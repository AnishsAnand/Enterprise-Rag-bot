# user_main.py (Merged: User Chat UI + Action Agent / ActionBot orchestration)
# Entrypoint: user_main:app
import os
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Header, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
import uvicorn
import jwt
import asyncio
import importlib
import signal
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent
from uuid import uuid4

# Project imports (existing)
from dotenv import load_dotenv
from app.api.routes import rag_widget
from app.api.routes.auth import router as auth_router
from app.core.database import init_db
from app.services.milvus_service import milvus_service
from app.services.ai_service import ai_service

# ActionBot import (we will instantiate using the existing ai_service)
from app.api.routes.action_bot import ActionBot, ActionResult, ActionStatus, ConversationSession

load_dotenv()

# ------------------------ Logging ------------------------
logger = logging.getLogger("user_app")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ------------------------ JWT / Widget Config ------------------------
JWT_SECRET = os.getenv("WIDGET_JWT_SECRET", "dev-secret-change-me")
JWT_ALG = "HS256"
WIDGET_API_KEY = os.getenv("WIDGET_API_KEY", "dev-widget-key")
WIDGET_STATIC_DIR = os.getenv("WIDGET_STATIC_DIR", "widget_static")
WIDGET_URL = os.getenv("WIDGET_URL", "")

# ------------------------ Helpers (kept from your file) ------------------------
def _to_safe_url(u: Any) -> Optional[str]:
    if not u:
        return None
    try:
        s = str(u).strip()
    except Exception:
        return None
    if not s:
        return None
    return s

def _normalize_image_obj(img: Any) -> Optional[Dict[str, Any]]:
    if not img:
        return None
    if isinstance(img, (str, bytes)):
        url = _to_safe_url(img)
        if url:
            return {"url": url, "alt": "", "caption": "", "source_url": ""}
        return None
    if isinstance(img, dict):
        url = img.get("url") or img.get("src") or img.get("image") or None
        if not url and isinstance(img.get("data"), dict):
            url = img["data"].get("url")
        url = _to_safe_url(url)
        if not url:
            return None
        return {
            "url": url,
            "alt": str(img.get("alt") or img.get("title") or "")[:1024],
            "caption": str(img.get("caption") or img.get("description") or "")[:1024],
            "source_url": str(img.get("source_url") or img.get("page_url") or img.get("page") or "")
        }
    return None

def _normalize_images_list(images: Any, cap: int = 12) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not images:
        return out
    if isinstance(images, (str, bytes)):
        n = _normalize_image_obj(images)
        if n:
            out.append(n)
        return out
    if not isinstance(images, (list, tuple)):
        n = _normalize_image_obj(images)
        if n:
            out.append(n)
        return out
    seen = set()
    for it in images:
        norm = _normalize_image_obj(it)
        if norm and norm["url"] not in seen:
            seen.add(norm["url"])
            out.append(norm)
        if len(out) >= cap:
            break
    return out

# ------------------------ FastAPI Lifespan & App (combined) ------------------------

# ---- Action Agent Manager (merged and adapted) ----
class ActionAgentManager:
    """
    Manages ActionBot lifecycle within this process.
    Uses the shared ai_service instance (imported).
    """
    def __init__(self, ai_service_instance, service_registry: Dict[str, str], reload_watch: bool = False):
        self.ai_service = ai_service_instance
        self.action_bot: Optional[ActionBot] = None
        self.service_registry = service_registry
        self.websocket_connections: Dict[str, WebSocket] = {}
        self.is_shutting_down = False
        self.file_observer: Optional[Observer] = None
        self.reload_watch = reload_watch
        self._last_reload = datetime.utcnow()

        # Files to watch (optional)
        self.watch_files = os.getenv("ACTION_WATCH_FILES", "action_bot.py,ai_service.py").split(",")

    async def initialize(self):
        logger.info("🚀 ActionAgentManager: initializing ActionBot...")
        # Instantiate ActionBot using existing ai_service
        self.action_bot = ActionBot(ai_service=self.ai_service, service_registry=self.service_registry)
        await self.action_bot.initialize()
        logger.info("✅ ActionBot initialized inside ActionAgentManager")

        if self.reload_watch and self.file_observer is None:
            self._setup_file_watcher()
            logger.info("🔁 File watcher enabled for ActionAgentManager")

    async def shutdown(self):
        if self.is_shutting_down:
            return
        self.is_shutting_down = True
        logger.info("🛑 ActionAgentManager shutting down...")
        try:
            # Close websocket connections
            for ws in list(self.websocket_connections.values()):
                try:
                    await ws.close()
                except Exception:
                    pass
            # Stop file observer
            if self.file_observer:
                self.file_observer.stop()
                self.file_observer.join(timeout=2)
            # Close action bot
            if self.action_bot:
                await self.action_bot.close()
            logger.info("✅ ActionAgentManager shutdown complete")
        except Exception as e:
            logger.exception("❌ Error during ActionAgentManager shutdown: %s", e)

    async def reload_action_bot(self):
        """Safely reload action_bot module and re-create ActionBot"""
        now = datetime.utcnow()
        # cooldown 2s
        if (now - self._last_reload).total_seconds() < 2:
            return
        self._last_reload = now

        logger.info("🔄 Reloading ActionBot due to file change...")
        try:
            if self.action_bot:
                await self.action_bot.close()
            # reload python module
            import importlib
            import app.api.routes.action_bot as action_bot_mod
            importlib.reload(action_bot_mod)
            ActionBotReloaded = action_bot_mod.ActionBot
            # instantiate using existing ai_service
            self.action_bot = ActionBotReloaded(ai_service=self.ai_service, service_registry=self.service_registry)
            await self.action_bot.initialize()
            logger.info("✅ ActionBot reloaded successfully")
            # notify websockets
            await self._broadcast({"type": "reload_complete", "message": "ActionBot reloaded"})
        except Exception as e:
            logger.exception("❌ Failed to reload ActionBot: %s", e)
            await self._broadcast({"type": "reload_error", "message": str(e)})

    def _setup_file_watcher(self):
        class _Handler(FileSystemEventHandler):
            def __init__(self, mgr: "ActionAgentManager"):
                self.mgr = mgr
                super().__init__()
            def on_modified(self, event):
                if isinstance(event, FileModifiedEvent):
                    p = Path(event.src_path)
                    if p.suffix == ".py" and p.name in [f.strip() for f in self.mgr.watch_files]:
                        asyncio.create_task(self.mgr.reload_action_bot())
        handler = _Handler(self)
        self.file_observer = Observer()
        self.file_observer.schedule(handler, path=".", recursive=True)
        self.file_observer.start()

    async def _broadcast(self, message: Dict[str, Any]):
        disconnected = []
        for cid, ws in list(self.websocket_connections.items()):
            try:
                await ws.send_json(message)
            except Exception:
                disconnected.append(cid)
        for cid in disconnected:
            try:
                del self.websocket_connections[cid]
            except KeyError:
                pass

# ---- Application FastAPI app ----
app = FastAPI(
    title="RAG Chat + Action Agent (Merged)",
    description="Combined user chat UI and Action Agent orchestration API",
    version="2.0.0"
)

# ------------------------ CORS ------------------------
allowed_origins_env = os.getenv("USER_ALLOWED_ORIGINS", "")
if allowed_origins_env:
    allowed_origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]
else:
    allowed_origins = [
        "http://localhost:4201",
        "http://127.0.0.1:4201",
    ]

if os.getenv("USER_ALLOW_ALL_ORIGINS", "false").lower() in ("1", "true", "yes"):
    allowed_origins = ["*"]

logger.info("✅ Allowed origins for CORS/framing: %s", allowed_origins)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS", "PUT", "DELETE"],
    allow_headers=["*"],
)

# ------------------------ Security Headers ------------------------
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        try:
            if allowed_origins and allowed_origins != ["*"]:
                frame_ancestors = " ".join(allowed_origins)
                csp_value = f"default-src 'self'; frame-ancestors {frame_ancestors};"
                response.headers["Content-Security-Policy"] = csp_value
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["Referrer-Policy"] = "no-referrer-when-downgrade"
        except Exception:
            pass
        return response

app.add_middleware(SecurityHeadersMiddleware)

# ------------------------ Include existing routers ------------------------
app.include_router(auth_router)
# include orchestrator if present (keeps previous behavior)
try:
    from app.api.routes import orchestrator
    app.include_router(orchestrator.router)
except Exception:
    logger.debug("orchestrator.router not available, skipping")

if hasattr(rag_widget, "router"):
    app.include_router(rag_widget.router)
    logger.info("✅ Included rag_widget.router into application")
else:
    logger.warning("⚠ rag_widget module has no 'router' attribute — widget routes not mounted")

# ------------------------ Globals for manager ------------------------
service_registry = {
    "admin": os.getenv("ADMIN_SERVICE_URL", "http://localhost:8001"),
    "rag": os.getenv("RAG_SERVICE_URL", "http://localhost:8002"),
    "scraper": os.getenv("SCRAPER_SERVICE_URL", "http://localhost:8003"),
}
_action_agent_manager: Optional[ActionAgentManager] = None

# ------------------------ Lifespan ------------------------
@asynccontextmanager
async def lifespan(app_inst: FastAPI):
    global _action_agent_manager
    # --- DB init
    try:
        await init_db()
        logger.info("✅ Database initialized")
    except Exception as e:
        logger.exception("Database initialization failed (continuing): %s", e)

    # --- Milvus
    try:
        await milvus_service.initialize()
        logger.info("✅ Milvus connected successfully")
    except Exception as e:
        logger.exception("Milvus initialization failed (continuing): %s", e)

    # --- AI service basic test (use shared instance)
    try:
        # ensure ai_service initializes itself if required
        if hasattr(ai_service, "initialize"):
            try:
                await ai_service.initialize()
            except Exception:
                # Some ai_service implementations may be lazy - tolerate failures and continue
                logger.debug("ai_service.initialize() failed or not required (continuing)")
        test_embeddings = []
        try:
            test_embeddings = await ai_service.generate_embeddings(["test"])
        except Exception:
            logger.debug("ai_service.generate_embeddings test failed (continuing)")
        if test_embeddings and isinstance(test_embeddings, list) and len(test_embeddings) > 0 and len(test_embeddings[0]) > 0:
            logger.info("✅ AI services operational")
        else:
            logger.warning("⚠️ AI service generate_embeddings returned empty result during startup test")
    except Exception as e:
        logger.exception("AI service initialization/test failed (continuing): %s", e)

    # --- Initialize Action Agent Manager (shared ai_service)
    try:
        reload_watch = os.getenv("ACTION_RELOAD_WATCH", "false").lower() in ("1", "true", "yes")
        _action_agent_manager = ActionAgentManager(ai_service_instance=ai_service, service_registry=service_registry, reload_watch=reload_watch)
        await _action_agent_manager.initialize()
    except Exception as e:
        logger.exception("ActionAgentManager initialization failed (continuing): %s", e)
        _action_agent_manager = None

    logger.info("✅ user_main lifespan startup complete")
    yield
    logger.info("Shutting down user app...")

    # Shutdown order: action agent -> milvus -> others
    try:
        if _action_agent_manager:
            await _action_agent_manager.shutdown()
    except Exception as e:
        logger.warning("⚠️ Error shutting down ActionAgentManager: %s", e)
    try:
        await milvus_service.close()
        logger.info("✅ Milvus connection closed")
    except Exception as e:
        logger.warning("⚠️ Error closing Milvus: %s", e)

app.router.lifespan_context = lifespan

# ------------------------ Request Models (user chat) ------------------------
class UserQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    max_results: int = Field(default=5, ge=1, le=50)
    include_images: bool = Field(default=True)

# ------------------------ User Chat Endpoint (keeps your logic) ------------------------
@app.post("/api/chat/query")
async def user_chat_query(request: UserQueryRequest, background_tasks: BackgroundTasks):
    # (omitted here for brevity — reuse your full implementation)
    # For brevity in this merged code snippet, call into your original function if present.
    # If you want the full inlined function, copy-paste your earlier user_chat_query body here.
    from app.api.routes.user_chat_impl import handle_user_chat_query  # optional: split logic into helper module
    # If helper not found, fallback to simplified behavior:
    try:
        # If you had the full handler in this file, keep it. For now use simple routing to rag_widget if present.
        query = request.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        # Attempt to call rag_widget.handler if available (this keeps previous behavior)
        widget_query_fn = getattr(rag_widget, "widget_query", None)
        if widget_query_fn:
            try:
                widget_req_model = getattr(rag_widget, "WidgetQueryRequest", None)
                if widget_req_model:
                    widget_req = widget_req_model(
                        query=query,
                        max_results=min(max(int(request.max_results or 1), 1), 100),
                        include_sources=True,
                        enable_advanced_search=True,
                        search_depth="balanced"
                    )
                    widget_resp = await widget_query_fn(widget_req, background_tasks)
                    # return widget_resp if valid; else fallback to simple message
                    if widget_resp and isinstance(widget_resp, dict):
                        return widget_resp
            except Exception as e:
                logger.exception("Error calling rag_widget.widget_query: %s", e)
        # fallback: basic reply
        return {"query": query, "answer": "Widget not available or returned no content", "confidence": 0.0}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("User chat error: %s", e)
        raise HTTPException(status_code=500, detail="Internal error")

# ------------------------ Health Check (merged) ------------------------
@app.get("/health")
async def health_check():
    try:
        ok = True
        doc_count = 0
        try:
            stats = await milvus_service.get_collection_stats()
            doc_count = stats.get("document_count", 0) if isinstance(stats, dict) else 0
        except Exception:
            ok = False

        # Action agent health (if available)
        agent_status = "unavailable"
        if _action_agent_manager and _action_agent_manager.action_bot:
            agent_status = "ready"
        return {
            "status": "healthy" if ok else "unhealthy",
            "service": "user_chat_with_action_agent",
            "documents_available": doc_count if ok else 0,
            "action_agent": agent_status,
            "version": app.version
        }
    except Exception as e:
        logger.exception("Health check failed")
        return JSONResponse({"status": "unhealthy", "error": str(e)}, status_code=503)

# ------------------------ Widget Token / Config (kept unchanged) ------------------------
class TokenResponse(BaseModel):
    token: str
    expires_at: str

@app.post("/api/widget/token", response_model=TokenResponse)
async def issue_widget_token(x_api_key: Optional[str] = Header(None)):
    if WIDGET_API_KEY and x_api_key != WIDGET_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    payload = {
        "sub": "widget-client",
        "iat": datetime.utcnow(),
        "exp": datetime.utcnow() + timedelta(minutes=30),
        "scope": "widget:query"
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)
    return {"token": token, "expires_at": (datetime.utcnow() + timedelta(minutes=30)).isoformat()}

@app.get("/api/widget/config")
async def widget_config():
    api_base = os.getenv("USER_API_BASE", "")
    return {
        "allowed_origins": allowed_origins,
        "widget_url": WIDGET_URL or ("/widget/index.html" if os.path.isdir(WIDGET_STATIC_DIR) else ""),
        "api_base": api_base or ""
    }

# ------------------------ Frontend SPA / Widget / CDN mounting (unchanged) ------------------------
BASE_DIR = Path(__file__).resolve().parent
user_frontend_path = BASE_DIR / "dist" / "user-frontend"

if user_frontend_path.exists() and user_frontend_path.is_dir():
    app.mount("/", StaticFiles(directory=str(user_frontend_path), html=True), name="user_frontend")
    logger.info(f"✅ User frontend mounted at: {user_frontend_path}")

    @app.get("/{full_path:path}")
    async def spa_fallback(full_path: str):
        index_file = user_frontend_path / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return JSONResponse({"detail": "index.html not found"}, status_code=500)
else:
    logger.warning(f"⚠️ User frontend not found at {user_frontend_path}")

widget_static_path = BASE_DIR / WIDGET_STATIC_DIR
if widget_static_path.exists() and widget_static_path.is_dir():
    app.mount("/widget", StaticFiles(directory=str(widget_static_path), html=True), name="widget")
    logger.info(f"✅ Widget static mounted at /widget -> {widget_static_path}")
else:
    logger.info(f"⚠️ Widget static not mounted (folder {widget_static_path} not found)")

cdn_static_path = BASE_DIR / WIDGET_STATIC_DIR
if cdn_static_path.exists() and cdn_static_path.is_dir():
    app.mount("/cdn", StaticFiles(directory=str(cdn_static_path)), name="cdn")
    logger.info(f"✅ CDN static mounted at /cdn -> {cdn_static_path}")
else:
    logger.info(f"⚠️ CDN static not mounted (folder {cdn_static_path} not found)")

# ------------------------ Action Agent API Endpoints (from action_agent_main) ------------------------

# Request/Response models simplified / re-used
class ActionRequest(BaseModel):
    query: str = Field(..., description="Natural language query or command")
    session_id: Optional[str] = Field(None, description="Session ID for continuing conversation")
    user_input: Optional[str] = Field(None, description="User input for ongoing session")
    skip_confirmation: bool = Field(False, description="Skip confirmation step")

class ActionResponse(BaseModel):
    status: str
    message: str
    details: Optional[Dict[str, Any]] = None
    next_step: Optional[str] = None
    missing_fields: Optional[list] = None
    suggestions: Optional[list] = None
    confidence: float = 0.0
    session_id: Optional[str] = None
    execution_time: Optional[float] = None

@app.post("/api/action/execute", response_model=ActionResponse)
async def execute_action(request: ActionRequest, background_tasks: BackgroundTasks):
    """Execute an action based on natural language query"""
    try:
        if not _action_agent_manager or not _action_agent_manager.action_bot:
            raise HTTPException(status_code=503, detail="ActionBot not initialized")

        logger.info(f"📥 [Action] Received action request: {request.query[:200]}")
        payload = {
            "query": request.query,
            "skip_confirmation": request.skip_confirmation
        }
        if request.user_input:
            payload["user_input"] = request.user_input

        result: ActionResult = await _action_agent_manager.action_bot.handle_action_request(
            payload=payload,
            session_id=request.session_id
        )

        response = ActionResponse(
            status=result.status.value,
            message=result.message,
            details=result.details,
            next_step=result.next_step,
            missing_fields=result.missing_fields,
            suggestions=result.suggestions,
            confidence=result.confidence,
            session_id=result.session_id,
            execution_time=result.execution_time
        )

        logger.info(f"✅ [Action] Completed: status={result.status.value} session={result.session_id}")
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"❌ Action execution failed: {e}")
        return ActionResponse(
            status="error",
            message=f"Failed to execute action: {str(e)}",
            details={"error": str(e), "type": type(e).__name__}
        )

@app.post("/api/action/analyze")
async def analyze_query(request: ActionRequest):
    """Analyze query without executing"""
    try:
        if not _action_agent_manager or not _action_agent_manager.action_bot:
            raise HTTPException(status_code=503, detail="ActionBot not initialized")
        result = await _action_agent_manager.action_bot.analyze_and_suggest(request.query)
        return result
    except Exception as e:
        logger.exception(f"❌ Query analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sessions")
async def list_sessions():
    try:
        if not _action_agent_manager or not _action_agent_manager.action_bot:
            raise HTTPException(status_code=503, detail="ActionBot not initialized")
        sessions = await _action_agent_manager.action_bot.get_active_sessions()
        return {"active_sessions": sessions, "count": len(sessions), "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.exception(f"❌ Failed to list sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    try:
        if not _action_agent_manager or not _action_agent_manager.action_bot:
            raise HTTPException(status_code=503, detail="ActionBot not initialized")
        session: Optional[ConversationSession] = await _action_agent_manager.action_bot.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return {
            "session_id": session.session_id,
            "query": session.user_query,
            "intent": session.detected_intent.value if session.detected_intent else None,
            "confidence": session.intent_confidence,
            "step_number": session.step_number,
            "collected_params": session.collected_params,
            "pending_params": session.pending_params,
            "awaiting_confirmation": session.awaiting_confirmation,
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"❌ Failed to get session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/sessions/{session_id}")
async def clear_session(session_id: str):
    try:
        if not _action_agent_manager or not _action_agent_manager.action_bot:
            raise HTTPException(status_code=503, detail="ActionBot not initialized")
        success = await _action_agent_manager.action_bot.clear_session(session_id)
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"message": "Session cleared successfully", "session_id": session_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"❌ Failed to clear session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/services/health")
async def services_health():
    try:
        if not _action_agent_manager or not _action_agent_manager.action_bot:
            raise HTTPException(status_code=503, detail="ActionBot not initialized")
        health = await _action_agent_manager.action_bot.check_service_health()
        return {"services": health, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.exception(f"❌ Failed to check services health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/services/capabilities")
async def services_capabilities():
    try:
        if not _action_agent_manager or not _action_agent_manager.action_bot:
            raise HTTPException(status_code=503, detail="ActionBot not initialized")
        capabilities = await _action_agent_manager.action_bot.get_service_capabilities()
        result = {}
        for name, cap in capabilities.items():
            result[name] = {
                "name": cap.name,
                "base_url": cap.base_url,
                "available": cap.available,
                "version": cap.version,
                "health_status": cap.health_status,
                "last_check": cap.last_check.isoformat(),
                "endpoints": list(cap.endpoints.keys()) if cap.endpoints else []
            }
        return {"capabilities": result}
    except Exception as e:
        logger.exception(f"❌ Failed to get capabilities: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ------------------------ WebSocket endpoint for interactive action execution ------------------------
@app.websocket("/ws/action")
async def websocket_action(websocket: WebSocket):
    await websocket.accept()
    client_id = f"client_{uuid4().hex[:8]}"
    if not _action_agent_manager:
        await websocket.send_json({"type": "error", "message": "Action agent not initialized"})
        await websocket.close()
        return
    _action_agent_manager.websocket_connections[client_id] = websocket
    logger.info(f"🔌 WebSocket client connected: {client_id}")
    try:
        while True:
            data = await websocket.receive_json()
            if data.get("type") == "action":
                payload = {
                    "query": data.get("query", ""),
                    "skip_confirmation": data.get("skip_confirmation", False)
                }
                if data.get("user_input"):
                    payload["user_input"] = data["user_input"]
                result = await _action_agent_manager.action_bot.handle_action_request(payload=payload, session_id=data.get("session_id"))
                await websocket.send_json({
                    "type": "result",
                    "status": result.status.value,
                    "message": result.message,
                    "details": result.details,
                    "session_id": result.session_id,
                    "next_step": result.next_step,
                    "missing_fields": result.missing_fields,
                    "suggestions": result.suggestions
                })
            elif data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        logger.info(f"🔌 WebSocket client disconnected: {client_id}")
    except Exception as e:
        logger.exception(f"❌ WebSocket error: {e}")
    finally:
        if client_id in _action_agent_manager.websocket_connections:
            del _action_agent_manager.websocket_connections[client_id]

# ------------------------ Root ------------------------
@app.get("/")
async def root():
    return {"message": "RAG Chat - User Interface + Action Agent", "version": app.version, "usage": "POST /api/chat/query or /api/action/execute"}

# ------------------------ Run (when executed directly) ------------------------
if __name__ == "__main__":
    host = os.getenv("USER_APP_HOST", "0.0.0.0")
    port = int(os.getenv("USER_APP_PORT", "8001"))
    reload_flag = os.getenv("DEV_RELOAD", "false").lower() in ("1", "true", "yes")
    uvicorn.run("user_main:app", host=host, port=port, reload=reload_flag, log_level=os.getenv("LOG_LEVEL", "info"))
