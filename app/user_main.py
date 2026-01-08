"""
RAG Chat - User Interface API Server (Port 8001)
Provides end-user chat, widget endpoints, and OpenAI-compatible API.
"""

import os
import logging
import time
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Header, BackgroundTasks
from starlette.requests import Request as StarletteRequest
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, Response, RedirectResponse, HTMLResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
import uvicorn
import jwt
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from app.api.routes import rag_widget, agent_chat, user_credentials, tata_auth, openwebui_auth
from app.api.routes.auth import router as auth_router
from app.routers import openai_compatible, agentic_metrics
from app.core.database import init_db
from app.services.milvus_service import milvus_service
from app.services.ai_service import ai_service
from app.services.prometheus_metrics import metrics

load_dotenv()

logger = logging.getLogger("user_app")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── Configuration ───────────────────────────────────────────────────────────
JWT_SECRET = os.getenv("WIDGET_JWT_SECRET", "dev-secret-change-me")
JWT_ALG = "HS256"
WIDGET_API_KEY = os.getenv("WIDGET_API_KEY", "dev-widget-key")
WIDGET_STATIC_DIR = os.getenv("WIDGET_STATIC_DIR", "widget_static")
WIDGET_URL = os.getenv("WIDGET_URL", "")


# ── Image Normalization Helpers ─────────────────────────────────────────────
def _to_safe_url(u: Any) -> Optional[str]:
    """Safely convert value to URL string."""
    if not u:
        return None
    try:
        s = str(u).strip()
        return s if s else None
    except Exception:
        return None


def _normalize_image_obj(img: Any) -> Optional[Dict[str, Any]]:
    """Normalize image entry to {url, alt, caption, source_url}."""
    if not img:
        return None
    if isinstance(img, (str, bytes)):
        url = _to_safe_url(img)
        return {"url": url, "alt": "", "caption": "", "source_url": ""} if url else None
    if isinstance(img, dict):
        url = img.get("url") or img.get("src") or img.get("image")
        if not url and isinstance(img.get("data"), dict):
            url = img["data"].get("url")
        url = _to_safe_url(url)
        if not url:
            return None
        return {
            "url": url,
            "alt": str(img.get("alt") or img.get("title") or "")[:1024],
            "caption": str(img.get("caption") or img.get("description") or "")[:1024],
            "source_url": str(img.get("source_url") or img.get("page_url") or "")
        }
    return None


def _normalize_images_list(images: Any, cap: int = 12) -> List[Dict[str, Any]]:
    """Normalize a list of images, deduplicating by URL."""
    if not images:
        return []
    if isinstance(images, (str, bytes)):
        n = _normalize_image_obj(images)
        return [n] if n else []
    if not isinstance(images, (list, tuple)):
        n = _normalize_image_obj(images)
        return [n] if n else []
    
    out, seen = [], set()
    for it in images:
        norm = _normalize_image_obj(it)
        if norm and norm["url"] not in seen:
            seen.add(norm["url"])
            out.append(norm)
        if len(out) >= cap:
            break
    return out


# ── Lifespan Management ─────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup (DB, Milvus) and shutdown."""
    try:
        await init_db()
        logger.info("✅ Database initialized")
    except Exception as e:
        logger.exception("Database init failed: %s", e)

    try:
        await milvus_service.initialize()
        logger.info("✅ Milvus initialized")
    except Exception as e:
        logger.exception("Milvus init failed: %s", e)

    logger.info("✅ User API startup complete")
    yield
    
    logger.info("Shutting down...")
    try:
        await milvus_service.close()
        logger.info("✅ Milvus closed")
    except Exception as e:
        logger.warning("Milvus close error: %s", e)


# ── FastAPI App ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG Chat - User Interface",
    description="End-user chat interface for RAG knowledge base",
    version="2.0.0",
    lifespan=lifespan,
)


# ── CORS Configuration ──────────────────────────────────────────────────────
allowed_origins_env = os.getenv("USER_ALLOWED_ORIGINS", "")
if allowed_origins_env:
    allowed_origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]
else:
    allowed_origins = ["http://localhost:4201", "http://127.0.0.1:4201"]

if os.getenv("USER_ALLOW_ALL_ORIGINS", "false").lower() in ("1", "true", "yes"):
    allowed_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ── Middleware ──────────────────────────────────────────────────────────────
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Adds security headers (CSP, nosniff, referrer-policy)."""
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        try:
            if allowed_origins and allowed_origins != ["*"]:
                csp = f"default-src 'self'; frame-ancestors {' '.join(allowed_origins)};"
                response.headers["Content-Security-Policy"] = csp
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["Referrer-Policy"] = "no-referrer-when-downgrade"
        except Exception:
            pass
        return response


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Tracks HTTP request metrics for Prometheus."""
    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/metrics":
            return await call_next(request)
        
        # Normalize path
        parts = request.url.path.split('/')
        normalized = ['{id}' if p and (len(p) > 20 or p.isdigit() or (len(p) == 36 and p.count('-') == 4)) else p for p in parts]
        endpoint = '/'.join(normalized) or '/'
        method = request.method
        start_time = time.time()
        
        if cl := request.headers.get('content-length'):
            metrics.http_request_size.labels(method=method, endpoint=endpoint).observe(int(cl))
        
        response = await call_next(request)
        duration = time.time() - start_time
        
        metrics.http_requests_total.labels(method=method, endpoint=endpoint, status_code=str(response.status_code)).inc()
        metrics.http_request_duration.labels(method=method, endpoint=endpoint).observe(duration)
        
        if rs := response.headers.get('content-length'):
            metrics.http_response_size.labels(method=method, endpoint=endpoint).observe(int(rs))
        
        return response


app.add_middleware(SecurityHeadersMiddleware)
if os.getenv("ENABLE_PROMETHEUS_METRICS", "true").lower() in ("1", "true", "yes"):
    app.add_middleware(PrometheusMiddleware)


# ── Routes ──────────────────────────────────────────────────────────────────
app.include_router(auth_router)
app.include_router(user_credentials.router)
app.include_router(tata_auth.router)
app.include_router(openwebui_auth.router)
app.include_router(openai_compatible.router)
app.include_router(agentic_metrics.router)

if hasattr(rag_widget, "router"):
    app.include_router(rag_widget.router)
if hasattr(agent_chat, "router"):
    app.include_router(agent_chat.router, tags=["agent-chat"])


# ── Request Models ──────────────────────────────────────────────────────────
class UserQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    max_results: int = Field(default=5, ge=1, le=50)
    include_images: bool = Field(default=True)


class TokenResponse(BaseModel):
    token: str
    expires_at: str


# ── Chat Endpoint ───────────────────────────────────────────────────────────
@app.post("/api/chat/query")
async def user_chat_query(request: UserQueryRequest, background_tasks: BackgroundTasks):
    """End-user query endpoint with enhanced retrieval fallback."""
    try:
        query = request.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        logger.info(f"Query: '{query}' (max={request.max_results})")

        # Try widget_query first
        widget_resp = None
        try:
            WidgetReqModel = getattr(rag_widget, "WidgetQueryRequest", None)
            widget_query_fn = getattr(rag_widget, "widget_query", None)
            if WidgetReqModel and widget_query_fn:
                widget_req = WidgetReqModel(
                    query=query,
                    max_results=min(max(request.max_results, 1), 100),
                    include_sources=True,
                    enable_advanced_search=True,
                    search_depth="balanced",
                )
                widget_resp = await asyncio.wait_for(widget_query_fn(widget_req, background_tasks), timeout=20)
        except asyncio.TimeoutError:
            logger.warning("widget_query timed out")
        except Exception as e:
            logger.exception("widget_query error: %s", e)

        # Check response quality
        def is_weak(resp):
            if not resp or not isinstance(resp, dict):
                return True
            return (
                resp.get("results_found", 0) < 3 or
                (resp.get("confidence") or 0) < 0.60 or
                len((resp.get("answer") or "").strip()) < 80 or
                (request.include_images and not resp.get("images") and (resp.get("confidence") or 0) < 0.95)
            )

        # Return if response is good
        if widget_resp and not is_weak(widget_resp):
            return _format_response(query, widget_resp, request.include_images)

        # Enhanced retrieval fallback
        logger.info("Weak response - performing enhanced retrieval")
        return await _enhanced_retrieval(query, request, widget_resp)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Query error: %s", e)
        raise HTTPException(status_code=500, detail="Request processing failed")


def _format_response(query: str, resp: Dict, include_images: bool) -> Dict:
    """Format widget response for output."""
    images = _normalize_images_list(resp.get("images", []), cap=24) if include_images else []
    steps = []
    for s in (resp.get("steps") or []):
        if isinstance(s, dict):
            img_url = None
            if raw_img := (s.get("image") or s.get("image_url")):
                if n := _normalize_image_obj(raw_img):
                    img_url = n["url"]
            steps.append({"index": s.get("index"), "text": s.get("text") or s.get("content") or "", "image": img_url, "caption": s.get("caption")})
        else:
            steps.append({"index": None, "text": str(s), "image": None})
    
    return {
        "query": resp.get("query", query),
        "answer": resp.get("answer", ""),
        "steps": steps,
        "stepsTitle": resp.get("stepsTitle") or "Step-by-step instructions",
        "images": images,
        "summary": resp.get("summary"),
        "summaryTitle": resp.get("summaryTitle") or "Quick Summary",
        "timestamp": resp.get("timestamp") or datetime.now().isoformat(),
        "confidence": resp.get("confidence", 0.0),
        "results_found": resp.get("results_found", 0),
        "results_used": resp.get("results_used", 0),
        "has_sources": resp.get("has_sources", False),
    }


async def _enhanced_retrieval(query: str, request: UserQueryRequest, fallback_resp: Optional[Dict]) -> Dict:
    """Perform enhanced Milvus retrieval when widget response is weak."""
    try:
        results = await milvus_service.search_documents(query, n_results=min(200, request.max_results * 3))
        if not results:
            return _fallback_response(query, fallback_resp)

        # Build context
        context_texts = [r.get("content", "")[:5000] for r in results[:40] if len((r.get("content") or "").strip()) > 50]
        if not context_texts:
            context_texts = [r.get("content", "")[:2000] for r in results[:3]]

        # Generate enhanced response
        enhanced = await ai_service.generate_enhanced_response(query, context_texts, query_type=None) or {}
        answer = enhanced.get("text", "") if isinstance(enhanced, dict) else str(enhanced)
        confidence = float(enhanced.get("quality_score", 0) if isinstance(enhanced, dict) else 0)

        # Extract images
        images = _extract_images_from_results(results, request.include_images)

        # Generate steps
        steps_data = await ai_service.generate_stepwise_response(query, context_texts[:3]) or []
        if not steps_data and answer:
            steps_data = [{"text": s.strip() + ".", "type": "info"} for s in answer.split(".")[:6] if s.strip()]

        steps = []
        image_urls = [img["url"] for img in images if img.get("url")]
        for i, step in enumerate(steps_data):
            text = step.get("text") if isinstance(step, dict) else str(step)
            img_url = image_urls[i] if i < len(image_urls) else None
            steps.append({"index": i + 1, "text": text, "image": img_url})

        # Generate summary
        try:
            summary = await ai_service.generate_summary(answer or "\n\n".join(context_texts[:3]), max_sentences=3, max_chars=600)
        except Exception:
            summary = (answer[:600] + "...") if len(answer) > 600 else answer

        return {
            "query": query,
            "answer": answer or "Found information but couldn't generate a summary.",
            "steps": steps,
            "stepsTitle": "Step-by-step instructions",
            "images": images,
            "summary": summary or "No summary available.",
            "summaryTitle": "Quick Summary",
            "timestamp": datetime.now().isoformat(),
            "confidence": round(confidence, 3),
            "results_found": len(results),
            "results_used": min(len(results), request.max_results),
            "has_sources": False,
        }
    except Exception as e:
        logger.exception("Enhanced retrieval failed: %s", e)
        return _fallback_response(query, fallback_resp)


def _extract_images_from_results(results: List[Dict], include: bool) -> List[Dict]:
    """Extract and normalize images from search results."""
    if not include:
        return []
    images, seen = [], set()
    for r in results:
        meta = r.get("metadata") or {}
        for img in (meta.get("images") or []):
            if isinstance(img, dict):
                n = _normalize_image_obj(img)
                if n and n["url"] not in seen and not any(x in n["url"].lower() for x in ["logo", "icon", "favicon", "sprite", "banner"]):
                    n["source_url"] = meta.get("url", "")
                    seen.add(n["url"])
                    images.append(n)
            if len(images) >= 24:
                break
        if len(images) >= 24:
            break
    return images


def _fallback_response(query: str, fallback: Optional[Dict]) -> Dict:
    """Return fallback response when all else fails."""
    if fallback:
        return fallback
    return {
        "query": query,
        "answer": "I don't have enough information to answer that question.",
        "images": [],
        "timestamp": datetime.now().isoformat(),
        "confidence": 0.0,
        "results_found": 0,
    }


# ── Other Endpoints ─────────────────────────────────────────────────────────
@app.get("/metrics", include_in_schema=False)
async def prometheus_metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/debug/llm-calls", tags=["debug"])
async def debug_llm_calls():
    """Debug: view LLM call tracking."""
    return {
        "total_calls": len(metrics.get_llm_call_log()),
        "call_sources": metrics.get_llm_call_sources(),
        "recent_calls": metrics.get_llm_call_log()[-20:],
    }


@app.post("/debug/llm-calls/clear", tags=["debug"])
async def debug_llm_calls_clear():
    """Debug: clear LLM call log."""
    metrics.clear_llm_call_log()
    return {"status": "cleared"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        stats = await milvus_service.get_collection_stats()
        doc_count = stats.get("document_count", 0) if isinstance(stats, dict) else 0
        return {"status": "healthy", "service": "user_chat", "documents": doc_count, "version": app.version}
    except Exception as e:
        return JSONResponse({"status": "unhealthy", "error": str(e)}, status_code=503)


@app.post("/api/widget/token", response_model=TokenResponse)
async def issue_widget_token(x_api_key: Optional[str] = Header(None)):
    """Issue JWT token for widget authentication."""
    if WIDGET_API_KEY and x_api_key != WIDGET_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    expires = datetime.utcnow() + timedelta(minutes=30)
    token = jwt.encode({"sub": "widget-client", "iat": datetime.utcnow(), "exp": expires, "scope": "widget:query"}, JWT_SECRET, algorithm=JWT_ALG)
    return {"token": token, "expires_at": expires.isoformat()}


@app.get("/api/widget/config")
async def widget_config():
    """Get widget configuration."""
    return {
        "allowed_origins": allowed_origins,
        "widget_url": WIDGET_URL or ("/widget/index.html" if os.path.isdir(WIDGET_STATIC_DIR) else ""),
        "api_base": os.getenv("USER_API_BASE", ""),
    }


@app.get("/auth")
@app.get("/auth/login")
async def auth_redirect(request: StarletteRequest):
    """Redirect /auth routes to login page (OpenWebUI compatibility)."""
    referer = request.headers.get("referer", "")
    origin = request.headers.get("origin", "")
    
    if referer:
        try:
            from urllib.parse import urlparse
            parsed = urlparse(referer)
            return RedirectResponse(url=f"{parsed.scheme}://{parsed.netloc}/", status_code=302)
        except Exception:
            pass
    if origin:
        return RedirectResponse(url=f"{origin}/", status_code=302)
    
    login_url = os.getenv("OPENWEBUI_LOGIN_URL", "http://localhost:3000") + "/"
    return HTMLResponse(f'<meta http-equiv="refresh" content="0; url={login_url}"><a href="{login_url}">Redirecting...</a>')


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "RAG Chat - User Interface", "version": app.version, "usage": "POST /api/chat/query"}


# ── Static Files ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
user_frontend_path = BASE_DIR / "dist" / "user-frontend"

if user_frontend_path.exists() and user_frontend_path.is_dir():
    app.mount("/", StaticFiles(directory=str(user_frontend_path), html=True), name="user_frontend")
    logger.info("✅ Frontend mounted: %s", user_frontend_path)

    @app.get("/{full_path:path}")
    async def spa_fallback(full_path: str):
        index = user_frontend_path / "index.html"
        return FileResponse(index) if index.exists() else JSONResponse({"detail": "index.html not found"}, status_code=500)

widget_static_path = BASE_DIR / WIDGET_STATIC_DIR
if widget_static_path.exists():
    app.mount("/widget", StaticFiles(directory=str(widget_static_path), html=True), name="widget")
    app.mount("/cdn", StaticFiles(directory=str(widget_static_path)), name="cdn")


# ── Main Entry Point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "user_main:app",
        host=os.getenv("USER_APP_HOST", "0.0.0.0"),
        port=int(os.getenv("USER_APP_PORT", "8001")),
        reload=os.getenv("DEV_RELOAD", "false").lower() in ("1", "true"),
        workers=int(os.getenv("USER_WORKERS", "2")),
        log_level=os.getenv("LOG_LEVEL", "info"),
    )
