# user_main.py (Milvus migration - production-ready)
import os
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from fastapi import BackgroundTasks
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
import uvicorn
import jwt

from urllib.parse import urljoin
from app.api.routes import rag_widget
from app.api.routes.auth import router as auth_router
from app.core.database import init_db
from app.services.milvus_service import milvus_service
from app.services.ai_service import ai_service

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

# ---------- Helpers to normalize images & steps ----------
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
    """
    Normalize image entries to {url, alt, caption, source_url}
    Accepts str or dict.
    """
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

# ------------------------ FastAPI Lifespan ------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await init_db()
        logger.info("✅ Database initialized")
    except Exception as e:
        logger.exception("Database initialization failed (continuing): %s", e)

    try:
        await milvus_service.initialize()
        logger.info("✅ Milvus connected successfully")
    except Exception as e:
        logger.exception("Milvus initialization failed (continuing): %s", e)

    try:
        test_embeddings = await ai_service.generate_embeddings(["test"])
        if test_embeddings and len(test_embeddings[0]) > 0:
            logger.info("✅ AI services operational")
        else:
            logger.warning("⚠️ AI service generate_embeddings returned empty result during startup test")
    except Exception as e:
        logger.exception("AI service initialization/test failed (continuing): %s", e)

    logger.info("✅ user_main lifespan startup complete")
    yield
    logger.info("Shutting down user app...")
    try:
        await milvus_service.close()
        logger.info("✅ Milvus connection closed")
    except Exception as e:
        logger.warning("⚠️ Error closing Milvus: %s", e)

# ------------------------ FastAPI App ------------------------
app = FastAPI(
    title="RAG Chat - User Interface",
    description="End-user chat interface for RAG knowledge base",
    version="2.0.0",
    lifespan=lifespan,
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
    allow_methods=["GET", "POST", "OPTIONS"],
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

# ------------------------ Routers ------------------------
app.include_router(auth_router)

if hasattr(rag_widget, "router"):
    app.include_router(rag_widget.router)
    logger.info("✅ Included rag_widget.router into application")
else:
    logger.warning("⚠ rag_widget module has no 'router' attribute — widget routes not mounted")

# ------------------------ Request Models ------------------------
class UserQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    max_results: int = Field(default=5, ge=1, le=50)
    include_images: bool = Field(default=True)

# ------------------------ Chat API (end-user) ------------------------
@app.post("/api/chat/query")
async def user_chat_query(request: UserQueryRequest, background_tasks: BackgroundTasks):
    """
    End-user query endpoint using rag_widget.widget_query handler.
    Enhanced fallback will perform broader Milvus search + generate stepwise response
    and attach step images so the frontend can render inline step images.
    """
    try:
        query = request.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        logger.info(f"User query: '{query}' (max_results={request.max_results}, include_images={request.include_images})")

        # build a widget request (balanced by default) and call widget_query
        WidgetReqModel = getattr(rag_widget, "WidgetQueryRequest")
        if WidgetReqModel is None:
            logger.error("rag_widget.WidgetQueryRequest model missing")
            raise HTTPException(status_code=500, detail="Widget model not available")
        
        widget_max = min(max(int(request.max_results or 1), 1), 100)
        logger.debug("Building widget request with max_results=%d", widget_max)

        widget_req = WidgetReqModel(
            query=query,
            max_results=widget_max,
            include_sources=True,
            enable_advanced_search=True,
            search_depth="balanced",
        )


        widget_resp = None 
        try:
            widget_query_fn = getattr(rag_widget, "widget_query", None)
            if widget_query_fn is None:
                raise RuntimeError("rag_widget.widget_query not available")

            # use a timeout to protect caller (adjust as needed)
            import asyncio
            try:
                widget_resp = await asyncio.wait_for(widget_query_fn(widget_req, background_tasks), timeout=20)
            except asyncio.TimeoutError:
                logger.warning("rag_widget.widget_query timed out")
                widget_resp = None
        except Exception as e:
            logger.exception("Error calling rag_widget.widget_query: %s", e)
            widget_resp = None
        

        # Basic validation
        def _is_weak_widget_response(resp: Optional[dict]) -> bool:
            if not resp or not isinstance(resp, dict):
                return True
            results_found = int(resp.get("results_found", 0))
            confidence = float(resp.get("confidence", 0.0) or 0.0)
            answer = (resp.get("answer") or "").strip()
            images = resp.get("images") or []
            if results_found < 3:
                return True
            if confidence < 0.60:
                return True
            if len(answer) < 80:
                return True
            # Skip image requirement for high-confidence responses (API data like cluster listings)
            # Images are only needed for RAG documentation responses
            if request.include_images and (not images) and confidence < 0.95:
                return True
            return False

        # If widget_resp is acceptable, use it.
        if widget_resp and not _is_weak_widget_response(widget_resp):
            # Ensure images normalized and steps image strings present if rag_widget produced them
            images = _normalize_images_list(widget_resp.get("images", []), cap=24) if request.include_images else []
            steps = []
            raw_steps = widget_resp.get("steps") or []
            if isinstance(raw_steps, list):
                for s in raw_steps:
                    # step might be dict with image as object or string
                    if isinstance(s, dict):
                        idx = s.get("index") or None
                        text = s.get("text") or s.get("content") or ""
                        raw_img = s.get("image") or s.get("image_url") or s.get("image_data") or None
                        img_url = None
                        if raw_img:
                            n = _normalize_image_obj(raw_img)
                            if n:
                                img_url = n["url"]
                        steps.append({"index": idx if idx is not None else None, "text": text, "image": img_url, "caption": s.get("caption")})
                    else:
                        steps.append({"index": None, "text": str(s), "image": None})
            return {
                "query": widget_resp.get("query", query),
                "answer": widget_resp.get("answer", ""),
                "steps": steps,
                "stepsTitle": widget_resp.get("stepsTitle") or widget_resp.get("stepsTitle") or "Step-by-step instructions",
                "images": images,
                "summary": widget_resp.get("summary"),
                "summaryTitle": widget_resp.get("summaryTitle") or "Quick Summary",
                "timestamp": widget_resp.get("timestamp") or datetime.now().isoformat(),
                "confidence": widget_resp.get("confidence", 0.0),
                "results_found": widget_resp.get("results_found", 0),
                "results_used": widget_resp.get("results_used", 0),
                "has_sources": widget_resp.get("has_sources", False),
            }

        # Enhanced retrieval + re-synthesis when widget response weak
        logger.info("Widget response weak or missing — performing enhanced retrieval and step generation.")
        final_resp: Dict[str, Any] = widget_resp or {}

        try:
            # broaden Milvus results
            milvus_n = min(200, max(request.max_results * 3, 120))
            milvus_results = []
            try:
                milvus_results = await milvus_service.search_documents(query, n_results=milvus_n)
            except Exception as e:
                logger.exception("Direct milvus_service.search_documents failed: %s", e)
                milvus_results = []

            if milvus_results:
                # build contexts from top docs
                context_texts = []
                for r in milvus_results[:min(40, len(milvus_results))]:
                    content = r.get("content", "") or ""
                    if content and len(content.strip()) > 50:
                        context_texts.append(content[:5000])
                if not context_texts:
                    context_texts = [r.get("content", "")[:2000] for r in milvus_results[:3]]

                # ask ai_service to synthesize an enhanced answer
                try:
                    enhanced = await ai_service.generate_enhanced_response(query, context_texts, query_type=None)
                except Exception as e:
                    logger.exception("ai_service.generate_enhanced_response failed: %s", e)
                    enhanced = {}

                answer_text = ""
                confidence_score = 0.0
                expanded_context = ""
                if isinstance(enhanced, dict):
                    answer_text = enhanced.get("text", "") or ""
                    confidence_score = float(enhanced.get("quality_score", 0.0) or 0.0)
                    expanded_context = enhanced.get("expanded_context", "") or ""
                elif isinstance(enhanced, str):
                    answer_text = enhanced
                    confidence_score = 0.0

                # Collect candidate images from Milvus metadata
                candidate_images = []
                seen_urls = set()
                for r in milvus_results:
                    meta = r.get("metadata") or {}
                    imgs = meta.get("images") if isinstance(meta.get("images"), list) else []
                    page_url = meta.get("url") or ""
                    page_title = meta.get("title") or ""
                    for img in imgs:
                        if isinstance(img, dict):
                            n = _normalize_image_obj(img)
                            if not n:
                                continue
                            u_low = n["url"].lower()
                            if any(noise in u_low for noise in ["logo", "icon", "favicon", "sprite", "banner"]):
                                continue
                            n["source_url"] = page_url
                            n["source_title"] = page_title
                            if n["url"] in seen_urls:
                                continue
                            seen_urls.add(n["url"])
                            candidate_images.append(n)
                        elif isinstance(img, str) and img.startswith("http"):
                            n = _normalize_image_obj(img)
                            if n and n["url"] not in seen_urls:
                                seen_urls.add(n["url"])
                                n["source_url"] = page_url
                                n["source_title"] = page_title
                                candidate_images.append(n)
                        if len(candidate_images) >= 24:
                            break
                    if len(candidate_images) >= 24:
                        break

                # If no candidate images, keep empty list
                selected_images = candidate_images[:24]

                # generate a stepwise response and attach images to steps
                steps_with_images: List[Dict[str, Any]] = []
                try:
                    steps_data = await ai_service.generate_stepwise_response(query, context_texts[:3])
                except Exception as e:
                    logger.warning(f"ai_service.generate_stepwise_response failed: {e}")
                    steps_data = []

                if not steps_data:
                    if answer_text:
                        sentences = [s.strip() for s in answer_text.split(".") if s.strip()]
                        steps_data = [{"text": (s + ".") if not s.endswith(".") else s, "type": "info"} for s in sentences[:6]]
                    else:
                        steps_data = [{"text": "Unable to generate structured response.", "type": "info"}]

                # Prefer explicit step images returned by step generator; else map Nth selected image to Nth step
                step_level_added = 0
                # build a simple list of selected image URLs for mapping
                selected_image_urls = [img.get("url") for img in selected_images if img.get("url")]

                for i, step in enumerate(steps_data):
                    text = step.get("text") if isinstance(step, dict) else str(step)
                    step_entry: Dict[str, Any] = {"index": i + 1, "text": text}
                    assigned_img_url = None

                    # explicit step image
                    if isinstance(step, dict):
                        raw_img = step.get("image") or step.get("image_url") or step.get("image_data")
                        if raw_img:
                            n = _normalize_image_obj(raw_img)
                            if n:
                                assigned_img_url = n["url"]

                    # else use i-th selected image
                    if not assigned_img_url and i < len(selected_image_urls):
                        assigned_img_url = selected_image_urls[i]

                    if assigned_img_url:
                        step_entry["image"] = assigned_img_url
                        step_level_added += 1

                    steps_with_images.append(step_entry)

                # summary best-effort
                try:
                    summary_text = await ai_service.generate_summary(answer_text or "\n\n".join(context_texts[:3]), max_sentences=3, max_chars=600)
                except Exception:
                    summary_text = (answer_text[:600] + "...") if answer_text and len(answer_text) > 600 else (answer_text or "")

                final_resp = {
                    "query": query,
                    "answer": answer_text or "I found information in the knowledge base but couldn't generate a comprehensive summary.",
                    "expanded_context": expanded_context or None,
                    "step_count": len(steps_with_images),
                    "steps": steps_with_images,
                    "stepsTitle": "Step-by-step instructions",
                    "images": selected_images,
                    "sources": [],  # we could build sources similar to rag_widget if needed
                    "has_sources": False,
                    "confidence": round(confidence_score if confidence_score else 0.0, 3),
                    "search_depth": "enhanced",
                    "results_found": len(milvus_results),
                    "results_used": min(len(milvus_results), request.max_results),
                    "timestamp": datetime.now().isoformat(),
                    "summary": summary_text or "No summary available.",
                    "summaryTitle": "Quick Summary",
                }

            else:
                logger.warning("Milvus returned no additional documents during enhanced retrieval.")
                final_resp = widget_resp or {
                    "query": query,
                    "answer": "I don't have enough information in my knowledge base to answer that question.",
                    "images": [],
                    "timestamp": datetime.now().isoformat(),
                    "source": "no_results",
                    "confidence": 0.0,
                    "results_found": 0
                }

        except Exception as e:
            logger.exception("Enhanced retrieval flow failed: %s", e)
            final_resp = widget_resp or {
                "query": query,
                "answer": "I don't have enough information in my knowledge base to answer that question.",
                "images": [],
                "timestamp": datetime.now().isoformat(),
                "source": "no_results",
                "confidence": 0.0,
                "results_found": 0
            }

        # Final normalization before returning to frontend: ensure shapes are consistent
        resp = final_resp or {}
        answer = (resp.get("answer") or "").strip()

        # normalize images
        images = _normalize_images_list(resp.get("images", []), cap=24) if request.include_images else []

        # normalize steps: ensure image is string URL or null
        steps = []
        raw_steps = resp.get("steps") or []
        for s in raw_steps:
            if isinstance(s, dict):
                idx = s.get("index") or None
                text = s.get("text") or ""
                image = None
                raw_img = s.get("image")
                if raw_img:
                    # raw_img might be dict or str
                    n = _normalize_image_obj(raw_img)
                    if n:
                        image = n["url"]
                    else:
                        # maybe it's already a string URL
                        image = _to_safe_url(raw_img)
                caption = s.get("caption") or None
                steps.append({"index": idx if idx is not None else None, "text": text, "image": image, "caption": caption})
            else:
                steps.append({"index": None, "text": str(s), "image": None})

        logger.info(
            f"Returning answer (results_found={resp.get('results_found', 0)}, confidence={resp.get('confidence', 0.0):.3f}, answer_length={len(answer)}, steps={len(steps)}, images={len(images)})"
        )

        return {
            "query": query,
            "answer": answer if answer else "I found some information but couldn't generate a comprehensive answer.",
            "steps": steps,
            "stepsTitle": resp.get("stepsTitle") or resp.get("stepsTitle") or "Step-by-step instructions",
            "images": images,
            "summary": resp.get("summary"),
            "summaryTitle": resp.get("summaryTitle") or "Quick Summary",
            "timestamp": resp.get("timestamp") or datetime.now().isoformat(),
            "source": resp.get("source", "widget_retrieval"),
            "confidence": resp.get("confidence", 0.0),
            "results_found": resp.get("results_found", 0),
            "results_used": resp.get("results_used", 0),
            "has_sources": resp.get("has_sources", False)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"User query error: {e}")
        raise HTTPException(status_code=500, detail="An error occurred processing your request")


# ------------------------ Health Check ------------------------
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

        return {
            "status": "healthy" if ok else "unhealthy",
            "service": "user_chat",
            "documents_available": doc_count if ok else 0,
            "version": app.version
        }
    except Exception as e:
        logger.exception("Health check failed")
        return JSONResponse({"status": "unhealthy", "error": str(e)}, status_code=503)

# ------------------------ Widget API: token + config ------------------------
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

# ------------------------ Frontend SPA ------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
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

# ------------------------ Widget & CDN ------------------------
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

# ------------------------ Root ------------------------
@app.get("/")
async def root():
    return {
        "message": "RAG Chat - User Interface",
        "version": app.version,
        "usage": "Send POST requests to /api/chat/query"
    }

# ------------------------ Run ------------------------
if __name__ == "__main__":
    host = os.getenv("USER_APP_HOST", "0.0.0.0")
    port = int(os.getenv("USER_APP_PORT", "8001"))
    uvicorn.run(
        "user_main:app",
        host=host,
        port=port,
        reload=os.getenv("DEV_RELOAD", "false").lower() in ("1", "true"),
        workers=int(os.getenv("USER_WORKERS", "2")),
        log_level=os.getenv("LOG_LEVEL", "info"),
    )
