"""
Enterprise RAG Bot - Admin API Server (Port 8000)
Provides scraping, RAG, admin tools, and OpenAI-compatible endpoints.
"""

import os
import logging
import time
from pathlib import Path
from typing import List
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware
import uvicorn
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from app.core.config import settings
from app.api.routes import scraper, rag, admin, support, rag_widget, agent_chat
from app.routers import openai_compatible, agentic_metrics
from app.services.milvus_service import milvus_service
from app.services.ai_service import ai_service
from app.api.routes.auth import router as auth_router
from app.api.routes import user_credentials
from app.core.database import init_db
from app.services.prometheus_metrics import metrics

load_dotenv()

logger = logging.getLogger("enterprise_rag_bot")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ── Prometheus Middleware ───────────────────────────────────────────────────
class PrometheusMiddleware(BaseHTTPMiddleware):
    """Tracks HTTP request metrics (count, duration, sizes) for Prometheus."""
    
    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/metrics":
            return await call_next(request)
        
        # Normalize path (replace IDs with placeholders)
        path_parts = request.url.path.split('/')
        normalized = []
        for part in path_parts:
            if part and (len(part) > 20 or part.isdigit() or (len(part) == 36 and part.count('-') == 4)):
                normalized.append('{id}')
            else:
                normalized.append(part)
        endpoint = '/'.join(normalized) or '/'
        method = request.method
        start_time = time.time()
        
        if content_length := request.headers.get('content-length'):
            metrics.http_request_size.labels(method=method, endpoint=endpoint).observe(int(content_length))
        
        response = await call_next(request)
        duration = time.time() - start_time
        
        metrics.http_requests_total.labels(method=method, endpoint=endpoint, status_code=str(response.status_code)).inc()
        metrics.http_request_duration.labels(method=method, endpoint=endpoint).observe(duration)
        
        if response_size := response.headers.get('content-length'):
            metrics.http_response_size.labels(method=method, endpoint=endpoint).observe(int(response_size))
        
        return response


# ── Lifespan Management ─────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup (DB, Milvus, AI services) and shutdown."""
    logger.info("Starting Enterprise RAG Bot...")
    logger.info("=" * 60)

    # Initialize database
    try:
        await init_db()
        logger.info("✅ Database initialized")
    except Exception as e:
        logger.exception("❌ Database init failed: %s", e)

    # Initialize Milvus
    try:
        await milvus_service.initialize()
        logger.info("✅ Milvus initialized")
    except Exception as e:
        logger.exception("❌ Milvus init failed: %s", e)

    # Test AI services
    try:
        embeddings = await ai_service.generate_embeddings(["test"])
        if embeddings and len(embeddings[0]) > 0:
            logger.info("✅ Embedding service OK (dim=%d)", len(embeddings[0]))
    except Exception as e:
        logger.warning("❌ Embedding test failed: %s", e)

    try:
        resp = await ai_service.generate_enhanced_response("Hello", ["Test"])
        if resp and resp.get("text"):
            logger.info("✅ Generation service OK")
    except Exception as e:
        logger.warning("❌ Generation test failed: %s", e)

    # Log Milvus stats
    try:
        stats = await milvus_service.get_collection_stats()
        doc_count = stats.get("document_count", 0) if isinstance(stats, dict) else 0
        logger.info("✅ Milvus docs: %d", doc_count)
    except Exception:
        pass

    logger.info("=" * 60)
    logger.info("✅ Startup complete | Docs: /docs | Health: /health")

    yield

    # Shutdown
    logger.info("Shutting down...")
    try:
        await milvus_service.close()
        logger.info("✅ Milvus closed")
    except Exception as e:
        logger.warning("⚠️ Milvus close error: %s", e)


# ── FastAPI App ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Enterprise RAG Bot",
    description="RAG system with Milvus, web scraping, and AI-powered retrieval",
    version="2.0.0",
    lifespan=lifespan,
)

# ── CORS Configuration ──────────────────────────────────────────────────────
allowed_origins: List[str] = [
    "http://localhost:4200", "http://127.0.0.1:4200",  # Angular admin
    "http://localhost:4201", "http://127.0.0.1:4201",  # Angular user
    "http://localhost:3000", "http://127.0.0.1:3000",  # Open WebUI
    "http://localhost:8000", "http://127.0.0.1:8000",  # Same origin
]

# Dev port forwarding range
for port in range(57000, 58000):
    allowed_origins.append(f"http://localhost:{port}")

if extra := os.getenv("ALLOWED_ORIGINS", ""):
    allowed_origins.extend([o.strip() for o in extra.split(",") if o.strip()])

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics middleware
if os.getenv("ENABLE_PROMETHEUS_METRICS", "true").lower() in ("1", "true", "yes"):
    app.add_middleware(PrometheusMiddleware)

# ── Routes ──────────────────────────────────────────────────────────────────
app.include_router(scraper.router, prefix="/api/scraper", tags=["scraper"])
app.include_router(rag.router, prefix="/api/rag", tags=["rag"])
app.include_router(rag_widget.router, prefix="/api/rag-widget", tags=["rag-widget"])
app.include_router(admin.router, prefix="/api/admin", tags=["admin"])
app.include_router(support.router, prefix="/api/support", tags=["support"])
app.include_router(agent_chat.router, tags=["agent-chat"])
app.include_router(openai_compatible.router)
app.include_router(auth_router)
app.include_router(user_credentials.router)
app.include_router(agentic_metrics.router)


# ── Static Files & Frontend ─────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
frontend_path = BASE_DIR / "dist" / "enterprise-rag-frontend"
embedded_static_mounted = False

if frontend_path.exists() and frontend_path.is_dir():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="static")
    embedded_static_mounted = True
    logger.info("✅ Frontend mounted: %s", frontend_path)


# ── Endpoints ───────────────────────────────────────────────────────────────
@app.get("/metrics", include_in_schema=False)
async def prometheus_metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/widget/embed.js")
def serve_embed_script():
    """Serve the embeddable widget script."""
    candidates = [
        BASE_DIR / "app" / "static" / "embed.js",
        frontend_path / "embed.js",
        frontend_path / "assets" / "embed.js",
    ]
    for p in candidates:
        if p and p.exists():
            return FileResponse(p, media_type="application/javascript")
    raise HTTPException(status_code=404, detail="embed.js not found")


@app.get("/auth")
@app.get("/auth/login")
async def auth_redirect():
    """Redirect /auth routes to login page (OpenWebUI compatibility)."""
    from fastapi.responses import RedirectResponse
    login_url = os.getenv("OPENWEBUI_LOGIN_URL", "http://localhost:3000")
    return RedirectResponse(url=f"{login_url}/login", status_code=302)


if embedded_static_mounted:
    @app.get("/{full_path:path}")
    async def spa_fallback(full_path: str):
        """SPA fallback: serve index.html for non-API routes."""
        if full_path.startswith("api/") or full_path.startswith("widget/"):
            raise HTTPException(status_code=404, detail="Route not found")
        index_file = frontend_path / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return JSONResponse({"detail": "index.html not found"}, status_code=500)


@app.get("/health/liveness")
async def liveness_check():
    """Lightweight liveness probe."""
    return {"status": "alive", "service": "enterprise-rag-bot"}


@app.get("/health")
async def readiness_check():
    """Readiness check - verifies Milvus and AI services."""
    from datetime import datetime
    
    milvus_status, milvus_docs, milvus_error = "unavailable", 0, None
    try:
        stats = await milvus_service.get_collection_stats()
        if isinstance(stats, dict):
            milvus_status = stats.get("status", "unknown")
            milvus_docs = int(stats.get("document_count", 0))
    except Exception as e:
        milvus_error = str(e)

    ai_status = {"embedding": False, "generation": False}
    try:
        if await ai_service.generate_embeddings(["health"]):
            ai_status["embedding"] = True
    except Exception:
        pass
    try:
        if await ai_service.generate_response("health", []):
            ai_status["generation"] = True
    except Exception:
        pass

    overall = "healthy"
    if milvus_status not in ("active", "healthy"):
        overall = "degraded"
    if not ai_status["embedding"] or not ai_status["generation"]:
        overall = "degraded" if overall == "healthy" else "unhealthy"

    return JSONResponse({
        "status": overall,
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "milvus": {"status": milvus_status, "documents": milvus_docs, "error": milvus_error},
            "ai_services": {k: "operational" if v else "unavailable" for k, v in ai_status.items()},
            "database": "operational",
        },
        "version": app.version,
    }, status_code=200 if overall == "healthy" else 503)


@app.get("/")
async def api_root():
    """Root endpoint: serves frontend or API info."""
    if embedded_static_mounted:
        index_file = frontend_path / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
    return {
        "message": "Enterprise RAG Bot API",
        "version": app.version,
        "documentation": "/docs",
        "health_check": "/health",
    }


@app.get("/api/info")
async def app_info():
    """Application information and status."""
    try:
        stats = await milvus_service.get_collection_stats()
        milvus_info = stats if isinstance(stats, dict) else {}
    except Exception as e:
        milvus_info = {"error": str(e)}

    return {
        "application": {"name": "Enterprise RAG Bot", "version": app.version},
        "infrastructure": {"milvus": milvus_info},
        "endpoints": {
            "scraper": "/api/scraper", "rag": "/api/rag",
            "widget": "/api/rag-widget", "admin": "/api/admin",
        },
    }


# ── Exception Handlers ──────────────────────────────────────────────────────
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.warning(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(status_code=500, content={"error": "Internal server error"})


# ── Main Entry Point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("DEV_RELOAD", "false").lower() in ("1", "true", "yes"),
        workers=int(os.getenv("UVICORN_WORKERS", "1")),
        log_level=os.getenv("UVICORN_LOG_LEVEL", "info"),
    )
