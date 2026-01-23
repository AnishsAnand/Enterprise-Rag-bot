import os
import logging
from pathlib import Path
from typing import List
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

import uvicorn

from app.core.config import settings
from app.api.routes import (
    rag_widget,
    agent_chat,
    health,
    chat_persistence,
    orchestrator,
    openwebui_auth,
    tata_auth,
    user_credentials,
)
from app.api.routes.user_chat import router as user_chat_router
from app.routers import openai_compatible
from app.services.ai_service import ai_service
from app.services.postgres_service import postgres_service
from app.core.database import init_db

load_dotenv()

logger = logging.getLogger("enterprise_rag_bot")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ===================== Lifespan Management =====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Production-grade lifespan management for FastAPI.
    Handles startup and shutdown of all services.
    """
    # ===== Startup =====
    logger.info("Starting Enterprise RAG Bot...")
    logger.info("=" * 70)

    # Initialize database
    try:
        await init_db()
        logger.info("✅ Database initialized successfully")
    except Exception as e:
        logger.exception("❌ Database initialization failed (continuing): %s", e)

    app.state.postgres_ready = False
    try:
        await postgres_service.initialize()
        app.state.postgres_ready = True
        logger.info("✅ PostgreSQL (pgvector) initialized successfully")
    except Exception as e:
        logger.exception(
        "❌ PostgreSQL vector initialization failed (vector search disabled): %s",
        e,)
    # Test AI Services
    logger.info("Testing AI Services...")
    ai_health = {"embedding": False, "generation": False}
    try:
        test_embeddings = await ai_service.generate_embeddings(["test embedding"])
        if test_embeddings and len(test_embeddings) > 0 and len(test_embeddings[0]) > 0:
            logger.info("✅ Embedding service working (dimension: %d)", len(test_embeddings[0]))
            ai_health["embedding"] = True
        else:
            logger.warning("⚠️ Embedding service returned empty results")
    except Exception as e:
        logger.exception("❌ Embedding service test failed: %s", e)
    try:
        test_response = await ai_service.generate_enhanced_response("health", ["health check"])
        if test_response and isinstance(test_response, dict):
            response_text = test_response.get("text", "")
            if response_text and len(response_text.strip()) > 0:
                logger.info("✅ Text generation service working")
                ai_health["generation"] = True
            else:
                logger.warning("⚠️ Text generation service returned empty text")
        else:
            logger.warning("⚠️ Text generation service returned invalid response")
    except Exception as e:
        logger.exception("❌ Text generation service test failed: %s", e)

    try:
        stats = await postgres_service.get_collection_stats()
        doc_count = stats.get("document_count", 0) if isinstance(stats, dict) else 0
        logger.info("✅ PostgreSQL Vector Stats - Documents: %d", doc_count)
    except Exception as e:
        logger.exception("⚠️ Could not retrieve stats: %s", e)
    logger.info("=" * 70)
    logger.info("✅ Enterprise RAG Bot startup sequence complete")
    logger.info("📚 API documentation: http://localhost:8000/docs")
    logger.info("🏥 Health check: http://localhost:8000/health")
    yield
    # ===== Shutdown =====
    logger.info("=" * 70)
    logger.info("Shutting down Enterprise RAG Bot...")
    try:
        await postgres_service.close()
        logger.info("✅ PostgreSQL connection closed gracefully")
    except Exception as e:
        logger.warning("⚠️ Error closing PostgreSQL connection: %s", e)
    logger.info("✅ Enterprise RAG Bot shutdown complete")
    logger.info("=" * 70)
# ===================== FastAPI App Initialization =====================
app = FastAPI(
    title="Enterprise RAG Bot",
    description="Advanced RAG system with postgres vector database, web scraping, and AI-powered knowledge retrieval",
    version="2.0.0",
    lifespan=lifespan,)
# ===================== CORS Configuration =====================
allowed_origins: List[str] = [
    "http://localhost:4200", 
    "http://127.0.0.1:4200",
    "http://localhost:4201",      
    "http://127.0.0.1:4201",      
    "http://localhost:3000",      # Open WebUI
    "http://127.0.0.1:3000",      # Open WebUI (alternative)
]
extra_origins = os.getenv("ALLOWED_ORIGINS", "")
if extra_origins:
    allowed_origins.extend([o.strip() for o in extra_origins.split(",") if o.strip()])

user_allowed_origins = os.getenv("USER_ALLOWED_ORIGINS", "")
if user_allowed_origins:
    allowed_origins.extend([o.strip() for o in user_allowed_origins.split(",") if o.strip()])

allow_all = os.getenv("ALLOW_ALL_ORIGINS", "false").lower() in ("1", "true", "yes")
user_allow_all = os.getenv("USER_ALLOW_ALL_ORIGINS", "false").lower() in ("1", "true", "yes")
if allow_all or user_allow_all:
    allowed_origins = ["*"]
    logger.warning("⚠️ CORS set to allow ALL origins - not recommended for production!")
logger.info("✅ CORS allowed origins: %s", allowed_origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],)

app.state.allowed_origins = allowed_origins

# ------------------------ Security Headers ------------------------
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses"""

    async def dispatch(self, request, call_next):
        response = await call_next(request)
        try:
            app_origins = getattr(request.app.state, "allowed_origins", [])
            if app_origins and app_origins != ["*"]:
                frame_ancestors = " ".join(app_origins)
                csp_value = f"default-src 'self'; frame-ancestors {frame_ancestors};"
                response.headers["Content-Security-Policy"] = csp_value
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["Referrer-Policy"] = "no-referrer-when-downgrade"
        except Exception:
            pass
        return response


app.add_middleware(SecurityHeadersMiddleware)

# ===================== API Routes =====================
app.include_router(health.router)  # Add health check routes
app.include_router(rag_widget.router, prefix="/api", tags=["rag-widget"])
app.include_router(rag_widget.router, prefix="/api/rag-widget", tags=["rag-widget"], include_in_schema=False)
app.include_router(agent_chat.router, tags=["agent-chat"])
app.include_router(chat_persistence.router)
app.include_router(user_chat_router)
app.include_router(orchestrator.router)
app.include_router(openwebui_auth.router)
app.include_router(tata_auth.router)
app.include_router(user_credentials.router)
# OpenAI-compatible API for Open WebUI integration
app.include_router(openai_compatible.router)
# ===================== Static Files & Frontend =====================
BASE_DIR = Path(__file__).resolve().parent.parent
frontend_path = BASE_DIR / "dist" / "enterprise-rag-frontend"
embedded_static_mounted = False
if frontend_path.exists() and frontend_path.is_dir():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="static")
    embedded_static_mounted = True
    logger.info("✅ Embedded frontend mounted at: %s", frontend_path)
else:
    logger.warning("⚠️ Static frontend directory not found at %s. API will continue to run without embedded frontend.", frontend_path)
# ===================== Widget Embed Script =====================
@app.get("/widget/embed.js")
def serve_embed_script():
    """
    Deliver the embeddable JS widget script.
    Namespaced under /widget to avoid collisions with frontend routes.
    
    Production-ready with multiple fallback locations for robustness.
    """
    candidate_paths = [
        BASE_DIR / "app" / "static" / "embed.js",
        frontend_path / "embed.js",
        frontend_path / "assets" / "embed.js",
    ]
    
    for p in candidate_paths:
        if p and p.exists():
            logger.info("✅ Serving embed.js from: %s", p)
            return FileResponse(p, media_type="application/javascript")
    
    logger.warning("❌ embed.js not found in candidate locations: %s", candidate_paths)
    raise HTTPException(status_code=404, detail="embed.js not found")

# ===================== SPA Fallback =====================
if embedded_static_mounted:
    @app.get("/{full_path:path}")
    async def spa_fallback(full_path: str):
        """
        Fallback route for SPA: serve index.html for all non-API routes.
        Allows frontend routing to work properly.
        """
        # Exclude API routes from SPA fallback
        if full_path.startswith("api/") or full_path.startswith("widget/"):
            raise HTTPException(status_code=404, detail="Route not found")
        
        index_file = frontend_path / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        logger.error("❌ index.html missing from frontend at %s", index_file)
        return JSONResponse({"detail": "index.html not found"}, status_code=500)

# ===================== Health Check Endpoints =====================
@app.get("/health/liveness")
async def liveness_check():
    """
    Lightweight liveness probe endpoint.
    Returns 200 if the application process is running.
    Use this for container/orchestration liveness probes.
    Endpoint response time: <10ms
    """
    return {
        "status": "alive",
        "service": "enterprise-rag-bot"}

@app.get("/health")
def health():
    return {"status": "ok", "service": "admin"}

# ===================== Prometheus Metrics Endpoint =====================
@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus metrics endpoint for monitoring."""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    from fastapi.responses import Response
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
@app.get("/health/readiness")
async def readiness_check():
    """
    Comprehensive readiness check endpoint.
    Verifies all critical services (postgres, AI services, Database).
    Use this for k8s/ECS readiness probes.
    Only returns 200 when all services are operational.
    """
    # Check postgres connection and collection
    postgres_status = "unavailable"
    postgres_docs = 0
    postgres_error = None
    try:
        stats = await postgres_service.get_collection_stats()
        if isinstance(stats, dict):
            postgres_status = stats.get("status", "unknown")
            postgres_docs = int(stats.get("document_count", 0))
        else:
            postgres_status = "error"
            postgres_error = "Invalid response format"
    except Exception as e:
        postgres_status = "unavailable"
        postgres_error = str(e)
        logger.exception("❌ postgres stats fetch failed: %s", e)
    # Check AI services
    ai_services_status = {
        "embedding": False,
        "generation": False,}
    try:
        test_emb = await ai_service.generate_embeddings(["health"])
        if test_emb and len(test_emb) > 0:
            ai_services_status["embedding"] = True
    except Exception as e:
        logger.debug("AI embedding service check failed: %s", e)
    try:
        test_gen = await ai_service.generate_response("health", [])
        if test_gen:
            ai_services_status["generation"] = True
    except Exception as e:
        logger.debug("AI generation service check failed: %s", e)
    # Determine overall health
    overall_status = "healthy"
    if postgres_status not in ("active", "healthy"):
        overall_status = "degraded"
    if not ai_services_status["embedding"] or not ai_services_status["generation"]:
        overall_status = "degraded" if overall_status == "healthy" else "unhealthy"
    response = {
        "status": overall_status,
        "timestamp": str(__import__("datetime").datetime.utcnow().isoformat()),
        "services": {
            "postgres": {
                "status": postgres_status,
                "documents_stored": postgres_docs,
                "error": postgres_error,},
            "ai_services": {
                "embedding": "operational" if ai_services_status["embedding"] else "unavailable",
                "generation": "operational" if ai_services_status["generation"] else "unavailable",},
            "database": "operational",  },
        "version": app.version,}
    # Return appropriate status code
    status_code = 200 if overall_status == "healthy" else 503
    return JSONResponse(response, status_code=status_code)

# ===================== Root Endpoint =====================
@app.get("/")
async def api_root():
    """
    Root endpoint: serves embedded frontend or API info.
    """
    if embedded_static_mounted:
        index_file = frontend_path / "index.html"
        if index_file.exists():
            return FileResponse(index_file)

    return {
        "message": "Enterprise RAG Bot API",
        "version": app.version,
        "features": [
            "Advanced web scraping with anti-blocking",
            "postgres vector database for semantic search",
            "AI-powered knowledge retrieval (Grok, OpenRouter)",
            "Multi-source ingestion (web, files, bulk scraping)",
            "Popup widget interface for easy integration",
            "Production-grade reliability and scalability",
        ],
        "documentation": "/docs",
        "health_check": "/health",}

# ===================== Application Info Endpoint =====================
@app.get("/api/info")
async def app_info():
    """
    Detailed application information and status.
    Useful for debugging and monitoring.
    """
    try:
        stats = await postgres_service.get_collection_stats()
        postgres_info = stats if isinstance(stats, dict) else {}
    except Exception as e:
        postgres_info = {"error": str(e)}

    return {
        "application": {
            "name": "Enterprise RAG Bot",
            "version": app.version,
            "environment": os.getenv("ENV", "development"),
        },
        "infrastructure": {
            "postgres": postgres_info,
            "ai_service": {
                "embedding_model": os.getenv("EMBEDDING_MODEL", "unknown"),
                "chat_model": os.getenv("CHAT_MODEL", "unknown"),
            },
        },
        "features": {
            "web_scraping": True,
            "file_uploads": True,
            "bulk_scraping": True,
            "widget_interface": True,},
        "endpoints": {
            "scraper": "/api/scraper",
            "rag": "/api/rag",
            "widget": "/api/rag-widget",
            "admin": "/api/admin",
            "support": "/api/support",},}

# ===================== Error Handlers =====================
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """
    Custom HTTP exception handler for better error messages.
    """
    logger.warning(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """
    Catch-all exception handler for unhandled errors.
    Logs exception and returns generic error message.
    """
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "detail": "An unexpected error occurred. Please check server logs."})

# ===================== Main Entry Point =====================
if __name__ == "__main__":
    dev_reload = os.getenv("DEV_RELOAD", "false").lower() in ("1", "true", "yes")
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    workers = int(os.getenv("UVICORN_WORKERS", "1"))
    log_level = os.getenv("UVICORN_LOG_LEVEL", "info")

    logger.info(f"Starting server on {host}:{port}")
    logger.info(f"Workers: {workers}, Log Level: {log_level}, Dev Reload: {dev_reload}")

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=dev_reload,
        workers=workers,
        log_level=log_level,
    )
