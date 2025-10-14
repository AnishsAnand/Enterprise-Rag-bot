import os
import logging
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

import uvicorn

from app.core.config import settings
from app.api.routes import scraper, rag, admin, support, rag_widget
from app.services.chroma_service import chroma_service
from app.services.ai_service import ai_service
from app.api.routes.auth import router as auth_router
from app.core.database import init_db

load_dotenv()

logger = logging.getLogger("enterprise_rag_bot")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI(
    title="Enterprise RAG Bot",
    description="Advanced RAG system with web scraping and AI-powered knowledge retrieval",
    version="2.0.0",
)

@app.on_event("startup")
async def initialize_database():
    try:
        await init_db()
        logger.info("Database initialized")
    except Exception as e:

        logger.exception("Database initialization failed (continuing): %s", e)


@app.on_event("startup")
async def startup_services():
    logger.info("Starting Enterprise RAG Bot...")
    logger.info("=" * 50)

    try:
        await chroma_service.initialize()
        logger.info("ChromaDB initialized successfully")
    except Exception as e:
        logger.exception("ChromaDB initialization failed (continuing without): %s", e)

    logger.info("Testing AI Services:")
    try:
        test_embeddings = await ai_service.generate_embeddings(["test"])
        if test_embeddings and len(test_embeddings[0]) > 0:
            logger.info("Embedding service working")
        else:
            logger.warning("Embedding service returned empty results")
    except Exception as e:
        logger.exception("Embedding service test failed: %s", e)

    try:
        test_response = await ai_service.generate_enhanced_response("Hello", ["Test context"])
        if test_response:
            logger.info("Text generation service working")
        else:
            logger.warning("Text generation service returned empty response")
    except Exception as e:
        logger.exception("Text generation service test failed: %s", e)

    logger.info("=" * 50)
    logger.info("Enterprise RAG Bot startup sequence complete")
    logger.info("API docs at: http://localhost:8000/docs")  

allowed_origins: List[str] = [
    "http://localhost:4200",
    "http://127.0.0.1:4200",
]

extra_origins = os.getenv("ALLOWED_ORIGINS", "")
if extra_origins:
    allowed_origins.extend([o.strip() for o in extra_origins.split(",") if o.strip()])

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(scraper.router, prefix="/api/scraper", tags=["scraper"])
app.include_router(rag.router, prefix="/api/rag", tags=["rag"])
app.include_router(rag_widget.router, prefix="/api/rag-widget", tags=["rag-widget"])
app.include_router(admin.router, prefix="/api/admin", tags=["admin"])
app.include_router(support.router, prefix="/api/support", tags=["support"])
app.include_router(auth_router)


BASE_DIR = Path(__file__).resolve().parent.parent  
frontend_path = BASE_DIR / "dist" / "enterprise-rag-frontend"
embedded_static_mounted = False

if frontend_path.exists() and frontend_path.is_dir():

    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="static")
    embedded_static_mounted = True
    logger.info("Embedded frontend mounted at: %s", frontend_path)
else:
    logger.warning("Static frontend directory not found at %s. API will continue to run without embedded frontend.", frontend_path)

@app.get("/widget/embed.js")
def serve_embed_script():
    """
    Deliver the embeddable JS widget script. 
    Namespaced under /widget to avoid collisions.
    """
    candidate_paths = [
        BASE_DIR / "app" / "static" / "embed.js", 
        frontend_path / "embed.js",              
        frontend_path / "assets" / "embed.js",    
    ]
    for p in candidate_paths:
        if p and p.exists():
            return FileResponse(p, media_type="application/javascript")
    logger.warning("embed.js not found in any candidate locations: %s", candidate_paths)
    raise HTTPException(status_code=404, detail="embed.js not found")

if embedded_static_mounted:
    @app.get("/{full_path:path}")
    async def spa_fallback(full_path: str):
        index_file = frontend_path / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        logger.error("index.html missing from frontend at %s", index_file)
        return JSONResponse({"detail": "index.html not found"}, status_code=500)


@app.get("/health/liveness")
async def liveness_check():
    """
    Lightweight endpoint: always return 200.
    Use this for container liveness probes.
    """
    return {"status": "alive"}


@app.get("/health")
async def readiness_check():
    """
    Readiness check: verifies Chroma and AI services.
    Use this for k8s/ecs readiness probes.
    """
    chroma_status = "healthy"
    chroma_docs = 0
    try:
        stats = await chroma_service.get_collection_stats()
        chroma_docs = int(stats.get("document_count", 0)) if isinstance(stats, dict) else 0
    except Exception:
        chroma_status = "unavailable"
        logger.exception("ChromaDB stats fetch failed")

    ai_services = []
    try:
        if getattr(ai_service, "ollama_client", None):
            ai_services.append("ollama")
        if getattr(ai_service, "openrouter_client", None):
            ai_services.append("openrouter")
        if getattr(ai_service, "voyage_client", None):
            ai_services.append("voyage")
    except Exception:
        logger.exception("Failed to inspect AI clients")

    return {
        "status": "healthy",
        "services": {
            "chromadb": chroma_status,
            "ai_services": ai_services,
            "documents_stored": chroma_docs,
        },
        "version": app.version,
    }


@app.get("/")
async def api_root():
    if embedded_static_mounted:
        index_file = frontend_path / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
 
    return {
        "message": "Enterprise RAG Bot API",
        "version": app.version,
        "features": [
            "Advanced web scraping with anti-blocking",
            "AI-powered knowledge retrieval",
            "ChromaDB vector storage",
            "Multi-model AI support (Ollama, OpenRouter, Voyage)",
            "Popup widget interface",
        ],
    }

if __name__ == "__main__":
    dev_reload = os.getenv("DEV_RELOAD", "false").lower() in ("1", "true", "yes")
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=dev_reload,
        workers=int(os.getenv("UVicorn_WORKERS", "1")),
        log_level=os.getenv("UVICORN_LOG_LEVEL", "info"),
    )
