# user_main.py - PRODUCTION-READY with OpenWebUI Step-by-Step Formatting
"""
Enterprise RAG User Interface with OpenWebUI Integration

PRODUCTION UPDATES:
- ✅ Integrated OpenWebUI formatter for step-by-step instructions with images
- ✅ Enhanced response formatting with markdown optimization
- ✅ Proper image embedding in steps
- ✅ Agent response formatting with tables
- ✅ Maintains backward compatibility with existing API
"""

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
from app.routers import openai_compatible
from app.core.database import init_db
from app.services.postgres_service import postgres_service
from app.services.ai_service import ai_service

# ============================================================================
# PRODUCTION FIX: Import OpenWebUI Formatter
# ============================================================================
from app.services.openwebui_formatter import (
    format_for_openwebui,
    format_agent_response_for_openwebui,
    format_error_for_openwebui
)

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

# ========================================================================================
# PRODUCTION FIX #1: Enhanced Image Normalization for OpenWebUI
# These functions ensure images display properly in OpenWebUI by handling all formats
# ========================================================================================

def _to_safe_url(u: Any) -> Optional[str]:
    """Safely convert any input to a valid URL string"""
    if not u:
        return None
    try:
        s = str(u).strip()
    except Exception:
        return None
    if not s:
        return None
    # Only return if it's a valid HTTP(S) URL
    if s.startswith('http://') or s.startswith('https://'):
        return s
    return None


def _normalize_image_for_display(img: Any) -> Optional[str]:
    """
    Extract image URL from any format and return just the URL string.
    OpenWebUI needs simple URL strings in the response.
    
    Handles:
    - String URLs: "https://example.com/image.png"
    - Dict with url key: {"url": "https://...", "alt": "..."}
    - Nested dict: {"data": {"url": "https://..."}}
    - Various key names: url, src, image, image_url, href
    
    Returns:
        str: Valid HTTP(S) URL or None
    """
    if not img:
        return None
    
    # Already a URL string
    if isinstance(img, str):
        url = img.strip()
        if url.startswith('http://') or url.startswith('https://'):
            return url
        return None
    
    # Dict with various possible keys
    if isinstance(img, dict):
        # Try common keys
        for key in ['url', 'src', 'image', 'image_url', 'href']:
            if key in img:
                url_val = img[key]
                if isinstance(url_val, str):
                    url = url_val.strip()
                    if url.startswith('http://') or url.startswith('https://'):
                        return url
        
        # Check nested data
        if 'data' in img and isinstance(img['data'], dict):
            if 'url' in img['data']:
                url_val = img['data']['url']
                if isinstance(url_val, str):
                    url = url_val.strip()
                    if url.startswith('http://') or url.startswith('https://'):
                        return url
    
    return None


def _format_steps_for_display(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Format steps for OpenWebUI display with proper image handling.
    Each step should have: text, step_number, type, and optional image URL.
    
    Args:
        steps: List of step dictionaries from AI service
        
    Returns:
        List of formatted step dictionaries ready for OpenWebUI
    """
    formatted_steps = []
    
    for idx, step in enumerate(steps, 1):
        if not isinstance(step, dict):
            # Handle string steps
            formatted_steps.append({
                "step_number": idx,
                "text": str(step),
                "image": None,
                "type": "info"
            })
            continue
        
        # Extract text (try multiple possible keys)
        text = step.get('text') or step.get('content') or step.get('description') or ''
        if not text or not text.strip():
            continue
        
        # Extract and normalize image
        image_url = None
        raw_image = step.get('image') or step.get('image_url') or step.get('img')
        if raw_image:
            image_url = _normalize_image_for_display(raw_image)
        
        formatted_steps.append({
            "step_number": step.get('step_number') or step.get('index') or idx,
            "text": text.strip(),
            "image": image_url,  # Will be None or valid URL string
            "type": step.get('type', 'action')
        })
    
    return formatted_steps


def _normalize_image_obj(img: Any) -> Optional[Dict[str, Any]]:
    """
    Normalize image entries to standard format: {url, alt, caption, source_url}
    
    Args:
        img: Image in any format (str, dict, or nested structure)
        
    Returns:
        Dict with standardized image fields or None
    """
    if not img:
        return None
    
    # Handle string URLs
    if isinstance(img, (str, bytes)):
        url = _to_safe_url(img)
        if url:
            return {
                "url": url,
                "alt": "",
                "caption": "",
                "source_url": ""
            }
        return None
    
    # Handle dict format
    if isinstance(img, dict):
        # Try to extract URL from various keys
        url = img.get("url") or img.get("src") or img.get("image") or None
        
        # Check nested data structure
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


def _normalize_images_list(images: Any, cap: int = 24) -> List[Dict[str, Any]]:
    """
    Normalize a list of images to standard format with deduplication.
    
    Args:
        images: Images in any format (list, single item, or None)
        cap: Maximum number of images to return
        
    Returns:
        List of normalized image dicts
    """
    out: List[Dict[str, Any]] = []
    
    if not images:
        return out
    
    # Handle single string/bytes
    if isinstance(images, (str, bytes)):
        n = _normalize_image_obj(images)
        if n:
            out.append(n)
        return out
    
    # Handle single dict
    if not isinstance(images, (list, tuple)):
        n = _normalize_image_obj(images)
        if n:
            out.append(n)
        return out
    
    # Handle list - deduplicate by URL
    seen = set()
    for it in images:
        norm = _normalize_image_obj(it)
        if norm and norm["url"] not in seen:
            seen.add(norm["url"])
            out.append(norm)
        if len(out) >= cap:
            break
    
    return out


# ========================================================================================
# PRODUCTION FIX #2: Improved Weak Response Detection
# Ensures agent responses and high-quality AI answers aren't rejected
# ========================================================================================

def _is_weak_widget_response(resp: Optional[dict]) -> bool:
    """
    Determine if widget response is too weak to use.
    
    A response is considered STRONG if:
    1. It has a meaningful answer (30+ chars)
    2. It's from the agent manager (always trusted)
    3. It has results + decent confidence (1+ results, 0.3+ confidence)
    4. It has high confidence even without results (0.7+ confidence, 100+ chars)
    
    Args:
        resp: Response dictionary from widget_query
        
    Returns:
        bool: True if response is weak and should trigger fallback
    """
    if not resp or not isinstance(resp, dict):
        return True

    answer = (resp.get("answer") or "").strip()
    results_found = int(resp.get("results_found", 0))
    confidence = float(resp.get("confidence", 0.0))
    routed_to = resp.get("routed_to", "")
    
    # CRITICAL FIX: Agent responses are ALWAYS considered strong
    # Agent responses include cluster lists, endpoint data, etc.
    if routed_to == "agent_manager":
        logger.info(f"✅ Agent response detected - considered STRONG (routed_to={routed_to})")
        return False
    
    # Need meaningful answer text
    if len(answer) < 30:
        logger.info(f"❌ Weak response: answer too short ({len(answer)} chars)")
        return True
    
    # Check if we have supporting data
    has_results = results_found >= 1
    has_decent_confidence = confidence >= 0.3
    
    # Strong if we have both results and confidence
    if has_results and has_decent_confidence:
        logger.info(f"✅ Strong response: {results_found} results, {confidence:.2f} confidence")
        return False
    
    # IMPROVED: High confidence AI answers are also strong
    # This handles cases where AI generates good answers from training data
    if confidence >= 0.7 and len(answer) >= 100:
        logger.info(f"✅ Strong AI response: {confidence:.2f} confidence, {len(answer)} chars")
        return False
    
    # Otherwise, considered weak
    logger.info(f"❌ Weak response: results={results_found}, confidence={confidence:.2f}, answer_length={len(answer)}")
    return True


# ------------------------ FastAPI Lifespan ------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Production-grade lifespan management with proper error handling"""
    # ===== Startup =====
    logger.info("=" * 70)
    logger.info("🚀 Starting RAG Chat - User Interface")
    logger.info("=" * 70)
    
    # Initialize database
    try:
        await init_db()
        logger.info("✅ Database initialized")
    except Exception as e:
        logger.exception("❌ Database initialization failed (continuing): %s", e)

    # Initialize PostgreSQL vector store
    try:
        await postgres_service.initialize()
        logger.info("✅ PostgreSQL (pgvector) connected successfully")
    except Exception as e:
        logger.exception("❌ PostgreSQL initialization failed (continuing): %s", e)

    # Test AI services
    try:
        test_embeddings = await ai_service.generate_embeddings(["test"])
        if test_embeddings and len(test_embeddings[0]) > 0:
            logger.info("✅ AI services operational (embedding dimension: %d)", len(test_embeddings[0]))
        else:
            logger.warning("⚠️ AI service returned empty embeddings during startup test")
    except Exception as e:
        logger.exception("❌ AI service initialization failed (continuing): %s", e)

    logger.info("=" * 70)
    logger.info("✅ User app startup complete - Ready to serve requests")
    logger.info("=" * 70)
    
    yield
    
    # ===== Shutdown =====
    logger.info("=" * 70)
    logger.info("🛑 Shutting down user app...")
    try:
        await postgres_service.close()
        logger.info("✅ PostgreSQL connection closed")
    except Exception as e:
        logger.warning("⚠️ Error closing PostgreSQL: %s", e)
    logger.info("✅ Shutdown complete")
    logger.info("=" * 70)


# ------------------------ FastAPI App ------------------------
app = FastAPI(
    title="RAG Chat - User Interface",
    description="End-user chat interface for RAG knowledge base with OpenWebUI support",
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
        "http://localhost:3000",  # OpenWebUI
        "http://127.0.0.1:3000",  # OpenWebUI
    ]

if os.getenv("USER_ALLOW_ALL_ORIGINS", "false").lower() in ("1", "true", "yes"):
    allowed_origins = ["*"]
    logger.warning("⚠️ CORS set to allow ALL origins - not recommended for production!")

logger.info("✅ CORS allowed origins: %s", allowed_origins)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ------------------------ Security Headers ------------------------
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses"""
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
    app.include_router(rag_widget.router, prefix="/api")
    logger.info("✅ RAG widget router included at /api")
else:
    logger.warning("⚠️ rag_widget module has no 'router' attribute")

# OpenAI-compatible API for OpenWebUI
app.include_router(openai_compatible.router)
logger.info("✅ OpenAI-compatible router included for OpenWebUI integration")

# ------------------------ Request Models ------------------------
class UserQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="User's search query")
    max_results: int = Field(default=5, ge=1, le=50, description="Maximum number of results")
    include_images: bool = Field(default=True, description="Include images in response")


# ========================================================================================
# PRODUCTION ENDPOINT: Chat Query with OpenWebUI Formatting
# ========================================================================================

@app.post("/api/chat/query")
async def user_chat_query(request: UserQueryRequest, background_tasks: BackgroundTasks):
    """
    End-user query endpoint with production-grade OpenWebUI formatting.
    
    PRODUCTION FEATURES:
    - ✅ Step-by-step instructions with embedded images
    - ✅ Markdown optimization for OpenWebUI display
    - ✅ Agent response formatting with tables
    - ✅ Proper error handling with formatted messages
    - ✅ Maintains backward compatibility
    
    Flow:
    1. Call widget_query for initial processing
    2. Check response quality (agent responses always accepted)
    3. If weak, perform enhanced retrieval with postgres
    4. Generate step-by-step instructions with images
    5. Format EVERYTHING for OpenWebUI display with markdown
    
    Returns:
        JSON with formatted answer (markdown), steps, images, and metadata
    """
    try:
        query = request.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        logger.info(
            f"📥 User query: '{query}' "
            f"(max_results={request.max_results}, include_images={request.include_images})"
        )

        # =====================================================================
        # STEP 1: Call widget for initial processing
        # =====================================================================
        WidgetReqModel = getattr(rag_widget, "WidgetQueryRequest")
        if WidgetReqModel is None:
            logger.error("❌ rag_widget.WidgetQueryRequest model missing")
            raise HTTPException(status_code=500, detail="Widget model not available")
        
        widget_max = min(max(int(request.max_results or 1), 1), 100)

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

            import asyncio
            try:
                widget_resp = await asyncio.wait_for(
                    widget_query_fn(widget_req, background_tasks),
                    timeout=20.0
                )
            except asyncio.TimeoutError:
                logger.warning("⚠️ widget_query timed out after 20s")
                widget_resp = None
        except Exception as e:
            logger.exception("❌ Error calling rag_widget.widget_query: %s", e)
            widget_resp = None

        # =====================================================================
        # STEP 2: Check response quality and format for OpenWebUI
        # =====================================================================
        if widget_resp and not _is_weak_widget_response(widget_resp):
            logger.info("✅ Using widget response (quality check passed)")
            
            # Normalize images
            images = _normalize_images_list(
                widget_resp.get("images", []),
                cap=24
            ) if request.include_images else []
            
            # Format steps with images
            raw_steps = widget_resp.get("steps") or []
            steps = _format_steps_for_display(raw_steps)
            
            # Get raw answer
            raw_answer = widget_resp.get("answer", "")
            summary = widget_resp.get("summary")
            confidence = widget_resp.get("confidence", 0.0)
            routed_to = widget_resp.get("routed_to", "rag_system")
            
            # ============= PRODUCTION FIX: Format for OpenWebUI =============
            if routed_to == "agent_manager":
                # Agent response with tables
                logger.info("🎯 Formatting agent response for OpenWebUI")
                formatted_answer = format_agent_response_for_openwebui(
                    response_text=raw_answer,
                    execution_result=widget_resp.get("execution_result"),
                    session_id=widget_resp.get("session_id"),
                    metadata=widget_resp.get("metadata", {})
                )
            else:
                # Regular RAG response with steps and images
                logger.info(
                    f"📝 Formatting RAG response for OpenWebUI: "
                    f"{len(steps)} steps, {len(images)} images"
                )
                formatted_answer = format_for_openwebui(
                    answer=raw_answer,
                    steps=steps,
                    images=images,
                    query=query,
                    confidence=confidence,
                    summary=summary,
                    metadata={
                        "results_found": widget_resp.get("results_found", 0),
                        "results_used": widget_resp.get("results_used", 0),
                        "routed_to": routed_to,
                        "has_sources": widget_resp.get("has_sources", False)
                    }
                )
            # ================================================================
            
            return {
                "query": widget_resp.get("query", query),
                "answer": formatted_answer,  
                "steps": steps,  
                "stepsTitle": "Step-by-Step Instructions",
                "images": images,  
                "summary": summary,
                "summaryTitle": "Quick Summary",
                "timestamp": widget_resp.get("timestamp") or datetime.now().isoformat(),
                "confidence": confidence,
                "results_found": widget_resp.get("results_found", 0),
                "results_used": widget_resp.get("results_used", 0),
                "has_sources": widget_resp.get("has_sources", False),
                "source": routed_to
            }

        # =====================================================================
        # STEP 3: Enhanced Retrieval Fallback
        # =====================================================================
        logger.info("⚡ Widget response weak - performing enhanced retrieval")
        
        # Broaden postgres search
        postgres_n = min(200, max(request.max_results * 3, 120))
        postgres_results = []
        
        try:
            postgres_results = await postgres_service.search_documents(
                query, 
                n_results=postgres_n
            )
        except Exception as e:
            logger.exception("❌ postgres search failed: %s", e)
            postgres_results = []
        
        # No results fallback
        if not postgres_results:
            logger.warning("⚠️ No postgres results - returning formatted error")
            
            # ============= PRODUCTION FIX: Format error for OpenWebUI =============
            formatted_error = format_error_for_openwebui(
                error_message=(
                    "I couldn't find relevant information in my knowledge base. "
                    "The information you're looking for may not have been added yet."
                ),
                suggestions=[
                    "Try rephrasing your question with different keywords",
                    "Use more specific terms related to your topic",
                    "Check if the documentation has been uploaded to the system",
                    "Contact support if you believe this information should be available"
                ],
                error_type="not_found"
            )
            # ======================================================================
            
            return {
                "query": query,
                "answer": formatted_error,  
                "steps": [],
                "stepsTitle": "Step-by-Step Instructions",
                "images": [],
                "summary": "No relevant information found.",
                "summaryTitle": "Quick Summary",
                "confidence": 0.0,
                "results_found": 0,
                "results_used": 0,
                "has_sources": False,
                "source": "no_results",
                "timestamp": datetime.now().isoformat()
            }

        # Build context from results
        context_texts = []
        for r in postgres_results[:min(40, len(postgres_results))]:
            content = r.get("content", "") or ""
            if content and len(content.strip()) > 50:
                context_texts.append(content[:5000])
        
        if not context_texts:
            context_texts = [r.get("content", "")[:2000] for r in postgres_results[:3]]

        try:
            enhanced = await ai_service.generate_enhanced_response(
                query,
                context_texts,
                query_type=None
            )
        except Exception as e:
            logger.exception("❌ Enhanced response generation failed: %s", e)
            enhanced = {}

        # Extract answer and confidence
        answer_text = ""
        confidence_score = 0.0
        if isinstance(enhanced, dict):
            answer_text = enhanced.get("text", "") or ""
            confidence_score = float(enhanced.get("quality_score", 0.0) or 0.0)
        elif isinstance(enhanced, str):
            answer_text = enhanced
            confidence_score = 0.6

        # Collect images from postgres metadata
        candidate_images = []
        seen_urls = set()
        
        for r in postgres_results[:20]:  # Check first 20 results
            meta = r.get("metadata") or {}
            imgs = meta.get("images") if isinstance(meta.get("images"), list) else []
            page_url = meta.get("url") or ""
            
            for img in imgs:
                if isinstance(img, dict):
                    n = _normalize_image_obj(img)
                    if not n:
                        continue
                    
                    # Filter out logos/icons
                    u_low = n["url"].lower()
                    if any(noise in u_low for noise in ["logo", "icon", "favicon", "sprite", "banner"]):
                        continue
                    
                    n["source_url"] = page_url
                    
                    if n["url"] not in seen_urls:
                        seen_urls.add(n["url"])
                        candidate_images.append(n)
                
                elif isinstance(img, str) and img.startswith("http"):
                    n = _normalize_image_obj(img)
                    if n and n["url"] not in seen_urls:
                        seen_urls.add(n["url"])
                        n["source_url"] = page_url
                        candidate_images.append(n)
                
                if len(candidate_images) >= 24:
                    break
            
            if len(candidate_images) >= 24:
                break

        selected_images = candidate_images[:24]

        # Generate step-by-step response
        steps_with_images: List[Dict[str, Any]] = []
        try:
            steps_data = await ai_service.generate_stepwise_response(
                query,
                context_texts[:3]
            )
        except Exception as e:
            logger.warning(f"⚠️ Stepwise response generation failed: {e}")
            steps_data = []

        # Fallback: Create steps from answer sentences
        if not steps_data and answer_text:
            sentences = [s.strip() for s in answer_text.split(".") if s.strip()]
            steps_data = [
                {"text": (s + ".") if not s.endswith(".") else s, "type": "info"}
                for s in sentences[:6]
            ]

        
        selected_image_urls = [img.get("url") for img in selected_images if img.get("url")]

        # Assign images to steps
        for i, step in enumerate(steps_data):
            text = step.get("text") if isinstance(step, dict) else str(step)
            step_entry: Dict[str, Any] = {
                "index": i + 1,
                "text": text,
                "type": step.get("type", "action") if isinstance(step, dict) else "info"
            }
            
            assigned_img_url = None

            # Check if step has explicit image
            if isinstance(step, dict):
                raw_img = step.get("image") or step.get("image_url")
                if raw_img:
                    assigned_img_url = _normalize_image_for_display(raw_img)

            # Otherwise, use Nth image from selected images
            if not assigned_img_url and i < len(selected_image_urls):
                assigned_img_url = selected_image_urls[i]

            if assigned_img_url:
                step_entry["image"] = assigned_img_url

            steps_with_images.append(step_entry)

        # Generate summary
        try:
            summary_text = await ai_service.generate_summary(
                answer_text or "\n\n".join(context_texts[:3]),
                max_sentences=3,
                max_chars=600
            )
        except Exception:
            summary_text = (
                (answer_text[:600] + "...") 
                if answer_text and len(answer_text) > 600 
                else (answer_text or "")
            )

        # Format final response
        steps = _format_steps_for_display(steps_with_images)
        images = _normalize_images_list(selected_images, cap=24) if request.include_images else []

        # ============= PRODUCTION FIX: Format for OpenWebUI =============
        formatted_answer = format_for_openwebui(
            answer=answer_text or (
                "I found information in the knowledge base but couldn't "
                "generate a comprehensive answer."
            ),
            steps=steps,
            images=images,
            query=query,
            confidence=confidence_score,
            summary=summary_text or "No summary available.",
            metadata={
                "results_found": len(postgres_results),
                "results_used": min(len(postgres_results), request.max_results),
                "routed_to": "enhanced_retrieval",
                "has_sources": True
            }
        )
        # ================================================================

        logger.info(
            f"📤 Returning enhanced response: answer={len(answer_text)} chars, "
            f"steps={len(steps)}, images={len(images)}, confidence={confidence_score:.2f}"
        )

        return {
            "query": query,
            "answer": formatted_answer,  
            "steps": steps,  # Keep for API consumers
            "stepsTitle": "Step-by-Step Instructions",
            "images": images,  # Keep for API consumers
            "summary": summary_text or "No summary available.",
            "summaryTitle": "Quick Summary",
            "confidence": round(confidence_score, 3),
            "results_found": len(postgres_results),
            "results_used": min(len(postgres_results), request.max_results),
            "has_sources": True,
            "source": "enhanced_retrieval",
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"❌ User query error: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred processing your request. Please try again."
        )


# ------------------------ Metrics Endpoint ------------------------
@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus metrics endpoint"""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    from fastapi.responses import Response
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ------------------------ Health Check ------------------------
@app.get("/health")
async def health():
    """Simple health check endpoint"""
    return {"status": "ok", "service": "user"}


@app.get("/health/detailed")
async def health_check_detailed():
    """Detailed health check with service status"""
    try:
        doc_count = 0
        postgres_ok = False
        
        try:
            stats = await postgres_service.get_collection_stats()
            doc_count = stats.get("document_count", 0) if isinstance(stats, dict) else 0
            postgres_ok = True
        except Exception:
            postgres_ok = False

        return {
            "status": "healthy" if postgres_ok else "degraded",
            "service": "user_chat",
            "postgres": {
                "available": postgres_ok,
                "documents": doc_count
            },
            "version": app.version,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.exception("Health check failed")
        return JSONResponse(
            {"status": "unhealthy", "error": str(e)},
            status_code=503
        )


# ------------------------ Widget API: token + config ------------------------
class TokenResponse(BaseModel):
    token: str
    expires_at: str


@app.post("/api/widget/token", response_model=TokenResponse)
async def issue_widget_token(x_api_key: Optional[str] = Header(None)):
    """Issue JWT token for widget authentication"""
    if WIDGET_API_KEY and x_api_key != WIDGET_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    payload = {
        "sub": "widget-client",
        "iat": datetime.utcnow(),
        "exp": datetime.utcnow() + timedelta(minutes=30),
        "scope": "widget:query"
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)
    return {
        "token": token,
        "expires_at": (datetime.utcnow() + timedelta(minutes=30)).isoformat()
    }


@app.get("/api/widget/config")
async def widget_config():
    """Get widget configuration"""
    api_base = os.getenv("USER_API_BASE", "")
    return {
        "allowed_origins": allowed_origins,
        "widget_url": WIDGET_URL or (
            "/widget/index.html" if os.path.isdir(WIDGET_STATIC_DIR) else ""
        ),
        "api_base": api_base or ""
    }


# ------------------------ Frontend SPA ------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
user_frontend_path = BASE_DIR / "dist" / "user-frontend"

if user_frontend_path.exists() and user_frontend_path.is_dir():
    app.mount(
        "/", 
        StaticFiles(directory=str(user_frontend_path), html=True), 
        name="user_frontend"
    )
    logger.info(f"✅ User frontend mounted at: {user_frontend_path}")

    @app.get("/{full_path:path}")
    async def spa_fallback(full_path: str):
        """SPA fallback for client-side routing"""
        # Don't serve index.html for API routes
        if full_path.startswith("api/") or full_path.startswith("health"):
            raise HTTPException(status_code=404, detail="Route not found")
        
        index_file = user_frontend_path / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return JSONResponse({"detail": "index.html not found"}, status_code=500)
else:
    logger.warning(f"⚠️ User frontend not found at {user_frontend_path}")


# ------------------------ Widget & CDN Static Files ------------------------
widget_static_path = BASE_DIR / WIDGET_STATIC_DIR
if widget_static_path.exists() and widget_static_path.is_dir():
    app.mount(
        "/widget", 
        StaticFiles(directory=str(widget_static_path), html=True), 
        name="widget"
    )
    logger.info(f"✅ Widget static mounted at /widget")
else:
    logger.info(f"⚠️ Widget static not mounted (folder not found)")


# ------------------------ Root ------------------------
@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "RAG Chat - User Interface",
        "version": app.version,
        "endpoints": {
            "chat": "/api/chat/query",
            "health": "/health",
            "openwebui": "/api/v1/chat/completions"
        },
        "documentation": "/docs"
    }


# ------------------------ Run ------------------------
if __name__ == "__main__":
    host = os.getenv("USER_APP_HOST", "0.0.0.0")
    port = int(os.getenv("USER_APP_PORT", "8001"))
    reload = os.getenv("DEV_RELOAD", "false").lower() in ("1", "true")
    workers = int(os.getenv("USER_WORKERS", "2"))
    
    logger.info(f"🚀 Starting server on {host}:{port}")
    logger.info(f"   Workers: {workers}, Reload: {reload}")
    
    uvicorn.run(
        "user_main:app",
        host=host,
        port=port,
        reload=reload,
        workers=1 if reload else workers,  # Use 1 worker in reload mode
        log_level=os.getenv("LOG_LEVEL", "info"),
    )