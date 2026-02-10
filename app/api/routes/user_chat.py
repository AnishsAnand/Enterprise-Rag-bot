import os
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, Header, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import jwt

from app.api.routes import rag_widget
from app.services.postgres_service import postgres_service
from app.services.ai_service import ai_service
from app.services.webui_formatter import (
    format_for_webui,
    format_agent_response_for_webui,
    format_error_for_webui)

logger = logging.getLogger("user_app")
router = APIRouter(tags=["user-chat"])
# ------------------------ JWT / Widget Config ------------------------
JWT_SECRET = os.getenv("WIDGET_JWT_SECRET", "dev-secret-change-me")
JWT_ALG = "HS256"
WIDGET_API_KEY = os.getenv("WIDGET_API_KEY", "dev-widget-key")
WIDGET_STATIC_DIR = os.getenv("WIDGET_STATIC_DIR", "widget_static")
WIDGET_URL = os.getenv("WIDGET_URL", "")

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
    if s.startswith("http://") or s.startswith("https://"):
        return s
    return None

def _normalize_image_for_display(img: Any) -> Optional[str]:
    """
    Extract image URL from any format and return just the URL string.
    OpenWebUI needs simple URL strings in the response.
    """
    if not img:
        return None

    if isinstance(img, str):
        url = img.strip()
        if url.startswith("http://") or url.startswith("https://"):
            return url
        return None

    if isinstance(img, dict):
        for key in ["url", "src", "image", "image_url", "href"]:
            if key in img:
                url_val = img[key]
                if isinstance(url_val, str):
                    url = url_val.strip()
                    if url.startswith("http://") or url.startswith("https://"):
                        return url

        if "data" in img and isinstance(img["data"], dict):
            if "url" in img["data"]:
                url_val = img["data"]["url"]
                if isinstance(url_val, str):
                    url = url_val.strip()
                    if url.startswith("http://") or url.startswith("https://"):
                        return url

    return None

def _format_steps_for_display(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Format steps for OpenWebUI display with proper image handling."""
    formatted_steps = []

    for idx, step in enumerate(steps, 1):
        if not isinstance(step, dict):
            formatted_steps.append(
                {
                    "step_number": idx,
                    "text": str(step),
                    "image": None,
                    "type": "info",
                }
            )
            continue

        text = step.get("text") or step.get("content") or step.get("description") or ""
        if not text or not text.strip():
            continue
        image_url = None
        raw_image = step.get("image") or step.get("image_url") or step.get("img")
        if raw_image:
            image_url = _normalize_image_for_display(raw_image)
        formatted_steps.append(
            {
                "step_number": step.get("step_number") or step.get("index") or idx,
                "text": text.strip(),
                "image": image_url,
                "type": step.get("type", "action"),
            }
        )

    return formatted_steps

def _normalize_image_obj(img: Any) -> Optional[Dict[str, Any]]:
    """Normalize image entries to standard format: {url, alt, caption, source_url}"""
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
            "source_url": str(
                img.get("source_url") or img.get("page_url") or img.get("page") or ""
            ),
        }

    return None

def _normalize_images_list(images: Any, cap: int = 24) -> List[Dict[str, Any]]:
    """Normalize a list of images to standard format with deduplication."""
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

def _is_weak_widget_response(resp: Optional[dict]) -> bool:
    """Determine if widget response is too weak to use."""
    if not resp or not isinstance(resp, dict):
        return True
    answer = (resp.get("answer") or "").strip()
    results_found = int(resp.get("results_found", 0))
    confidence = float(resp.get("confidence", 0.0))
    routed_to = resp.get("routed_to", "")
    if routed_to == "agent_manager":
        logger.info("✅ Agent response detected - considered STRONG")
        return False
    if len(answer) < 30:
        logger.info("❌ Weak response: answer too short (%d chars)", len(answer))
        return True
    has_results = results_found >= 1
    has_decent_confidence = confidence >= 0.3
    if has_results and has_decent_confidence:
        logger.info("✅ Strong response: %d results, %.2f confidence", results_found, confidence)
        return False
    if confidence >= 0.7 and len(answer) >= 100:
        logger.info("✅ Strong AI response: %.2f confidence, %d chars", confidence, len(answer))
        return False
    logger.info(
        "❌ Weak response: results=%d, confidence=%.2f, answer_length=%d",
        results_found,
        confidence,
        len(answer),
    )
    return True

class UserQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="User's search query")
    max_results: int = Field(default=5, ge=1, le=50, description="Maximum number of results")
    include_images: bool = Field(default=True, description="Include images in response")

@router.post("/api/chat/query")
async def user_chat_query(request: UserQueryRequest, background_tasks: BackgroundTasks):
    """
    End-user query endpoint with OpenWebUI formatting.
    """
    try:
        query = request.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        logger.info(
            "📥 User query: '%s' (max_results=%s, include_images=%s)",
            query,
            request.max_results,
            request.include_images)
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
            search_depth="balanced")
        widget_resp = None
        try:
            widget_query_fn = getattr(rag_widget, "widget_query", None)
            if widget_query_fn is None:
                raise RuntimeError("rag_widget.widget_query not available")
            import asyncio
            try:
                widget_resp = await asyncio.wait_for(
                    widget_query_fn(widget_req, background_tasks),
                    timeout=20.0)
            except asyncio.TimeoutError:
                logger.warning("⚠️ widget_query timed out after 20s")
                widget_resp = None
        except Exception as e:
            logger.exception("❌ Error calling rag_widget.widget_query: %s", e)
            widget_resp = None
        if widget_resp and not _is_weak_widget_response(widget_resp):
            logger.info("✅ Using widget response (quality check passed)")
            images = (
                _normalize_images_list(widget_resp.get("images", []), cap=24)
                if request.include_images
                else [])
            raw_steps = widget_resp.get("steps") or []
            steps = _format_steps_for_display(raw_steps)
            raw_answer = widget_resp.get("answer", "")
            summary = widget_resp.get("summary")
            confidence = widget_resp.get("confidence", 0.0)
            routed_to = widget_resp.get("routed_to", "rag_system")
            if routed_to == "agent_manager":
                logger.info("🎯 Formatting agent response for OpenWebUI")
                formatted_answer = format_agent_response_for_webui(
                    response_text=raw_answer,
                    execution_result=widget_resp.get("execution_result"),
                    session_id=widget_resp.get("session_id"),
                    metadata=widget_resp.get("metadata", {}) )
            else:
                logger.info(
                    "📝 Formatting RAG response for OpenWebUI: %d steps, %d images",
                    len(steps),
                    len(images))
                formatted_answer = format_for_webui(
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
                        "has_sources": widget_resp.get("has_sources", False)},)
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
                "source": routed_to}

        logger.info("⚡ Widget response weak - performing enhanced retrieval")
        postgres_n = min(200, max(request.max_results * 3, 120))
        postgres_results = []
        try:
            postgres_results = await postgres_service.search_documents(
                query,
                n_results=postgres_n)
        except Exception as e:
            logger.exception("❌ postgres search failed: %s", e)
            postgres_results = []
        if not postgres_results:
            logger.warning("⚠️ No postgres results - returning formatted error")
            formatted_error = format_error_for_webui(
                error_message=(
                    "I couldn't find relevant information in my knowledge base. "
                    "The information you're looking for may not have been added yet."
                ),
                suggestions=[
                    "Try rephrasing your question with different keywords",
                    "Use more specific terms related to your topic",
                    "Check if the documentation has been uploaded to the system",
                    "Contact support if you believe this information should be available",
                ],
                error_type="not_found")
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
                "timestamp": datetime.now().isoformat()}

        context_texts = []
        for r in postgres_results[: min(40, len(postgres_results))]:
            content = r.get("content", "") or ""
            if content and len(content.strip()) > 50:
                context_texts.append(content[:5000])

        if not context_texts:
            context_texts = [r.get("content", "")[:2000] for r in postgres_results[:3]]
        try:
            enhanced = await ai_service.generate_enhanced_response(query,context_texts,query_type=None)
        except Exception as e:
            logger.exception("❌ Enhanced response generation failed: %s", e)
            enhanced = {}
        answer_text = ""
        confidence_score = 0.0
        if isinstance(enhanced, dict):
            answer_text = enhanced.get("text", "") or ""
            confidence_score = float(enhanced.get("quality_score", 0.0) or 0.0)
        elif isinstance(enhanced, str):
            answer_text = enhanced
            confidence_score = 0.6
        candidate_images = []
        seen_urls = set()
        for r in postgres_results[:20]:
            meta = r.get("metadata") or {}
            imgs = meta.get("images") if isinstance(meta.get("images"), list) else []
            page_url = meta.get("url") or ""
            for img in imgs:
                if isinstance(img, dict):
                    n = _normalize_image_obj(img)
                    if not n:
                        continue
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
        steps_with_images: List[Dict[str, Any]] = []
        try:
            steps_data = await ai_service.generate_stepwise_response(query,context_texts[:3])
        except Exception as e:
            logger.warning("⚠️ Stepwise response generation failed: %s", e)
            steps_data = []
        if not steps_data and answer_text:
            sentences = [s.strip() for s in answer_text.split(".") if s.strip()]
            steps_data = [
                {"text": (s + ".") if not s.endswith(".") else s, "type": "info"}
                for s in sentences[:6]]
        selected_image_urls = [img.get("url") for img in selected_images if img.get("url")]
        for i, step in enumerate(steps_data):
            text = step.get("text") if isinstance(step, dict) else str(step)
            step_entry: Dict[str, Any] = {
                "index": i + 1,
                "text": text,
                "type": step.get("type", "action") if isinstance(step, dict) else "info"}
            assigned_img_url = None
            if isinstance(step, dict):
                raw_img = step.get("image") or step.get("image_url")
                if raw_img:
                    assigned_img_url = _normalize_image_for_display(raw_img)
            if not assigned_img_url and i < len(selected_image_urls):
                assigned_img_url = selected_image_urls[i]
            if assigned_img_url:
                step_entry["image"] = assigned_img_url
            steps_with_images.append(step_entry)
        try:
            summary_text = await ai_service.generate_summary(answer_text or "\n\n".join(context_texts[:3]),max_sentences=3,max_chars=600)
        except Exception:
            summary_text = (
                (answer_text[:600] + "...")
                if answer_text and len(answer_text) > 600
                else (answer_text or ""))
        steps = _format_steps_for_display(steps_with_images)
        images = _normalize_images_list(selected_images, cap=24) if request.include_images else []
        formatted_answer = format_for_webui(
            answer=answer_text
            or (
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
                "has_sources": True},)
        logger.info(
            "📤 Returning enhanced response: answer=%d chars, steps=%d, images=%d, confidence=%.2f",
            len(answer_text),
            len(steps),
            len(images),
            confidence_score)
        return {
            "query": query,
            "answer": formatted_answer,
            "steps": steps,
            "stepsTitle": "Step-by-Step Instructions",
            "images": images,
            "summary": summary_text or "No summary available.",
            "summaryTitle": "Quick Summary",
            "confidence": round(confidence_score, 3),
            "results_found": len(postgres_results),
            "results_used": min(len(postgres_results), request.max_results),
            "has_sources": True,
            "source": "enhanced_retrieval",
            "timestamp": datetime.now().isoformat()}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("❌ User query error: %s", e)
        raise HTTPException(
            status_code=500,
            detail="An error occurred processing your request. Please try again.")

@router.get("/health/detailed")
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
            "postgres": {"available": postgres_ok, "documents": doc_count},
            "version": os.getenv("APP_VERSION", "2.0.0"),
            "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.exception("Health check failed")
        return JSONResponse({"status": "unhealthy", "error": str(e)}, status_code=503)

class TokenResponse(BaseModel):
    token: str
    expires_at: str

@router.post("/api/widget/token", response_model=TokenResponse)
async def issue_widget_token(x_api_key: Optional[str] = Header(None)):
    """Issue JWT token for widget authentication"""
    if WIDGET_API_KEY and x_api_key != WIDGET_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    payload = {
        "sub": "widget-client",
        "iat": datetime.utcnow(),
        "exp": datetime.utcnow() + timedelta(minutes=30),
        "scope": "widget:query"}
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)
    return {"token": token, "expires_at": (datetime.utcnow() + timedelta(minutes=30)).isoformat()}

@router.get("/api/widget/config")
async def widget_config(request: Request):
    """Get widget configuration"""
    allowed_origins = getattr(request.app.state, "allowed_origins", [])
    api_base = os.getenv("USER_API_BASE", "")
    return {
        "allowed_origins": allowed_origins,
        "widget_url": WIDGET_URL or ("/widget/index.html" if os.path.isdir(WIDGET_STATIC_DIR) else ""),
        "api_base": api_base or ""}

@router.websocket("/socket.io/")
async def socketio_stub(websocket: WebSocket):
    """Stub endpoint for Socket.IO connections to silence 403s."""
    await websocket.accept()
    logger.info("Socket.IO stub: connection accepted")
    try:
        while True:
            data = await websocket.receive_text()
            if data == "2":
                await websocket.send_text("3")
            elif data.startswith("40"):
                await websocket.send_text("40")
    except WebSocketDisconnect:
        logger.debug("Socket.IO stub: client disconnected")
    except Exception as e:
        logger.debug("Socket.IO stub: connection closed - %s", e)
