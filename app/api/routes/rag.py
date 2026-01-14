from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from datetime import datetime
import difflib
import inspect
import logging

from app.services.postgres_service import postgres_service
from app.services.ai_service import ai_service
import re
router = APIRouter()
logger = logging.getLogger(__name__)

# ---------- Request models ----------
class WidgetQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    max_results: int = Field(default=50, ge=1, le=100)
    search_depth: str = Field(default="balanced", pattern="^(quick|balanced|deep)$")

# ---------- Utilities ----------
def _enhanced_similarity(query: str, text: str) -> float:
    if not query or not text:
        return 0.0
    query_norm = " ".join(query.lower().split())
    text_norm = " ".join(text.lower().split())
    seq_ratio = difflib.SequenceMatcher(None, query_norm, text_norm).ratio()
    query_words, text_words = set(query_norm.split()), set(text_norm.split())
    overlap = len(query_words & text_words) / len(query_words | text_words) if query_words else 0.0
    substring_bonus = 0.2 if query_norm in text_norm else 0.0
    score = 0.4 * seq_ratio + 0.4 * overlap + 0.2 * substring_bonus
    return min(1.0, max(0.0, score))

def _extract_key_concepts(text: str) -> List[str]:
    if not text:
        return []
    patterns = [r"\b[A-Z]{2,}\b", r"\b\w+(?:_\w+)+\b", r"\b\w+(?:-\w+)+\b", r"\b\d+(?:\.\d+)?\w*\b"]
    concepts = []
    for p in patterns:
        concepts.extend(re.findall(p, text))
    words = re.findall(r"\b[a-zA-Z]{5,}\b", text)
    stopwords = {"about", "after", "again", "before", "being", "could", "while", "would"}
    meaningful = [w.lower() for w in words if w.lower() not in stopwords]
    combined = list(dict.fromkeys(concepts + meaningful))
    return combined[:15]

async def call_maybe_async(fn, *args, **kwargs):
    if not callable(fn):
        raise RuntimeError("Provided object is not callable")
    result = fn(*args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result

# ---------- Endpoints ----------
@router.post("/widget/query")
async def widget_query(request: WidgetQueryRequest):
    """User-facing query endpoint (chatbot responses with images only)"""
    try:
        query = request.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        logger.info(f"Processing query: '{query}'")

        # Fetch search results from Chroma (optional for user, fallback to LLM)
        search_params = {
            "quick": {"max_results": min(request.max_results, 30)},
            "balanced": {"max_results": request.max_results},
            "deep": {"max_results": min(request.max_results * 2, 100)},
        }
        search_config = search_params.get(request.search_depth, search_params["balanced"])
        
        try:
            search_results = await call_maybe_async(
                getattr(milvus_service, "search_documents", milvus_service),
                query,
                n_results=search_config["max_results"]
            )
        except Exception as e:
            logger.warning(f"Milvus search failed: {e}")
            search_results = []

        base_context = [r.get("content", "")[:1500] for r in search_results[:3]] if search_results else []

        try:
            enhanced_result = await call_maybe_async(ai_service.generate_enhanced_response, query, base_context, None)
            answer = (enhanced_result or {}).get("text", "") if isinstance(enhanced_result, dict) else (enhanced_result or "")
            expanded_context = (enhanced_result or {}).get("expanded_context", "") if isinstance(enhanced_result, dict) else ""
            confidence = (enhanced_result or {}).get("quality_score", 0.0) if isinstance(enhanced_result, dict) else 0.0
        except Exception:
            answer = await call_maybe_async(ai_service.generate_response, query, base_context[:3]) if base_context else "Sorry, I couldn't find an answer."
            expanded_context = "\n\n".join(base_context[:2]) if base_context else ""
            confidence = 0.6

        # Stepwise response
        try:
            steps_data = await call_maybe_async(ai_service.generate_stepwise_response, query, [expanded_context] if expanded_context else base_context[:3])
        except Exception:
            steps_data = [{"text": answer, "type": "info"}]

        # Image selection
        candidate_images = []
        all_concepts = set(_extract_key_concepts(query.lower()))
        for result in search_results:
            meta = result.get("metadata", {}) or {}
            images = meta.get("images", []) if isinstance(meta.get("images", []), list) else []
            for img in images:
                if img.get("url"):
                    candidate_images.append({"url": img.get("url"), "alt": img.get("alt", ""), "caption": img.get("caption", "")})

        selected_images = candidate_images[:12]

        # Assign images to steps
        steps_with_images = []
        for i, step in enumerate(steps_data):
            step_obj = {"index": i + 1, "text": step.get("text", ""), "type": step.get("type", "action")}
            if i < len(selected_images):
                step_obj["image"] = selected_images[i]
            steps_with_images.append(step_obj)

        # Summary
        try:
            summary = await call_maybe_async(ai_service.generate_summary, answer, max_sentences=4, max_chars=600)
        except Exception:
            summary = answer[:600] + "..." if len(answer) > 600 else answer

        return {
            "query": query,
            "answer": answer,
            "step_count": len(steps_with_images),
            "steps": steps_with_images,
            "images": selected_images,
            "confidence": round(confidence, 3),
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
