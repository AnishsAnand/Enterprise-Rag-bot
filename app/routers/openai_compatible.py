"""
Production-grade OpenAI-compatible API for Open WebUI integration.
UPDATED: Properly formats responses with step-by-step instructions and embedded images.

CRITICAL CHANGES:
1. Added OpenWebUI formatter integration for proper markdown display
2. Enhanced RAG search to retrieve step-by-step content with images
3. Added intelligent routing to widget endpoint for rich content
4. Proper streaming with formatted markdown
5. Maintains OpenAI API compatibility
"""

from fastapi import APIRouter, HTTPException, Header, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, AsyncGenerator
import time
import uuid
import json
import logging
from datetime import datetime
import hashlib
import asyncio
import httpx

from app.agents import get_agent_manager
from app.services.postgres_service import postgres_service
from app.services.ai_service import ai_service
from app.services.rag_search_service import rag_search_service
from app.core.database import SessionLocal
from app.models.database_models import RAGQuery

# ============================================================================
# PRODUCTION FIX: Import OpenWebUI Formatter
# ============================================================================
from app.services.openwebui_formatter import (
    format_for_openwebui,
    format_agent_response_for_openwebui,
    format_error_for_openwebui
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["OpenAI Compatible"])

# ============================================================================
# Pydantic Models (OpenAI-compatible schemas)
# ============================================================================

class Message(BaseModel):
    role: str = Field(..., description="Role: user/assistant/system")
    content: str = Field(..., description="Message content")
    name: Optional[str] = Field(None, description="Optional sender name")


class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="Model identifier")
    messages: List[Message] = Field(..., description="Conversation messages")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(4096, gt=0)
    stream: Optional[bool] = Field(False)
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0)
    frequency_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    user: Optional[str] = Field(None)
    stop: Optional[List[str]] = Field(None)


class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Optional[Usage] = None


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "Tata Communications"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


# ============================================================================
# Helper Functions
# ============================================================================

def estimate_tokens(text: str) -> int:
    """Estimate token count (1 token ≈ 4 characters)."""
    return max(1, min(8192, max(1, len(text) // 3)))


def create_completion_id() -> str:
    """Generate unique completion ID."""
    return f"chatcmpl-{uuid.uuid4().hex[:8]}"


def create_stable_session_id(user_id: str, conversation_context: str) -> str:
    """Create stable session ID for conversations."""
    context_hash = hashlib.md5(f"{user_id}:{conversation_context}".encode()).hexdigest()[:16]
    return f"openwebui_{context_hash}"


def create_stream_chunk(
    completion_id: str,
    model: str,
    content: str,
    is_first: bool = False,
    finish_reason: Optional[str] = None
) -> str:
    """Create OpenAI-compatible streaming response chunk"""
    choice = {
        "index": 0,
        "delta": {"content": content} if content else {"role": "assistant"},
        "finish_reason": finish_reason if finish_reason else None
    }
    if is_first:
        delta = {"role": "assistant"}
    elif content:
        delta = {"content": content}
    else:
        delta = {}
    
    response = {
    "id": completion_id,
    "object": "chat.completion.chunk",
    "created": int(datetime.utcnow().timestamp()),
    "model": model,
    "choices": [choice]
}
    
    return f"data: {json.dumps(response)}\n\n"


# ============================================================================
# PRODUCTION FIX: Enhanced Widget Integration
# ============================================================================

async def get_rich_content_from_widget(
    query: str,
    user_id: Optional[str],
    session_id: str
) -> Optional[Dict[str, Any]]:
    """
    Call the widget endpoint to get rich content with steps and images.
    This ensures we get properly formatted, step-by-step responses.
    
    Returns:
        Dict with answer, steps, images, and metadata, or None if call fails
    """
    try:
        # Construct widget request
        widget_url = "http://127.0.0.1:8001/api/widget/query"  # Internal call
        
        widget_payload = {
            "query": query,
            "max_results": 10,
            "include_sources": True,
            "enable_advanced_search": True,
            "search_depth": "balanced",
            "auto_execute": True,
            "store_interaction": False,  # Don't store from API calls
            "session_id": session_id,
            "user_id": user_id or "openwebui_user"
        }
        
        logger.info(f"[OpenWebUI] Calling widget endpoint for rich content: {query[:50]}")
        
        # Make async HTTP call to widget
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(widget_url, json=widget_payload)
            
            if response.status_code == 200:
                widget_response = response.json()
                logger.info(
                    f"[OpenWebUI] Widget returned: "
                    f"steps={len(widget_response.get('steps', []))}, "
                    f"images={len(widget_response.get('images', []))}, "
                    f"confidence={widget_response.get('confidence', 0)}"
                )
                return widget_response
            else:
                logger.warning(
                    f"[OpenWebUI] Widget call failed: {response.status_code}"
                )
                return None
                
    except Exception as e:
        logger.warning(f"[OpenWebUI] Widget call exception: {e}")
        return None


# ============================================================================
# PRODUCTION FIX: Enhanced Response Builder
# ============================================================================

async def build_rich_response(
    query: str,
    user_id: Optional[str],
    session_id: str,
    temperature: float = 0.7
) -> Dict[str, Any]:
    """
    Build a rich response with steps, images, and proper formatting.
    
    Flow:
    1. Try widget endpoint first (gets steps + images)
    2. If widget fails, fall back to direct RAG
    3. Format everything for OpenWebUI display
    
    Returns:
        Dict with formatted_answer, metadata, and confidence
    """
    
    # STEP 1: Try to get rich content from widget
    widget_response = await get_rich_content_from_widget(query, user_id, session_id)
    
    if widget_response:
        # Widget succeeded - extract rich content
        answer = widget_response.get("answer", "")
        steps = widget_response.get("steps", [])
        images = widget_response.get("images", [])
        summary = widget_response.get("summary")
        confidence = widget_response.get("confidence", 0.0)
        
        # Check if this is an agent response (cluster listings, etc.)
        routed_to = widget_response.get("routed_to", "")
        if routed_to == "agent_manager":
            logger.info(f"[OpenWebUI] Agent response detected, using specialized formatting")
            formatted_answer = format_agent_response_for_openwebui(
                response_text=answer,
                execution_result=widget_response.get("execution_result"),
                session_id=session_id,
                metadata=widget_response.get("metadata", {})
            )
        else:
            # Regular RAG response - format with steps and images
            logger.info(
                f"[OpenWebUI] Formatting RAG response: "
                f"{len(steps)} steps, {len(images)} images"
            )
            formatted_answer = format_for_openwebui(
                answer=answer,
                steps=steps,
                images=images,
                query=query,
                confidence=confidence,
                summary=summary,
                show_metadata=False,
                metadata={
                    "results_found": widget_response.get("results_found", 0),
                    "results_used": widget_response.get("results_used", 0),
                    "routed_to": routed_to or "rag_system",
                    "has_sources": widget_response.get("has_sources", False)
                }
            
            )
        
        return {
            "formatted_answer": formatted_answer,
            "confidence": confidence,
            "metadata": widget_response.get("metadata", {}),
            "has_rich_content": bool(steps or images),
            "steps_count": len(steps),
            "images_count": len(images)
        }
    
    # STEP 2: Widget failed - fall back to direct RAG search
    logger.info(f"[OpenWebUI] Widget unavailable, using direct RAG search")
    
    rag_results = await rag_search_service.search(
        query=query,
        user_id=int(user_id) if user_id and user_id.isdigit() else None,
        top_k=10
    )
    
    has_results = rag_results.get("total_results", 0) > 0
    
    if not has_results:
        # No results found - return formatted error
        logger.warning(f"[OpenWebUI] No RAG results found for: {query}")
        
        formatted_answer = format_error_for_openwebui(
            error_message=(
                "I couldn't find relevant information in the knowledge base for your query."
            ),
            suggestions=[
                "Try rephrasing your question with different keywords",
                "Check if the information has been added to the knowledge base",
                "Contact support if you believe this information should be available"
            ],
            error_type="not_found"
        )
        
        return {
            "formatted_answer": formatted_answer,
            "confidence": 0.0,
            "metadata": {"results_found": 0},
            "has_rich_content": False,
            "steps_count": 0,
            "images_count": 0
        }
    
    # STEP 3: Generate enhanced response with available context
    chunks = rag_results.get("chunks", [])
    context_texts = [chunk.get("text", "") for chunk in chunks[:5]]
    
    # Generate main answer
    try:
        enhanced_result = await ai_service.generate_enhanced_response(
            query=query,
            context=context_texts,
            query_type=None,
            temperature=temperature
        )
        
        answer = (
            enhanced_result.get("text", "") 
            if isinstance(enhanced_result, dict) 
            else str(enhanced_result)
        )
        confidence = (
            enhanced_result.get("quality_score", 0.0)
            if isinstance(enhanced_result, dict)
            else 0.7
        )
    except Exception as e:
        logger.error(f"[OpenWebUI] Enhanced response generation failed: {e}")
        answer = "\n\n".join(context_texts[:2])
        confidence = 0.5
    
    # Generate steps
    try:
        steps = await ai_service.generate_stepwise_response(
            query=query,
            context=context_texts[:3]
        )
    except Exception as e:
        logger.warning(f"[OpenWebUI] Step generation failed: {e}")
        # Create basic steps from answer
        if answer:
            sentences = [s.strip() for s in answer.split(".") if s.strip()]
            steps = [
                {"text": (s + "."), "type": "info", "step_number": i+1}
                for i, s in enumerate(sentences[:5])
            ]
        else:
            steps = []
    
    # Extract images from context metadata
    images = []
    for chunk in chunks[:10]:
        chunk_meta = chunk.get("metadata", {})
        chunk_images = chunk_meta.get("images", [])
        
        if isinstance(chunk_images, list):
            for img in chunk_images:
                if isinstance(img, dict) and img.get("url"):
                    images.append({
                        "url": img.get("url"),
                        "alt": img.get("alt", ""),
                        "caption": img.get("caption", ""),
                        "relevance_score": chunk.get("confidence_score", 0.5)
                    })
                    
                    if len(images) >= 12:
                        break
        
        if len(images) >= 12:
            break
    
    # Generate summary
    try:
        summary = await ai_service.generate_summary(
            answer if answer else "\n\n".join(context_texts[:2]),
            max_sentences=3,
            max_chars=600
        )
    except Exception:
        summary = (answer[:600] + "...") if answer and len(answer) > 600 else answer
    
    # Format everything for OpenWebUI
    formatted_answer = format_for_openwebui(
        answer=answer or "Unable to generate response from available context.",
        steps=steps,
        images=images,
        query=query,
        confidence=confidence,
        summary=summary,
        metadata={
            "results_found": len(chunks),
            "results_used": len(context_texts),
            "routed_to": "direct_rag",
            "has_sources": True
        }
    )
    
    return {
        "formatted_answer": formatted_answer,
        "confidence": confidence,
        "metadata": {
            "results_found": len(chunks),
            "results_used": len(context_texts)
        },
        "has_rich_content": bool(steps or images),
        "steps_count": len(steps),
        "images_count": len(images)
    }


# ============================================================================
# Core Chat Endpoint with Enhanced RAG Integration
# ============================================================================

@router.post("/chat/completions")
async def chat_completions(
    request_data: ChatCompletionRequest,
    request: Request,
    background_tasks: BackgroundTasks
):
    """
    OpenAI-compatible chat completions endpoint with rich content support.
    
    PRODUCTION FEATURES:
    - Proper step-by-step instructions with embedded images
    - Agent response formatting for cluster/endpoint operations
    - Markdown optimized for OpenWebUI display
    - Maintains OpenAI API compatibility
    - Proper error handling and fallbacks
    """
    start_time = datetime.utcnow()
    completion_id = create_completion_id()
    
    try:
        # =====================================================================
        # STEP 1: Extract and validate request
        # =====================================================================
        if not request_data.messages or len(request_data.messages) == 0:
            raise HTTPException(status_code=400, detail="No messages provided")
        
        user_message = request_data.messages[-1]
        if user_message.role != "user":
            raise HTTPException(
                status_code=400, 
                detail="Last message must be from user"
            )
        
        query = user_message.content.strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        user_id = request_data.user or "openwebui_user"
        
        # Create stable session ID for conversation continuity
        conversation_hash = hashlib.md5(
            json.dumps([m.dict() for m in request_data.messages[:-1]]).encode()
        ).hexdigest()[:16]
        session_id = create_stable_session_id(user_id, conversation_hash)
        
        logger.info(
            f"[OpenWebUI] Chat request - ID: {completion_id}, "
            f"Query: '{query[:100]}', Session: {session_id}"
        )
        
        # =====================================================================
        # STEP 2: Build rich response with steps and images
        # =====================================================================
        response_data = await build_rich_response(
            query=query,
            user_id=user_id,
            session_id=session_id,
            temperature=request_data.temperature or 0.7
        )
        
        formatted_answer = response_data["formatted_answer"]
        confidence = response_data["confidence"]
        metadata = response_data["metadata"]
        
        logger.info(
            f"[OpenWebUI] Response built - "
            f"Rich: {response_data['has_rich_content']}, "
            f"Steps: {response_data['steps_count']}, "
            f"Images: {response_data['images_count']}, "
            f"Confidence: {confidence:.2f}"
        )
        
        # =====================================================================
        # STEP 3: Validate response quality
        # =====================================================================
        if not formatted_answer or len(formatted_answer.strip()) < 20:
            logger.error(
                f"[OpenWebUI] Response too short or empty for query: {query}"
            )
            formatted_answer = format_error_for_openwebui(
                error_message=(
                    "Unable to generate a comprehensive response. "
                    "Please try rephrasing your question or contact support."
                ),
                suggestions=[
                    "Use more specific keywords",
                    "Break down complex questions into simpler parts",
                    "Check if related information is available in the knowledge base"
                ],
                error_type="service_unavailable"
            )
            confidence = 0.0
        
        # =====================================================================
        # STEP 4: Log query (background task)
        # =====================================================================
        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        background_tasks.add_task(
            _log_rag_query,
            user_id=user_id,
            query=query,
            response=formatted_answer,
            rag_results={
                "total_results": metadata.get("results_found", 0),
                "chunks": [],
                "metadata": metadata
            },
            execution_time=execution_time,
            confidence=confidence
        )
        
        # =====================================================================
        # STEP 5: Return response (streaming or non-streaming)
        # =====================================================================
        prompt_tokens = estimate_tokens(query)
        completion_tokens = estimate_tokens(formatted_answer)
        
        if request_data.stream:
            logger.info(f"[OpenWebUI] Streaming response ({len(formatted_answer)} chars)")
            return StreamingResponse(
                _stream_response(
                    completion_id, 
                    request_data.model, 
                    formatted_answer
                ),
                media_type="text/event-stream; charset=utf-8"
            )
        else:
            logger.info(f"[OpenWebUI] Non-streaming response ({len(formatted_answer)} chars)")
            return ChatCompletionResponse(
                id=completion_id,
                created=int(start_time.timestamp()),
                model=request_data.model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=Message(
                            role="assistant", 
                            content=formatted_answer
                        ),
                        finish_reason="stop"
                    )
                ],
                usage=Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens
                )
            )
    
    except HTTPException as e:
        logger.exception(f"[OpenWebUI] HTTP error: {e.detail}")
        raise
    except Exception as e:
        logger.exception(f"[OpenWebUI] Unhandled error: {e}")
        
        # Return formatted error response
        error_response = format_error_for_openwebui(
            error_message=f"An unexpected error occurred: {str(e)[:100]}",
            suggestions=[
                "Please try again in a moment",
                "Contact support if the issue persists",
                "Check system logs for more details"
            ],
            error_type="general"
        )
        
        return ChatCompletionResponse(
            id=completion_id,
            created=int(datetime.utcnow().timestamp()),
            model=request_data.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=Message(role="assistant", content=error_response),
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0
            )
        )


# ============================================================================
# Streaming Response Handler
# ============================================================================

async def _stream_response(
    completion_id: str,
    model: str,
    response_text: str
) -> AsyncGenerator[str, None]:
    """
    Stream response in OpenAI format with proper pacing.
    
    PRODUCTION NOTE: Streams character-by-character with small delays
    to ensure smooth display in OpenWebUI.
    """
    
    # Send initial chunk
    yield create_stream_chunk(completion_id, model, "",is_first=True)
    
    # Stream content
    chunk_size = 5  # Stream 5 characters at a time for smooth rendering
    
    for i in range(0, len(response_text), chunk_size):
        chunk_text = response_text[i:i+chunk_size]
        yield create_stream_chunk(completion_id, model, chunk_text)
        
        # Small delay for smooth streaming (adjust based on network)
        await asyncio.sleep(0.02)
    
    # Send final chunk with finish reason
    yield create_stream_chunk(
        completion_id,
        model,
        "",
        finish_reason="stop"
    )
    
    yield "data: [DONE]\n\n"


# ============================================================================
# Query Logging
# ============================================================================

async def _log_rag_query(
    user_id: Optional[str],
    query: str,
    response: str,
    rag_results: Dict,
    execution_time: float,
    confidence: float
):
    """
    Log RAG query for analytics (background task).
    
    PRODUCTION NOTE: Runs in background, won't block response.
    """
    try:
        db = SessionLocal()
        
        rag_query = RAGQuery(
            user_id=int(user_id) if user_id and user_id.isdigit() else None,
            query_text=query[:500],  # Truncate long queries
            response_text=response[:1000],  # Truncate long responses
            retrieved_chunks=rag_results.get("total_results", 0),
            response_sources=[
                {
                    "id": chunk.get("id"),
                    "title": chunk.get("document_title", "Unknown"),
                    "confidence": chunk.get("confidence_score")
                }
                for chunk in rag_results.get("chunks", [])[:5]
            ],
            query_latency_ms=execution_time,
            relevance_score=confidence
        )
        
        db.add(rag_query)
        db.commit()
        db.close()
        
        logger.info(
            f"[OpenWebUI] Query logged - "
            f"Sources: {rag_results.get('total_results', 0)}, "
            f"Confidence: {confidence:.2f}, "
            f"Time: {execution_time:.0f}ms"
        )
    
    except Exception as e:
        logger.exception(f"[OpenWebUI] Failed to log query: {e}")


# ============================================================================
# Model List Endpoint
# ============================================================================

@router.get("/models")
async def list_models():
    """
    List available models (OpenAI-compatible).
    
    Returns two models:
    - enterprise-rag-bot: Standard RAG with rich formatting
    - enterprise-rag-bot-rag: Alias for compatibility
    """
    return ModelListResponse(
        object="list",
        data=[
            ModelInfo(
                id="enterprise-rag-bot",
                created=int(datetime.utcnow().timestamp()),
                owned_by="Tata Communications"
            ),
            ModelInfo(
                id="enterprise-rag-bot-rag",
                created=int(datetime.utcnow().timestamp()),
                owned_by="Tata Communications"
            )
        ]
    )


# ============================================================================
# Health Check
# ============================================================================

@router.get("/health")
async def health_check():
    """Health check endpoint for OpenAI-compatible API"""
    try:
        # Test database connection
        db_healthy = False
        try:
            stats = await postgres_service.get_collection_stats()
            db_healthy = isinstance(stats, dict) and stats.get("status") in ("active", "healthy")
        except Exception as e:
            logger.warning(f"Database health check failed: {e}")
        
        # Test AI service
        ai_healthy = False
        try:
            ai_health = await ai_service.get_service_health()
            ai_healthy = ai_health.get("overall_status") == "healthy"
        except Exception as e:
            logger.warning(f"AI service health check failed: {e}")
        
        overall_status = "healthy" if (db_healthy and ai_healthy) else "degraded"
        
        return {
            "status": overall_status,
            "service": "openai_compatible_api",
            "components": {
                "database": "healthy" if db_healthy else "degraded",
                "ai_service": "healthy" if ai_healthy else "degraded"
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.exception(f"Health check error: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }