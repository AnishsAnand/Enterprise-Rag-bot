"""
OpenAI-compatible API endpoints for Open WebUI integration.
COMPLETE FIX: Proper RAG context extraction with images.
"""

from fastapi import APIRouter, HTTPException, Header, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, AsyncGenerator
import time
import uuid
import json
import logging
from datetime import datetime
import hashlib
import re

# Import actual agent system components
from app.agents import get_agent_manager
from app.services.milvus_service import milvus_service
from app.services.ai_service import ai_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["OpenAI Compatible"])


# ============================================================================
# Pydantic Models (OpenAI-compatible schemas)
# ============================================================================

class Message(BaseModel):
    """Chat message in OpenAI format."""
    role: str = Field(..., description="Role of the message sender (user/assistant/system)")
    content: str = Field(..., description="Content of the message")
    name: Optional[str] = Field(None, description="Optional name of the sender")


class ChatCompletionRequest(BaseModel):
    """OpenAI chat completion request format."""
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
    """Single completion choice."""
    index: int
    message: Message
    finish_reason: str


class Usage(BaseModel):
    """Token usage statistics."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """OpenAI chat completion response format."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Optional[Usage] = None


class ModelInfo(BaseModel):
    """Model information in OpenAI format."""
    id: str
    object: str = "model"
    created: int
    owned_by: str = "Tata Communications"


class ModelListResponse(BaseModel):
    """List of available models."""
    object: str = "list"
    data: List[ModelInfo]


# ============================================================================
# Helper Functions
# ============================================================================

def estimate_tokens(text: str) -> int:
    """Estimate token count (rough approximation: 1 token ≈ 4 characters)."""
    return len(text) // 4


def create_completion_id() -> str:
    """Generate a unique completion ID."""
    return f"chatcmpl-{uuid.uuid4().hex[:8]}"


def create_stable_session_id(user_id: str, conversation_context: str) -> str:
    """Create a stable session ID for Open WebUI conversations."""
    context_hash = hashlib.md5(f"{user_id}:{conversation_context}".encode()).hexdigest()[:16]
    return f"openwebui_{context_hash}"


def is_resource_operation(query: str) -> bool:
    """
    Detect if query is about resource operations (clusters, endpoints, etc.).
    Returns True for cluster/infrastructure operations, False for RAG queries.
    """
    query_lower = query.lower()
    query_words = query_lower.split()
    
    # Action keywords
    action_keywords = [
        "create", "make", "build", "deploy", "provision",
        "delete", "remove", "destroy", "terminate",
        "update", "modify", "change", "edit",
        "list", "show", "get", "view", "display", "fetch"
    ]
    
    # Resource keywords
    resource_keywords = [
        "cluster", "clusters", "k8s", "kubernetes",
        "firewall", "rule", "rules",
        "load balancer", "loadbalancer", "lb",
        "database", "db", "storage", "volume", "volumes",
        "endpoint", "endpoints", "datacenter", "datacenters", "dc"
    ]
    
    # Check for combinations
    has_action = any(keyword in query_lower for keyword in action_keywords) or "all" in query_words
    has_resource = any(keyword in query_lower for keyword in resource_keywords)
    
    return has_action and has_resource


async def get_agent_service():
    """Dependency to get the agent manager instance."""
    try:
        manager = get_agent_manager(
            vector_service=milvus_service,
            ai_service=ai_service
        )
        logger.debug("Agent manager initialized successfully")
        return manager
    except Exception as e:
        logger.error(f"Failed to initialize agent manager: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Agent service unavailable: {str(e)}"
        )


# ============================================================================
# API Endpoints
# ============================================================================

@router.get("/models", response_model=ModelListResponse)
async def list_models(authorization: Optional[str] = Header(None)) -> ModelListResponse:
    """List available models (OpenAI-compatible endpoint)."""
    logger.info("Models list requested")
    
    models = [
        ModelInfo(
            id="Vayu Maya",
            created=int(time.time()),
            owned_by="Tata Communications"
        )
    ]
    
    return ModelListResponse(data=models)


@router.post("/chat/completions", response_model=None)
async def chat_completions(
    request: ChatCompletionRequest,
    http_request: Request,
    authorization: Optional[str] = Header(None),
):
    """
    OpenAI-compatible chat completions endpoint.
    FIXED: Proper routing and context extraction.
    """
    try:
        # Log request
        logger.info(f"📨 Chat completion: model={request.model}, messages={len(request.messages)}, stream={request.stream}")
        
        # Validate messages
        if not request.messages:
            raise HTTPException(status_code=400, detail="No messages provided")
        
        # Extract user message and conversation history
        user_message = request.messages[-1].content
        conversation_history = [
            {"role": msg.role, "content": msg.content}
            for msg in request.messages[:-1]
        ]
        
        logger.info(f"🔍 Query: '{user_message[:100]}...'")
        
        # Determine user ID
        user_id = request.user or "openwebui_user"
        
        # Create stable session ID
        if len(conversation_history) > 0:
            first_msg = conversation_history[0].get("content", "")[:100]
            session_id = create_stable_session_id(user_id, first_msg)
        else:
            time_bucket = str(datetime.utcnow().hour) + str(datetime.utcnow().minute // 10)
            session_id = create_stable_session_id(user_id, time_bucket)
        
        # Route based on query type
        if is_resource_operation(user_message):
            logger.info(f"🎯 Routing to Agent Manager (resource operation)")
            return await _handle_agent_operation(
                user_message=user_message,
                session_id=session_id,
                user_id=user_id,
                model=request.model,
                stream=request.stream
            )
        else:
            logger.info(f"📚 Routing to RAG system (documentation query)")
            return await _handle_rag_query(
                user_message=user_message,
                conversation_history=conversation_history,
                model=request.model,
                stream=request.stream
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in chat completions: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


async def _handle_agent_operation(
    user_message: str,
    session_id: str,
    user_id: str,
    model: str,
    stream: bool
) -> Any:
    """Handle resource operations via agent manager."""
    try:
        agent_manager = await get_agent_service()
        
        logger.info(f"🤖 Processing agent request: session={session_id}")
        
        result = await agent_manager.process_request(
            user_input=user_message,
            session_id=session_id,
            user_id=user_id,
            user_roles=["user"]
        )
        
        response_text = result.get("response", "")
        
        if not response_text:
            response_text = "I encountered an error processing your request. Please try again."
        
        logger.info(f"✅ Agent response generated: {len(response_text)} chars")
        
        if stream:
            return await _stream_response(response_text, model)
        else:
            return _create_response(response_text, model)
            
    except Exception as e:
        logger.error(f"❌ Agent operation failed: {str(e)}")
        error_msg = "I encountered an error processing your request. Please try again."
        
        if stream:
            return await _stream_response(error_msg, model)
        else:
            return _create_response(error_msg, model)


async def _handle_rag_query(
    user_message: str,
    conversation_history: List[Dict],
    model: str,
    stream: bool
) -> Any:
    """
    Handle RAG documentation queries.
    CRITICAL FIX: Direct Milvus search with proper context extraction.
    """
    try:
        logger.info(f"🔍 Performing direct Milvus search...")
        
        # CRITICAL FIX: Direct Milvus search with AGGRESSIVE parameters
        search_results = await milvus_service.search_documents(
            query=user_message,
            n_results=150  # High limit for better coverage
        )
        
        if not search_results:
            logger.warning("⚠️ No Milvus results found")
            fallback_msg = (
                "I don't have specific documentation for that query in my knowledge base. "
                "Could you rephrase your question or provide more context?"
            )
            if stream:
                return await _stream_response(fallback_msg, model)
            else:
                return _create_response(fallback_msg, model)
        
        logger.info(f"✅ Found {len(search_results)} Milvus results")
        
        # CRITICAL FIX: Extract context and images with LOWER threshold
        context_texts = []
        all_images = []
        
        for result in search_results:
            relevance_score = result.get("relevance_score", 0.0)
            
            # Use LOWER threshold (0.10 instead of 0.30)
            if relevance_score >= 0.10:
                content = result.get("content", "")
                if content and len(content.strip()) > 50:
                    # Take MORE context per result (4000 chars vs 1500)
                    context_texts.append(content[:4000])
                
                # Extract images from metadata
                metadata = result.get("metadata", {})
                
                # Try images_json field first
                images_json = metadata.get("images_json", "")
                if images_json:
                    try:
                        images = json.loads(images_json)
                        if isinstance(images, list):
                            for img in images[:5]:
                                if isinstance(img, dict) and img.get("url"):
                                    all_images.append(img)
                    except:
                        pass
                
                # Also try direct images field
                images_field = metadata.get("images", [])
                if isinstance(images_field, list):
                    for img in images_field[:5]:
                        if isinstance(img, dict) and img.get("url"):
                            all_images.append(img)
        
        if not context_texts:
            logger.warning("⚠️ No valid context extracted from results")
            fallback_msg = (
                "I found some results but couldn't extract meaningful content. "
                "Please try rephrasing your question with more specific terms."
            )
            if stream:
                return await _stream_response(fallback_msg, model)
            else:
                return _create_response(fallback_msg, model)
        
        logger.info(f"📝 Extracted {len(context_texts)} context chunks, {len(all_images)} images")
        
        # CRITICAL FIX: Generate enhanced response with full context
        try:
            enhanced_result = await ai_service.generate_enhanced_response(
                query=user_message,
                context=context_texts[:30],  # Use top 30 contexts
                query_type=None  # Auto-detect
            )
            
            response_text = enhanced_result.get("text", "") if isinstance(enhanced_result, dict) else str(enhanced_result)
        except Exception as e:
            logger.error(f"❌ Enhanced response failed: {str(e)}")
            # Fallback: basic response generation
            try:
                response_text = await ai_service.generate_response(user_message, context_texts[:10])
            except:
                response_text = ""
        
        if not response_text or len(response_text.strip()) < 50:
            logger.warning("⚠️ Generated response too short, using context summary")
            # Create a basic response from context
            response_text = (
                f"Based on the documentation, here's what I found:\n\n"
                f"{context_texts[0][:800]}...\n\n"
            )
            if len(context_texts) > 1:
                response_text += f"\n\nAdditional context:\n{context_texts[1][:500]}..."
        
        # CRITICAL FIX: Add images as markdown for Open WebUI
        if all_images:
            # Deduplicate images by URL
            seen_urls = set()
            unique_images = []
            for img in all_images:
                url = img.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    unique_images.append(img)
            
            # Format top 8 images as markdown
            if unique_images:
                image_section = "\n\n---\n\n**📸 Related Images:**\n\n"
                for idx, img in enumerate(unique_images[:8], 1):
                    img_url = img.get("url", "")
                    img_alt = img.get("alt", f"Documentation Image {idx}")
                    img_caption = img.get("caption", "")
                    
                    if img_url:
                        image_section += f"![{img_alt}]({img_url})"
                        if img_caption:
                            image_section += f"\n*{img_caption}*"
                        image_section += "\n\n"
                
                response_text += image_section
                logger.info(f"✅ Added {len(unique_images[:8])} images to response")
        
        logger.info(f"✅ Final RAG response: {len(response_text)} chars")
        
        # Return formatted response
        if stream:
            return await _stream_response(response_text, model)
        else:
            return _create_response(response_text, model)
            
    except Exception as e:
        logger.error(f"❌ RAG query failed: {str(e)}", exc_info=True)
        error_msg = (
            "I encountered an error searching the documentation. "
            "Please try rephrasing your question or contact support if the issue persists."
        )
        
        if stream:
            return await _stream_response(error_msg, model)
        else:
            return _create_response(error_msg, model)


def _create_response(text: str, model: str) -> ChatCompletionResponse:
    """Create non-streaming OpenAI response."""
    return ChatCompletionResponse(
        id=create_completion_id(),
        created=int(time.time()),
        model=model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=Message(role="assistant", content=text),
                finish_reason="stop"
            )
        ],
        usage=Usage(
            prompt_tokens=100,
            completion_tokens=estimate_tokens(text),
            total_tokens=100 + estimate_tokens(text)
        )
    )


async def _stream_response(text: str, model: str) -> StreamingResponse:
    """Create streaming OpenAI response."""
    async def generate():
        completion_id = create_completion_id()
        created = int(time.time())
        
        # Split by paragraphs first, then by lines
        paragraphs = text.split('\n\n')
        
        for para in paragraphs:
            if para.strip():
                lines = para.split('\n')
                for line in lines:
                    if line.strip():
                        chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": line + "\n"},
                                "finish_reason": None
                            }]
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                
                # Paragraph break
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": "\n"},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
        
        # Final chunk
        final_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# ============================================================================
# Health Check
# ============================================================================

@router.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "service": "enterprise-rag-bot-openai-compatible",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.1-complete-fix"
    }