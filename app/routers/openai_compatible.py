"""
OpenAI-compatible API endpoints for Open WebUI integration.

This module provides OpenAI-compatible endpoints (/v1/chat/completions, /v1/models)
that allow Open WebUI to communicate with the Enterprise RAG Bot backend.
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
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(4096, gt=0, description="Maximum tokens to generate")
    stream: Optional[bool] = Field(False, description="Enable streaming responses")
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    frequency_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    user: Optional[str] = Field(None, description="User identifier")
    stop: Optional[List[str]] = Field(None, description="Stop sequences")
    
    class Config:
        schema_extra = {
            "example": {
                "model": "Vayu Maya",
                "messages": [
                    {"role": "user", "content": "Create a Kubernetes cluster"}
                ],
                "temperature": 0.7,
                "stream": False
            }
        }


class ChatCompletionChoice(BaseModel):
    """Single completion choice."""
    index: int
    message: Message
    finish_reason: str  # "stop", "length", "function_call"


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
    
    class Config:
        schema_extra = {
            "example": {
                "id": "chatcmpl-abc123",
                "object": "chat.completion",
                "created": 1677652288,
                "model": "Vayu Maya",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "I'll help you create a Kubernetes cluster..."
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 50,
                    "total_tokens": 60
                }
            }
        }


class ModelInfo(BaseModel):
    """Model information in OpenAI format."""
    id: str
    object: str = "model"
    created: int
    owned_by: str = "Tata Communications"
    permission: Optional[List[Dict]] = None
    root: Optional[str] = None
    parent: Optional[str] = None


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


async def get_agent_service():
    """
    Dependency to get the actual agent manager instance.
    
    This initializes and returns the multi-agent system that handles:
    - Intent classification
    - RAG-based question answering
    - CRUD operation orchestration
    - Multi-turn conversations
    """
    try:
        # Get the agent manager with vector and AI services
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
async def list_models(
    authorization: Optional[str] = Header(None)
) -> ModelListResponse:
    """
    List available models (OpenAI-compatible endpoint).
    
    This endpoint is called by Open WebUI to discover available models.
    You can customize the model list based on your deployment.
    
    Headers:
        authorization: Optional Bearer token for authentication
    
    Returns:
        List of available models in OpenAI format
    """
    
    logger.info("Models list requested")
    
    # Define available models
    # You can make this dynamic based on your configuration
    models = [
        ModelInfo(
            id="Vayu Maya",
            created=int(time.time()),
            owned_by="Tata Communications"
        ),
        ModelInfo(
            id="Vayu Maya v2",
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
    
    This is the main endpoint that Open WebUI uses to send chat messages
    to your Enterprise RAG Bot backend.
    
    Args:
        request: Chat completion request in OpenAI format
        http_request: FastAPI request object
        authorization: Optional Bearer token
    
    Returns:
        ChatCompletionResponse (non-streaming) or StreamingResponse (streaming)
    """
    
    try:
        # Log request
        logger.info(f"Chat completion request: model={request.model}, "
                   f"messages={len(request.messages)}, stream={request.stream}")
        
        # Get agent manager (initializes with vector and AI services)
        agent_manager = await get_agent_service()
        
        # Extract the latest user message
        if not request.messages:
            raise HTTPException(status_code=400, detail="No messages provided")
        
        user_message = request.messages[-1].content
        
        # Extract conversation history (all messages except the last one)
        conversation_history = [
            {"role": msg.role, "content": msg.content}
            for msg in request.messages[:-1]
        ]
        
        # Determine user ID (from request or default)
        user_id = request.user or "openwebui_user"
        
        # Grant full permissions for Open WebUI users
        # Since Open WebUI is the authenticated frontend, we trust its users
        # TODO: If you want role-based access, extract roles from JWT token or OpenWebUI headers
        user_roles = ["admin", "developer", "viewer"]  # Full permissions for authenticated users
        
        logger.info(f"👤 User: {user_id} | Roles: {user_roles}")
        
        # Generate stable session ID for Open WebUI conversations
        # Open WebUI maintains conversation context via messages array
        # We create a stable session ID based on the conversation
        import hashlib
        
        # Extract conversation ID from Open WebUI metadata if available
        # Open WebUI passes chat_id in some contexts, but primarily uses message history
        conversation_signature = ""
        
        # Try to get a stable identifier from the conversation
        if len(conversation_history) > 0:
            # Use first message as conversation anchor
            first_msg = conversation_history[0].get("content", "")[:100]
            conversation_signature = hashlib.md5(f"{user_id}:{first_msg}".encode()).hexdigest()[:16]
            session_id = f"openwebui_{conversation_signature}"
            logger.info(f"📋 Using stable session ID from conversation: {session_id}")
        else:
            # New conversation - create a time-bucketed session (10-minute windows)
            from datetime import datetime
            time_bucket = str(datetime.utcnow().hour) + str(datetime.utcnow().minute // 10)
            conversation_signature = hashlib.md5(f"{user_id}:{time_bucket}".encode()).hexdigest()[:16]
            session_id = f"openwebui_new_{conversation_signature}"
            logger.info(f"📋 New conversation session: {session_id}")
        
        # Handle streaming vs non-streaming
        if request.stream:
            return await _handle_streaming_response(
                agent_manager=agent_manager,
                user_message=user_message,
                user_id=user_id,
                user_roles=user_roles,
                conversation_history=conversation_history,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                session_id=session_id
            )
        else:
            return await _handle_non_streaming_response(
                agent_manager=agent_manager,
                user_message=user_message,
                user_id=user_id,
                user_roles=user_roles,
                conversation_history=conversation_history,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                session_id=session_id
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat completions: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


async def _handle_non_streaming_response(
    agent_manager,
    user_message: str,
    user_id: str,
    user_roles: List[str],
    conversation_history: List[Dict],
    model: str,
    temperature: float,
    max_tokens: int,
    session_id: str
) -> ChatCompletionResponse:
    """Handle non-streaming chat completion using the multi-agent system."""
    
    logger.info(f"Processing message for user {user_id} | Session: {session_id}")
    
    try:
        # Call the agent manager's process_request method
        # This goes through the full agent pipeline:
        # 1. Intent classification
        # 2. Route to appropriate agent (RAG, Execution, etc.)
        # 3. Multi-turn parameter collection if needed
        # 4. Execute operations via API executor
        result = await agent_manager.process_request(
            user_input=user_message,
            session_id=session_id,
            user_id=user_id,
            user_roles=user_roles
        )
        
        # Extract response content from agent result
        if not result.get("success", True):
            response_content = f"I apologize, but I encountered an error: {result.get('error', 'Unknown error')}"
            logger.warning(f"Agent returned error: {result.get('error')}")
        else:
            response_content = result.get("response", "No response generated")
            logger.info(f"Agent response generated | Length: {len(response_content)} chars")
        
        # Add metadata about the agent routing if available
        routing_info = result.get("routing", "Unknown")
        execution_result = result.get("execution_result")
        
        # Log additional context for debugging
        logger.debug(f"Routing: {routing_info} | Execution: {bool(execution_result)}")
        
    except Exception as e:
        logger.error(f"Error in agent processing: {str(e)}", exc_info=True)
        response_content = f"I apologize, but I encountered a technical error. Please try again."
    
    # Estimate token usage
    prompt_tokens = estimate_tokens(user_message)
    completion_tokens = estimate_tokens(response_content)
    
    # Create OpenAI-compatible response
    return ChatCompletionResponse(
        id=create_completion_id(),
        created=int(time.time()),
        model=model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=Message(
                    role="assistant",
                    content=response_content
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


async def _handle_streaming_response(
    agent_manager,
    user_message: str,
    user_id: str,
    user_roles: List[str],
    conversation_history: List[Dict],
    model: str,
    temperature: float,
    max_tokens: int,
    session_id: str
) -> StreamingResponse:
    """Handle streaming chat completion using the multi-agent system."""
    
    async def generate_stream():
        """Generate SSE (Server-Sent Events) stream in OpenAI format."""
        
        completion_id = create_completion_id()
        created = int(time.time())
        
        try:
            logger.info(f"Starting streaming response for session {session_id}")
            
            # Process through agent system (non-streaming for now)
            # TODO: Implement true streaming at agent level if needed
            result = await agent_manager.process_request(
                user_input=user_message,
                session_id=session_id,
                user_id=user_id,
                user_roles=user_roles
            )
            
            # Get response text
            response_text = result.get("response", "No response generated")
            
            # Stream the response preserving markdown structure
            # Split by lines instead of words to preserve formatting
            lines = response_text.split('\n')
            
            for i, line in enumerate(lines):
                # Add line with newline (except we handle them separately)
                chunk_content = line
                if i < len(lines) - 1:
                    chunk_content += "\n"
                
                # Format as OpenAI streaming response
                delta = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": chunk_content},
                            "finish_reason": None
                        }
                    ]
                }
                
                yield f"data: {json.dumps(delta)}\n\n"
            
            # Send final chunk
            final_delta = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }
                ]
            }
            
            yield f"data: {json.dumps(final_delta)}\n\n"
            yield "data: [DONE]\n\n"
            
            logger.info(f"Streaming complete for session {session_id}")
            
        except Exception as e:
            logger.error(f"Streaming error: {str(e)}", exc_info=True)
            error_delta = {
                "error": {
                    "message": str(e),
                    "type": "internal_error"
                }
            }
            yield f"data: {json.dumps(error_delta)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
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
        "version": "1.0.0"
    }

