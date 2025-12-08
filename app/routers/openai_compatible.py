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
                "model": "enterprise-rag-bot",
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
                "model": "enterprise-rag-bot",
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
    owned_by: str = "enterprise-rag-bot"
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
    Dependency to get the agent service.
    
    TODO: Replace this with your actual agent service initialization.
    This should import and return your AgentService instance.
    """
    # Example:
    # from app.services.agent_service import get_agent_service
    # return get_agent_service()
    
    # For now, we'll create a mock service
    # You should replace this with your actual implementation
    class MockAgentService:
        async def process_message(self, message: str, user_id: str, 
                                 user_roles: List[str], 
                                 conversation_history: List[Dict] = None,
                                 **kwargs) -> Dict[str, Any]:
            """Mock implementation - replace with actual agent service."""
            return {
                "success": True,
                "response": f"[Mock Response] Received: {message}",
                "agent_name": "MockAgent",
                "metadata": {}
            }
        
        async def process_message_stream(self, message: str, user_id: str, 
                                        user_roles: List[str], 
                                        conversation_history: List[Dict] = None,
                                        **kwargs) -> AsyncGenerator[str, None]:
            """Mock streaming implementation - replace with actual agent service."""
            response = f"[Mock Streaming Response] Processing: {message}"
            for word in response.split():
                yield word + " "
    
    return MockAgentService()


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
            id="enterprise-rag-bot",
            created=int(time.time()),
            owned_by="enterprise"
        ),
        ModelInfo(
            id="enterprise-rag-bot-v2",
            created=int(time.time()),
            owned_by="enterprise"
        )
    ]
    
    return ModelListResponse(data=models)


@router.post("/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    http_request: Request,
    authorization: Optional[str] = Header(None),
) -> ChatCompletionResponse | StreamingResponse:
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
        
        # Get agent service (replace with your actual service)
        agent_service = await get_agent_service()
        
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
        
        # Default roles (you might want to extract this from JWT or session)
        user_roles = ["user"]
        
        # Handle streaming vs non-streaming
        if request.stream:
            return await _handle_streaming_response(
                agent_service=agent_service,
                user_message=user_message,
                user_id=user_id,
                user_roles=user_roles,
                conversation_history=conversation_history,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
        else:
            return await _handle_non_streaming_response(
                agent_service=agent_service,
                user_message=user_message,
                user_id=user_id,
                user_roles=user_roles,
                conversation_history=conversation_history,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat completions: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


async def _handle_non_streaming_response(
    agent_service,
    user_message: str,
    user_id: str,
    user_roles: List[str],
    conversation_history: List[Dict],
    model: str,
    temperature: float,
    max_tokens: int
) -> ChatCompletionResponse:
    """Handle non-streaming chat completion."""
    
    # Call your agent service
    result = await agent_service.process_message(
        message=user_message,
        user_id=user_id,
        user_roles=user_roles,
        conversation_history=conversation_history,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    # Extract response content
    if not result.get("success", False):
        response_content = f"I apologize, but I encountered an error: {result.get('error', 'Unknown error')}"
    else:
        response_content = result.get("response", "No response generated")
    
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
    agent_service,
    user_message: str,
    user_id: str,
    user_roles: List[str],
    conversation_history: List[Dict],
    model: str,
    temperature: float,
    max_tokens: int
) -> StreamingResponse:
    """Handle streaming chat completion."""
    
    async def generate_stream():
        """Generate SSE (Server-Sent Events) stream."""
        
        completion_id = create_completion_id()
        created = int(time.time())
        
        try:
            # Get streaming response from agent
            async for chunk in agent_service.process_message_stream(
                message=user_message,
                user_id=user_id,
                user_roles=user_roles,
                conversation_history=conversation_history,
                temperature=temperature,
                max_tokens=max_tokens
            ):
                # Format as OpenAI streaming response
                delta = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": chunk},
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

