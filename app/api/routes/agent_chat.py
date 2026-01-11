"""
Agent Chat API - Endpoint for multi-agent conversational CRUD operations.
Integrates the LangChain-based agent system with the FastAPI backend.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
import logging
import uuid
from datetime import datetime

from app.agents import get_agent_manager
from app.services.ai_service import ai_service
from app.services.postgres_service import postgres_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agent", tags=["agent-chat"])


# Request/Response Models
class AgentChatRequest(BaseModel):
    """Request model for agent chat."""
    message: str = Field(..., description="User's message")
    session_id: Optional[str] = Field(None, description="Conversation session ID")
    user_id: str = Field(default="default_user", description="User identifier")
    user_roles: Optional[List[str]] = Field(default=["viewer"], description="User's roles")


class AgentChatResponse(BaseModel):
    """Response model for agent chat."""
    success: bool
    response: str
    session_id: str
    routing: Optional[str] = None
    execution_result: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ConversationStatusResponse(BaseModel):
    """Response model for conversation status."""
    found: bool
    session_id: str
    state: Optional[Dict[str, Any]] = None
    message: Optional[str] = None


class AgentStatsResponse(BaseModel):
    """Response model for agent statistics."""
    initialized: bool
    total_requests: int
    active_sessions: int
    agents: Dict[str, Optional[str]]
    initialization_time: Optional[str] = None


# Endpoints
@router.post("/chat", response_model=AgentChatResponse)
async def agent_chat(request: AgentChatRequest):
    """
    Process a user message through the multi-agent system.
    
    This endpoint handles:
    - Intent detection for CRUD operations
    - Multi-turn parameter collection
    - Parameter validation
    - Operation execution
    - RAG-based question answering
    
    Example requests:
    - "Create a new Kubernetes cluster named prod-cluster"
    - "Delete the firewall rule allow-http"
    - "Show me all clusters"
    - "How do I configure a load balancer?"
    """
    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        logger.info(
            f"📨 Agent chat request | Session: {session_id} | "
            f"User: {request.user_id} | Message: {request.message[:50]}..."
        )
        
        # Get agent manager (initializes if needed)
        manager = get_agent_manager(
            vector_service=milvus_service,
            ai_service=ai_service
        )
        
        # Process request through agent system
        result = await manager.process_request(
            user_input=request.message,
            session_id=session_id,
            user_id=request.user_id,
            user_roles=request.user_roles
        )
        
        # Format response
        return AgentChatResponse(
            success=result.get("success", True),
            response=result.get("response", ""),
            session_id=session_id,
            routing=result.get("routing"),
            execution_result=result.get("execution_result"),
            metadata=result.get("metadata"),
            error=result.get("error")
        )
        
    except Exception as e:
        logger.error(f"❌ Agent chat failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Agent chat processing failed: {str(e)}"
        )


@router.get("/conversation/{session_id}", response_model=ConversationStatusResponse)
async def get_conversation_status(session_id: str):
    """
    Get the status of a conversation session.
    
    Returns information about:
    - Current intent and operation
    - Collected parameters
    - Missing parameters
    - Conversation history
    - Execution status
    """
    try:
        manager = get_agent_manager(
            vector_service=milvus_service,
            ai_service=ai_service
        )
        
        status = await manager.get_conversation_status(session_id)
        
        return ConversationStatusResponse(
            found=status.get("found", False),
            session_id=session_id,
            state=status.get("state"),
            message=status.get("message")
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to get conversation status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get conversation status: {str(e)}"
        )


@router.delete("/conversation/{session_id}")
async def reset_conversation(session_id: str):
    """
    Reset/delete a conversation session.
    
    This clears all conversation state, collected parameters, and history.
    """
    try:
        manager = get_agent_manager(
            vector_service=milvus_service,
            ai_service=ai_service
        )
        
        result = await manager.reset_conversation(session_id)
        
        return {
            "success": result.get("success", False),
            "message": result.get("message", ""),
            "session_id": session_id
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to reset conversation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reset conversation: {str(e)}"
        )


@router.get("/stats", response_model=AgentStatsResponse)
async def get_agent_stats():
    """
    Get statistics about the agent system.
    
    Returns:
    - Initialization status
    - Total requests processed
    - Active conversation sessions
    - Agent information
    """
    try:
        manager = get_agent_manager(
            vector_service=milvus_service,
            ai_service=ai_service
        )
        
        stats = manager.get_stats()
        
        return AgentStatsResponse(**stats)
        
    except Exception as e:
        logger.error(f"❌ Failed to get agent stats: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get agent stats: {str(e)}"
        )


@router.post("/cleanup")
async def cleanup_old_sessions(max_age_hours: int = 24):
    """
    Clean up old conversation sessions.
    
    Removes completed/cancelled sessions older than the specified age.
    """
    try:
        manager = get_agent_manager(
            vector_service=milvus_service,
            ai_service=ai_service
        )
        
        cleaned_count = manager.cleanup_old_sessions(max_age_hours)
        
        return {
            "success": True,
            "cleaned_sessions": cleaned_count,
            "max_age_hours": max_age_hours,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to cleanup sessions: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cleanup sessions: {str(e)}"
        )


@router.get("/health")
async def agent_health_check():
    """
    Health check for the agent system.
    """
    try:
        manager = get_agent_manager(
            vector_service=milvus_service,
            ai_service=ai_service
        )
        
        return {
            "status": "healthy",
            "initialized": manager.initialized,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Agent health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
