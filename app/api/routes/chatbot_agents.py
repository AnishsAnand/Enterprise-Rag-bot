"""
API Routes for Chatbot Agent System
Endpoints for interacting with the captain agent and specialized agents
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import logging
from datetime import datetime

from app.agents.captain_chatbot_agent import captain_chatbot_agent

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chatbot-agents", tags=["chatbot-agents"])


# ============ REQUEST/RESPONSE MODELS ============

class ChatbotQueryRequest(BaseModel):
    """User query request"""
    query: str = Field(..., min_length=1, max_length=1000)
    context: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None


class ChatbotQueryResponse(BaseModel):
    """Chatbot response"""
    status: str
    query: str
    intent: str
    summary: str
    agent_responses: Dict[str, Any]
    actions_taken: List[str]
    next_steps: List[str]
    timestamp: str


class AgentStatusResponse(BaseModel):
    """Agent status response"""
    id: str
    name: str
    role: str
    status: str
    message_count: int
    tool_calls: int


# ============ ENDPOINTS ============

@router.post("/query", response_model=ChatbotQueryResponse)
async def process_chatbot_query(request: ChatbotQueryRequest):
    """
    Process a user query through the captain agent
    Routes to appropriate specialized agents based on intent
    """
    try:
        logger.info(f"Processing chatbot query: {request.query}")
        
        response = await captain_chatbot_agent.process_user_query(
            request.query,
            context=request.context
        )
        
        return ChatbotQueryResponse(
            status=response.get("status", "success"),
            query=response.get("query", ""),
            intent=response.get("intent", ""),
            summary=response.get("summary", ""),
            agent_responses=response.get("agent_responses", {}),
            actions_taken=response.get("actions_taken", []),
            next_steps=response.get("next_steps", []),
            timestamp=response.get("timestamp", datetime.now().isoformat())
        )
    
    except Exception as e:
        logger.error(f"Query processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@router.get("/agents/status")
async def get_agents_status():
    """Get status of all agents"""
    try:
        status = captain_chatbot_agent.get_all_agents_status()
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "agents": status
        }
    except Exception as e:
        logger.error(f"Failed to get agents status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/{agent_role}/status")
async def get_agent_status(agent_role: str):
    """Get status of a specific agent"""
    try:
        agent = captain_chatbot_agent.get_agent_by_role(agent_role)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_role} not found")
        
        return {
            "status": "success",
            "agent": agent.get_status(),
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get agent status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversation-history")
async def get_conversation_history(limit: int = 50):
    """Get conversation history"""
    try:
        history = captain_chatbot_agent.get_conversation_history()
        return {
            "status": "success",
            "total": len(history),
            "history": history[-limit:],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get conversation history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/{agent_role}/tools")
async def get_agent_tools(agent_role: str):
    """Get available tools for a specific agent"""
    try:
        agent = captain_chatbot_agent.get_agent_by_role(agent_role)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_role} not found")
        
        tools_info = [
            {
                "name": tool.name,
                "description": tool.description
            }
            for tool in agent.tools
        ]
        
        return {
            "status": "success",
            "agent": agent_role,
            "tools": tools_info,
            "total": len(tools_info),
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get agent tools: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/{agent_role}/message-history")
async def get_agent_message_history(agent_role: str, limit: int = 20):
    """Get message history for a specific agent"""
    try:
        agent = captain_chatbot_agent.get_agent_by_role(agent_role)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_role} not found")
        
        history = agent.get_message_history()
        messages = [
            {
                "id": msg.id,
                "sender": msg.sender,
                "receiver": msg.receiver,
                "type": msg.message_type.value,
                "timestamp": msg.timestamp
            }
            for msg in history[-limit:]
        ]
        
        return {
            "status": "success",
            "agent": agent_role,
            "messages": messages,
            "total": len(history),
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get message history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agents/{agent_role}/execute-tool")
async def execute_agent_tool(agent_role: str, tool_name: str, parameters: Dict[str, Any]):
    """Execute a specific tool on an agent"""
    try:
        agent = captain_chatbot_agent.get_agent_by_role(agent_role)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_role} not found")
        
        result = await agent.execute_tool(tool_name, parameters)
        
        return {
            "status": "success",
            "agent": agent_role,
            "tool": tool_name,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Tool execution failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def chatbot_agents_health():
    """Health check for chatbot agent system"""
    try:
        status = captain_chatbot_agent.get_all_agents_status()
        
        return {
            "status": "healthy",
            "captain_agent": status["captain"]["status"],
            "agents_count": len(status["agents"]),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
