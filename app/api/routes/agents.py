"""
API routes for agent management and orchestration.
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging

from app.agents.captain_agent import captain_agent
from app.agents.agent_communication import message_broker

logger = logging.getLogger(__name__)
router = APIRouter()


class AgentQueryRequest(BaseModel):
    """Request model for agent queries"""
    query: str = Field(..., min_length=1, max_length=2000)
    task_type: str = Field(default="auto", description="Type of task: auto, search, analysis, api, data")
    user_id: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(default=1, ge=1, le=10)


class ToolCallRequest(BaseModel):
    """Request model for tool calls"""
    tool_name: str
    parameters: Dict[str, Any]
    agent_id: Optional[str] = None


@router.post("/agents/query")
async def agent_query(request: AgentQueryRequest):
    """
    Submit a query to the agent system.
    The captain agent will orchestrate specialized agents to handle the request.
    """
    try:
        logger.info(f"Received agent query: {request.query[:100]}")
        
        result = await captain_agent.process_request({
            "query": request.query,
            "task_type": request.task_type,
            "user_id": request.user_id,
            "context": request.context,
            "priority": request.priority
        })
        
        return result
    
    except Exception as e:
        logger.exception(f"Agent query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/status")
async def get_agents_status(agent_name: Optional[str] = Query(None)):
    """Get status of agents"""
    try:
        status = captain_agent.get_agent_status(agent_name)
        return {"status": "success", "data": status}
    
    except Exception as e:
        logger.exception(f"Failed to get agent status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/metrics")
async def get_performance_metrics():
    """Get performance metrics for all agents"""
    try:
        metrics = captain_agent.get_performance_metrics()
        return {"status": "success", "metrics": metrics}
    
    except Exception as e:
        logger.exception(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/task/{task_id}")
async def get_task_status(task_id: str):
    """Get status of a specific task"""
    try:
        task_status = captain_agent.get_task_status(task_id)
        if not task_status:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        return {"status": "success", "task": task_status}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to get task status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/messages")
async def get_message_history(agent_id: Optional[str] = Query(None), limit: int = Query(100, ge=1, le=1000)):
    """Get message history"""
    try:
        history = message_broker.get_message_history(agent_id, limit)
        return {"status": "success", "messages": history}
    
    except Exception as e:
        logger.exception(f"Failed to get message history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agents/tool-call")
async def execute_tool_call(request: ToolCallRequest):
    """
    Execute a tool call through the agent system.
    """
    try:
        logger.info(f"Tool call: {request.tool_name}")
        
        # Route to appropriate agent based on tool
        if request.agent_id:
            agent_name = request.agent_id.split("-")[0]
        else:
            agent_name = "search"  # Default
        
        if agent_name not in captain_agent.agents:
            raise HTTPException(status_code=400, detail=f"Unknown agent: {agent_name}")
        
        agent = captain_agent.agents[agent_name]
        result = await agent.execute_tool(request.tool_name, request.parameters)
        
        return {"status": "success", "result": result}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Tool call failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
