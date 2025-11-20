from typing import Any, Dict, List, Optional, Set
import logging
import asyncio
from datetime import datetime
from enum import Enum

from app.agents.base_agent import BaseAgent, AgentRole, AgentStatus, AgentMessage
from app.agents.specialized_agents import SearchAgent, AnalysisAgent, APIAgent, DataAgent

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Task execution status"""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Task:
    """Represents a task to be executed by agents"""
    
    def __init__(self, task_id: str, description: str, required_agents: List[str], priority: int = 1):
        self.task_id = task_id
        self.description = description
        self.required_agents = required_agents
        self.priority = priority
        self.status = TaskStatus.PENDING
        self.assigned_agent: Optional[str] = None
        self.created_at = datetime.utcnow().isoformat()
        self.started_at: Optional[str] = None
        self.completed_at: Optional[str] = None
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.retry_count = 0
        self.max_retries = 3
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "description": self.description,
            "status": self.status.value,
            "assigned_agent": self.assigned_agent,
            "priority": self.priority,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "result": self.result,
            "error": self.error,
            "retry_count": self.retry_count
        }


class CaptainAgent(BaseAgent):
    """
    Captain Agent - Orchestrates all specialized agents.
    Manages task distribution, monitoring, and error handling.
    """
    
    def __init__(self):
        super().__init__(
            agent_id="captain-agent-001",
            role=AgentRole.CAPTAIN,
            name="Captain Agent"
        )
        
        
        self.agents: Dict[str, BaseAgent] = {
            "search": SearchAgent(),
            "analysis": AnalysisAgent(),
            "api": APIAgent(),
            "data": DataAgent(),
        }
        
        
        self.tasks: Dict[str, Task] = {}
        self.task_queue: List[Task] = []
        self.completed_tasks: List[Task] = []
        self.failed_tasks: List[Task] = []
        
        
        self.execution_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Any] = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "average_execution_time": 0.0,
            "agent_performance": {}
        }
        
        self.capabilities = [
            "task_orchestration",
            "agent_coordination",
            "error_handling",
            "performance_monitoring",
            "dynamic_routing"
        ]
        
        logger.info(f"Captain Agent initialized with {len(self.agents)} specialized agents")
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process incoming request and orchestrate agent execution.
        Main entry point for the captain agent.
        """
        try:
            self.status = AgentStatus.PROCESSING
            
            
            query = request.get("query", "")
            task_type = request.get("task_type", "auto")
            user_id = request.get("user_id")
            context = request.get("context", {})
            
            if not query:
                return {"error": "Query is required", "status": "failed"}
            
            
            task = Task(
                task_id=f"task-{datetime.utcnow().timestamp()}",
                description=query,
                required_agents=self._determine_required_agents(query, task_type),
                priority=request.get("priority", 1)
            )
            
            logger.info(f"Captain Agent created task {task.task_id} with agents: {task.required_agents}")
            
            
            result = await self._execute_task(task, context)
            
            self.status = AgentStatus.COMPLETED
            return {
                "status": "success",
                "task_id": task.task_id,
                "result": result,
                "agents_used": task.required_agents,
                "execution_time": (datetime.utcnow().isoformat())
            }
        
        except Exception as e:
            logger.exception(f"CaptainAgent error: {e}")
            self.increment_error_count()
            self.status = AgentStatus.ERROR
            return {"error": str(e), "status": "failed"}
    
    def _determine_required_agents(self, query: str, task_type: str) -> List[str]:
        """Determine which agents are needed for the query"""
        query_lower = query.lower()
        required = []
        
        if task_type == "auto":
            
            if any(word in query_lower for word in ["search", "find", "look for", "query"]):
                required.append("search")
            
            if any(word in query_lower for word in ["analyze", "sentiment", "extract", "summarize"]):
                required.append("analysis")
            
            if any(word in query_lower for word in ["api", "call", "request", "fetch"]):
                required.append("api")
            
            if any(word in query_lower for word in ["user", "data", "document", "retrieve"]):
                required.append("data")
            
            
            if not required:
                required.append("search")
        else:
            
            if task_type in self.agents:
                required.append(task_type)
            else:
                required.append("search")
        
        return required
    
    async def _execute_task(self, task: Task, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task using appropriate agents"""
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.utcnow().isoformat()
        
        results = {}
        
        try:
            
            agent_tasks = []
            
            for agent_name in task.required_agents:
                if agent_name not in self.agents:
                    logger.warning(f"Agent {agent_name} not found")
                    continue
                
                agent = self.agents[agent_name]
                
                
                agent_request = self._prepare_agent_request(agent_name, task.description, context)
                
                
                agent_tasks.append(self._execute_agent(agent, agent_request, task.task_id))
            
            
            if agent_tasks:
                agent_results = await asyncio.gather(*agent_tasks, return_exceptions=True)
                
                for i, agent_name in enumerate(task.required_agents):
                    if i < len(agent_results):
                        result = agent_results[i]
                        if isinstance(result, Exception):
                            results[agent_name] = {"error": str(result), "status": "failed"}
                        else:
                            results[agent_name] = result
            
            
            task.result = results
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.utcnow().isoformat()
            self.completed_tasks.append(task)
            
            logger.info(f"Task {task.task_id} completed successfully")
            
            return {
                "task_id": task.task_id,
                "results": results,
                "agents_executed": task.required_agents
            }
        
        except Exception as e:
            logger.exception(f"Task execution failed: {e}")
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.utcnow().isoformat()
            self.failed_tasks.append(task)
            
            
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                logger.info(f"Retrying task {task.task_id} (attempt {task.retry_count})")
                await asyncio.sleep(2 ** task.retry_count)  # Exponential backoff
                return await self._execute_task(task, context)
            
            raise
    
    async def _execute_agent(self, agent: BaseAgent, request: Dict[str, Any], task_id: str) -> Dict[str, Any]:
        """Execute a single agent with timeout and error handling"""
        try:
            logger.info(f"Executing agent {agent.agent_id} for task {task_id}")
            
           
            result = await asyncio.wait_for(
                agent.process_request(request),
                timeout=30.0
            )
            
            logger.info(f"Agent {agent.agent_id} completed successfully")
            return result
        
        except asyncio.TimeoutError:
            logger.error(f"Agent {agent.agent_id} timed out")
            return {"error": "Agent execution timed out", "status": "failed"}
        
        except Exception as e:
            logger.exception(f"Agent {agent.agent_id} failed: {e}")
            return {"error": str(e), "status": "failed"}
    
    def _prepare_agent_request(self, agent_name: str, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare request for specific agent"""
        base_request = {
            "query": query,
            "context": context
        }
        
        if agent_name == "search":
            return {
                **base_request,
                "search_type": context.get("search_type", "knowledge_base"),
                "max_results": context.get("max_results", 10),
                "search_depth": context.get("search_depth", "balanced")
            }
        
        elif agent_name == "analysis":
            return {
                **base_request,
                "analysis_type": context.get("analysis_type", "summary"),
                "content": context.get("content", query),
                "entity_types": context.get("entity_types", ["PERSON", "ORG", "LOCATION"])
            }
        
        elif agent_name == "api":
            return {
                **base_request,
                "url": context.get("url", ""),
                "method": context.get("method", "GET"),
                "payload": context.get("payload"),
                "headers": context.get("headers")
            }
        
        elif agent_name == "data":
            return {
                **base_request,
                "data_type": context.get("data_type", "user"),
                "user_id": context.get("user_id", ""),
                "document_id": context.get("document_id", "")
            }
        
        return base_request
    
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Captain agent doesn't directly execute tools, delegates to specialized agents"""
        logger.warning(f"Captain agent received direct tool call: {tool_name}")
        return {"error": "Captain agent delegates to specialized agents", "status": "failed"}
    
    def get_agent_status(self, agent_name: Optional[str] = None) -> Dict[str, Any]:
        """Get status of agents"""
        if agent_name:
            if agent_name in self.agents:
                return self.agents[agent_name].get_status()
            return {"error": f"Agent {agent_name} not found"}
        
        return {
            "captain": self.get_status(),
            "agents": {name: agent.get_status() for name, agent in self.agents.items()}
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            **self.performance_metrics,
            "total_tasks": len(self.completed_tasks) + len(self.failed_tasks),
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "pending_tasks": len(self.task_queue),
            "agents": {name: agent.get_status() for name, agent in self.agents.items()}
        }
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task"""
        if task_id in self.tasks:
            return self.tasks[task_id].to_dict()
        
        for task in self.completed_tasks + self.failed_tasks:
            if task.task_id == task_id:
                return task.to_dict()
        
        return None



captain_agent = CaptainAgent()
