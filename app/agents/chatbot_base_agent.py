"""
Base Chatbot Agent Class
Foundation for all specialized chatbot agents
"""

from typing import List, Dict, Any, Optional, Callable
from enum import Enum
from datetime import datetime
import logging
import uuid
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AgentRole(str, Enum):
    """Agent role types"""
    CAPTAIN = "captain"
    SEARCH = "search"
    INFRASTRUCTURE = "infrastructure"
    API = "api"
    DATA = "data"
    ANALYSIS = "analysis"


class MessageType(str, Enum):
    """Message types for inter-agent communication"""
    QUERY = "query"
    RESPONSE = "response"
    TASK = "task"
    STATUS = "status"
    ERROR = "error"


class AgentMessage(BaseModel):
    """Message structure for agent communication"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender: str
    receiver: str
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    parent_id: Optional[str] = None


class ToolCall(BaseModel):
    """Record of a tool call"""
    tool_name: str
    parameters: Dict[str, Any]
    result: Dict[str, Any]
    execution_time: float
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class ChatbotBaseAgent:
    """Base class for all chatbot agents"""
    
    def __init__(self, name: str, role: AgentRole, description: str = ""):
        self.name = name
        self.role = role
        self.description = description
        self.id = str(uuid.uuid4())
        self.created_at = datetime.now()
        self.status = "idle"
        self.message_history: List[AgentMessage] = []
        self.tool_calls: List[ToolCall] = []
        self.current_task: Optional[Dict] = None
        self.tools: List[Any] = []
        
        logger.info(f"Initialized agent: {name} (ID: {self.id}) with role: {role}")
    
    def add_tool(self, tool: Any):
        """Add a tool to the agent"""
        self.tools.append(tool)
        logger.debug(f"Added tool to {self.name}: {tool.name if hasattr(tool, 'name') else tool}")
    
    def add_tools(self, tools: List[Any]):
        """Add multiple tools to the agent"""
        for tool in tools:
            self.add_tool(tool)
    
    async def process_message(self, message: AgentMessage) -> AgentMessage:
        """Process incoming message"""
        self.message_history.append(message)
        logger.info(f"Agent {self.name} received message from {message.sender}")
        
        # Override in subclasses
        response = AgentMessage(
            sender=self.name,
            receiver=message.sender,
            message_type=MessageType.RESPONSE,
            content={"status": "processed"}
        )
        
        self.message_history.append(response)
        return response
    
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool"""
        try:
            tool = next((t for t in self.tools if t.name == tool_name), None)
            if not tool:
                raise ValueError(f"Tool {tool_name} not found")
            
            logger.info(f"Executing tool {tool_name} with parameters: {parameters}")
            
            # Execute tool
            result = await self._call_maybe_async(tool.func, **parameters)
            
            # Record tool call
            tool_call = ToolCall(
                tool_name=tool_name,
                parameters=parameters,
                result=result if isinstance(result, dict) else {"output": str(result)},
                execution_time=0.0
            )
            self.tool_calls.append(tool_call)
            
            return result
        except Exception as e:
            logger.error(f"Tool execution failed: {str(e)}")
            raise
    
    def get_message_history(self) -> List[AgentMessage]:
        """Get message history"""
        return self.message_history
    
    def get_tool_calls(self) -> List[ToolCall]:
        """Get tool call history"""
        return self.tool_calls
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role.value,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "message_count": len(self.message_history),
            "tool_calls": len(self.tool_calls),
            "current_task": self.current_task
        }
    
    async def _call_maybe_async(self, fn: Callable, *args, **kwargs) -> Any:
        """Call function whether it's async or sync"""
        import inspect
        
        result = fn(*args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result
