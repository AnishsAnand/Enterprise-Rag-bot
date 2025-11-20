from typing import Any, Dict, List, Optional, Callable
from abc import ABC, abstractmethod
from datetime import datetime
import logging
import json
from enum import Enum

logger = logging.getLogger(__name__)


class AgentRole(str, Enum):
    """Agent roles in the system"""
    CAPTAIN = "captain"
    SEARCH = "search"
    ANALYSIS = "analysis"
    API = "api"
    DATA = "data"
    KNOWLEDGE = "knowledge"


class AgentStatus(str, Enum):
    """Agent execution status"""
    IDLE = "idle"
    PROCESSING = "processing"
    WAITING = "waiting"
    COMPLETED = "completed"
    ERROR = "error"


class ToolCall(Dict[str, Any]):
    """Represents a tool call made by an agent"""
    
    def __init__(self, tool_name: str, parameters: Dict[str, Any], agent_id: str):
        super().__init__()
        self["tool_name"] = tool_name
        self["parameters"] = parameters
        self["agent_id"] = agent_id
        self["timestamp"] = datetime.utcnow().isoformat()
        self["status"] = "pending"
        self["result"] = None
        self["error"] = None


class AgentMessage:
    """Message passed between agents"""
    
    def __init__(
        self,
        sender_id: str,
        recipient_id: str,
        content: str,
        message_type: str = "request",
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.sender_id = sender_id
        self.recipient_id = recipient_id
        self.content = content
        self.message_type = message_type  
        self.metadata = metadata or {}
        self.timestamp = datetime.utcnow().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "content": self.content,
            "message_type": self.message_type,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }


class BaseAgent(ABC):
    """Base class for all agents"""
    
    def __init__(self, agent_id: str, role: AgentRole, name: str):
        self.agent_id = agent_id
        self.role = role
        self.name = name
        self.status = AgentStatus.IDLE
        self.tool_calls: List[ToolCall] = []
        self.message_history: List[AgentMessage] = []
        self.capabilities: List[str] = []
        self.created_at = datetime.utcnow().isoformat()
        self.last_activity = datetime.utcnow().isoformat()
        self.error_count = 0
        self.max_errors = 5
    
    @abstractmethod
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a request and return response"""
        pass
    
    @abstractmethod
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool and return result"""
        pass
    
    async def send_message(self, recipient_id: str, content: str, message_type: str = "request") -> AgentMessage:
        """Send message to another agent"""
        message = AgentMessage(
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            content=content,
            message_type=message_type
        )
        self.message_history.append(message)
        logger.info(f"Agent {self.agent_id} sent message to {recipient_id}: {message_type}")
        return message
    
    async def receive_message(self, message: AgentMessage) -> Dict[str, Any]:
        """Receive and process message from another agent"""
        self.message_history.append(message)
        self.last_activity = datetime.utcnow().isoformat()
        logger.info(f"Agent {self.agent_id} received message from {message.sender_id}")
        return {"status": "received", "message_id": id(message)}
    
    def record_tool_call(self, tool_call: ToolCall):
        """Record a tool call"""
        self.tool_calls.append(tool_call)
        logger.debug(f"Agent {self.agent_id} recorded tool call: {tool_call['tool_name']}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "role": self.role.value,
            "status": self.status.value,
            "capabilities": self.capabilities,
            "tool_calls_count": len(self.tool_calls),
            "messages_count": len(self.message_history),
            "error_count": self.error_count,
            "created_at": self.created_at,
            "last_activity": self.last_activity
        }
    
    def increment_error_count(self):
        """Increment error count"""
        self.error_count += 1
        if self.error_count >= self.max_errors:
            self.status = AgentStatus.ERROR
            logger.error(f"Agent {self.agent_id} reached max error count")
    
    def reset_error_count(self):
        """Reset error count"""
        self.error_count = 0
