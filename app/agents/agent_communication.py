from typing import Any, Dict, List, Optional, Callable
import logging
import asyncio
from datetime import datetime
from enum import Enum

from app.agents.base_agent import AgentMessage

logger = logging.getLogger(__name__)


class MessageBroker:
    """
    Central message broker for agent communication.
    Handles message routing, queuing, and delivery.
    """
    
    def __init__(self):
        self.message_queue: Dict[str, List[AgentMessage]] = {}
        self.subscribers: Dict[str, List[Callable]] = {}
        self.message_history: List[AgentMessage] = []
        self.max_history_size = 10000
    
    async def send_message(self, message: AgentMessage) -> bool:
        """Send message to recipient"""
        try:
            recipient_id = message.recipient_id
            
            # Add to recipient's queue
            if recipient_id not in self.message_queue:
                self.message_queue[recipient_id] = []
            
            self.message_queue[recipient_id].append(message)
            self.message_history.append(message)
            
            # Trim history if too large
            if len(self.message_history) > self.max_history_size:
                self.message_history = self.message_history[-self.max_history_size:]
            
            # Notify subscribers
            if recipient_id in self.subscribers:
                for callback in self.subscribers[recipient_id]:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(message)
                        else:
                            callback(message)
                    except Exception as e:
                        logger.error(f"Subscriber callback failed: {e}")
            
            logger.debug(f"Message sent from {message.sender_id} to {recipient_id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False
    
    async def receive_messages(self, agent_id: str) -> List[AgentMessage]:
        """Receive all messages for an agent"""
        messages = self.message_queue.get(agent_id, [])
        self.message_queue[agent_id] = []  # Clear queue
        return messages
    
    def subscribe(self, agent_id: str, callback: Callable):
        """Subscribe to messages for an agent"""
        if agent_id not in self.subscribers:
            self.subscribers[agent_id] = []
        self.subscribers[agent_id].append(callback)
        logger.info(f"Subscribed {agent_id} to message broker")
    
    def unsubscribe(self, agent_id: str, callback: Callable):
        """Unsubscribe from messages"""
        if agent_id in self.subscribers:
            self.subscribers[agent_id].remove(callback)
            logger.info(f"Unsubscribed {agent_id} from message broker")
    
    def get_message_history(self, agent_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get message history"""
        if agent_id:
            messages = [m for m in self.message_history if m.sender_id == agent_id or m.recipient_id == agent_id]
        else:
            messages = self.message_history
        
        return [m.to_dict() for m in messages[-limit:]]


# Global message broker instance
message_broker = MessageBroker()
