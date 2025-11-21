"""
Conversation State Manager for multi-turn parameter collection.
Tracks user intent, collected parameters, and conversation flow.
"""

from typing import Any, Dict, List, Optional, Set
from datetime import datetime
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)


class ConversationStatus(Enum):
    """Status of the conversation flow."""
    INITIATED = "initiated"
    COLLECTING_PARAMS = "collecting_params"
    VALIDATING = "validating"
    READY_TO_EXECUTE = "ready_to_execute"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ConversationState:
    """
    Manages the state of a multi-turn conversation for CRUD operations.
    Tracks intent, parameters, validation status, and execution flow.
    """
    
    def __init__(self, session_id: str, user_id: str):
        """
        Initialize conversation state.
        
        Args:
            session_id: Unique session identifier
            user_id: User identifier for permission checks
        """
        self.session_id = session_id
        self.user_id = user_id
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        
        # Intent and operation tracking
        self.intent: Optional[str] = None  # e.g., "create_k8s_cluster"
        self.resource_type: Optional[str] = None  # e.g., "k8s_cluster"
        self.operation: Optional[str] = None  # e.g., "create", "update", "delete", "list"
        
        # Parameter collection
        self.required_params: Set[str] = set()
        self.optional_params: Set[str] = set()
        self.collected_params: Dict[str, Any] = {}
        self.missing_params: Set[str] = set()
        self.invalid_params: Dict[str, str] = {}  # param_name -> error_message
        
        # Conversation flow
        self.status: ConversationStatus = ConversationStatus.INITIATED
        self.conversation_history: List[Dict[str, Any]] = []
        self.clarification_count: int = 0
        self.max_clarifications: int = 5
        
        # Execution tracking
        self.execution_result: Optional[Dict[str, Any]] = None
        self.error_message: Optional[str] = None
        
        # Agent tracking
        self.active_agent: Optional[str] = None
        self.agent_handoffs: List[Dict[str, str]] = []
        
        logger.info(f"✅ Created conversation state for session {session_id}, user {user_id}")
    
    def set_intent(
        self,
        resource_type: str,
        operation: str,
        required_params: List[str],
        optional_params: Optional[List[str]] = None
    ) -> None:
        """
        Set the detected intent and initialize parameter tracking.
        
        Args:
            resource_type: Type of resource (k8s_cluster, firewall, etc.)
            operation: CRUD operation (create, read, update, delete, list)
            required_params: List of required parameter names
            optional_params: List of optional parameter names
        """
        self.resource_type = resource_type
        self.operation = operation
        self.intent = f"{operation}_{resource_type}"
        self.required_params = set(required_params)
        self.optional_params = set(optional_params or [])
        self.missing_params = self.required_params.copy()
        self.status = ConversationStatus.COLLECTING_PARAMS
        self.updated_at = datetime.utcnow()
        
        logger.info(
            f"📋 Intent set: {self.intent} | "
            f"Required: {len(self.required_params)} | "
            f"Optional: {len(self.optional_params)}"
        )
    
    def add_parameter(self, param_name: str, param_value: Any, is_valid: bool = True) -> None:
        """
        Add a collected parameter to the state.
        
        Args:
            param_name: Name of the parameter
            param_value: Value of the parameter
            is_valid: Whether the parameter passed validation
        """
        if is_valid:
            self.collected_params[param_name] = param_value
            self.missing_params.discard(param_name)
            if param_name in self.invalid_params:
                del self.invalid_params[param_name]
            logger.info(f"✅ Parameter collected: {param_name} = {param_value}")
        else:
            self.invalid_params[param_name] = f"Invalid value: {param_value}"
            logger.warning(f"⚠️ Invalid parameter: {param_name} = {param_value}")
        
        self.updated_at = datetime.utcnow()
    
    def add_parameters(self, params: Dict[str, Any]) -> None:
        """
        Add multiple parameters at once.
        
        Args:
            params: Dictionary of parameter name-value pairs
        """
        for param_name, param_value in params.items():
            self.add_parameter(param_name, param_value)
    
    def mark_parameter_invalid(self, param_name: str, error_message: str) -> None:
        """
        Mark a parameter as invalid with an error message.
        
        Args:
            param_name: Name of the invalid parameter
            error_message: Description of why it's invalid
        """
        self.invalid_params[param_name] = error_message
        if param_name in self.collected_params:
            del self.collected_params[param_name]
        self.missing_params.add(param_name)
        self.updated_at = datetime.utcnow()
        
        logger.warning(f"❌ Parameter marked invalid: {param_name} - {error_message}")
    
    def is_ready_to_execute(self) -> bool:
        """
        Check if all required parameters are collected and valid.
        
        Returns:
            True if ready to execute, False otherwise
        """
        ready = (
            len(self.missing_params) == 0 and
            len(self.invalid_params) == 0 and
            self.status != ConversationStatus.FAILED
        )
        
        if ready and self.status == ConversationStatus.COLLECTING_PARAMS:
            self.status = ConversationStatus.READY_TO_EXECUTE
            self.updated_at = datetime.utcnow()
            logger.info(f"✅ Conversation ready to execute: {self.intent}")
        
        return ready
    
    def get_missing_params_message(self) -> str:
        """
        Generate a user-friendly message about missing parameters.
        
        Returns:
            Message describing missing parameters
        """
        if not self.missing_params:
            return "All required parameters collected."
        
        missing_list = ", ".join(sorted(self.missing_params))
        return f"I still need the following information: {missing_list}"
    
    def get_invalid_params_message(self) -> str:
        """
        Generate a user-friendly message about invalid parameters.
        
        Returns:
            Message describing invalid parameters
        """
        if not self.invalid_params:
            return ""
        
        messages = []
        for param, error in self.invalid_params.items():
            messages.append(f"- {param}: {error}")
        
        return "Please correct the following:\n" + "\n".join(messages)
    
    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a message to conversation history.
        
        Args:
            role: Message role (user, assistant, system)
            content: Message content
            metadata: Additional metadata
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }
        self.conversation_history.append(message)
        self.updated_at = datetime.utcnow()
    
    def add_clarification(self, question: str) -> bool:
        """
        Add a clarification question to the conversation.
        
        Args:
            question: Clarification question to ask user
            
        Returns:
            True if clarification added, False if max clarifications reached
        """
        if self.clarification_count >= self.max_clarifications:
            logger.warning(f"⚠️ Max clarifications ({self.max_clarifications}) reached")
            return False
        
        self.clarification_count += 1
        self.add_message("assistant", question, {"type": "clarification"})
        return True
    
    def handoff_to_agent(self, from_agent: str, to_agent: str, reason: str) -> None:
        """
        Record an agent handoff.
        
        Args:
            from_agent: Agent handing off
            to_agent: Agent receiving handoff
            reason: Reason for handoff
        """
        handoff = {
            "from": from_agent,
            "to": to_agent,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.agent_handoffs.append(handoff)
        self.active_agent = to_agent
        self.updated_at = datetime.utcnow()
        
        logger.info(f"🔄 Agent handoff: {from_agent} -> {to_agent} ({reason})")
    
    def set_execution_result(self, result: Dict[str, Any]) -> None:
        """
        Set the execution result.
        
        Args:
            result: Execution result dictionary
        """
        self.execution_result = result
        self.status = ConversationStatus.COMPLETED if result.get("success") else ConversationStatus.FAILED
        self.updated_at = datetime.utcnow()
        
        if result.get("success"):
            logger.info(f"✅ Execution completed successfully: {self.intent}")
        else:
            self.error_message = result.get("error", "Unknown error")
            logger.error(f"❌ Execution failed: {self.error_message}")
    
    def cancel(self, reason: str = "User cancelled") -> None:
        """
        Cancel the conversation.
        
        Args:
            reason: Reason for cancellation
        """
        self.status = ConversationStatus.CANCELLED
        self.error_message = reason
        self.updated_at = datetime.utcnow()
        logger.info(f"🚫 Conversation cancelled: {reason}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert state to dictionary for serialization.
        
        Returns:
            Dictionary representation of state
        """
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "intent": self.intent,
            "resource_type": self.resource_type,
            "operation": self.operation,
            "status": self.status.value,
            "required_params": list(self.required_params),
            "optional_params": list(self.optional_params),
            "collected_params": self.collected_params,
            "missing_params": list(self.missing_params),
            "invalid_params": self.invalid_params,
            "conversation_history": self.conversation_history,
            "clarification_count": self.clarification_count,
            "execution_result": self.execution_result,
            "error_message": self.error_message,
            "active_agent": self.active_agent,
            "agent_handoffs": self.agent_handoffs
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationState":
        """
        Create ConversationState from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            ConversationState instance
        """
        state = cls(data["session_id"], data["user_id"])
        state.created_at = datetime.fromisoformat(data["created_at"])
        state.updated_at = datetime.fromisoformat(data["updated_at"])
        state.intent = data.get("intent")
        state.resource_type = data.get("resource_type")
        state.operation = data.get("operation")
        state.status = ConversationStatus(data["status"])
        state.required_params = set(data.get("required_params", []))
        state.optional_params = set(data.get("optional_params", []))
        state.collected_params = data.get("collected_params", {})
        state.missing_params = set(data.get("missing_params", []))
        state.invalid_params = data.get("invalid_params", {})
        state.conversation_history = data.get("conversation_history", [])
        state.clarification_count = data.get("clarification_count", 0)
        state.execution_result = data.get("execution_result")
        state.error_message = data.get("error_message")
        state.active_agent = data.get("active_agent")
        state.agent_handoffs = data.get("agent_handoffs", [])
        return state
    
    def __repr__(self) -> str:
        return (
            f"<ConversationState(session={self.session_id}, "
            f"intent={self.intent}, status={self.status.value}, "
            f"missing={len(self.missing_params)})>"
        )


class ConversationStateManager:
    """
    Manages multiple conversation states across sessions.
    Provides session storage and retrieval.
    """
    
    def __init__(self):
        """Initialize the conversation state manager."""
        self.states: Dict[str, ConversationState] = {}
        logger.info("✅ ConversationStateManager initialized")
    
    def create_session(self, session_id: str, user_id: str) -> ConversationState:
        """
        Create a new conversation session.
        
        Args:
            session_id: Unique session identifier
            user_id: User identifier
            
        Returns:
            New ConversationState instance
        """
        if session_id in self.states:
            logger.warning(f"⚠️ Session {session_id} already exists, returning existing")
            return self.states[session_id]
        
        state = ConversationState(session_id, user_id)
        self.states[session_id] = state
        logger.info(f"✅ Created new session: {session_id}")
        return state
    
    def get_session(self, session_id: str) -> Optional[ConversationState]:
        """
        Get an existing conversation session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            ConversationState if exists, None otherwise
        """
        return self.states.get(session_id)
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a conversation session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if deleted, False if not found
        """
        if session_id in self.states:
            del self.states[session_id]
            logger.info(f"🗑️ Deleted session: {session_id}")
            return True
        return False
    
    def get_active_sessions(self) -> List[ConversationState]:
        """
        Get all active (non-completed) sessions.
        
        Returns:
            List of active ConversationState instances
        """
        return [
            state for state in self.states.values()
            if state.status not in [ConversationStatus.COMPLETED, ConversationStatus.CANCELLED]
        ]
    
    def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """
        Clean up old completed/cancelled sessions.
        
        Args:
            max_age_hours: Maximum age in hours before cleanup
            
        Returns:
            Number of sessions cleaned up
        """
        from datetime import timedelta
        
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        sessions_to_delete = []
        
        for session_id, state in self.states.items():
            if (
                state.status in [ConversationStatus.COMPLETED, ConversationStatus.CANCELLED]
                and state.updated_at < cutoff_time
            ):
                sessions_to_delete.append(session_id)
        
        for session_id in sessions_to_delete:
            del self.states[session_id]
        
        if sessions_to_delete:
            logger.info(f"🧹 Cleaned up {len(sessions_to_delete)} old sessions")
        
        return len(sessions_to_delete)


# Global instance
conversation_state_manager = ConversationStateManager()

