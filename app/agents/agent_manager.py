"""
Agent Manager - Central manager for all agents in the multi-agent system.
Initializes agents, manages their lifecycle, and provides a unified interface.
"""
from typing import Any, Dict, List, Optional
import logging
from datetime import datetime
from app.agents.orchestrator_agent import OrchestratorAgent
from app.agents.intent_agent import IntentAgent
from app.agents.validation_agent import ValidationAgent
from app.agents.execution_agent import ExecutionAgent
from app.agents.rag_agent import RAGAgent
from app.agents.state.conversation_state import conversation_state_manager

logger = logging.getLogger(__name__)

class AgentManager:
    """
    Central manager for the multi-agent system.
    Agent Flow:
        User → Orchestrator → IntentAgent → ValidationAgent → ExecutionAgent → ResourceAgents → API
    """
    def __init__(self):
        """
        Initialize agent manager.
        Note: RAGAgent uses the existing widget_query system, so no services needed.
        """
        self.orchestrator: Optional[OrchestratorAgent] = None
        self.intent_agent: Optional[IntentAgent] = None
        self.validation_agent: Optional[ValidationAgent] = None
        self.execution_agent: Optional[ExecutionAgent] = None
        self.rag_agent: Optional[RAGAgent] = None
        # Manager state
        self.initialized = False
        self.initialization_time: Optional[datetime] = None
        self.total_requests = 0
        logger.info("✅ AgentManager created")
    
    def initialize(self) -> None:
        """Initialize all agents and wire them together."""
        try:
            logger.info("🚀 Initializing multi-agent system...")
            
            self.intent_agent = IntentAgent()
            self.validation_agent = ValidationAgent()
            self.execution_agent = ExecutionAgent()
            self.rag_agent = RAGAgent()  
            self.orchestrator = OrchestratorAgent()
            self.orchestrator.set_specialized_agents(
                intent_agent=self.intent_agent,
                validation_agent=self.validation_agent,
                execution_agent=self.execution_agent,
                rag_agent=self.rag_agent)
            self.initialized = True
            self.initialization_time = datetime.utcnow()
            logger.info("✅ Multi-agent system initialized successfully")
            logger.info(f"   - OrchestratorAgent: {self.orchestrator.agent_name}")
            logger.info(f"   - IntentAgent: {self.intent_agent.agent_name}")
            logger.info(f"   - ValidationAgent: {self.validation_agent.agent_name}")
            logger.info(f"   - ExecutionAgent: {self.execution_agent.agent_name}")
            logger.info(f"   - RAGAgent: {self.rag_agent.agent_name}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize agent manager: {str(e)}")
            raise
    
    async def process_request(self,user_input: str,session_id: str,user_id: str,user_roles: List[str] = None) -> Dict[str, Any]:
        """
        Process a user request through the multi-agent system.
        Args:
            user_input: User's message
            session_id: Conversation session ID
            user_id: User identifier
            user_roles: User's roles for permission checking   
        Returns:
            Dict with response and metadata
        """
        if not self.initialized:
            self.initialize()
        try:
            self.total_requests += 1
            start_time = datetime.utcnow()
            logger.info(
                f"📥 Processing request #{self.total_requests} | "
                f"Session: {session_id} | User: {user_id}")
            # Process through orchestrator
            result = await self.orchestrator.orchestrate(
                user_input=user_input,
                session_id=session_id,
                user_id=user_id,
                user_roles=user_roles or [])
            # Add metadata
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            result["metadata"] = {
                "request_number": self.total_requests,
                "duration_seconds": duration,
                "timestamp": end_time.isoformat(),
                "session_id": session_id,
                "user_id": user_id}
            logger.info(
                f"✅ Request #{self.total_requests} completed in {duration:.2f}s | "
                f"Success: {result.get('success', False)}")
            return result
        except Exception as e:
            logger.error(f"❌ Request processing failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "response": f"I encountered an error while processing your request: {str(e)}",
                "metadata": {
                    "request_number": self.total_requests,
                    "timestamp": datetime.utcnow().isoformat(),
                    "session_id": session_id,
                    "user_id": user_id}}
    
    async def get_conversation_status(self, session_id: str) -> Dict[str, Any]:
        """
        Get the status of a conversation session.
        Args:
            session_id: Session identifier
        Returns:
            Dict with conversation status
        """
        state = conversation_state_manager.get_session(session_id)
        if not state:
            return {
                "found": False,
                "message": "No active conversation found for this session" }
        return {
            "found": True,
            "state": state.to_dict()}
    
    async def reset_conversation(self, session_id: str) -> Dict[str, Any]:
        """
        Reset a conversation session.
        Args:
            session_id: Session identifier
        Returns:
            Dict with reset result
        """
        deleted = conversation_state_manager.delete_session(session_id)
        return {
            "success": deleted,
            "message": "Conversation reset successfully" if deleted else "No conversation found"}
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the agent manager.
        Returns:
            Dict with statistics
        """
        active_sessions = conversation_state_manager.get_active_sessions()
        
        return {
            "initialized": self.initialized,
            "initialization_time": self.initialization_time.isoformat() if self.initialization_time else None,
            "total_requests": self.total_requests,
            "active_sessions": len(active_sessions),
            "agents": {
                "orchestrator": self.orchestrator.agent_name if self.orchestrator else None,
                "intent": self.intent_agent.agent_name if self.intent_agent else None,
                "validation": self.validation_agent.agent_name if self.validation_agent else None,
                "execution": self.execution_agent.agent_name if self.execution_agent else None,
                "rag": self.rag_agent.agent_name if self.rag_agent else None}}
    
    def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """
        Clean up old conversation sessions.
        """
        return conversation_state_manager.cleanup_old_sessions(max_age_hours)
    
    def __repr__(self) -> str:
        return (
            f"<AgentManager(initialized={self.initialized}, "
            f"requests={self.total_requests}, "
            f"active_sessions={len(conversation_state_manager.get_active_sessions())})>")
# Global agent manager instance
agent_manager: Optional[AgentManager] = None

def get_agent_manager(vector_service=None, ai_service=None) -> AgentManager:
    """
    Get or create the global agent manager instance.
    Args:
        vector_service: Vector database service (not used, kept for compatibility)
        ai_service: AI service (not used, kept for compatibility)
    Returns:
        AgentManager instance
    """
    global agent_manager
    if agent_manager is None:
        agent_manager = AgentManager()
        agent_manager.initialize()
    return agent_manager