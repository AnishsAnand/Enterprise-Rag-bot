"""
Orchestrator Agent - The main coordinator that routes tasks to specialized agents.
This is the entry point for all user requests in the multi-agent system.
"""

from typing import Any, Dict, List, Optional
from langchain.tools import Tool
import logging
import json

from app.agents.base_agent import BaseAgent
from app.agents.state.conversation_state import (
    ConversationState,
    ConversationStatus,
    conversation_state_manager
)

logger = logging.getLogger(__name__)


class OrchestratorAgent(BaseAgent):
    """
    Main orchestrator agent that coordinates between specialized agents.
    Routes user requests to appropriate agents and manages conversation flow.
    """
    
    def __init__(self):
        super().__init__(
            agent_name="OrchestratorAgent",
            agent_description=(
                "Main coordinator agent that routes user requests to specialized agents. "
                "Manages conversation flow, parameter collection, and execution orchestration."
            ),
            temperature=0.3
        )
        
        # References to specialized agents (will be set after initialization)
        self.intent_agent: Optional[BaseAgent] = None
        self.validation_agent: Optional[BaseAgent] = None
        self.execution_agent: Optional[BaseAgent] = None
        self.rag_agent: Optional[BaseAgent] = None
        
        # Setup agent with tools
        self.setup_agent()
    
    def set_specialized_agents(
        self,
        intent_agent: BaseAgent,
        validation_agent: BaseAgent,
        execution_agent: BaseAgent,
        rag_agent: BaseAgent
    ) -> None:
        """
        Set references to specialized agents.
        
        Args:
            intent_agent: Agent for intent detection
            validation_agent: Agent for parameter validation
            execution_agent: Agent for API execution
            rag_agent: Agent for RAG-based responses
        """
        self.intent_agent = intent_agent
        self.validation_agent = validation_agent
        self.execution_agent = execution_agent
        self.rag_agent = rag_agent
        logger.info("✅ Orchestrator agent configured with specialized agents")
    
    def get_system_prompt(self) -> str:
        """Return system prompt for orchestrator."""
        return """You are the Orchestrator Agent, the main coordinator in a multi-agent system for managing cloud resources.

Your responsibilities:
1. **Route user requests** to appropriate specialized agents:
   - IntentAgent: Detect user intent and extract parameters
   - ValidationAgent: Validate parameters and check permissions
   - ExecutionAgent: Execute CRUD operations on resources
   - RAGAgent: Answer questions using documentation

2. **Manage conversation flow**:
   - Track conversation state and collected parameters
   - Ask clarifying questions when needed
   - Guide users through multi-step operations

3. **Coordinate agents**:
   - Decide which agent should handle each step
   - Pass context between agents
   - Synthesize responses from multiple agents

4. **Handle operations** on these resources:
   - k8s_cluster: Kubernetes clusters (create, update, delete, list)
   - firewall: Firewall rules (create, read, update, delete, list)
   - load_balancer: Load balancers
   - database: Managed databases
   - storage: Storage volumes

**Decision making:**
- If user asks a question about documentation → RAGAgent
- If user wants to perform an action (create, delete, etc.) → IntentAgent → ValidationAgent → ExecutionAgent
- If unclear intent → Ask clarifying questions
- If missing parameters → Collect them conversationally

Be conversational, helpful, and guide users through complex operations step by step.
Always confirm destructive operations (delete, update) before executing."""
    
    def get_tools(self) -> List[Tool]:
        """Return tools for orchestrator agent."""
        return [
            Tool(
                name="route_to_intent_agent",
                func=self._route_to_intent,
                description=(
                    "Route to Intent Agent to detect user intent and extract parameters "
                    "for CRUD operations. Use when user wants to create, update, delete, "
                    "or list resources."
                )
            ),
            Tool(
                name="route_to_rag_agent",
                func=self._route_to_rag,
                description=(
                    "Route to RAG Agent to answer questions using documentation. "
                    "Use when user asks questions about how things work, documentation, "
                    "or needs information."
                )
            ),
            Tool(
                name="get_conversation_state",
                func=self._get_conversation_state,
                description=(
                    "Get current conversation state including collected parameters, "
                    "missing parameters, and conversation status. Use to check progress."
                )
            ),
            Tool(
                name="ask_clarifying_question",
                func=self._ask_clarifying_question,
                description=(
                    "Ask user a clarifying question to collect missing parameters "
                    "or resolve ambiguity. Provide the question as input."
                )
            )
        ]
    
    def _route_to_intent(self, user_input: str) -> str:
        """Route to intent agent."""
        try:
            if not self.intent_agent:
                return "Intent agent not configured"
            
            logger.info(f"🔀 Routing to IntentAgent: {user_input[:50]}...")
            # This will be called by the specialized orchestration logic
            return f"Routing to Intent Agent for: {user_input}"
        except Exception as e:
            logger.error(f"❌ Failed to route to intent agent: {str(e)}")
            return f"Error routing to intent agent: {str(e)}"
    
    def _route_to_rag(self, user_input: str) -> str:
        """Route to RAG agent."""
        try:
            if not self.rag_agent:
                return "RAG agent not configured"
            
            logger.info(f"🔀 Routing to RAGAgent: {user_input[:50]}...")
            return f"Routing to RAG Agent for: {user_input}"
        except Exception as e:
            logger.error(f"❌ Failed to route to RAG agent: {str(e)}")
            return f"Error routing to RAG agent: {str(e)}"
    
    def _get_conversation_state(self, session_id: str) -> str:
        """Get conversation state."""
        try:
            state = conversation_state_manager.get_session(session_id)
            if not state:
                return "No active conversation state found"
            
            return json.dumps({
                "intent": state.intent,
                "status": state.status.value,
                "collected_params": state.collected_params,
                "missing_params": list(state.missing_params),
                "invalid_params": state.invalid_params
            }, indent=2)
        except Exception as e:
            logger.error(f"❌ Failed to get conversation state: {str(e)}")
            return f"Error getting conversation state: {str(e)}"
    
    def _ask_clarifying_question(self, question: str) -> str:
        """Ask clarifying question."""
        logger.info(f"❓ Clarifying question: {question}")
        return f"CLARIFICATION_NEEDED: {question}"
    
    async def orchestrate(
        self,
        user_input: str,
        session_id: str,
        user_id: str,
        user_roles: List[str] = None
    ) -> Dict[str, Any]:
        """
        Main orchestration method that coordinates the entire flow.
        
        Args:
            user_input: User's message
            session_id: Conversation session ID
            user_id: User identifier
            user_roles: User's roles for permission checking
            
        Returns:
            Dict with orchestration result
        """
        try:
            logger.info(f"🎭 Orchestrating request for session {session_id}")
            
            # Get or create conversation state
            state = conversation_state_manager.get_session(session_id)
            if not state:
                state = conversation_state_manager.create_session(session_id, user_id)
            
            # Add user message to history
            state.add_message("user", user_input)
            
            # Determine routing based on conversation state and input
            routing_decision = await self._decide_routing(user_input, state, user_roles)
            
            # Execute based on routing decision
            result = await self._execute_routing(routing_decision, user_input, state, user_roles)
            
            # Add assistant response to history
            state.add_message("assistant", result.get("response", ""), {
                "routing": routing_decision["route"],
                "success": result.get("success", True)
            })
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Orchestration failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "response": f"I encountered an error while processing your request: {str(e)}"
            }
    
    async def _decide_routing(
        self,
        user_input: str,
        state: ConversationState,
        user_roles: List[str]
    ) -> Dict[str, Any]:
        """
        Decide which agent should handle the request.
        
        Args:
            user_input: User's message
            state: Current conversation state
            user_roles: User's roles
            
        Returns:
            Dict with routing decision
        """
        # Check if we're in the middle of parameter collection
        if state.status == ConversationStatus.COLLECTING_PARAMS:
            if state.missing_params:
                return {
                    "route": "validation",
                    "reason": "Collecting missing parameters"
                }
        
        # Check if ready to execute
        if state.status == ConversationStatus.READY_TO_EXECUTE:
            return {
                "route": "execution",
                "reason": "All parameters collected, ready to execute"
            }
        
        # Detect intent from user input
        user_input_lower = user_input.lower()
        
        # Action keywords
        action_keywords = [
            "create", "deploy", "provision", "setup", "configure",
            "delete", "remove", "destroy", "terminate",
            "update", "modify", "change", "edit",
            "list", "show", "get", "view", "display"
        ]
        
        # Resource keywords
        resource_keywords = [
            "cluster", "k8s", "kubernetes",
            "firewall", "security", "rule",
            "load balancer", "lb",
            "database", "db",
            "storage", "volume"
        ]
        
        # Check for action intent
        has_action = any(keyword in user_input_lower for keyword in action_keywords)
        has_resource = any(keyword in user_input_lower for keyword in resource_keywords)
        
        if has_action and has_resource:
            return {
                "route": "intent",
                "reason": "Detected action intent with resource"
            }
        
        # Question keywords
        question_keywords = ["what", "how", "why", "when", "where", "explain", "tell me", "?"]
        has_question = any(keyword in user_input_lower for keyword in question_keywords)
        
        if has_question:
            return {
                "route": "rag",
                "reason": "Detected question about documentation"
            }
        
        # Default to intent detection for ambiguous cases
        return {
            "route": "intent",
            "reason": "Ambiguous input, routing to intent detection"
        }
    
    async def _execute_routing(
        self,
        routing_decision: Dict[str, Any],
        user_input: str,
        state: ConversationState,
        user_roles: List[str]
    ) -> Dict[str, Any]:
        """
        Execute the routing decision.
        
        Args:
            routing_decision: Routing decision dict
            user_input: User's message
            state: Conversation state
            user_roles: User's roles
            
        Returns:
            Dict with execution result
        """
        route = routing_decision["route"]
        
        try:
            if route == "intent":
                # Route to intent agent
                state.handoff_to_agent("OrchestratorAgent", "IntentAgent", routing_decision["reason"])
                
                if self.intent_agent:
                    result = await self.intent_agent.execute(user_input, {
                        "session_id": state.session_id,
                        "user_id": state.user_id,
                        "conversation_state": state.to_dict()
                    })
                    
                    # Update state based on intent detection
                    if result.get("success") and result.get("intent_detected"):
                        intent_data = result.get("intent_data", {})
                        state.set_intent(
                            resource_type=intent_data.get("resource_type"),
                            operation=intent_data.get("operation"),
                            required_params=intent_data.get("required_params", []),
                            optional_params=intent_data.get("optional_params", [])
                        )
                        
                        # Add extracted parameters
                        extracted_params = intent_data.get("extracted_params", {})
                        if extracted_params:
                            state.add_parameters(extracted_params)
                    
                    return {
                        "success": True,
                        "response": result.get("output", ""),
                        "routing": route
                    }
                else:
                    return {
                        "success": False,
                        "response": "Intent agent not available",
                        "routing": route
                    }
            
            elif route == "validation":
                # Route to validation agent
                state.handoff_to_agent("OrchestratorAgent", "ValidationAgent", routing_decision["reason"])
                
                if self.validation_agent:
                    result = await self.validation_agent.execute(user_input, {
                        "session_id": state.session_id,
                        "conversation_state": state.to_dict()
                    })
                    
                    return {
                        "success": True,
                        "response": result.get("output", ""),
                        "routing": route
                    }
                else:
                    return {
                        "success": False,
                        "response": "Validation agent not available",
                        "routing": route
                    }
            
            elif route == "execution":
                # Route to execution agent
                state.handoff_to_agent("OrchestratorAgent", "ExecutionAgent", routing_decision["reason"])
                state.status = ConversationStatus.EXECUTING
                
                if self.execution_agent:
                    result = await self.execution_agent.execute("", {
                        "session_id": state.session_id,
                        "conversation_state": state.to_dict(),
                        "user_roles": user_roles or []
                    })
                    
                    # Update state with execution result
                    if result.get("success"):
                        state.set_execution_result(result.get("execution_result", {}))
                    else:
                        state.set_execution_result({
                            "success": False,
                            "error": result.get("error", "Execution failed")
                        })
                    
                    return {
                        "success": True,
                        "response": result.get("output", ""),
                        "routing": route,
                        "execution_result": result.get("execution_result")
                    }
                else:
                    return {
                        "success": False,
                        "response": "Execution agent not available",
                        "routing": route
                    }
            
            elif route == "rag":
                # Route to RAG agent
                state.handoff_to_agent("OrchestratorAgent", "RAGAgent", routing_decision["reason"])
                
                if self.rag_agent:
                    result = await self.rag_agent.execute(user_input, {
                        "session_id": state.session_id
                    })
                    
                    return {
                        "success": True,
                        "response": result.get("output", ""),
                        "routing": route
                    }
                else:
                    return {
                        "success": False,
                        "response": "RAG agent not available",
                        "routing": route
                    }
            
            else:
                return {
                    "success": False,
                    "response": f"Unknown routing: {route}",
                    "routing": route
                }
        
        except Exception as e:
            logger.error(f"❌ Routing execution failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "response": f"Error executing routing: {str(e)}",
                "routing": route
            }

