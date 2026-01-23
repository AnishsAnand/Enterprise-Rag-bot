"""
Orchestrator Agent - The main coordinator that routes tasks to specialized agents.
"""
from typing import Any, Dict, List, Optional
from langchain.tools import Tool
import logging
import json
from app.agents.base_agent import BaseAgent
from app.agents.state.conversation_state import (ConversationState,ConversationStatus,conversation_state_manager)

logger = logging.getLogger(__name__)

class OrchestratorAgent(BaseAgent):
    """
    Main orchestrator agent 
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
    def set_specialized_agents(self,intent_agent: BaseAgent,validation_agent: BaseAgent,execution_agent: BaseAgent,rag_agent: BaseAgent) -> None:
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
            )]
    
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
    
    async def orchestrate(self,user_input: str,session_id: str,user_id: str,user_roles: List[str] = None) -> Dict[str, Any]:
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
        import time
        from app.services.prometheus_metrics import metrics
        start_time = time.time()
        metrics.agent_active_sessions.labels(agent_name="OrchestratorAgent").inc()
        try:
            logger.info(f"🎭 Orchestrating request for session {session_id}")
            # Get or create conversation state
            state = conversation_state_manager.get_session(session_id)
            # Detect OpenWebUI metadata requests (title, tags, follow-ups generation)
            is_metadata_request = any([
                user_input.strip().startswith("### Task:"),
                "Generate a concise" in user_input and "title" in user_input.lower(),
                "Suggest 3-5 relevant follow-up" in user_input,
                "Generate 1-3 broad tags" in user_input])
            # If state exists but is COMPLETED/FAILED, delete it and start fresh
            if state and state.status in [ConversationStatus.COMPLETED, ConversationStatus.FAILED, ConversationStatus.CANCELLED]:
                if is_metadata_request:
                    logger.info(f"📋 Metadata request detected, preserving session state")
                else:
                    logger.info(f"🔄 Previous conversation {state.status.value}, starting fresh session")
                    conversation_state_manager.delete_session(session_id)
                    state = None
            if not state:
                state = conversation_state_manager.create_session(session_id, user_id)
                state.user_query = user_input
            # Add user message to history
            state.add_message("user", user_input)
            # Determine routing based on conversation state and input
            routing_decision = await self._decide_routing(user_input, state, user_roles)
            # Execute based on routing decision
            result = await self._execute_routing(routing_decision, user_input, state, user_roles)
            # Add assistant response to history
            state.add_message("assistant", result.get("response", ""), {
                "routing": routing_decision["route"],
                "success": result.get("success", True)})
            # Persist state after each interaction 
            conversation_state_manager.update_session(state)
            # Track success metrics
            duration = time.time() - start_time
            metrics.agent_execution_duration.labels(agent_name="OrchestratorAgent", operation="orchestrate").observe(duration)
            metrics.agent_sessions_total.labels(agent_name="OrchestratorAgent", status="success").inc()
            return result
        except Exception as e:
            logger.error(f"❌ Orchestration failed: {str(e)}")
            # Track error metrics
            duration = time.time() - start_time
            metrics.agent_execution_duration.labels(agent_name="OrchestratorAgent", operation="orchestrate").observe(duration)
            metrics.agent_sessions_total.labels(agent_name="OrchestratorAgent", status="error").inc()
            return {
                "success": False,
                "error": str(e),
                "response": f"I encountered an error while processing your request: {str(e)}"}
        finally:
            # Always decrement active sessions
            metrics.agent_active_sessions.labels(agent_name="OrchestratorAgent").dec()
    
    async def _decide_routing(self,user_input: str,state: ConversationState,user_roles: List[str]) -> Dict[str, Any]:
        """
        Decide which agent should handle the request using LLM-based routing.
        Args:
            user_input: User's message
            state: Current conversation state
            user_roles: User's roles
        Returns:
            Dict with routing decision
        """
        if state.status == ConversationStatus.COLLECTING_PARAMS:
            # For cluster creation workflow, always route to validation
            if state.operation == "create" and state.resource_type == "k8s_cluster":
                logger.info(f"🎯 Cluster creation in progress, routing to validation")
                return {
                    "route": "validation",
                    "reason": "Continuing cluster creation workflow"
                }
            if state.missing_params:
                return {
                    "route": "validation",
                    "reason": "Collecting missing parameters"
                }
        # Check if we're awaiting user selection 
        if state.status == ConversationStatus.AWAITING_SELECTION:
            logger.info(f"🔄 Session in AWAITING_SELECTION status, routing to validation to process user selection")
            return {
                "route": "validation",
                "reason": "Processing user selection from available options"
            }
        
        # Check if we're awaiting filter selection (BU/Environment/Zone)
        if state.status == ConversationStatus.AWAITING_FILTER_SELECTION:
            logger.info(f"🔄 Session in AWAITING_FILTER_SELECTION status, routing to filter_selection handler")
            return {
                "route": "filter_selection",
                "reason": "Processing user filter selection (BU/Environment/Zone)"
            }
        # Check if ready to execute
        if state.status == ConversationStatus.READY_TO_EXECUTE:
            return {
                "route": "execution",
                "reason": "All parameters collected, ready to execute"}
        
        # PRE-CHECK: Fast rule-based routing for obvious resource operations.This avoids LLM call latency for clear-cut cases
        query_lower = user_input.lower()
        resource_action_keywords = ["create", "delete", "remove", "update", "modify", "list", "show", "get", "deploy", "provision"]
        resource_type_keywords = ["cluster", "clusters", "firewall", "database", "vm", "volume", "endpoint", "jenkins", "kafka", "gitlab", "registry", "postgres"]
        has_action = any(kw in query_lower for kw in resource_action_keywords)
        has_resource = any(kw in query_lower for kw in resource_type_keywords)
        if has_action and has_resource:
            logger.info(f"⚡ Fast-path routing: detected resource operation '{user_input}' → intent")
            return {
                "route": "intent",
                "reason": "Fast-path routing: clear resource operation detected"}

        from app.services.ai_service import ai_service
        routing_prompt = f"""You are a routing specialist for a cloud resource management chatbot. Determine if the user's query is about:

A) **RESOURCE OPERATIONS**: Managing/viewing cloud resources (clusters, firewalls, databases, load balancers, storage, etc.)
   - Examples: "list clusters", "show clusters in delhi", "what are the clusters in mumbai?", "how many clusters?", "create a cluster", "delete firewall", "count clusters in bengaluru"
   
B) **DOCUMENTATION**: Questions about how to use the platform, concepts, procedures, troubleshooting, or explanations
   - Examples: "how do I create a cluster?", "what is kubernetes?", "explain load balancing", "why did my deployment fail?", "what are the requirements?"

User Query: "{user_input}"

Instructions:
1. If the query is asking to VIEW, COUNT, LIST, CREATE, UPDATE, or DELETE actual resources → return "RESOURCE_OPERATIONS"
2. If the query is asking HOW TO do something, WHY something works, or WHAT a concept means → return "DOCUMENTATION"
3. "What are the clusters?" = RESOURCE_OPERATIONS (listing actual clusters)
4. "What is a cluster?" = DOCUMENTATION (explaining the concept)
5. "How many clusters in delhi?" = RESOURCE_OPERATIONS (counting actual clusters)
6. "How do I create a cluster?" = DOCUMENTATION (explaining the process)

Respond with ONLY ONE of these:
- ROUTE: RESOURCE_OPERATIONS
- ROUTE: DOCUMENTATION"""

        try:
            logger.debug(f"🔍 Routing prompt for query: {user_input}")
            llm_response = await ai_service._call_chat_with_retries(
                prompt=routing_prompt,
                max_tokens=100,  
                temperature=0.1,  
                timeout=15  )
            
            logger.info(f"🤖 LLM routing decision (length={len(llm_response)} chars): {llm_response}")
            # Check for empty or too short response
            if not llm_response or len(llm_response.strip()) < 5:
                logger.error(f"❌ LLM returned empty/very short response ('{llm_response}') for routing")
                # Use rule-based fallback
                query_lower = user_input.lower()
                resource_action_keywords = ["create", "delete", "remove", "update", "modify", "list", "show", "get"]
                resource_type_keywords = ["cluster", "firewall", "database", "vm", "volume", "endpoint", "jenkins", "kafka"]
                has_action = any(kw in query_lower for kw in resource_action_keywords)
                has_resource = any(kw in query_lower for kw in resource_type_keywords)
                if has_action and has_resource:
                    logger.info(f"🎯 Rule-based fallback: detected resource operation in '{user_input}'")
                    return {
                        "route": "intent",
                        "reason": "Rule-based routing: resource operation detected (LLM response empty)"}
                # Documentation patterns
                doc_patterns = ["how to", "how do", "how can", "what is", "explain", "why", 
                               "tutorial", "guide", "documentation", "help me", "tell me about"]
                if any(pattern in query_lower for pattern in doc_patterns):
                    logger.info(f"🎯 Rule-based fallback: detected documentation pattern in '{user_input}'")
                    return {
                        "route": "rag",
                        "reason": "Rule-based routing: documentation question detected (LLM response empty)"}
                else:
                    logger.info(f"🎯 Rule-based fallback: assuming resource operation for '{user_input}'")
                    return {
                        "route": "intent",
                        "reason": "Rule-based routing: resource operation assumed (LLM response empty)" }
            if "RESOURCE_OPERATIONS" in llm_response.upper():
                logger.info(f"✅ LLM routing: RESOURCE_OPERATIONS → IntentAgent")
                return {
                    "route": "intent",
                    "reason": "LLM detected resource operation intent"
                }
            elif "DOCUMENTATION" in llm_response.upper():
                logger.info(f"✅ LLM routing: DOCUMENTATION → RAGAgent")
                return {
                    "route": "rag",
                    "reason": "LLM detected documentation question"
                }
            else:
                logger.warning(f"⚠️ LLM routing unclear: '{llm_response}', defaulting to intent")
                return {
                    "route": "intent",
                    "reason": "Ambiguous routing, defaulting to intent detection"
                }
        except Exception as e:
            logger.error(f"❌ LLM routing failed with exception: {e}, using rule-based fallback")
            # Use rule-based fallback on exception
            query_lower = user_input.lower()
            doc_patterns = ["how to", "how do", "how can", "what is", "what are", "explain", "why", 
                           "tutorial", "guide", "documentation", "help me", "tell me about"]
            if any(pattern in query_lower for pattern in doc_patterns):
                logger.info(f"🎯 Rule-based fallback (exception): detected documentation pattern")
                return {
                    "route": "rag",
                    "reason": "Rule-based routing: documentation question detected (LLM error)"
                }
            else:
                logger.info(f"🎯 Rule-based fallback (exception): assuming resource operation")
                return {
                    "route": "intent",
                    "reason": "Rule-based routing: resource operation assumed (LLM error)"
                }
    
    async def _execute_routing(self,routing_decision: Dict[str, Any],user_input: str,state: ConversationState,user_roles: List[str]) -> Dict[str, Any]:
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
                        "conversation_state": state.to_dict()})
                    # STEP 1 : Update state based on intent detection
                    if result.get("success") and result.get("intent_detected"):
                        intent_data = result.get("intent_data", {})
                        state.set_intent(
                            resource_type=intent_data.get("resource_type"),
                            operation=intent_data.get("operation"),
                            required_params=intent_data.get("required_params", []),
                            optional_params=intent_data.get("optional_params", []))
                        # Add extracted parameters
                        extracted_params = intent_data.get("extracted_params", {})
                        if extracted_params:
                            state.add_parameters(extracted_params)
                        logger.info(f"📋 State after intent: required={state.required_params}, collected={list(state.collected_params.keys())}, missing={state.missing_params}")
                        # STEP 2: Check if we need more parameters OR if ready to execute
                        needs_validation = bool(state.missing_params)
                        if state.operation == "create" and state.resource_type == "k8s_cluster":
                            # Cluster creation uses a multi-step workflow, always route to validation
                            needs_validation = True
                            logger.info("🎯 Cluster creation detected - routing to ValidationAgent for workflow")
                        if needs_validation:
                            logger.info(f"🔄 Missing params detected: {state.missing_params}, routing to ValidationAgent")
                            state.status = ConversationStatus.COLLECTING_PARAMS
                            state.handoff_to_agent("IntentAgent", "ValidationAgent", "Need to collect missing parameters")
                            # Immediately route to validation agent
                            if self.validation_agent:
                                validation_result = await self.validation_agent.execute(user_input, {
                                    "session_id": state.session_id,
                                    "conversation_state": state.to_dict()})
                                # Check if validation made us ready to execute
                                if validation_result.get("ready_to_execute") and validation_result.get("success"):
                                    logger.info("🚀 ValidationAgent says ready - routing to ExecutionAgent")
                                    state.handoff_to_agent("ValidationAgent", "ExecutionAgent", "All parameters collected")
                                    state.status = ConversationStatus.EXECUTING
                                    if self.execution_agent:
                                        exec_result = await self.execution_agent.execute("", {
                                            "session_id": state.session_id,
                                            "conversation_state": state.to_dict(),
                                            "user_roles": user_roles or []
                                        })
                                        
                                        # Check if awaiting filter selection - don't mark as completed
                                        if state.status != ConversationStatus.AWAITING_FILTER_SELECTION:
                                            if exec_result.get("success"):
                                                state.set_execution_result(exec_result.get("execution_result", {}))
                                        return {
                                            "success": True,
                                            "response": exec_result.get("output", ""),
                                            "routing": "execution",
                                            "execution_result": exec_result.get("execution_result"),
                                            "metadata": {
                                                "collected_params": state.collected_params,
                                                "resource_type": state.resource_type,
                                                "operation": state.operation}}
                              
                                return {
                                    "success": True,
                                    "response": validation_result.get("output", ""),
                                    "routing": "validation",
                                    "intent_data": intent_data,
                                    "metadata": {
                                        "collected_params": state.collected_params,
                                        "missing_params": list(state.missing_params) }}
                        else:
                            # No missing params - proceed directly to execution!
                            logger.info(f"✅ All params collected for {state.operation} {state.resource_type}, executing immediately")
                            state.status = ConversationStatus.EXECUTING
                            state.handoff_to_agent("IntentAgent", "ExecutionAgent", "No parameters needed, executing immediately")
                            if self.execution_agent:
                                exec_result = await self.execution_agent.execute("", {
                                    "session_id": state.session_id,
                                    "conversation_state": state.to_dict(),
                                    "user_roles": user_roles or []
                                })
                                
                                # Check if awaiting filter selection - don't mark as completed
                                if state.status != ConversationStatus.AWAITING_FILTER_SELECTION:
                                    if exec_result.get("success"):
                                        state.set_execution_result(exec_result.get("execution_result", {}))
                                return {
                                    "success": True,
                                    "response": exec_result.get("output", ""),
                                    "routing": "execution",
                                    "execution_result": exec_result.get("execution_result"),
                                    "metadata": {
                                        "collected_params": state.collected_params,
                                        "resource_type": state.resource_type,
                                        "operation": state.operation}}
                    return {
                        "success": True,
                        "response": result.get("output", ""),
                        "routing": route}
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
                    # CHECK: If user cancelled the workflow
                    if result.get("cancelled"):
                        logger.info("🚫 User cancelled the workflow")
                        return {
                            "success": True,
                            "response": result.get("output", "Workflow cancelled."),
                            "routing": "cancelled",
                            "cancelled": True,
                            "metadata": {} }
                    # CHECK: If validation says "ready to execute", route to execution NOW!
                    if result.get("ready_to_execute") and result.get("success"):
                        logger.info("🚀 ValidationAgent says ready - routing to ExecutionAgent")
                        # Update state to executing
                        state.handoff_to_agent("ValidationAgent", "ExecutionAgent", "All parameters collected")
                        state.status = ConversationStatus.EXECUTING
                        # Execute immediately
                        if self.execution_agent:
                            exec_result = await self.execution_agent.execute("", {
                                "session_id": state.session_id,
                                "conversation_state": state.to_dict(),
                                "user_roles": user_roles or []
                            })
                            
                            # Check if execution is awaiting filter selection (BU/Env/Zone)
                            # In this case, DON'T mark as completed - state is already set to AWAITING_FILTER_SELECTION
                            exec_metadata = exec_result.get("metadata", {})
                            is_awaiting_filter = exec_metadata.get("awaiting_filter_selection") or \
                                                 state.status == ConversationStatus.AWAITING_FILTER_SELECTION
                            
                            if is_awaiting_filter:
                                logger.info("🔄 Execution returned filter options - keeping AWAITING_FILTER_SELECTION status")
                                # Don't call set_execution_result - it would mark as COMPLETED
                                # State is already updated by ExecutionAgent
                            elif exec_result.get("success"):
                                state.set_execution_result(exec_result.get("execution_result", {}))
                            return {
                                "success": True,
                                "response": exec_result.get("output", ""),
                                "routing": "execution",
                                "execution_result": exec_result.get("execution_result"),
                                "metadata": {
                                    "collected_params": state.collected_params,
                                    "resource_type": state.resource_type,
                                    "operation": state.operation}}
                        else:
                            return {
                                "success": False,
                                "response": "Execution agent not available",
                                "routing": "execution" }

                    # Otherwise, return validation response (asking for more info)
                    return {
                        "success": True,
                        "response": result.get("output", ""),
                        "routing": route,
                        "metadata": {
                            "missing_params": result.get("missing_params", []),
                            "ready_to_execute": result.get("ready_to_execute", False)} }
                else:
                    return {
                        "success": False,
                        "response": "Validation agent not available",
                        "routing": route }
            
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
                    
                    # Check if awaiting filter selection - don't mark as completed
                    if state.status != ConversationStatus.AWAITING_FILTER_SELECTION:
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
                        "execution_result": result.get("execution_result")}
                else:
                    return {
                        "success": False,
                        "response": "Execution agent not available",
                        "routing": route}
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
                        "routing": route}
                else:
                    return {
                        "success": False,
                        "response": "RAG agent not available",
                        "routing": route
                    }
            
            elif route == "filter_selection":
                # Process user's filter selection (BU/Environment/Zone)
                return await self._handle_filter_selection(user_input, state, user_roles)
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

    async def _handle_filter_selection(
        self,
        user_input: str,
        state: ConversationState,
        user_roles: List[str]
    ) -> Dict[str, Any]:
        """
        Handle user's selection from BU/Environment/Zone filter options.
        
        This is called when state.status == AWAITING_FILTER_SELECTION.
        User input could be:
        - A number: "9" → select option at index 8
        - A name: "qatest332" → match by name
        - Multiple: "1, 3" or "1 and 3"
        
        Args:
            user_input: User's response
            state: Conversation state with pending_filter_options
            user_roles: User's roles
            
        Returns:
            Dict with result (either proceed to execution or ask again)
        """
        try:
            logger.info(f"🔍 Processing filter selection: '{user_input}'")
            
            # Get the pending filter options from state
            options = state.pending_filter_options
            filter_type = state.pending_filter_type
            
            if not options or not filter_type:
                logger.error("❌ No pending filter options found in state")
                # Clear the state and start fresh
                state.status = ConversationStatus.INITIATED
                state.pending_filter_options = None
                state.pending_filter_type = None
                conversation_state_manager.update_session(state)
                return {
                    "success": False,
                    "response": "I lost track of the filter options. Could you please ask again?",
                    "routing": "filter_selection"
                }
            
            if filter_type and filter_type.startswith("report_"):
                selection_result = self._parse_generic_selection(user_input, options)
            else:
                # Use K8sClusterAgent to parse the selection
                from app.agents.resource_agents.k8s_cluster_agent import K8sClusterAgent
                cluster_agent = K8sClusterAgent()
                
                selection_result = cluster_agent.parse_filter_selection(
                    user_input=user_input,
                    options=options,
                    filter_type=filter_type
                )
            
            if not selection_result or not selection_result.get("matched"):
                # Could not parse selection - ask user again
                logger.warning(f"⚠️ Could not match selection '{user_input}' to any option")
                
                # Build a helpful message
                response = f"I couldn't match '{user_input}' to any option.\n\n"
                response += "Please reply with:\n"
                response += "- A **number** from the table (e.g., `3`)\n"
                response += "- A **name** from the list (e.g., `qatest332`)\n"
                response += "- **Multiple selections** separated by commas (e.g., `1, 3, 5`)\n"
                
                return {
                    "success": True,
                    "response": response,
                    "routing": "filter_selection"
                }
            
            # Successfully parsed selection!
            if filter_type and filter_type.startswith("report_"):
                selected_names = selection_result.get("selected_names", [])
                selected_options = selection_result.get("selected_options", [])
                
                logger.info(f"✅ Report filter selection: {selected_names} ({filter_type})")
                
                # Clear filter selection state
                state.pending_filter_options = None
                state.pending_filter_type = None
                
                # Apply selected filter to report params
                if filter_type == "report_catalog" and selected_options:
                    selected_option = selected_options[0]
                    report_type = selected_option.get("value") or selected_option.get("name")
                    if report_type:
                        state.add_parameter("report_type", report_type, is_valid=True)
                elif filter_type == "report_cluster" and selected_names:
                    state.add_parameter("clusterName", ",".join(selected_names), is_valid=True)
                elif filter_type == "report_datacenter" and selected_names:
                    state.add_parameter("datacenter", ",".join(selected_names), is_valid=True)
                elif filter_type == "report_dates":
                    selected = selected_options[0] if selected_options else {}
                    if selected.get("name") == "Custom Date Range":
                        state.pending_filter_options = []
                        state.pending_filter_type = "report_custom_dates"
                        state.status = ConversationStatus.AWAITING_FILTER_SELECTION
                        conversation_state_manager.update_session(state)
                        return {
                            "success": True,
                            "response": "Please provide a custom date range in the format `YYYY-MM-DD to YYYY-MM-DD`.",
                            "routing": "filter_selection"
                        }
                    if selected.get("startDate") and selected.get("endDate"):
                        state.add_parameter("startDate", selected["startDate"], is_valid=True)
                        state.add_parameter("endDate", selected["endDate"], is_valid=True)
                elif filter_type == "report_custom_dates":
                    dates = self._extract_date_range(user_input)
                    if not dates:
                        return {
                            "success": True,
                            "response": "Please provide dates in the format `YYYY-MM-DD to YYYY-MM-DD`.",
                            "routing": "filter_selection"
                        }
                    state.add_parameter("startDate", dates["startDate"], is_valid=True)
                    state.add_parameter("endDate", dates["endDate"], is_valid=True)
            else:
                filter_ids = selection_result["filter_ids"]
                endpoint_ids = selection_result["endpoint_ids"]
                endpoint_names = selection_result["endpoint_names"]
                selected_names = selection_result["selected_names"]
                filter_key = selection_result["filter_key"]  # "businessUnits", "environments", or "zones"
                
                logger.info(f"✅ Filter selection: {selected_names} (endpoints: {endpoint_ids}, {filter_key}: {filter_ids})")
                
                # Clear filter selection state
                state.pending_filter_options = None
                state.pending_filter_type = None
                
                # Add the collected parameters to state
                state.add_parameter("endpoints", endpoint_ids, is_valid=True)
                state.add_parameter("endpoint_names", endpoint_names, is_valid=True)
                state.add_parameter(filter_key, filter_ids, is_valid=True)
            
            # Update status to executing
            state.status = ConversationStatus.EXECUTING
            state.handoff_to_agent("OrchestratorAgent", "ExecutionAgent", f"Filter selection complete: {selected_names}")
            
            # Persist state
            conversation_state_manager.update_session(state)
            
            # Execute the cluster listing with the filters
            if self.execution_agent:
                exec_result = await self.execution_agent.execute("", {
                    "session_id": state.session_id,
                    "conversation_state": state.to_dict(),
                    "user_roles": user_roles or []
                })
                
                # Only mark as completed if not awaiting another filter selection
                if state.status != ConversationStatus.AWAITING_FILTER_SELECTION:
                    if exec_result.get("success"):
                        state.set_execution_result(exec_result.get("execution_result", {}))
                
                return {
                    "success": True,
                    "response": exec_result.get("output", ""),
                    "routing": "execution",
                    "execution_result": exec_result.get("execution_result"),
                    "metadata": {
                        "collected_params": state.collected_params,
                        "resource_type": state.resource_type,
                        "operation": state.operation,
                        "filter_applied": {
                            "type": filter_type,
                            "selected": selected_names
                        }
                    }
                }
            else:
                return {
                    "success": False,
                    "response": "Execution agent not available",
                    "routing": "filter_selection"
                }
            
        except Exception as e:
            logger.error(f"❌ Filter selection handling failed: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "response": f"Error processing your selection: {str(e)}",
                "routing": "filter_selection"
            }

    def _parse_generic_selection(
        self,
        user_input: str,
        options: Optional[List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        if not options:
            return {"matched": False}
        user_input = (user_input or "").strip()
        selected = []
        import re
        parts = re.split(r'[,;]|\band\b', user_input)
        parts = [p.strip() for p in parts if p.strip()]
        for part in parts:
            try:
                idx = int(part) - 1
                if 0 <= idx < len(options):
                    selected.append(options[idx])
                    continue
            except ValueError:
                pass
            part_lower = part.lower()
            for opt in options:
                opt_name = (opt.get("name") or "").lower()
                if part_lower in opt_name or opt_name in part_lower:
                    if opt not in selected:
                        selected.append(opt)
                    break
        if not selected:
            return {"matched": False}
        return {
            "matched": True,
            "selected_names": [opt.get("name") for opt in selected if opt.get("name")],
            "selected_options": selected
        }

    def _extract_date_range(self, user_input: str) -> Optional[Dict[str, str]]:
        import re
        matches = re.findall(r"\d{4}-\d{2}-\d{2}", user_input or "")
        if len(matches) >= 2:
            return {"startDate": matches[0], "endDate": matches[1]}
        return None
