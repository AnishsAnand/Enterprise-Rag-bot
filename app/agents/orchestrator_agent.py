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
        logger.info(f"Clarifying question: {question}")
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
        import time
        from app.services.prometheus_metrics import metrics
        
        start_time = time.time()
        metrics.agent_active_sessions.labels(agent_name="OrchestratorAgent").inc()
        
        try:
            logger.info(f"Orchestrating request for session {session_id}")
            
            # Get or create conversation state
            state = conversation_state_manager.get_session(session_id)
            
            # Detect OpenWebUI metadata requests (title, tags, follow-ups generation)
            # These should NOT trigger session resets
            is_metadata_request = any([
                user_input.strip().startswith("### Task:"),
                "Generate a concise" in user_input and "title" in user_input.lower(),
                "Suggest 3-5 relevant follow-up" in user_input,
                "Generate 1-3 broad tags" in user_input
            ])
            
            # If state exists but is COMPLETED/FAILED, delete it and start fresh
            # BUT skip this for metadata requests to preserve context
            if state and state.status in [ConversationStatus.COMPLETED, ConversationStatus.FAILED, ConversationStatus.CANCELLED]:
                if is_metadata_request:
                    # Don't reset session for metadata requests, just continue
                    logger.info(f"Metadata request detected, preserving session state")
                else:
                    # Regular completed conversation - start fresh for new user query
                    logger.info(f"Previous conversation {state.status.value}, starting fresh session")
                    conversation_state_manager.delete_session(session_id)
                    state = None
            
            if not state:
                state = conversation_state_manager.create_session(session_id, user_id)
                # Store original query for context (e.g., extracting location from "clusters in delhi")
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
                "success": result.get("success", True)
            })
            
            # Persist state after each interaction (for scalability across restarts/instances)
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
                "response": f"I encountered an error while processing your request: {str(e)}"
            }
        finally:
            # Always decrement active sessions
            metrics.agent_active_sessions.labels(agent_name="OrchestratorAgent").dec()
    
    async def _decide_routing(
        self,
        user_input: str,
        state: ConversationState,
        user_roles: List[str]
    ) -> Dict[str, Any]:
        """
        Decide which agent should handle the request using LLM-based routing.
        
        Args:
            user_input: User's message
            state: Current conversation state
            user_roles: User's roles
            
        Returns:
            Dict with routing decision
        """
        # Check if we're in the middle of parameter collection
        if state.status == ConversationStatus.COLLECTING_PARAMS:
            # For cluster creation workflow, always route to validation (it has internal step tracking)
            if state.operation == "create" and state.resource_type == "k8s_cluster":
                logger.info(f"Cluster creation in progress, routing to validation")
                return {
                    "route": "validation",
                    "reason": "Continuing cluster creation workflow"
                }
            if state.missing_params:
                return {
                    "route": "validation",
                    "reason": "Collecting missing parameters"
                }
        
        # Check if we're awaiting user selection (e.g., which endpoint to use)
        if state.status == ConversationStatus.AWAITING_SELECTION:
            logger.info(f"Session in AWAITING_SELECTION status, routing to validation to process user selection")
            return {
                "route": "validation",
                "reason": "Processing user selection from available options"
            }
        
        # Check if ready to execute
        if state.status == ConversationStatus.READY_TO_EXECUTE:
            return {
                "route": "execution",
                "reason": "All parameters collected, ready to execute"
            }
        
        # PRE-CHECK: Fast rule-based routing for obvious resource operations
        # This avoids LLM call latency for clear-cut cases
        query_lower = user_input.lower()
        resource_action_keywords = ["create", "delete", "remove", "update", "modify", "list", "show", "get", "deploy", "provision"]
        resource_type_keywords = ["cluster", "clusters", "firewall", "database", "vm", "volume", "endpoint", "jenkins", "kafka", "gitlab", "registry", "postgres"]
        
        has_action = any(kw in query_lower for kw in resource_action_keywords)
        has_resource = any(kw in query_lower for kw in resource_type_keywords)
        
        # If it's clearly a resource operation, skip LLM routing
        if has_action and has_resource:
            logger.info(f"Fast-path routing: detected resource operation '{user_input}' → intent")
            return {
                "route": "intent",
                "reason": "Fast-path routing: clear resource operation detected"
            }
        
        # Use LLM to intelligently route the query
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
            logger.debug(f"Routing prompt for query: {user_input}")
            llm_response = await ai_service._call_chat_with_retries(
                prompt=routing_prompt,
                max_tokens=100,  # Increased from 20 to 100 for more reliable responses
                temperature=0.1,  # Slightly increased from 0.0 to avoid potential model issues
                timeout=15  # Add explicit timeout
            )
            
            logger.info(f"LLM routing decision (length={len(llm_response)} chars): {llm_response}")
            
            # Check for empty or too short response
            if not llm_response or len(llm_response.strip()) < 5:
                logger.error(f"❌ LLM returned empty/very short response ('{llm_response}') for routing")
                # Use rule-based fallback
                query_lower = user_input.lower()
                
                # RESOURCE OPERATION keywords - these should ALWAYS go to intent
                resource_action_keywords = ["create", "delete", "remove", "update", "modify", "list", "show", "get"]
                resource_type_keywords = ["cluster", "firewall", "database", "vm", "volume", "endpoint", "jenkins", "kafka"]
                
                has_action = any(kw in query_lower for kw in resource_action_keywords)
                has_resource = any(kw in query_lower for kw in resource_type_keywords)
                
                if has_action and has_resource:
                    logger.info(f"Rule-based fallback: detected resource operation in '{user_input}'")
                    return {
                        "route": "intent",
                        "reason": "Rule-based routing: resource operation detected (LLM response empty)"
                    }
                
                # Documentation patterns
                doc_patterns = ["how to", "how do", "how can", "what is", "explain", "why", 
                               "tutorial", "guide", "documentation", "help me", "tell me about"]
                if any(pattern in query_lower for pattern in doc_patterns):
                    logger.info(f"Rule-based fallback: detected documentation pattern in '{user_input}'")
                    return {
                        "route": "rag",
                        "reason": "Rule-based routing: documentation question detected (LLM response empty)"
                    }
                else:
                    logger.info(f"Rule-based fallback: assuming resource operation for '{user_input}'")
                    return {
                        "route": "intent",
                        "reason": "Rule-based routing: resource operation assumed (LLM response empty)"
                    }
            
            # Parse LLM response
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
                # Fallback to intent if unclear
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
                logger.info(f"Rule-based fallback (exception): detected documentation pattern")
                return {
                    "route": "rag",
                    "reason": "Rule-based routing: documentation question detected (LLM error)"
                }
            else:
                logger.info(f"Rule-based fallback (exception): assuming resource operation")
                return {
                    "route": "intent",
                    "reason": "Rule-based routing: resource operation assumed (LLM error)"
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
                        
                        logger.info(f"State after intent: required={state.required_params}, collected={list(state.collected_params.keys())}, missing={state.missing_params}")
                        
                        # STEP 2: Check if we need more parameters OR if ready to execute
                        # SPECIAL CASE: Cluster creation has its own workflow and always needs validation
                        needs_validation = bool(state.missing_params)
                        if state.operation == "create" and state.resource_type == "k8s_cluster":
                            # Cluster creation uses a multi-step workflow, always route to validation
                            needs_validation = True
                            logger.info("Cluster creation detected - routing to ValidationAgent for workflow")
                        
                        if needs_validation:
                            logger.info(f"Missing params detected: {state.missing_params}, routing to ValidationAgent")
                            state.status = ConversationStatus.COLLECTING_PARAMS
                            state.handoff_to_agent("IntentAgent", "ValidationAgent", "Need to collect missing parameters")
                            
                            # Immediately route to validation agent
                            if self.validation_agent:
                                validation_result = await self.validation_agent.execute(user_input, {
                                    "session_id": state.session_id,
                                    "conversation_state": state.to_dict()
                                })
                                
                                # Check if validation made us ready to execute
                                if validation_result.get("ready_to_execute") and validation_result.get("success"):
                                    logger.info("ValidationAgent says ready - routing to ExecutionAgent")
                                    
                                    state.handoff_to_agent("ValidationAgent", "ExecutionAgent", "All parameters collected")
                                    state.status = ConversationStatus.EXECUTING
                                    
                                    if self.execution_agent:
                                        exec_result = await self.execution_agent.execute("", {
                                            "session_id": state.session_id,
                                            "conversation_state": state.to_dict(),
                                            "user_roles": user_roles or []
                                        })
                                        
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
                                                "operation": state.operation
                                            }
                                        }
                                
                                return {
                                    "success": True,
                                    "response": validation_result.get("output", ""),
                                    "routing": "validation",
                                    "intent_data": intent_data,
                                    "metadata": {
                                        "collected_params": state.collected_params,
                                        "missing_params": list(state.missing_params)
                                    }
                                }
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
                                        "operation": state.operation
                                    }
                                }
                    
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
                    
                    # CHECK: If user cancelled the workflow
                    if result.get("cancelled"):
                        logger.info("User cancelled the workflow")
                        return {
                            "success": True,
                            "response": result.get("output", "Workflow cancelled."),
                            "routing": "cancelled",
                            "cancelled": True,
                            "metadata": {}
                        }
                    
                    # CHECK: If validation says "ready to execute", route to execution NOW!
                    if result.get("ready_to_execute") and result.get("success"):
                        logger.info("ValidationAgent says ready - routing to ExecutionAgent")
                        
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
                            
                            # Update state with execution result
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
                                    "operation": state.operation
                                }
                            }
                        else:
                            return {
                                "success": False,
                                "response": "Execution agent not available",
                                "routing": "execution"
                            }
                    
                    # Otherwise, return validation response (asking for more info)
                    return {
                        "success": True,
                        "response": result.get("output", ""),
                        "routing": route,
                        "metadata": {
                            "missing_params": result.get("missing_params", []),
                            "ready_to_execute": result.get("ready_to_execute", False)
                        }
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
