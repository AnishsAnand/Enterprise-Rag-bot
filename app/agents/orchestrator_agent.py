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
        self.function_calling_agent: Optional[BaseAgent] = None  # OLD: Kept for backwards compatibility
        
        # Feature flag for function calling mode
        # 🆕 DISABLED: Now using multi-agent flow with specialized Resource Agents
        self.use_function_calling = False  # Set to True to use old FunctionCallingAgent (bypasses multi-agent flow)
        
        # Setup agent with tools
        self.setup_agent()
    
    def set_specialized_agents(
        self,
        intent_agent: BaseAgent,
        validation_agent: BaseAgent,
        execution_agent: BaseAgent,
        rag_agent: BaseAgent,
        function_calling_agent: Optional[BaseAgent] = None
    ) -> None:
        """
        Set references to specialized agents.
        
        Args:
            intent_agent: Agent for intent detection
            validation_agent: Agent for parameter validation
            execution_agent: Agent for API execution
            rag_agent: Agent for RAG-based responses
            function_calling_agent: Modern function calling agent (optional)
        """
        self.intent_agent = intent_agent
        self.validation_agent = validation_agent
        self.execution_agent = execution_agent
        self.rag_agent = rag_agent
        self.function_calling_agent = function_calling_agent
        logger.info("✅ Orchestrator agent configured with specialized agents")
        if function_calling_agent:
            logger.info("🎯 Function calling agent enabled - using modern approach")
    
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
                    logger.info(f"📋 Metadata request detected, preserving session state")
                else:
                    # Regular completed conversation - start fresh for new user query
                    logger.info(f"🔄 Previous conversation {state.status.value}, starting fresh session")
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
        Decide which agent should handle the request using LLM-based routing.
        
        Args:
            user_input: User's message
            state: Current conversation state
            user_roles: User's roles
            
        Returns:
            Dict with routing decision
        """
        # OPTION 2: Early detection of OpenWebUI metadata requests - skip LLM routing entirely
        # These requests are for UI enhancement and should not trigger agent routing
        is_metadata_request = any([
            user_input.strip().startswith("### Task:"),
            "Generate a concise" in user_input and ("title" in user_input.lower() or "tag" in user_input.lower()),
            "Suggest 3-5 relevant follow-up" in user_input or "Suggest relevant follow-up" in user_input,
            "Generate 1-3 broad tags" in user_input or "Generate broad tags" in user_input,
            "suggest follow-up questions" in user_input.lower(),
            "generate a title" in user_input.lower()
        ])
        
        if is_metadata_request:
            logger.info(f"📋 Metadata request detected early, skipping LLM routing: {user_input[:80]}...")
            return {
                "route": "skip",
                "reason": "OpenWebUI metadata request - no agent routing needed"
            }
        
        # 🆕 CHECK FOR FILTER/REFINEMENT REQUESTS ON PREVIOUS RESULTS
        # Detect when user wants to filter/refine the last result instead of making a new query
        filter_keywords = [
            "filter", "show only", "just show", "only show",
            "version below", "version above", "version less than", "version greater than",
            "filter the above", "filter that", "from the above", "from that result",
            "in the above", "from those", "just the", "only the", 
            "exclude", "without", "remove", "except"
        ]
        
        is_filter_request = any(keyword in user_input.lower() for keyword in filter_keywords)
        has_previous_result = (
            state.execution_result is not None 
            and state.execution_result.get("data") 
            and len(state.execution_result.get("data", [])) > 0
        )
        
        if is_filter_request and has_previous_result:
            logger.info(f"🔍 Filter/refinement request detected on previous results")
            return {
                "route": "filter",
                "reason": "User wants to filter/refine previous execution results",
                "filter_query": user_input
            }
        
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
            logger.debug(f"🔍 Routing prompt for query: {user_input}")
            llm_response = await ai_service._call_chat_with_retries(
                prompt=routing_prompt,
                max_tokens=250,  # OPTION 1: Increased from 100 to 250 for more reliable responses from LLM
                temperature=0.1,  # Slightly increased from 0.0 to avoid potential model issues
                timeout=15  # Add explicit timeout
            )
            
            logger.info(f"🤖 LLM routing decision (length={len(llm_response)} chars): {llm_response}")
            
            # Check for empty or too short response
            if not llm_response or len(llm_response.strip()) < 5:
                logger.error(f"❌ LLM returned empty/very short response ('{llm_response}') for routing")
                # Use rule-based fallback for common documentation patterns
                query_lower = user_input.lower()
                doc_patterns = ["how to", "how do", "how can", "what is", "what are", "explain", "why", 
                               "tutorial", "guide", "documentation", "help me", "tell me about"]
                if any(pattern in query_lower for pattern in doc_patterns):
                    logger.info(f"🎯 Rule-based fallback: detected documentation pattern in '{user_input}'")
                    return {
                        "route": "rag",
                        "reason": "Rule-based routing: documentation question detected (LLM response empty)"
                    }
                else:
                    # Default to function_calling if available, otherwise intent
                    if self.use_function_calling and self.function_calling_agent:
                        logger.info(f"🎯 Rule-based fallback: assuming resource operation → FunctionCallingAgent")
                        return {
                            "route": "function_calling",
                            "reason": "Rule-based routing: resource operation assumed (LLM response empty)"
                        }
                    else:
                        logger.info(f"🎯 Rule-based fallback: assuming resource operation → IntentAgent")
                        return {
                            "route": "intent",
                            "reason": "Rule-based routing: resource operation assumed (LLM response empty)"
                        }
            
            # Parse LLM response
            if "RESOURCE_OPERATIONS" in llm_response.upper():
                # Route to FunctionCallingAgent if available, otherwise IntentAgent
                if self.use_function_calling and self.function_calling_agent:
                    logger.info(f"✅ LLM routing: RESOURCE_OPERATIONS → FunctionCallingAgent")
                    return {
                        "route": "function_calling",
                        "reason": "LLM detected resource operation - using modern function calling"
                    }
                else:
                    logger.info(f"✅ LLM routing: RESOURCE_OPERATIONS → IntentAgent (fallback)")
                    return {
                        "route": "intent",
                        "reason": "LLM detected resource operation intent (traditional flow)"
                    }
            elif "DOCUMENTATION" in llm_response.upper():
                logger.info(f"✅ LLM routing: DOCUMENTATION → RAGAgent")
                return {
                    "route": "rag",
                    "reason": "LLM detected documentation question"
                }
            else:
                # Fallback to function_calling (if available) or intent
                if self.use_function_calling and self.function_calling_agent:
                    logger.warning(f"⚠️ LLM routing unclear: '{llm_response}', defaulting to function_calling")
                    return {
                        "route": "function_calling",
                        "reason": "Ambiguous routing, defaulting to function calling"
                    }
                else:
                    logger.warning(f"⚠️ LLM routing unclear: '{llm_response}', defaulting to intent")
                    return {
                        "route": "intent",
                        "reason": "Ambiguous routing, defaulting to intent detection"
                    }
        except Exception as e:
            logger.error(f"❌ LLM routing failed with exception: {e}, using rule-based fallback")
            
            # Check if this is a metadata request - don't route to RAG or Intent
            if any([
                user_input.strip().startswith("### Task:"),
                "Generate a concise" in user_input and "title" in user_input.lower(),
                "Suggest 3-5 relevant follow-up" in user_input,
                "Generate 1-3 broad tags" in user_input
            ]):
                logger.info(f"🎯 Metadata request detected with LLM failure - skipping routing")
                return {
                    "route": "skip",
                    "reason": "Metadata request should be handled by OpenWebUI, not by agents"
                }
            
            # Use rule-based fallback on exception for regular queries
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
                # Default to function_calling if available, otherwise intent
                if self.use_function_calling and self.function_calling_agent:
                    logger.info(f"🎯 Rule-based fallback (exception): assuming resource operation → FunctionCallingAgent")
                    return {
                        "route": "function_calling",
                        "reason": "Rule-based routing: resource operation assumed (LLM error)"
                    }
                else:
                    logger.info(f"🎯 Rule-based fallback (exception): assuming resource operation → IntentAgent")
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
            if route == "skip":
                # Metadata request - return simple acknowledgment
                logger.info(f"⏭️ Skipping routing for metadata request")
                return {
                    "success": True,
                    "response": "Processing metadata request...",
                    "route": "skip"
                }
            
            elif route == "filter":
                # 🆕 FILTER/REFINEMENT REQUEST - Apply filter to previous results
                logger.info(f"🔍 Processing filter request on previous execution results")
                
                # Get previous execution result
                previous_result = state.execution_result
                previous_data = previous_result.get("data", [])
                
                if not previous_data:
                    return {
                        "success": False,
                        "response": "I don't have any previous results to filter. Could you please make a query first?",
                        "route": "filter"
                    }
                
                # Determine which resource agent to use based on previous operation
                resource_type = state.resource_type
                if not resource_type:
                    return {
                        "success": False,
                        "response": "I couldn't determine what type of data to filter. Could you please specify?",
                        "route": "filter"
                    }
                
                # Get the appropriate resource agent
                resource_agent = None
                if self.execution_agent and hasattr(self.execution_agent, 'resource_agent_map'):
                    resource_agent = self.execution_agent.resource_agent_map.get(resource_type)
                
                if not resource_agent:
                    # Fallback: Use LLM to filter manually
                    logger.warning(f"⚠️ No resource agent found for {resource_type}, using direct LLM filtering")
                    from app.services.ai_service import ai_service
                    
                    filter_prompt = f"""Given this user query: "{user_input}"
And this data from a previous query:
{json.dumps(previous_data[:5], indent=2)}... ({len(previous_data)} total items)

Identify what filter criteria the user wants to apply.
Respond with a JSON object containing:
- filter_field: The field name to filter on (e.g., "k8sVersion", "status", "name")
- filter_operator: The operator (e.g., "less_than", "equals", "contains", "greater_than")
- filter_value: The value to compare against

Example: "version below 1.30" → {{"filter_field": "k8sVersion", "filter_operator": "less_than", "filter_value": "1.30"}}
"""
                    
                    try:
                        filter_criteria_json = await ai_service._call_chat_with_retries(
                            prompt=filter_prompt,
                            max_tokens=300,
                            temperature=0.1
                        )
                        filter_criteria = json.loads(filter_criteria_json)
                        
                        # Apply filter manually
                        filtered_data = []
                        filter_field = filter_criteria.get("filter_field")
                        filter_operator = filter_criteria.get("filter_operator")
                        filter_value = filter_criteria.get("filter_value")
                        
                        for item in previous_data:
                            item_value = item.get(filter_field)
                            if filter_operator == "less_than":
                                # Handle version comparison
                                if item_value and str(item_value) < str(filter_value):
                                    filtered_data.append(item)
                            elif filter_operator == "equals":
                                if item_value == filter_value:
                                    filtered_data.append(item)
                            elif filter_operator == "contains":
                                if filter_value.lower() in str(item_value).lower():
                                    filtered_data.append(item)
                        
                        # Format response with LLM
                        format_prompt = f"""The user asked: "{user_input}"

I filtered {len(previous_data)} items and found {len(filtered_data)} matching items:
{json.dumps(filtered_data, indent=2)}

Please provide a natural, helpful response to the user showing these filtered results.
Include relevant details in a formatted way (use tables if appropriate)."""
                        
                        formatted_response = await ai_service._call_chat_with_retries(
                            prompt=format_prompt,
                            max_tokens=2000,
                            temperature=0.3
                        )
                        
                        # Update execution result with filtered data
                        state.execution_result = {
                            "success": True,
                            "data": filtered_data,
                            "resource_type": resource_type,
                            "operation": "filter",
                            "filter_applied": filter_criteria
                        }
                        state.status = ConversationStatus.COMPLETED
                        
                        return {
                            "success": True,
                            "response": formatted_response,
                            "route": "filter",
                            "execution_result": state.execution_result,
                            "metadata": {
                                "original_count": len(previous_data),
                                "filtered_count": len(filtered_data),
                                "filter_criteria": filter_criteria
                            }
                        }
                    except Exception as e:
                        logger.error(f"❌ Filter processing failed: {e}")
                        return {
                            "success": False,
                            "response": f"I had trouble applying that filter: {str(e)}",
                            "route": "filter"
                        }
                else:
                    # Use resource agent's filter_with_llm method
                    logger.info(f"🎯 Using {resource_agent.agent_name} to filter data")
                    
                    try:
                        filtered_data = await resource_agent.filter_with_llm(
                            data=previous_data,
                            filter_criteria=user_input,
                            resource_type=resource_type
                        )
                        
                        # Format response
                        formatted_response = await resource_agent.format_response_with_llm(
                            operation="list",
                            raw_data=filtered_data,
                            user_query=user_input,
                            context={
                                "filtered": True,
                                "original_count": len(previous_data),
                                "endpoint_names": state.collected_params.get("endpoint_names", [])
                            }
                        )
                        
                        # Update execution result with filtered data
                        state.execution_result = {
                            "success": True,
                            "data": filtered_data,
                            "resource_type": resource_type,
                            "operation": "filter"
                        }
                        state.status = ConversationStatus.COMPLETED
                        
                        return {
                            "success": True,
                            "response": formatted_response,
                            "route": "filter",
                            "execution_result": state.execution_result,
                            "metadata": {
                                "original_count": len(previous_data),
                                "filtered_count": len(filtered_data)
                            }
                        }
                    except Exception as e:
                        logger.error(f"❌ Resource agent filter failed: {e}")
                        return {
                            "success": False,
                            "response": f"I had trouble applying that filter: {str(e)}",
                            "route": "filter"
                        }
            
            elif route == "function_calling":
                # NEW: Route to modern function calling agent
                logger.info(f"🎯 Routing to FunctionCallingAgent (modern approach)")
                state.handoff_to_agent("OrchestratorAgent", "FunctionCallingAgent", routing_decision["reason"])
                
                if self.function_calling_agent:
                    # Build conversation history for context
                    conversation_history = []
                    for msg in state.conversation_history[-10:]:  # Last 10 messages
                        conversation_history.append({
                            "role": msg.get("role", "user"),
                            "content": msg.get("content", "")
                        })
                    
                    result = await self.function_calling_agent.execute(user_input, {
                        "session_id": state.session_id,
                        "user_id": state.user_id,
                        "user_roles": user_roles or [],
                        "conversation_history": conversation_history,
                        "conversation_state": state.to_dict()
                    })
                    
                    # Mark conversation as completed
                    if result.get("success"):
                        state.status = ConversationStatus.COMPLETED
                    
                    return {
                        "success": result.get("success", True),
                        "response": result.get("output", ""),
                        "routing": "function_calling",
                        "function_calls_made": result.get("function_calls_made", []),
                        "iterations": result.get("iterations", 1),
                        "metadata": {
                            "agent_type": "function_calling",
                            "modern_approach": True
                        }
                    }
                else:
                    # Fallback to traditional intent agent if function calling not available
                    logger.warning("⚠️ Function calling agent not available, falling back to intent agent")
                    return await self._execute_routing(
                        {"route": "intent", "reason": "Fallback to traditional flow"},
                        user_input,
                        state,
                        user_roles
                    )
            
            elif route == "intent":
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
                        
                        # Handle resource_type: convert list to comma-separated string
                        resource_type = intent_data.get("resource_type")
                        if isinstance(resource_type, list):
                            resource_type = ",".join(resource_type)
                            logger.info(f"🔧 Converted resource_type list to string: {resource_type}")
                        
                        state.set_intent(
                            resource_type=resource_type,
                            operation=intent_data.get("operation"),
                            required_params=intent_data.get("required_params", []),
                            optional_params=intent_data.get("optional_params", [])
                        )
                        
                        # Add extracted parameters
                        extracted_params = intent_data.get("extracted_params", {})
                        if extracted_params:
                            state.add_parameters(extracted_params)
                        
                        # 🆕 CHECK FOR CLARIFICATIONS OR AMBIGUITIES FIRST!
                        clarification_needed = intent_data.get("clarification_needed")
                        ambiguities = intent_data.get("ambiguities", [])
                        
                        logger.info(f"📋 State resource_type: {state.resource_type}, Ambiguities: {ambiguities}")
                        
                        # SPECIAL CASE: Multi-resource requests with null resource_type
                        # Extract resources from ambiguity/clarification text
                        if not state.resource_type and (ambiguities or clarification_needed):
                            logger.info(f"🔧 Attempting to extract resources from ambiguity text")
                            
                            # Try to extract from ambiguities first
                            for ambiguity in ambiguities:
                                # Simple extraction: look for resource names we know about
                                resources_found = []
                                known_resources = ["gitlab", "kafka", "jenkins", "postgres", "documentdb", "container_registry", "registry"]
                                for resource in known_resources:
                                    if resource in ambiguity.lower():
                                        resources_found.append(resource)
                                
                                if len(resources_found) >= 2:
                                    resources_str = ",".join(resources_found)
                                    logger.info(f"✅ Extracted multi-resource: '{resources_str}'")
                                    state.resource_type = resources_str
                                    state.intent = f"list_{resources_str}"
                                    break
                            
                            # Also try clarification_needed text
                            if not state.resource_type and clarification_needed:
                                resources_found = []
                                for resource in ["gitlab", "kafka", "jenkins", "postgres", "documentdb", "container_registry", "registry"]:
                                    if resource in clarification_needed.lower():
                                        resources_found.append(resource)
                                
                                if len(resources_found) >= 2:
                                    resources_str = ",".join(resources_found)
                                    logger.info(f"✅ Extracted multi-resource from clarification: '{resources_str}'")
                                    state.resource_type = resources_str
                                    state.intent = f"list_{resources_str}"
                        
                        # EXCEPTION: If resource_type has comma/and, this is multi-resource
                        # Don't ask for clarification - proceed with execution
                        is_multi_resource = state.resource_type and ("," in str(state.resource_type) or " and " in str(state.resource_type).lower())
                        
                        if is_multi_resource:
                            logger.info(f"✅ Multi-resource confirmed: {state.resource_type}, skipping clarification")
                            # Clear ambiguities about multiple resources
                            ambiguities = [amb for amb in ambiguities if not any(
                                keyword in amb.lower() for keyword in ["multiple resource", "two resource", "two different"]
                            )]
                            if not ambiguities:
                                clarification_needed = None
                        
                        if clarification_needed or ambiguities:
                            # IntentAgent needs clarification from user
                            logger.info(f"🤔 Intent clarification needed or ambiguities detected")
                            
                            # Format ambiguities for user
                            ambiguity_text = ""
                            if ambiguities:
                                ambiguity_text = f"\n\n**Ambiguities detected:**\n" + "\n".join(f"- {amb}" for amb in ambiguities)
                            
                            response_text = clarification_needed or "I need some clarification to proceed."
                            response_text += ambiguity_text
                            
                            return {
                                "success": True,
                                "response": response_text,
                                "routing": "intent",
                                "intent_data": intent_data,
                                "metadata": {
                                    "clarification_needed": True,
                                    "ambiguities": ambiguities
                                }
                            }
                        
                        # 🆕 CHECK IF RESOURCE TYPE IS NULL/NONE
                        logger.info(f"🔍 Checking resource type: '{state.resource_type}' (type: {type(state.resource_type).__name__})")
                        if not state.resource_type or state.resource_type == "None":
                            logger.error(f"❌ IntentAgent failed to determine resource type")
                            return {
                                "success": False,
                                "response": "I couldn't determine what resource you're asking about. Could you please clarify?",
                                "routing": "intent",
                                "intent_data": intent_data
                            }
                        
                        # STEP 2: Check if we need more parameters OR if ready to execute
                        if state.missing_params:
                            logger.info(f"🔄 Missing params detected: {state.missing_params}, routing to ValidationAgent")
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
                                    logger.info("🚀 ValidationAgent says ready - routing to ExecutionAgent")
                                    
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

