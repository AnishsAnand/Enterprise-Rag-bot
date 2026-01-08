"""
Orchestrator Agent - The main coordinator that routes tasks to specialized agents.
This is the entry point for all user requests in the multi-agent system.

Flow: User → Orchestrator → IntentAgent → ValidationAgent → ExecutionAgent → ResourceAgents → API
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
            
            # Handle status - it might be enum or string
            status_value = state.status.value if hasattr(state.status, 'value') else str(state.status)
            
            return json.dumps({
                "intent": state.intent,
                "status": status_value,
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
                "success": result.get("success", True)
            })
            
            # Persist state after each interaction
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
        # Early detection of OpenWebUI metadata requests - skip LLM routing
        is_metadata_request = any([
            user_input.strip().startswith("### Task:"),
            "Generate a concise" in user_input and ("title" in user_input.lower() or "tag" in user_input.lower()),
            "Suggest 3-5 relevant follow-up" in user_input or "Suggest relevant follow-up" in user_input,
            "Generate 1-3 broad tags" in user_input or "Generate broad tags" in user_input,
            "suggest follow-up questions" in user_input.lower(),
            "generate a title" in user_input.lower()
        ])
        
        if is_metadata_request:
            logger.info(f"📋 Metadata request detected early, skipping LLM routing")
            return {
                "route": "skip",
                "reason": "OpenWebUI metadata request - no agent routing needed"
            }
        
        # Check for filter/refinement requests on previous results
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
        
        # Check if we're awaiting confirmation (review step)
        # This is tracked via last_asked_param = "_confirmation" in cluster creation handler
        if (hasattr(state, 'last_asked_param') and 
            state.last_asked_param == "_confirmation" and
            state.status == ConversationStatus.COLLECTING_PARAMS):
            return {
                "route": "validation",
                "reason": "Awaiting user confirmation after review"
            }
        
        # Check if we're in the middle of parameter collection
        if state.status == ConversationStatus.COLLECTING_PARAMS:
            # SPECIAL CASE: For k8s_cluster create, always route to validation
            # (even if missing_params is empty, since it has its own workflow)
            if state.operation == "create" and state.resource_type == "k8s_cluster":
                return {
                    "route": "validation",
                    "reason": "Continuing cluster creation workflow"
                }
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
   - Examples: "list clusters", "show clusters in delhi", "what are the clusters in mumbai?", "how many clusters?", "create a cluster", "delete firewall"
   
B) **DOCUMENTATION**: Questions about how to use the platform, concepts, procedures, troubleshooting, or explanations
   - Examples: "how do I create a cluster?", "what is kubernetes?", "explain load balancing", "why did my deployment fail?"

User Query: "{user_input}"

Instructions:
1. If the query is asking to VIEW, COUNT, LIST, CREATE, UPDATE, or DELETE actual resources → return "RESOURCE_OPERATIONS"
2. If the query is asking HOW TO do something, WHY something works, or WHAT a concept means → return "DOCUMENTATION"

Respond with ONLY ONE of these:
- ROUTE: RESOURCE_OPERATIONS
- ROUTE: DOCUMENTATION"""

        try:
            llm_response = await ai_service._call_chat_with_retries(
                prompt=routing_prompt,
                max_tokens=250,
                temperature=0.1,
                timeout=30  # Increased from 15s - allow time for LLM to respond
            )
            
            logger.info(f"🤖 LLM routing decision: {llm_response}")
            
            # Check for empty response
            if not llm_response or len(llm_response.strip()) < 5:
                logger.error(f"❌ LLM returned empty response for routing")
                return self._rule_based_routing(user_input)
            
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
                logger.warning(f"⚠️ LLM routing unclear: '{llm_response}', using rule-based fallback")
                return self._rule_based_routing(user_input)
                
        except Exception as e:
            logger.error(f"❌ LLM routing failed: {e}, using rule-based fallback")
            return self._rule_based_routing(user_input)
    
    def _normalize_param_names(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize parameter names from LLM extraction to expected formats.
        
        Converts snake_case (from LLM) to camelCase (expected by handlers).
        Also handles common aliases.
        
        Args:
            params: Dictionary of extracted parameters
            
        Returns:
            Normalized parameter dictionary
        """
        # Mapping from LLM-extracted names to expected handler names
        name_mapping = {
            # Cluster parameters
            "cluster_name": "clusterName",
            "clustername": "clusterName",
            "name": "clusterName",  # When context is cluster creation
            "cluster": "clusterName",
            
            # Datacenter/location parameters  
            "datacenter": "datacenter",
            "data_center": "datacenter",
            "location": "_detected_location",
            "endpoint": "_detected_location",
            
            # Version parameters
            "k8s_version": "k8sVersion",
            "kubernetes_version": "k8sVersion",
            "version": "k8sVersion",
            
            # Worker pool parameters
            "worker_pool_name": "workerPoolName",
            "pool_name": "workerPoolName",
            
            # Node parameters
            "node_type": "nodeType",
            "replica_count": "replicaCount",
            "node_count": "replicaCount",
            "replicas": "replicaCount",
            
            # Other parameters
            "cni_driver": "cniDriver",
            "cni": "cniDriver",
            "business_unit": "businessUnit",
            "operating_system": "operatingSystem",
            "os": "operatingSystem",
            "enable_autoscaling": "enableAutoscaling",
            "autoscaling": "enableAutoscaling",
            "max_replicas": "maxReplicas",
        }
        
        normalized = {}
        for key, value in params.items():
            # Check if we have a mapping for this key
            normalized_key = name_mapping.get(key.lower(), key)
            normalized[normalized_key] = value
            
            if normalized_key != key:
                logger.info(f"🔄 Normalized param: {key} → {normalized_key}")
        
        return normalized
    
    def _rule_based_routing(self, user_input: str) -> Dict[str, Any]:
        """
        Rule-based fallback routing when LLM fails.
        
        Args:
            user_input: User's message
            
        Returns:
            Dict with routing decision
        """
        query_lower = user_input.lower()
        
        # Check for documentation patterns
        doc_patterns = [
            "how to", "how do", "how can", "what is", "explain", "why", 
            "tutorial", "guide", "documentation", "help me", "tell me about"
        ]
        
        if any(pattern in query_lower for pattern in doc_patterns):
            # Exception: "what are the clusters" is a resource operation
            if "what are the" in query_lower and any(
                resource in query_lower for resource in 
                ["cluster", "firewall", "vm", "database", "service", "endpoint"]
            ):
                logger.info(f"🎯 Rule-based: 'what are the X' → IntentAgent")
                return {
                    "route": "intent",
                    "reason": "Rule-based routing: resource listing question"
                }
            
            logger.info(f"🎯 Rule-based: documentation pattern → RAGAgent")
            return {
                "route": "rag",
                "reason": "Rule-based routing: documentation question detected"
            }
        
        # Default to intent agent for resource operations
        logger.info(f"🎯 Rule-based: assuming resource operation → IntentAgent")
        return {
            "route": "intent",
            "reason": "Rule-based routing: resource operation assumed"
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
                logger.info(f"⏭️ Skipping routing for metadata request")
                return {
                    "success": True,
                    "response": "Processing metadata request...",
                    "route": "skip"
                }
            
            elif route == "filter":
                return await self._handle_filter_request(user_input, state)
            
            elif route == "intent":
                return await self._handle_intent_routing(user_input, state, user_roles)
            
            elif route == "validation":
                return await self._handle_validation_routing(user_input, state, user_roles)
            
            elif route == "execution":
                return await self._handle_execution_routing(state, user_roles)
            
            elif route == "rag":
                return await self._handle_rag_routing(user_input, state)
            
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
    
    async def _handle_filter_request(
        self,
        user_input: str,
        state: ConversationState
    ) -> Dict[str, Any]:
        """Handle filter/refinement request on previous results."""
        logger.info(f"🔍 Processing filter request on previous execution results")
        
        previous_result = state.execution_result
        previous_data = previous_result.get("data", [])
        
        if not previous_data:
            return {
                "success": False,
                "response": "I don't have any previous results to filter. Could you please make a query first?",
                "route": "filter"
            }
        
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
        
        if resource_agent:
            try:
                filtered_data = await resource_agent.filter_with_llm(
                    data=previous_data,
                    filter_criteria=user_input,
                    resource_type=resource_type
                )
                
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
        else:
            # Fallback: Use LLM to filter manually
            return await self._filter_with_llm_fallback(user_input, previous_data, resource_type, state)
    
    async def _filter_with_llm_fallback(
        self,
        user_input: str,
        previous_data: list,
        resource_type: str,
        state: ConversationState
    ) -> Dict[str, Any]:
        """Fallback LLM-based filtering when no resource agent is available."""
        from app.services.ai_service import ai_service
        
        filter_prompt = f"""Given this user query: "{user_input}"
And this data from a previous query:
{json.dumps(previous_data[:5], indent=2)}... ({len(previous_data)} total items)

Identify what filter criteria the user wants to apply.
Respond with a JSON object containing:
- filter_field: The field name to filter on
- filter_operator: The operator (less_than, equals, contains, greater_than)
- filter_value: The value to compare against"""
        
        try:
            filter_criteria_json = await ai_service._call_chat_with_retries(
                prompt=filter_prompt,
                max_tokens=300,
                temperature=0.1
            )
            filter_criteria = json.loads(filter_criteria_json)
            
            # Apply filter
            filtered_data = []
            filter_field = filter_criteria.get("filter_field")
            filter_operator = filter_criteria.get("filter_operator")
            filter_value = filter_criteria.get("filter_value")
            
            for item in previous_data:
                item_value = item.get(filter_field)
                if filter_operator == "less_than":
                    if item_value and str(item_value) < str(filter_value):
                        filtered_data.append(item)
                elif filter_operator == "equals":
                    if item_value == filter_value:
                        filtered_data.append(item)
                elif filter_operator == "contains":
                    if filter_value.lower() in str(item_value).lower():
                        filtered_data.append(item)
            
            # Format response
            format_prompt = f"""The user asked: "{user_input}"
I filtered {len(previous_data)} items and found {len(filtered_data)} matching items:
{json.dumps(filtered_data, indent=2)}

Please provide a natural, helpful response showing these filtered results."""
            
            formatted_response = await ai_service._call_chat_with_retries(
                prompt=format_prompt,
                max_tokens=2000,
                temperature=0.3
            )
            
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
                "execution_result": state.execution_result
            }
        except Exception as e:
            logger.error(f"❌ Filter processing failed: {e}")
            return {
                "success": False,
                "response": f"I had trouble applying that filter: {str(e)}",
                "route": "filter"
            }
    
    async def _handle_intent_routing(
        self,
        user_input: str,
        state: ConversationState,
        user_roles: List[str]
    ) -> Dict[str, Any]:
        """Handle routing to intent agent."""
        state.handoff_to_agent("OrchestratorAgent", "IntentAgent", "Intent detection")
        
        if not self.intent_agent:
            return {
                "success": False,
                "response": "Intent agent not available",
                "routing": "intent"
            }
        
        result = await self.intent_agent.execute(user_input, {
            "session_id": state.session_id,
            "user_id": state.user_id,
            "conversation_state": state.to_dict()
        })
        
        if not (result.get("success") and result.get("intent_detected")):
            return {
                "success": True,
                "response": result.get("output", ""),
                "routing": "intent"
            }
        
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
        
        # Record intent for agentic metrics evaluation
        self.record_intent(
            intent=f"{intent_data.get('operation', 'unknown')}_{resource_type}",
            resource_type=resource_type,
            operation=intent_data.get("operation")
        )
        
        # Add extracted parameters (normalize naming conventions)
        extracted_params = intent_data.get("extracted_params", {})
        if extracted_params:
            normalized_params = self._normalize_param_names(extracted_params)
            state.add_parameters(normalized_params)
        
        # Check for clarifications or ambiguities
        clarification_needed = intent_data.get("clarification_needed")
        ambiguities = intent_data.get("ambiguities", [])
        
        # SPECIAL CASE: For CREATE operations on k8s_cluster, skip ambiguities
        # The ClusterCreationHandler has its own step-by-step workflow that guides users
        # through all required parameters. We don't want IntentAgent's ambiguities
        # to interfere with that workflow.
        if state.operation == "create" and state.resource_type == "k8s_cluster":
            logger.info(f"🎯 Create cluster detected - skipping IntentAgent ambiguities, using step-by-step workflow")
            clarification_needed = None
            ambiguities = []
        
        # Handle multi-resource requests
        is_multi_resource = state.resource_type and (
            "," in str(state.resource_type) or " and " in str(state.resource_type).lower()
        )
        
        if is_multi_resource:
            logger.info(f"✅ Multi-resource confirmed: {state.resource_type}")
            ambiguities = [amb for amb in ambiguities if not any(
                keyword in amb.lower() for keyword in ["multiple resource", "two resource", "two different"]
            )]
            if not ambiguities:
                clarification_needed = None
        
        if clarification_needed or ambiguities:
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
                "metadata": {"clarification_needed": True, "ambiguities": ambiguities}
            }
        
        # Check if resource type is valid
        if not state.resource_type or state.resource_type == "None":
            logger.error(f"❌ IntentAgent failed to determine resource type")
            return {
                "success": False,
                "response": "I couldn't determine what resource you're asking about. Could you please clarify?",
                "routing": "intent",
                "intent_data": intent_data
            }
        
        # Check if we need more parameters OR if ready to execute
        # SPECIAL CASE: For k8s_cluster create, always start with ValidationAgent
        # to initiate the step-by-step workflow (even if missing_params is empty)
        if state.operation == "create" and state.resource_type == "k8s_cluster":
            logger.info(f"🎯 Starting cluster creation workflow - routing to ValidationAgent")
            state.status = ConversationStatus.COLLECTING_PARAMS
            return await self._handle_validation_routing(user_input, state, user_roles)
        elif state.missing_params:
            logger.info(f"🔄 Missing params: {state.missing_params}, routing to ValidationAgent")
            state.status = ConversationStatus.COLLECTING_PARAMS
            return await self._handle_validation_routing(user_input, state, user_roles)
        else:
            # No missing params - proceed directly to execution
            logger.info(f"✅ All params collected, executing immediately")
            state.status = ConversationStatus.EXECUTING
            return await self._handle_execution_routing(state, user_roles)
    
    async def _handle_validation_routing(
        self,
        user_input: str,
        state: ConversationState,
        user_roles: List[str]
    ) -> Dict[str, Any]:
        """Handle routing to validation agent."""
        state.handoff_to_agent("OrchestratorAgent", "ValidationAgent", "Parameter validation")
        
        if not self.validation_agent:
            return {
                "success": False,
                "response": "Validation agent not available",
                "routing": "validation"
            }
        
        result = await self.validation_agent.execute(user_input, {
            "session_id": state.session_id,
            "conversation_state": state.to_dict()
        })
        
        # Check if validation made us ready to execute
        if result.get("ready_to_execute") and result.get("success"):
            logger.info("🚀 ValidationAgent says ready - routing to ExecutionAgent")
            state.status = ConversationStatus.EXECUTING
            return await self._handle_execution_routing(state, user_roles)
        
        return {
            "success": True,
            "response": result.get("output", ""),
            "routing": "validation",
            "metadata": {
                "missing_params": result.get("missing_params", []),
                "ready_to_execute": result.get("ready_to_execute", False)
            }
        }
    
    async def _handle_execution_routing(
        self,
        state: ConversationState,
        user_roles: List[str]
    ) -> Dict[str, Any]:
        """Handle routing to execution agent."""
        state.handoff_to_agent("OrchestratorAgent", "ExecutionAgent", "Executing operation")
        state.status = ConversationStatus.EXECUTING
        
        if not self.execution_agent:
            return {
                "success": False,
                "response": "Execution agent not available",
                "routing": "execution"
            }
        
        result = await self.execution_agent.execute("", {
            "session_id": state.session_id,
            "conversation_state": state.to_dict(),
            "user_roles": user_roles or []
        })
        
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
            "routing": "execution",
            "execution_result": result.get("execution_result"),
            "metadata": {
                "collected_params": state.collected_params,
                "resource_type": state.resource_type,
                "operation": state.operation
            }
        }
    
    async def _handle_rag_routing(
        self,
        user_input: str,
        state: ConversationState
    ) -> Dict[str, Any]:
        """Handle routing to RAG agent."""
        state.handoff_to_agent("OrchestratorAgent", "RAGAgent", "Documentation query")
        
        if not self.rag_agent:
            return {
                "success": False,
                "response": "RAG agent not available",
                "routing": "rag"
            }
        
        result = await self.rag_agent.execute(user_input, {
            "session_id": state.session_id
        })
        
        return {
            "success": True,
            "response": result.get("output", ""),
            "routing": "rag"
        }
