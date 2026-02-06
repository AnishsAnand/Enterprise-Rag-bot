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

    def _slim_state_for_llm(self, state: ConversationState) -> Dict[str, Any]:
        """
        Return a minimal, high-signal snapshot of state for LLM prompts.
        Never include large caches / histories (can exceed model context windows).
        """
        return {
            "status": state.status.value,
            "selected_engagement_id": state.selected_engagement_id,
            "user_type": state.user_type,
            "resource_type": state.resource_type,
            "operation": state.operation,
            # Common per-chat selections (small + useful)
            "endpoints": state.collected_params.get("endpoints"),
            "endpoint_names": state.collected_params.get("endpoint_names"),
            "businessUnits": state.collected_params.get("businessUnits"),
            "environments": state.collected_params.get("environments"),
            "zones": state.collected_params.get("zones"),
            # Internal derived context (small)
            "ipc_engagement_id": state.collected_params.get("ipc_engagement_id"),
        }

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
    
    async def _check_greeting_or_capability(self, user_input: str) -> Optional[Dict[str, Any]]:
        """
        Use AI to intelligently determine if user is greeting or asking about capabilities.
        
        This handles first-time users who don't know what the bot can do.
        The AI must distinguish between actual greetings and real operational requests.
        
        Args:
            user_input: User's message
            
        Returns:
            Dict with greeting type if detected, None otherwise
        """
        from app.services.ai_service import ai_service
        
        if not user_input or not user_input.strip():
            return None
        
        # FAST PRE-CHECK: If input contains resource keywords, skip LLM check entirely
        # This prevents misclassification of obvious resource operations
        input_lower = user_input.lower()
        resource_keywords = [
            "cluster", "clusters", "vm", "vms", "virtual machine", 
            "firewall", "firewalls", "load balancer", "database", "databases",
            "endpoint", "endpoints", "jenkins", "kafka", "gitlab", "postgres",
            "datacenter", "data center", "registry", "volume", "storage",
            "kubernetes", "k8s", "node", "nodes", "pod", "pods", "service"
        ]
        action_keywords = [
            "list", "listing", "show", "view", "get", "fetch", "display",
            "create", "deploy", "provision", "delete", "remove", "update",
            "modify", "scale", "restart", "stop", "start", "check", "status"
        ]
        
        has_resource = any(kw in input_lower for kw in resource_keywords)
        has_action = any(kw in input_lower for kw in action_keywords)
        
        # If input has resource OR action keywords, it's an operation - skip greeting check
        if has_resource or has_action:
            logger.debug(f"⚡ Fast-path: detected resource/action keywords in '{user_input[:30]}', skipping greeting check")
            return None
        
        try:
            prompt = f"""You are analyzing user input for a cloud infrastructure management assistant.
Determine if the user is GREETING you or asking about your CAPABILITIES, versus making an actual operational request.

User input: "{user_input}"

Classification rules:
- GREETING: Pure social greetings like "Hi", "Hello", "Good morning", "Hey there", etc. with NO operational intent
- CAPABILITY: Questions about what you can do like "What can you help with?", "What are your features?", "Help me" (without specific task)
- OPERATION: ANY request that mentions resources, actions, or specific tasks:
  * Mentioning resources: clusters, VMs, load balancers, firewalls, databases, etc.
  * Mentioning actions: list, show, view, create, delete, update, deploy, get, check, etc.
  * Mentioning locations: Delhi, Mumbai, Bengaluru, Chennai, datacenter, etc.
  * Mentioning any specific operational context

CRITICAL: If the message contains ANY operational intent (like "view clusters", "show VMs", "list resources"), it is OPERATION, not GREETING or CAPABILITY.

Examples:
- "Hi" → GREETING
- "Hello there!" → GREETING
- "Good morning" → GREETING
- "What can you do?" → CAPABILITY
- "Help" → CAPABILITY
- "What are your features?" → CAPABILITY
- "view existing clusters" → OPERATION (mentions resource)
- "show me VMs" → OPERATION (mentions action and resource)
- "list clusters in Delhi" → OPERATION (mentions action, resource, location)
- "Hi, show me clusters" → OPERATION (has operational intent despite greeting)

Respond with ONLY one word: GREETING, CAPABILITY, or OPERATION"""

            response = await ai_service._call_chat_with_retries(
                prompt=prompt,
                temperature=0.0,
                max_tokens=20
            )
            
            result = response.strip().upper()
            logger.info(f"🤖 Greeting/capability check for '{user_input[:50]}...': {result}")
            
            if result == "GREETING":
                return {"type": "greeting"}
            elif result == "CAPABILITY":
                return {"type": "capability"}
            else:
                # OPERATION or anything else - not a greeting
                return None
                
        except Exception as e:
            logger.warning(f"⚠️ Greeting check AI call failed: {e}")
            # On error, don't block - assume it's not a greeting
            return None
    
    def _get_greeting_response(self, greeting_type: str) -> str:
        """
        Generate a friendly response for greetings and capability questions.
        
        Args:
            greeting_type: "greeting" or "capability"
            
        Returns:
            Friendly response string
        """
        if greeting_type == "greeting":
            return """👋 Hello! I'm your AI assistant for managing cloud infrastructure.

I can help you with:

**🔧 Resource Management**
• Create, view, and manage Kubernetes clusters
• Check load balancers and their configurations  
• View virtual machines, firewalls, and other resources

**📊 Information & Reports**
• List clusters across different datacenters
• Show cluster details and configurations
• Generate reports and summaries

**❓ Questions & Help**
• Answer questions about the platform
• Guide you through complex operations
• Explain concepts and best practices

What would you like to do today?"""
        
        else:  # capability
            return """I'm your AI assistant for **Vayu Cloud Infrastructure Management**.

**Here's what I can help you with:**

🚀 **Create Resources**
• Create new Kubernetes clusters with guided setup
• Configure worker nodes, networking, and more

📋 **View & Manage**
• List clusters, VMs, load balancers, firewalls
• View detailed configurations and status
• Filter by datacenter, business unit, environment

🔍 **Query & Analyze**
• "What clusters are in Delhi?"
• "Show me load balancer details"
• "How many VMs do we have?"

📚 **Learn & Troubleshoot**
• Ask how to do things
• Get explanations of concepts
• Troubleshoot issues

**Try saying:**
• "Create a cluster"
• "List clusters in Mumbai"
• "Show me load balancers"
• "How do I scale a cluster?"

How can I help you today?"""
    
    def _check_context_switch_intent(self, user_input: str) -> Optional[Dict[str, Any]]:
        """
        Check if user wants to switch context (datacenter, engagement, etc.)
        
        Args:
            user_input: User's message
            
        Returns:
            Dict with entity_type and target_value if switch detected, None otherwise
        """
        import re
        text_lower = user_input.lower().strip()
        
        # Context switch keywords
        switch_keywords = ["switch", "change", "use", "set default", "update default", "select"]
        
        # Entity types that can be switched
        entity_patterns = {
            "engagement": r"\b(engagement|engagements)\b",
            "datacenter": r"\b(datacenter|data\s*center|dc|location|endpoint)\b",
            "cluster": r"\b(cluster|k8s\s*cluster)\b",
            "firewall": r"\b(firewall|fw)\b",
            "business_unit": r"\b(business\s*unit|bu|department)\b",
            "environment": r"\b(environment|env)\b",
            "zone": r"\b(zone)\b",
        }
        
        # Check for switch intent
        has_switch_intent = any(kw in text_lower for kw in switch_keywords)
        
        if not has_switch_intent:
            return None
        
        # Detect which entity type is being switched
        detected_entity = None
        for entity_type, pattern in entity_patterns.items():
            if re.search(pattern, text_lower):
                detected_entity = entity_type
                break
        
        if not detected_entity:
            # Check for generic "switch to X" without entity type
            # e.g., "switch to Mumbai" - assume datacenter
            if re.search(r"\bswitch\s+to\s+\w+", text_lower):
                detected_entity = "datacenter"
        
        if detected_entity:
            # Extract the target value (e.g., "Mumbai" from "switch to Mumbai")
            target_value = None
            to_match = re.search(r"\bto\s+(\w+(?:\s*-\s*\w+)?)", text_lower)
            if to_match:
                target_value = to_match.group(1).strip()
            
            return {
                "entity_type": detected_entity,
                "target_value": target_value
            }
        
        return None
    
    async def _generate_follow_ups(self, user_input: str, response: str, context: Dict[str, Any] = None) -> List[str]:
        """
        Generate intelligent follow-up suggestions based on the conversation context.
        
        Args:
            user_input: Original user query
            response: The response that was generated
            context: Additional context (resource type, operation, etc.)
            
        Returns:
            List of 3-5 suggested follow-up questions
        """
        from app.services.ai_service import ai_service
        
        try:
            context_info = ""
            if context:
                resource_type = context.get("resource_type", "")
                operation = context.get("operation", "")
                if resource_type:
                    context_info = f"\nContext: Working with {resource_type}"
                    if operation:
                        context_info += f" ({operation} operation)"
            
            # Truncate response for prompt efficiency
            response_snippet = response[:500] + "..." if len(response) > 500 else response
            
            prompt = f"""Based on this conversation, suggest 3-5 relevant follow-up questions the user might want to ask.

User asked: "{user_input}"

Assistant response summary: "{response_snippet}"
{context_info}

Requirements:
1. Suggestions should be natural follow-up questions
2. They should be actionable and relevant to cloud infrastructure management
3. Mix of: drilling deeper, related operations, and clarifications
4. Keep each suggestion under 80 characters
5. Make them specific, not generic

Examples of good follow-ups:
- "Show me cluster details for <name>"
- "Filter by production environment only"
- "What's the resource usage for these clusters?"
- "How do I add more worker nodes?"

Return ONLY a JSON array of strings, like:
["question 1", "question 2", "question 3"]"""

            llm_response = await ai_service._call_chat_with_retries(
                prompt=prompt,
                temperature=0.7,
                max_tokens=300
            )
            
            # Parse JSON response
            import json
            # Try to extract JSON array from response
            response_text = llm_response.strip()
            if response_text.startswith("["):
                follow_ups = json.loads(response_text)
            else:
                # Try to find JSON array in response
                import re
                match = re.search(r'\[.*\]', response_text, re.DOTALL)
                if match:
                    follow_ups = json.loads(match.group())
                else:
                    logger.warning(f"Could not parse follow-ups: {response_text[:100]}")
                    return []
            
            # Validate and clean
            if isinstance(follow_ups, list):
                return [str(f).strip() for f in follow_ups[:5] if f and len(str(f).strip()) > 5]
            return []
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to generate follow-ups: {e}")
            return []
    
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
    
    async def orchestrate(self,user_input: str,session_id: str,user_id: str,user_roles: List[str] = None,auth_token: str = None,user_type: str = None) -> Dict[str, Any]:
        """
        Main orchestration method that coordinates the entire flow.
        Args:
            user_input: User's message
            session_id: Conversation session ID
            user_id: User identifier
            user_roles: User's roles for permission checking
            auth_token: Bearer token from UI (Keycloak) for API authentication
            user_type: User type (ENG or CUS) for engagement selection logic
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
            
            # Debug: Log state details when loaded
            if state:
                logger.info(f"📂 Loaded existing session: status={state.status.value}, engagement_id={state.selected_engagement_id}, user_type={state.user_type}")
            # Detect OpenWebUI metadata requests (title, tags, follow-ups generation)
            is_metadata_request = any([
                user_input.strip().startswith("### Task:"),
                "Generate a concise" in user_input and "title" in user_input.lower(),
                "Suggest 3-5 relevant follow-up" in user_input,
                "Generate 1-3 broad tags" in user_input])
            # If state exists but is COMPLETED/FAILED, reset for new operation but PRESERVE engagement
            if state and state.status in [ConversationStatus.COMPLETED, ConversationStatus.FAILED, ConversationStatus.CANCELLED]:
                if is_metadata_request:
                    logger.info(f"📋 Metadata request detected, preserving session state")
                else:
                    # Preserve context before resetting (engagement, datacenter/endpoints)
                    preserved_engagement_id = state.selected_engagement_id
                    preserved_user_type = state.user_type
                    # Preserve endpoint/datacenter selection for follow-up queries in same chat
                    preserved_endpoints = state.collected_params.get("endpoints")
                    preserved_endpoint_names = state.collected_params.get("endpoint_names")
                    # Preserve BU/Env/Zone filters (per-chat context)
                    preserved_business_units = state.collected_params.get("businessUnits")
                    preserved_environments = state.collected_params.get("environments")
                    preserved_zones = state.collected_params.get("zones")
                    # Preserve derived IPC engagement id (per-chat context)
                    preserved_ipc_engagement_id = state.collected_params.get("ipc_engagement_id")
                    # Preserve last list cache so "from above response" can filter without refetching
                    preserved_last_list_cache = state.collected_params.get("_last_list_cache")
                    preserved_last_list_resource = state.collected_params.get("_last_list_resource_type")
                    preserved_last_list_endpoints = state.collected_params.get("_last_list_endpoints")
                    preserved_last_list_engagement = state.collected_params.get("_last_list_engagement_id")
                    
                    logger.info(f"🔄 Previous conversation {state.status.value}, resetting for new operation")
                    logger.info(f"   ♻️ Preserving: engagement={preserved_engagement_id}, endpoints={preserved_endpoints}")
                    
                    # Reset operation-specific state but keep session
                    state.status = ConversationStatus.INITIATED
                    state.resource_type = None
                    state.operation = None
                    state.intent = None
                    state.required_params = set()
                    state.optional_params = set()
                    state.collected_params = {}
                    state.missing_params = set()
                    state.invalid_params = {}
                    state.execution_result = None
                    state.error_message = None
                    state.pending_filter_options = None
                    state.pending_filter_type = None
                    state.pending_engagements = None
                    
                    # Restore preserved context (engagement + datacenter/endpoints)
                    state.selected_engagement_id = preserved_engagement_id
                    state.user_type = preserved_user_type
                    if preserved_endpoints:
                        state.collected_params["endpoints"] = preserved_endpoints
                        state.collected_params["endpoint_names"] = preserved_endpoint_names
                        logger.info(f"   ♻️ Restored endpoints: {preserved_endpoint_names}")
                    if preserved_business_units:
                        state.collected_params["businessUnits"] = preserved_business_units
                    if preserved_environments:
                        state.collected_params["environments"] = preserved_environments
                    if preserved_zones:
                        state.collected_params["zones"] = preserved_zones
                    if preserved_ipc_engagement_id:
                        state.collected_params["ipc_engagement_id"] = preserved_ipc_engagement_id
                    if preserved_last_list_cache is not None:
                        state.collected_params["_last_list_cache"] = preserved_last_list_cache
                        state.collected_params["_last_list_resource_type"] = preserved_last_list_resource
                        state.collected_params["_last_list_endpoints"] = preserved_last_list_endpoints
                        state.collected_params["_last_list_engagement_id"] = preserved_last_list_engagement
                    state.user_query = user_input
                    
                    # Persist the reset state
                    conversation_state_manager.update_session(state)
                    
            if not state:
                state = conversation_state_manager.create_session(session_id, user_id, auth_token=auth_token, user_type=user_type)
                state.user_query = user_input
            # Update auth token and user_type in case they changed
            else:
                if auth_token:
                    state.auth_token = auth_token
                if user_type:
                    state.user_type = user_type
                
                # Restore engagement ID to api_executor_service if it exists in state
                # (needed after server restart since api_executor cache is in-memory only)
                if state.selected_engagement_id and state.user_id:
                    from app.services.api_executor_service import api_executor_service
                    await api_executor_service.set_engagement_id(
                        user_id=state.user_id,
                        engagement_id=state.selected_engagement_id
                    )
                    logger.info(f"♻️ Restored engagement ID {state.selected_engagement_id} from persisted state")
                
                # Update user_query for new operational queries
                # Skip only when in engagement selection (preserve original query for continuation)
                if state.status != ConversationStatus.AWAITING_ENGAGEMENT_SELECTION:
                    state.user_query = user_input
                    logger.debug(f"📝 Updated user_query to: '{user_input[:50]}...'")
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
            
            # Generate follow-up suggestions (async, non-blocking if it fails)
            try:
                # Skip follow-ups for metadata requests, workflow prompts, or error responses
                should_generate_followups = (
                    not is_metadata_request and
                    result.get("success", True) and
                    not result.get("workflow_interrupted") and
                    routing_decision.get("route") not in ["greeting"]  # Greetings already suggest actions
                )
                
                if should_generate_followups:
                    response_text = result.get("response", "")
                    # Only generate if we have a substantial response
                    if response_text and len(response_text) > 50:
                        context = {
                            "resource_type": state.resource_type,
                            "operation": state.operation
                        }
                        follow_ups = await self._generate_follow_ups(user_input, response_text, context)
                        if follow_ups:
                            result["follow_ups"] = follow_ups
                            logger.info(f"💡 Generated {len(follow_ups)} follow-up suggestions")
            except Exception as e:
                logger.warning(f"⚠️ Follow-ups generation failed (non-critical): {e}")
            
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
        # PRIORITY 0: Check for greeting/capability questions
        # But ONLY when session is in a neutral state (INITIATED or COMPLETED)
        # When in active workflow (AWAITING_SELECTION, etc.), user input is contextual - don't check
        is_neutral_state = state.status in [ConversationStatus.INITIATED, 
                                            ConversationStatus.COMPLETED,
                                            ConversationStatus.FAILED,
                                            ConversationStatus.CANCELLED]
        
        # PRIORITY 0: For AWAITING_ENGAGEMENT_SELECTION, only reset if user says an EXPLICIT greeting
        # DO NOT use LLM here - it misclassifies numbers like "20" as capability queries
        if state.status == ConversationStatus.AWAITING_ENGAGEMENT_SELECTION:
            # Simple keyword check for explicit greetings only
            explicit_greetings = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening"]
            input_lower = user_input.lower().strip()
            is_explicit_greeting = input_lower in explicit_greetings or any(input_lower.startswith(g + " ") for g in explicit_greetings)
            
            if is_explicit_greeting:
                logger.info(f"👋 Explicit greeting detected during engagement selection, resetting workflow")
                state.status = ConversationStatus.INITIATED
                state.pending_engagements = None
                conversation_state_manager.update_session(state)
                return {
                    "route": "greeting",
                    "reason": "User explicit greeting during engagement selection",
                    "greeting_type": "greeting"
                }
            # Otherwise, proceed to engagement_selection handler (handled in PRIORITY 1 below)
        
        # For neutral states, check for greeting/capability using LLM
        # BUT FIRST: Check if this is a follow-up to a recent operation
        elif is_neutral_state:
            # Check if there's a recent resource operation that this might be a follow-up to
            has_recent_operation = (
                state.resource_type and 
                state.operation and 
                state.status == ConversationStatus.COMPLETED
            )
            
            # Short responses (1-3 words) after a completed operation are likely follow-ups
            input_words = user_input.strip().split()
            is_short_response = len(input_words) <= 3 and len(user_input.strip()) < 50
            
            # Common follow-up patterns
            follow_up_keywords = ["all", "everything", "show all", "list all", "yes", "ok", "okay", "sure", "go ahead"]
            is_follow_up_keyword = user_input.lower().strip() in follow_up_keywords
            
            # If we have a recent operation and user gives a short response, treat as follow-up
            if has_recent_operation and (is_short_response or is_follow_up_keyword):
                logger.info(f"🔄 Detected follow-up to recent operation ({state.operation} {state.resource_type}): '{user_input}'")
                # Reset state and route to intent agent to re-process with context
                state.status = ConversationStatus.INITIATED
                # Build a contextual query that includes the follow-up
                contextual_query = f"{state.operation} {state.resource_type} {user_input}"
                logger.info(f"📝 Contextual query: '{contextual_query}'")
                # Route to intent agent with the contextual query
                return {
                    "route": "intent",
                    "reason": f"Follow-up to recent {state.operation} {state.resource_type} operation",
                    "contextual_query": contextual_query
                }
            
            # Only check for greeting/capability if it's not a follow-up
            greeting_check = await self._check_greeting_or_capability(user_input)
            if greeting_check:
                greeting_type = greeting_check.get('type', 'greeting')
                logger.info(f"👋 Detected greeting/capability question: {greeting_type}")
                return {
                    "route": "greeting",
                    "reason": f"User greeting or capability question: {greeting_type}",
                    "greeting_type": greeting_type
                }
        
        # PRIORITY 0.5: Check for context switch intent (switch datacenter, change engagement, etc.)
        # This should be checked early as it's a high-priority user intent
        context_switch_result = self._check_context_switch_intent(user_input)
        if context_switch_result:
            logger.info(f"🔄 Context switch detected: {context_switch_result.get('entity_type')} -> {context_switch_result.get('target_value')}")
            return {
                "route": "context_switch",
                "reason": f"User wants to switch {context_switch_result.get('entity_type')}",
                "entity_type": context_switch_result.get("entity_type"),
                "target_value": context_switch_result.get("target_value")
            }
        
        # PRIORITY 1: Check for active workflow states
        # These states indicate user is responding to a prompt for a resource operation
        
        # Check if we're awaiting engagement selection (ENG users)
        if state.status == ConversationStatus.AWAITING_ENGAGEMENT_SELECTION:
            logger.info(f"🔄 Session in AWAITING_ENGAGEMENT_SELECTION status, routing to engagement_selection handler")
            return {
                "route": "engagement_selection",
                "reason": "Processing user engagement selection"
            }
        
        # Check if we're awaiting user selection (endpoints, etc.)
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
        
        # Check if we're collecting parameters
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
        
        # Check if ready to execute
        if state.status == ConversationStatus.READY_TO_EXECUTE:
            return {
                "route": "execution",
                "reason": "All parameters collected, ready to execute"
            }
        
        # Note: Greeting check already done at PRIORITY 0 above
        
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
        auth_token = state.auth_token  # Extract auth token from conversation state
        try:
            # Handle greetings and capability questions directly
            if route == "greeting":
                greeting_type = routing_decision.get("greeting_type", "greeting")
                response = self._get_greeting_response(greeting_type)
                logger.info(f"👋 Handled {greeting_type} directly")
                return {
                    "success": True,
                    "response": response,
                    "routing": "greeting",
                    "metadata": {"greeting_type": greeting_type}
                }
            
            # Handle context switch requests (switch datacenter, change engagement, etc.)
            if route == "context_switch":
                from app.agents.handlers.context_switch_handler import context_switch_handler
                
                entity_type = routing_decision.get("entity_type")
                target_value = routing_decision.get("target_value")
                
                logger.info(f"🔄 Executing context switch: {entity_type} -> {target_value}")
                
                result = await context_switch_handler.handle_switch(
                    state=state,
                    entity_type=entity_type,
                    target_value=target_value,
                    auth_token=auth_token
                )
                
                # Update state if awaiting selection
                if result.get("awaiting_selection"):
                    conversation_state_manager.update_session(state)
                
                return {
                    "success": result.get("success", False),
                    "response": result.get("response", ""),
                    "routing": "context_switch",
                    "metadata": {
                        "entity_type": entity_type,
                        "target_value": target_value,
                        "switched_to": result.get("switched_to")
                    }
                }
            
            if route == "intent":
                # Route to intent agent
                state.handoff_to_agent("OrchestratorAgent", "IntentAgent", routing_decision["reason"])
                if self.intent_agent:
                    # Use contextual query if provided (for follow-ups), otherwise use original user_input
                    query_to_use = routing_decision.get("contextual_query", user_input)
                    # Update state.user_query to reflect the contextual query for follow-ups
                    if routing_decision.get("contextual_query"):
                        state.user_query = query_to_use
                        logger.info(f"📝 Updated user_query to contextual query: '{query_to_use[:50]}...'")
                    # IMPORTANT: Do NOT pass full state.to_dict() to the LLM (can exceed context window).
                    # Provide only a slim, high-signal snapshot.
                    slim_state = {
                        "status": state.status.value,
                        "selected_engagement_id": state.selected_engagement_id,
                        "user_type": state.user_type,
                        "resource_type": state.resource_type,
                        "operation": state.operation,
                        "endpoints": state.collected_params.get("endpoints"),
                        "endpoint_names": state.collected_params.get("endpoint_names"),
                        "businessUnits": state.collected_params.get("businessUnits"),
                        "environments": state.collected_params.get("environments"),
                        "zones": state.collected_params.get("zones"),
                    }
                    result = await self.intent_agent.execute(query_to_use, {
                        "session_id": state.session_id,
                        "user_id": state.user_id,
                        "conversation_state": slim_state})
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
                                    "conversation_state": slim_state,
                                    "auth_token": auth_token,
                                    "user_type": state.user_type})
                                
                                # CHECK: If workflow was aborted to handle a different request
                                if validation_result.get("workflow_aborted") and validation_result.get("pending_request"):
                                    pending_request = validation_result.get("pending_request")
                                    logger.info(f"🔀 Workflow aborted - re-routing pending request: '{pending_request}'")
                                    return await self.execute(pending_request, {
                                        "session_id": state.session_id,
                                        "user_roles": user_roles
                                    })
                                
                                # CHECK: If workflow was paused to handle a different request
                                if validation_result.get("workflow_paused") and validation_result.get("pending_request"):
                                    pending_request = validation_result.get("pending_request")
                                    logger.info(f"💾 Workflow paused - handling pending request: '{pending_request}'")
                                    pending_result = await self.execute(pending_request, {
                                        "session_id": state.session_id,
                                        "user_roles": user_roles
                                    })
                                    combined_response = validation_result.get("output", "") + "\n\n---\n\n" + pending_result.get("response", "")
                                    pending_result["response"] = combined_response
                                    return pending_result
                                
                                # CHECK: If workflow interruption prompt was shown
                                if validation_result.get("workflow_interrupted"):
                                    logger.info("⚠️ Workflow interrupted - waiting for user choice")
                                    return {
                                        "success": True,
                                        "response": validation_result.get("output", ""),
                                        "routing": "validation",
                                        "workflow_interrupted": True,
                                        "metadata": {}}
                                
                                # Check if validation made us ready to execute
                                if validation_result.get("ready_to_execute") and validation_result.get("success"):
                                    logger.info("🚀 ValidationAgent says ready - routing to ExecutionAgent")
                                    state.handoff_to_agent("ValidationAgent", "ExecutionAgent", "All parameters collected")
                                    state.status = ConversationStatus.EXECUTING
                                    if self.execution_agent:
                                        exec_result = await self.execution_agent.execute("", {
                                            "session_id": state.session_id,
                                            "user_roles": user_roles or [],
                                            "auth_token": auth_token,
                                            "user_type": state.user_type
                                        })
                                        
                                        # Check if awaiting selection - don't mark as completed
                                        is_awaiting = (
                                            state.status in [ConversationStatus.AWAITING_FILTER_SELECTION, ConversationStatus.AWAITING_ENGAGEMENT_SELECTION] or
                                            exec_result.get("engagement_selection_required")
                                        )
                                        if not is_awaiting and exec_result.get("success"):
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
                                    "user_roles": user_roles or [],
                                    "auth_token": auth_token,
                                    "user_type": state.user_type
                                })
                                
                                # Check if awaiting selection - don't mark as completed
                                is_awaiting = (
                                    state.status in [ConversationStatus.AWAITING_FILTER_SELECTION, ConversationStatus.AWAITING_ENGAGEMENT_SELECTION] or
                                    exec_result.get("engagement_selection_required")
                                )
                                if not is_awaiting and exec_result.get("success"):
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
                    slim_state = self._slim_state_for_llm(state)
                    result = await self.validation_agent.execute(user_input, {
                        "session_id": state.session_id,
                        "conversation_state": slim_state,
                        "auth_token": auth_token,
                        "user_type": state.user_type
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
                    
                    # CHECK: If workflow was aborted to handle a different request
                    if result.get("workflow_aborted") and result.get("pending_request"):
                        pending_request = result.get("pending_request")
                        logger.info(f"🔀 Workflow aborted - re-routing pending request: '{pending_request}'")
                        # Recursively process the pending request
                        return await self.execute(pending_request, {
                            "session_id": state.session_id,
                            "user_roles": user_roles
                        })
                    
                    # CHECK: If workflow was paused to handle a different request
                    if result.get("workflow_paused") and result.get("pending_request"):
                        pending_request = result.get("pending_request")
                        logger.info(f"💾 Workflow paused - handling pending request: '{pending_request}'")
                        # Process the pending request (workflow state is preserved for later resume)
                        pending_result = await self.execute(pending_request, {
                            "session_id": state.session_id,
                            "user_roles": user_roles
                        })
                        # Combine the save message with the result
                        combined_response = result.get("output", "") + "\n\n---\n\n" + pending_result.get("response", "")
                        pending_result["response"] = combined_response
                        return pending_result
                    
                    # CHECK: If workflow interruption prompt was shown (waiting for user choice)
                    if result.get("workflow_interrupted"):
                        logger.info("⚠️ Workflow interrupted - waiting for user choice")
                        return {
                            "success": True,
                            "response": result.get("output", ""),
                            "routing": "validation",
                            "workflow_interrupted": True,
                            "metadata": {}}
                    
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
                                "user_roles": user_roles or [],
                                "auth_token": auth_token,
                                "user_type": state.user_type
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
                        "user_roles": user_roles or [],
                        "auth_token": auth_token,
                        "user_type": state.user_type
                    })
                    
                    # Check if awaiting filter or engagement selection - don't mark as completed
                    is_awaiting_selection = (
                        state.status == ConversationStatus.AWAITING_FILTER_SELECTION or
                        state.status == ConversationStatus.AWAITING_ENGAGEMENT_SELECTION or
                        result.get("engagement_selection_required")
                    )
                    
                    if not is_awaiting_selection:
                        # Update state with execution result
                        if result.get("success"):
                            state.set_execution_result(result.get("execution_result", {}))
                        else:
                            state.set_execution_result({
                                "success": False,
                                "error": result.get("error", "Execution failed")
                            })
                    else:
                        # Persist state with engagement selection status (already set by ExecutionAgent)
                        conversation_state_manager.update_session(state)
                        logger.info(f"📝 State persisted with engagement_selection_required (status={state.status.value})")
                    
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
            
            elif route == "engagement_selection":
                # Process user's engagement selection (ENG users)
                return await self._handle_engagement_selection(user_input, state, user_roles)
            
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
                    "user_roles": user_roles or [],
                    "auth_token": state.auth_token,
                    "user_type": state.user_type
                })
                
                # Only mark as completed if not awaiting selection
                is_awaiting = (
                    state.status in [ConversationStatus.AWAITING_FILTER_SELECTION, ConversationStatus.AWAITING_ENGAGEMENT_SELECTION] or
                    exec_result.get("engagement_selection_required")
                )
                if not is_awaiting and exec_result.get("success"):
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

    async def _handle_engagement_selection(
        self,
        user_input: str,
        state: ConversationState,
        user_roles: List[str]
    ) -> Dict[str, Any]:
        """
        Handle user's engagement selection (for ENG users with multiple engagements).
        
        This is called when state.status == AWAITING_ENGAGEMENT_SELECTION.
        User input could be:
        - A number: "1" → select first engagement
        - A name: "MyProject" → match by engagement name
        - An ID: "1923" → match by engagement ID
        
        Args:
            user_input: User's response
            state: Conversation state with pending_engagements
            user_roles: User's roles
            
        Returns:
            Dict with result (either continue with operation or ask again)
        """
        try:
            logger.info(f"🏢 Processing engagement selection: '{user_input}'")
            
            engagements = state.pending_engagements
            if not engagements:
                logger.error("❌ No pending engagements found in state")
                state.status = ConversationStatus.INITIATED
                state.pending_engagements = None
                conversation_state_manager.update_session(state)
                return {
                    "success": False,
                    "response": "I lost track of the engagement options. Could you please ask again?",
                    "routing": "engagement_selection"
                }
            
            # Parse user selection
            selected_engagement = None
            user_input_lower = user_input.strip().lower()
            
            # Try to match by number (index)
            try:
                index = int(user_input_lower) - 1  # Convert to 0-based index
                if 0 <= index < len(engagements):
                    selected_engagement = engagements[index]
                    logger.info(f"✅ Matched engagement by index: {index + 1}")
            except ValueError:
                pass
            
            # Try to match by ID
            if not selected_engagement:
                try:
                    eng_id = int(user_input_lower)
                    for eng in engagements:
                        if eng.get("id") == eng_id:
                            selected_engagement = eng
                            logger.info(f"✅ Matched engagement by ID: {eng_id}")
                            break
                except ValueError:
                    pass
            
            # Try to match by name (partial match)
            if not selected_engagement:
                for eng in engagements:
                    eng_name = (eng.get("engagementName") or eng.get("name") or "").lower()
                    if user_input_lower in eng_name or eng_name in user_input_lower:
                        selected_engagement = eng
                        logger.info(f"✅ Matched engagement by name: {eng_name}")
                        break
            
            if not selected_engagement:
                # Could not parse - ask again
                logger.warning(f"⚠️ Could not match '{user_input}' to any engagement")
                
                # Build a helpful message
                response = f"I couldn't match '{user_input}' to any engagement.\n\n"
                response += "Please select from the available engagements:\n\n"
                for i, eng in enumerate(engagements, 1):
                    name = eng.get("engagementName") or eng.get("name") or "Unknown"
                    eng_id = eng.get("id")
                    response += f"**{i}. {name}** (ID: {eng_id})\n"
                response += "\nYou can say the number, name, or ID."
                
                return {
                    "success": True,
                    "response": response,
                    "routing": "engagement_selection"
                }
            
            # Successfully selected - store in state and session
            engagement_id = selected_engagement.get("id")
            engagement_name = selected_engagement.get("engagementName") or selected_engagement.get("name")
            
            logger.info(f"✅ User selected engagement: {engagement_name} (ID: {engagement_id})")
            
            # If this was an explicit context-switch flow, clear dependent context (datacenter/endpoints, cluster, etc.)
            # and DO NOT continue with the original "change engagement" text as an operation.
            is_explicit_switch = state.collected_params.get("_pending_context_switch") == "engagement"
            if is_explicit_switch:
                from app.agents.handlers.context_switch_handler import context_switch_handler
                await context_switch_handler._clear_dependents(state, state.user_id, "engagement")
                state.collected_params.pop("_pending_context_switch", None)

            # Store in state
            state.selected_engagement_id = engagement_id
            state.pending_engagements = None
            # Resume the original flow only when this engagement selection was needed for an operation.
            state.status = ConversationStatus.COLLECTING_PARAMS if not is_explicit_switch else ConversationStatus.INITIATED
            
            # Store in API executor session for future calls
            from app.services.api_executor_service import api_executor_service
            await api_executor_service.set_engagement_id(
                user_id=state.user_id,
                engagement_id=engagement_id,
                engagement_data=selected_engagement
            )
            
            # Persist state
            conversation_state_manager.update_session(state)
            
            response = f"✅ Great! I've selected **{engagement_name}** (ID: {engagement_id}) for this session.\n\n"
            if is_explicit_switch:
                response += (
                    "ℹ️ Since the engagement changed, I cleared the dependent context (datacenter/endpoints, "
                    "cluster selection, derived IPC engagement ID, and last list cache)."
                )
                state.user_query = None
                conversation_state_manager.update_session(state)
                return {
                    "success": True,
                    "response": response,
                    "routing": "engagement_selection",
                    "metadata": {"context_cleared": ["datacenter", "cluster"]}
                }

            response += "Now let me continue with your request..."
            
            # Re-process the original query with the engagement now set
            if state.user_query:
                logger.info(f"🔄 Continuing with original query: '{state.user_query}'")
                # Route back to validation to continue
                state.handoff_to_agent("OrchestratorAgent", "ValidationAgent", "Engagement selected, continuing operation")
                if self.validation_agent:
                    slim_state = self._slim_state_for_llm(state)
                    validation_result = await self.validation_agent.execute(state.user_query, {
                        "session_id": state.session_id,
                        "conversation_state": slim_state,
                        "auth_token": state.auth_token,
                        "user_type": state.user_type
                    })
                    
                    # Check if ready to execute
                    if validation_result.get("ready_to_execute") and validation_result.get("success"):
                        logger.info("🚀 ValidationAgent says ready after engagement selection - routing to ExecutionAgent")
                        state.status = ConversationStatus.EXECUTING
                        if self.execution_agent:
                            exec_result = await self.execution_agent.execute("", {
                                "session_id": state.session_id,
                                "user_roles": user_roles or [],
                                "auth_token": state.auth_token,
                                "user_type": state.user_type
                            })
                            
                            if exec_result.get("success"):
                                state.set_execution_result(exec_result.get("execution_result", {}))
                            
                            return {
                                "success": True,
                                "response": response + "\n\n" + exec_result.get("output", ""),
                                "routing": "execution",
                                "execution_result": exec_result.get("execution_result")
                            }
                    
                    return {
                        "success": True,
                        "response": response + "\n\n" + validation_result.get("output", ""),
                        "routing": "validation"
                    }
            
            return {
                "success": True,
                "response": response,
                "routing": "engagement_selection"
            }
            
        except Exception as e:
            logger.error(f"❌ Engagement selection handling failed: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "response": f"Error processing your selection: {str(e)}",
                "routing": "engagement_selection"
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
