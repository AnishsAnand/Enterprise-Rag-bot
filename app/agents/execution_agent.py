"""
Execution Agent - Executes validated CRUD operations via Resource Agents.
Routes operations to specialized agents for domain-specific handling.
"""
from typing import Any, Dict, List, Optional
from langchain.tools import Tool
import logging
import json
from app.agents.base_agent import BaseAgent
from app.agents.state.conversation_state import (conversation_state_manager,ConversationStatus,ConversationState)
from app.services.api_executor_service import api_executor_service
from app.agents.resource_agents.k8s_cluster_agent import K8sClusterAgent
from app.agents.resource_agents.managed_services_agent import ManagedServicesAgent
from app.agents.resource_agents.virtual_machine_agent import VirtualMachineAgent
from app.agents.resource_agents.network_agent import NetworkAgent
from app.agents.resource_agents.load_balancer_agent import LoadBalancerAgent
from app.agents.resource_agents.generic_resource_agent import GenericResourceAgent
from app.agents.resource_agents.reports_agent import ReportsAgent

logger = logging.getLogger(__name__)

class ExecutionAgent(BaseAgent):
    """
    Agent specialized in executing validated CRUD operations.
    Routes ALL operations to specialized resource agents for clean, maintainable code.
    """
    def __init__(self):
        super().__init__(
            agent_name="ExecutionAgent",
            agent_description=(
                "Executes validated CRUD operations on cloud resources. "
                "Routes to specialized resource agents for domain-specific handling."
            ),
            temperature=0.3)
        # Initialize specialized resource agents
        self.k8s_agent = K8sClusterAgent()
        self.managed_services_agent = ManagedServicesAgent()
        self.vm_agent = VirtualMachineAgent()
        self.network_agent = NetworkAgent()  
        self.load_balancer_agent = LoadBalancerAgent()
        self.generic_agent = GenericResourceAgent()
        self.reports_agent = ReportsAgent()
        
        # Resource type to agent mapping - ALL resources go through agents
        self.resource_agent_map = {
            # Kubernetes
            "k8s_cluster": self.k8s_agent,
            "cluster": self.k8s_agent,
            "kubernetes": self.k8s_agent,
            "kafka": self.managed_services_agent,
            "gitlab": self.managed_services_agent,
            "jenkins": self.managed_services_agent,
            "postgres": self.managed_services_agent,
            "postgresql": self.managed_services_agent,
            "documentdb": self.managed_services_agent,
            "container_registry": self.managed_services_agent,
            "registry": self.managed_services_agent,
            "vm": self.vm_agent,
            "virtual_machine": self.vm_agent,
            "firewall": self.network_agent,
            "load_balancer": self.load_balancer_agent,
            "lb": self.load_balancer_agent,
            "loadbalancer": self.load_balancer_agent,

            # Reports
            "reports": self.reports_agent,
            "report": self.reports_agent,
            "common_cluster_report": self.reports_agent,
            
            # Generic (fallback for other resources)
            "endpoint": self.generic_agent,
            "business_unit": self.generic_agent,
            "environment": self.generic_agent,
            "zone": self.generic_agent,}

        logger.info(f"✅ ExecutionAgent initialized with {len(self.resource_agent_map)} resource agent mappings")
        logger.info(f"🔧 Load balancer operations → LoadBalancerAgent")
        logger.info(f"🔥 Firewall operations → NetworkAgent")
        self.setup_agent()
    
    def get_system_prompt(self) -> str:
        return """You are the Execution Agent, responsible for executing validated operations on cloud resources.

**Your responsibilities:**
1. **Route operations** to specialized resource agents
2. **Handle execution results** (success and errors)
3. **Provide clear feedback** to users about what happened

**Supported Resources:**
- Kubernetes clusters (K8sClusterAgent)
- Managed services: Kafka, GitLab, Jenkins, PostgreSQL, DocumentDB, Container Registry (ManagedServicesAgent)
- Virtual machines (VirtualMachineAgent)
- Firewalls (NetworkAgent)
- Load balancers (LoadBalancerAgent)
- Reports: Common Cluster Report (ReportsAgent)
- Generic: Endpoints, Business Units, Environments, Zones (GenericResourceAgent)

**Load Balancer Operations:**
- list: List all load balancers (uses IPC engagement ID)
- get_details: Get detailed configuration for specific LB
- get_virtual_services: Get VIPs/listeners for specific LB

All operations are routed through specialized resource agents for proper handling."""
    
    def get_tools(self) -> List[Tool]:
        """Return tools for execution agent."""
        return [
            Tool(
                name="execute_operation",
                func=self._tool_execute_operation,
                description="Execute a CRUD operation on a resource. Input: JSON with resource_type, operation, params"
            ),
            Tool(
                name="get_available_endpoints",
                func=self._tool_get_endpoints,
                description="Get available data center endpoints. Input: empty dict {}"
            )]

    def _should_use_previous_list_cache(self, user_query: str) -> bool:
        if not user_query:
            return False
        q = user_query.lower()
        return any(
            phrase in q
            for phrase in [
                "from above response",
                "from above",
                "above response",
                "previous response",
                "from the above",
                "from the previous",
                "from earlier",
                "based on above",
                # Common follow-up filtering phrasing (no refetch; filter last list)
                "show only",
                "only show",
                "now show",
                "filter",
            ]
        )
    
    def _tool_execute_operation(self, input_json: str) -> str:
        """Tool wrapper for execute_operation."""
        try:
            data = json.loads(input_json)
            import asyncio
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(
                self._execute_via_resource_agent(
                    resource_type=data.get("resource_type"),
                    operation=data.get("operation"),
                    params=data.get("params", {}),
                    context=data.get("context", {})))
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})
    
    def _tool_get_endpoints(self, input_json: str) -> str:
        """Tool wrapper for get endpoints."""
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(api_executor_service.list_endpoints())
            return json.dumps(result, indent=2)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})
    # =========================================================================
    # HELPER: Endpoint Name to ID Conversion
    # =========================================================================
    async def _convert_endpoint_names_to_ids(self,endpoint_names: List[str]) -> tuple[List[int], Optional[str]]:
        """
        Convert endpoint names to endpoint IDs.
        Args:
            endpoint_names: List of endpoint names (e.g., ["Delhi", "Mumbai"])
        Returns:
            Tuple of (endpoint_ids, error_message)
        """
        if not endpoint_names:
            return [], None
        # Check if already IDs
        if all(isinstance(e, int) or (isinstance(e, str) and e.isdigit()) for e in endpoint_names):
            return [int(e) for e in endpoint_names], None
        try:
            # Fetch available endpoints
            endpoints_result = await api_executor_service.list_endpoints()
            if not endpoints_result.get("success"):
                return [], "Failed to fetch endpoints for name-to-ID conversion"
            available_endpoints = endpoints_result.get("data", {}).get("endpoints", [])
            # Build name -> ID mapping
            name_to_id = {}
            for ep in available_endpoints:
                ep_name = ep.get("name", "").strip()
                ep_id = ep.get("id")
                if ep_name and ep_id:
                    # Add exact match and normalized versions
                    name_to_id[ep_name.lower()] = ep_id
                    name_to_id[ep_name.lower().replace("-", "").replace(" ", "")] = ep_id
            # Convert names to IDs
            converted_ids = []
            not_found = []
            for name in endpoint_names:
                name_clean = name.lower().strip().replace("-", "").replace(" ", "")
                if name_clean in name_to_id:
                    converted_ids.append(name_to_id[name_clean])
                    logger.info(f"  ✅ '{name}' -> ID {name_to_id[name_clean]}")
                else:
                    not_found.append(name)
            if not_found and not converted_ids:
                return [], f"Could not find endpoint IDs for: {', '.join(not_found)}"
            return converted_ids, None
        except Exception as e:
            logger.error(f"❌ Error converting endpoint names to IDs: {str(e)}")
            return [], f"Error converting endpoint names: {str(e)}"
    # =========================================================================
    # CORE: Execute via Resource Agent
    # =========================================================================
    async def _execute_via_resource_agent(self,resource_type: str,operation: str,params: Dict[str, Any],context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute operation via the appropriate resource agent.
        Args:
            resource_type: Type of resource (k8s_cluster, kafka, load_balancer, etc.)
            operation: Operation to perform (list, create, get_details, etc.)
            params: Operation parameters
            context: Execution context (session_id, user_id, user_roles, etc.)
        Returns:
            Dict with execution result
        """
        # Get the resource agent
        resource_agent = self.resource_agent_map.get(resource_type, self.generic_agent)
        logger.info(f"🎯 Routing {operation} {resource_type} to {resource_agent.agent_name}")
        # Pre-process: Convert endpoint names to IDs if needed
        if resource_type not in ["load_balancer", "lb", "loadbalancer"]:
            endpoint_names = params.get("endpoints") or params.get("endpoint_ids") or params.get("endpoint_names")
            if endpoint_names and isinstance(endpoint_names, list):
                if any(isinstance(e, str) and not e.isdigit() for e in endpoint_names):
                    converted_ids, error = await self._convert_endpoint_names_to_ids(endpoint_names)
                    if error and not converted_ids:
                        return {"success": False, "error": error}
                    if converted_ids:
                        params["endpoints"] = converted_ids
                        # Store original names for display
                        params["endpoint_names"] = endpoint_names
                        logger.info(f"✅ Converted endpoints: {endpoint_names} -> {converted_ids}")
        # Execute via resource agent
        try:
            result = await resource_agent.execute_operation(
                operation=operation,
                params=params,
                context={
                    **context,
                    "resource_type": resource_type})
            return result
        except Exception as e:
            logger.error(f"❌ Resource agent execution failed: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "response": f"Failed to execute {operation} on {resource_type}: {str(e)}"}
    
    # =========================================================================
    # MAIN: Execute Method (Clean Version)
    # =========================================================================
    
    async def execute(self,input_text: str,context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute the operation from conversation state.
        Routes ALL operations through resource agents for clean, maintainable code.
        Args:
            input_text: User's message (usually confirmation)
            context: Context including session_id, conversation_state, user_roles
        Returns:
            Dict with execution result
        """
        try:
            logger.info(f"⚡ ExecutionAgent executing operation...")
            # Get conversation state
            session_id = context.get("session_id") if context else None
            state = conversation_state_manager.get_session(session_id) if session_id else None
            if not state:
                logger.error("❌ No conversation state found!")
                return {
                    "agent_name": self.agent_name,
                    "success": False,
                    "error": "No conversation state found",
                    "output": "I couldn't find the operation to execute."}
            logger.info(f"📌 State: resource={state.resource_type}, operation={state.operation}, ready={state.is_ready_to_execute()}")
            # Check if ready to execute
            if not state.is_ready_to_execute():
                logger.warning(f"⚠️ Not ready to execute! Missing: {state.missing_params}")
                return {
                    "agent_name": self.agent_name,
                    "success": False,
                    "error": "Not ready to execute",
                    "output": state.get_missing_params_message()}
            # Get user context
            user_roles = context.get("user_roles", []) if context else []
            user_id = context.get("user_id") if context else None
            user_query = context.get("user_query", state.user_query) if context else state.user_query
            auth_token = context.get("auth_token") if context else state.auth_token

            # If user asked to filter "from above response", reuse last list data (no refetch)
            if (
                state.operation == "list"
                and self._should_use_previous_list_cache(user_query)
                and state.collected_params.get("_last_list_cache") is not None
                and state.collected_params.get("_last_list_resource_type") == state.resource_type
            ):
                logger.info("♻️ Using cached previous list response (no API refetch)")
                cached_items = state.collected_params.get("_last_list_cache") or []
                resource_agent = self.resource_agent_map.get(state.resource_type, self.generic_agent)
                # Apply client-side filtering on cached data
                filter_result = await resource_agent.apply_client_side_llm_filter(
                    items=cached_items if isinstance(cached_items, list) else [],
                    user_query=user_query,
                    params=state.collected_params,
                )
                filtered_items = filter_result.get("items", cached_items)
                formatted = await resource_agent.format_response_agentic(
                    operation="list",
                    raw_data=filtered_items,
                    user_query=user_query,
                    context={
                        "query_type": "cached_followup",
                        "original_count": filter_result.get("original_count", len(cached_items) if isinstance(cached_items, list) else 0),
                        "filtered_count": filter_result.get("filtered_count", len(filtered_items) if isinstance(filtered_items, list) else 0),
                        "client_side_filter_applied": bool(filter_result.get("filter_applied")),
                        "cached": True,
                    },
                )
                execution_result = {
                    "success": True,
                    "data": filtered_items,
                    "response": formatted,
                    "metadata": {
                        "query_type": "cached_followup",
                        "cached": True,
                        "client_side_filter_applied": bool(filter_result.get("filter_applied")),
                        "original_count": filter_result.get("original_count"),
                        "count": filter_result.get("filtered_count"),
                    },
                }
                state.set_execution_result(execution_result)
                state.status = ConversationStatus.COMPLETED
                conversation_state_manager.update_session(state)
                return {
                    "agent_name": self.agent_name,
                    "success": True,
                    "output": formatted,
                    "execution_result": execution_result,
                    "metadata": execution_result.get("metadata", {}),
                }
            
            # IPC ENGAGEMENT DERIVATION (per-chat): BU/Env/Zone lists require ipc_engagement_id.
            # Never prompt the user for it; derive from selected_engagement_id and persist in session.
            if state.operation == "list" and state.resource_type in ["business_unit", "environment", "zone"]:
                if not state.collected_params.get("ipc_engagement_id"):
                    if state.selected_engagement_id:
                        try:
                            ipc_id = await api_executor_service.get_ipc_engagement_id(
                                engagement_id=state.selected_engagement_id,
                                user_id=user_id,
                                auth_token=auth_token,
                                force_refresh=False,
                            )
                            if ipc_id:
                                state.collected_params["ipc_engagement_id"] = ipc_id
                                conversation_state_manager.update_session(state)
                                logger.info(f"✅ Derived & persisted ipc_engagement_id={ipc_id} for {state.resource_type} list")
                            else:
                                return {
                                    "agent_name": self.agent_name,
                                    "success": False,
                                    "error": "Failed to derive IPC engagement ID",
                                    "output": (
                                        "I couldn't derive the internal IPC engagement context from your selected engagement.\n\n"
                                        "Please retry. If it still fails, re-select the engagement (or refresh your login token)."
                                    ),
                                }
                        except Exception as e:
                            logger.error(f"❌ IPC derivation failed: {e}")
                            return {
                                "agent_name": self.agent_name,
                                "success": False,
                                "error": "IPC engagement derivation error",
                                "output": (
                                    "I hit an error while deriving IPC engagement context needed for this request.\n\n"
                                    "Please retry, or re-select the engagement."
                                ),
                            }

            # ENGAGEMENT CHECK: For operations that need IPC engagement ID, ensure we have a selected engagement
            # This applies to ALL resource operations that call APIs
            # - CUS users: Auto-select their engagement (they typically have one)
            # - ENG users: Must select from multiple engagements
            resources_needing_engagement = [
                "k8s_cluster", "vm", "firewall", "kafka", "gitlab", "jenkins", 
                "postgres", "documentdb", "container_registry", "managed_service", 
                "load_balancer", "lb", "cluster", "virtual_machine", "endpoint",
                "business_unit", "zone", "environment"
            ]
            
            user_type = state.user_type or context.get("user_type") if context else None
            
            if state.resource_type in resources_needing_engagement and not state.selected_engagement_id:
                logger.info(f"🏢 Resource {state.resource_type} needs engagement ID - user_type={user_type}")
                
                # Fetch engagements
                engagements = await api_executor_service.get_engagements_list(auth_token=auth_token, user_id=user_id)
                
                if not engagements:
                    logger.error("❌ No engagements found for user")
                    return {
                        "agent_name": self.agent_name,
                        "success": False,
                        "error": "No engagements found",
                        "output": "I couldn't find any engagements for your account. Please contact support."
                    }
                
                # CUS users: Always auto-select (they typically have one engagement assigned)
                if user_type == "CUS":
                    single_eng = engagements[0]
                    state.selected_engagement_id = single_eng.get("id")
                    logger.info(f"✅ CUS user - auto-selected engagement: {state.selected_engagement_id}")
                    # Also set in api_executor_service for API calls
                    api_executor_service.set_engagement_id(session_id, state.selected_engagement_id)
                    conversation_state_manager.update_session(state)
                
                # Multiple engagements AND not CUS → Must select (ENG or unknown user_type)
                # This is the safe default - if we don't know the user type, ask them to select
                elif len(engagements) > 1:
                    logger.info(f"🔄 User has {len(engagements)} engagements (user_type={user_type}) - prompting for selection")
                    
                    # Store engagements in state
                    state.pending_engagements = engagements
                    state.status = ConversationStatus.AWAITING_ENGAGEMENT_SELECTION
                    conversation_state_manager.update_session(state)
                    
                    # Format the selection prompt
                    prompt = "Before I can proceed, please select an engagement to work with:\n\n"
                    for i, eng in enumerate(engagements, 1):
                        eng_name = eng.get("engagementName") or eng.get("name", "Unknown")
                        eng_id = eng.get("id")
                        prompt += f"{i}. {eng_name} (ID: {eng_id})\n"
                    prompt += "\nYou can say the number, name, or ID. You can change this later by saying 'switch engagement'."
                    
                    return {
                        "agent_name": self.agent_name,
                        "success": True,
                        "output": prompt,
                        "engagement_selection_required": True
                    }
                
                # Single engagement - auto-select (any user type)
                else:
                    single_eng = engagements[0]
                    state.selected_engagement_id = single_eng.get("id")
                    logger.info(f"✅ Auto-selected single engagement: {state.selected_engagement_id} (user_type={user_type})")
                    # Also set in api_executor_service for API calls
                    api_executor_service.set_engagement_id(session_id, state.selected_engagement_id)
                    conversation_state_manager.update_session(state)
            
            # Check for multi-resource requests (e.g., "gitlab, kafka")
            if state.resource_type and ("," in state.resource_type or " and " in state.resource_type.lower()):
                logger.info(f"🔀 Multi-resource request detected: {state.resource_type}")
                return await self._execute_multi_resource(state, user_roles, session_id, auth_token)
            # Execute via resource agent (single resource)
            execution_result = await self._execute_via_resource_agent(
                resource_type=state.resource_type,
                operation=state.operation,
                params=state.collected_params,
                context={
                    "session_id": session_id,
                    "user_id": user_id,
                    "user_query": user_query,
                    "user_roles": user_roles,
                    "auth_token": auth_token,
                    "user_type": state.user_type,
                    "selected_engagement_id": state.selected_engagement_id
                }
            )

            # Cache last successful list data for "from above response" follow-ups
            if execution_result.get("success") and state.operation == "list":
                data = execution_result.get("data")
                if isinstance(data, list):
                    # Prevent DB bloat: cap cache size
                    capped = data[:200]
                    state.collected_params["_last_list_cache"] = capped
                    state.collected_params["_last_list_resource_type"] = state.resource_type
                    state.collected_params["_last_list_endpoints"] = state.collected_params.get("endpoints")
                    state.collected_params["_last_list_engagement_id"] = state.selected_engagement_id
                    conversation_state_manager.update_session(state)
                    logger.info(f"💾 Cached last list: resource={state.resource_type}, items={len(capped)} (capped)")
            
            # Check if this is a filter selection response (needs user to select BU/Env/Zone)
            if execution_result.get("set_filter_state"):
                logger.info(f"🔄 Filter selection required - storing options in state")
                
                # Store filter options in state for later retrieval
                state.pending_filter_options = execution_result.get("filter_options_for_state", [])
                state.pending_filter_type = execution_result.get("filter_type_for_state", "bu")
                state.status = ConversationStatus.AWAITING_FILTER_SELECTION
                
                # Persist state
                conversation_state_manager.update_session(state)
                
                return {
                    "agent_name": self.agent_name,
                    "success": True,
                    "output": execution_result.get("response", ""),
                    "execution_result": execution_result,
                    "metadata": execution_result.get("metadata", {})
                }
            # Update conversation state with result
            state.set_execution_result(execution_result)
            # Update status based on result
            if execution_result.get("success"):
                state.status = ConversationStatus.COMPLETED
                logger.info(f"✅ Execution successful: {state.operation} {state.resource_type}")
            else:
                state.status = ConversationStatus.FAILED
                logger.error(f"❌ Execution failed: {execution_result.get('error')}")
            # Get formatted response from resource agent
            response = execution_result.get("response", "")
            if not response:
                # Fallback formatting
                if execution_result.get("success"):
                    response = self._format_success_message(state, execution_result)
                else:
                    response = self._format_error_message(state, execution_result)
            return {
                "agent_name": self.agent_name,
                "success": execution_result.get("success", False),
                "output": response,
                "execution_result": execution_result,
                "metadata": execution_result.get("metadata", {})}
        except Exception as e:
            logger.error(f"❌ Execution agent failed: {str(e)}", exc_info=True)
            return {
                "agent_name": self.agent_name,
                "success": False,
                "error": str(e),
                "output": f"I encountered an error while executing the operation: {str(e)}"}
    # ========================================================================
    # MULTI-RESOURCE: Execute Multiple Resources in Parallel
    # =========================================================================
    async def _execute_multi_resource(self,state: ConversationState,user_roles: List[str],session_id: str,auth_token: str = None) -> Dict[str, Any]:
        """
        Execute operations on multiple resource types in parallel.
        PRODUCTION: Supports load balancers in multi-resource queries. 
        Args:
            state: Conversation state with comma-separated resource_type
            user_roles: User roles for permission checking
            session_id: Session identifier
            auth_token: Bearer token from UI (Keycloak) for API authentication
        Returns:
            Combined execution result
        """
        import asyncio
        # Parse resource types
        resource_type_str = state.resource_type or ""
        if "," in resource_type_str:
            resource_types = [r.strip() for r in resource_type_str.split(",")]
        elif " and " in resource_type_str.lower():
            resource_types = [r.strip() for r in resource_type_str.lower().split(" and ")]
        else:
            resource_types = [resource_type_str.strip()]
        # Remove empty strings and duplicates
        resource_types = list(set([rt for rt in resource_types if rt]))
        logger.info(f"🔀 Executing {len(resource_types)} resource operations in parallel: {resource_types}")
        endpoint_names = state.collected_params.get("endpoints") or state.collected_params.get("endpoint_ids")
        if endpoint_names and isinstance(endpoint_names, list):
            needs_conversion = any(rt not in ["load_balancer", "lb", "loadbalancer"] for rt in resource_types)
            if needs_conversion and any(isinstance(e, str) and not e.isdigit() for e in endpoint_names):
                converted_ids, error = await self._convert_endpoint_names_to_ids(endpoint_names)
                if error and not converted_ids:
                    return {
                        "agent_name": self.agent_name,
                        "success": False,
                        "output": f"❌ {error}",
                        "execution_result": {"success": False, "error": error}}
                if converted_ids:
                    state.collected_params["endpoints"] = converted_ids
                    state.collected_params["endpoint_names"] = endpoint_names
        tasks = []
        for resource_type in resource_types:
            resource_agent = self.resource_agent_map.get(resource_type, self.generic_agent)
            logger.info(f"  📦 Adding {resource_type} to execution queue -> {resource_agent.agent_name}")
            
            task = resource_agent.execute_operation(
                operation=state.operation,
                params=state.collected_params.copy(),
                context={
                    "session_id": session_id,
                    "user_id": state.user_id,
                    "user_query": state.user_query,
                    "user_roles": user_roles,
                    "resource_type": resource_type,
                    "auth_token": auth_token})
            tasks.append((resource_type, task))
        if not tasks:
            return {
                "agent_name": self.agent_name,
                "success": False,
                "output": f"❌ No handlers found for: {', '.join(resource_types)}",
                "execution_result": {}}
        # Execute all tasks in parallel
        logger.info(f"⚡ Executing {len(tasks)} tasks in parallel...")
        results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
        # Collect results
        combined_results = {}
        combined_data = []
        all_success = True
        error_messages = []
        for (resource_type, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                logger.error(f"❌ {resource_type} failed with exception: {result}")
                all_success = False
                error_messages.append(f"{resource_type}: {str(result)}")
            else:
                combined_results[resource_type] = result
                if result.get("success"):
                    logger.info(f"✅ {resource_type} completed successfully")
                    resource_data = result.get("data", [])
                    if isinstance(resource_data, list):
                        combined_data.extend([{**item, "_resource_type": resource_type} for item in resource_data])
                else:
                    logger.error(f"❌ {resource_type} failed: {result.get('error')}")
                    all_success = False
                    error_messages.append(f"{resource_type}: {result.get('error', 'Unknown error')}")

        if all_success or any(r.get("success") for r in combined_results.values()):
            state.status = ConversationStatus.COMPLETED
            # Combine responses from successful results
            combined_responses = []
            total_count = 0
            for resource_type, result in combined_results.items():
                if result.get("success"):
                    response_text = result.get("response", "")
                    count = result.get("metadata", {}).get("count", 0)
                    total_count += count
                    combined_responses.append(f"## {resource_type.replace('_', ' ').title()}\n{response_text}")
            # Add error summary if partial failure
            if error_messages:
                combined_responses.append(f"\n⚠️ **Some resources failed:**\n" + "\n".join(f"- {err}" for err in error_messages))
            final_response = "\n\n---\n\n".join(combined_responses)
            return {
                "agent_name": self.agent_name,
                "success": True,
                "output": final_response,
                "execution_result": {
                    "success": True,
                    "data": combined_data,
                    "multi_resource": True,
                    "resource_types": resource_types,
                    "individual_results": combined_results,
                    "total_items": total_count
                },
                "metadata": {
                    "resource_types": resource_types,
                    "operation": state.operation,
                    "total_items": total_count,
                    "multi_resource_execution": True
                }
            }
        else:
            state.status = ConversationStatus.FAILED
            error_summary = f"❌ I encountered errors while fetching {', '.join(resource_types)}:\n\n"
            error_summary += "\n".join(f"- **{err}**" for err in error_messages)
            return {
                "agent_name": self.agent_name,
                "success": False,
                "output": error_summary,
                "execution_result": {
                    "success": False,
                    "errors": error_messages,
                    "partial_results": combined_results}}
    # =========================================================================
    # FORMATTING: Fallback Message Formatters
    # =========================================================================
    def _format_success_message(self,state: ConversationState,execution_result: Dict[str, Any]) -> str:
        """
        Fallback success message formatter.
        Used when resource agent doesn't provide a formatted response.
        """
        operation_verb = {
            "create": "created",
            "update": "updated",
            "delete": "deleted",
            "list": "retrieved",
            "read": "retrieved",
            "get_details": "retrieved details for",
            "get_virtual_services": "retrieved virtual services for"
        }.get(state.operation, "processed")
        resource_name = state.resource_type.replace("_", " ")
        message = f"✅ Successfully {operation_verb} {resource_name}!\n\n"
        result_data = execution_result.get("data", {})
        if isinstance(result_data, list):
            message += f"📊 Found {len(result_data)} item(s).\n"
        elif isinstance(result_data, dict):
            message += "**Details:**\n"
            for key, value in list(result_data.items())[:5]:
                if not key.startswith("_"):
                    message += f"- {key}: {value}\n"
        return message
    
    def _format_error_message(self,state: ConversationState,execution_result: Dict[str, Any]) -> str:
        """
        Fallback error message formatter.
        Used when resource agent doesn't provide a formatted response.
        """
        error = execution_result.get("error", "Unknown error occurred")
        resource_name = state.resource_type.replace("_", " ") if state.resource_type else "resource"
        message = f"❌ I couldn't complete the {state.operation} operation on {resource_name}.\n\n"
        message += f"**Error:** {error}\n\n"
        # Add helpful suggestions based on error type
        error_lower = error.lower()
        if "permission" in error_lower or "denied" in error_lower:
            message += "💡 Please contact your administrator to request access."
        elif "not found" in error_lower:
            message += "💡 Please check the resource name/ID and try again."
        elif "timeout" in error_lower or "connection" in error_lower:
            message += "💡 Please try again in a moment."
        else:
            message += "💡 Please check your parameters and try again."
        return message