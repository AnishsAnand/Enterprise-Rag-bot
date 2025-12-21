"""
Execution Agent - Executes validated CRUD operations via API calls.
Handles actual execution of operations and provides feedback to users.
"""

from typing import Any, Dict, List, Optional
from langchain.tools import Tool
import logging
import json

from app.agents.base_agent import BaseAgent
from app.agents.state.conversation_state import conversation_state_manager, ConversationStatus, ConversationState
from app.services.api_executor_service import api_executor_service
from app.services.llm_formatter_service import llm_formatter

# Import specialized resource agents
from app.agents.resource_agents.k8s_cluster_agent import K8sClusterAgent
from app.agents.resource_agents.managed_services_agent import ManagedServicesAgent
from app.agents.resource_agents.virtual_machine_agent import VirtualMachineAgent
from app.agents.resource_agents.network_agent import NetworkAgent

logger = logging.getLogger(__name__)


class ExecutionAgent(BaseAgent):
    """
    Agent specialized in executing validated CRUD operations.
    Makes API calls and provides user-friendly feedback on results.
    """
    
    def __init__(self):
        super().__init__(
            agent_name="ExecutionAgent",
            agent_description=(
                "Executes validated CRUD operations on cloud resources. "
                "Routes to specialized resource agents for domain-specific handling."
            ),
            temperature=0.3
        )
        
        # Initialize specialized resource agents
        self.k8s_agent = K8sClusterAgent()
        self.managed_services_agent = ManagedServicesAgent()
        self.vm_agent = VirtualMachineAgent()
        self.network_agent = NetworkAgent()
        
        # Resource type to agent mapping
        self.resource_agent_map = {
            "k8s_cluster": self.k8s_agent,
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
            "load_balancer": self.network_agent
        }
        
        logger.info(f"✅ ExecutionAgent initialized with {len(self.resource_agent_map)} resource agent mappings")
        
        # Setup agent
        self.setup_agent()
    
    def get_system_prompt(self) -> str:
        """Return system prompt for execution agent."""
        return """You are the Execution Agent, responsible for executing validated operations on cloud resources.

**Your responsibilities:**
1. **Execute API calls** for CRUD operations (create, list, update, delete)
2. **Handle execution results** (success and errors)
3. **Provide clear feedback** to users about what happened
4. **Format responses** in a user-friendly way
5. **Handle errors gracefully** with helpful messages

**Special Operations:**

For listing Kubernetes clusters:
- Use `list_k8s_clusters` tool to get clusters across all or specific endpoints
- First, you can optionally call `get_available_endpoints` to show user available data centers
- If user asks for "all clusters" or "list clusters", use `list_k8s_clusters` with no parameters
- If user specifies locations like "Mumbai" or "Delhi", map them to endpoint IDs and pass to the tool
- The system automatically handles authentication and engagement ID fetching

**When reporting success:**
- Confirm what was done
- Provide key details (resource ID, name, etc.)
- Mention next steps if relevant
- Be positive and clear

**When reporting errors:**
- Explain what went wrong in simple terms
- Suggest how to fix the issue
- Don't expose technical stack traces to users
- Offer to try again or alternative approaches

**Example responses:**

Success (create):
"✅ Great news! I've successfully created your Kubernetes cluster 'prod-cluster-01'.

**Cluster Details:**
- Name: prod-cluster-01
- Region: us-east-1
- Status: Provisioning
- Cluster ID: cls-abc123

The cluster is now being provisioned. It should be ready in about 10-15 minutes. 
You can check its status anytime by asking me 'show cluster prod-cluster-01'."

Success (delete):
"✅ The firewall rule 'allow-http' has been successfully deleted. 
The changes will take effect immediately."

Error (permission denied):
"❌ I wasn't able to create the cluster because you don't have the required permissions. 
This operation requires 'admin' or 'developer' role. Please contact your administrator 
if you need access."

Error (validation failed):
"❌ The operation couldn't be completed because the cluster name 'prod_cluster' 
contains invalid characters. Cluster names must use only lowercase letters, 
numbers, and hyphens. Would you like to try with a different name?"

Be professional, helpful, and always provide actionable information."""
    
    def get_tools(self) -> List[Tool]:
        """Return tools for execution agent."""
        return [
            Tool(
                name="execute_api_operation",
                func=self._execute_api_operation,
                description=(
                    "Execute a CRUD operation via API. "
                    "Input: JSON with resource_type, operation, params, and user_roles"
                )
            ),
            Tool(
                name="list_k8s_clusters",
                func=self._list_k8s_clusters,
                description=(
                    "List Kubernetes clusters across endpoints. "
                    "Input: JSON with optional endpoint_ids (list of integers) and engagement_id. "
                    "If not provided, lists all clusters across all available endpoints. "
                    "Example: {\"endpoint_ids\": [11, 12]} or {} for all"
                )
            ),
            Tool(
                name="get_available_endpoints",
                func=self._get_available_endpoints,
                description=(
                    "Get available data center endpoints for cluster operations. "
                    "Returns list of endpoints with IDs and names. Input: empty dict {}"
                )
            ),
            Tool(
                name="format_success_response",
                func=self._format_success_response,
                description=(
                    "Format a success response for the user. "
                    "Input: JSON with operation, resource_type, and result_data"
                )
            ),
            Tool(
                name="format_error_response",
                func=self._format_error_response,
                description=(
                    "Format an error response for the user. "
                    "Input: JSON with error message and context"
                )
            )
        ]
    
    def _execute_api_operation(self, input_json: str) -> str:
        """Execute API operation."""
        try:
            data = json.loads(input_json)
            resource_type = data.get("resource_type")
            operation = data.get("operation")
            params = data.get("params", {})
            user_roles = data.get("user_roles", [])
            user_id = data.get("user_id")  # Get user_id from context if available
            
            # Execute operation
            import asyncio
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(
                api_executor_service.execute_operation(
                    resource_type=resource_type,
                    operation=operation,
                    params=params,
                    user_roles=user_roles,
                    dry_run=False,
                    user_id=user_id  # Pass user_id to retrieve credentials
                )
            )
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})
    
    def _list_k8s_clusters(self, input_json: str) -> str:
        """
        List Kubernetes clusters across endpoints.
        Uses the multi-step workflow: engagement -> endpoints -> clusters.
        """
        try:
            logger.info(f"🎯 ExecutionAgent._list_k8s_clusters called with input: {input_json}")
            data = json.loads(input_json) if input_json and input_json != "{}" else {}
            endpoint_ids = data.get("endpoint_ids")
            engagement_id = data.get("engagement_id")
            
            logger.info(f"📊 Calling api_executor_service.list_clusters(endpoint_ids={endpoint_ids}, engagement_id={engagement_id})")
            
            # Execute cluster listing workflow
            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                # No event loop in current thread, create a new one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            result = loop.run_until_complete(
                api_executor_service.list_clusters(
                    endpoint_ids=endpoint_ids,
                    engagement_id=engagement_id
                )
            )
            
            logger.info(f"📊 api_executor_service.list_clusters returned: success={result.get('success')}")
            
            # Format the response for better readability
            if result.get("success") and result.get("data"):
                data = result["data"]
                if isinstance(data, dict) and "data" in data:
                    clusters = data["data"]
                    
                    # Group by endpoint
                    by_endpoint = {}
                    for cluster in clusters:
                        endpoint = cluster.get("displayNameEndpoint", "Unknown")
                        if endpoint not in by_endpoint:
                            by_endpoint[endpoint] = []
                        by_endpoint[endpoint].append(cluster)
                    
                    # Create summary
                    summary = {
                        "success": True,
                        "total_clusters": len(clusters),
                        "endpoints": len(by_endpoint),
                        "clusters_by_endpoint": {
                            endpoint: len(endpoint_clusters)
                            for endpoint, endpoint_clusters in by_endpoint.items()
                        },
                        "clusters": clusters[:20],  # Return first 20 for details
                        "message": f"Found {len(clusters)} clusters across {len(by_endpoint)} endpoints"
                    }
                    return json.dumps(summary, indent=2)
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            logger.error(f"❌ Failed to list clusters: {str(e)}")
            return json.dumps({"success": False, "error": str(e)})
    
    def _get_available_endpoints(self, input_json: str) -> str:
        """Get available endpoints (data centers)."""
        try:
            # Fetch endpoints
            import asyncio
            loop = asyncio.get_event_loop()
            endpoints = loop.run_until_complete(
                api_executor_service.get_endpoints()
            )
            
            if endpoints:
                result = {
                    "success": True,
                    "endpoints": endpoints,
                    "total": len(endpoints),
                    "message": f"Found {len(endpoints)} available endpoints"
                }
                return json.dumps(result, indent=2)
            else:
                return json.dumps({
                    "success": False,
                    "error": "Failed to fetch endpoints"
                })
            
        except Exception as e:
            logger.error(f"❌ Failed to get endpoints: {str(e)}")
            return json.dumps({"success": False, "error": str(e)})
    
    def _format_success_response(self, input_json: str) -> str:
        """Format success response."""
        try:
            data = json.loads(input_json)
            operation = data.get("operation", "operation")
            resource_type = data.get("resource_type", "resource")
            result_data = data.get("result_data", {})
            
            response = f"✅ Successfully completed {operation} on {resource_type}.\n\n"
            
            if result_data:
                response += "**Details:**\n"
                for key, value in result_data.items():
                    response += f"- {key}: {value}\n"
            
            return response
            
        except Exception as e:
            return f"Operation completed successfully. (Error formatting response: {str(e)})"
    
    def _format_error_response(self, input_json: str) -> str:
        """Format error response."""
        try:
            data = json.loads(input_json)
            error = data.get("error", "Unknown error")
            context = data.get("context", {})
            
            response = f"❌ I encountered an issue: {error}\n\n"
            
            if context:
                response += "**Context:**\n"
                for key, value in context.items():
                    response += f"- {key}: {value}\n"
            
            return response
            
        except Exception as e:
            return f"An error occurred: {str(e)}"
    
    
    def _map_endpoint_name_to_display(self, endpoint_name: str) -> str:
        """
        Map technical endpoint name to display name.
        
        Args:
            endpoint_name: Technical name like "EP_V2_BL"
            
        Returns:
            Display name like "Bengaluru"
        """
        endpoint_map = {
            "EP_V2_BL": "Bengaluru",
            "EP_V2_DEL": "Delhi",
            "EP_V2_CHN_AMB": "Chennai-AMB",
            "EP_V2_MUM_BKC": "Mumbai-BKC",
            "EP_V2_MUM_DC3": "Mumbai-DC3",
            "EP_GCC_DEL": "GCCDelhi",
            "EP_GCC_MUM": "GCCMumbai",
            "EP_V2_UKCX": "Cressex",
            "EP_V2_UKHB": "Highbridge",
            "EP_V2_SG_TCX": "Singapore East"
        }
        return endpoint_map.get(endpoint_name, endpoint_name)
    
    async def _build_cluster_create_payload(self, state: Any) -> Dict[str, Any]:
        """
        Build the complete payload for cluster creation (customer version).
        
        Based on createcluster.ts lines 3994-4042.
        
        Args:
            state: Conversation state with collected params
            
        Returns:
            Complete payload dict for API
        """
        params = state.collected_params
        logger.info(f"🔧 Building payload from collected params: {list(params.keys())}")
        
        # Get engagement ID and circuit ID
        engagement_id = await api_executor_service.get_engagement_id()
        circuit_id = params.get("_circuit_id") or await api_executor_service.get_circuit_id(engagement_id)
        
        # Extract OS and flavor details
        os_info = params["operatingSystem"]
        flavor_info = params["flavor"]
        zone_id = params["_zone_id"]
        
        # Build master node config (customer default: 3x D8)
        master_node = {
            "vmHostName": "",
            "vmFlavor": "D8",
            "skuCode": "D8.UBN",
            "nodeType": "Master",
            "replicaCount": 3,
            "maxReplicaCount": None,
            "additionalDisk": {},
            "labelsNTaints": "no"
        }
        
        # Build worker node config
        worker_node = {
            "vmHostName": params["workerPoolName"],
            "vmFlavor": flavor_info["flavor_name"],
            "skuCode": flavor_info["sku_code"],
            "nodeType": "Worker",
            "replicaCount": params["replicaCount"],
            "maxReplicaCount": params.get("maxReplicas"),
            "additionalDisk": {},
            "labelsNTaints": "no"
        }
        
        # Build vmSpecificInput array (master + workers)
        vm_specific_input = [master_node, worker_node]
        
        # Build imageDetails
        image_details = {
            "valueOSModel": os_info["os_model"],
            "valueOSMake": os_info["os_make"],
            "valueOSVersion": os_info["os_version"],
            "valueOSServicePack": None
        }
        
        # Build networking driver (optional)
        networking_driver = []
        if params.get("cniDriver"):
            networking_driver = [{"name": params["cniDriver"]}]
        
        # Build tags (optional)
        tags = params.get("tags", [])
        
        # Construct final payload
        payload = {
            "name": "",  # Empty string as per UI
            "hypervisor": os_info.get("hypervisor", "VCD_ESXI"),
            "purpose": "ipc",
            "vmPurpose": "",  # Empty for customer
            "imageId": os_info["os_id"],
            "zoneId": zone_id,
            "alertSuppression": True,
            "iops": 1,
            "isKdumpOrPageEnabled": "No",
            "applicationType": "Container",
            "application": "Containers",
            "vmSpecificInput": vm_specific_input,
            "clusterMode": "High availability",
            "dedicatedDeployment": False,
            "clusterName": params["clusterName"],
            "k8sVersion": params["k8sVersion"],
            "circuitId": circuit_id,
            "vApp": "",
            "imageDetails": image_details
        }
        
        # Add optional fields
        if networking_driver:
            payload["networkingDriver"] = networking_driver
        
        if tags:
            payload["tags"] = tags
        
        logger.info(f"✅ Built complete payload for cluster: {payload['clusterName']}")
        logger.debug(f"📦 Full payload: {json.dumps(payload, indent=2)}")
        
        return payload
    
    async def execute(
        self,
        input_text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute the operation from conversation state.
        
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
            logger.info(f"📌 Session ID: {session_id}")
            state = conversation_state_manager.get_session(session_id) if session_id else None
            
            if not state:
                logger.error("❌ No conversation state found!")
                return {
                    "agent_name": self.agent_name,
                    "success": False,
                    "error": "No conversation state found",
                    "output": "I couldn't find the operation to execute."
                }
            
            logger.info(f"📌 State: resource={state.resource_type}, operation={state.operation}, ready={state.is_ready_to_execute()}, status={state.status}")
            
            # Check if ready to execute
            if not state.is_ready_to_execute():
                logger.warning(f"⚠️ Not ready to execute! Missing: {state.missing_params}")
                return {
                    "agent_name": self.agent_name,
                    "success": False,
                    "error": "Not ready to execute",
                    "output": state.get_missing_params_message()
                }
            
            # Get user roles and user_id for permission checking and credential retrieval
            user_roles = context.get("user_roles", []) if context else []
            user_id = context.get("user_id") if context else None
            
            # Execute the operation
            logger.info(
                f"🚀 Executing {state.operation} on {state.resource_type} "
                f"with params: {list(state.collected_params.keys())}"
            )
            
            # 🆕 CHECK FOR MULTI-RESOURCE REQUESTS (e.g., "gitlab, kafka")
            if state.resource_type and ("," in state.resource_type or " and " in state.resource_type.lower()):
                logger.info(f"🔀 Multi-resource request detected: {state.resource_type}")
                return await self._execute_multi_resource(state, user_roles, session_id)
            
            # 🆕 NEW ROUTING LOGIC: Check if we have a specialized resource agent
            resource_agent = self.resource_agent_map.get(state.resource_type)
            
            if resource_agent:
                # Route to specialized resource agent for intelligent handling
                logger.info(f"🎯 Routing to {resource_agent.agent_name} for {state.resource_type}")
                
                agent_result = await resource_agent.execute_operation(
                    operation=state.operation,
                    params=state.collected_params,
                    context={
                        "session_id": session_id,
                        "user_id": state.user_id,
                        "user_query": state.user_query,
                        "user_roles": user_roles,
                        "resource_type": state.resource_type
                    }
                )
                
                if agent_result.get("success"):
                    state.status = ConversationStatus.COMPLETED
                    logger.info(f"✅ {resource_agent.agent_name} completed successfully")
                else:
                    state.status = ConversationStatus.FAILED
                    logger.error(f"❌ {resource_agent.agent_name} failed: {agent_result.get('error')}")
                
                return {
                    "agent_name": self.agent_name,
                    "success": agent_result.get("success"),
                    "output": agent_result.get("response", ""),
                    "execution_result": agent_result,
                    "metadata": {
                        "routed_to": resource_agent.agent_name,
                        "resource_type": state.resource_type,
                        "operation": state.operation
                    }
                }
            
            # 🔄 FALLBACK: Traditional execution logic (if no resource agent)
            logger.info(f"⚠️ No specialized agent for {state.resource_type}, using traditional execution")
            
            # Special handling for endpoint listing - use the list_endpoints workflow method
            if state.resource_type == "endpoint" and state.operation == "list":
                logger.info("📋 Using list_endpoints workflow method")
                execution_result = await api_executor_service.list_endpoints()
            
            # Special handling for cluster listing - use the full workflow method
            elif state.resource_type == "k8s_cluster" and state.operation == "list":
                logger.info("📋 Using list_clusters workflow method")
                endpoint_ids = state.collected_params.get("endpoints") or state.collected_params.get("endpoint_ids")
                
                # IMPORTANT: Convert endpoint names to IDs if needed
                # The IntentAgent might extract ["Delhi", "Chennai"] but API needs [11, 204]
                conversion_error = None
                if endpoint_ids and isinstance(endpoint_ids, list) and len(endpoint_ids) > 0:
                    if isinstance(endpoint_ids[0], str) and not endpoint_ids[0].isdigit():
                        # We have names, need to convert to IDs
                        logger.info(f"🔄 Converting endpoint names {endpoint_ids} to IDs...")
                        try:
                            # Fetch available endpoints
                            endpoints_result = await api_executor_service.list_endpoints()
                            if endpoints_result.get("success"):
                                available_endpoints = endpoints_result.get("data", {}).get("endpoints", [])
                                
                                # Build name -> ID mapping
                                name_to_id = {}
                                for ep in available_endpoints:
                                    ep_name = ep.get("name", "").strip()
                                    ep_id = ep.get("id")
                                    if ep_name and ep_id:
                                        # Add exact match
                                        name_to_id[ep_name.lower()] = ep_id
                                        # Add without hyphens/spaces for fuzzy matching
                                        name_to_id[ep_name.lower().replace("-", "").replace(" ", "")] = ep_id
                                
                                # Convert names to IDs
                                converted_ids = []
                                for name in endpoint_ids:
                                    name_clean = name.lower().strip().replace("-", "").replace(" ", "")
                                    if name_clean in name_to_id:
                                        converted_ids.append(name_to_id[name_clean])
                                        logger.info(f"  ✅ '{name}' -> ID {name_to_id[name_clean]}")
                                    else:
                                        logger.warning(f"  ⚠️ Could not find ID for endpoint '{name}'")
                                
                                if converted_ids:
                                    endpoint_ids = converted_ids
                                    logger.info(f"✅ Converted to IDs: {endpoint_ids}")
                                else:
                                    conversion_error = f"Could not find endpoint IDs for: {', '.join(endpoint_ids)}"
                                    logger.error(f"❌ {conversion_error}")
                            else:
                                conversion_error = "Failed to fetch endpoints for name-to-ID conversion"
                                logger.error(f"❌ {conversion_error}")
                        except Exception as e:
                            conversion_error = f"Error converting endpoint names to IDs: {str(e)}"
                            logger.error(f"❌ {conversion_error}")
                
                # Call API if no conversion error
                if conversion_error:
                    execution_result = {
                        "success": False,
                        "error": conversion_error
                    }
                else:
                    execution_result = await api_executor_service.list_clusters(
                        endpoint_ids=endpoint_ids,
                        engagement_id=None  # Will be fetched automatically
                    )
            
            # Special handling for Kafka listing
            elif state.resource_type == "kafka" and state.operation == "list":
                logger.info("📋 Using list_kafka workflow method")
                endpoint_ids = state.collected_params.get("endpoints") or state.collected_params.get("endpoint_ids")
                
                # Convert endpoint names to IDs if needed (same logic as clusters)
                conversion_error = None
                if endpoint_ids and isinstance(endpoint_ids, list) and len(endpoint_ids) > 0:
                    if isinstance(endpoint_ids[0], str) and not endpoint_ids[0].isdigit():
                        logger.info(f"🔄 Converting endpoint names {endpoint_ids} to IDs...")
                        try:
                            endpoints_result = await api_executor_service.list_endpoints()
                            if endpoints_result.get("success"):
                                available_endpoints = endpoints_result.get("data", {}).get("endpoints", [])
                                name_to_id = {}
                                for ep in available_endpoints:
                                    ep_name = ep.get("name", "").strip()
                                    ep_id = ep.get("id")
                                    if ep_name and ep_id:
                                        name_to_id[ep_name.lower()] = ep_id
                                        name_to_id[ep_name.lower().replace("-", "").replace(" ", "")] = ep_id
                                
                                converted_ids = []
                                for name in endpoint_ids:
                                    name_clean = name.lower().strip().replace("-", "").replace(" ", "")
                                    if name_clean in name_to_id:
                                        converted_ids.append(name_to_id[name_clean])
                                        logger.info(f"  ✅ '{name}' -> ID {name_to_id[name_clean]}")
                                
                                if converted_ids:
                                    endpoint_ids = converted_ids
                                    logger.info(f"✅ Converted to IDs: {endpoint_ids}")
                                else:
                                    conversion_error = f"Could not find endpoint IDs for: {', '.join(endpoint_ids)}"
                            else:
                                conversion_error = "Failed to fetch endpoints for name-to-ID conversion"
                        except Exception as e:
                            conversion_error = f"Error converting endpoint names to IDs: {str(e)}"
                
                if conversion_error:
                    execution_result = {"success": False, "error": conversion_error}
                else:
                    execution_result = await api_executor_service.list_kafka(
                        endpoint_ids=endpoint_ids,
                        ipc_engagement_id=None  # Will be fetched and converted automatically
                    )
            
            # Special handling for GitLab listing
            elif state.resource_type == "gitlab" and state.operation == "list":
                logger.info("📋 Using list_gitlab workflow method")
                endpoint_ids = state.collected_params.get("endpoints") or state.collected_params.get("endpoint_ids")
                
                # Convert endpoint names to IDs if needed (same logic as clusters)
                conversion_error = None
                if endpoint_ids and isinstance(endpoint_ids, list) and len(endpoint_ids) > 0:
                    if isinstance(endpoint_ids[0], str) and not endpoint_ids[0].isdigit():
                        logger.info(f"🔄 Converting endpoint names {endpoint_ids} to IDs...")
                        try:
                            endpoints_result = await api_executor_service.list_endpoints()
                            if endpoints_result.get("success"):
                                available_endpoints = endpoints_result.get("data", {}).get("endpoints", [])
                                name_to_id = {}
                                for ep in available_endpoints:
                                    ep_name = ep.get("name", "").strip()
                                    ep_id = ep.get("id")
                                    if ep_name and ep_id:
                                        name_to_id[ep_name.lower()] = ep_id
                                        name_to_id[ep_name.lower().replace("-", "").replace(" ", "")] = ep_id
                                
                                converted_ids = []
                                for name in endpoint_ids:
                                    name_clean = name.lower().strip().replace("-", "").replace(" ", "")
                                    if name_clean in name_to_id:
                                        converted_ids.append(name_to_id[name_clean])
                                        logger.info(f"  ✅ '{name}' -> ID {name_to_id[name_clean]}")
                                
                                if converted_ids:
                                    endpoint_ids = converted_ids
                                    logger.info(f"✅ Converted to IDs: {endpoint_ids}")
                                else:
                                    conversion_error = f"Could not find endpoint IDs for: {', '.join(endpoint_ids)}"
                            else:
                                conversion_error = "Failed to fetch endpoints for name-to-ID conversion"
                        except Exception as e:
                            conversion_error = f"Error converting endpoint names to IDs: {str(e)}"
                
                if conversion_error:
                    execution_result = {"success": False, "error": conversion_error}
                else:
                    execution_result = await api_executor_service.list_gitlab(
                        endpoint_ids=endpoint_ids,
                        ipc_engagement_id=None  # Will be fetched and converted automatically
                    )
            
            # Special handling for Container Registry listing
            elif state.resource_type == "container_registry" and state.operation == "list":
                logger.info("📋 Using list_container_registry workflow method")
                endpoint_ids = state.collected_params.get("endpoints") or state.collected_params.get("endpoint_ids")
                
                conversion_error = None
                if endpoint_ids and isinstance(endpoint_ids, list) and len(endpoint_ids) > 0:
                    if isinstance(endpoint_ids[0], str) and not endpoint_ids[0].isdigit():
                        logger.info(f"🔄 Converting endpoint names {endpoint_ids} to IDs...")
                        try:
                            endpoints_result = await api_executor_service.list_endpoints()
                            if endpoints_result.get("success"):
                                available_endpoints = endpoints_result.get("data", {}).get("endpoints", [])
                                name_to_id = {}
                                for ep in available_endpoints:
                                    ep_name = ep.get("name", "").strip()
                                    ep_id = ep.get("id")
                                    if ep_name and ep_id:
                                        name_to_id[ep_name.lower()] = ep_id
                                        name_to_id[ep_name.lower().replace("-", "").replace(" ", "")] = ep_id
                                
                                converted_ids = []
                                for name in endpoint_ids:
                                    name_clean = name.lower().strip().replace("-", "").replace(" ", "")
                                    if name_clean in name_to_id:
                                        converted_ids.append(name_to_id[name_clean])
                                        logger.info(f"  ✅ '{name}' -> ID {name_to_id[name_clean]}")
                                
                                if converted_ids:
                                    endpoint_ids = converted_ids
                                    logger.info(f"✅ Converted to IDs: {endpoint_ids}")
                                else:
                                    conversion_error = f"Could not find endpoint IDs for: {', '.join(endpoint_ids)}"
                            else:
                                conversion_error = "Failed to fetch endpoints for name-to-ID conversion"
                        except Exception as e:
                            conversion_error = f"Error converting endpoint names to IDs: {str(e)}"
                
                if conversion_error:
                    execution_result = {"success": False, "error": conversion_error}
                else:
                    execution_result = await api_executor_service.list_container_registry(
                        endpoint_ids=endpoint_ids,
                        ipc_engagement_id=None
                    )
            
            # Special handling for Jenkins listing
            elif state.resource_type == "jenkins" and state.operation == "list":
                logger.info("📋 Using list_jenkins workflow method")
                endpoint_ids = state.collected_params.get("endpoints") or state.collected_params.get("endpoint_ids")
                
                conversion_error = None
                if endpoint_ids and isinstance(endpoint_ids, list) and len(endpoint_ids) > 0:
                    if isinstance(endpoint_ids[0], str) and not endpoint_ids[0].isdigit():
                        logger.info(f"🔄 Converting endpoint names {endpoint_ids} to IDs...")
                        try:
                            endpoints_result = await api_executor_service.list_endpoints()
                            if endpoints_result.get("success"):
                                available_endpoints = endpoints_result.get("data", {}).get("endpoints", [])
                                name_to_id = {}
                                for ep in available_endpoints:
                                    ep_name = ep.get("name", "").strip()
                                    ep_id = ep.get("id")
                                    if ep_name and ep_id:
                                        name_to_id[ep_name.lower()] = ep_id
                                        name_to_id[ep_name.lower().replace("-", "").replace(" ", "")] = ep_id
                                
                                converted_ids = []
                                for name in endpoint_ids:
                                    name_clean = name.lower().strip().replace("-", "").replace(" ", "")
                                    if name_clean in name_to_id:
                                        converted_ids.append(name_to_id[name_clean])
                                        logger.info(f"  ✅ '{name}' -> ID {name_to_id[name_clean]}")
                                
                                if converted_ids:
                                    endpoint_ids = converted_ids
                                    logger.info(f"✅ Converted to IDs: {endpoint_ids}")
                                else:
                                    conversion_error = f"Could not find endpoint IDs for: {', '.join(endpoint_ids)}"
                            else:
                                conversion_error = "Failed to fetch endpoints for name-to-ID conversion"
                        except Exception as e:
                            conversion_error = f"Error converting endpoint names to IDs: {str(e)}"
                
                if conversion_error:
                    execution_result = {"success": False, "error": conversion_error}
                else:
                    execution_result = await api_executor_service.list_jenkins(
                        endpoint_ids=endpoint_ids,
                        ipc_engagement_id=None
                    )
            
            # Special handling for PostgreSQL listing
            elif state.resource_type == "postgres" and state.operation == "list":
                logger.info("📋 Using list_postgres workflow method")
                endpoint_ids = state.collected_params.get("endpoints") or state.collected_params.get("endpoint_ids")
                
                conversion_error = None
                if endpoint_ids and isinstance(endpoint_ids, list) and len(endpoint_ids) > 0:
                    if isinstance(endpoint_ids[0], str) and not endpoint_ids[0].isdigit():
                        logger.info(f"🔄 Converting endpoint names {endpoint_ids} to IDs...")
                        try:
                            endpoints_result = await api_executor_service.list_endpoints()
                            if endpoints_result.get("success"):
                                available_endpoints = endpoints_result.get("data", {}).get("endpoints", [])
                                name_to_id = {}
                                for ep in available_endpoints:
                                    ep_name = ep.get("name", "").strip()
                                    ep_id = ep.get("id")
                                    if ep_name and ep_id:
                                        name_to_id[ep_name.lower()] = ep_id
                                        name_to_id[ep_name.lower().replace("-", "").replace(" ", "")] = ep_id
                                
                                converted_ids = []
                                for name in endpoint_ids:
                                    name_clean = name.lower().strip().replace("-", "").replace(" ", "")
                                    if name_clean in name_to_id:
                                        converted_ids.append(name_to_id[name_clean])
                                        logger.info(f"  ✅ '{name}' -> ID {name_to_id[name_clean]}")
                                
                                if converted_ids:
                                    endpoint_ids = converted_ids
                                    logger.info(f"✅ Converted to IDs: {endpoint_ids}")
                                else:
                                    conversion_error = f"Could not find endpoint IDs for: {', '.join(endpoint_ids)}"
                            else:
                                conversion_error = "Failed to fetch endpoints for name-to-ID conversion"
                        except Exception as e:
                            conversion_error = f"Error converting endpoint names to IDs: {str(e)}"
                
                if conversion_error:
                    execution_result = {"success": False, "error": conversion_error}
                else:
                    execution_result = await api_executor_service.list_postgres(
                        endpoint_ids=endpoint_ids,
                        ipc_engagement_id=None
                    )
            
            # Special handling for DocumentDB listing
            elif state.resource_type == "documentdb" and state.operation == "list":
                logger.info("📋 Using list_documentdb workflow method")
                endpoint_ids = state.collected_params.get("endpoints") or state.collected_params.get("endpoint_ids")
                
                conversion_error = None
                if endpoint_ids and isinstance(endpoint_ids, list) and len(endpoint_ids) > 0:
                    if isinstance(endpoint_ids[0], str) and not endpoint_ids[0].isdigit():
                        logger.info(f"🔄 Converting endpoint names {endpoint_ids} to IDs...")
                        try:
                            endpoints_result = await api_executor_service.list_endpoints()
                            if endpoints_result.get("success"):
                                available_endpoints = endpoints_result.get("data", {}).get("endpoints", [])
                                name_to_id = {}
                                for ep in available_endpoints:
                                    ep_name = ep.get("name", "").strip()
                                    ep_id = ep.get("id")
                                    if ep_name and ep_id:
                                        name_to_id[ep_name.lower()] = ep_id
                                        name_to_id[ep_name.lower().replace("-", "").replace(" ", "")] = ep_id
                                
                                converted_ids = []
                                for name in endpoint_ids:
                                    name_clean = name.lower().strip().replace("-", "").replace(" ", "")
                                    if name_clean in name_to_id:
                                        converted_ids.append(name_to_id[name_clean])
                                        logger.info(f"  ✅ '{name}' -> ID {name_to_id[name_clean]}")
                                
                                if converted_ids:
                                    endpoint_ids = converted_ids
                                    logger.info(f"✅ Converted to IDs: {endpoint_ids}")
                                else:
                                    conversion_error = f"Could not find endpoint IDs for: {', '.join(endpoint_ids)}"
                            else:
                                conversion_error = "Failed to fetch endpoints for name-to-ID conversion"
                        except Exception as e:
                            conversion_error = f"Error converting endpoint names to IDs: {str(e)}"
                
                if conversion_error:
                    execution_result = {"success": False, "error": conversion_error}
                else:
                    execution_result = await api_executor_service.list_documentdb(
                        endpoint_ids=endpoint_ids,
                        ipc_engagement_id=None
                    )
            
            # Special handling for Business Unit listing with LLM formatting
            elif state.resource_type == "business_unit" and state.operation == "list":
                logger.info("📋 Using get_business_units_list method with LLM formatting")
                execution_result = await api_executor_service.get_business_units_list(
                    ipc_engagement_id=None,  # Will be fetched automatically
                    user_id=user_id
                )
                
                # Apply LLM formatting if successful
                if execution_result.get("success"):
                    raw_data = execution_result.get("data", {})
                    user_query = context.get("user_query", "list business units") if context else "list business units"
                    formatted_response = await llm_formatter.format_response(
                        resource_type="business_unit",
                        operation="list",
                        raw_data=raw_data,
                        user_query=user_query
                    )
                    # Return early with formatted response
                    state.status = ConversationStatus.COMPLETED
                    return {
                        "agent_name": self.agent_name,
                        "success": True,
                        "output": formatted_response,
                        "execution_result": execution_result
                    }
            
            # Special handling for Environment listing with LLM formatting
            elif state.resource_type == "environment" and state.operation == "list":
                logger.info("📋 Using get_environments_list method with LLM formatting")
                execution_result = await api_executor_service.get_environments_list(
                    ipc_engagement_id=None,  # Will be fetched automatically
                    user_id=user_id
                )
                
                # Apply LLM formatting if successful
                if execution_result.get("success"):
                    raw_data = execution_result.get("data", execution_result.get("environments", []))
                    user_query = context.get("user_query", "list environments") if context else "list environments"
                    formatted_response = await llm_formatter.format_response(
                        resource_type="environment",
                        operation="list",
                        raw_data=raw_data,
                        user_query=user_query
                    )
                    # Return early with formatted response
                    state.status = ConversationStatus.COMPLETED
                    return {
                        "agent_name": self.agent_name,
                        "success": True,
                        "output": formatted_response,
                        "execution_result": execution_result
                    }
            
            # Special handling for Zone listing with LLM formatting
            elif state.resource_type == "zone" and state.operation == "list":
                logger.info("🌐 Using get_zones_list method with LLM formatting")
                execution_result = await api_executor_service.get_zones_list(
                    ipc_engagement_id=None,  # Will be fetched automatically
                    user_id=user_id
                )
                
                # Apply LLM formatting if successful
                if execution_result.get("success"):
                    raw_data = execution_result.get("data", execution_result.get("zones", []))
                    user_query = context.get("user_query", "list zones") if context else "list zones"
                    formatted_response = await llm_formatter.format_response(
                        resource_type="zone",
                        operation="list",
                        raw_data=raw_data,
                        user_query=user_query
                    )
                    # Return early with formatted response
                    state.status = ConversationStatus.COMPLETED
                    return {
                        "agent_name": self.agent_name,
                        "success": True,
                        "output": formatted_response,
                        "execution_result": execution_result
                    }
            
            # Special handling for VM listing
            elif state.resource_type == "vm" and state.operation == "list":
                logger.info("📋 Using list_vms method")
                
                # Extract optional filters from collected params
                endpoint_filter = state.collected_params.get("endpoint")
                zone_filter = state.collected_params.get("zone")
                department_filter = state.collected_params.get("department")
                
                execution_result = await api_executor_service.list_vms(
                    ipc_engagement_id=None,  # Will be fetched automatically
                    endpoint_filter=endpoint_filter,
                    zone_filter=zone_filter,
                    department_filter=department_filter
                )
            
            # Special handling for Firewall listing
            elif state.resource_type == "firewall" and state.operation == "list":
                logger.info("📋 Using list_firewalls method")
                endpoint_ids = state.collected_params.get("endpoints") or state.collected_params.get("endpoint_ids")
                
                # Convert endpoint names to IDs if needed
                conversion_error = None
                if endpoint_ids and isinstance(endpoint_ids, list) and len(endpoint_ids) > 0:
                    if isinstance(endpoint_ids[0], str) and not endpoint_ids[0].isdigit():
                        logger.info(f"🔄 Converting endpoint names {endpoint_ids} to IDs...")
                        try:
                            endpoints_result = await api_executor_service.list_endpoints()
                            if endpoints_result.get("success"):
                                available_endpoints = endpoints_result.get("data", {}).get("endpoints", [])
                                name_to_id = {}
                                for ep in available_endpoints:
                                    ep_name = ep.get("name", "").strip()
                                    ep_id = ep.get("id")
                                    if ep_name and ep_id:
                                        name_to_id[ep_name.lower()] = ep_id
                                        name_to_id[ep_name.lower().replace("-", "").replace(" ", "")] = ep_id
                                
                                converted_ids = []
                                for name in endpoint_ids:
                                    name_clean = name.lower().strip().replace("-", "").replace(" ", "")
                                    if name_clean in name_to_id:
                                        converted_ids.append(name_to_id[name_clean])
                                        logger.info(f"  ✅ '{name}' -> ID {name_to_id[name_clean]}")
                                
                                if converted_ids:
                                    endpoint_ids = converted_ids
                                    logger.info(f"✅ Converted to IDs: {endpoint_ids}")
                                else:
                                    conversion_error = f"Could not find endpoint IDs for: {', '.join(endpoint_ids)}"
                            else:
                                conversion_error = "Failed to fetch endpoints for name-to-ID conversion"
                        except Exception as e:
                            conversion_error = f"Error converting endpoint names to IDs: {str(e)}"
                
                if conversion_error:
                    execution_result = {"success": False, "error": conversion_error}
                else:
                    execution_result = await api_executor_service.list_firewalls(
                        endpoint_ids=endpoint_ids,
                        ipc_engagement_id=None,  # Will be fetched automatically
                        variant=""
                    )
            
            # Special handling for cluster creation - build custom payload
            elif state.resource_type == "k8s_cluster" and state.operation == "create":
                logger.info("🏗️ Building cluster creation payload")
                try:
                    payload = await self._build_cluster_create_payload(state)
                    logger.info(f"📦 Built payload with keys: {list(payload.keys())}")
                    
                    # **DRY-RUN MODE** - Show payload without hitting API
                    # Set DRY_RUN = True to test, False to actually create
                    DRY_RUN = True
                    
                    if DRY_RUN:
                        logger.info(f"🔍 DRY RUN MODE - Showing payload without API call")
                        execution_result = {
                            "success": True,
                            "dry_run": True,
                            "data": {
                                "message": "Cluster creation payload generated (DRY RUN - no API call made)",
                                "payload": payload,
                                "payload_json": json.dumps(payload, indent=2)
                            }
                        }
                    else:
                        # Execute creation via API (when ready)
                        execution_result = await api_executor_service.execute_operation(
                            resource_type=state.resource_type,
                            operation=state.operation,
                            params=payload,
                            user_roles=user_roles,
                            dry_run=False
                        )
                except Exception as e:
                    logger.error(f"❌ Failed to build cluster payload: {e}")
                    execution_result = {
                        "success": False,
                        "error": f"Failed to build cluster payload: {str(e)}"
                    }
            
            else:
                # Standard execution for other operations
                        execution_result = await api_executor_service.execute_operation(
                            resource_type=state.resource_type,
                            operation=state.operation,
                            params=state.collected_params,
                            user_roles=user_roles,
                            dry_run=False,
                            user_id=user_id  # Pass user_id to retrieve credentials
                        )
            
            # Update conversation state with result
            state.set_execution_result(execution_result)
            
            # Format response for user
            if execution_result.get("success"):
                response = self._format_success_message(state, execution_result)
                logger.info(f"✅ Execution successful: {state.operation} {state.resource_type}")
            else:
                response = self._format_error_message(state, execution_result)
                logger.error(f"❌ Execution failed: {execution_result.get('error')}")
            
            return {
                "agent_name": self.agent_name,
                "success": True,
                "output": response,
                "execution_result": execution_result
            }
            
        except Exception as e:
            logger.error(f"❌ Execution agent failed: {str(e)}")
            return {
                "agent_name": self.agent_name,
                "success": False,
                "error": str(e),
                "output": f"I encountered an error while executing the operation: {str(e)}"
            }
    
    def _format_success_message(
        self,
        state,
        execution_result: Dict[str, Any]
    ) -> str:
        """
        Format a user-friendly success message.
        
        Args:
            state: Conversation state
            execution_result: Execution result dict
            
        Returns:
            Formatted success message
        """
        # Handle endpoint listing
        if state.resource_type == "endpoint" and state.operation == "list":
            result_data = execution_result.get("data", {})
            endpoints = result_data.get("endpoints", [])
            total = result_data.get("total", len(endpoints))
            
            if endpoints:
                message = f"📍 **Available Endpoints/Datacenters** ({total} found)\n\n"
                message += "| # | Name | ID | Type |\n"
                message += "|---|------|----|----- |\n"
                
                for i, ep in enumerate(endpoints, 1):
                    name = ep.get("name", "Unknown")
                    ep_id = ep.get("id", "N/A")
                    ep_type = ep.get("type", "")
                    message += f"| {i} | {name} | {ep_id} | {ep_type} |\n"
                
                message += "\n💡 You can use these endpoints when listing or creating clusters."
                return message
            else:
                return "❌ No endpoints found for your engagement."
        
        # Handle DRY-RUN mode for cluster creation
        if execution_result.get("dry_run"):
            payload_json = execution_result.get("data", {}).get("payload_json", "{}")
            message = f"""
🔍 **DRY RUN MODE - Cluster Creation Payload Preview**

The cluster creation payload has been generated successfully!  
**No API call was made** - this is for testing/validation only.

**📦 Complete Payload:**
```json
{payload_json}
```

**💡 To actually create the cluster:**
1. Set `DRY_RUN = False` in `execution_agent.py` (line ~380)
2. Ensure all API endpoints are correctly configured
3. Re-run the cluster creation workflow

**✅ All parameters were successfully collected and validated!**
"""
            return message
        
        operation_verb = {
            "create": "created",
            "update": "updated",
            "delete": "deleted",
            "list": "retrieved",
            "read": "retrieved"
        }.get(state.operation, "processed")
        
        resource_name = state.resource_type.replace("_", " ")
        
        # Handle cluster listing with beautiful formatting
        if state.resource_type == "k8s_cluster" and state.operation == "list":
            result_data = execution_result.get("data", {})
            
            # Handle both old format (dict with "data" key) and new streaming format (list directly)
            if isinstance(result_data, dict) and "data" in result_data:
                clusters = result_data["data"]
            elif isinstance(result_data, list):
                clusters = result_data
            else:
                clusters = []
            
            # Get the requested endpoint names from state
            requested_endpoint_names = state.collected_params.get("endpoint_names", [])
            
            # Get endpoint data from streaming response if available
            endpoint_data_map = execution_result.get("endpoint_data", {})
            
            # Group by endpoint
            by_endpoint = {}
            for cluster in clusters:
                endpoint = cluster.get("displayNameEndpoint", "Unknown")
                if endpoint not in by_endpoint:
                    by_endpoint[endpoint] = []
                by_endpoint[endpoint].append(cluster)
            
            # Add empty entries for requested endpoints that had no clusters
            # Map endpoint names from the streaming response
            for endpoint_id, endpoint_info in endpoint_data_map.items():
                endpoint_name = endpoint_info.get("endpoint_name", "")
                # Try to find the display name from existing clusters
                display_name = None
                for cluster in endpoint_info.get("clusters", []):
                    if "displayNameEndpoint" in cluster:
                        display_name = cluster["displayNameEndpoint"]
                        break
                
                # If no clusters, use endpoint_name as fallback
                if not display_name:
                    display_name = self._map_endpoint_name_to_display(endpoint_name)
                
                if display_name and display_name not in by_endpoint:
                    by_endpoint[display_name] = []
            
            # Also add requested endpoints that might not be in the response
            for endpoint_name in requested_endpoint_names:
                if endpoint_name not in by_endpoint:
                    by_endpoint[endpoint_name] = []
            
            # Build formatted response optimized for OpenWebUI markdown rendering
            total_clusters = len(clusters)
            total_endpoints = len(requested_endpoint_names) if requested_endpoint_names else len(by_endpoint)
            
            message = f"## ✅ Found {total_clusters} Kubernetes Cluster{'s' if total_clusters != 1 else ''}\n"
            message += f"*Across {total_endpoints} data center{'s' if total_endpoints != 1 else ''}*\n\n"
            message += "---\n\n"
            
            # Display clusters grouped by endpoint (use requested order if available)
            endpoints_to_show = requested_endpoint_names if requested_endpoint_names else sorted(by_endpoint.keys())
            
            for endpoint in endpoints_to_show:
                endpoint_clusters = by_endpoint.get(endpoint, [])
                cluster_count = len(endpoint_clusters)
                
                message += f"### 📍 {endpoint}\n"
                message += f"*{cluster_count} cluster{'s' if cluster_count != 1 else ''}*\n\n"
                
                if cluster_count == 0:
                    # Check if there was an error for this endpoint
                    error_msg = None
                    for ep_info in endpoint_data_map.values():
                        ep_display = self._map_endpoint_name_to_display(ep_info.get("endpoint_name", ""))
                        if ep_display == endpoint and ep_info.get("error"):
                            error_msg = ep_info.get("error")
                            break
                    
                    if error_msg:
                        message += f"⚠️ _{error_msg.capitalize()}_\n\n\n"
                    else:
                        message += "_No clusters found in this data center._\n\n\n"
                    continue
                
                for cluster in endpoint_clusters:
                    status = cluster.get("status", "Unknown")
                    status_emoji = "✅" if status == "Healthy" else ("⚠️" if status == "Draft" else "❌")
                    
                    cluster_name = cluster.get("clusterName", "Unknown")
                    node_count = cluster.get("nodescount", "?")
                    k8s_version = cluster.get("kubernetesVersion") or "N/A"
                    cluster_type = cluster.get("type", "")
                    cluster_id = cluster.get("clusterId", "")
                    backup_enabled = cluster.get("isIksBackupEnabled", False)
                    
                    # Create a compact card format
                    message += f"**{status_emoji} {cluster_name}**\n"
                    message += f"> **Status:** {status} | **Nodes:** {node_count} | **Version:** {k8s_version}\n"
                    message += f"> **Type:** {cluster_type} | **ID:** `{cluster_id}`"
                    
                    if str(backup_enabled).lower() == "true":
                        message += f" | **Backup:** 🔒 Enabled"
                    
                    message += "\n\n"
                
                message += "\n"
            
            # Add duration if available
            duration = execution_result.get("duration_seconds")
            if duration:
                message += f"---\n\n"
                message += f"⏱️ *Completed in {duration:.2f} seconds*\n"
            
            return message
        
        # Handle Kafka service listing
        if state.resource_type == "kafka" and state.operation == "list":
            services = execution_result.get("data", [])
            total = execution_result.get("total", len(services))
            endpoints_queried = execution_result.get("endpoints", [])
            
            message = f"## ✅ Found {total} Kafka Service{'s' if total != 1 else ''}\n"
            message += f"*Queried {len(endpoints_queried)} endpoint{'s' if len(endpoints_queried) != 1 else ''}*\n\n"
            message += "---\n\n"
            
            if total == 0:
                message += "_No Kafka services found in the selected endpoints._\n\n"
                message += "💡 Kafka services may not be deployed yet, or they might be in different endpoints.\n"
            else:
                for service in services:
                    # Handle case where service might be a string or unexpected type
                    if not isinstance(service, dict):
                        logger.warning(f"⚠️ Service is not a dict: {type(service)} = {service}")
                        message += f"• {service}\n\n"
                        continue
                    
                    # Map actual API fields
                    service_name = service.get("name", service.get("serviceName", "Unknown"))
                    status = service.get("status", "Unknown")
                    location = service.get("locationName", service.get("endpointName", "Unknown"))
                    version = service.get("version", "N/A")
                    cluster_name = service.get("clusterName", "N/A")
                    replicas = service.get("replicas", "N/A")
                    namespace = service.get("instanceNamespace", service.get("namespace", "N/A"))
                    
                    status_emoji = "✅" if status == "Active" else ("⚠️" if status == "Pending" else "❌")
                    
                    message += f"**{status_emoji} {service_name}**\n"
                    message += f"> **Status:** {status} | **Version:** {version}\n"
                    message += f"> **Location:** {location} | **Cluster:** {cluster_name}\n"
                    message += f"> **Replicas:** {replicas} | **Namespace:** {namespace}\n\n"
            
            # Add duration if available
            duration = execution_result.get("duration_seconds")
            if duration:
                message += f"---\n\n⏱️ *Completed in {duration:.2f} seconds*\n"
            
            return message
        
        # Handle GitLab service listing
        if state.resource_type == "gitlab" and state.operation == "list":
            services = execution_result.get("data", [])
            total = execution_result.get("total", len(services))
            endpoints_queried = execution_result.get("endpoints", [])
            
            message = f"## ✅ Found {total} GitLab Service{'s' if total != 1 else ''}\n"
            message += f"*Queried {len(endpoints_queried)} endpoint{'s' if len(endpoints_queried) != 1 else ''}*\n\n"
            message += "---\n\n"
            
            if total == 0:
                message += "_No GitLab services found in the selected endpoints._\n\n"
                message += "💡 GitLab services may not be deployed yet, or they might be in different endpoints.\n"
            else:
                for service in services:
                    # Handle case where service might be a string or unexpected type
                    if not isinstance(service, dict):
                        logger.warning(f"⚠️ Service is not a dict: {type(service)} = {service}")
                        message += f"• {service}\n\n"
                        continue
                    
                    # Map actual API fields
                    service_name = service.get("name", service.get("serviceName", "Unknown"))
                    status = service.get("status", "Unknown")
                    location = service.get("locationName", service.get("endpointName", "Unknown"))
                    version = service.get("version", "N/A")
                    cluster_name = service.get("clusterName", "N/A")
                    ingress_url = service.get("ingressUrl", service.get("url", "N/A"))
                    replicas = service.get("replicas", "N/A")
                    namespace = service.get("instanceNamespace", service.get("namespace", "N/A"))
                    
                    status_emoji = "✅" if status == "Active" else ("⚠️" if status == "Pending" else "❌")
                    
                    message += f"**{status_emoji} {service_name}**\n"
                    message += f"> **Status:** {status} | **Version:** {version}\n"
                    message += f"> **Location:** {location} | **Cluster:** {cluster_name}\n"
                    message += f"> **Replicas:** {replicas} | **Namespace:** {namespace}\n"
                    if ingress_url != "N/A":
                        message += f"> **Ingress URL:** `{ingress_url}`\n"
                    message += "\n"
            
            # Add duration if available
            duration = execution_result.get("duration_seconds")
            if duration:
                message += f"---\n\n⏱️ *Completed in {duration:.2f} seconds*\n"
            
            return message
        
        # Handle Container Registry service listing
        if state.resource_type == "container_registry" and state.operation == "list":
            services = execution_result.get("data", [])
            total = execution_result.get("total", len(services))
            endpoints_queried = execution_result.get("endpoints", [])
            
            message = f"## ✅ Found {total} Container Registry Service{'s' if total != 1 else ''}\n"
            message += f"*Queried {len(endpoints_queried)} endpoint{'s' if len(endpoints_queried) != 1 else ''}*\n\n"
            message += "---\n\n"
            
            if total == 0:
                message += "_No Container Registry services found in the selected endpoints._\n\n"
                message += "💡 Container Registry services may not be deployed yet, or they might be in different endpoints.\n"
            else:
                for service in services:
                    if not isinstance(service, dict):
                        logger.warning(f"⚠️ Service is not a dict: {type(service)} = {service}")
                        message += f"• {service}\n\n"
                        continue
                    
                    service_name = service.get("name", service.get("serviceName", "Unknown"))
                    status = service.get("status", "Unknown")
                    location = service.get("locationName", service.get("endpointName", "Unknown"))
                    version = service.get("version", "N/A")
                    cluster_name = service.get("clusterName", "N/A")
                    ingress_url = service.get("ingressUrl", service.get("url", "N/A"))
                    replicas = service.get("replicas", "N/A")
                    namespace = service.get("instanceNamespace", service.get("namespace", "N/A"))
                    
                    status_emoji = "✅" if status == "Active" or status == "Running" else ("⚠️" if status == "Pending" else "❌")
                    
                    message += f"**{status_emoji} {service_name}**\n"
                    message += f"> **Status:** {status} | **Version:** {version}\n"
                    message += f"> **Location:** {location} | **Cluster:** {cluster_name}\n"
                    message += f"> **Replicas:** {replicas} | **Namespace:** {namespace}\n"
                    if ingress_url != "N/A":
                        message += f"> **Registry URL:** `{ingress_url}`\n"
                    message += "\n"
            
            duration = execution_result.get("duration_seconds")
            if duration:
                message += f"---\n\n⏱️ *Completed in {duration:.2f} seconds*\n"
            
            return message
        
        # Handle Jenkins service listing
        if state.resource_type == "jenkins" and state.operation == "list":
            services = execution_result.get("data", [])
            total = execution_result.get("total", len(services))
            endpoints_queried = execution_result.get("endpoints", [])
            
            message = f"## ✅ Found {total} Jenkins Service{'s' if total != 1 else ''}\n"
            message += f"*Queried {len(endpoints_queried)} endpoint{'s' if len(endpoints_queried) != 1 else ''}*\n\n"
            message += "---\n\n"
            
            if total == 0:
                message += "_No Jenkins services found in the selected endpoints._\n\n"
                message += "💡 Jenkins services may not be deployed yet, or they might be in different endpoints.\n"
            else:
                for service in services:
                    if not isinstance(service, dict):
                        logger.warning(f"⚠️ Service is not a dict: {type(service)} = {service}")
                        message += f"• {service}\n\n"
                        continue
                    
                    service_name = service.get("name", service.get("serviceName", "Unknown"))
                    status = service.get("status", "Unknown")
                    location = service.get("locationName", service.get("endpointName", "Unknown"))
                    version = service.get("version", "N/A")
                    cluster_name = service.get("clusterName", "N/A")
                    ingress_url = service.get("ingressUrl", service.get("url", "N/A"))
                    replicas = service.get("replicas", "N/A")
                    namespace = service.get("instanceNamespace", service.get("namespace", "N/A"))
                    
                    status_emoji = "✅" if status == "Active" or status == "Running" else ("⚠️" if status == "Pending" else "❌")
                    
                    message += f"**{status_emoji} {service_name}**\n"
                    message += f"> **Status:** {status} | **Version:** {version}\n"
                    message += f"> **Location:** {location} | **Cluster:** {cluster_name}\n"
                    message += f"> **Replicas:** {replicas} | **Namespace:** {namespace}\n"
                    if ingress_url != "N/A":
                        message += f"> **Jenkins URL:** `{ingress_url}`\n"
                    message += "\n"
            
            duration = execution_result.get("duration_seconds")
            if duration:
                message += f"---\n\n⏱️ *Completed in {duration:.2f} seconds*\n"
            
            return message
        
        # Handle PostgreSQL service listing
        if state.resource_type == "postgres" and state.operation == "list":
            services = execution_result.get("data", [])
            total = execution_result.get("total", len(services))
            endpoints_queried = execution_result.get("endpoints", [])
            
            message = f"## ✅ Found {total} PostgreSQL Service{'s' if total != 1 else ''}\n"
            message += f"*Queried {len(endpoints_queried)} endpoint{'s' if len(endpoints_queried) != 1 else ''}*\n\n"
            message += "---\n\n"
            
            if total == 0:
                message += "_No PostgreSQL services found in the selected endpoints._\n\n"
                message += "💡 PostgreSQL services may not be deployed yet, or they might be in different endpoints.\n"
            else:
                for service in services:
                    if not isinstance(service, dict):
                        logger.warning(f"⚠️ Service is not a dict: {type(service)} = {service}")
                        message += f"• {service}\n\n"
                        continue
                    
                    service_name = service.get("name", service.get("serviceName", "Unknown"))
                    status = service.get("status", "Unknown")
                    location = service.get("locationName", service.get("endpointName", "Unknown"))
                    version = service.get("version", "N/A")
                    cluster_name = service.get("clusterName", "N/A")
                    replicas = service.get("replicas", "N/A")
                    namespace = service.get("instanceNamespace", service.get("namespace", "N/A"))
                    db_size = service.get("volumeSize", "N/A")
                    
                    status_emoji = "✅" if status == "Active" or status == "Running" else ("⚠️" if status == "Pending" else "❌")
                    
                    message += f"**{status_emoji} {service_name}**\n"
                    message += f"> **Status:** {status} | **Version:** {version}\n"
                    message += f"> **Location:** {location} | **Cluster:** {cluster_name}\n"
                    message += f"> **Replicas:** {replicas} | **Namespace:** {namespace}\n"
                    message += f"> **Storage:** {db_size}GB\n"
                    message += "\n"
            
            duration = execution_result.get("duration_seconds")
            if duration:
                message += f"---\n\n⏱️ *Completed in {duration:.2f} seconds*\n"
            
            return message
        
        # Handle DocumentDB service listing
        if state.resource_type == "documentdb" and state.operation == "list":
            services = execution_result.get("data", [])
            total = execution_result.get("total", len(services))
            endpoints_queried = execution_result.get("endpoints", [])
            
            message = f"## ✅ Found {total} DocumentDB Service{'s' if total != 1 else ''}\n"
            message += f"*Queried {len(endpoints_queried)} endpoint{'s' if len(endpoints_queried) != 1 else ''}*\n\n"
            message += "---\n\n"
            
            if total == 0:
                message += "_No DocumentDB services found in the selected endpoints._\n\n"
                message += "💡 DocumentDB services may not be deployed yet, or they might be in different endpoints.\n"
            else:
                for service in services:
                    if not isinstance(service, dict):
                        logger.warning(f"⚠️ Service is not a dict: {type(service)} = {service}")
                        message += f"• {service}\n\n"
                        continue
                    
                    service_name = service.get("name", service.get("serviceName", "Unknown"))
                    status = service.get("status", "Unknown")
                    location = service.get("locationName", service.get("endpointName", "Unknown"))
                    version = service.get("version", "N/A")
                    cluster_name = service.get("clusterName", "N/A")
                    replicas = service.get("replicas", "N/A")
                    namespace = service.get("instanceNamespace", service.get("namespace", "N/A"))
                    db_size = service.get("volumeSize", "N/A")
                    
                    status_emoji = "✅" if status == "Active" or status == "Running" else ("⚠️" if status == "Pending" else "❌")
                    
                    message += f"**{status_emoji} {service_name}**\n"
                    message += f"> **Status:** {status} | **Version:** {version}\n"
                    message += f"> **Location:** {location} | **Cluster:** {cluster_name}\n"
                    message += f"> **Replicas:** {replicas} | **Namespace:** {namespace}\n"
                    message += f"> **Storage:** {db_size}GB\n"
                    message += "\n"
            
            duration = execution_result.get("duration_seconds")
            if duration:
                message += f"---\n\n⏱️ *Completed in {duration:.2f} seconds*\n"
            
            return message
        
        # Handle VM listing
        if state.resource_type == "vm" and state.operation == "list":
            vms = execution_result.get("data", [])
            total = execution_result.get("total", len(vms))
            total_unfiltered = execution_result.get("total_unfiltered", total)
            last_synced = execution_result.get("last_synced", "N/A")
            filters_applied = execution_result.get("filters_applied", {})
            
            message = f"## ✅ Found {total} Virtual Machine{'s' if total != 1 else ''}\n"
            if total != total_unfiltered:
                message += f"*Showing {total} of {total_unfiltered} total VMs (filtered)*\n"
            message += f"*Last synced: {last_synced}*\n\n"
            message += "---\n\n"
            
            # Show active filters
            active_filters = [f"{k}: {v}" for k, v in filters_applied.items() if v]
            if active_filters:
                message += f"**Filters:** {', '.join(active_filters)}\n\n"
            
            if total == 0:
                message += "_No virtual machines found._\n\n"
                if any(filters_applied.values()):
                    message += "💡 Try removing some filters to see more VMs.\n"
            else:
                # Group VMs by endpoint for better organization
                by_endpoint = {}
                for vm_wrapper in vms:
                    vm = vm_wrapper.get("virtualMachine", {})
                    endpoint_name = vm.get("endpoint", {}).get("endpointName", "Unknown")
                    if endpoint_name not in by_endpoint:
                        by_endpoint[endpoint_name] = []
                    by_endpoint[endpoint_name].append(vm)
                
                # Display VMs grouped by endpoint
                for endpoint_name, endpoint_vms in sorted(by_endpoint.items()):
                    message += f"### 📍 {endpoint_name} ({len(endpoint_vms)} VM{'s' if len(endpoint_vms) != 1 else ''})\n\n"
                    
                    for vm in endpoint_vms:
                        vm_name = vm.get("vmName", "Unknown")
                        vm_status = vm.get("vmAttributes", {}).get("VM Status", "Unknown")
                        vm_ip = vm.get("vmAttributes", {}).get("IP", "N/A")
                        vcpu = vm.get("vmAttributes", {}).get("vCPU", "N/A")
                        ram_mb = vm.get("vmAttributes", {}).get("RAM", "N/A")
                        ram_gb = f"{int(ram_mb)/1024:.1f}GB" if ram_mb != "N/A" and ram_mb else "N/A"
                        storage = vm.get("storage", "N/A")
                        os_version = vm.get("vmAttributes", {}).get("OSVersion", "N/A")
                        os_make = vm.get("vmAttributes", {}).get("OSMake", "N/A")
                        zone_name = vm.get("zone", {}).get("zoneName", "N/A")
                        department = vm.get("department", {}).get("departmentName", "N/A")
                        created_time = vm.get("createdTime", "N/A")
                        
                        # Status emoji
                        status_emoji = "✅" if vm_status == "ACTIVE" else ("⚠️" if vm_status in ["PENDING", "RESTORE"] else "❌")
                        
                        message += f"**{status_emoji} {vm_name}**\n"
                        message += f"> **Status:** {vm_status} | **IP:** `{vm_ip}`\n"
                        message += f"> **Resources:** {vcpu} vCPU, {ram_gb} RAM, {storage}GB Storage\n"
                        message += f"> **OS:** {os_make} {os_version}\n"
                        message += f"> **Zone:** {zone_name}\n"
                        message += f"> **Department:** {department}\n"
                        message += f"> **Created:** {created_time}\n"
                        message += "\n"
            
            duration = execution_result.get("duration_seconds")
            if duration:
                message += f"---\n\n⏱️ *Completed in {duration:.2f} seconds*\n"
            
            return message
        
        # Handle Firewall listing
        if state.resource_type == "firewall" and state.operation == "list":
            firewalls = execution_result.get("data", [])
            total = execution_result.get("total", len(firewalls))
            endpoints_queried = execution_result.get("endpoints_queried", [])
            endpoint_results = execution_result.get("endpoint_results", {})
            
            message = f"## ✅ Found {total} Firewall{'s' if total != 1 else ''}\n"
            message += f"*Queried {len(endpoints_queried)} endpoint{'s' if len(endpoints_queried) != 1 else ''}*\n\n"
            message += "---\n\n"
            
            if total == 0:
                message += "_No firewalls found in the selected endpoints._\n\n"
                message += "💡 Firewalls may not be configured yet, or they might be in different endpoints.\n"
            else:
                # Group firewalls by endpoint for better organization
                by_endpoint = {}
                for fw in firewalls:
                    endpoint_id = fw.get("endId", fw.get("_queried_endpoint_id", "Unknown"))
                    if endpoint_id not in by_endpoint:
                        by_endpoint[endpoint_id] = []
                    by_endpoint[endpoint_id].append(fw)
                
                # Display firewalls grouped by endpoint
                for endpoint_id, endpoint_fws in sorted(by_endpoint.items()):
                    # Get endpoint name from first firewall or use ID
                    endpoint_name = f"Endpoint {endpoint_id}"
                    if endpoint_fws:
                        # Try to get a readable endpoint name from the firewall data
                        first_fw = endpoint_fws[0]
                        # Endpoint name might be in various places, try to find it
                        endpoint_name = f"Endpoint {endpoint_id}"
                    
                    message += f"### 📍 {endpoint_name} ({len(endpoint_fws)} firewall{'s' if len(endpoint_fws) != 1 else ''})\n\n"
                    
                    for fw in endpoint_fws:
                        fw_name = fw.get("displayName", fw.get("technicalName", "Unknown"))
                        fw_tech_name = fw.get("technicalName", "N/A")
                        fw_ip = fw.get("ip", "N/A")
                        component = fw.get("component", "N/A")
                        component_type = fw.get("componentType", "N/A")
                        hypervisor = fw.get("hypervisor", "N/A")
                        
                        # Get departments
                        departments = fw.get("department", [])
                        dept_names = [d.get("name", "Unknown") for d in departments[:3]]  # Show first 3
                        dept_str = ", ".join(dept_names) if dept_names else "N/A"
                        if len(departments) > 3:
                            dept_str += f" (+{len(departments)-3} more)"
                        
                        # Get basic details
                        basic = fw.get("basicDetails", {})
                        throughput = basic.get("throughput", "N/A")
                        iks_enabled = basic.get("iksEnabled", "N/A")
                        project_name = basic.get("projectName", "N/A")
                        
                        # Get config
                        config = fw.get("config", {})
                        vdom_name = config.get("vdomName", "N/A")
                        category = config.get("category", "N/A")
                        
                        # Status based on various flags
                        self_provisioned = fw.get("selfProvisioned", False)
                        status_emoji = "✅" if not fw.get("migrationStatus") else "⚠️"
                        
                        message += f"**{status_emoji} {fw_name}**\n"
                        message += f"> **Technical Name:** `{fw_tech_name}` | **IP:** `{fw_ip}`\n"
                        message += f"> **Component:** {component} ({component_type}) | **Category:** {category}\n"
                        message += f"> **VDOM:** {vdom_name} | **Hypervisor:** {hypervisor}\n"
                        if throughput != "N/A":
                            message += f"> **Throughput:** {throughput}\n"
                        message += f"> **IKS Enabled:** {iks_enabled} | **Self-Provisioned:** {'Yes' if self_provisioned else 'No'}\n"
                        if project_name != "N/A":
                            message += f"> **Project:** {project_name}\n"
                        message += f"> **Departments:** {dept_str}\n"
                        message += "\n"
            
            duration = execution_result.get("duration_seconds")
            if duration:
                message += f"---\n\n⏱️ *Completed in {duration:.2f} seconds*\n"
            
            return message
        
        # Default formatting for other operations
        message = f"✅ Successfully {operation_verb} {resource_name}!\n\n"
        
        # Add details from result
        result_data = execution_result.get("data", {})
        if result_data and isinstance(result_data, dict):
            message += "**Details:**\n"
            for key, value in result_data.items():
                if not key.startswith("_"):  # Skip internal fields
                    message += f"- {key}: {value}\n"
        
        # Add operation-specific guidance
        if state.operation == "create":
            message += "\n💡 Your resource is now being provisioned. It may take a few minutes to become fully available."
        elif state.operation == "delete":
            message += "\n💡 The resource has been removed. This action cannot be undone."
        elif state.operation == "update":
            message += "\n💡 Your changes have been applied and will take effect shortly."
        
        # Add duration if available
        duration = execution_result.get("duration_seconds")
        if duration:
            message += f"\n\n⏱️ Operation completed in {duration:.2f} seconds."
        
        return message
    
    def _format_error_message(
        self,
        state,
        execution_result: Dict[str, Any]
    ) -> str:
        """
        Format a user-friendly error message.
        
        Args:
            state: Conversation state
            execution_result: Execution result dict
            
        Returns:
            Formatted error message
        """
        error = execution_result.get("error", "Unknown error occurred")
        
        message = f"❌ I couldn't complete the {state.operation} operation on {state.resource_type}.\n\n"
        message += f"**Error:** {error}\n\n"
        
        # Add helpful suggestions based on error type
        error_lower = error.lower()
        
        if "permission" in error_lower or "denied" in error_lower:
            required_perms = execution_result.get("required_permissions", [])
            message += "**Reason:** You don't have the required permissions for this operation.\n"
            if required_perms:
                message += f"**Required roles:** {', '.join(required_perms)}\n"
            message += "\n💡 Please contact your administrator to request access."
        
        elif "validation" in error_lower or "invalid" in error_lower:
            validation_errors = execution_result.get("validation_errors", [])
            message += "**Reason:** Some parameters didn't pass validation.\n"
            if validation_errors:
                message += "**Issues:**\n"
                for err in validation_errors:
                    message += f"- {err}\n"
            message += "\n💡 Please correct the issues and try again."
        
        elif "not found" in error_lower or "404" in error_lower:
            message += "**Reason:** The resource you're trying to access doesn't exist.\n"
            message += "\n💡 Please check the resource name/ID and try again."
        
        elif "timeout" in error_lower or "connection" in error_lower:
            message += "**Reason:** The API service is not responding or is unreachable.\n"
            message += "\n💡 Please try again in a moment. If the issue persists, contact support."
        
        else:
            message += "💡 Please check your parameters and try again. If the issue persists, contact support."
        
        return message
    
    async def _execute_multi_resource(
        self,
        state: ConversationState,
        user_roles: List[str],
        session_id: str
    ) -> Dict[str, Any]:
        """
        Execute operations on multiple resource types in parallel.
        
        Example: "show gitlab and kafka in all endpoints"
        → Executes both gitlab.list and kafka.list in parallel
        → Combines results intelligently
        
        Args:
            state: Conversation state with comma-separated resource_type
            user_roles: User roles for permission checking
            session_id: Session identifier
            
        Returns:
            Combined execution result
        """
        import asyncio
        from app.services.ai_service import ai_service
        
        # Parse resource types from comma-separated or "and"-separated string
        resource_type_str = state.resource_type or ""
        
        # Split by comma or " and "
        if "," in resource_type_str:
            resource_types = [r.strip() for r in resource_type_str.split(",")]
        elif " and " in resource_type_str.lower():
            resource_types = [r.strip() for r in resource_type_str.lower().split(" and ")]
        else:
            resource_types = [resource_type_str.strip()]
        
        # Remove empty strings and duplicates
        resource_types = list(set([rt for rt in resource_types if rt]))
        
        logger.info(f"🔀 Executing {len(resource_types)} resource operations in parallel: {resource_types}")
        logger.info(f"👥 User roles for multi-resource execution: {user_roles}")
        
        # Execute all resources in parallel
        tasks = []
        for resource_type in resource_types:
            # Get the resource agent
            resource_agent = self.resource_agent_map.get(resource_type)
            
            if resource_agent:
                logger.info(f"  📦 Adding {resource_type} to execution queue")
                task = resource_agent.execute_operation(
                    operation=state.operation,
                    params=state.collected_params,
                    context={
                        "session_id": session_id,
                        "user_id": state.user_id,
                        "user_query": state.user_query,
                        "user_roles": user_roles,
                        "resource_type": resource_type
                    }
                )
                tasks.append((resource_type, task))
            else:
                logger.warning(f"  ⚠️ No agent found for {resource_type}, skipping")
        
        if not tasks:
            return {
                "agent_name": self.agent_name,
                "success": False,
                "output": f"I don't have specialized handlers for these resources: {', '.join(resource_types)}",
                "execution_result": {}
            }
        
        # Execute all tasks in parallel
        logger.info(f"⚡ Executing {len(tasks)} tasks in parallel...")
        results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
        
        # Collect results
        combined_results = {}
        combined_data = []
        all_success = True
        error_messages = []
        
        for i, (resource_type, result) in enumerate(zip([rt for rt, _ in tasks], results)):
            if isinstance(result, Exception):
                logger.error(f"❌ {resource_type} execution failed with exception: {result}")
                all_success = False
                error_messages.append(f"{resource_type}: {str(result)}")
            else:
                combined_results[resource_type] = result
                if result.get("success"):
                    logger.info(f"✅ {resource_type} completed successfully")
                    # Collect data from each resource
                    resource_data = result.get("data", [])
                    combined_data.extend([{
                        **item,
                        "_resource_type": resource_type
                    } for item in resource_data])
                else:
                    logger.error(f"❌ {resource_type} failed: {result.get('error')}")
                    all_success = False
                    error_messages.append(f"{resource_type}: {result.get('error', 'Unknown error')}")
        
        # Format combined response using LLM
        if all_success:
            state.status = ConversationStatus.COMPLETED
            
            # Use LLM to intelligently combine and format the results
            try:
                combined_text_responses = []
                total_count = 0
                
                for resource_type, result in combined_results.items():
                    if result.get("success"):
                        response_text = result.get("response", "")
                        count = result.get("metadata", {}).get("count", 0)
                        total_count += count
                        
                        combined_text_responses.append(f"## {resource_type.title()}\n{response_text}")
                
                # Combine with LLM for natural flow
                combine_prompt = f"""You are a cloud infrastructure assistant. The user asked to see multiple resource types, and I have the results for each.

**User's Query:** {state.user_query}

**Results:**
{chr(10).join(combined_text_responses)}

**Instructions:**
1. Combine these results into a single, coherent response
2. Start with a summary: "Found X resources across Y types"
3. Present each resource type clearly (use headings/sections)
4. Keep the formatting from each individual result (tables, emojis, etc.)
5. Be conversational and helpful
6. If there are interesting patterns or insights across resource types, mention them

Format as markdown. Be concise yet informative."""
                
                final_response = await ai_service._call_chat_with_retries(
                    prompt=combine_prompt,
                    max_tokens=3000,
                    temperature=0.3,
                    timeout=20
                )
                
                if not final_response:
                    # Fallback: simple concatenation
                    final_response = f"# Combined Results\n\n" + "\n\n---\n\n".join(combined_text_responses)
            
            except Exception as e:
                logger.error(f"Error combining results with LLM: {e}")
                final_response = f"# Combined Results\n\n" + "\n\n---\n\n".join(combined_text_responses)
            
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
            
            error_summary = f"I encountered errors while fetching {', '.join(resource_types)}:\n\n"
            error_summary += "\n".join(f"- **{err}**" for err in error_messages)
            
            # Include partial results if any succeeded
            if combined_results:
                success_types = [rt for rt, res in combined_results.items() if res.get("success")]
                if success_types:
                    error_summary += f"\n\n✅ Successfully retrieved: {', '.join(success_types)}"
                    error_summary += "\n\nShowing partial results..."
                    
                    # Format successful results
                    for resource_type in success_types:
                        result = combined_results[resource_type]
                        error_summary += f"\n\n## {resource_type.title()}\n{result.get('response', '')}"
            
            return {
                "agent_name": self.agent_name,
                "success": False,
                "output": error_summary,
                "execution_result": {
                    "success": False,
                    "errors": error_messages,
                    "partial_results": combined_results
                }
            }

