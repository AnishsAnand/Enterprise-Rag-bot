"""
Cluster Creation Handler - Manages the multi-step workflow for creating Kubernetes clusters.

This handler encapsulates the 17-step customer workflow for cluster creation,
making it easier to maintain and test independently.
"""

import logging
import re
from typing import Any, Dict, Optional
import json

from app.services.api_executor_service import api_executor_service
from app.agents.tools.parameter_extraction import ParameterExtractor
from app.agents.state.conversation_state import conversation_state_manager

logger = logging.getLogger(__name__)


class ClusterCreationHandler:
    """
    Handles the complete workflow for customer cluster creation.
    
    Implements the 17-step process defined in resource_schema.json,
    collecting parameters conversationally and validating them.
    """
    
    def __init__(self):
        self.param_extractor = ParameterExtractor()
        
        # Define the parameter collection workflow order for customers
        self.workflow = [
            "clusterName",
            "datacenter", 
            "k8sVersion",
            "cniDriver",
            "businessUnit",
            "environment",
            "zone",
            "operatingSystem",
            "workerPoolName",
            "nodeType",
            "flavor",
            "replicaCount",
            "enableAutoscaling",  # optional
            "maxReplicas",  # conditional
            "tags"  # optional
        ]
    
    def _log_cluster_payload(self, state: Any, step: str) -> None:
        """
        Log the current cluster creation payload for debugging.
        
        Args:
            state: Conversation state
            step: Current step/action name
        """
        params = state.collected_params.copy()
        
        # Build a readable payload representation
        payload = {}
        for key, value in params.items():
            # Skip internal params starting with _
            if key.startswith('_'):
                continue
            
            # Extract readable values from dict objects
            if isinstance(value, dict):
                payload[key] = value.get('name') or value.get('display_name') or value
            else:
                payload[key] = value
        
        logger.info(f"📦 PAYLOAD [{step}] Collected {len(payload)} params: {json.dumps(payload, indent=2)}")
        
        # Log the internal IDs separately for reference
        internal_params = {k: v for k, v in params.items() if k.startswith('_')}
        if internal_params:
            logger.debug(f"🔐 Internal IDs: {internal_params}")
    
    async def handle(self, input_text: str, state: Any) -> Dict[str, Any]:
        """
        Main entry point for handling cluster creation workflow.
        
        Args:
            input_text: User's current input
            state: Conversation state
            
        Returns:
            Dict with next prompt or ready_to_execute flag
        """
        logger.info(f"🎯 Cluster creation handler - collected params: {list(state.collected_params.keys())}")
        
        # Log current payload state at entry
        self._log_cluster_payload(state, "ENTRY")
        
        # If user provided input AND we previously asked for a parameter, process it
        if input_text and hasattr(state, 'last_asked_param') and state.last_asked_param:
            logger.info(f"📥 Processing response for: {state.last_asked_param}")
            result = await self._process_user_input(input_text, state)
            
            # Log payload after processing
            self._log_cluster_payload(state, f"AFTER_{state.last_asked_param}")
            
            if result:
                # Check if we should continue to next parameter (success with feedback)
                if result.get("continue_workflow"):
                    # Get the next parameter after this success
                    next_param = self._find_next_parameter(state)
                    if next_param is None:
                        return self._build_summary(state)
                    # Combine success message with next question
                    next_question = await self._ask_for_parameter(next_param, state)
                    result["output"] = result["output"] + "\n\n" + next_question["output"]
                    return result
                # Error or validation failure - ask again
                return result
        
        # NEW: On first entry (no last_asked_param), try to extract params from user's initial response
        # This handles the case where user provides params in response to IntentAgent's clarification
        elif input_text and not hasattr(state, 'last_asked_param'):
            logger.info(f"🔍 First entry - attempting to extract params from: '{input_text[:100]}...'")
            extracted = await self._extract_initial_params(input_text, state)
            if extracted:
                logger.info(f"✅ Extracted initial params: {list(extracted.keys())}")
        
        # Find the next parameter to collect (after processing or on initial call)
        next_param = self._find_next_parameter(state)
        
        # If all parameters collected, mark as ready
        if next_param is None:
            return self._build_summary(state)
        
        # Ask for the next parameter
        return await self._ask_for_parameter(next_param, state)
    
    async def _extract_initial_params(self, input_text: str, state: Any) -> Dict[str, Any]:
        """
        Extract parameters from user's initial response (before any specific param was asked).
        
        This handles the case where user provides params in response to IntentAgent's clarification,
        e.g., "prod-clus as the name and bengaluru location"
        
        Args:
            input_text: User's input text
            state: Conversation state
            
        Returns:
            Dict of extracted parameters
        """
        extracted = {}
        text_lower = input_text.lower()
        
        # First, normalize any existing params from IntentAgent
        self._normalize_collected_params(state)
        
        # If we already have clusterName from IntentAgent, validate it
        if "clusterName" in state.collected_params and not state.collected_params.get("_clusterName_validated"):
            cluster_name = state.collected_params["clusterName"]
            logger.info(f"🔍 Validating clusterName from IntentAgent: '{cluster_name}'")
            
            # Validate format
            if not re.match(r'^[a-zA-Z][a-zA-Z0-9-]{2,17}$', cluster_name):
                logger.info(f"❌ Name format invalid: '{cluster_name}'")
                del state.collected_params["clusterName"]
            else:
                # Check availability
                check_result = await api_executor_service.check_cluster_name_available(cluster_name)
                if check_result.get("available"):
                    state.collected_params["_clusterName_validated"] = True
                    extracted["clusterName"] = cluster_name
                    logger.info(f"✅ clusterName '{cluster_name}' validated and available")
                    conversation_state_manager.update_session(state)
                else:
                    logger.info(f"❌ clusterName '{cluster_name}' is not available")
                    del state.collected_params["clusterName"]
        
        # If we already have clusterName validated, skip extraction
        if "clusterName" not in state.collected_params:
            # Try to extract cluster name
            # Patterns: "name is X", "X as the name", "name: X", "called X", "name it X"
            name_patterns = [
                r'(?:name\s+is|named?)\s+["\']?([a-zA-Z][a-zA-Z0-9-]{2,17})["\']?',
                r'["\']?([a-zA-Z][a-zA-Z0-9-]{2,17})["\']?\s+(?:as\s+)?(?:the\s+)?name',
                r'name[:\s]+["\']?([a-zA-Z][a-zA-Z0-9-]{2,17})["\']?',
                r'(?:call|name)\s+it\s+["\']?([a-zA-Z][a-zA-Z0-9-]{2,17})["\']?',
            ]
            
            cluster_name = None
            for pattern in name_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    potential_name = match.group(1)
                    # Validate format
                    if re.match(r'^[a-zA-Z][a-zA-Z0-9-]{2,17}$', potential_name):
                        cluster_name = potential_name
                        logger.info(f"🔍 Extracted cluster name from pattern: '{cluster_name}'")
                        break
            
            # If we found a cluster name, validate and store it
            if cluster_name:
                logger.info(f"🔍 Checking availability for extracted name: '{cluster_name}'")
                check_result = await api_executor_service.check_cluster_name_available(cluster_name)
                
                if check_result.get("available"):
                    state.collected_params["clusterName"] = cluster_name
                    extracted["clusterName"] = cluster_name
                    logger.info(f"✅ Stored extracted clusterName: '{cluster_name}'")
                    conversation_state_manager.update_session(state)
                else:
                    logger.info(f"⚠️ Extracted name '{cluster_name}' is not available, will ask user")
        else:
            logger.info(f"✅ clusterName already collected from IntentAgent: '{state.collected_params['clusterName']}'")
            extracted["clusterName"] = state.collected_params["clusterName"]
        
        # Try to extract datacenter/location
        # Common location keywords that map to datacenters
        location_keywords = {
            'bengaluru': 'bengaluru',
            'bangalore': 'bengaluru',
            'blr': 'bengaluru',
            'delhi': 'delhi',
            'del': 'delhi',
            'mumbai': 'mumbai',
            'mum': 'mumbai',
            'bkc': 'mumbai-bkc',
            'chennai': 'chennai',
            'amb': 'chennai-amb',
            'cressex': 'cressex',
        }
        
        # Check for location mentions (only if not already detected)
        if not hasattr(state, '_detected_location') or not state._detected_location:
            detected_location = None
            for keyword, location in location_keywords.items():
                if keyword in text_lower:
                    detected_location = location
                    logger.info(f"🔍 Detected location keyword '{keyword}' -> '{location}'")
                    break
            
            # Store detected location for later use in datacenter selection
            if detected_location:
                state._detected_location = detected_location
                extracted["_detected_location"] = detected_location
                logger.info(f"📍 Stored detected location hint: '{detected_location}'")
                conversation_state_manager.update_session(state)
        
        return extracted
    
    def _find_next_parameter(self, state: Any) -> Optional[str]:
        """
        Find the next parameter that needs to be collected.
        
        Args:
            state: Conversation state
            
        Returns:
            Next parameter name or None if all collected
        """
        # First, normalize any alternate param names that might be in state
        self._normalize_collected_params(state)
        
        for param in self.workflow:
            if param not in state.collected_params:
                # Skip optional params based on user choice
                if param == "maxReplicas" and not state.collected_params.get("enableAutoscaling"):
                    continue
                    
                return param
        
        return None
    
    def _normalize_collected_params(self, state: Any) -> None:
        """
        Normalize alternate parameter names in collected_params.
        
        This handles cases where IntentAgent extracted params with different naming.
        E.g., cluster_name → clusterName
        
        Args:
            state: Conversation state (modified in place)
        """
        # Mapping of alternate names to canonical names
        aliases = {
            "cluster_name": "clusterName",
            "clustername": "clusterName",
            "name": "clusterName",
            "k8s_version": "k8sVersion",
            "kubernetes_version": "k8sVersion",
            "version": "k8sVersion",
            "data_center": "datacenter",
            "location": "_detected_location",
            "endpoint": "_detected_location",
            "worker_pool_name": "workerPoolName",
            "pool_name": "workerPoolName",
            "node_type": "nodeType",
            "replica_count": "replicaCount",
            "node_count": "replicaCount",
            "replicas": "replicaCount",
            "cni_driver": "cniDriver",
            "business_unit": "businessUnit",
            "operating_system": "operatingSystem",
            "enable_autoscaling": "enableAutoscaling",
            "max_replicas": "maxReplicas",
        }
        
        params_to_normalize = []
        for key in list(state.collected_params.keys()):
            canonical = aliases.get(key.lower())
            if canonical and canonical != key and canonical not in state.collected_params:
                params_to_normalize.append((key, canonical))
        
        for old_key, new_key in params_to_normalize:
            value = state.collected_params.pop(old_key)
            state.collected_params[new_key] = value
            logger.info(f"🔄 Normalized param in state: {old_key} → {new_key} = {value}")
    
    def _build_summary(self, state: Any) -> Dict[str, Any]:
        """
        Build final summary when all parameters are collected.
        
        Args:
            state: Conversation state
            
        Returns:
            Dict with summary and ready_to_execute flag
        """
        logger.info(f"✅ All cluster creation parameters collected!")
        
        # Log final payload
        self._log_cluster_payload(state, "COMPLETE")
        
        state.status = "READY_TO_EXECUTE"
        state.missing_params = []
        
        params = state.collected_params
        autoscaling_info = ""
        if params.get("enableAutoscaling"):
            autoscaling_info = f" (autoscaling up to {params.get('maxReplicas', 8)} nodes)"
        
        # Helper to safely get name from dict or string
        def get_name(val, key='name'):
            if isinstance(val, dict):
                return val.get(key, val.get('display_name', str(val)))
            return str(val) if val else 'N/A'
        
        summary = f"""
**🎉 Cluster Configuration Complete!**

**Basic Configuration:**
- **Cluster Name**: `{params.get('clusterName', 'N/A')}`
- **Datacenter**: {get_name(params.get('datacenter'))}
- **Kubernetes Version**: {params.get('k8sVersion', 'N/A')}
- **CNI Driver**: {params.get('cniDriver', 'N/A')}

**Network Setup:**
- **Business Unit**: {get_name(params.get('businessUnit'))}
- **Environment**: {get_name(params.get('environment'))}
- **Zone**: {get_name(params.get('zone'))}

**Worker Node Configuration:**
- **Operating System**: {get_name(params.get('operatingSystem'), 'display_name')}
- **Worker Pool Name**: `{params.get('workerPoolName', 'N/A')}`
- **Node Type**: {params.get('nodeType', 'N/A').replace('generalPurpose', 'General Purpose').replace('computeOptimized', 'Compute Optimized').replace('memoryOptimized', 'Memory Optimized') if params.get('nodeType') else 'N/A'}
- **Flavor**: {get_name(params.get('flavor'))}
- **Node Count**: {params.get('replicaCount', 'N/A')}{autoscaling_info}

**Master Nodes** (auto-configured):
- **Type**: Virtual Control Plane
- **Mode**: High Availability (3x D8 nodes)

---

⏱️ **Estimated creation time**: 15-30 minutes

Would you like me to proceed with creating this cluster?
"""
        
        return {
            "agent_name": "ValidationAgent",
            "success": True,
            "ready_to_execute": True,
            "output": summary
        }
    
    async def _process_user_input(self, input_text: str, state: Any) -> Optional[Dict[str, Any]]:
        """
        Process user's input for the last asked parameter.
        
        Args:
            input_text: User's response
            state: Conversation state
            
        Returns:
            Dict with result or None to continue
        """
        last_param = state.last_asked_param
        logger.info(f"📝 Processing user input for: {last_param}")
        
        # Delegate to specific handler methods
        handlers = {
            "clusterName": self._handle_cluster_name,
            "datacenter": self._handle_datacenter,
            "k8sVersion": self._handle_k8s_version,
            "cniDriver": self._handle_cni_driver,
            "businessUnit": self._handle_business_unit,
            "environment": self._handle_environment,
            "zone": self._handle_zone,
            "operatingSystem": self._handle_operating_system,
            "workerPoolName": self._handle_worker_pool_name,
            "nodeType": self._handle_node_type,
            "flavor": self._handle_flavor,
            "replicaCount": self._handle_replica_count,
            "enableAutoscaling": self._handle_autoscaling,
            "maxReplicas": self._handle_max_replicas,
            "tags": self._handle_tags
        }
        
        handler = handlers.get(last_param)
        if handler:
            return await handler(input_text, state)
        
        return None
    
    async def _handle_cluster_name(self, input_text: str, state: Any) -> Optional[Dict[str, Any]]:
        """Validate and collect cluster name."""
        cluster_name = input_text.strip()
        logger.info(f"🔍 Validating cluster name: '{cluster_name}'")
        
        # Validate format
        if not re.match(r'^[a-zA-Z][a-zA-Z0-9-]{2,17}$', cluster_name):
            logger.info(f"❌ Name format invalid")
            return {
                "agent_name": "ValidationAgent",
                "success": True,
                "output": "❌ Cluster name must start with a letter and be 3-18 characters (letters, numbers, hyphens). Please try again:"
            }
        
        logger.info(f"✅ Name format valid, checking availability...")
        # Check availability
        check_result = await api_executor_service.check_cluster_name_available(cluster_name)
        logger.info(f"📋 Availability check result: {check_result}")
        
        if not check_result.get("available"):
            logger.info(f"❌ Name already taken")
            return {
                "agent_name": "ValidationAgent",
                "success": True,
                "output": f"❌ Cluster name '{cluster_name}' is already taken. Please choose another name:"
            }
        
        logger.info(f"✅ Name is available, storing...")
        state.collected_params["clusterName"] = cluster_name
        logger.info(f"✅✅ Stored clusterName = '{cluster_name}', collected params now: {list(state.collected_params.keys())}")
        # Persist state after collecting parameter
        conversation_state_manager.update_session(state)
        
        # Return success message to user (don't return None - that skips feedback)
        return {
            "agent_name": "ValidationAgent",
            "success": True,
            "output": f"✅ Great! Cluster name **`{cluster_name}`** is available and reserved.\n\nLet me continue with the next step...",
            "continue_workflow": True  # Signal to continue to next parameter
        }
    
    async def _handle_datacenter(self, input_text: str, state: Any) -> Optional[Dict[str, Any]]:
        """Fetch and match datacenter selection."""
        # Fetch datacenters if not cached
        if not hasattr(state, '_datacenter_options'):
            engagement_id = await api_executor_service.get_engagement_id()
            dc_result = await api_executor_service.get_iks_images_and_datacenters(engagement_id)
            state._datacenter_options = dc_result.get("datacenters", [])
            state._all_images = dc_result.get("images", [])
        
        # Match user selection
        matched = await self.param_extractor.match_user_selection(input_text, state._datacenter_options)
        matched_data = json.loads(matched)
        
        if matched_data.get("matched"):
            dc_info = matched_data.get("matched_item")
            state.collected_params["datacenter"] = dc_info
            state.collected_params["_datacenter_id"] = dc_info["id"]
            conversation_state_manager.update_session(state)
            return None
        else:
            return {
                "agent_name": "ValidationAgent",
                "success": True,
                "output": "❌ I couldn't match that datacenter. Please choose from the list above:"
            }
    
    async def _handle_k8s_version(self, input_text: str, state: Any) -> Optional[Dict[str, Any]]:
        """Match Kubernetes version selection."""
        if not hasattr(state, '_k8s_versions'):
            dc_id = state.collected_params["_datacenter_id"]
            versions = await api_executor_service.get_k8s_versions_for_datacenter(
                dc_id, 
                state._all_images
            )
            state._k8s_versions = versions
        
        if input_text.strip() in state._k8s_versions:
            state.collected_params["k8sVersion"] = input_text.strip()
            conversation_state_manager.update_session(state)
            return None
        else:
            return {
                "agent_name": "ValidationAgent",
                "success": True,
                "output": "❌ Please select one of the versions listed above:"
            }
    
    async def _handle_cni_driver(self, input_text: str, state: Any) -> Optional[Dict[str, Any]]:
        """Match CNI driver selection."""
        if not hasattr(state, '_cni_drivers'):
            dc_id = state.collected_params["_datacenter_id"]
            k8s_version = state.collected_params["k8sVersion"]
            driver_result = await api_executor_service.get_network_drivers(dc_id, k8s_version)
            state._cni_drivers = driver_result.get("drivers", [])
        
        if input_text.strip() in state._cni_drivers:
            state.collected_params["cniDriver"] = input_text.strip()
            conversation_state_manager.update_session(state)
            return None
        else:
            return {
                "agent_name": "ValidationAgent",
                "success": True,
                "output": "❌ Please select one of the CNI drivers listed above:"
            }
    
    async def _handle_business_unit(self, input_text: str, state: Any) -> Optional[Dict[str, Any]]:
        """Fetch and match business unit selection (filtered by selected datacenter endpoint)."""
        if not hasattr(state, '_business_units') or not state._business_units:
            # Fetch all business units using the correct API
            logger.info(f"🏢 Fetching business units from API (handler)...")
            bu_result = await api_executor_service.get_business_units_list(force_refresh=True)
            logger.info(f"🏢 API result: success={bu_result.get('success')}, departments count={len(bu_result.get('departments', []))}")
            
            if not bu_result.get("success"):
                logger.error(f"❌ Failed to fetch business units: {bu_result.get('error')}")
            
            all_departments = bu_result.get("departments", [])
            
            # Filter by selected datacenter's endpoint ID
            selected_endpoint_id = state.collected_params.get("_datacenter_id")
            logger.info(f"🏢 Filtering business units for endpoint ID: {selected_endpoint_id}")
            
            # Filter departments that match the selected endpoint
            filtered_bus = []
            for dept in all_departments:
                dept_endpoint_id = dept.get("endpoint", {}).get("id")
                if dept_endpoint_id == selected_endpoint_id:
                    filtered_bus.append({
                        "id": dept["id"],
                        "name": dept["name"],
                        "endpoint_id": dept_endpoint_id,
                        "location": dept.get("endpoint", {}).get("location")
                    })
            
            logger.info(f"🏢 Found {len(filtered_bus)} business units for endpoint {selected_endpoint_id} (from {len(all_departments)} total)")
            state._business_units = filtered_bus
            state._all_departments = all_departments  # Keep all for reference
            
            # Also fetch environments for later use
            env_result = await api_executor_service.get_environments_list()
            state._all_environments = env_result.get("environments", [])
        
        matched = await self.param_extractor.match_user_selection(input_text, state._business_units)
        matched_data = json.loads(matched)
        
        if matched_data.get("matched"):
            bu_info = matched_data.get("matched_item")
            state.collected_params["businessUnit"] = bu_info
            logger.info(f"✅ Selected business unit: {bu_info['name']} (ID: {bu_info['id']})")
            conversation_state_manager.update_session(state)
            return None
        else:
            return {
                "agent_name": "ValidationAgent",
                "success": True,
                "output": "❌ I couldn't match that business unit. Please choose from the list above:"
            }
    
    async def _handle_environment(self, input_text: str, state: Any) -> Optional[Dict[str, Any]]:
        """Filter and match environment selection (filtered by selected business unit)."""
        bu_value = state.collected_params.get("businessUnit")
        
        # Handle both dict and string cases
        if isinstance(bu_value, dict):
            bu_id = bu_value.get("id")
            bu_name = bu_value.get("name")
        else:
            # If it's a string, try to find the matching BU from cached data
            logger.warning(f"⚠️ businessUnit is a string in handler: {bu_value}")
            bu_name = bu_value
            bu_id = None
            if hasattr(state, '_business_units'):
                for bu in state._business_units:
                    if bu.get("name") == bu_value:
                        bu_id = bu.get("id")
                        state.collected_params["businessUnit"] = bu
                        break
        
        # Ensure environments are fetched
        if not hasattr(state, '_all_environments') or not state._all_environments:
            logger.info(f"🔄 Fetching environments in handler (not cached)")
            env_result = await api_executor_service.get_environments_list()
            state._all_environments = env_result.get("environments", [])
        
        # Filter environments by department ID (business unit)
        # Environments have departmentId or department field
        filtered_envs = []
        for env in state._all_environments:
            env_dept_id = env.get("departmentId") or env.get("department_id")
            if env_dept_id == bu_id:
                filtered_envs.append(env)
        
        logger.info(f"🔧 Found {len(filtered_envs)} environments for BU '{bu_name}' (ID: {bu_id})")
        
        env_options = [{"id": env["id"], "name": env["name"]} for env in filtered_envs]
        matched = await self.param_extractor.match_user_selection(input_text, env_options)
        matched_data = json.loads(matched)
        
        if matched_data.get("matched"):
            env_info = matched_data.get("matched_item")
            full_env = next((e for e in filtered_envs if e["id"] == env_info["id"]), None)
            if full_env:
                state.collected_params["environment"] = {"id": full_env["id"], "name": full_env["name"]}
                state.collected_params["_environment_name"] = full_env["name"]
                state.collected_params["_department_id"] = bu_id  # Store for zone filtering
                logger.info(f"✅ Selected environment: {full_env['name']} (ID: {full_env['id']})")
                conversation_state_manager.update_session(state)
                return None
        
        return {
            "agent_name": "ValidationAgent",
            "success": True,
            "output": "❌ I couldn't match that environment. Please choose from the list above:"
        }
    
    async def _handle_zone(self, input_text: str, state: Any) -> Optional[Dict[str, Any]]:
        """Fetch, filter, and match zone selection."""
        if not hasattr(state, '_zones'):
            engagement_id = await api_executor_service.get_engagement_id()
            zone_result = await api_executor_service.get_zones_list(engagement_id)
            state._zones = zone_result.get("zones", [])
        
        bu_value = state.collected_params.get("businessUnit")
        bu_name = bu_value.get("name") if isinstance(bu_value, dict) else bu_value
        env_name = state.collected_params.get("_environment_name", "")
        filtered_zones = [z for z in state._zones 
                         if z["departmentName"] == bu_name and z["environmentName"] == env_name]
        
        zone_options = [{"id": z["zoneId"], "name": z["zoneName"]} for z in filtered_zones]
        matched = await self.param_extractor.match_user_selection(input_text, zone_options)
        matched_data = json.loads(matched)
        
        if matched_data.get("matched"):
            zone_info = matched_data.get("matched_item")
            state.collected_params["zone"] = zone_info
            state.collected_params["_zone_id"] = zone_info["id"]
            conversation_state_manager.update_session(state)
            return None
        else:
            return {
                "agent_name": "ValidationAgent",
                "success": True,
                "output": "❌ I couldn't match that zone. Please choose from the list above:"
            }
    
    async def _handle_operating_system(self, input_text: str, state: Any) -> Optional[Dict[str, Any]]:
        """Fetch and match operating system selection."""
        if not hasattr(state, '_os_options'):
            zone_id = state.collected_params["_zone_id"]
            circuit_id = await api_executor_service.get_circuit_id(None)
            k8s_version = state.collected_params["k8sVersion"]
            os_result = await api_executor_service.get_os_images(zone_id, circuit_id, k8s_version)
            state._os_options = os_result.get("os_options", [])
            state._circuit_id = circuit_id
        
        os_options = [{"id": i, "name": opt["display_name"]} for i, opt in enumerate(state._os_options)]
        matched = await self.param_extractor.match_user_selection(input_text, os_options)
        matched_data = json.loads(matched)
        
        if matched_data.get("matched"):
            os_idx = matched_data.get("matched_item")["id"]
            state.collected_params["operatingSystem"] = state._os_options[os_idx]
            conversation_state_manager.update_session(state)
            return None
        else:
            return {
                "agent_name": "ValidationAgent",
                "success": True,
                "output": "❌ I couldn't match that OS. Please choose from the list above:"
            }
    
    async def _handle_worker_pool_name(self, input_text: str, state: Any) -> Optional[Dict[str, Any]]:
        """Validate worker pool name format."""
        pool_name = input_text.strip().lower()
        if not re.match(r'^[a-z0-9]{1,5}$', pool_name):
            return {
                "agent_name": "ValidationAgent",
                "success": True,
                "output": "❌ Worker pool name must be 1-5 lowercase alphanumeric characters. Please try again:"
            }
        
        state.collected_params["workerPoolName"] = pool_name
        conversation_state_manager.update_session(state)
        return None
    
    async def _handle_node_type(self, input_text: str, state: Any) -> Optional[Dict[str, Any]]:
        """Fetch flavors and match node type selection."""
        if not hasattr(state, '_node_types'):
            zone_id = state.collected_params["_zone_id"]
            circuit_id = state._circuit_id
            os_model = state.collected_params["operatingSystem"]["os_model"]
            flavor_result = await api_executor_service.get_flavors(zone_id, circuit_id, os_model)
            state._node_types = flavor_result.get("node_types", [])
            state._all_flavors = flavor_result.get("flavors", [])
        
        # Map display names
        node_type_map = {
            "general": "generalPurpose",
            "compute": "computeOptimized",
            "memory": "memoryOptimized"
        }
        
        user_input_lower = input_text.lower()
        matched_type = None
        for key, value in node_type_map.items():
            if key in user_input_lower and value in state._node_types:
                matched_type = value
                break
        
        if matched_type:
            state.collected_params["nodeType"] = matched_type
            conversation_state_manager.update_session(state)
            return None
        else:
            return {
                "agent_name": "ValidationAgent",
                "success": True,
                "output": "❌ Please choose General Purpose, Compute Optimized, or Memory Optimized:"
            }
    
    async def _handle_flavor(self, input_text: str, state: Any) -> Optional[Dict[str, Any]]:
        """Filter and match flavor selection."""
        node_type = state.collected_params["nodeType"]
        filtered_flavors = [f for f in state._all_flavors if f["node_type"] == node_type]
        
        flavor_options = [{"id": f["id"], "name": f["name"]} for f in filtered_flavors]
        matched = await self.param_extractor.match_user_selection(input_text, flavor_options)
        matched_data = json.loads(matched)
        
        if matched_data.get("matched"):
            flavor_id = matched_data.get("matched_item")["id"]
            flavor_info = next(f for f in filtered_flavors if f["id"] == flavor_id)
            state.collected_params["flavor"] = flavor_info
            conversation_state_manager.update_session(state)
            return None
        else:
            return {
                "agent_name": "ValidationAgent",
                "success": True,
                "output": "❌ I couldn't match that flavor. Please choose from the list above:"
            }
    
    async def _handle_replica_count(self, input_text: str, state: Any) -> Optional[Dict[str, Any]]:
        """Validate replica count."""
        try:
            count = int(input_text.strip())
            if 1 <= count <= 8:
                state.collected_params["replicaCount"] = count
                conversation_state_manager.update_session(state)
                return None
            else:
                return {
                    "agent_name": "ValidationAgent",
                    "success": True,
                    "output": "❌ Replica count must be between 1 and 8. Please try again:"
                }
        except ValueError:
            return {
                "agent_name": "ValidationAgent",
                "success": True,
                "output": "❌ Please enter a number between 1 and 8:"
            }
    
    async def _handle_autoscaling(self, input_text: str, state: Any) -> Optional[Dict[str, Any]]:
        """Handle autoscaling yes/no."""
        user_response = input_text.lower().strip()
        if "yes" in user_response or "enable" in user_response:
            state.collected_params["enableAutoscaling"] = True
            conversation_state_manager.update_session(state)
            return None
        elif "no" in user_response or "skip" in user_response:
            state.collected_params["enableAutoscaling"] = False
            conversation_state_manager.update_session(state)
            return None
        else:
            return {
                "agent_name": "ValidationAgent",
                "success": True,
                "output": "❌ Please answer 'yes' or 'no' for autoscaling:"
            }
    
    async def _handle_max_replicas(self, input_text: str, state: Any) -> Optional[Dict[str, Any]]:
        """Validate max replicas."""
        try:
            max_count = int(input_text.strip())
            min_count = state.collected_params["replicaCount"]
            if min_count <= max_count <= 8:
                state.collected_params["maxReplicas"] = max_count
                conversation_state_manager.update_session(state)
                return None
            else:
                return {
                    "agent_name": "ValidationAgent",
                    "success": True,
                    "output": f"❌ Max replicas must be between {min_count} and 8. Please try again:"
                }
        except ValueError:
            min_count = state.collected_params["replicaCount"]
            return {
                "agent_name": "ValidationAgent",
                "success": True,
                "output": f"❌ Please enter a number between {min_count} and 8:"
            }
    
    async def _handle_tags(self, input_text: str, state: Any) -> Optional[Dict[str, Any]]:
        """Handle tags (skip for MVP)."""
        user_response = input_text.lower().strip()
        if "no" in user_response or "skip" in user_response:
            state.collected_params["tags"] = []
            conversation_state_manager.update_session(state)
            return None
        else:
            # For MVP, just skip tags
            state.collected_params["tags"] = []
            conversation_state_manager.update_session(state)
            return None
    
    async def _ask_for_parameter(self, param_name: str, state: Any) -> Dict[str, Any]:
        """
        Ask user for a specific parameter with context and available options.
        
        Args:
            param_name: Name of parameter to ask for
            state: Conversation state
            
        Returns:
            Dict with prompt for user
        """
        logger.info(f"❓ Asking for parameter: {param_name}")
        
        state.last_asked_param = param_name
        # Persist state with last_asked_param so it survives restarts
        conversation_state_manager.update_session(state)
        
        # Build prompts with available options
        if param_name == "clusterName":
            output = "**Step 1/15**: What would you like to name your cluster?\n\n📝 Requirements: Start with a letter, 3-18 characters (letters, numbers, hyphens)"
        
        elif param_name == "datacenter":
            if not hasattr(state, '_datacenter_options'):
                engagement_id = await api_executor_service.get_engagement_id()
                dc_result = await api_executor_service.get_iks_images_and_datacenters(engagement_id)
                state._datacenter_options = dc_result.get("datacenters", [])
                state._all_images = dc_result.get("images", [])
            
            # Check if we have a detected location from initial extraction
            if hasattr(state, '_detected_location') and state._detected_location:
                detected = state._detected_location.lower()
                logger.info(f"🔍 Trying to auto-match detected location: '{detected}'")
                
                for dc in state._datacenter_options:
                    dc_name_lower = dc.get('name', '').lower()
                    if detected in dc_name_lower or dc_name_lower.startswith(detected):
                        # Found a match! Auto-select it
                        state.collected_params["datacenter"] = dc
                        state.collected_params["_datacenter_id"] = dc["id"]
                        # Clear the hint
                        del state._detected_location
                        conversation_state_manager.update_session(state)
                        logger.info(f"✅ Auto-matched datacenter: {dc['name']}")
                        
                        # Ask for the next param instead
                        next_param = self._find_next_parameter(state)
                        if next_param:
                            return await self._ask_for_parameter(next_param, state)
                        else:
                            return self._build_summary(state)
            
            dc_list = "\n".join([f"  • {dc['name']}" for dc in state._datacenter_options])
            output = f"**Step 2/15**: Which data center would you like to deploy the cluster in?\n\n📍 **Available data centers:**\n{dc_list}"
        
        elif param_name == "k8sVersion":
            dc_id = state.collected_params["_datacenter_id"]
            versions = await api_executor_service.get_k8s_versions_for_datacenter(dc_id, state._all_images)
            state._k8s_versions = versions
            
            version_list = "\n".join([f"  • {v}" for v in versions])
            output = f"**Step 3/15**: Which Kubernetes version would you like to use?\n\n🎯 **Available versions:**\n{version_list}"
        
        elif param_name == "cniDriver":
            dc_id = state.collected_params["_datacenter_id"]
            k8s_version = state.collected_params["k8sVersion"]
            driver_result = await api_executor_service.get_network_drivers(dc_id, k8s_version)
            state._cni_drivers = driver_result.get("drivers", [])
            
            driver_list = "\n".join([f"  • {d}" for d in state._cni_drivers])
            output = f"**Step 4/15**: Which CNI (Container Network Interface) driver?\n\n🌐 **Available drivers:**\n{driver_list}"
        
        elif param_name == "businessUnit":
            # Fetch all business units using the correct API
            logger.info(f"🏢 Fetching business units from API...")
            bu_result = await api_executor_service.get_business_units_list(force_refresh=True)
            logger.info(f"🏢 API result: success={bu_result.get('success')}, departments count={len(bu_result.get('departments', []))}")
            
            if not bu_result.get("success"):
                logger.error(f"❌ Failed to fetch business units: {bu_result.get('error')}")
            
            all_departments = bu_result.get("departments", [])
            
            # Filter by selected datacenter's endpoint ID
            selected_endpoint_id = state.collected_params.get("_datacenter_id")
            datacenter_name = state.collected_params.get("datacenter", {}).get("name", "selected")
            logger.info(f"🏢 Filtering business units for endpoint ID: {selected_endpoint_id}")
            
            # Filter departments that match the selected endpoint
            filtered_bus = []
            for dept in all_departments:
                dept_endpoint_id = dept.get("endpoint", {}).get("id")
                if dept_endpoint_id == selected_endpoint_id:
                    filtered_bus.append({
                        "id": dept["id"],
                        "name": dept["name"],
                        "endpoint_id": dept_endpoint_id,
                        "location": dept.get("endpoint", {}).get("location")
                    })
            
            logger.info(f"🏢 Found {len(filtered_bus)} business units for endpoint {selected_endpoint_id} (from {len(all_departments)} total)")
            state._business_units = filtered_bus
            state._all_departments = all_departments
            
            # Also fetch environments for later use
            env_result = await api_executor_service.get_environments_list()
            state._all_environments = env_result.get("environments", [])
            
            if filtered_bus:
                bu_list = "\n".join([f"  • {bu['name']}" for bu in filtered_bus])
                output = f"**Step 5/15**: Which business unit should this cluster belong to?\n\n🏢 **Available business units** (for {datacenter_name}):\n{bu_list}"
            else:
                output = f"**Step 5/15**: ⚠️ No business units found for datacenter {datacenter_name}.\n\nPlease contact your administrator to create a business unit for this location."
        
        elif param_name == "environment":
            # Debug: Log the actual businessUnit value
            bu_value = state.collected_params.get("businessUnit")
            logger.info(f"🔍 DEBUG businessUnit value type: {type(bu_value)}, value: {bu_value}")
            
            # Handle both dict and string cases
            if isinstance(bu_value, dict):
                bu_id = bu_value.get("id")
                bu_name = bu_value.get("name")
            else:
                # If it's a string, try to find the matching BU from cached data
                logger.warning(f"⚠️ businessUnit is a string, not a dict. Trying to resolve...")
                bu_name = bu_value
                bu_id = None
                if hasattr(state, '_business_units'):
                    for bu in state._business_units:
                        if bu.get("name") == bu_value:
                            bu_id = bu.get("id")
                            # Fix the stored value
                            state.collected_params["businessUnit"] = bu
                            logger.info(f"✅ Fixed businessUnit to dict: {bu}")
                            break
            
            logger.info(f"🏢 Using BU: name={bu_name}, id={bu_id}")
            
            # Ensure environments are fetched
            if not hasattr(state, '_all_environments') or not state._all_environments:
                logger.info(f"🔄 Fetching environments (not cached in state)")
                env_result = await api_executor_service.get_environments_list()
                state._all_environments = env_result.get("environments", [])
                logger.info(f"✅ Fetched {len(state._all_environments)} environments")
            
            # Filter environments by business unit
            filtered_envs = []
            for env in state._all_environments:
                env_dept_id = env.get("departmentId") or env.get("department_id")
                if env_dept_id == bu_id:
                    filtered_envs.append(env)
            
            logger.info(f"🔧 Found {len(filtered_envs)} environments for BU '{bu_name}'")
            
            if filtered_envs:
                env_list = "\n".join([f"  • {env['name']}" for env in filtered_envs])
                output = f"**Step 6/15**: Which environment is this cluster for?\n\n🔧 **Available environments** (for {bu_name}):\n{env_list}"
            else:
                output = f"**Step 6/15**: ⚠️ No environments found for business unit '{bu_name}'.\n\nPlease contact your administrator to create an environment for this business unit."
        
        elif param_name == "zone":
            engagement_id = await api_executor_service.get_engagement_id()
            zone_result = await api_executor_service.get_zones_list(engagement_id)
            state._zones = zone_result.get("zones", [])
            
            bu_value = state.collected_params.get("businessUnit")
            bu_name = bu_value.get("name") if isinstance(bu_value, dict) else bu_value
            env_name = state.collected_params.get("_environment_name", "")
            filtered_zones = [z for z in state._zones 
                             if z["departmentName"] == bu_name and z["environmentName"] == env_name]
            
            zone_list = "\n".join([f"  • {z['zoneName']}" for z in filtered_zones])
            output = f"**Step 7/15**: Which network zone (VLAN) should the cluster use?\n\n🗺️ **Available zones:**\n{zone_list}"
        
        elif param_name == "operatingSystem":
            zone_id = state.collected_params["_zone_id"]
            circuit_id = await api_executor_service.get_circuit_id(None)
            k8s_version = state.collected_params["k8sVersion"]
            os_result = await api_executor_service.get_os_images(zone_id, circuit_id, k8s_version)
            state._os_options = os_result.get("os_options", [])
            state._circuit_id = circuit_id
            
            os_list = "\n".join([f"  • {opt['display_name']}" for opt in state._os_options])
            output = f"**Step 8/15**: Which operating system for the worker nodes?\n\n💿 **Available OS options:**\n{os_list}"
        
        elif param_name == "workerPoolName":
            output = "**Step 9/15**: What would you like to name this worker node pool?\n\n📝 Requirements: 1-5 lowercase alphanumeric characters"
        
        elif param_name == "nodeType":
            zone_id = state.collected_params["_zone_id"]
            circuit_id = state._circuit_id
            os_model = state.collected_params["operatingSystem"]["os_model"]
            flavor_result = await api_executor_service.get_flavors(zone_id, circuit_id, os_model)
            state._node_types = flavor_result.get("node_types", [])
            state._all_flavors = flavor_result.get("flavors", [])
            
            type_display = {
                "generalPurpose": "General Purpose - Balanced compute, memory, and storage",
                "computeOptimized": "Compute Optimized - Higher CPU-to-memory ratio",
                "memoryOptimized": "Memory Optimized - Higher memory-to-CPU ratio"
            }
            type_list = "\n".join([f"  • {type_display.get(t, t)}" for t in state._node_types])
            output = f"**Step 10/15**: What type of worker nodes do you need?\n\n💻 **Available types:**\n{type_list}"
        
        elif param_name == "flavor":
            node_type = state.collected_params["nodeType"]
            filtered_flavors = [f for f in state._all_flavors if f["node_type"] == node_type]
            
            flavor_list = "\n".join([f"  • {f['name']}" for f in filtered_flavors])
            output = f"**Step 11/15**: Which compute configuration?\n\n⚡ **Available flavors:**\n{flavor_list}"
        
        elif param_name == "replicaCount":
            output = "**Step 12/15**: How many worker nodes?\n\n📊 Enter a number between 1 and 8:"
        
        elif param_name == "enableAutoscaling":
            output = "**Step 13/15**: Would you like to enable autoscaling?\n\n🔄 This allows automatic scaling based on load. Answer 'yes' or 'no':"
        
        elif param_name == "maxReplicas":
            min_count = state.collected_params["replicaCount"]
            output = f"**Step 14/15**: What should be the maximum number of replicas when autoscaling?\n\n📈 Enter a number between {min_count} and 8:"
        
        elif param_name == "tags":
            output = "**Step 15/15**: Would you like to add any tags (key-value pairs for organization)?\n\n🏷️ Answer 'yes' or 'no' (you can skip for now):"
        
        else:
            output = f"Please provide: {param_name}"
        
        return {
            "agent_name": "ValidationAgent",
            "success": True,
            "output": output
        }

