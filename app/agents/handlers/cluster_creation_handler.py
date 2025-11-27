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
        
        # If user provided input AND we previously asked for a parameter, process it
        if input_text and hasattr(state, 'last_asked_param') and state.last_asked_param:
            logger.info(f"📥 Processing response for: {state.last_asked_param}")
            result = await self._process_user_input(input_text, state)
            if result:
                # Error or validation failure - ask again
                return result
        
        # Find the next parameter to collect (after processing or on initial call)
        next_param = self._find_next_parameter(state)
        
        # If all parameters collected, mark as ready
        if next_param is None:
            return self._build_summary(state)
        
        # Ask for the next parameter
        return await self._ask_for_parameter(next_param, state)
    
    def _find_next_parameter(self, state: Any) -> Optional[str]:
        """
        Find the next parameter that needs to be collected.
        
        Args:
            state: Conversation state
            
        Returns:
            Next parameter name or None if all collected
        """
        for param in self.workflow:
            if param not in state.collected_params:
                # Skip optional params based on user choice
                if param == "maxReplicas" and not state.collected_params.get("enableAutoscaling"):
                    continue
                    
                return param
        
        return None
    
    def _build_summary(self, state: Any) -> Dict[str, Any]:
        """
        Build final summary when all parameters are collected.
        
        Args:
            state: Conversation state
            
        Returns:
            Dict with summary and ready_to_execute flag
        """
        logger.info(f"✅ All cluster creation parameters collected!")
        state.status = "READY_TO_EXECUTE"
        state.missing_params = []
        
        params = state.collected_params
        autoscaling_info = ""
        if params.get("enableAutoscaling"):
            autoscaling_info = f" (autoscaling up to {params.get('maxReplicas', 8)} nodes)"
        
        summary = f"""
**🎉 Cluster Configuration Complete!**

**Basic Configuration:**
- **Cluster Name**: `{params['clusterName']}`
- **Datacenter**: {params['datacenter']['name']}
- **Kubernetes Version**: {params['k8sVersion']}
- **CNI Driver**: {params['cniDriver']}

**Network Setup:**
- **Business Unit**: {params['businessUnit']['name']}
- **Environment**: {params['environment']['name']}
- **Zone**: {params['zone']['name']}

**Worker Node Configuration:**
- **Operating System**: {params['operatingSystem']['display_name']}
- **Worker Pool Name**: `{params['workerPoolName']}`
- **Node Type**: {params.get('nodeType', 'N/A').replace('generalPurpose', 'General Purpose').replace('computeOptimized', 'Compute Optimized').replace('memoryOptimized', 'Memory Optimized')}
- **Flavor**: {params['flavor']['name']}
- **Node Count**: {params['replicaCount']}{autoscaling_info}

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
        return None
    
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
            return None
        else:
            return {
                "agent_name": "ValidationAgent",
                "success": True,
                "output": "❌ Please select one of the CNI drivers listed above:"
            }
    
    async def _handle_business_unit(self, input_text: str, state: Any) -> Optional[Dict[str, Any]]:
        """Fetch and match business unit selection."""
        if not hasattr(state, '_business_units'):
            engagement_id = await api_executor_service.get_engagement_id()
            env_result = await api_executor_service.get_environments_and_business_units(engagement_id)
            state._business_units = env_result.get("business_units", [])
            state._all_environments = env_result.get("environments", [])
        
        matched = await self.param_extractor.match_user_selection(input_text, state._business_units)
        matched_data = json.loads(matched)
        
        if matched_data.get("matched"):
            bu_info = matched_data.get("matched_item")
            state.collected_params["businessUnit"] = bu_info
            return None
        else:
            return {
                "agent_name": "ValidationAgent",
                "success": True,
                "output": "❌ I couldn't match that business unit. Please choose from the list above:"
            }
    
    async def _handle_environment(self, input_text: str, state: Any) -> Optional[Dict[str, Any]]:
        """Filter and match environment selection."""
        bu_id = state.collected_params["businessUnit"]["id"]
        filtered_envs = [env for env in state._all_environments if env["departmentId"] == bu_id]
        
        env_options = [{"id": env["id"], "name": env["name"]} for env in filtered_envs]
        matched = await self.param_extractor.match_user_selection(input_text, env_options)
        matched_data = json.loads(matched)
        
        if matched_data.get("matched"):
            env_info = matched_data.get("matched_item")
            full_env = next(e for e in filtered_envs if e["id"] == env_info["id"])
            state.collected_params["environment"] = {"id": full_env["id"], "name": full_env["name"]}
            state.collected_params["_environment_name"] = full_env["name"]
            return None
        else:
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
        
        bu_name = state.collected_params["businessUnit"]["name"]
        env_name = state.collected_params["_environment_name"]
        filtered_zones = [z for z in state._zones 
                         if z["departmentName"] == bu_name and z["environmentName"] == env_name]
        
        zone_options = [{"id": z["zoneId"], "name": z["zoneName"]} for z in filtered_zones]
        matched = await self.param_extractor.match_user_selection(input_text, zone_options)
        matched_data = json.loads(matched)
        
        if matched_data.get("matched"):
            zone_info = matched_data.get("matched_item")
            state.collected_params["zone"] = zone_info
            state.collected_params["_zone_id"] = zone_info["id"]
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
            return None
        elif "no" in user_response or "skip" in user_response:
            state.collected_params["enableAutoscaling"] = False
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
            return None
        else:
            # For MVP, just skip tags
            state.collected_params["tags"] = []
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
        
        # Build prompts with available options
        if param_name == "clusterName":
            output = "**Step 1/15**: What would you like to name your cluster?\n\n📝 Requirements: Start with a letter, 3-18 characters (letters, numbers, hyphens)"
        
        elif param_name == "datacenter":
            if not hasattr(state, '_datacenter_options'):
                engagement_id = await api_executor_service.get_engagement_id()
                dc_result = await api_executor_service.get_iks_images_and_datacenters(engagement_id)
                state._datacenter_options = dc_result.get("datacenters", [])
                state._all_images = dc_result.get("images", [])
            
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
            engagement_id = await api_executor_service.get_engagement_id()
            env_result = await api_executor_service.get_environments_and_business_units(engagement_id)
            state._business_units = env_result.get("business_units", [])
            state._all_environments = env_result.get("environments", [])
            
            bu_list = "\n".join([f"  • {bu['name']}" for bu in state._business_units])
            output = f"**Step 5/15**: Which business unit should this cluster belong to?\n\n🏢 **Available business units:**\n{bu_list}"
        
        elif param_name == "environment":
            bu_id = state.collected_params["businessUnit"]["id"]
            filtered_envs = [env for env in state._all_environments if env["departmentId"] == bu_id]
            
            env_list = "\n".join([f"  • {env['name']}" for env in filtered_envs])
            output = f"**Step 6/15**: Which environment is this cluster for?\n\n🔧 **Available environments:**\n{env_list}"
        
        elif param_name == "zone":
            engagement_id = await api_executor_service.get_engagement_id()
            zone_result = await api_executor_service.get_zones_list(engagement_id)
            state._zones = zone_result.get("zones", [])
            
            bu_name = state.collected_params["businessUnit"]["name"]
            env_name = state.collected_params["_environment_name"]
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

