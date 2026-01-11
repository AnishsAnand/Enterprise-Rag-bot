"""
Cluster Creation Handler - Manages the multi-step workflow for creating Kubernetes clusters.

This handler encapsulates the 17-step customer workflow for cluster creation,
making it easier to maintain and test independently.
"""

import logging
import re
from typing import Any, Dict, Optional, List
import json

from app.services.api_executor_service import api_executor_service
from app.agents.tools.parameter_extraction import ParameterExtractor
from app.agents.state.conversation_state import conversation_state_manager, ConversationStatus

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
            "additionalStorage",  # optional - additional disk storage
            "replicaCount",
            "enableAutoscaling",  # optional
            "maxReplicas",  # conditional
            "tags"  # optional
        ]
    
    def _check_cancel_intent(self, input_text: str, state: Any) -> Optional[Dict[str, Any]]:
        """
        Check if user wants to cancel/exit the cluster creation workflow.
        
        Args:
            input_text: User's current input
            state: Conversation state
            
        Returns:
            Dict with cancel response if user wants to cancel, None otherwise
        """
        if not input_text:
            return None
        
        user_lower = input_text.lower().strip()
        
        # Check if we're expecting a yes/no answer (for autoscaling, additional storage, etc.)
        # In this case, "no" is a valid answer, not a cancel request
        yes_no_params = ["enableAutoscaling", "additionalStorage"]
        last_asked = getattr(state, 'last_asked_param', None)
        if last_asked in yes_no_params and user_lower in ["no", "yes", "n", "y"]:
            # This is an answer to a yes/no question, not a cancel request
            return None
        
        # Cancel/exit keywords
        cancel_keywords = [
            "cancel", "exit", "quit", "stop", "abort", "nevermind", "never mind",
            "forget it", "back", "go back", "start over", "restart", "end",
            "no thanks", "don't want", "dont want", "changed my mind"
        ]
        
        # Check if user wants to cancel
        is_cancel = any(keyword in user_lower for keyword in cancel_keywords)
        
        # Also check for exact matches of short words (but NOT "no" by itself - too ambiguous)
        exact_cancel_words = ["cancel", "exit", "quit", "stop", "abort", "end", "back"]
        is_exact_cancel = user_lower in exact_cancel_words
        
        if is_cancel or is_exact_cancel:
            logger.info(f"🚫 User requested to cancel cluster creation: '{input_text}'")
            
            # Clear the conversation state
            state.status = ConversationStatus.CANCELLED
            state.collected_params = {}
            state.missing_params = []
            if hasattr(state, 'last_asked_param'):
                state.last_asked_param = None
            
            # Persist the cancelled state
            conversation_state_manager.update_session(state)
            
            return {
                "agent_name": "ValidationAgent",
                "success": True,
                "cancelled": True,
                "output": "✅ **Cluster creation cancelled.**\n\nNo worries! I've cancelled the cluster creation process. Your progress has been discarded.\n\n💡 When you're ready, you can:\n- **Create a new cluster**: Just say \"create cluster\"\n- **List existing clusters**: Say \"list clusters\"\n- **Ask a question**: I'm here to help!"
            }
        
        return None
    
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
        # Check for cancel/exit intent first
        cancel_result = self._check_cancel_intent(input_text, state)
        if cancel_result:
            return cancel_result
        
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
                    
                    # Add summary of collected params
                    summary = self._format_collected_params_summary(state)
                    if summary:
                        result["output"] = result["output"] + "\n\n" + summary
                    
                    # Combine success message with next question
                    next_question = await self._ask_for_parameter(next_param, state)
                    result["output"] = result["output"] + "\n\n" + next_question["output"]
                    return result
                # Error or validation failure - ask again
                return result
            else:
                # Successful processing without explicit result - continue to next param
                # Add summary of collected params
                summary = self._format_collected_params_summary(state)
                next_param = self._find_next_parameter(state)
                if next_param is None:
                    return self._build_summary(state)
                
                next_question = await self._ask_for_parameter(next_param, state)
                if summary:
                    next_question["output"] = summary + "\n\n" + next_question["output"]
                return next_question
        
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
                check_result = await api_executor_service.execute_operation(
                    resource_type="k8s_cluster",
                    operation="check_cluster_name",
                    params={"clusterName": cluster_name}
                )
                # Parse response: empty data = available
                is_available = not check_result.get("data", {}).get("data", {})
                check_result = {"available": is_available, "success": check_result.get("success", False)}
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
                check_result = await api_executor_service.execute_operation(
                    resource_type="k8s_cluster",
                    operation="check_cluster_name",
                    params={"clusterName": cluster_name}
                )
                # Parse response: empty data = available
                is_available = not check_result.get("data", {}).get("data", {})
                check_result = {"available": is_available, "success": check_result.get("success", False)}
                
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
            Dict with summary and awaiting confirmation
        """
        logger.info(f"✅ All cluster creation parameters collected!")
        
        # Log final payload
        self._log_cluster_payload(state, "COMPLETE")
        
        # Set status to COLLECTING_PARAMS (we're awaiting confirmation)
        # Use last_asked_param = "_confirmation" to track that we're in confirmation step
        state.status = ConversationStatus.COLLECTING_PARAMS
        state.last_asked_param = "_confirmation"  # Special marker for confirmation step
        state.missing_params = []
        conversation_state_manager.update_session(state)
        
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
- **Storage**: {params.get('additionalStorage') or params.get('flavor', {}).get('disk_gb', 'N/A')} GB
- **Node Count**: {params.get('replicaCount', 'N/A')}{autoscaling_info}

**Master Nodes** (auto-configured):
- **Type**: Virtual Control Plane
- **Mode**: High Availability (3x D8 nodes)

---

⏱️ **Estimated creation time**: 5-10 minutes

**Please review the configuration above.**

Reply with:
- **"yes"** or **"proceed"** to create the cluster
- **"change [parameter]"** to modify a specific parameter (e.g., "change cluster name")
- **"cancel"** to abort cluster creation
"""
        
        return {
            "agent_name": "ValidationAgent",
            "success": True,
            "ready_to_execute": False,  # Not ready yet - awaiting confirmation
            "output": summary
        }
    
    def _check_for_special_commands(self, input_text: str, state: Any) -> Optional[Dict[str, Any]]:
        """
        Check if user wants to exit, cancel, or modify previous parameters.
        
        Args:
            input_text: User's input
            state: Conversation state
            
        Returns:
            Dict with response if special command detected, None otherwise
        """
        text_lower = input_text.lower().strip()
        
        # Exit/Cancel commands
        exit_keywords = ['exit', 'cancel', 'quit', 'stop', 'abort', 'nevermind', 'never mind']
        if any(keyword in text_lower for keyword in exit_keywords):
            # Clear the workflow state
            state.collected_params.clear()
            state.last_asked_param = None
            conversation_state_manager.update_session(state)
            return {
                "agent_name": "ValidationAgent",
                "success": True,
                "output": "❌ Cluster creation cancelled. All progress has been cleared. Let me know if you'd like to start over!"
            }
        
        # Go back / Change parameter commands
        change_keywords = ['go back', 'change', 'modify', 'update', 'edit']
        if any(keyword in text_lower for keyword in change_keywords):
            # Try to extract which parameter they want to change
            for param in self.workflow:
                param_lower = param.lower()
                # Check for parameter name mentions
                if param_lower in text_lower or self._get_param_display_name(param).lower() in text_lower:
                    # Clear this parameter and all its dependents (cascading)
                    self._clear_parameter_and_dependents(param, state)
                    
                    state.last_asked_param = None
                    conversation_state_manager.update_session(state)
                    
                    return {
                        "agent_name": "ValidationAgent",
                        "success": True,
                        "output": f"✅ Cleared '{self._get_param_display_name(param)}' and all dependent parameters. Let's start from there again."
                    }
            
            # Generic "go back" without specific parameter
            if 'back' in text_lower and state.collected_params:
                # Go back to the last collected parameter
                collected_params_list = [p for p in self.workflow if p in state.collected_params]
                if collected_params_list:
                    last_param = collected_params_list[-1]
                    del state.collected_params[last_param]
                    # Clear internal params too
                    internal_key = f"_{last_param}_id"
                    if internal_key in state.collected_params:
                        del state.collected_params[internal_key]
                    
                    state.last_asked_param = None
                    conversation_state_manager.update_session(state)
                    
                    return {
                        "agent_name": "ValidationAgent",
                        "success": True,
                        "output": f"✅ Removed the last parameter ('{self._get_param_display_name(last_param)}'). Let's continue from there."
                    }
        
        return None
    
    def _clear_parameter_and_dependents(self, param_name: str, state: Any) -> None:
        """
        Clear a parameter and all its dependent parameters based on dependency chain.
        
        Dependency chain:
        - datacenter → businessUnit → environment → zone → operatingSystem, nodeType, flavor
        - businessUnit → environment → zone → operatingSystem, nodeType, flavor
        - environment → zone → operatingSystem, nodeType, flavor
        - zone → operatingSystem, nodeType, flavor
        - nodeType → flavor
        - k8sVersion → cniDriver (and potentially OS/flavors if they depend on version)
        
        Args:
            param_name: Name of parameter to clear
            state: Conversation state (modified in place)
        """
        logger.info(f"🧹 Clearing parameter '{param_name}' and its dependents")
        
        # Define dependency mapping: param -> list of dependent params to clear
        dependencies = {
            "datacenter": [
                "businessUnit", "environment", "zone", 
                "operatingSystem", "nodeType", "flavor", "additionalStorage"
            ],
            "businessUnit": [
                "environment", "zone", 
                "operatingSystem", "nodeType", "flavor", "additionalStorage"
            ],
            "environment": [
                "zone", "operatingSystem", "nodeType", "flavor", "additionalStorage"
            ],
            "zone": [
                "operatingSystem", "nodeType", "flavor", "additionalStorage"
            ],
            "k8sVersion": [
                "cniDriver"  # CNI driver depends on k8s version
            ],
            "nodeType": [
                "flavor", "additionalStorage"  # Flavor depends on node type
            ],
            "operatingSystem": [
                "nodeType", "flavor", "additionalStorage"  # Node type/flavor depend on OS
            ]
        }
        
        # Get list of params to clear (including the param itself)
        params_to_clear = [param_name]
        if param_name in dependencies:
            params_to_clear.extend(dependencies[param_name])
        
        # Clear each parameter and its internal tracking params
        for param in params_to_clear:
            if param in state.collected_params:
                del state.collected_params[param]
                logger.info(f"  ✓ Cleared: {param}")
            
            # Clear internal tracking params
            internal_keys = [
                f"_{param}_id",
                f"_{param}_name",
                f"_{param}_validated"
            ]
            for key in internal_keys:
                if key in state.collected_params:
                    del state.collected_params[key]
        
        # Clear cached options that depend on the cleared params
        if param_name == "datacenter":
            if hasattr(state, '_business_units'):
                delattr(state, '_business_units')
            if hasattr(state, '_department_details'):
                delattr(state, '_department_details')
        elif param_name == "businessUnit":
            if hasattr(state, '_business_units'):
                delattr(state, '_business_units')
        elif param_name == "environment":
            # No specific cache to clear
            pass
        elif param_name == "zone":
            if hasattr(state, '_os_options'):
                delattr(state, '_os_options')
            if hasattr(state, '_node_types'):
                delattr(state, '_node_types')
            if hasattr(state, '_all_flavors'):
                delattr(state, '_all_flavors')
        elif param_name == "k8sVersion":
            if hasattr(state, '_k8s_versions'):
                delattr(state, '_k8s_versions')
            if hasattr(state, '_cni_drivers'):
                delattr(state, '_cni_drivers')
        elif param_name == "nodeType":
            if hasattr(state, '_node_types'):
                delattr(state, '_node_types')
            if hasattr(state, '_all_flavors'):
                delattr(state, '_all_flavors')
        
        logger.info(f"✅ Cleared {param_name} and {len(params_to_clear) - 1} dependent parameter(s)")
    
    async def _safe_match_selection(
        self,
        input_text: str,
        available_options: List[Dict[str, Any]],
        param_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Safely match user selection with proper error handling.
        
        Args:
            input_text: User's input
            available_options: List of available options to match against
            param_name: Name of parameter being matched (for error messages)
            
        Returns:
            Dict with matched_item if successful, None if no match or error
        """
        try:
            if not available_options:
                logger.error(f"❌ No {param_name} options available")
                return None
            
            matched = await self.param_extractor.match_user_selection(input_text, available_options)
            matched_data = json.loads(matched)
            
            if not matched_data or not matched_data.get("matched"):
                logger.info(f"❌ No match found for {param_name}: '{input_text}'")
                return None
            
            matched_item = matched_data.get("matched_item")
            if not matched_item:
                logger.error(f"❌ LLM returned matched=true but matched_item is None for {param_name}")
                return None
            
            if not isinstance(matched_item, dict) or "id" not in matched_item:
                logger.error(f"❌ matched_item missing 'id' field for {param_name}: {matched_item}")
                return None
            
            logger.info(f"✅ Matched {param_name}: {matched_item.get('name')} (ID: {matched_item['id']})")
            return matched_item
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse LLM matching response for {param_name}: {e}")
            logger.error(f"   Raw response: {matched[:200] if 'matched' in locals() and matched else 'None'}")
            return None
        except Exception as e:
            logger.error(f"❌ Error matching {param_name}: {e}", exc_info=True)
            return None
    
    def _get_param_display_name(self, param_name: str) -> str:
        """Get user-friendly display name for parameter."""
        display_names = {
            "clusterName": "Cluster Name",
            "datacenter": "Datacenter",
            "k8sVersion": "Kubernetes Version",
            "cniDriver": "CNI Driver",
            "businessUnit": "Business Unit",
            "environment": "Environment",
            "zone": "Zone",
            "operatingSystem": "Operating System",
            "workerPoolName": "Worker Pool Name",
            "nodeType": "Node Type",
            "flavor": "Flavor",
            "additionalStorage": "Additional Storage",
            "replicaCount": "Replica Count",
            "enableAutoscaling": "Enable Autoscaling",
            "maxReplicas": "Max Replicas",
            "tags": "Tags"
        }
        return display_names.get(param_name, param_name)
    
    def _format_collected_params_summary(self, state: Any) -> str:
        """
        Format a pretty-printed summary of collected parameters for user display.
        
        Args:
            state: Conversation state
            
        Returns:
            Formatted string with collected parameters
        """
        if not state.collected_params:
            return ""
        
        # Build summary
        summary_lines = ["📋 **Current Configuration:**"]
        
        for param in self.workflow:
            if param in state.collected_params and not param.startswith('_'):
                value = state.collected_params[param]
                display_name = self._get_param_display_name(param)
                
                # Format value based on type
                if isinstance(value, dict):
                    # For complex objects like flavor, show the name
                    display_value = value.get('name') or value.get('display_name') or str(value)
                elif isinstance(value, bool):
                    display_value = "Yes" if value else "No"
                else:
                    display_value = str(value)
                
                summary_lines.append(f"  ✓ **{display_name}**: {display_value}")
        
        return "\n".join(summary_lines)
    
    async def _process_user_input(self, input_text: str, state: Any) -> Optional[Dict[str, Any]]:
        """
        Process user's input for the last asked parameter.
        
        Args:
            input_text: User's response
            state: Conversation state
            
        Returns:
            Dict with result or None to continue
        """
        # Check for special commands first (exit, cancel, go back, etc.)
        special_cmd = self._check_for_special_commands(input_text, state)
        if special_cmd:
            return special_cmd
        
        last_param = state.last_asked_param
        logger.info(f"📝 Processing user input for: {last_param}")
        
        # Special handler for confirmation step
        if last_param == "_confirmation":
            return await self._handle_confirmation(input_text, state)
        
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
            "additionalStorage": self._handle_additional_storage,
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
            # Get IPC engagement ID first
            ipc_engagement_id = await api_executor_service.get_ipc_engagement_id(engagement_id)
            
            # Get IKS images with datacenters
            dc_result = await api_executor_service.execute_operation(
                resource_type="k8s_cluster",
                operation="get_iks_images",
                params={"ipc_engagement_id": ipc_engagement_id}
            )
            
            # Extract datacenters from images
            if dc_result.get("success") and dc_result.get("data"):
                api_data = dc_result["data"]
                if api_data.get("status") == "success" and api_data.get("data"):
                    all_images = []
                    for category, images in api_data["data"].items():
                        if isinstance(images, list):
                            all_images.extend(images)
                    
                    # Extract unique datacenters
                    datacenters = {}
                    for img in all_images:
                        dc_id = img.get("endpointId")
                        if dc_id and dc_id not in datacenters:
                            datacenters[dc_id] = {
                                "id": dc_id,
                                "name": img.get("endpointName", f"DC-{dc_id}"),
                                "endpoint": img.get("endpoint", "")
                            }
                    
                    dc_result = {
                        "success": True,
                        "datacenters": list(datacenters.values()),
                        "images": all_images
                    }
                else:
                    dc_result = {"success": False, "datacenters": [], "images": []}
            else:
                dc_result = {"success": False, "datacenters": [], "images": []}
            state._datacenter_options = dc_result.get("datacenters", [])
            state._all_images = dc_result.get("images", [])
        
        # Check if we have datacenter options
        if not state._datacenter_options:
            logger.error("❌ No datacenter options available")
            return {
                "agent_name": "ValidationAgent",
                "success": True,
                "output": "❌ No datacenters are currently available. Please contact your administrator."
            }
        
        # Match user selection using LLM (with safe error handling)
        dc_info = await self._safe_match_selection(input_text, state._datacenter_options, "datacenter")
        
        if dc_info:
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
        """Match Kubernetes version selection using intelligent LLM matching."""
        if not hasattr(state, '_k8s_versions'):
            dc_id = state.collected_params["_datacenter_id"]
            # Extract k8s versions from images for this datacenter
            dc_images = [img for img in state._all_images if img.get("endpointId") == dc_id]
            import re
            version_set = set()
            for img in dc_images:
                match = re.search(r'v\d+\.\d+\.\d+', img.get("ImageName", ""))
                if match:
                    version_set.add(match.group(0))
            
            # Sort semantically (latest first)
            versions = sorted(list(version_set), key=lambda v: [int(x) for x in v[1:].split('.')], reverse=True)
            
            state._k8s_versions = versions
        
        # Clean input (remove bullet chars, extra whitespace)
        cleaned_input = input_text.strip().lstrip('•·-*').strip()
        
        # Direct match first
        if cleaned_input in state._k8s_versions:
            state.collected_params["k8sVersion"] = cleaned_input
            conversation_state_manager.update_session(state)
            return None
        
        # Use LLM for intelligent matching (e.g., "1.30" → "v1.30.9")
        version_options = [{"id": v, "name": v} for v in state._k8s_versions]
        matched = await self.param_extractor.match_user_selection(cleaned_input, version_options)
        matched_data = json.loads(matched)
        
        if matched_data.get("matched") and matched_data.get("matched_item"):
            matched_version = matched_data["matched_item"]["id"]
            state.collected_params["k8sVersion"] = matched_version
            logger.info(f"✅ LLM matched k8s version: '{cleaned_input}' → '{matched_version}'")
            conversation_state_manager.update_session(state)
            return None
        else:
            versions_list = ", ".join(state._k8s_versions[:5])  # Show first 5
            return {
                "agent_name": "ValidationAgent",
                "success": True,
                "output": f"❌ Could not match '{cleaned_input}' to available versions.\n\nAvailable: {versions_list}..."
            }
    
    async def _handle_cni_driver(self, input_text: str, state: Any) -> Optional[Dict[str, Any]]:
        """Match CNI driver selection using intelligent LLM matching."""
        if not hasattr(state, '_cni_drivers'):
            dc_id = state.collected_params["_datacenter_id"]
            k8s_version = state.collected_params["k8sVersion"]
            
            logger.info(f"🌐 Fetching CNI drivers for endpoint {dc_id}, k8s version {k8s_version}")
            
            driver_result = await api_executor_service.execute_operation(
                resource_type="k8s_cluster",
                operation="get_network_list",
                params={"endpointId": dc_id, "k8sVersion": k8s_version}
            )
            
            # Parse response to extract drivers list
            # API returns: {"status": "success", "data": {"data": ["calico-v3.29.3", ...], "status": "success"}}
            drivers = []
            if driver_result.get("success") and driver_result.get("data"):
                api_data = driver_result["data"]
                if api_data.get("status") == "success" and api_data.get("data"):
                    inner_data = api_data["data"]
                    if isinstance(inner_data, dict):
                        drivers = inner_data.get("data", [])
                    elif isinstance(inner_data, list):
                        drivers = inner_data
            
            logger.info(f"✅ Extracted {len(drivers)} CNI drivers: {drivers}")
            state._cni_drivers = drivers
        
        # Clean input (remove bullet chars, extra whitespace)
        cleaned_input = input_text.strip().lstrip('•·-*').strip()
        
        # Direct match first
        if cleaned_input in state._cni_drivers:
            state.collected_params["cniDriver"] = cleaned_input
            conversation_state_manager.update_session(state)
            return None
        
        # Use LLM for intelligent matching
        driver_options = [{"id": d, "name": d} for d in state._cni_drivers]
        matched = await self.param_extractor.match_user_selection(cleaned_input, driver_options)
        matched_data = json.loads(matched)
        
        if matched_data.get("matched") and matched_data.get("matched_item"):
            matched_driver = matched_data["matched_item"]["id"]
            state.collected_params["cniDriver"] = matched_driver
            logger.info(f"✅ LLM matched CNI driver: '{cleaned_input}' → '{matched_driver}'")
            conversation_state_manager.update_session(state)
            return None
        else:
            driver_list = ", ".join(state._cni_drivers)
            return {
                "agent_name": "ValidationAgent",
                "success": True,
                "output": f"❌ Could not match '{cleaned_input}' to available drivers.\n\nPlease select one of: {driver_list}"
            }
    
    async def _handle_business_unit(self, input_text: str, state: Any) -> Optional[Dict[str, Any]]:
        """Fetch and match business unit selection (filtered by selected datacenter endpoint)."""
        if not hasattr(state, '_department_details') or not state._department_details:
            # Fetch full department hierarchy (BU -> Environment -> Zone)
            logger.info(f"🏢 Fetching department details with nested hierarchy...")
            # Get IPC engagement ID
            ipc_engagement_id = await api_executor_service.get_ipc_engagement_id()
            
            # Get department details with nested hierarchy
            dept_result = await api_executor_service.execute_operation(
                resource_type="k8s_cluster",
                operation="get_department_details",
                params={"ipc_engagement_id": ipc_engagement_id}
            )
            
            # Parse response
            if dept_result.get("success") and dept_result.get("data"):
                api_data = dept_result["data"]
                if api_data.get("status") == "success":
                    dept_data = api_data.get("data", {})
                    dept_result = {
                        "success": True,
                        "data": dept_data,
                        "departmentList": dept_data.get("departmentList", [])
                    }
                else:
                    dept_result = {"success": False, "departmentList": []}
            else:
                dept_result = {"success": False, "departmentList": []}
            logger.info(f"🏢 API result: success={dept_result.get('success')}, departments count={len(dept_result.get('departmentList', []))}")
            
            if not dept_result.get("success"):
                logger.error(f"❌ Failed to fetch department details: {dept_result.get('error')}")
            
            # Store full hierarchy for later use (environments, zones)
            state._department_details = dept_result.get("departmentList", [])
            
            # Filter by selected datacenter's endpoint ID
            selected_endpoint_id = state.collected_params.get("_datacenter_id")
            logger.info(f"🏢 Filtering departments for endpoint ID: {selected_endpoint_id}")
            
            # Filter departments that match the selected endpoint
            filtered_bus = []
            for dept in state._department_details:
                dept_endpoint_id = dept.get("endpointId")
                if dept_endpoint_id == selected_endpoint_id:
                    filtered_bus.append({
                        "id": dept["departmentId"],
                        "name": dept["departmentName"],
                        "endpoint_id": dept_endpoint_id,
                        "environmentList": dept.get("environmentList", [])  # Keep nested data
                    })
            
            logger.info(f"🏢 Found {len(filtered_bus)} business units for endpoint {selected_endpoint_id}")
            state._business_units = filtered_bus
        
        # Match user selection using LLM (with safe error handling)
        bu_info = await self._safe_match_selection(input_text, state._business_units, "businessUnit")
        
        if bu_info:
            # Find the full BU data with environmentList
            full_bu = next((bu for bu in state._business_units if bu["id"] == bu_info["id"]), bu_info)
            state.collected_params["businessUnit"] = full_bu
            logger.info(f"✅ Selected business unit: {full_bu['name']} (ID: {full_bu['id']}, {len(full_bu.get('environmentList', []))} environments)")
            conversation_state_manager.update_session(state)
            return None
        else:
            return {
                "agent_name": "ValidationAgent",
                "success": True,
                "output": "❌ I couldn't match that business unit. Please choose from the list above:"
            }
    
    async def _handle_environment(self, input_text: str, state: Any) -> Optional[Dict[str, Any]]:
        """Filter and match environment selection using nested data from selected business unit."""
        import traceback
        try:
            bu_value = state.collected_params.get("businessUnit")
            logger.info(f"🔍 _handle_environment: bu_value type={type(bu_value)}")
            
            # Get environments from the nested BU data
            if isinstance(bu_value, dict):
                bu_id = bu_value.get("id")
                bu_name = bu_value.get("name")
                # Get environments directly from the BU's nested data
                env_list = bu_value.get("environmentList", [])
                logger.info(f"🏢 BU '{bu_name}' has {len(env_list)} environments in nested data")
            else:
                logger.error(f"❌ businessUnit is not a dict: {bu_value}")
                return {
                    "agent_name": "ValidationAgent",
                    "success": True,
                    "output": "❌ Error: Business unit data not found. Please go back and select a business unit."
                }
            
            if not env_list:
                logger.warning(f"⚠️ No environments in BU '{bu_name}'")
                return {
                    "agent_name": "ValidationAgent",
                    "success": True,
                    "output": f"❌ No environments found for business unit '{bu_name}'. Please contact your administrator."
                }
            
            # Build options list for matching - use environmentId and environmentName
            env_options = [
                {"id": env["environmentId"], "name": env["environmentName"]} 
                for env in env_list
            ]
            logger.info(f"🔍 Matching '{input_text}' against {len(env_options)} environments: {[e['name'] for e in env_options]}")
            
            matched = await self.param_extractor.match_user_selection(input_text, env_options)
            matched_data = json.loads(matched)
            
            if matched_data.get("matched"):
                env_info = matched_data.get("matched_item")
                # Find the full environment data including zoneList
                full_env = next((e for e in env_list if e["environmentId"] == env_info["id"]), None)
                if full_env:
                    state.collected_params["environment"] = {
                        "id": full_env["environmentId"], 
                        "name": full_env["environmentName"],
                        "zoneList": full_env.get("zoneList", [])  # Keep zones for next step
                    }
                    state.collected_params["_environment_name"] = full_env["environmentName"]
                    state.collected_params["_department_id"] = bu_id
                    logger.info(f"✅ Selected environment: {full_env['environmentName']} (ID: {full_env['environmentId']}, {len(full_env.get('zoneList', []))} zones)")
                    conversation_state_manager.update_session(state)
                    return None
            
            return {
                "agent_name": "ValidationAgent",
                "success": True,
                "output": "❌ I couldn't match that environment. Please choose from the list above:"
            }
        except Exception as e:
            logger.error(f"❌ _handle_environment error: {e}")
            logger.error(f"🔍 Traceback:\n{traceback.format_exc()}")
            raise
    
    async def _handle_zone(self, input_text: str, state: Any) -> Optional[Dict[str, Any]]:
        """Match zone selection using nested data from selected environment."""
        env_value = state.collected_params.get("environment")
        logger.info(f"🔍 _handle_zone: env_value type={type(env_value)}")
        
        # Get zones from the nested environment data
        if isinstance(env_value, dict):
            env_name = env_value.get("name")
            # Get zones directly from the environment's nested data
            zone_list = env_value.get("zoneList", [])
            logger.info(f"🗺️ Environment '{env_name}' has {len(zone_list)} zones in nested data")
        else:
            logger.error(f"❌ environment is not a dict: {env_value}")
            return {
                "agent_name": "ValidationAgent",
                "success": True,
                "output": "❌ Error: Environment data not found. Please go back and select an environment."
            }
        
        if not zone_list:
            logger.warning(f"⚠️ No zones in environment '{env_name}'")
            return {
                "agent_name": "ValidationAgent",
                "success": True,
                "output": f"❌ No zones found for environment '{env_name}'. Please contact your administrator."
            }
        
        # Build options list for matching - use zoneId and zoneName
        zone_options = [
            {"id": zone["zoneId"], "name": zone["zoneName"]} 
            for zone in zone_list
        ]
        logger.info(f"🔍 Matching '{input_text}' against {len(zone_options)} zones: {[z['name'] for z in zone_options]}")
        
        matched = await self.param_extractor.match_user_selection(input_text, zone_options)
        matched_data = json.loads(matched)
        
        if matched_data.get("matched"):
            zone_info = matched_data.get("matched_item")
            state.collected_params["zone"] = zone_info
            state.collected_params["_zone_id"] = zone_info["id"]
            logger.info(f"✅ Selected zone: {zone_info['name']} (ID: {zone_info['id']})")
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
            k8s_version = state.collected_params["k8sVersion"]
            
            # Get OS images for zone
            os_result = await api_executor_service.execute_operation(
                resource_type="k8s_cluster",
                operation="get_os_images",
                params={"zoneId": zone_id}
            )
            
            # Parse and filter by k8s version
            if os_result.get("success") and os_result.get("data"):
                api_data = os_result["data"]
                if api_data.get("status") == "success" and api_data.get("data"):
                    images = api_data["data"].get("image", {}).get("options", [])
                    
                    # Filter by k8s version
                    version_patterns = [k8s_version]
                    if k8s_version.startswith("v"):
                        version_patterns.append(k8s_version[1:])
                    
                    filtered = [
                        img for img in images 
                        if any(p in (img.get("label", "") or img.get("ImageName", "")) for p in version_patterns)
                    ]
                    
                    # Group by osMake + osVersion
                    grouped = {}
                    for img in filtered:
                        os_make = img.get('osMake', 'Unknown')
                        os_version = img.get('osVersion', '')
                        key = f"{os_make} {os_version}".strip()
                        
                        if key not in grouped:
                            grouped[key] = {
                                "display_name": key,
                                "os_id": img.get("id"),
                                "os_make": os_make,
                                "os_model": img.get("osModel"),
                                "os_version": os_version,
                                "hypervisor": img.get("hypervisor"),
                                "image_id": img.get("IMAGEID"),
                                "image_name": img.get("ImageName")
                            }
                    
                    os_result = {"success": True, "os_options": list(grouped.values())}
                else:
                    os_result = {"success": False, "os_options": []}
            else:
                os_result = {"success": False, "os_options": []}
            state._os_options = os_result.get("os_options", [])
        
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
    
    def _get_node_type_display(self, node_type: str) -> str:
        """Get user-friendly display name for node type (like UI does)."""
        if not node_type:
            return node_type
        # Map API values to display names (matching createcluster.ts line 4456-4466)
        display_map = {
            'generalPurpose': 'General Purpose',
            'memoryOptimized': 'Memory Optimized',
            'computeOptimized': 'Compute Optimized'
        }
        return display_map.get(node_type, node_type)
    
    async def _handle_node_type(self, input_text: str, state: Any) -> Optional[Dict[str, Any]]:
        """Fetch flavors and match node type selection using intelligent LLM matching."""
        if not hasattr(state, '_node_types') or not state._node_types:
            zone_id = state.collected_params["_zone_id"]
            os_model = state.collected_params["operatingSystem"].get("os_model", "ubuntu")
            # Get flavors for zone
            flavor_result = await api_executor_service.execute_operation(
                resource_type="k8s_cluster",
                operation="get_flavors",
                params={"zoneId": zone_id}
            )
            
            # Parse and filter flavors
            if flavor_result.get("success") and flavor_result.get("data"):
                api_data = flavor_result["data"]
                if api_data.get("status") == "success" and api_data.get("data"):
                    all_flavors = api_data["data"].get("flavor", [])
                    
                    # Filter by applicationType = Container (strict match)
                    container_flavors = [f for f in all_flavors if f.get("applicationType") == "Container"]
                    
                    # Filter by OS model if provided
                    if os_model and container_flavors:
                        os_model_lower = os_model.lower()
                        filtered = [f for f in container_flavors 
                                   if os_model_lower in f.get("osModel", "").lower()]
                        if filtered:
                            container_flavors = filtered
                    
                    # Extract unique node types (raw flavorCategory values - no normalization)
                    node_types = list(set(f.get("flavorCategory") for f in container_flavors if f.get("flavorCategory")))
                    
                    # Format flavors
                    formatted_flavors = []
                    for flavor in container_flavors:
                        vcpu = flavor.get("vCpu", 0)
                        vram_mb = flavor.get("vRam", 0)
                        vram_gb = vram_mb // 1024 if vram_mb else 0
                        vdisk = flavor.get("vDisk", 0)
                        
                        formatted_flavors.append({
                            "id": flavor.get("artifactId"),
                            "name": f"{vcpu} vCPU / {vram_gb} GB RAM / {vdisk} GB Storage",
                            "display_name": flavor.get("display_name", flavor.get("FlavorName")),
                            "flavor_name": flavor.get("FlavorName"),
                            "sku_code": flavor.get("skuCode"),
                            "circuit_id": flavor.get("circuitId"),
                            "vcpu": vcpu,
                            "vram_gb": vram_gb,
                            "disk_gb": vdisk,
                            "node_type": flavor.get("flavorCategory"),
                            "storage_type": flavor.get("storageType"),
                            "os_model": flavor.get("osModel")
                        })
                    
                    flavor_result = {
                        "success": True,
                        "node_types": node_types,
                        "flavors": formatted_flavors
                    }
                else:
                    flavor_result = {"success": False, "node_types": [], "flavors": []}
            else:
                flavor_result = {"success": False, "node_types": [], "flavors": []}
            
            # Use raw node types from API (no transformation)
            state._node_types = flavor_result.get("node_types", [])
            state._all_flavors = flavor_result.get("flavors", [])
            
            logger.info(f"📋 Node types from API: {state._node_types}")
        
        # Clean input (remove bullet chars, extra whitespace)
        cleaned_input = input_text.strip().lstrip('•·-*').strip()
        
        # Build options - use RAW node type names (no pretty display)
        type_options = [
            {"id": nt, "name": nt}  # Use raw name as-is from API
            for nt in state._node_types
        ]
        
        # Use LLM for intelligent matching
        matched = await self.param_extractor.match_user_selection(cleaned_input, type_options)
        matched_data = json.loads(matched)
        
        if matched_data.get("matched") and matched_data.get("matched_item"):
            matched_type = matched_data["matched_item"]["id"]
            state.collected_params["nodeType"] = matched_type
            logger.info(f"✅ LLM matched node type: '{cleaned_input}' → '{matched_type}'")
            conversation_state_manager.update_session(state)
            return None
        else:
            # Show raw node types in error message
            types_list = ", ".join(state._node_types) if state._node_types else "General Purpose, Compute Optimized, or Memory Optimized"
            return {
                "agent_name": "ValidationAgent",
                "success": True,
                "output": f"❌ Could not match '{cleaned_input}' to available types.\n\nPlease choose one of: {types_list}"
            }
    
    async def _handle_flavor(self, input_text: str, state: Any) -> Optional[Dict[str, Any]]:
        """Filter and match flavor selection by node type."""
        node_type = state.collected_params["nodeType"]
        
        # Filter flavors by selected node type
        filtered_flavors = [f for f in state._all_flavors if f.get("node_type") == node_type]
        logger.info(f"🔍 Filtering flavors for node type '{node_type}': {len(filtered_flavors)} flavors")
        
        # Build options for matching - use the formatted name like "8 vCPU / 32 GB RAM / 100 GB Storage"
        flavor_options = [{"id": f["id"], "name": f["name"]} for f in filtered_flavors]
        matched = await self.param_extractor.match_user_selection(input_text, flavor_options)
        matched_data = json.loads(matched)
        
        if matched_data.get("matched"):
            flavor_id = matched_data.get("matched_item")["id"]
            flavor_info = next((f for f in filtered_flavors if f["id"] == flavor_id), None)
            if flavor_info:
                state.collected_params["flavor"] = flavor_info
                logger.info(f"✅ Selected flavor: {flavor_info['name']} (ID: {flavor_id})")
                conversation_state_manager.update_session(state)
                return None
        
        return {
            "agent_name": "ValidationAgent",
            "success": True,
            "output": "❌ I couldn't match that flavor. Please choose from the list above:"
        }
    
    async def _handle_additional_storage(self, input_text: str, state: Any) -> Optional[Dict[str, Any]]:
        """Handle optional additional disk storage."""
        user_response = input_text.lower().strip()
        
        # Get minimum storage from selected flavor
        flavor = state.collected_params.get("flavor", {})
        min_storage = flavor.get("disk_gb", 50)
        
        # Check for skip/no responses
        if any(word in user_response for word in ["no", "skip", "default", "none"]):
            state.collected_params["additionalStorage"] = None
            logger.info(f"⏭️ Skipping additional storage, using default: {min_storage} GB")
            conversation_state_manager.update_session(state)
            return None
        
        # Try to parse a number
        try:
            # Extract number from input
            import re
            numbers = re.findall(r'\d+', user_response)
            if numbers:
                storage = int(numbers[0])
                if storage > min_storage:
                    state.collected_params["additionalStorage"] = storage
                    logger.info(f"✅ Additional storage set to: {storage} GB")
                    conversation_state_manager.update_session(state)
                    return None
                else:
                    return {
                        "agent_name": "ValidationAgent",
                        "success": True,
                        "output": f"❌ Storage must be greater than the flavor's default ({min_storage} GB). Please enter a larger value or type 'skip':"
                    }
            else:
                return {
                    "agent_name": "ValidationAgent",
                    "success": True,
                    "output": f"❌ Please enter a number greater than {min_storage} GB, or type 'skip' to use the default:"
                }
        except ValueError:
            return {
                "agent_name": "ValidationAgent",
                "success": True,
                "output": f"❌ Please enter a valid number greater than {min_storage} GB, or type 'skip':"
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
    
    async def _handle_confirmation(self, input_text: str, state: Any) -> Optional[Dict[str, Any]]:
        """
        Handle user's confirmation or edit request after review.
        
        Args:
            input_text: User's response (yes/no/change X/cancel)
            state: Conversation state
            
        Returns:
            Dict with result or None to continue
        """
        user_response = input_text.lower().strip()
        
        # Check for confirmation
        if any(word in user_response for word in ["yes", "proceed", "confirm", "create", "go ahead"]):
            logger.info("✅ User confirmed cluster creation")
            state.status = ConversationStatus.READY_TO_EXECUTE
            state.last_asked_param = None
            conversation_state_manager.update_session(state)
            
            return {
                "agent_name": "ValidationAgent",
                "success": True,
                "ready_to_execute": True,
                "output": "🚀 Creating your cluster... This will take 15-30 minutes."
            }
        
        # Check for cancellation
        elif any(word in user_response for word in ["cancel", "abort", "stop", "no"]):
            logger.info("❌ User cancelled cluster creation")
            state.status = ConversationStatus.CANCELLED
            state.last_asked_param = None
            conversation_state_manager.update_session(state)
            
            return {
                "agent_name": "ValidationAgent",
                "success": True,
                "ready_to_execute": False,
                "output": "❌ Cluster creation cancelled. No resources were created."
            }
        
        # Check for edit/change request
        elif "change" in user_response or "edit" in user_response or "modify" in user_response:
            # Extract which parameter to change
            param_map = {
                "cluster name": "clusterName",
                "name": "clusterName",
                "datacenter": "datacenter",
                "data center": "datacenter",
                "location": "datacenter",
                "kubernetes": "k8sVersion",
                "k8s": "k8sVersion",
                "version": "k8sVersion",
                "cni": "cniDriver",
                "network": "cniDriver",
                "business unit": "businessUnit",
                "bu": "businessUnit",
                "environment": "environment",
                "env": "environment",
                "zone": "zone",
                "operating system": "operatingSystem",
                "os": "operatingSystem",
                "worker pool": "workerPoolName",
                "pool name": "workerPoolName",
                "node type": "nodeType",
                "flavor": "flavor",
                "storage": "additionalStorage",
                "replica": "replicaCount",
                "count": "replicaCount",
                "autoscaling": "enableAutoscaling"
            }
            
            # Find which parameter user wants to change
            param_to_change = None
            for key, value in param_map.items():
                if key in user_response:
                    param_to_change = value
                    break
            
            if param_to_change:
                logger.info(f"🔄 User wants to change: {param_to_change}")
                
                # Clear the parameter and all dependent parameters
                self._clear_parameter_and_dependents(param_to_change, state)
                
                # Reset status to collecting
                state.status = ConversationStatus.COLLECTING_PARAMS
                state.last_asked_param = None
                conversation_state_manager.update_session(state)
                
                # Ask for the parameter again
                return await self._ask_for_parameter(param_to_change, state)
            else:
                return {
                    "agent_name": "ValidationAgent",
                    "success": True,
                    "output": "❌ I couldn't identify which parameter you want to change. Please specify (e.g., 'change cluster name', 'change datacenter', etc.)"
                }
        
        else:
            return {
                "agent_name": "ValidationAgent",
                "success": True,
                "output": "❌ Please reply with 'yes' to proceed, 'change [parameter]' to modify something, or 'cancel' to abort."
            }
    
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
                # Get IPC engagement ID first
                ipc_engagement_id = await api_executor_service.get_ipc_engagement_id(engagement_id)
                
                # Get IKS images with datacenters
                dc_result = await api_executor_service.execute_operation(
                    resource_type="k8s_cluster",
                    operation="get_iks_images",
                    params={"ipc_engagement_id": ipc_engagement_id}
                )
                
                # Extract datacenters from images
                if dc_result.get("success") and dc_result.get("data"):
                    api_data = dc_result["data"]
                    if api_data.get("status") == "success" and api_data.get("data"):
                        all_images = []
                        for category, images in api_data["data"].items():
                            if isinstance(images, list):
                                all_images.extend(images)
                        
                        # Extract unique datacenters
                        datacenters = {}
                        for img in all_images:
                            dc_id = img.get("endpointId")
                            if dc_id and dc_id not in datacenters:
                                datacenters[dc_id] = {
                                    "id": dc_id,
                                    "name": img.get("endpointName", f"DC-{dc_id}"),
                                    "endpoint": img.get("endpoint", "")
                                }
                        
                        dc_result = {
                            "success": True,
                            "datacenters": list(datacenters.values()),
                            "images": all_images
                        }
                    else:
                        dc_result = {"success": False, "datacenters": [], "images": []}
                else:
                    dc_result = {"success": False, "datacenters": [], "images": []}
                
                # Store in state (moved outside the if/else blocks)
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
            # Extract k8s versions from images for this datacenter
            dc_images = [img for img in state._all_images if img.get("endpointId") == dc_id]
            import re
            version_set = set()
            for img in dc_images:
                match = re.search(r'v\d+\.\d+\.\d+', img.get("ImageName", ""))
                if match:
                    version_set.add(match.group(0))
            
            # Sort semantically (latest first)
            versions = sorted(list(version_set), key=lambda v: [int(x) for x in v[1:].split('.')], reverse=True)
            
            state._k8s_versions = versions
            
            version_list = "\n".join([f"  • {v}" for v in versions])
            output = f"**Step 3/15**: Which Kubernetes version would you like to use?\n\n🎯 **Available versions:**\n{version_list}"
        
        elif param_name == "cniDriver":
            dc_id = state.collected_params["_datacenter_id"]
            k8s_version = state.collected_params["k8sVersion"]
            
            logger.info(f"🌐 Fetching CNI drivers for endpoint {dc_id}, k8s version {k8s_version}")
            
            driver_result = await api_executor_service.execute_operation(
                resource_type="k8s_cluster",
                operation="get_network_list",
                params={"endpointId": dc_id, "k8sVersion": k8s_version}
            )
            
            logger.info(f"📦 CNI API response: {driver_result}")
            
            # Parse response to extract drivers list
            # API returns: {"status": "success", "data": {"data": ["calico-v3.29.3", ...], "status": "success"}}
            drivers = []
            if driver_result.get("success") and driver_result.get("data"):
                api_data = driver_result["data"]
                if api_data.get("status") == "success" and api_data.get("data"):
                    inner_data = api_data["data"]
                    if isinstance(inner_data, dict):
                        drivers = inner_data.get("data", [])
                    elif isinstance(inner_data, list):
                        drivers = inner_data
            
            logger.info(f"✅ Extracted {len(drivers)} CNI drivers: {drivers}")
            state._cni_drivers = drivers
            
            driver_list = "\n".join([f"  • {d}" for d in state._cni_drivers])
            output = f"**Step 4/15**: Which CNI (Container Network Interface) driver?\n\n🌐 **Available drivers:**\n{driver_list}"
        
        elif param_name == "businessUnit":
            # Fetch full department hierarchy (BU -> Environment -> Zone)
            logger.info(f"🏢 Fetching department details with nested hierarchy...")
            # Get IPC engagement ID
            ipc_engagement_id = await api_executor_service.get_ipc_engagement_id()
            
            # Get department details with nested hierarchy
            dept_result = await api_executor_service.execute_operation(
                resource_type="k8s_cluster",
                operation="get_department_details",
                params={"ipc_engagement_id": ipc_engagement_id}
            )
            
            # Parse response
            if dept_result.get("success") and dept_result.get("data"):
                api_data = dept_result["data"]
                if api_data.get("status") == "success":
                    dept_data = api_data.get("data", {})
                    dept_result = {
                        "success": True,
                        "data": dept_data,
                        "departmentList": dept_data.get("departmentList", [])
                    }
                else:
                    dept_result = {"success": False, "departmentList": []}
            else:
                dept_result = {"success": False, "departmentList": []}
            logger.info(f"🏢 API result: success={dept_result.get('success')}, departments count={len(dept_result.get('departmentList', []))}")
            
            if not dept_result.get("success"):
                logger.error(f"❌ Failed to fetch department details: {dept_result.get('error')}")
            
            # Store full hierarchy for later use (environments, zones)
            state._department_details = dept_result.get("departmentList", [])
            
            # Filter by selected datacenter's endpoint ID
            selected_endpoint_id = state.collected_params.get("_datacenter_id")
            datacenter_name = state.collected_params.get("datacenter", "selected")
            if isinstance(datacenter_name, dict):
                datacenter_name = datacenter_name.get("name", "selected")
            logger.info(f"🏢 Filtering departments for endpoint ID: {selected_endpoint_id}")
            
            # Filter departments that match the selected endpoint
            filtered_bus = []
            for dept in state._department_details:
                dept_endpoint_id = dept.get("endpointId")
                if dept_endpoint_id == selected_endpoint_id:
                    filtered_bus.append({
                        "id": dept["departmentId"],
                        "name": dept["departmentName"],
                        "endpoint_id": dept_endpoint_id,
                        "environmentList": dept.get("environmentList", [])  # Keep nested data
                    })
            
            logger.info(f"🏢 Found {len(filtered_bus)} business units for endpoint {selected_endpoint_id}")
            state._business_units = filtered_bus
            
            if filtered_bus:
                bu_list = "\n".join([f"  • {bu['name']}" for bu in filtered_bus])
                output = f"**Step 5/15**: Which business unit should this cluster belong to?\n\n🏢 **Available business units** (for {datacenter_name}):\n{bu_list}"
            else:
                output = f"**Step 5/15**: ⚠️ No business units found for datacenter {datacenter_name}.\n\nPlease contact your administrator to create a business unit for this location."
        
        elif param_name == "environment":
            # Get environments from the nested BU data
            bu_value = state.collected_params.get("businessUnit")
            logger.info(f"🔍 Asking for environment. BU value type: {type(bu_value)}")
            
            if isinstance(bu_value, dict):
                bu_name = bu_value.get("name")
                # Get environments directly from the BU's nested data
                env_list = bu_value.get("environmentList", [])
                logger.info(f"🏢 BU '{bu_name}' has {len(env_list)} environments in nested data")
            else:
                logger.error(f"❌ businessUnit is not a dict: {bu_value}")
                output = f"**Step 6/15**: ⚠️ Error: Business unit data not found. Please go back and select a business unit."
                return {"agent_name": "ValidationAgent", "success": True, "output": output}
            
            if env_list:
                env_names = "\n".join([f"  • {env['environmentName']}" for env in env_list])
                output = f"**Step 6/15**: Which environment is this cluster for?\n\n🔧 **Available environments** (for {bu_name}):\n{env_names}"
            else:
                output = f"**Step 6/15**: ⚠️ No environments found for business unit '{bu_name}'.\n\nPlease contact your administrator to create an environment for this business unit."
        
        elif param_name == "zone":
            # Get zones from the nested environment data
            env_value = state.collected_params.get("environment")
            logger.info(f"🔍 Asking for zone. Environment value type: {type(env_value)}")
            
            if isinstance(env_value, dict):
                env_name = env_value.get("name")
                # Get zones directly from the environment's nested data
                zone_list = env_value.get("zoneList", [])
                logger.info(f"🗺️ Environment '{env_name}' has {len(zone_list)} zones in nested data")
            else:
                logger.error(f"❌ environment is not a dict: {env_value}")
                output = f"**Step 7/15**: ⚠️ Error: Environment data not found. Please go back and select an environment."
                return {"agent_name": "ValidationAgent", "success": True, "output": output}
            
            if zone_list:
                zone_names = "\n".join([f"  • {z['zoneName']}" for z in zone_list])
                output = f"**Step 7/15**: Which network zone (VLAN) should the cluster use?\n\n🗺️ **Available zones** (for {env_name}):\n{zone_names}"
            else:
                output = f"**Step 7/15**: ⚠️ No zones found for environment '{env_name}'.\n\nPlease contact your administrator to create a zone for this environment."
        
        elif param_name == "operatingSystem":
            zone_id = state.collected_params["_zone_id"]
            k8s_version = state.collected_params["k8sVersion"]
            
            # Get OS images for zone
            os_result = await api_executor_service.execute_operation(
                resource_type="k8s_cluster",
                operation="get_os_images",
                params={"zoneId": zone_id}
            )
            
            # Parse and filter by k8s version
            if os_result.get("success") and os_result.get("data"):
                api_data = os_result["data"]
                if api_data.get("status") == "success" and api_data.get("data"):
                    images = api_data["data"].get("image", {}).get("options", [])
                    
                    # Filter by k8s version
                    version_patterns = [k8s_version]
                    if k8s_version.startswith("v"):
                        version_patterns.append(k8s_version[1:])
                    
                    filtered = [
                        img for img in images 
                        if any(p in (img.get("label", "") or img.get("ImageName", "")) for p in version_patterns)
                    ]
                    
                    # Group by osMake + osVersion
                    grouped = {}
                    for img in filtered:
                        os_make = img.get('osMake', 'Unknown')
                        os_version = img.get('osVersion', '')
                        key = f"{os_make} {os_version}".strip()
                        
                        if key not in grouped:
                            grouped[key] = {
                                "display_name": key,
                                "os_id": img.get("id"),
                                "os_make": os_make,
                                "os_model": img.get("osModel"),
                                "os_version": os_version,
                                "hypervisor": img.get("hypervisor"),
                                "image_id": img.get("IMAGEID"),
                                "image_name": img.get("ImageName")
                            }
                    
                    os_result = {"success": True, "os_options": list(grouped.values())}
                else:
                    os_result = {"success": False, "os_options": []}
            else:
                os_result = {"success": False, "os_options": []}
            state._os_options = os_result.get("os_options", [])
            
            os_list = "\n".join([f"  • {opt['display_name']}" for opt in state._os_options])
            output = f"**Step 8/15**: Which operating system for the worker nodes?\n\n💿 **Available OS options:**\n{os_list}"
        
        elif param_name == "workerPoolName":
            output = "**Step 9/15**: What would you like to name this worker node pool?\n\n📝 Requirements: 1-5 lowercase alphanumeric characters"
        
        elif param_name == "nodeType":
            zone_id = state.collected_params["_zone_id"]
            os_model = state.collected_params["operatingSystem"].get("os_model", "ubuntu")
            # Get flavors for zone
            flavor_result = await api_executor_service.execute_operation(
                resource_type="k8s_cluster",
                operation="get_flavors",
                params={"zoneId": zone_id}
            )
            
            # Parse and filter flavors
            if flavor_result.get("success") and flavor_result.get("data"):
                api_data = flavor_result["data"]
                if api_data.get("status") == "success" and api_data.get("data"):
                    all_flavors = api_data["data"].get("flavor", [])
                    
                    # Filter by applicationType = Container (strict match)
                    container_flavors = [f for f in all_flavors if f.get("applicationType") == "Container"]
                    
                    # Filter by OS model if provided
                    if os_model and container_flavors:
                        os_model_lower = os_model.lower()
                        filtered = [f for f in container_flavors 
                                   if os_model_lower in f.get("osModel", "").lower()]
                        if filtered:
                            container_flavors = filtered
                    
                    # Extract unique node types (raw flavorCategory values - no normalization)
                    node_types = list(set(f.get("flavorCategory") for f in container_flavors if f.get("flavorCategory")))
                    
                    # Format flavors
                    formatted_flavors = []
                    for flavor in container_flavors:
                        vcpu = flavor.get("vCpu", 0)
                        vram_mb = flavor.get("vRam", 0)
                        vram_gb = vram_mb // 1024 if vram_mb else 0
                        vdisk = flavor.get("vDisk", 0)
                        
                        formatted_flavors.append({
                            "id": flavor.get("artifactId"),
                            "name": f"{vcpu} vCPU / {vram_gb} GB RAM / {vdisk} GB Storage",
                            "display_name": flavor.get("display_name", flavor.get("FlavorName")),
                            "flavor_name": flavor.get("FlavorName"),
                            "sku_code": flavor.get("skuCode"),
                            "circuit_id": flavor.get("circuitId"),
                            "vcpu": vcpu,
                            "vram_gb": vram_gb,
                            "disk_gb": vdisk,
                            "node_type": flavor.get("flavorCategory"),
                            "storage_type": flavor.get("storageType"),
                            "os_model": flavor.get("osModel")
                        })
                    
                    flavor_result = {
                        "success": True,
                        "node_types": node_types,
                        "flavors": formatted_flavors
                    }
                else:
                    flavor_result = {"success": False, "node_types": [], "flavors": []}
            else:
                flavor_result = {"success": False, "node_types": [], "flavors": []}
            
            # Use raw node types from API (no transformation)
            state._node_types = flavor_result.get("node_types", [])
            state._all_flavors = flavor_result.get("flavors", [])
            
            logger.info(f"📋 Node types from API: {state._node_types}")
            
            if state._node_types:
                # Show RAW node types as-is from API (no pretty display)
                type_list = "\n".join([f"  • {t}" for t in state._node_types])
            else:
                # Fallback if API returned no node types
                type_list = "  • No node types available - please check zone/OS selection"
                logger.warning("⚠️ No node types returned from flavor API!")
            
            output = f"**Step 10/16**: What type of worker nodes do you need?\n\n💻 **Available types:**\n{type_list}"
        
        elif param_name == "flavor":
            node_type = state.collected_params["nodeType"]
            filtered_flavors = [f for f in state._all_flavors if f.get("node_type") == node_type]
            
            logger.info(f"📋 Flavors for node type '{node_type}': {len(filtered_flavors)}")
            
            # Format: "8 vCPU / 32 GB RAM / 100 GB Storage"
            flavor_list = "\n".join([f"  • {f['name']}" for f in filtered_flavors])
            output = f"**Step 11/16**: Which compute configuration?\n\n⚡ **Available flavors** ({node_type}):\n{flavor_list}"
        
        elif param_name == "additionalStorage":
            flavor = state.collected_params.get("flavor", {})
            default_storage = flavor.get("disk_gb", 50)
            output = f"**Step 12/16**: Would you like additional disk storage?\n\n💾 Default storage: **{default_storage} GB**\n\nEnter a number greater than {default_storage} GB, or type 'skip' to use default:"
        
        elif param_name == "replicaCount":
            output = "**Step 13/16**: How many worker nodes?\n\n📊 Enter a number between 1 and 8:"
        
        elif param_name == "enableAutoscaling":
            output = "**Step 14/16**: Would you like to enable autoscaling?\n\n🔄 This allows automatic scaling based on load. Answer 'yes' or 'no':"
        
        elif param_name == "maxReplicas":
            min_count = state.collected_params["replicaCount"]
            output = f"**Step 15/16**: What should be the maximum number of replicas when autoscaling?\n\n📈 Enter a number between {min_count} and 8:"
        
        elif param_name == "tags":
            output = "**Step 16/16**: Would you like to add any tags (key-value pairs for organization)?\n\n🏷️ Answer 'yes' or 'no' (you can skip for now):"
        
        else:
            output = f"Please provide: {param_name}"
        
        return {
            "agent_name": "ValidationAgent",
            "success": True,
            "output": output
        }
