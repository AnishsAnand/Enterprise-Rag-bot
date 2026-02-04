"""
Virtual Machine Agent - Handles VM/Instance operations.
"""
from typing import Any, Dict, List
import logging
from app.agents.resource_agents.base_resource_agent import BaseResourceAgent
from app.services.api_executor_service import api_executor_service
logger = logging.getLogger(__name__)

class VirtualMachineAgent(BaseResourceAgent):
    """Agent for virtual machine operations."""
    def __init__(self):
        super().__init__(
            agent_name="VirtualMachineAgent",
            agent_description="Specialized agent for VM/Instance operations with intelligent filtering",
            resource_type="vm",
            temperature=0.2)

    def get_supported_operations(self) -> List[str]:
        return ["list", "create", "stop", "start", "delete"]
    
    async def execute_operation(self,operation: str,params: Dict[str, Any],context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            logger.info(f"🖥️ VirtualMachineAgent executing: {operation}")
            if operation == "list":
                return await self._list_vms(params, context)
            else:
                return {
                    "success": False,
                    "response": f"Operation '{operation}' for VMs is not yet implemented."}
        except Exception as e:
            logger.error(f"❌ VirtualMachineAgent error: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e), "response": f"Error: {str(e)}"}
    
    async def _list_vms(self, params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """List VMs with intelligent filtering."""
        try:
            # Extract auth info from context - MUST be done FIRST before any API calls
            auth_token = context.get("auth_token") if context else None
            user_id = context.get("user_id") if context else None
            user_type = context.get("user_type") if context else None
            selected_engagement_id = context.get("selected_engagement_id") if context else None
            
            logger.info(f"🔐 VM listing with auth_token: {'✓' if auth_token else '✗'}, user_id: {user_id}, engagement: {selected_engagement_id}")
            
            # Handle both parameter naming conventions
            endpoint_ids = params.get("endpoints") or params.get("endpoint_ids") or []
            endpoint_names = params.get("endpoint_names") or []
            endpoint_filter = params.get("endpoint_filter") or params.get("endpoint")
            zone_filter = params.get("zone_filter") or params.get("zone")
            department_filter = params.get("department_filter") or params.get("department")
            # INTELLIGENT MAPPING: If we have endpoint_ids but no endpoint_filter, 
            if endpoint_ids and not endpoint_filter and endpoint_names:
                endpoint_filter = endpoint_names[0] if len(endpoint_names) == 1 else None
                logger.info(f"🔍 Using endpoint filter from endpoint_names: {endpoint_filter}")
            elif endpoint_filter and not endpoint_names:
                # Try to map it to full name using ValidationAgent's logic
                logger.info(f"🔍 Got abbreviation '{endpoint_filter}', will fetch full endpoint name")
                # Map common abbreviations
                abbrev_map = {
                    "blr": "Bengaluru",
                    "del": "Delhi", 
                    "mum": "Mumbai-BKC",
                    "chennai": "Chennai-AMB",
                    "singapore": "Singapore East",
                    "sg": "Singapore East"}
                mapped_name = abbrev_map.get(endpoint_filter.lower())
                if mapped_name:
                    endpoint_filter = mapped_name
                    logger.info(f"✅ Mapped abbreviation to full name: {endpoint_filter}")
            # Fallback: infer endpoint filter from user query if still missing
            user_query = context.get("user_query", "") if context else ""
            if not endpoint_filter and user_query:
                inferred = self._extract_endpoint_filter(user_query)
                if inferred:
                    endpoint_filter = inferred
                    logger.info(f"✅ Inferred endpoint from query: {endpoint_filter}")
            if endpoint_filter and endpoint_filter not in endpoint_names:
                endpoint_names = endpoint_names + [endpoint_filter]
            
            # Get IPC engagement ID if we have a selected engagement
            ipc_engagement_id = None
            if selected_engagement_id:
                ipc_engagement_id = await api_executor_service.get_ipc_engagement_id(
                    engagement_id=selected_engagement_id,
                    auth_token=auth_token,
                    user_id=user_id
                )
                logger.info(f"✅ Using IPC engagement ID: {ipc_engagement_id} (from selected: {selected_engagement_id})")
            
            logger.info(f"🔍 Listing VMs with filters: endpoint={endpoint_filter}, zone={zone_filter}, dept={department_filter}")
            result = await api_executor_service.list_vms(
                ipc_engagement_id=ipc_engagement_id,
                endpoint_filter=endpoint_filter,
                zone_filter=zone_filter,
                department_filter=department_filter,
                auth_token=auth_token,
                user_id=user_id
            )
            if not result.get("success"):
                return {
                    "success": False,
                    "error": result.get("error"),
                    "response": f"Failed to list VMs: {result.get('error')}"}
            vms = result.get("data", [])
            logger.info(f"✅ Found {len(vms)} VMs")
            # Simplify VM data for better LLM processing
            simplified_vms = self._simplify_vm_data(vms)
            if endpoint_filter:
                simplified_vms = [
                    vm for vm in simplified_vms
                    if endpoint_filter.lower() in (vm.get("endpoint") or "").lower()
                ]
            # Use common LLM formatter
            user_query = context.get("user_query", "") if context else ""
            formatted_response = await self.format_response_agentic(
                operation="list",
                raw_data=simplified_vms,
                user_query=user_query,
                context={"query_type": "general", "endpoint_names": endpoint_names})
            return {
                "success": True,
                "data": vms,
                "response": formatted_response,
                "metadata": {"count": len(vms), "endpoints": endpoint_names}}
        except Exception as e:
            logger.error(f"❌ Error listing VMs: {str(e)}", exc_info=True)
            raise
    
    def _simplify_vm_data(self, vms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Simplify nested VM data for LLM processing."""
        simplified_vms = []
        for vm_item in vms:
            vm = vm_item.get("virtualMachine", {})
            simplified_vms.append({
                "vmName": vm.get("vmName", "N/A"),
                "vmuuid": vm.get("vmuuid", "N/A"),
                "endpoint": vm.get("endpoint", {}).get("endpointName", "N/A"),
                "engagement": vm.get("engagement", {}).get("engagementName", "N/A"),
                "vmId": vm.get("vmId", "N/A"),
                "storage": vm.get("storage", 0),
                "isPpuMeteringEnabled": vm.get("isPpuMeteringEnabled", "no"),
                "isBudgetingEnabled": vm.get("isBudgetingEnabled", "no"),
                "tags": vm.get("tags", [])})
        return simplified_vms

    def _extract_endpoint_filter(self, user_query: str) -> str:
        """Infer endpoint filter from the user query (e.g., 'vms in blr')."""
        query_lower = user_query.lower()
        abbrev_map = {
            "blr": "Bengaluru",
            "del": "Delhi",
            "mum": "Mumbai-BKC",
            "bkc": "Mumbai-BKC",
            "chennai": "Chennai-AMB",
            "singapore": "Singapore East",
            "sg": "Singapore East"
        }
        # Direct abbreviation match
        for abbr, full in abbrev_map.items():
            if f" {abbr} " in f" {query_lower} ":
                return full
        # Direct full-name match
        known_locations = [
            "bengaluru", "delhi", "mumbai-bkc", "chennai-amb",
            "singapore east", "mumbai"
        ]
        for loc in known_locations:
            if loc in query_lower:
                return loc.title() if loc != "mumbai-bkc" else "Mumbai-BKC"
        return ""