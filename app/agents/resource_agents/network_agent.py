"""
Network Agent - Handles firewall operations only.
PRODUCTION: Load balancer functionality moved to LoadBalancerAgent.
"""

from typing import Any, Dict, List, Optional
import json
import logging

from app.agents.resource_agents.base_resource_agent import BaseResourceAgent
from app.services.api_executor_service import api_executor_service

logger = logging.getLogger(__name__)


class NetworkAgent(BaseResourceAgent):
    """Agent for network operations (firewalls only)."""
    
    def __init__(self):
        super().__init__(
            agent_name="NetworkAgent",
            agent_description="Specialized agent for firewall operations",
            resource_type="firewall",
            temperature=0.2
        )
    
    def get_supported_operations(self) -> List[str]:
        return ["list", "create", "update", "delete"]
    
    async def execute_operation(self,operation: str,params: Dict[str, Any],context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute network operation for firewalls.
        Args:
            operation: Operation to perform (list, create, update, delete)
            params: Operation parameters
            context: Context including resource_type, user_query, etc.
        Returns:
            Dict with success status and formatted response
        """
        try:
            resource_type = context.get("resource_type", "")
            logger.info(f"🔥 NetworkAgent executing: {operation} for {resource_type}")
            # NetworkAgent now only handles firewall operations
            if resource_type != "firewall":
                return {
                    "success": False,
                    "error": f"NetworkAgent does not handle {resource_type}",
                    "response": f"NetworkAgent only handles firewall operations. For {resource_type}, please use the appropriate agent."}
            if operation == "list":
                return await self._list_firewalls(params, context)
            else:
                return {
                    "success": False,
                    "response": f"Operation '{operation}' for {resource_type} is not yet implemented."}
        except Exception as e:
            logger.error(f"❌ NetworkAgent error: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "response": f"An error occurred while {operation}ing {context.get('resource_type', 'resource')}: {str(e)}"}
    
    async def _list_firewalls(self,params: Dict[str, Any],context: Dict[str, Any]) -> Dict[str, Any]:
        """List firewalls with intelligent formatting."""
        try:
            endpoint_ids = params.get("endpoints", [])
            user_id = context.get("user_id")
            user_roles = context.get("user_roles", [])
            user_query = context.get("user_query", "")
            # Get IPC engagement ID if not provided
            ipc_engagement_id = params.get("ipc_engagement_id")
            if not ipc_engagement_id:
                engagement_id = await self.get_engagement_id(user_roles=user_roles)
                if not engagement_id:
                    return {
                        "success": False,
                        "error": "Failed to get engagement ID",
                        "response": "Unable to retrieve engagement information."}
                
                ipc_engagement_id = await api_executor_service.get_ipc_engagement_id(engagement_id=engagement_id,user_id=user_id)
                if not ipc_engagement_id:
                    return {
                        "success": False,
                        "error": "Failed to get IPC engagement ID",
                        "response": "Unable to retrieve IPC engagement information."}
            # Get endpoints if not provided
            if not endpoint_ids:
                datacenters = await self.get_datacenters()
                endpoint_ids = [dc.get("endpointId") for dc in datacenters if dc.get("endpointId")]
            logger.info(f"🔍 Listing firewalls for endpoints: {endpoint_ids}")
            # Call API executor service
            result = await api_executor_service.list_firewalls(
                endpoint_ids=endpoint_ids,
                ipc_engagement_id=ipc_engagement_id
            )
            if not result.get("success"):
                return {
                    "success": False,
                    "error": result.get("error"),
                    "response": f"Failed to list firewalls: {result.get('error')}"}
            firewalls = result.get("data", [])
            logger.info(f"✅ Found {len(firewalls)} firewalls")
            # Prefer deterministic formatting to avoid LB-style fields
            formatted_response = self._format_firewall_dynamic_table(
                firewalls,
                params.get("endpoint_names", []),
                endpoint_ids
            )
            return {
                "success": True,
                "data": firewalls,
                "response": formatted_response,
                "metadata": {
                    "count": len(firewalls),
                    "resource_type": "firewall"} }
        except Exception as e:
            logger.error(f"❌ Error listing firewalls: {str(e)}", exc_info=True)
            raise

    def _format_firewall_dynamic_table(
        self,
        firewalls: List[Dict[str, Any]],
        endpoint_names: List[str],
        endpoint_ids: Optional[List[int]] = None
    ) -> str:
        """Render firewall list using raw API fields."""
        if not firewalls:
            return "🔥 No firewalls found for the selected datacenter(s)."

        endpoint_map = {}
        if endpoint_ids and endpoint_names and len(endpoint_ids) == len(endpoint_names):
            endpoint_map = {eid: name for eid, name in zip(endpoint_ids, endpoint_names)}

        # Build a stable set of columns from the raw payload keys
        columns = []
        for fw in firewalls:
            if isinstance(fw, dict):
                for key in fw.keys():
                    if key not in columns and key not in ["_queried_endpoint_id"]:
                        columns.append(key)
            if len(columns) >= 8:
                break

        if not columns:
            return "🔥 No firewall fields available from the API response."

        summary = f"🔥 Found **{len(firewalls)} firewalls** across **{len(endpoint_names) or 1} datacenter(s)**"
        if len(endpoint_names) == 1:
            summary += f"\n\n**Location:** {endpoint_names[0]}"

        lines = [summary, ""]
        header = " | ".join(columns + (["endpoint"] if endpoint_map else []))
        separator = " | ".join(["---"] * (len(columns) + (1 if endpoint_map else 0)))
        lines.append(f"| {header} |")
        lines.append(f"| {separator} |")

        for fw in firewalls:
            row_values = []
            for col in columns:
                val = fw.get(col, "")
                if isinstance(val, (dict, list)):
                    val = json.dumps(val)[:120]
                row_values.append(str(val))
            if endpoint_map:
                endpoint_id = fw.get("_queried_endpoint_id")
                row_values.append(endpoint_map.get(endpoint_id, ""))
            lines.append("| " + " | ".join(row_values) + " |")

        return "\n".join(lines)