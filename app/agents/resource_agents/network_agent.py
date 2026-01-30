"""
Network Agent - Handles firewall operations only.
"""

from typing import Any, Dict, List, Optional
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
            
            # Enrich firewalls with location data
            endpoint_names = params.get("endpoint_names", [])
            enriched_firewalls = self._enrich_firewalls_with_location(
                firewalls, endpoint_ids, endpoint_names
            )
            
            # Use LLM formatter for intelligent, user-friendly formatting
            # (consistent with all other agents: VM, K8s, LoadBalancer, etc.)
            formatted_response = await self.format_response_with_llm(
                operation="list",
                raw_data=enriched_firewalls,
                user_query=user_query,
                context={
                    "query_type": "general",
                    "total_count": len(enriched_firewalls),
                    "location_filter": params.get("location_filter"),
                    "endpoint_names": endpoint_names
                }
            )
            
            return {
                "success": True,
                "data": enriched_firewalls,
                "response": formatted_response,
                "metadata": {
                    "count": len(enriched_firewalls),
                    "resource_type": "firewall"
                }
            }
        except Exception as e:
            logger.error(f"❌ Error listing firewalls: {str(e)}", exc_info=True)
            raise

    def _enrich_firewalls_with_location(
        self,
        firewalls: List[Dict[str, Any]],
        endpoint_ids: Optional[List[int]] = None,
        endpoint_names: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Enrich firewall data with location information.
        
        Args:
            firewalls: Raw firewall data from API
            endpoint_ids: List of endpoint IDs queried
            endpoint_names: List of endpoint/datacenter names
            
        Returns:
            Enriched firewall list with _location field
        """
        # Build endpoint ID to name mapping
        endpoint_map = {}
        if endpoint_ids and endpoint_names:
            for eid, name in zip(endpoint_ids, endpoint_names):
                if eid is not None:
                    endpoint_map[eid] = name
        
        enriched = []
        for fw in firewalls:
            enriched_fw = {**fw}  # Copy original data
            
            # Try to get location from endpoint mapping
            queried_endpoint = fw.get("_queried_endpoint_id")
            if queried_endpoint and queried_endpoint in endpoint_map:
                enriched_fw["_location"] = endpoint_map[queried_endpoint]
            else:
                # Try to extract from firewall data
                location = (
                    fw.get("endpointName") or 
                    fw.get("locationName") or 
                    fw.get("datacenter") or
                    self._extract_location_from_fw_name(fw)
                )
                enriched_fw["_location"] = location or "Unknown"
            
            enriched.append(enriched_fw)
        
        return enriched
    
    def _extract_location_from_fw_name(self, fw: Dict[str, Any]) -> Optional[str]:
        """Extract location from firewall name if possible."""
        name = fw.get("name") or fw.get("firewallName") or ""
        name_lower = name.lower()
        
        location_patterns = {
            "Mumbai": ["mumbai", "bkc", "mum"],
            "Delhi": ["delhi", "del", "ncr"],
            "Chennai": ["chennai", "che", "amb"],
            "Bengaluru": ["bengaluru", "bangalore", "blr"],
            "Hyderabad": ["hyderabad", "hyd"],
            "Pune": ["pune"]
        }
        
        for location, patterns in location_patterns.items():
            if any(p in name_lower for p in patterns):
                return location
        
        return None