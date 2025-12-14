"""
Network Agent - Handles firewall and load balancer operations.
"""

from typing import Any, Dict, List
import logging

from app.agents.resource_agents.base_resource_agent import BaseResourceAgent
from app.services.api_executor_service import api_executor_service

logger = logging.getLogger(__name__)


class NetworkAgent(BaseResourceAgent):
    """Agent for network operations (firewalls, load balancers)."""
    
    def __init__(self):
        super().__init__(
            agent_name="NetworkAgent",
            agent_description="Specialized agent for network operations including firewalls and load balancers",
            resource_type="network",
            temperature=0.2
        )
    
    def get_supported_operations(self) -> List[str]:
        return ["list", "create", "update", "delete"]
    
    async def execute_operation(
        self,
        operation: str,
        params: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            resource_type = context.get("resource_type", "")
            logger.info(f"🔥 NetworkAgent executing: {operation} for {resource_type}")
            
            if operation == "list":
                if resource_type == "firewall":
                    return await self._list_firewalls(params, context)
                else:
                    return {"success": False, "response": f"Listing {resource_type} is not yet implemented."}
            else:
                return {"success": False, "response": f"Operation '{operation}' is not yet implemented."}
        except Exception as e:
            logger.error(f"❌ NetworkAgent error: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e), "response": f"Error: {str(e)}"}
    
    async def _list_firewalls(self, params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """List firewalls with intelligent formatting."""
        try:
            endpoint_ids = params.get("endpoints", [])
            
            if not endpoint_ids:
                datacenters = await self.get_datacenters()
                endpoint_ids = [dc.get("endpointId") for dc in datacenters if dc.get("endpointId")]
            
            logger.info(f"🔍 Listing firewalls for endpoints: {endpoint_ids}")
            
            result = await api_executor_service.list_firewalls(endpoint_ids=endpoint_ids)
            
            if not result.get("success"):
                return {
                    "success": False,
                    "error": result.get("error"),
                    "response": f"Failed to list firewalls: {result.get('error')}"
                }
            
            firewalls = result.get("data", [])
            logger.info(f"✅ Found {len(firewalls)} firewalls")
            
            # Format with LLM
            user_query = context.get("user_query", "")
            formatted_response = await self.format_response_with_llm(
                operation="list",
                raw_data=firewalls,
                user_query=user_query,
                context={"endpoint_names": params.get("endpoint_names", [])}
            )
            
            return {
                "success": True,
                "data": firewalls,
                "response": formatted_response,
                "metadata": {"count": len(firewalls)}
            }
        except Exception as e:
            logger.error(f"❌ Error listing firewalls: {str(e)}", exc_info=True)
            raise

