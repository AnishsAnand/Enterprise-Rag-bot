"""
Network Agent - Handles firewall operations only.
PRODUCTION: Load balancer functionality moved to LoadBalancerAgent.
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
                    "response": f"NetworkAgent only handles firewall operations. For {resource_type}, please use the appropriate agent."
                }
            
            if operation == "list":
                return await self._list_firewalls(params, context)
            else:
                return {
                    "success": False,
                    "response": f"Operation '{operation}' for {resource_type} is not yet implemented."
                }
                
        except Exception as e:
            logger.error(f"❌ NetworkAgent error: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "response": f"An error occurred while {operation}ing {context.get('resource_type', 'resource')}: {str(e)}"
            }
    
    async def _list_firewalls(
        self,
        params: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
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
                        "response": "Unable to retrieve engagement information."
                    }
                
                ipc_engagement_id = await api_executor_service.get_ipc_engagement_id(
                    engagement_id=engagement_id,
                    user_id=user_id
                )
                
                if not ipc_engagement_id:
                    return {
                        "success": False,
                        "error": "Failed to get IPC engagement ID",
                        "response": "Unable to retrieve IPC engagement information."
                    }
            
            # Get endpoints if not provided
            if not endpoint_ids:
                datacenters = await self.get_datacenters()
                endpoint_ids = [dc.get("endpointId") for dc in datacenters if dc.get("endpointId")]
            
            logger.info(f"🔍 Listing firewalls for endpoints: {endpoint_ids}")
            
            # Call API executor service
            result = await api_executor_service.list_firewalls(
                endpoint_ids=endpoint_ids,
                ipc_engagement_id=ipc_engagement_id,
                user_id=user_id
            )
            
            if not result.get("success"):
                return {
                    "success": False,
                    "error": result.get("error"),
                    "response": f"Failed to list firewalls: {result.get('error')}"
                }
            
            firewalls = result.get("data", [])
            logger.info(f"✅ Found {len(firewalls)} firewalls")
            
            # Format with LLM
            formatted_response = await self.format_response_with_llm(
                operation="list",
                raw_data=firewalls,
                user_query=user_query,
                context={
                    "endpoint_names": params.get("endpoint_names", []),
                    "ipc_engagement_id": ipc_engagement_id,
                    "resource_type": "firewall"
                }
            )
            
            return {
                "success": True,
                "data": firewalls,
                "response": formatted_response,
                "metadata": {
                    "count": len(firewalls),
                    "resource_type": "firewall"
                }
            }
        except Exception as e:
            logger.error(f"❌ Error listing firewalls: {str(e)}", exc_info=True)
            raise