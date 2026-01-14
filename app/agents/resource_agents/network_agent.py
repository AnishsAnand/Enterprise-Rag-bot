"""
Network Agent - Handles firewall and load balancer operations.
PRODUCTION FIX: Comprehensive load balancer listing with proper error handling.
"""

from typing import Any, Dict, List, Optional
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
        """
        Execute network operation with proper resource type routing.
        
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
            
            if operation == "list":
                # Route to appropriate handler based on resource type
                if resource_type == "firewall":
                    return await self._list_firewalls(params, context)
                elif resource_type == "load_balancer":
                    return await self._list_load_balancers(params, context)
                else:
                    return {
                        "success": False,
                        "response": f"Listing {resource_type} is not yet implemented."
                    }
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
    
    async def _list_load_balancers(
    self,
    params: Dict[str, Any],
    context: Dict[str, Any]
) -> Dict[str, Any]:

        try:
            user_roles = context.get("user_roles", [])
            user_id = context.get("user_id")
            user_query = context.get("user_query", "").lower()
        
            logger.info(f"⚖️ Listing load balancers for user: {user_id}")
        
        # CRITICAL: Get IPC engagement ID (required for load balancer API)
            engagement_id = await self.get_engagement_id(user_roles=user_roles)
            if not engagement_id:
                return {
                "success": False,
                "error": "Failed to get engagement ID",
                "response": "Unable to retrieve engagement information. Please try again or contact support."
                }
        
        # Convert PAAS engagement ID to IPC engagement ID
            ipc_engagement_id = await api_executor_service.get_ipc_engagement_id(
                engagement_id=engagement_id,
                user_id=user_id
            )
        
            if not ipc_engagement_id:
                return {
                "success": False,
                "error": "Failed to get IPC engagement ID",
                "response": "Unable to retrieve IPC engagement information. Please try again or contact support."
            }
        
            logger.info(f"✅ Got IPC engagement ID: {ipc_engagement_id}")
        
        # CRITICAL: Single API call with IPC engagement ID
        # This returns ALL load balancers from ALL endpoints
            logger.info(f"📡 Calling load balancer API with IPC engagement ID: {ipc_engagement_id}")
        
            result = await api_executor_service.list_load_balancers(
            ipc_engagement_id=ipc_engagement_id,
            user_id=user_id
            )
        
            if not result.get("success"):
                return {
                "success": False,
                "error": result.get("error"),
                "response": f"Failed to list load balancers: {result.get('error')}"
            }
        
            load_balancers = result.get("data", [])
            total_count = result.get("total", len(load_balancers))
        
            logger.info(f"✅ Found {total_count} load balancer(s) across all endpoints")
        
        # Apply intelligent filtering if user specified criteria
            filter_criteria = self._extract_filter_criteria(user_query)
            if filter_criteria and load_balancers:
                logger.info(f"🔍 Applying filter criteria: {filter_criteria}")
                load_balancers = await self.filter_with_llm(
                load_balancers, 
                filter_criteria, 
                user_query
                )
                logger.info(f"✅ After filtering: {len(load_balancers)} load balancer(s)")
        
        # Format with LLM
            formatted_response = await self.format_response_with_llm(
            operation="list",
            raw_data=load_balancers,
            user_query=user_query,
                context={
                "ipc_engagement_id": ipc_engagement_id,
                "total_count": len(load_balancers),
                "resource_type": "load_balancer"
                }
            )
        
        # Handle empty results gracefully
            if len(load_balancers) == 0:
                formatted_response = (
                f"⚖️ **No Load Balancers Found**\n\n"
                f"Your engagement currently has no load balancers configured.\n\n"
                f"**This is normal if:**\n"
                f"- Your engagement is newly set up\n"
                f"- You haven't created any load balancers yet\n"
                f"- Load balancers are managed through a different system\n\n"
                f"💡 **Tip:** To create a load balancer, use the cloud portal or contact your administrator."
                )
        
            return {
            "success": True,
            "data": load_balancers,
            "response": formatted_response,
            "metadata": {
                "count": len(load_balancers),
                "ipc_engagement_id": ipc_engagement_id,
                "resource_type": "load_balancer"
                }
            }
        
        except Exception as e:
            logger.error(f"❌ Error listing load balancers: {str(e)}", exc_info=True)
            raise
    
    
    def _extract_filter_criteria(self, user_query: str) -> Optional[str]:

        if not user_query:
            return None
    
        query_lower = user_query.lower()
    
    # Common filter keywords for load balancers
        filter_keywords = [
        "active", "inactive", "enabled", "disabled",
        "production", "staging", "development", "prod", "stage", "dev",
        "ssl", "https", "http", "tcp", "udp",
        "healthy", "unhealthy", "degraded",
        "high", "low", "traffic", "load",
        "public", "private", "internal", "external"
        ]
    
    # Look for filter keywords in query
        for keyword in filter_keywords:
            if keyword in query_lower:
                words = query_lower.split()
                if keyword in words:
                    idx = words.index(keyword)
                    start = max(0, idx - 2)
                    end = min(len(words), idx + 3)
                    context = " ".join(words[start:end])
                    logger.info(f"🔍 Extracted filter context: '{context}'")
                    return context
    
        return None
    
    async def _list_firewalls(
        self,
        params: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """List firewalls with intelligent formatting."""
        try:
            endpoint_ids = params.get("endpoints", [])
            user_id = context.get("user_id")
            
            if not endpoint_ids:
                datacenters = await self.get_datacenters()
                endpoint_ids = [dc.get("endpointId") for dc in datacenters if dc.get("endpointId")]
            
            logger.info(f"🔍 Listing firewalls for endpoints: {endpoint_ids}")
            
            result = await api_executor_service.list_firewalls(
                endpoint_ids=endpoint_ids,
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
            user_query = context.get("user_query", "")
            formatted_response = await self.format_response_with_llm(
                operation="list",
                raw_data=firewalls,
                user_query=user_query,
                context={
                    "endpoint_names": params.get("endpoint_names", []),
                    "resource_type": "firewall"
                }
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