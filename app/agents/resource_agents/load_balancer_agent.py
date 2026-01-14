"""
Load Balancer Agent - PRODUCTION READY
FIXED: Uses IPC engagement ID for single API call (not endpoint iteration)

Following the business_unit pattern:
- Single API call with IPC engagement ID
- Returns ALL load balancers from ALL endpoints
- No endpoint iteration needed
"""

from typing import Any, Dict, List, Optional
import logging

from app.agents.resource_agents.base_resource_agent import BaseResourceAgent
from app.services.api_executor_service import api_executor_service

logger = logging.getLogger(__name__)


class LoadBalancerAgent(BaseResourceAgent):
    """
    Agent for load balancer operations.
    
    Supported Operations:
    - list: List ALL load balancers across all endpoints with single API call
    
    PRODUCTION NOTES:
    - Uses IPC engagement ID (NOT endpoint ID or iteration)
    - Single API call: GET /networkservice/loadbalancer/list/loadbalancers/{ipc_engagement_id}
    - Returns load balancers from ALL endpoints in one response
    - Handles 404 and status='failed' gracefully (means no LBs exist)
    - Uses LLM for user-friendly formatting
    """
    
    def __init__(self):
        super().__init__(
            agent_name="LoadBalancerAgent",
            agent_description=(
                "Specialized agent for load balancer operations. "
                "Lists and analyzes load balancers across all datacenters "
                "with intelligent filtering and formatting."
            ),
            resource_type="load_balancer",
            temperature=0.2
        )
    
    def get_supported_operations(self) -> List[str]:
        """Get list of supported operations."""
        return ["list"]
    
    async def execute_operation(
        self,
        operation: str,
        params: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a load balancer operation.
        
        Args:
            operation: Operation to perform (currently only 'list')
            params: Parameters (ipc_engagement_id, filters, etc.)
            context: Context (session_id, user_query, user_roles, user_id, etc.)
            
        Returns:
            Dict with success status and formatted response
        """
        try:
            logger.info(f"⚖️ LoadBalancerAgent executing: {operation}")
            
            if operation == "list":
                return await self._list_load_balancers(params, context)
            else:
                return {
                    "success": False,
                    "error": f"Unsupported operation: {operation}",
                    "response": f"I don't support the '{operation}' operation for load balancers yet. Currently, only listing is available."
                }
                
        except Exception as e:
            logger.error(f"❌ LoadBalancerAgent error: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "response": f"An error occurred while {operation}ing load balancers: {str(e)}"
            }
    
    async def _list_load_balancers(
        self,
        params: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        List ALL load balancers using IPC engagement ID.
        
        CRITICAL FIXES:
        1. Uses IPC engagement ID (NOT endpoint iteration)
        2. Single API call returns ALL load balancers from ALL endpoints
        3. Properly handles empty results (not errors)
        4. Caches results in user session
        
        WORKFLOW:
        1. Get IPC engagement ID (from cache or API)
        2. Call load balancer list API with IPC engagement ID
        3. Apply intelligent filtering if requested
        4. Format response using LLM
        
        Args:
            params: Parameters including:
                - ipc_engagement_id: int (optional - will be fetched if not provided)
                - force_refresh: bool (optional - bypass cache)
            context: Context including:
                - user_query: str - Original user query for intelligent formatting
                - user_roles: List[str] - User roles for permission checking
                - user_id: str - User ID for authentication and caching
            
        Returns:
            Dict containing:
                - success: bool - Whether operation succeeded
                - data: List[Dict] - Raw load balancer data
                - response: str - LLM-formatted user-friendly response
                - metadata: Dict - Additional metadata (count, engagement, etc.)
        """
        try:
            # Extract context
            user_roles = context.get("user_roles", [])
            user_id = context.get("user_id")
            user_query = context.get("user_query", "").lower()
            force_refresh = params.get("force_refresh", False)
            
            logger.info(f"📋 Listing load balancers for user: {user_id}")
            if force_refresh:
                logger.info("🔄 Force refresh requested - bypassing cache")
            
            # CRITICAL: Get IPC engagement ID (required for load balancer API)
            # This is fetched once and cached in user session
            ipc_engagement_id = params.get("ipc_engagement_id")
            
            if not ipc_engagement_id:
                # Get PAAS engagement ID first
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
                    user_id=user_id,
                    force_refresh=force_refresh
                )
                
                if not ipc_engagement_id:
                    return {
                        "success": False,
                        "error": "Failed to get IPC engagement ID",
                        "response": "Unable to retrieve IPC engagement information. Please try again or contact support."
                    }
                
                logger.info(f"✅ Got IPC engagement ID: {ipc_engagement_id}")
            
            # CRITICAL: Call load balancer API with IPC engagement ID
            # This is a SINGLE API call that returns ALL load balancers from ALL endpoints
            # Similar to business_unit API pattern
            logger.info(f"📡 Calling load balancer API with IPC engagement ID: {ipc_engagement_id}")
            
            result = await api_executor_service.list_load_balancers(
                ipc_engagement_id=ipc_engagement_id,
                user_id=user_id,
                force_refresh=force_refresh
            )
            
            if not result.get("success"):
                # API call failed
                error_msg = result.get("error", "Unknown error")
                logger.error(f"❌ Load balancer API failed: {error_msg}")
                
                return {
                    "success": False,
                    "error": error_msg,
                    "response": f"Failed to retrieve load balancers: {error_msg}. Please try again or contact support."
                }
            
            # Extract load balancer data
            load_balancers = result.get("data", [])
            total_count = result.get("total", len(load_balancers))
            is_cached = result.get("cached", False)
            
            logger.info(f"✅ Retrieved {total_count} load balancer(s) {'(cached)' if is_cached else ''}")
            
            # Apply intelligent filtering if user specified criteria
            original_count = total_count
            filter_criteria = self._extract_filter_criteria(user_query)
            if filter_criteria and load_balancers:
                logger.info(f"🔍 Applying filter criteria: {filter_criteria}")
                load_balancers = await self.filter_with_llm(
                    load_balancers, 
                    filter_criteria, 
                    user_query
                )
                total_count = len(load_balancers)
                logger.info(f"✅ After filtering: {total_count} load balancer(s) (from {original_count})")
            
            # Format response with LLM for user-friendly output
            formatted_response = await self.format_response_with_llm(
                operation="list",
                raw_data=load_balancers,
                user_query=user_query,
                context={
                    "ipc_engagement_id": ipc_engagement_id,
                    "total_count": total_count,
                    "original_count": original_count,
                    "filter_applied": filter_criteria is not None,
                    "cached": is_cached,
                    "resource_type": "load_balancer"
                }
            )
            
            # Special handling for empty results
            if total_count == 0:
                if original_count > 0 and filter_criteria:
                    # Filtered out all results
                    formatted_response = (
                        f"⚖️ **No Load Balancers Match Your Criteria**\n\n"
                        f"I found {original_count} total load balancer(s), but none matched your filter: '{filter_criteria}'\n\n"
                        f"**Suggestions:**\n"
                        f"- Try broadening your search criteria\n"
                        f"- Remove filters to see all load balancers\n"
                        f"- Check if you're looking for the right attributes (status, protocol, SSL, etc.)"
                    )
                else:
                    # No load balancers exist
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
                    "count": total_count,
                    "original_count": original_count,
                    "ipc_engagement_id": ipc_engagement_id,
                    "filter_applied": filter_criteria is not None,
                    "cached": is_cached,
                    "resource_type": "load_balancer"
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error listing load balancers: {str(e)}", exc_info=True)
            raise
    
    def _extract_filter_criteria(self, user_query: str) -> Optional[str]:
        """
        Extract filter criteria from user query for intelligent filtering.
        
        Examples:
            "list active load balancers" → "active"
            "show load balancers in production" → "production"
            "load balancers with SSL enabled" → "SSL enabled"
            "LBs handling high traffic" → "high traffic"
            "healthy load balancers" → "healthy"
        
        Args:
            user_query: User's original query
            
        Returns:
            Filter criteria string or None if no filters detected
        """
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
            "public", "private", "internal", "external",
            "running", "stopped", "failed"
        ]
        
        # Look for filter keywords in query
        for keyword in filter_keywords:
            if keyword in query_lower:
                # Extract surrounding context for better filtering
                words = query_lower.split()
                if keyword in words:
                    idx = words.index(keyword)
                    # Get 2 words before and after for context
                    start = max(0, idx - 2)
                    end = min(len(words), idx + 3)
                    context = " ".join(words[start:end])
                    logger.info(f"🔍 Extracted filter context: '{context}'")
                    return context
        
        return None