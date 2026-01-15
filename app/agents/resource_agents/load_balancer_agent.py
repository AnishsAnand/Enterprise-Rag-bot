"""
Load Balancer Agent - PRODUCTION READY
Complete implementation with list, get_details, and get_virtual_services operations.

PRODUCTION NOTES:
- Uses IPC engagement ID (NOT endpoint iteration)
- Directly wired to api_executor_service for all operations
- Intelligent query parsing for location/feature filtering
- Automatic detail fetching for specific queries
- Enhanced LLM formatting with context
"""

from typing import Any, Dict, List, Optional
import logging
import re

from app.agents.resource_agents.base_resource_agent import BaseResourceAgent
from app.services.api_executor_service import api_executor_service

logger = logging.getLogger(__name__)


class LoadBalancerAgent(BaseResourceAgent):
    """
    Complete agent for load balancer operations.
    
    Supported Operations:
    - list: List ALL load balancers with intelligent filtering
    - get_details: Get detailed configuration for a specific load balancer
    - get_virtual_services: Get virtual services (VIPs/listeners) for a load balancer
    
    PRODUCTION FEATURES:
    - Direct integration with api_executor_service
    - Uses IPC engagement ID (NOT endpoint iteration)
    - Intelligent query parsing for location/feature filtering
    - Automatic detail fetching for specific queries
    - Enhanced LLM formatting with context
    - User session caching for performance
    """
    
    def __init__(self):
        super().__init__(
            agent_name="LoadBalancerAgent",
            agent_description=(
                "Specialized agent for load balancer operations. "
                "Lists, filters, and analyzes load balancers with intelligent "
                "location and feature-based queries. Retrieves detailed "
                "configuration and virtual services."
            ),
            resource_type="load_balancer",
            temperature=0.2
        )
    
    def get_supported_operations(self) -> List[str]:
        """Get list of supported operations."""
        return ["list", "get_details", "get_virtual_services"]
    
    async def execute_operation(
        self,
        operation: str,
        params: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a load balancer operation with intelligent routing.
        
        Args:
            operation: Operation to perform
            params: Parameters
            context: Context (user_query, user_roles, user_id, etc.)
            
        Returns:
            Dict with success status and formatted response
        """
        try:
            logger.info(f"⚖️ LoadBalancerAgent executing: {operation}")
            
            if operation == "list":
                return await self._list_load_balancers(params, context)
            elif operation == "get_details":
                return await self._get_load_balancer_details(params, context)
            elif operation == "get_virtual_services":
                return await self._get_virtual_services(params, context)
            else:
                return {
                    "success": False,
                    "error": f"Unsupported operation: {operation}",
                    "response": f"I don't support the '{operation}' operation for load balancers yet. Currently supported: list, get_details, get_virtual_services."
                }
                
        except Exception as e:
            logger.error(f"❌ LoadBalancerAgent error: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "response": f"An error occurred while {operation}ing load balancers: {str(e)}"
            }
    
    async def _list_load_balancers(self,params: Dict[str, Any],context: Dict[str, Any]) -> Dict[str, Any]:

        try:
            user_roles = context.get("user_roles", [])
            user_id = context.get("user_id")
            user_query = context.get("user_query", "").lower()
            force_refresh = params.get("force_refresh", False)
        
            logger.info(f"📋 Listing load balancers for user: {user_id}")
            logger.info(f"🔍 User query: {user_query}")
        
        # Get IPC engagement ID
            ipc_engagement_id = params.get("ipc_engagement_id")
        
            if not ipc_engagement_id:
                engagement_id = await self.get_engagement_id(user_roles=user_roles)
                if not engagement_id:
                    return {
                    "success": False,
                    "error": "Failed to get engagement ID",
                    "response": "Unable to retrieve engagement information. Please contact support."
                    }
            
                ipc_engagement_id = await api_executor_service.get_ipc_engagement_id(
                    engagement_id=engagement_id,
                    user_id=user_id,
                    force_refresh=force_refresh
                )
            
                if not ipc_engagement_id:
                    return {
                    "success": False,
                    "error": "Failed to get IPC engagement ID",
                    "response": "Unable to retrieve IPC engagement information. Please contact support."
                }
            
                logger.info(f"✅ Got IPC engagement ID: {ipc_engagement_id}")
        
        # Call api_executor_service to list load balancers
            logger.info(f"📡 Calling api_executor_service.list_load_balancers")
        
            result = await api_executor_service.list_load_balancers(
            ipc_engagement_id=ipc_engagement_id,
            user_id=user_id,
            force_refresh=force_refresh
            )
        
            if not result.get("success"):
                error_msg = result.get("error", "Unknown error")
                logger.error(f"❌ api_executor_service.list_load_balancers failed: {error_msg}")
                return {
                "success": False,
                "error": error_msg,
                "response": f"Failed to retrieve load balancers: {error_msg}"
            }
        
            load_balancers = result.get("data", [])
            total_count = result.get("total", len(load_balancers))
            is_cached = result.get("cached", False)
        
            logger.info(f"✅ Retrieved {total_count} load balancer(s) {'(cached)' if is_cached else ''}")
        
        # ============================================================================
        # NEW: INTELLIGENT QUERY-BASED FILTERING
        # ============================================================================
        
            original_count = total_count
            filtered_lbs = load_balancers.copy()
            filter_reason = None
        
        # Extract specific LB name/LBCI from query
            specific_lb_name = self._extract_specific_lb_name(user_query, load_balancers)
        
            if specific_lb_name:
            # User asked for a SPECIFIC load balancer
                logger.info(f"🎯 User requested specific LB: '{specific_lb_name}'")
            
            # Filter to only this LB
                filtered_lbs = [
                lb for lb in load_balancers
                if (lb.get("name", "").lower() == specific_lb_name.lower() or
                    lb.get("lbci", "").lower() == specific_lb_name.lower() or
                    specific_lb_name.lower() in lb.get("name", "").lower())
                ]
            
                if filtered_lbs:
                    total_count = len(filtered_lbs)
                    filter_reason = f"specific LB '{specific_lb_name}'"
                    logger.info(f"✅ Filtered to {total_count} LB(s) matching '{specific_lb_name}'")
                else:
                # No match found
                    logger.warning(f"⚠️ No LB found matching '{specific_lb_name}'")
                    return {
                    "success": False,
                    "error": "Load balancer not found",
                    "response": (
                        f"❌ **Load Balancer Not Found**\n\n"
                        f"I couldn't find a load balancer named '{specific_lb_name}'.\n\n"
                        f"**Available load balancers:**\n" +
                        "\n".join([f"- {lb.get('name')} (LBCI: {lb.get('lbci')})" for lb in load_balancers[:5]]) +
                        f"\n\n💡 **Tip:** Check the spelling or use 'list load balancers' to see all."
                    ),
                    "metadata": {
                        "count": 0,
                        "available_lbs": [lb.get("name") for lb in load_balancers]
                    }
                }
        
        # Parse query intent for location/feature filters
            query_intent = self._parse_query_intent(user_query)
            logger.info(f"🎯 Query intent: {query_intent}")
        
        # Check if user wants details for a specific LB
            if query_intent.get("wants_details") and filtered_lbs:
                if len(filtered_lbs) == 1:
                # Exactly one LB - fetch details
                    logger.info(f"🔍 Fetching details for: {filtered_lbs[0].get('name')}")
                    return await self._get_details_and_format(
                    filtered_lbs[0],
                    user_query,
                    query_intent,
                    ipc_engagement_id,
                    user_id
                )
                else:
                # Multiple LBs - ask which one
                    return {
                    "success": True,
                    "response": (
                        f"🔍 **Multiple Load Balancers Found**\n\n"
                        f"I found {len(filtered_lbs)} load balancers. Which one would you like details for?\n\n" +
                        "\n".join([f"{i+1}. {lb.get('name')} ({lb.get('endpoint', 'Unknown')})" for i, lb in enumerate(filtered_lbs)])
                    ),
                    "data": filtered_lbs,
                    "metadata": {
                        "count": len(filtered_lbs),
                        "needs_clarification": True
                    }
                }
        
        # Apply additional location/feature filters if needed
            filter_criteria = self._extract_filter_criteria(user_query)
        
            if filter_criteria and not specific_lb_name and filtered_lbs:
                logger.info(f"🔍 Applying additional filter: {filter_criteria}")
                filtered_lbs = await self.filter_with_llm(
                filtered_lbs,
                filter_criteria,
                user_query
            )
                total_count = len(filtered_lbs)
                filter_reason = filter_criteria if not filter_reason else f"{filter_reason} + {filter_criteria}"
                logger.info(f"✅ After filtering: {total_count} LB(s)")
        
        # Format response with LLM
            formatted_response = await self.format_response_with_llm(
            operation="list",
            raw_data=filtered_lbs,
            user_query=user_query,
            context={
                "ipc_engagement_id": ipc_engagement_id,
                "total_count": total_count,
                "original_count": original_count,
                "filter_applied": filter_reason is not None,
                "filter_reason": filter_reason,
                "cached": is_cached,
                "query_intent": query_intent,
                "resource_type": "load_balancer",
                "show_detail_hint": total_count > 0 and total_count <= 3,
                "specific_lb_requested": specific_lb_name is not None
            }
        )
        
        # Handle empty results
            if total_count == 0:
                if original_count > 0 and filter_reason:
                    formatted_response = (
                    f"⚖️ **No Load Balancers Match Your Criteria**\n\n"
                    f"I found {original_count} total load balancer(s), but none matched: {filter_reason}\n\n"
                    f"**Suggestions:**\n"
                    f"- Check spelling of the load balancer name\n"
                    f"- Use 'list load balancers' to see all available LBs\n"
                    f"- Try a different search term"
                )
                else:
                    formatted_response = (
                    f"⚖️ **No Load Balancers Found**\n\n"
                    f"Your engagement currently has no load balancers configured.\n\n"
                    f"💡 **Tip:** Contact your administrator to create load balancers."
                )
        
            return {
            "success": True,
            "data": filtered_lbs,
            "response": formatted_response,
            "metadata": {
                "count": total_count,
                "original_count": original_count,
                "ipc_engagement_id": ipc_engagement_id,
                "filter_applied": filter_reason is not None,
                "filter_reason": filter_reason,
                "cached": is_cached,
                "query_intent": query_intent,
                "resource_type": "load_balancer",
                "specific_lb_requested": specific_lb_name is not None
            }
        }
        
        except Exception as e:
            logger.error(f"❌ Error listing load balancers: {str(e)}", exc_info=True)
            raise

    def _extract_specific_lb_name(self,user_query: str,load_balancers: List[Dict[str, Any]]) -> Optional[str]:

        query_lower = user_query.lower()
    
    # Keywords that indicate a GENERAL query (not specific)
        general_keywords = [
        "all load balancers",
        "all lbs",
        "list load balancers",
        "show load balancers",
        "show me load balancers",
        "list lbs",
        "show lbs"
    ]
    
    # If query is clearly general, return None
        for keyword in general_keywords:
            if query_lower.strip() == keyword:
                logger.info(f"🌍 General query detected: '{user_query}'")
                return None
    
    # Look for actual LB names in the query
        for lb in load_balancers:
            lb_name = lb.get("name", "")
            lbci = lb.get("lbci", "") or lb.get("circuitId", "")
        
        # Check if LB name appears in query
            if lb_name and lb_name.lower() in query_lower:
                logger.info(f"🎯 Found LB name in query: '{lb_name}'")
                return lb_name
        
        # Check if LBCI appears in query
            if lbci and lbci.lower() in query_lower:
                logger.info(f"🎯 Found LBCI in query: '{lbci}'")
                return lbci
    
    # Pattern matching for LB names (e.g., "EG_Tata_Com_167_LB_SEG_388")
    # LB names typically contain underscores and follow patterns
        import re
        lb_name_pattern = r'[A-Z][A-Za-z0-9_]+_LB_[A-Z0-9_]+'
        matches = re.findall(lb_name_pattern, user_query)
    
        if matches:
            potential_name = matches[0]
            logger.info(f"🎯 Pattern matched potential LB name: '{potential_name}'")
        
        # Verify it exists in our LB list
            for lb in load_balancers:
                if potential_name.lower() in (lb.get("name", "").lower() or lb.get("lbci", "").lower()):
                    return potential_name
    
    # No specific LB found
        logger.info(f"🌍 No specific LB name found in query")
        return None
    
    async def _get_load_balancer_details(
        self,
        params: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get detailed configuration for a specific load balancer.
        
        Args:
            params: Parameters including:
                - lbci: str - Load Balancer Circuit ID (REQUIRED)
                - load_balancer: dict - Optional LB data with lbci
            context: Context including:
                - user_query: str
                - user_id: str
                
        Returns:
            Dict with detailed load balancer information
        """
        try:
            lbci = params.get("lbci")
            user_id = context.get("user_id")
            user_query = context.get("user_query", "")
            
            # Try to extract LBCI from load_balancer data if not directly provided
            if not lbci:
                lb_data = params.get("load_balancer")
                if lb_data:
                    lbci = lb_data.get("lbci") or lb_data.get("circuitId") or lb_data.get("LBCI")
                
                if not lbci:
                    return {
                        "success": False,
                        "error": "LBCI (Load Balancer Circuit ID) is required",
                        "response": "Please specify which load balancer you want details for. I need the LBCI (Load Balancer Circuit ID)."
                    }
            
            logger.info(f"🔍 Fetching details for load balancer: {lbci}")
            
            # Call api_executor_service to get details
            result = await api_executor_service.get_load_balancer_details(
                lbci=lbci,
                user_id=user_id
            )
            
            if not result.get("success"):
                error_msg = result.get("error", "Unknown error")
                logger.error(f"❌ api_executor_service.get_load_balancer_details failed: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "response": f"Failed to get details for {lbci}: {error_msg}"
                }
            
            details = result.get("data", {})
            
            logger.info(f"✅ Retrieved details for load balancer: {lbci}")
            
            # Format with LLM
            formatted_response = await self.format_response_with_llm(
                operation="get_details",
                raw_data=details,
                user_query=user_query,
                context={
                    "lbci": lbci,
                    "resource_type": "load_balancer_details",
                    "detail_level": "comprehensive"
                }
            )
            
            return {
                "success": True,
                "data": details,
                "response": formatted_response,
                "metadata": {
                    "lbci": lbci,
                    "resource_type": "load_balancer_details",
                    "operation": "get_details"
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting LB details: {str(e)}", exc_info=True)
            raise
    
    async def _get_virtual_services(
        self,
        params: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get virtual services for a load balancer.
        
        Args:
            params: Parameters including:
                - lbci: str - Load Balancer Circuit ID (REQUIRED)
                - load_balancer: dict - Optional LB data with lbci
            context: Context including:
                - user_query: str
                - user_id: str
                
        Returns:
            Dict with virtual services list
        """
        try:
            lbci = params.get("lbci")
            user_id = context.get("user_id")
            user_query = context.get("user_query", "")
            
            # Try to extract LBCI from load_balancer data if not directly provided
            if not lbci:
                lb_data = params.get("load_balancer")
                if lb_data:
                    lbci = lb_data.get("lbci") or lb_data.get("circuitId") or lb_data.get("LBCI")
                
                if not lbci:
                    return {
                        "success": False,
                        "error": "LBCI is required",
                        "response": "Please specify which load balancer's virtual services you want. I need the LBCI (Load Balancer Circuit ID)."
                    }
            
            logger.info(f"🌐 Fetching virtual services for: {lbci}")
            
            # Call api_executor_service to get virtual services
            result = await api_executor_service.get_load_balancer_virtual_services(
                lbci=lbci,
                user_id=user_id
            )
            
            if not result.get("success"):
                error_msg = result.get("error", "Unknown error")
                logger.error(f"❌ api_executor_service.get_load_balancer_virtual_services failed: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "response": f"Failed to get virtual services for {lbci}: {error_msg}"
                }
            
            virtual_services = result.get("data", [])
            total = result.get("total", len(virtual_services))
            
            logger.info(f"✅ Retrieved {total} virtual service(s) for {lbci}")
            
            # Format with LLM
            formatted_response = await self.format_response_with_llm(
                operation="get_virtual_services",
                raw_data=virtual_services,
                user_query=user_query,
                context={
                    "lbci": lbci,
                    "total": total,
                    "resource_type": "load_balancer_virtual_services"
                }
            )
            
            # Handle empty results
            if total == 0:
                formatted_response = (
                    f"🌐 **No Virtual Services Configured**\n\n"
                    f"Load balancer '{lbci}' currently has no virtual services (VIPs) configured.\n\n"
                    f"**What are Virtual Services?**\n"
                    f"Virtual services are the individual VIP:port combinations that clients connect to. "
                    f"Each virtual service routes traffic to a backend pool of servers.\n\n"
                    f"💡 **Tip:** Configure virtual services through the cloud portal to start routing traffic."
                )
            
            return {
                "success": True,
                "data": virtual_services,
                "response": formatted_response,
                "metadata": {
                    "lbci": lbci,
                    "total": total,
                    "resource_type": "load_balancer_virtual_services"
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting virtual services: {str(e)}", exc_info=True)
            raise
    
    def _parse_query_intent(self, user_query: str) -> Dict[str, Any]:
        """
        Parse user query to understand intent.
        
        Returns dict with:
        - wants_details: bool - User wants detailed info
        - wants_virtual_services: bool - User wants VIP/listener info
        - specific_lb: str - Name/ID of specific LB
        - location_filter: str - Location/endpoint filter
        - feature_filter: str - Feature/protocol filter
        """
        query_lower = user_query.lower()
        
        intent = {
            "wants_details": False,
            "wants_virtual_services": False,
            "specific_lb": None,
            "location_filter": None,
            "feature_filter": None
        }
        
        # Detail indicators
        detail_keywords = [
            "details", "detail", "configuration", "config", "settings",
            "show me more", "more info", "information about",
            "describe", "explain", "what is"
        ]
        intent["wants_details"] = any(kw in query_lower for kw in detail_keywords)
        
        # Virtual service indicators
        vs_keywords = [
            "virtual service", "vip", "listener", "port", "protocol",
            "endpoint", "frontend", "backend pool"
        ]
        intent["wants_virtual_services"] = any(kw in query_lower for kw in vs_keywords)
        
        # Location indicators
        location_patterns = [
            r"in\s+(\w+)",
            r"at\s+(\w+)",
            r"from\s+(\w+)",
            r"(\w+)\s+datacenter",
            r"(\w+)\s+region",
            r"(\w+)\s+location"
        ]
        for pattern in location_patterns:
            match = re.search(pattern, query_lower)
            if match:
                intent["location_filter"] = match.group(1)
                break
        
        # Feature/protocol indicators
        feature_keywords = {
            "ssl": "ssl",
            "https": "https",
            "http": "http",
            "tcp": "tcp",
            "udp": "udp",
            "active": "active",
            "enabled": "enabled",
            "production": "production",
            "staging": "staging"
        }
        for keyword, value in feature_keywords.items():
            if keyword in query_lower:
                intent["feature_filter"] = value
                break
        
        return intent
    
    def _identify_target_lb(
        self,
        user_query: str,
        load_balancers: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Identify which specific load balancer the user is asking about.
        
        Returns the matching LB or None if ambiguous.
        """
        if not load_balancers:
            return None
        
        query_lower = user_query.lower()
        
        # Look for LB name in query
        for lb in load_balancers:
            lb_name = lb.get("name", "").lower()
            if lb_name and lb_name in query_lower:
                logger.info(f"🎯 Found target LB by name: {lb.get('name')}")
                return lb
        
        # Look for LBCI in query
        for lb in load_balancers:
            lbci = lb.get("lbci", "") or lb.get("circuitId", "") or lb.get("LBCI", "")
            if lbci and lbci.lower() in query_lower:
                logger.info(f"🎯 Found target LB by LBCI: {lbci}")
                return lb
        
        # If only one LB and query asks for details, assume it's that one
        if len(load_balancers) == 1 and ("details" in query_lower or "info" in query_lower):
            logger.info(f"🎯 Only one LB, assuming target: {load_balancers[0].get('name')}")
            return load_balancers[0]
        
        return None
    
    async def _get_details_and_format(
        self,
        load_balancer: Dict[str, Any],
        user_query: str,
        query_intent: Dict[str, Any],
        ipc_engagement_id: int,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Fetch details and/or virtual services for a specific LB and format response.
        """
        lbci = load_balancer.get("lbci") or load_balancer.get("circuitId") or load_balancer.get("LBCI")
        
        if not lbci:
            return {
                "success": False,
                "error": "Load balancer has no LBCI",
                "response": "Cannot fetch details: Load balancer ID not found."
            }
        
        # Fetch details if requested
        details = None
        if query_intent.get("wants_details"):
            logger.info(f"📋 Fetching details for {lbci}")
            details_result = await api_executor_service.get_load_balancer_details(
                lbci=lbci,
                user_id=user_id
            )
            if details_result.get("success"):
                details = details_result.get("data")
        
        # Fetch virtual services if requested
        virtual_services = None
        if query_intent.get("wants_virtual_services"):
            logger.info(f"🌐 Fetching virtual services for {lbci}")
            vs_result = await api_executor_service.get_load_balancer_virtual_services(
                lbci=lbci,
                user_id=user_id
            )
            if vs_result.get("success"):
                virtual_services = vs_result.get("data")
        
        # Combine all data
        combined_data = {
            "load_balancer": load_balancer,
            "details": details,
            "virtual_services": virtual_services
        }
        
        # Format with LLM
        formatted_response = await self.format_response_with_llm(
            operation="detailed_view",
            raw_data=combined_data,
            user_query=user_query,
            context={
                "lbci": lbci,
                "has_details": details is not None,
                "has_virtual_services": virtual_services is not None,
                "query_intent": query_intent,
                "ipc_engagement_id": ipc_engagement_id,
                "resource_type": "load_balancer_detailed"
            }
        )
        
        return {
            "success": True,
            "data": combined_data,
            "response": formatted_response,
            "metadata": {
                "lbci": lbci,
                "has_details": details is not None,
                "has_virtual_services": virtual_services is not None,
                "resource_type": "load_balancer_detailed"
            }
        }
    
    def _extract_filter_criteria(self, user_query: str) -> Optional[str]:
        """Extract filter criteria from user query."""
        if not user_query:
            return None
        
        query_lower = user_query.lower()
        
        # Filter keywords
        filter_keywords = [
            "active", "inactive", "enabled", "disabled",
            "production", "staging", "development", "prod", "stage", "dev",
            "ssl", "https", "http", "tcp", "udp",
            "healthy", "unhealthy", "degraded",
            "high", "low", "traffic", "load",
            "public", "private", "internal", "external",
            "running", "stopped", "failed",
            # Location keywords
            "mumbai", "bangalore", "delhi", "chennai", "hyderabad",
            "pune", "kolkata", "ahmedabad"
        ]
        
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