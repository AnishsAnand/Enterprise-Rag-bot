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
    MAX_LBS_FOR_LLM = 50
    
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
    
    async def _list_load_balancers(
        self,
        params: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        List load balancers with INTELLIGENT PRE-FILTERING.
        
        PRODUCTION LOGIC:
        1. Analyze user query BEFORE fetching data
        2. Detect if asking for specific LB or general list
        3. Use appropriate API endpoint
        4. Format response based on query type
        """
        try:
            user_roles = context.get("user_roles", [])
            user_id = context.get("user_id")
            user_query = context.get("user_query", "").lower()
            force_refresh = params.get("force_refresh", False)
            
            logger.info(f"📋 Processing LB query: '{user_query}'")
            
            # =====================================================================
            # CRITICAL FIX 1: Analyze query BEFORE fetching data
            # =====================================================================
            query_analysis = self._analyze_query_intent(user_query)
            
            if query_analysis["is_specific_lb"]:
                # User asked for SPECIFIC load balancer
                specific_lb_name = query_analysis["lb_identifier"]
                logger.info(f"🎯 SPECIFIC LB QUERY: '{specific_lb_name}'")
                
                # Try to get LBCI from name (need to fetch list first to get LBCI)
                # This is unavoidable - we need LBCI for details API
                raw_roles = context.get("user_roles")
                if isinstance(raw_roles, (list, tuple, set)):
                     user_roles = list(raw_roles)
                else:
                    user_roles = []
                ipc_engagement_id = await self._get_ipc_engagement_id(
                            user_id=user_id,
                            user_roles=user_roles,
                            force_refresh=force_refresh)

                if not ipc_engagement_id:
                    return {
                        "success": False,
                        "error": "Failed to get IPC engagement ID",
                        "response": "Unable to retrieve engagement information."
                    }
                
                # Fetch list to find LBCI (cached, so fast)
                list_result = await api_executor_service.list_load_balancers(
                    ipc_engagement_id=ipc_engagement_id,
                    user_id=user_id,
                    force_refresh=False  # Use cache
                )
                
                if not list_result.get("success"):
                    return {
                        "success": False,
                        "error": list_result.get("error"),
                        "response": f"Failed to find load balancer: {list_result.get('error')}"
                    }
                
                all_lbs = list_result.get("data", [])
                
                # Find matching LB
                matched_lb = self._find_matching_lb(specific_lb_name, all_lbs)
                
                if not matched_lb:
                    # No match found
                    available_names = [lb.get("name") for lb in all_lbs[:5]]
                    return {
                        "success": False,
                        "error": "Load balancer not found",
                        "response": (
                            f"❌ **Load Balancer Not Found**\n\n"
                            f"I couldn't find a load balancer named '{specific_lb_name}'.\n\n"
                            f"**Available load balancers:**\n" +
                            "\n".join([f"- {name}" for name in available_names if name]) +
                            f"\n\n💡 **Tip:** Use 'list load balancers' to see all available LBs."
                        ),
                        "metadata": {
                            "query_type": "specific",
                            "requested_name": specific_lb_name,
                            "available_lbs": available_names
                        }
                    }
                
                # Found the LB - get full details
                lbci = matched_lb.get("lbci") or matched_lb.get("circuitId") or matched_lb.get("LBCI")
                
                if lbci and query_analysis.get("wants_details"):
                    # User wants DETAILED information
                    logger.info(f"📊 Fetching DETAILED info for {matched_lb.get('name')}")
                    return await self._get_details_and_format(
                        matched_lb,
                        user_query,
                        query_analysis,
                        ipc_engagement_id,
                        user_id
                    )
                else:
                    # User just wants basic info about this specific LB
                    logger.info(f"📋 Returning BASIC info for {matched_lb.get('name')}")
                    
                    # Format single LB response
                    formatted_response = await self.format_response_with_llm(
                        operation="list",
                        raw_data=[matched_lb],  # Single LB in list
                        user_query=user_query,
                        context={
                            "query_type": "specific",
                            "lb_name": matched_lb.get("name"),
                            "total_count": 1,
                            "show_detail_hint": True,
                            "specific_lb_requested": True
                        }
                    )
                    
                    return {
                        "success": True,
                        "data": [matched_lb],
                        "response": formatted_response,
                        "metadata": {
                            "query_type": "specific",
                            "count": 1,
                            "lb_name": matched_lb.get("name"),
                            "lbci": lbci
                        }
                    }
            
            else:
                # User asked for GENERAL list (all LBs or filtered by location)
                logger.info(f"🌍 GENERAL LIST QUERY")
                
                # Get IPC engagement ID
                raw_roles = context.get("user_roles")
                if isinstance(raw_roles, (list, tuple, set)):
                    user_roles = list(raw_roles)
                else:
                    user_roles = []
                ipc_engagement_id = await self._get_ipc_engagement_id(
                        user_id=user_id,
                        user_roles=user_roles,
                        force_refresh=force_refresh)
                if not ipc_engagement_id:
                    return {
                        "success": False,
                        "error": "Failed to get IPC engagement ID",
                        "response": "Unable to retrieve engagement information."
                    }
                
                # Fetch all load balancers
                result = await api_executor_service.list_load_balancers(
                    ipc_engagement_id=ipc_engagement_id,
                    user_id=user_id,
                    force_refresh=force_refresh
                )
                
                if not result.get("success"):
                    error_msg = result.get("error", "Unknown error")
                    return {
                        "success": False,
                        "error": error_msg,
                        "response": f"Failed to retrieve load balancers: {error_msg}"
                    }
                
                load_balancers = result.get("data", [])
                total_count = len(load_balancers)
                is_cached = result.get("cached", False)
                
                logger.info(f"✅ Retrieved {total_count} load balancer(s)")
                
                # Apply location/feature filters if specified
                filtered_lbs = load_balancers
                filter_reason = None
                
                if query_analysis.get("location_filter"):
                    location = query_analysis["location_filter"]
                    logger.info(f"🔍 Filtering by location: {location}")
                    filtered_lbs = await self.filter_with_llm(
                        filtered_lbs,
                        f"location: {location}",
                        user_query
                    )
                    filter_reason = f"location: {location}"
                    total_count = len(filtered_lbs)
                
                if query_analysis.get("feature_filter"):
                    feature = query_analysis["feature_filter"]
                    logger.info(f"🔍 Filtering by feature: {feature}")
                    filtered_lbs = await self.filter_with_llm(
                        filtered_lbs,
                        f"feature: {feature}",
                        user_query
                    )
                    filter_reason = f"{filter_reason} + {feature}" if filter_reason else feature
                    total_count = len(filtered_lbs)
                
                # Format response
                formatted_response = await self.format_response_with_llm(
                    operation="list",
                    raw_data=filtered_lbs,
                    user_query=user_query,
                    context={
                        "query_type": "general",
                        "total_count": total_count,
                        "original_count": len(load_balancers),
                        "filter_applied": filter_reason is not None,
                        "filter_reason": filter_reason,
                        "cached": is_cached,
                        "ipc_engagement_id": ipc_engagement_id
                    }
                )
                
                # Handle empty results
                if total_count == 0:
                    if len(load_balancers) > 0:
                        formatted_response = (
                            f"⚖️ **No Load Balancers Match Your Criteria**\n\n"
                            f"Found {len(load_balancers)} total LBs, but none matched: {filter_reason}\n\n"
                            f"**Suggestions:**\n"
                            f"- Try 'list load balancers' to see all\n"
                            f"- Check spelling of location/feature name"
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
                        "query_type": "general",
                        "count": total_count,
                        "original_count": len(load_balancers),
                        "filter_applied": filter_reason is not None,
                        "filter_reason": filter_reason,
                        "cached": is_cached
                    }
                }
        
        except Exception as e:
            logger.error(f"❌ Error in _list_load_balancers: {str(e)}", exc_info=True)
            raise
    
    def _analyze_query_intent(self, user_query: str) -> Dict[str, Any]:
    
        query_lower = user_query.lower().strip()
        extracted_filters = self._extract_filter_criteria(query_lower)

        result = {
        "is_specific_lb": False,
        "lb_identifier": None,
        "wants_details": False,
        "wants_virtual_services": False,
        "location_filter": None,
        "feature_filters": [],
        "generic_filter": None}
        result.update({
    "location_filter": extracted_filters.get("location"),
    "status_filter": extracted_filters.get("status"),
    "feature_filters": extracted_filters.get("features", [])})


    # =====================================================================
    # Virtual services intent (independent signal)
    # =====================================================================
        vs_keywords = [
        "virtual service", "virtual services", "vip", "vips",
        "listener", "listeners", "frontend", "front end"
    ]
        result["wants_virtual_services"] = any(kw in query_lower for kw in vs_keywords)

    # =====================================================================
    # GENERAL vs SPECIFIC query detection (GENERAL takes precedence)
    # =====================================================================
        general_keywords = [
        "all load balancers",
        "all lbs",
        "list load balancers",
        "show load balancers",
        "show me load balancers",
        "list lbs",
        "show lbs",
        "how many load balancers",
        "count load balancers"
    ]

        if any(keyword in query_lower for keyword in general_keywords):
            logger.info(f"🌍 Detected GENERAL query: {user_query}")

        # Even for general queries, try to extract lightweight filters
            extracted_filter = self._extract_filter_criteria(user_query)
            if extracted_filter:
                result["generic_filter"] = extracted_filter

            return result

    # =====================================================================
    # SPECIFIC load balancer detection
    # =====================================================================
        specific_patterns = [
        r'load balancer[s]?\s+(?:named|called)\s+["\']?([a-zA-Z0-9_\-]+)',
        r'lb\s+(?:named|called)\s+["\']?([a-zA-Z0-9_\-]+)',
        r'(?:show|get|describe|details of)\s+(?:load balancer|lb)\s+["\']?([a-zA-Z0-9_\-]+)',
        r'([A-Z][A-Za-z0-9_]+_LB_[A-Z0-9_]+)'
        ]

        for pattern in specific_patterns:
            match = re.search(pattern, user_query, re.IGNORECASE)
            if match:
                result["is_specific_lb"] = True
                result["lb_identifier"] = match.group(1)
                logger.info(f"🎯 Detected SPECIFIC LB: {result['lb_identifier']}")
                break

    # =====================================================================
    # Detail intent
    # =====================================================================
        detail_keywords = [
        "details", "detail", "configuration", "config",
        "show me more", "more info", "information about",
        "describe", "explain", "what is"
    ]
        result["wants_details"] = any(kw in query_lower for kw in detail_keywords)

    # =====================================================================
    # Location filter (STRICT, avoids false positives)
    # =====================================================================
        location_pattern = r'\b(?:datacenter|dc|location|in|at|from)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)'
        match = re.search(location_pattern, user_query)
        if match:
            result["location_filter"] = match.group(1).strip()

    # =====================================================================
    # Feature filters (multi-value, non-overwriting)
    # =====================================================================
        feature_keywords = {
        "ssl": "ssl",
        "https": "https",
        "active": "active",
        "inactive": "inactive",
        "healthy": "healthy",
        "degraded": "degraded"
        }

        for keyword, value in feature_keywords.items():
            if keyword in query_lower:
                result["feature_filters"].append(value)

    # =====================================================================
    # Fallback: generic filter extraction (SAFE MODE)
    # Only if no explicit filters were found
    # =====================================================================
        if (
            not result["location_filter"]
            and not result["feature_filters"]
        ):
            extracted_filter = self._extract_filter_criteria(user_query)
            if extracted_filter:
                logger.info(f"🔍 Fallback generic filter detected: {extracted_filter}")
                result["generic_filter"] = extracted_filter

        return result

    
    def _find_matching_lb(
        self,
        lb_identifier: str,
        all_lbs: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Find load balancer matching the identifier.
        
        Checks:
        - name (exact and partial match)
        - lbci/circuitId (exact match)
        
        Returns first match or None.
        """
        identifier_lower = lb_identifier.lower()
        
        # Try exact name match first
        for lb in all_lbs:
            lb_name = lb.get("name", "").lower()
            if lb_name == identifier_lower:
                logger.info(f"✅ Exact name match: {lb.get('name')}")
                return lb
        
        # Try exact LBCI match
        for lb in all_lbs:
            lbci = (lb.get("lbci") or lb.get("circuitId") or lb.get("LBCI") or "").lower()
            if lbci == identifier_lower:
                logger.info(f"✅ Exact LBCI match: {lbci}")
                return lb
        
        # Try partial name match
        for lb in all_lbs:
            lb_name = lb.get("name", "").lower()
            if identifier_lower in lb_name or lb_name in identifier_lower:
                logger.info(f"✅ Partial name match: {lb.get('name')}")
                return lb
        
        logger.warning(f"❌ No match found for: {lb_identifier}")
        return None
    
    async def _get_ipc_engagement_id(
    self,
    user_id: str,
    user_roles=None,
    force_refresh: bool = False
) -> Optional[int]:
        """Get IPC engagement ID (helper method)."""

    # 🛡️ HARDEN against OpenWebUI / Gateway garbage
        if not user_roles or not isinstance(user_roles, (list, tuple, set)):
            user_roles = []

    # Fetch engagement based on roles
        engagement_id = await self.get_engagement_id(user_roles)

        if not engagement_id:
            return None

    # Fetch IPC engagement
        ipc_engagement_id = await api_executor_service.get_ipc_engagement_id(
        engagement_id=engagement_id,
        user_id=user_id,
        force_refresh=force_refresh
)

        return ipc_engagement_id

    
    async def _get_load_balancer_details(
        self,
        params: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get detailed configuration for a specific load balancer."""
        try:
            lbci = params.get("lbci")
            user_id = context.get("user_id")
            user_query = context.get("user_query", "")
            
            if not lbci:
                lb_data = params.get("load_balancer")
                if lb_data:
                    lbci = lb_data.get("lbci") or lb_data.get("circuitId")
                
                if not lbci:
                    return {
                        "success": False,
                        "error": "LBCI required",
                        "response": "Please specify which load balancer (need LBCI)."
                    }
            
            logger.info(f"🔍 Fetching details for: {lbci}")
            
            result = await api_executor_service.get_load_balancer_details(
                lbci=lbci,
                user_id=user_id
            )
            
            if not result.get("success"):
                return {
                    "success": False,
                    "error": result.get("error"),
                    "response": f"Failed to get details: {result.get('error')}"
                }
            
            details = result.get("data", {})
            
            formatted_response = await self.format_response_with_llm(
                operation="get_details",
                raw_data=details,
                user_query=user_query,
                context={
                    "lbci": lbci,
                    "query_type": "detailed",
                    "resource_type": "load_balancer_details"
                }
            )
            
            return {
                "success": True,
                "data": details,
                "response": formatted_response,
                "metadata": {
                    "lbci": lbci,
                    "query_type": "detailed"
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting details: {str(e)}", exc_info=True)
            raise
    
    async def _get_virtual_services(
        self,
        params: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get virtual services for a load balancer."""
        try:
            lbci = params.get("lbci")
            user_id = context.get("user_id")
            user_query = context.get("user_query", "")
            
            if not lbci:
                lb_data = params.get("load_balancer")
                if lb_data:
                    lbci = lb_data.get("lbci") or lb_data.get("circuitId")
                
                if not lbci:
                    return {
                        "success": False,
                        "error": "LBCI required",
                        "response": "Please specify which load balancer."
                    }
            
            logger.info(f"🌐 Fetching virtual services for: {lbci}")
            
            result = await api_executor_service.get_load_balancer_virtual_services(
                lbci=lbci,
                user_id=user_id
            )
            
            if not result.get("success"):
                return {
                    "success": False,
                    "error": result.get("error"),
                    "response": f"Failed to get virtual services: {result.get('error')}"
                }
            
            virtual_services = result.get("data", [])
            total = len(virtual_services)
            
            formatted_response = await self.format_response_with_llm(
                operation="get_virtual_services",
                raw_data=virtual_services,
                user_query=user_query,
                context={
                    "lbci": lbci,
                    "total": total,
                    "query_type": "virtual_services"
                }
            )
            
            if total == 0:
                formatted_response = (
                    f"🌐 **No Virtual Services Configured**\n\n"
                    f"Load balancer '{lbci}' has no virtual services configured."
                )
            
            return {
                "success": True,
                "data": virtual_services,
                "response": formatted_response,
                "metadata": {
                    "lbci": lbci,
                    "total": total,
                    "query_type": "virtual_services"
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting virtual services: {str(e)}", exc_info=True)
            raise
    
    async def _get_details_and_format(
        self,
        load_balancer: Dict[str, Any],
        user_query: str,
        query_intent: Dict[str, Any],
        ipc_engagement_id: int,
        user_id: str
    ) -> Dict[str, Any]:
        """Fetch and format detailed information for a specific LB."""
        lbci = load_balancer.get("lbci") or load_balancer.get("circuitId")
        
        if not lbci:
            return {
                "success": False,
                "error": "No LBCI found",
                "response": "Cannot fetch details: Load balancer ID not found."
            }
        
        # Fetch details
        details = None
        if query_intent.get("wants_details"):
            logger.info(f"📋 Fetching full details for {lbci}")
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
                "query_type": "specific_detailed",
                "has_details": details is not None,
                "has_virtual_services": virtual_services is not None
            }
        )
        
        return {
            "success": True,
            "data": combined_data,
            "response": formatted_response,
            "metadata": {
                "lbci": lbci,
                "query_type": "specific_detailed",
                "has_details": details is not None,
                "has_virtual_services": virtual_services is not None
            }
        }
    def _extract_filter_criteria(self, query_lower: str) -> Dict[str, Any]:
        filters = {
        "status": None,
        "location": None,
        "features": []
        }

    # Status
        if "active" in query_lower:
            filters["status"] = "active"
        elif "inactive" in query_lower:
            filters["status"] = "inactive"

    # Location (Mumbai, Delhi, Bengaluru, etc.)
        loc_match = re.search(r"\b(in|at|from)\s+([a-zA-Z\s]+)", query_lower)
        if loc_match:
            filters["location"] = loc_match.group(2).strip()

    # Features
        if "ssl" in query_lower or "https" in query_lower:
            filters["features"].append("ssl")

        return filters
