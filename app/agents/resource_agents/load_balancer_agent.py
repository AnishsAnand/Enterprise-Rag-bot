"""
Load Balancer Agent - PRODUCTION READY
Complete implementation with list, get_details, and get_virtual_services operations.
"""

from typing import Any, Dict, List, Optional
import logging
import re

from app.agents.resource_agents.base_resource_agent import BaseResourceAgent
from app.services.api_executor_service import api_executor_service
import json 
logger = logging.getLogger(__name__)


class LoadBalancerAgent(BaseResourceAgent):
    """
    Complete agent for load balancer operations.
    
    Supported Operations:
    - list: List ALL load balancers with intelligent filtering
    - get_details: Get detailed configuration for a specific load balancer
    - get_virtual_services: Get virtual services (VIPs/listeners) for a load balancer
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
    
    async def _list_load_balancers(self, params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        List load balancers with AUTOMATIC detail fetching for specific LB queries.
        """
        try:
            # Extract context - MUST be done FIRST
            user_roles = context.get("user_roles", [])
            user_id = context.get("user_id")
            auth_token = context.get("auth_token")
            user_type = context.get("user_type")
            selected_engagement_id = context.get("selected_engagement_id")
            user_query = context.get("user_query", "").lower()
            force_refresh = params.get("force_refresh", False)
            
            logger.info(f"🔐 LB listing with auth_token: {'✓' if auth_token else '✗'}, engagement: {selected_engagement_id}")
            logger.info(f"📋 Processing LB query: '{user_query}'")
            
            # ═════════════════════════════════════════════════════════════════
            # STEP 1: Analyze query intent
            # ═════════════════════════════════════════════════════════════════
            query_analysis = self._analyze_query_intent(user_query)
            if query_analysis["is_specific_lb"]:
                specific_lb_name = query_analysis["lb_identifier"]
                list_result = await api_executor_service.list_load_balancers(...)
                all_lbs = list_result.get("data", [])
                matched_lb = self._find_matching_lb(specific_lb_name, all_lbs)
                lbci = matched_lb.get("lbci") or matched_lb.get("circuitId")
                return await self._fetch_complete_lb_details(load_balancer=matched_lb,
                                                             user_query=user_query,
                                                             query_intent=query_analysis,user_id=user_id,lbci=lbci,auth_token=auth_token)
            # ═════════════════════════════════════════════════════════════════
            # STEP 1.5: Handle DIRECT LBCI queries FIRST (if user knows LBCI)
            # ═════════════════════════════════════════════════════════════════
            if query_analysis.get("is_lbci_query"):
                lbci = query_analysis.get("lb_identifier")
                logger.info(f"🎯 DIRECT LBCI QUERY DETECTED: {lbci}")
                
                return await self._handle_lbci_query(
                    lbci=lbci,
                    user_query=user_query,
                    user_id=user_id,
                    auth_token=auth_token
                )
            
            # ═════════════════════════════════════════════════════════════════
            # STEP 2: Handle SPECIFIC load balancer queries (by NAME)
            # ═════════════════════════════════════════════════════════════════
            if query_analysis["is_specific_lb"]:
                specific_lb_name = query_analysis["lb_identifier"]
                logger.info(f"🎯 SPECIFIC LB QUERY BY NAME: '{specific_lb_name}'")
                
                # First, get list of all LBs to find the one user wants
                raw_roles = context.get("user_roles")
                user_roles = list(raw_roles) if isinstance(raw_roles, (list, tuple, set)) else []
                ipc_engagement_id = await self._get_ipc_engagement_id(
                    user_id=user_id,
                    user_roles=user_roles,
                    force_refresh=force_refresh,
                    auth_token=auth_token,
                    selected_engagement_id=selected_engagement_id
                )
                
                if not ipc_engagement_id:
                    return {
                        "success": False,
                        "error": "Failed to get IPC engagement ID",
                        "response": "Unable to retrieve engagement information. Please select an engagement first."
                    }
                
                # Get list of all load balancers
                list_result = await api_executor_service.list_load_balancers(
                    ipc_engagement_id=ipc_engagement_id,
                    user_id=user_id,
                    force_refresh=False,
                    auth_token=auth_token
                )
                
                if not list_result.get("success"):
                    return {
                        "success": False,
                        "error": list_result.get("error"),
                        "response": f"Failed to find load balancer: {list_result.get('error')}"
                    }
                
                all_lbs = list_result.get("data", [])
                
                # Find matching LB (fuzzy matching)
                matched_lb = self._find_matching_lb(specific_lb_name, all_lbs)
                
                if not matched_lb:
                    # No match - provide helpful error
                    available_names = [lb.get("name") for lb in all_lbs[:5] if lb.get("name")]
                    return {
                        "success": False,
                        "error": "Load balancer not found",
                        "response": (
                            f"❌ **Load Balancer Not Found**\n\n"
                            f"I couldn't find a load balancer matching '{specific_lb_name}'.\n\n"
                            f"**Available load balancers:**\n" +
                            "\n".join([f"- {name}" for name in available_names]) +
                            f"\n\n💡 **Tip:** Use 'list load balancers' to see all available LBs."
                        ),
                        "metadata": {
                            "query_type": "specific",
                            "requested_name": specific_lb_name,
                            "available_lbs": available_names
                        }
                    }
                
                # Found the LB - extract LBCI
                lbci = matched_lb.get("lbci") or matched_lb.get("circuitId") or matched_lb.get("LBCI")
                
                if not lbci:
                    # LB found but no LBCI - return basic info only
                    logger.warning(f"⚠️ LB '{matched_lb.get('name')}' has no LBCI, returning basic info only")
                    
                    formatted_response = await self.format_response_with_llm(
                        operation="list",
                        raw_data=[matched_lb],
                        user_query=user_query,
                        context={
                            "query_type": "specific",
                            "lb_name": matched_lb.get("name"),
                            "total_count": 1,
                            "no_lbci": True
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
                            "lbci": None,
                            "warning": "No LBCI found for this load balancer"
                        }
                    }
                
                # ═════════════════════════════════════════════════════════
                # CRITICAL: ALWAYS fetch complete details when user asks about specific LB
                # ═════════════════════════════════════════════════════════
                logger.info(f"📊 AUTOMATICALLY fetching COMPLETE details for {matched_lb.get('name')} (LBCI: {lbci})")
                
                return await self._fetch_complete_lb_details(
                    load_balancer=matched_lb,
                    user_query=user_query,
                    query_intent=query_analysis,
                    user_id=user_id,
                    lbci=lbci,
                    auth_token=auth_token
                )
            
            # ═════════════════════════════════════════════════════════════════
            # STEP 3: Handle GENERAL list queries
            # ═════════════════════════════════════════════════════════════════
            else:
                logger.info(f"🌍 GENERAL LIST QUERY")
                raw_roles = context.get("user_roles")
                user_roles = list(raw_roles) if isinstance(raw_roles, (list, tuple, set)) else []
                ipc_engagement_id = await self._get_ipc_engagement_id(
                    user_id=user_id,
                    user_roles=user_roles,
                    force_refresh=force_refresh,
                    auth_token=auth_token,
                    selected_engagement_id=selected_engagement_id
                )
                
                if not ipc_engagement_id:
                    return {
                        "success": False,
                        "error": "Failed to get IPC engagement ID",
                        "response": "Unable to retrieve engagement information. Please select an engagement first."
                    }
                
                result = await api_executor_service.list_load_balancers(
                    ipc_engagement_id=ipc_engagement_id,
                    user_id=user_id,
                    force_refresh=force_refresh,
                    auth_token=auth_token
                )
                
                if not result.get("success"):
                    error_msg = result.get("error", "Unknown error")
                    return {
                        "success": False,
                        "error": error_msg,
                        "response": f"Failed to retrieve load balancers: {error_msg}"
                    }
                
                load_balancers = result.get("data", [])
                original_count = len(load_balancers)
                is_cached = result.get("cached", False)
                
                logger.info(f"✅ Retrieved {original_count} load balancer(s) from API")
                
                # Enrich with location data
                enriched_lbs = await self._enrich_load_balancers_with_location(
                    load_balancers,
                    user_id,
                    ipc_engagement_id
                )
                
                # Apply filters
                filtered_lbs = enriched_lbs
                filter_reasons = []
                
                if query_analysis.get("location_filter"):
                    location = query_analysis["location_filter"]
                    logger.info(f"🔍 Applying location filter: {location}")
                    filtered_lbs = self._filter_by_location(filtered_lbs, location)
                    filter_reasons.append(f"location: {location}")
                    logger.info(f"   → {len(filtered_lbs)} LBs matched")
                
                if query_analysis.get("status_filter"):
                    status = query_analysis["status_filter"]
                    logger.info(f"🔍 Applying status filter: {status}")
                    filtered_lbs = self._filter_by_status(filtered_lbs, status)
                    filter_reasons.append(f"status: {status}")
                    logger.info(f"   → {len(filtered_lbs)} LBs matched")
                
                if query_analysis.get("feature_filters"):
                    for feature in query_analysis["feature_filters"]:
                        logger.info(f"🔍 Applying feature filter: {feature}")
                        filtered_lbs = self._filter_by_feature(filtered_lbs, feature)
                        filter_reasons.append(f"feature: {feature}")
                        logger.info(f"   → {len(filtered_lbs)} LBs matched")
                
                total_count = len(filtered_lbs)
                filter_reason = " + ".join(filter_reasons) if filter_reasons else None
                
                # Format response with agentic formatter (prevents hallucination for large lists)
                formatted_response = await self.format_response_agentic(
                    operation="list",
                    raw_data=filtered_lbs,
                    user_query=user_query,
                    context={
                        "query_type": "general",
                        "total_count": total_count,
                        "original_count": original_count,
                        "filter_applied": filter_reason is not None,
                        "filter_reason": filter_reason,
                        "cached": is_cached,
                        "ipc_engagement_id": ipc_engagement_id
                    }
                )
                
                # Handle empty results
                if total_count == 0:
                    if original_count > 0:
                        available_locations = sorted(list(set([
                            lb.get("_location") 
                            for lb in enriched_lbs 
                            if lb.get("_location") and lb.get("_location") != "Unknown"
                        ])))
                        
                        formatted_response = (
                            f"⚖️ **No Load Balancers Match Your Criteria**\n\n"
                            f"Found {original_count} total LBs, but none matched: **{filter_reason}**\n\n"
                            f"**Available locations:**\n" +
                            "\n".join([f"- {loc}" for loc in available_locations[:10]]) +
                            f"\n\n**Suggestions:**\n"
                            f"- Try 'list load balancers' to see all\n"
                            f"- Check spelling: '{filter_reason}'\n"
                            f"- Use format: 'list load balancers in [location]'"
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
                        "original_count": original_count,
                        "filter_applied": filter_reason is not None,
                        "filter_reason": filter_reason,
                        "cached": is_cached
                    }
                }
        
        except Exception as e:
            logger.error(f"❌ Error in _list_load_balancers: {str(e)}", exc_info=True)
            raise

    async def _fetch_complete_lb_details(self, load_balancer: Dict[str, Any], user_query: str, query_intent: Dict[str, Any], user_id: str, lbci: str, auth_token: str = None) -> Dict[str, Any]:
        """
    Fetch COMPLETE details for a specific load balancer.
    Called when user asks about a specific LB by name or LBCI.
    
    ALWAYS fetches:
    1. Configuration details (getDetails API)
    2. Virtual services (virtualservices API)
    
    Args:
        load_balancer: Basic LB info from list
        user_query: Original user query
        query_intent: Parsed query intent
        user_id: User ID
        lbci: Load Balancer Circuit ID
        auth_token: Bearer token for API authentication
        
    Returns:
        Dict with complete LB details
    """
        logger.info(f"📊 Fetching COMPLETE details for {load_balancer.get('name')} (LBCI: {lbci})")
    
    # 1️⃣ Fetch configuration details
        logger.info(f"📋 Fetching configuration details...")
        details_result = await api_executor_service.get_load_balancer_details(
            lbci=lbci,
            user_id=user_id,
            auth_token=auth_token)
        details = None
        details_error = None
        if details_result.get("success"):
            details = details_result.get("data")
            logger.info(f"✅ Got configuration details")
        else:
            details_error = details_result.get("error")
            logger.warning(f"⚠️ Details failed: {details_error}")
    # 2️⃣ Fetch virtual services
        logger.info(f"🌐 Fetching virtual services...")
        vs_result = await api_executor_service.get_load_balancer_virtual_services(
            lbci=lbci,
            user_id=user_id,
            auth_token=auth_token)
    
        virtual_services = []
        vs_error = None
        if vs_result.get("success"):
            virtual_services = vs_result.get("data", [])
            logger.info(f"✅ Got {len(virtual_services)} virtual services")
        else:
            vs_error = vs_result.get("error")
            logger.warning(f"⚠️ Virtual services failed: {vs_error}")
    # 3️⃣ Build combined data structure
        combined_data = {
        "load_balancer": load_balancer,
        "details": details,
        "virtual_services": virtual_services,
        "errors": {
            "details": details_error,
            "virtual_services": vs_error}}
    # 4️⃣ Format using ENHANCED detailed formatter
        response = await self._format_detailed_response_with_llm(
        raw_data=combined_data,
        user_query=user_query,
        context={
            "lbci": lbci,
            "lb_name": load_balancer.get("name"),
            "vs_count": len(virtual_services),
            "query_type": "specific_detailed",
            "has_details": details is not None,
            "has_virtual_services": len(virtual_services) > 0})
        return {
        "success": True,
        "data": combined_data,
        "response": response,
        "metadata": {
            "lbci": lbci,
            "lb_name": load_balancer.get("name"),
            "vs_count": len(virtual_services),
            "query_type": "specific_detailed",
            "has_details": details is not None,
            "has_virtual_services": len(virtual_services) > 0}}

    async def _handle_lbci_query(self, lbci: str, user_query: str, user_id: str = None, auth_token: str = None) -> Dict[str, Any]:
        logger.info(f"⚖️ LBCI QUERY HANDLER: {lbci}")
    # 1️⃣ Fetch virtual services (MANDATORY - this is what user wants!)
        logger.info(f"🌐 Fetching virtual services for LBCI {lbci}...")
        vs_result = await api_executor_service.get_load_balancer_virtual_services(
            lbci=lbci,
            user_id=user_id,
            auth_token=auth_token
        )
        virtual_services = []
        vs_error = None
        if vs_result.get("success"):
            virtual_services = vs_result.get("data", [])
            logger.info(f"✅ Got {len(virtual_services)} virtual services")
        else:
            vs_error = vs_result.get("error")
            logger.warning(f"⚠️ Virtual services failed: {vs_error}")
    # 2️⃣ Fetch LB details (OPTIONAL but recommended for complete picture)
        logger.info(f"📋 Fetching configuration details for LBCI {lbci}...")
        details_result = await api_executor_service.get_load_balancer_details(
            lbci=lbci,
            user_id=user_id,
            auth_token=auth_token)
        details = None
        details_error = None
        if details_result.get("success"):
            details = details_result.get("data")
            logger.info(f"✅ Got configuration details")
        else:
            details_error = details_result.get("error")
            logger.warning(f"⚠️ Details failed: {details_error}")
    
    # 3️⃣ Build combined data structure
        combined_data = {
        "load_balancer": details or {"lbci": lbci, "name": f"LB-{lbci}"},
        "details": details,
        "virtual_services": virtual_services,
        "errors": {
            "details": details_error,
            "virtual_services": vs_error}}
    # 4️⃣ Format using ENHANCED detailed formatter
        response = await self._format_detailed_response_with_llm(
        raw_data=combined_data,
        user_query=user_query,
        context={
            "lbci": lbci,
            "lb_name": details.get("name") if details else f"LB-{lbci}",
            "vs_count": len(virtual_services),
            "query_type": "lbci_direct",
            "is_lbci_query": True,
            "has_details": details is not None,
            "has_virtual_services": len(virtual_services) > 0
        }
    )
        return {
        "success": True,
        "data": combined_data,
        "response": response,
        "metadata": {
            "lbci": lbci,
            "vs_count": len(virtual_services),
            "query_type": "lbci_direct",
            "has_details": details is not None,
            "has_virtual_services": len(virtual_services) > 0
        }
    }

    def _filter_by_location(self, load_balancers: List[Dict[str, Any]], location: str) -> List[Dict[str, Any]]:
        location_normalized = location.lower().replace("-", "").replace(" ", "")
        filtered = [
            lb for lb in load_balancers
            if location_normalized in lb.get("_location_normalized", "")
        ]
        logger.info(f"🔍 Location filter '{location}': {len(filtered)}/{len(load_balancers)} matched")
        return filtered
    def _filter_by_status(self, load_balancers: List[Dict[str, Any]], status: str) -> List[Dict[str, Any]]:
        status_normalized = status.lower()
        filtered = [
        lb for lb in load_balancers
        if status_normalized in lb.get("status", "").lower()
        ]
        return filtered
    def _filter_by_feature(self, load_balancers: List[Dict[str, Any]], feature: str) -> List[Dict[str, Any]]:
        feature_lower = feature.lower()
        filtered = []
        for lb in load_balancers:
        # Check SSL/HTTPS
            if feature_lower in ["ssl", "https"]:
                if (lb.get("ssl_enabled") or 
                "https" in lb.get("protocol", "").lower() or
                lb.get("port") == 443):
                    filtered.append(lb)
        # Check protocol
            elif feature_lower in lb.get("protocol", "").lower():
                filtered.append(lb)
        return filtered
    
    async def _format_detailed_response_with_llm(self,raw_data: Dict[str, Any],user_query: str,context: Dict[str, Any]) -> str:
        """Format load balancer with virtual services in user-friendly format."""
        from app.services.ai_service import ai_service
    
        lb = raw_data.get("load_balancer", {})
        details = raw_data.get("details", {})
        virtual_services = raw_data.get("virtual_services", [])
        errors = raw_data.get("errors", {})
    
        lb_name = context.get("lb_name", "Unknown")
        lbci = context.get("lbci", "N/A")
        vs_count = len(virtual_services)
    
    # Build PRODUCTION-READY prompt
        prompt = f"""You are formatting load balancer information for a network engineer. 
Format this data in a clear, professional manner suitable for production operations.

**Load Balancer:** {lb_name}
**LBCI:** {lbci}

**Virtual Services Data:**
{json.dumps(virtual_services, indent=2)}

**Configuration Details:**
{json.dumps(details, indent=2) if details else "Configuration details unavailable"}

**REQUIRED FORMAT:**

# Load Balancer: {lb_name}

**Circuit ID (LBCI):** `{lbci}`

## Virtual Services ({vs_count} configured)

For each virtual service, display:

### [Number]. [Virtual Server Name]

| Property | Value |
|----------|-------|
| **VIP Address** | [vipIp]:[virtualServerport] |
| **Protocol** | [protocol] |
| **Status** | [emoji] [status] |
| **Load Balancing** | [poolAlgorithm] |
| **Health Monitor** | [monitor array as comma-separated] |
| **Persistence** | [persistenceType] ([persistenceValue]) |
| **Pool Members** | [poolMembers count or details] |
| **Pool Path** | `[virtualServerPath]` |

**Status Icons:**
- UP = ✅
- DOWN = ⚠️
- DEGRADED = 🟡
- UNKNOWN = ❓

**Special Notes:**
- If certificate is configured, mention: **SSL Certificate:** [certificateName]
- If pool members exist, list them
- If persistence is null, show "None"
- Use proper formatting with tables for readability

**Example Output:**

### 1. TESTPUBLIC

| Property | Value |
|----------|-------|
| **VIP Address** | 100.94.45.12:9056 |
| **Protocol** | HTTP |
| **Status** | ⚠️ DOWN |
| **Load Balancing** | Round Robin |
| **Health Monitor** | System-TCP |
| **Persistence** | None |
| **Pool Path** | `IPC_VS_1602_DWZ_4762_TESTPUBLIC` |

---

## Configuration Summary

If configuration details are available, add a summary section with:
- Total virtual services
- Active vs. inactive services
- Most common protocol
- Health monitor types in use

**Important Rules:**
1. Use EXACT field names from the API response
2. Handle null values gracefully (show "N/A" or "None")
3. Format arrays as comma-separated strings
4. Use code blocks for technical paths
5. Include ALL virtual services
6. If no virtual services: show "ℹ️ No virtual services configured"

Return ONLY the formatted markdown. NO preamble or explanation."""

        try:
            response = await ai_service._call_chat_with_retries(
            prompt=prompt,
            max_tokens=5000,  
            temperature=0.1,   
            timeout=30)
            return response.strip()
    
        except Exception as e:
            logger.error(f"❌ LLM formatting failed: {e}")
        # Fallback to manual formatting
            return self._manual_format_virtual_services(lb_name, lbci, virtual_services, errors)
        
    def _manual_format_virtual_services(self,lb_name: str,lbci: str,virtual_services: List[Dict],errors: Dict) -> str:
        """Manual fallback formatter with production-quality output."""
    
        output = f"# Load Balancer: {lb_name}\n\n"
        output += f"**Circuit ID (LBCI):** `{lbci}`\n\n"
    
    # Check for errors
        vs_error = errors.get("virtual_services")
        details_error = errors.get("details")
    
        if vs_error and details_error:
            output += "⚠️ **Error:** Unable to retrieve load balancer information\n\n"
            output += f"- Virtual Services Error: {vs_error}\n"
            output += f"- Details Error: {details_error}\n\n"
            return output
    
    # Check if empty
        if not virtual_services:
            output += "## Virtual Services (0 configured)\n\n"
            output += "ℹ️ No virtual services are currently configured for this load balancer.\n"
            return output
    
        output += f"## Virtual Services ({len(virtual_services)} configured)\n\n"
    
    # Format each virtual service
        for idx, vs in enumerate(virtual_services, 1):
            vs_name = vs.get("virtualServerName", "Unknown")
            vip = vs.get("vipIp", "N/A")
            port = vs.get("virtualServerport", "N/A")
            protocol = vs.get("protocol", "N/A")
            status = vs.get("status", "Unknown").upper()
            algorithm = vs.get("poolAlgorithm", "N/A")
            monitors = vs.get("monitor", [])
            pool_path = vs.get("virtualServerPath", "N/A")
            persistence_type = vs.get("persistenceType") or "None"
            persistence_value = vs.get("persistenceValue", "")
            pool_members = vs.get("poolMembers", [])
            cert_name = vs.get("certificateName")
        
        # Status emoji
            status_map = {
            "UP": "✅",
            "DOWN": "⚠️",
            "DEGRADED": "🟡",
            "UNKNOWN": "❓"}
            status_emoji = status_map.get(status, "❓")
        
        # Format monitors
            if isinstance(monitors, list) and monitors:
                monitor_str = ", ".join(monitors)
            elif monitors:
                monitor_str = str(monitors)
            else:
                monitor_str = "N/A"
        
        # Format persistence
            if persistence_type != "None" and persistence_value:
                persistence_str = f"{persistence_type} ({persistence_value})"
            else:
                persistence_str = persistence_type
        
        # Build virtual service entry
            output += f"### {idx}. {vs_name}\n\n"
            output += "| Property | Value |\n"
            output += "|----------|-------|\n"
            output += f"| **VIP Address** | {vip}:{port} |\n"
            output += f"| **Protocol** | {protocol} |\n"
            output += f"| **Status** | {status_emoji} {status} |\n"
            output += f"| **Load Balancing** | {algorithm} |\n"
            output += f"| **Health Monitor** | {monitor_str} |\n"
            output += f"| **Persistence** | {persistence_str} |\n"
        
            if pool_members:
                member_count = len(pool_members) if isinstance(pool_members, list) else "Available"
                output += f"| **Pool Members** | {member_count} |\n"
        
            if cert_name:
                output += f"| **SSL Certificate** | {cert_name} |\n"
        
            output += f"| **Pool Path** | `{pool_path}` |\n\n"
    
    # Add summary
        output += "---\n\n## Configuration Summary\n\n"
    
        active_count = sum(1 for vs in virtual_services if vs.get("status", "").upper() == "UP")
        down_count = sum(1 for vs in virtual_services if vs.get("status", "").upper() == "DOWN")
    
        protocols = {}
        for vs in virtual_services:
            proto = vs.get("protocol", "Unknown")
            protocols[proto] = protocols.get(proto, 0) + 1
    
        output += f"- **Total Services:** {len(virtual_services)}\n"
        output += f"- **Active (UP):** {active_count}\n"
        output += f"- **Down:** {down_count}\n"
        output += f"- **Protocols:** {', '.join(f'{k}({v})' for k, v in protocols.items())}\n"
    
        return output

        
    def _manual_format_lb_details(self,lb: Dict,virtual_services: List,lbci: str,lb_name: str) -> str:
        """Manual fallback formatting if LLM fails."""
        status_emoji = "✅" if lb.get("status", "").upper() == "ACTIVE" else "⚠️"
        output = f"⚖️ **{lb_name}** {status_emoji}\n\n"
        output += f"**LBCI:** `{lbci}`\n"
        output += f"**Status:** {lb.get('status', 'Unknown')}\n"
        output += f"**Location:** {lb.get('location', 'N/A')}\n\n"
        if virtual_services:
            output += f"### Virtual Services ({len(virtual_services)})\n\n"
            for vs in virtual_services:
                vs_name = vs.get("virtualServerName", "Unknown")
                vip = vs.get("vipIp", "N/A")
                port = vs.get("virtualServerport", "N/A")
                protocol = vs.get("protocol", "N/A")
                status = vs.get("status", "Unknown")
                algo = vs.get("poolAlgorithm", "N/A")
                monitors = vs.get("monitor", [])
                pool_path = vs.get("virtualServerPath", "N/A")
                status_emoji = "✅" if status.upper() == "UP" else "⚠️"
                output += f"🌐 **{vs_name}**\n"
                output += f"- VIP: {vip}:{port}\n"
                output += f"- Protocol: {protocol}\n"
                output += f"- Status: {status_emoji} {status}\n"
                output += f"- Load Balancing: {algo}\n"
                if monitors:
                    monitor_str = ", ".join(monitors) if isinstance(monitors, list) else monitors
                    output += f"- Health Monitor: {monitor_str}\n"
                if pool_path and pool_path != "N/A":
                    output += f"- Pool Path: `{pool_path}`\n"
                output += "\n"
        else:
            output += "### Virtual Services\n\nℹ️ No virtual services configured\n"
        return output

    def _analyze_query_intent(self, user_query: str) -> Dict[str, Any]:
        """
        ENHANCED query analysis - detects LB names from production patterns.
    """
        query_lower = user_query.lower().strip()
        original_query = user_query.strip()
    
        result = {
        "is_specific_lb": False,
        "lb_identifier": None,
        "is_lbci_query": False,
        "wants_details": True, 
        "wants_virtual_services": True, 
        "location_filter": None,
        "status_filter": None,
        "feature_filters": []}
    # ═══════════════════════════════════════════════════════════
    # PRIORITY 1: LBCI Detection (5-6 digit numbers)
    # ═══════════════════════════════════════════════════════════
        lbci_patterns = [
        r'\blbci[:\s=]+(\d{5,6})\b',                            
        r'\blb[:\s=]+(\d{5,6})\b',                              
        r'(?:load\s*balancer|lb)\s+(\d{5,6})\b',                 
        r'^\s*(\d{5,6})\s*$',                                   
        r'\b(\d{5,6})\b(?=\s*(?:details?|info|virtual\s*service))', 
        r'\b(\d{5,6})\b'                                       ]
    
        for pattern in lbci_patterns:
            match = re.search(pattern, query_lower)
            if match:
                lbci = match.group(1)
                result.update({
                "is_specific_lb": True,
                "is_lbci_query": True,
                "lb_identifier": lbci,
                "wants_details": True,       
                "wants_virtual_services": True 
                })
                logger.info(f"🎯 DETECTED LBCI: {lbci}")
                return result
    
    # ═══════════════════════════════════════════════════════════
    # PRIORITY 2: Check for GENERAL query keywords (before specific LB detection)
    # ═══════════════════════════════════════════════════════════
        general_keywords = [
        "all load balancers", "all lbs",
        "list load balancers", "list lbs",
        "show load balancers", "show lbs",
        "show me load balancers", "show me lbs",
        "how many load balancers", "how many lbs",
        "count load balancers", "count lbs"]
        is_general_query = any(keyword in query_lower for keyword in general_keywords)
        if is_general_query:
            logger.info(f"🌍 Detected GENERAL query: {user_query}")
    
    # ═══════════════════════════════════════════════════════════
    # PRIORITY 3: Production LB Name Patterns (only if not general query)
    # ═══════════════════════════════════════════════════════════
        if not is_general_query:
        # Pattern A: EG_* 
            match = re.search(r'\b(EG_[A-Za-z0-9_]+)\b', original_query)
            if match:
                lb_name = match.group(1)
                result.update({
                "is_specific_lb": True,
                "lb_identifier": lb_name,
                "wants_details": True,
                "wants_virtual_services": True
            })
                logger.info(f"🎯 DETECTED LB NAME (EG_): {lb_name}")
                return result
        
        # Pattern B: LB_* 
            match = re.search(r'\b(LB_[A-Za-z0-9_]+)\b', original_query)
            if match:
                lb_name = match.group(1)
                result.update({
                "is_specific_lb": True,
                "lb_identifier": lb_name,
                "wants_details": True,
                "wants_virtual_services": True
            })
                logger.info(f"🎯 DETECTED LB NAME (LB_): {lb_name}")
                return result
        # Pattern C: *_LB_*
            match = re.search(r'\b([A-Za-z0-9]+_LB_[A-Za-z0-9_]+)\b', original_query)
            if match:
                lb_name = match.group(1)
            # Validate it's not a false positive
                if len(lb_name.split("_")) >= 3:
                    result.update({
                    "is_specific_lb": True,
                    "lb_identifier": lb_name,
                    "wants_details": True,
                    "wants_virtual_services": True
                    })
                    logger.info(f"🎯 DETECTED LB NAME (_LB_): {lb_name}")
                    return result
        # Pattern D: "details/show/info/get about X" where X has underscores
            match = re.search(
            r'(?:details?|show|info(?:rmation)?|get|describe|list)\s+(?:about|for|on|of|the)?\s*([A-Z][A-Za-z0-9_\-]+)',
            original_query)
            if match:
                potential_name = match.group(1)
            # Must have underscore to be LB name (avoid matching single words)
                if '_' in potential_name and len(potential_name.split("_")) >= 2:
                    result.update({
                    "is_specific_lb": True,
                    "lb_identifier": potential_name,
                    "wants_details": True,
                    "wants_virtual_services": True})
                    logger.info(f"🎯 DETECTED LB NAME (details pattern): {potential_name}")
                    return result
        # Pattern E: Standalone LB name at start of query (e.g., "EG_Tata_Com_167")
            match = re.search(r'^\s*([A-Z][A-Za-z0-9_\-]+_LB_[A-Z0-9_]+)', original_query)
            if match:
                lb_name = match.group(1)
                result.update({
                "is_specific_lb": True,
                "lb_identifier": lb_name,
                "wants_details": True,
                "wants_virtual_services": True
            })
                logger.info(f"🎯 DETECTED LB NAME (standalone): {lb_name}")
                return result
        # Pattern F: "load balancer named/called X"
            match = re.search(
            r'(?:load balancer|lb)s?\s+(?:named|called)\s+["\']?([a-zA-Z0-9_\-]+)',
            query_lower)
            if match:
                lb_name = match.group(1)
            # Find in original query to preserve case
                original_match = re.search(
                r'(?:load balancer|lb)s?\s+(?:named|called)\s+["\']?([a-zA-Z0-9_\-]+)',original_query,
                re.IGNORECASE)
                if original_match:
                    lb_name = original_match.group(1)
                    if '_' in lb_name:
                        result.update({
                        "is_specific_lb": True,
                        "lb_identifier": lb_name,
                        "wants_details": True,
                        "wants_virtual_services": True
                    })
                        logger.info(f"🎯 DETECTED LB NAME (named/called pattern): {lb_name}")
                        return result
    # ═══════════════════════════════════════════════════════════
    # PRIORITY 4: Detail & Virtual Services Intent Detection
    # ═══════════════════════════════════════════════════════════
        vs_keywords = [
        "virtual service", "virtual services", "vip", "vips",
        "listener", "listeners", "frontend", "front end"
    ]
        result["wants_virtual_services"] = any(kw in query_lower for kw in vs_keywords)
    # Detail intent detection (context-aware)
        strong_detail_keywords = [
        "details", "detail", "configuration", "config", "information about",
        "tell me about", "show me about", "describe", "explain",
        "full details", "complete details", "more information",
        "complete configuration"]
        weak_detail_keywords = [
        "about", "more", "get", "show"]
        has_strong_signal = any(kw in query_lower for kw in strong_detail_keywords)
        has_weak_signal = any(kw in query_lower for kw in weak_detail_keywords)
    # Set wants_details based on context
        if result["is_specific_lb"]:
        # For specific LB queries, default to True if any signal exists
            result["wants_details"] = has_strong_signal or has_weak_signal
        else:
        # For general queries, only set True for strong signals
            result["wants_details"] = has_strong_signal
    # ═══════════════════════════════════════════════════════════
    # PRIORITY 5: Location Filter Extraction
    # ═══════════════════════════════════════════════════════════
        location_map = {
        "mumbai": "Mumbai", "mum": "Mumbai", "bom": "Mumbai", "bombay": "Mumbai", "bkc": "Mumbai",
        "delhi": "Delhi", "del": "Delhi", "ncr": "Delhi", "new delhi": "Delhi",
        "chennai": "Chennai", "che": "Chennai", "maa": "Chennai", "madras": "Chennai", "amb": "Chennai",
        "bangalore": "Bengaluru", "bengaluru": "Bengaluru", "blr": "Bengaluru",
        "hyderabad": "Hyderabad", "hyd": "Hyderabad",
        "pune": "Pune", "pun": "Pune",
        "kolkata": "Kolkata", "kol": "Kolkata", "calcutta": "Kolkata",
        "ahmedabad": "Ahmedabad", "amd": "Ahmedabad"}
    # Pattern-based location extraction
        location_patterns = [
        r'\b(?:in|at|from|for)\s+([a-z]+(?:\s*-?\s*[a-z]+)?)\b',  
        r'\b(?:datacenter|dc|location)\s+([a-z]+(?:\s*-?\s*[a-z]+)?)\b',  ]
    # Stopwords to filter out
        stopwords = {
        "the", "this", "that", "these", "those", "with", "from",
        "what", "where", "when", "which", "who", "how", "all",
        "details", "list", "show", "get"}
        for pattern in location_patterns:
            match = re.search(pattern, query_lower)
            if match:
                location_raw = match.group(1).strip()
                location_clean = location_raw.replace(" ", "").replace("-", "")
            # Skip stopwords
                if location_raw not in stopwords:
                # Try to match in location map
                    if location_clean in location_map:
                        result["location_filter"] = location_map[location_clean]
                        logger.info(f"📍 DETECTED LOCATION (pattern): {result['location_filter']}")
                        break
                    elif location_raw in location_map:
                        result["location_filter"] = location_map[location_raw]
                        logger.info(f"📍 DETECTED LOCATION (pattern): {result['location_filter']}")
                        break
    # Fallback: Direct city name matching
        if not result["location_filter"]:
            for keyword, location in location_map.items():
                if keyword in query_lower:
                    result["location_filter"] = location
                    logger.info(f"📍 DETECTED LOCATION (direct): {location}")
                    break
    # ═══════════════════════════════════════════════════════════
    # PRIORITY 6: Status Filter Detection
    # ═══════════════════════════════════════════════════════════
        if "active" in query_lower and "inactive" not in query_lower:
            result["status_filter"] = "active"
            logger.info(f"🔍 STATUS FILTER: active")
        elif "inactive" in query_lower:
            result["status_filter"] = "inactive"
            logger.info(f"🔍 STATUS FILTER: inactive")
        elif "down" in query_lower:
            result["status_filter"] = "down"
            logger.info(f"🔍 STATUS FILTER: down")
        elif "degraded" in query_lower:
            result["status_filter"] = "degraded"
            logger.info(f"🔍 STATUS FILTER: degraded")
        elif "healthy" in query_lower:
            result["status_filter"] = "healthy"
            logger.info(f"🔍 STATUS FILTER: healthy")
        elif "unhealthy" in query_lower:
            result["status_filter"] = "unhealthy"
            logger.info(f"🔍 STATUS FILTER: unhealthy")
    # ═══════════════════════════════════════════════════════════
    # PRIORITY 7: Feature Filters Detection
    # ═══════════════════════════════════════════════════════════
        if any(kw in query_lower for kw in ["ssl", "secure", "encrypted"]):
            result["feature_filters"].append("ssl")
            logger.info(f"🔍 FEATURE FILTER: ssl")
        if "https" in query_lower:
            if "ssl" not in result["feature_filters"]:
                result["feature_filters"].append("https")
            logger.info(f"🔍 FEATURE FILTER: https")
        elif "http" in query_lower and "https" not in query_lower:
            result["feature_filters"].append("http")
            logger.info(f"🔍 FEATURE FILTER: http")
        if "tcp" in query_lower:
            result["feature_filters"].append("tcp")
            logger.info(f"🔍 FEATURE FILTER: tcp")
        if "udp" in query_lower:
            result["feature_filters"].append("udp")
            logger.info(f"🔍 FEATURE FILTER: udp")
    # ═══════════════════════════════════════════════════════════
    # Final logging
    # ═══════════════════════════════════════════════════════════
        logger.debug(f"🔍 Query analysis complete: {result}")
        return result
    def _find_matching_lb(self, lb_identifier: str, all_lbs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Find load balancer matching the identifier.
        ENHANCED fuzzy matching for production names.
        """
        identifier_lower = lb_identifier.lower()
        identifier_clean = identifier_lower.replace("_", "").replace("-", "")
        # Try exact match first (case-insensitive)
        for lb in all_lbs:
            lb_name = (lb.get("name") or "").lower()
            if lb_name == identifier_lower:
                logger.info(f"✅ EXACT MATCH: {lb.get('name')}")
                return lb
        # Try exact LBCI match
        for lb in all_lbs:
            lbci = str(lb.get("lbci") or lb.get("circuitId") or "").lower()
            if lbci == identifier_lower:
                logger.info(f"✅ EXACT LBCI MATCH: {lbci}")
                return lb
        # Try partial match (removing underscores/hyphens)
        for lb in all_lbs:
            lb_name = (lb.get("name") or "").lower()
            lb_name_clean = lb_name.replace("_", "").replace("-", "")
            
            if identifier_clean in lb_name_clean or lb_name_clean in identifier_clean:
                logger.info(f"✅ PARTIAL MATCH: {lb.get('name')}")
                return lb
        # Try fuzzy matching (words in any order)
        identifier_words = set(re.findall(r'\w+', identifier_lower))
        best_match = None
        best_score = 0
        for lb in all_lbs:
            lb_name = (lb.get("name") or "").lower()
            lb_words = set(re.findall(r'\w+', lb_name))
            common_words = identifier_words & lb_words
            score = len(common_words) / max(len(identifier_words), len(lb_words))
            if score > best_score and score > 0.5: 
                best_score = score
                best_match = lb
        if best_match:
            logger.info(f"✅ FUZZY MATCH ({best_score:.0%}): {best_match.get('name')}")
            return best_match
        logger.warning(f"❌ NO MATCH for: {lb_identifier}")
        return None
    
    async def _get_ipc_engagement_id(self, user_id: str, user_roles=None, force_refresh: bool = False, auth_token: str = None, selected_engagement_id: int = None) -> Optional[int]:
        """Get IPC engagement ID (helper method)."""

        # 🛡️ HARDEN against OpenWebUI / Gateway garbage
        if not user_roles or not isinstance(user_roles, (list, tuple, set)):
            user_roles = []
        
        # Use selected engagement ID if available
        engagement_id = selected_engagement_id
        if not engagement_id:
            # Fetch engagement based on roles
            engagement_id = await self.get_engagement_id(user_roles=user_roles, auth_token=auth_token, user_id=user_id, selected_engagement_id=selected_engagement_id)
        
        if not engagement_id:
            return None
        
        # Fetch IPC engagement
        ipc_engagement_id = await api_executor_service.get_ipc_engagement_id(
            engagement_id=engagement_id,
            user_id=user_id,
            auth_token=auth_token,
            force_refresh=force_refresh
        )
        return ipc_engagement_id

    async def _get_load_balancer_details(self,params: Dict[str, Any],context: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed configuration for a specific load balancer."""
        try:
            lbci = params.get("lbci")
            user_id = context.get("user_id")
            auth_token = context.get("auth_token")
            user_query = context.get("user_query", "")
            
            if not lbci:
                lb_data = params.get("load_balancer")
                if lb_data:
                    lbci = lb_data.get("lbci") or lb_data.get("circuitId")
                
                if not lbci:
                    return {
                        "success": False,
                        "error": "LBCI required",
                        "response": "Please specify which load balancer (need LBCI)."}
            logger.info(f"🔍 Fetching details for: {lbci}") 
            result = await api_executor_service.get_load_balancer_details(
                lbci=lbci,
                user_id=user_id,
                auth_token=auth_token)
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
                    "resource_type": "load_balancer_details"})
            return {
                "success": True,
                "data": details,
                "response": formatted_response,
                "metadata": {
                    "lbci": lbci,
                    "query_type": "detailed"}}
        except Exception as e:
            logger.error(f"❌ Error getting details: {str(e)}", exc_info=True)
            raise

    async def _get_virtual_services(self,params: Dict[str, Any],context: Dict[str, Any]) -> Dict[str, Any]:
        """Get virtual services for a load balancer."""
        try:
            lbci = params.get("lbci")
            user_id = context.get("user_id")
            auth_token = context.get("auth_token")
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
                user_id=user_id,
                auth_token=auth_token)
            if not result.get("success"):
                return {
                    "success": False,
                    "error": result.get("error"),
                    "response": f"Failed to get virtual services: {result.get('error')}"
                }
            virtual_services = result.get("data", [])
            total = len(virtual_services)
            formatted_response = await self.format_response_with_llm(
                raw_data=virtual_services,
                user_query=user_query,
                context={
                    "lbci": lbci,
                    "total": total,
                    "query_type": "virtual_services"})
            if total == 0:
                formatted_response = (
                    f"🌐 **No Virtual Services Configured**\n\n"
                    f"Load balancer '{lbci}' has no virtual services configured.")
            return {
                "success": True,
                "data": virtual_services,
                "response": formatted_response,
                "metadata": {
                    "lbci": lbci,
                    "total": total,
                    "query_type": "virtual_services"}}
        except Exception as e:
            logger.error(f"❌ Error getting virtual services: {str(e)}", exc_info=True)
            raise

    async def _enrich_load_balancers_with_location(self,load_balancers: List[Dict[str, Any]],user_id: str,ipc_engagement_id: int) -> List[Dict[str, Any]]:

        logger.info(f"🌍 Enriching {len(load_balancers)} LBs with location data")
    
    # Fetch business units API (has endpoint data)
        bu_result = await api_executor_service.get_business_units_list(
        ipc_engagement_id=ipc_engagement_id,
        user_id=user_id,
        force_refresh=False  # Use cache
        )
    
    # Build endpoint_id -> location mapping
        endpoint_map = {}
        if bu_result.get("success") and bu_result.get("departments"):
            for dept in bu_result["departments"]:
                endpoint = dept.get("endpoint", {})
                if endpoint:
                    endpoint_id = endpoint.get("id")
                    location = endpoint.get("location", "Unknown")
                    endpoint_name = endpoint.get("name", "Unknown")
                
                    if endpoint_id:
                        endpoint_map[endpoint_id] = {
                        "location": location,
                        "name": endpoint_name
                    }
    
        logger.info(f"📍 Built endpoint map: {len(endpoint_map)} locations")
    
    # Enrich each LB
        enriched = []
        for lb in load_balancers:
            lb_name = lb.get("name", "")
            endpoint_id = lb.get("endpointId")
        
        # Strategy 1: Look up endpoint_id in map
            if endpoint_id and endpoint_id in endpoint_map:
                location = endpoint_map[endpoint_id]["location"]
                endpoint_name = endpoint_map[endpoint_id]["name"]
            else:
            # Strategy 2: Extract from LB name
                location = self._extract_location_from_name(lb_name) or "Unknown"
                endpoint_name = location
        
        # Add enriched fields
            enriched_lb = {**lb}
            enriched_lb["_location"] = location
            enriched_lb["_endpoint_name"] = endpoint_name
            enriched_lb["_location_normalized"] = location.lower().replace("-", "").replace(" ", "")
        
            enriched.append(enriched_lb)
    
        logger.info(f"✅ Enriched {len(enriched)} LBs with location data")
        return enriched

    def _extract_location_from_name(self, lb_name: str) -> Optional[str]:
        
        location_patterns = {
        "Mumbai": ["mumbai", "bkc", "mum"],
        "Delhi": ["delhi", "del", "ncr"],
        "Chennai": ["chennai", "che", "amb"],
        "Bengaluru": ["bengaluru", "bangalore", "blr"],
        "Hyderabad": ["hyderabad", "hyd"],
        "Pune": ["pune"]
    }
    
        name_lower = lb_name.lower()
        for location, patterns in location_patterns.items():
            if any(p in name_lower for p in patterns):
                return location
    
        return None
    
    async def _get_details_and_format(
        self,
        load_balancer: Dict[str, Any],
        user_query: str,
        query_intent: Dict[str, Any],
        ipc_engagement_id: int,
        user_id: str,
        auth_token: str = None
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
                user_id=user_id,
                auth_token=auth_token
            )
            if details_result.get("success"):
                details = details_result.get("data")
        
        # Fetch virtual services if requested
        virtual_services = None
        if query_intent.get("wants_virtual_services"):
            logger.info(f"🌐 Fetching virtual services for {lbci}")
            vs_result = await api_executor_service.get_load_balancer_virtual_services(
                lbci=lbci,
                user_id=user_id,
                auth_token=auth_token
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
