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
    
    async def _list_load_balancers(self,params: Dict[str, Any],context: Dict[str, Any]) -> Dict[str, Any]:

        try:
        # Extract context
            user_roles = context.get("user_roles", [])
            user_id = context.get("user_id")
            user_query = context.get("user_query", "").lower()
            force_refresh = params.get("force_refresh", False)
        
            logger.info(f"📋 Processing LB query: '{user_query}'")
        
        # ═════════════════════════════════════════════════════════════════
        # STEP 1: Analyze query intent
        # ═════════════════════════════════════════════════════════════════
            query_analysis = self._analyze_query_intent(user_query)
        
        # ═════════════════════════════════════════════════════════════════
        # STEP 1.5: Handle DIRECT LBCI queries FIRST (CRITICAL FIX)
        # ═════════════════════════════════════════════════════════════════
            if query_analysis.get("is_lbci_query"):
                lbci = query_analysis.get("lb_identifier")
                logger.info(f"🎯 DIRECT LBCI QUERY DETECTED: {lbci}")
            
            # Route to dedicated LBCI handler
                return await self._handle_lbci_query(
                lbci=lbci,
                user_query=user_query,
                user_id=user_id
            )
        
        # ═════════════════════════════════════════════════════════════════
        # STEP 2: Handle SPECIFIC load balancer queries (by name)
        # ═════════════════════════════════════════════════════════════════
            if query_analysis["is_specific_lb"]:
                specific_lb_name = query_analysis["lb_identifier"]
                logger.info(f"🎯 SPECIFIC LB QUERY: '{specific_lb_name}'")

                raw_roles = context.get("user_roles")
                user_roles = list(raw_roles) if isinstance(raw_roles, (list, tuple, set)) else []
                ipc_engagement_id = await self._get_ipc_engagement_id(
                user_id=user_id,
                user_roles=user_roles,
                force_refresh=force_refresh
            )
            
                if not ipc_engagement_id:
                    return {
                    "success": False,
                    "error": "Failed to get IPC engagement ID",
                    "response": "Unable to retrieve engagement information."
                }

                list_result = await api_executor_service.list_load_balancers(
                ipc_engagement_id=ipc_engagement_id,
                user_id=user_id,
                force_refresh=False 
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
            
            # ═════════════════════════════════════════════════════════════
            # Check if user wants detailed information
            # ═════════════════════════════════════════════════════════════
                detail_keywords = [
                "more", "detail", "info", "about", "configuration", "config",
                "show me", "tell me", "give me", "get", "describe",
                "virtual service", "vip", "listener", "pool", "members",
                "complete", "full", "all information", "everything",
                "settings", "properties", "attributes"
            ]
                wants_details = (
                query_analysis.get("wants_details") or
                query_analysis.get("wants_virtual_services") or
                any(keyword in user_query for keyword in detail_keywords)
            )
            
                if lbci and wants_details:
                # Auto-fetch detailed configuration
                    logger.info(f"📊 Auto-fetching DETAILED info for {matched_lb.get('name')}")
                    return await self._fetch_complete_lb_details(
                    load_balancer=matched_lb,
                    user_query=user_query,
                    query_intent=query_analysis,
                    user_id=user_id,
                    lbci=lbci
                )
                else:
                    logger.info(f"📋 Returning BASIC info for {matched_lb.get('name')}")
                
                    formatted_response = await self.format_response_with_llm(
                    operation="list",
                    raw_data=[matched_lb],
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
                force_refresh=force_refresh
            )
                if not ipc_engagement_id:
                    return {
                    "success": False,
                    "error": "Failed to get IPC engagement ID",
                    "response": "Unable to retrieve engagement information."
                }
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
                original_count = len(load_balancers)
                is_cached = result.get("cached", False)
            
                logger.info(f"✅ Retrieved {original_count} load balancer(s) from API")
                enriched_lbs = await self._enrich_load_balancers_with_location(
                load_balancers,
                user_id,
                ipc_engagement_id
            )
            
            # ═════════════════════════════════════════════════════════════
            # STEP 4: Apply filters
            # ═════════════════════════════════════════════════════════════
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
            # ═════════════════════════════════════════════════════════════
            # STEP 5: Format response
            # ═════════════════════════════════════════════════════════════
                formatted_response = await self.format_response_with_llm(
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
            # ═════════════════════════════════════════════════════════════
            # STEP 6: Handle empty results with helpful messages
            # ═════════════════════════════════════════════════════════════
                if total_count == 0:
                    if original_count > 0:
                    # Filters removed all LBs - suggest alternatives
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
    async def _handle_lbci_query(self,lbci: str,user_query: str,user_id: str = None) -> Dict[str, Any]:
        logger.info(f"⚖️ LBCI QUERY HANDLER: {lbci}")
    # 1️⃣ Fetch virtual services (MANDATORY - this is what user wants!)
        logger.info(f"🌐 Fetching virtual services for LBCI {lbci}...")
        vs_result = await api_executor_service.get_load_balancer_virtual_services(
            lbci=lbci,
            user_id=user_id
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
        user_id=user_id)
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
        """Format detailed LB response with virtual services - PRODUCTION READY."""
        from app.services.ai_service import ai_service
        lb = raw_data.get("load_balancer", {})
        details = raw_data.get("details", {})
        virtual_services = raw_data.get("virtual_services", [])
        errors = raw_data.get("errors", {})
        lb_name = context.get("lb_name", "Unknown")
        lbci = context.get("lbci", "N/A")
        vs_count = context.get("vs_count", 0)
    # Build comprehensive prompt for LLM
        prompt = f"""Format this load balancer information in a production-ready, user-friendly way.

**User Query:** {user_query}

**Load Balancer:** {lb_name} (LBCI: {lbci})

**Basic Information:**
{json.dumps(lb, indent=2) if lb else "⚠️ Basic info not available"}

**Configuration Details:**
{json.dumps(details, indent=2) if details else "⚠️ Configuration details not available"}

**Virtual Services ({vs_count}):**
{json.dumps(virtual_services, indent=2) if virtual_services else "⚠️ No virtual services configured"}

**Errors (if any):**
- Details API: {errors.get('details') or '✅ Success'}
- Virtual Services API: {errors.get('virtual_services') or '✅ Success'}

**FORMATTING REQUIREMENTS:**

1. **Header** - Clear LB name with status emoji
   ⚖️ **EG_Tata_Com_167_LB_SEG_388** ✅

2. **Basic Configuration Section:**
   - LBCI (Load Balancer Circuit ID): `312798`
   - Location/Endpoint: Mumbai / Delhi / Chennai
   - Status: ✅ ACTIVE / ⚠️ DEGRADED / ❌ DOWN
   - Protocol & Port

3. **Virtual Services Section - CRITICAL:**
   For EACH virtual service, display:
```
   🌐 **Virtual Service: TESTPUBLIC**
   - VIP: 100.94.45.12:9056
   - Protocol: HTTP
   - Status: ⚠️ DOWN (use ✅ for UP, ⚠️ for DOWN/degraded)
   - Load Balancing: Round Robin
   - Health Monitor: System-TCP
   - Pool Path: IPC_VS_1602_DWZ_4762_TESTPUBLIC
```
   
   **IMPORTANT Virtual Service Fields:**
   - virtualServerName → Service name
   - vipIp + virtualServerport → VIP address
   - status → UP/DOWN (with emoji)
   - poolAlgorithm → Load balancing method
   - monitor → Health check configuration
   - virtualServerPath → Pool path
   - protocol → HTTP/HTTPS/TCP
   - persistenceType → Session persistence (if any)

4. **Handle Missing Data Gracefully:**
   - If configuration details failed: Show basic info only + mention error
   - If virtual services failed/empty: Say "No virtual services found" or show error
   - Use "N/A" for missing individual fields

5. **Visual Hierarchy:**
   - Use ### headers for sections
   - Use emojis sparingly: ✅ (up/success), ⚠️ (down/warning), ❌ (error), 🔒 (SSL), 🌐 (virtual service)
   - Use bullet points for properties
   - Use code blocks ` ` for technical IDs/paths

6. **Production Requirements:**
   - Be concise but complete
   - Highlight DOWN status or errors prominently
   - Easy to scan quickly
   - Include actionable information

**CRITICAL:** Return ONLY the formatted markdown response. NO preamble, NO "Here's the formatted output", NO meta-commentary. Start directly with the ⚖️ header."""

        try:
            response = await ai_service._call_chat_with_retries(
            prompt=prompt,
            max_tokens=2500,
            temperature=0.3,
            timeout=20
        )
            return response.strip()
        except Exception as e:
            logger.error(f"❌ LLM formatting failed: {e}")
        # Fallback: Manual formatting
            return self._manual_format_detailed_lb(lb, details, virtual_services, errors, lbci, lb_name)
        
    def _manual_format_detailed_lb(self,lb: Dict,details: Dict,virtual_services: List,errors: Dict,lbci: str,lb_name: str) -> str:
        output = f"⚖️ **{lb_name}**\n\n"
        output += f"**LBCI:** `{lbci}`\n"
        output += f"**Status:** {lb.get('status', 'Unknown')}\n\n"
        if virtual_services:
            output += f"### Virtual Services ({len(virtual_services)})\n\n"
            for vs in virtual_services:
                vs_name = vs.get("virtualServerName", "Unknown")
                vip = vs.get("vipIp", "N/A")
                port = vs.get("virtualServerport", "N/A")
                protocol = vs.get("protocol", "N/A")
                status = vs.get("status", "Unknown")
                algo = vs.get("poolAlgorithm", "N/A")
                monitor = vs.get("monitor", [])
                status_emoji = "✅" if status.upper() == "UP" else "⚠️"
                output += f"🌐 **{vs_name}**\n"
                output += f"- VIP: {vip}:{port}\n"
                output += f"- Protocol: {protocol}\n"
                output += f"- Status: {status_emoji} {status}\n"
                output += f"- Load Balancing: {algo}\n"
                if monitor:
                    monitors_str = ", ".join(monitor) if isinstance(monitor, list) else monitor
                    output += f"- Health Monitor: {monitors_str}\n"
                pool_path = vs.get("virtualServerPath")
                if pool_path:
                    output += f"- Pool Path: `{pool_path}`\n"
                output += "\n"
        else:
            output += f"\n### Virtual Services\n\n"
            if errors.get("virtual_services"):
                output += f"⚠️ Failed to retrieve: {errors.get('virtual_services')}\n"
            else:
                output += f"ℹ️ No virtual services configured\n"
        return output
    def _analyze_query_intent(self, user_query: str) -> Dict[str, Any]:
        """
        Query intent analysis with better LB name detection.
        Improvements:
        - More accurate detail intent detection
        - Handles both "list details" and specific LB queries
        """
        query_lower = user_query.lower().strip()
        original_query = user_query.strip()  
    
    # Initialize result
        result = {
        "is_specific_lb": False,
        "lb_identifier": None,
        "is_lbci_query": False,
        "wants_details": False,
        "wants_virtual_services": False,
        "location_filter": None,
        "status_filter": None,
        "feature_filters": []
        }

        lbci_patterns = [
    r'\blbci[:\s=]+(\d{5,6})\b',             
    r'\blb[:\s=]+(\d{5,6})\b',               
    r'(?:load\s*balancer|lb)\s+(\d{5,6})\b',  
    r'^\s*(\d{5,6})\s*$',                    
    r'\b(\d{5,6})\b(?=\s*(?:details|info|virtual\s*service))', 
]
        for pattern in lbci_patterns:
            match = re.search(pattern, query_lower)
            if match:
                lbci = match.group(1)
                result["is_specific_lb"] = True
                result["is_lbci_query"] = True
                result["lb_identifier"] = lbci  
                result["wants_details"] = True  
                result["wants_virtual_services"] = True  
                logger.info(f"🎯 LBCI QUERY DETECTED: {lbci}")
                return result
    # ═════════════════════════════════════════════════════════════════════
    # STEP 1: Virtual services intent (independent signal)
    # ═════════════════════════════════════════════════════════════════════
        vs_keywords = [
        "virtual service", "virtual services", "vip", "vips",
        "listener", "listeners", "frontend", "front end"
    ]
        result["wants_virtual_services"] = any(kw in query_lower for kw in vs_keywords)
    # ═════════════════════════════════════════════════════════════════════
    # STEP 2: GENERAL vs SPECIFIC query detection (GENERAL takes precedence)
    # ═════════════════════════════════════════════════════════════════════
        general_keywords = [
        "all load balancers", "all lbs",
        "list load balancers", "list lbs",
        "show load balancers", "show lbs",
        "show me load balancers",
        "how many load balancers", "how many lbs",
        "count load balancers", "count lbs"
        ]
        is_general_query = any(keyword in query_lower for keyword in general_keywords)
        if is_general_query:
            logger.info(f"🌍 Detected GENERAL query: {user_query}")
    # ═════════════════════════════════════════════════════════════════════
    # STEP 3: SPECIFIC load balancer detection (ENHANCED PATTERNS)
    # ═════════════════════════════════════════════════════════════════════
        if not is_general_query:
            specific_patterns = [
            # Pattern 1: Exact LB name patterns (case-sensitive, covers Tata LB naming)
            r'\b((?:EG_|LB_)?[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*_LB_[A-Z0-9_]+)\b',
            # Pattern 2: "load balancer named X" or "lb called X"
            r'(?:load balancer|lb)s?\s+(?:named|called)\s+["\']?([a-zA-Z0-9_\-]+)',
            # Pattern 3: "show/get/describe/list/details (for/of/about) X"
            r'(?:show|get|describe|list|details?)\s+(?:for|of|about|the)?\s*(?:load balancer|lb)?\s*["\']?([A-Z][A-Za-z0-9_\-]+)',
            # Pattern 4: "X details" or "X configuration"
            r'\b([A-Z][A-Za-z0-9_\-]+_LB_[A-Z0-9_]+)\s+(?:details?|info|config)',
            # Pattern 5: Standalone LB name at start of query
            r'^\s*([A-Z][A-Za-z0-9_\-]+_LB_[A-Z0-9_]+)',
            # Pattern 6: "details/info on/about X"
            r'(?:details?|info(?:rmation)?|configuration|config)\s+(?:on|about|for)\s+["\']?([a-zA-Z0-9_\-]+)'
            ]
        
            for pattern in specific_patterns:
                match = re.search(pattern, original_query, re.IGNORECASE if "(?:" in pattern else 0)
                if match:
                    extracted_name = match.group(1)
                    if "_LB_" in extracted_name or len(extracted_name.split("_")) >= 3:
                        result["is_specific_lb"] = True
                        result["lb_identifier"] = extracted_name
                        logger.info(f"🎯 Detected SPECIFIC LB: '{result['lb_identifier']}' (pattern match)")
                        break
    
    # ═════════════════════════════════════════════════════════════════════
    # STEP 4: Detail intent detection (ENHANCED - context-aware)
    # ═════════════════════════════════════════════════════════════════════
        strong_detail_keywords = [
        "details", "detail", "configuration", "config", "information about",
        "tell me about", "show me about", "describe", "explain",
        "full details", "complete details", "more information"
        ]
        weak_detail_keywords = [
        "about", "more", "get", "show"
        ]
        has_strong_signal = any(kw in query_lower for kw in strong_detail_keywords)
        has_weak_signal = any(kw in query_lower for kw in weak_detail_keywords)

        result["wants_details"] = has_strong_signal or (has_weak_signal and result["is_specific_lb"])
    
    # ═════════════════════════════════════════════════════════════════════
    # STEP 5: Location filter extraction (ENHANCED - handles lowercase)
    # ═════════════════════════════════════════════════════════════════════
        location_patterns = [
        # Pattern 1: "in/at/from Location" (case-insensitive)
        r'\b(?:in|at|from|for)\s+([a-zA-Z]+(?:\s*-?\s*[a-zA-Z]+)?)\b',
        # Pattern 2: "datacenter/dc/location Location"
        r'\b(?:datacenter|dc|location)\s+([a-zA-Z]+(?:\s*-?\s*[a-zA-Z]+)?)\b',
        # Pattern 3: Direct city names
        r'\b(mumbai|delhi|chennai|bengaluru|bangalore|hyderabad|pune|kolkata|ahmedabad)\b'
        ]
    
        for pattern in location_patterns:
            match = re.search(pattern, query_lower, re.IGNORECASE)
            if match:
                location_raw = match.group(1).strip()
            
            # Filter out common English stopwords
                stopwords = ["the", "this", "that", "these", "those", "with", "from", 
                        "what", "where", "when", "which", "who", "how"]
            
                if location_raw.lower() not in stopwords:
                    location_normalized = location_raw.lower()
                    if location_normalized in ["bangalore", "bengaluru", "blr"]:
                        result["location_filter"] = "Bengaluru"
                    elif location_normalized in ["mumbai", "bombay", "mum", "bom"]:
                        result["location_filter"] = "Mumbai"
                    elif location_normalized in ["delhi", "new delhi", "ncr", "del"]:
                        result["location_filter"] = "Delhi"
                    elif location_normalized in ["chennai", "madras", "che", "maa"]:
                        result["location_filter"] = "Chennai"
                    elif location_normalized in ["hyderabad", "hyd"]:
                        result["location_filter"] = "Hyderabad"
                    elif location_normalized in ["pune", "pun"]:
                        result["location_filter"] = "Pune"
                    else:
                        result["location_filter"] = location_raw.title()
                
                    logger.info(f"📍 Detected location filter: '{result['location_filter']}'")
                    break
    
    # ═════════════════════════════════════════════════════════════════════
    # STEP 6: Status filter detection
    # ═════════════════════════════════════════════════════════════════════
        if "active" in query_lower and "inactive" not in query_lower:
            result["status_filter"] = "active"
        elif "inactive" in query_lower:
            result["status_filter"] = "inactive"
        elif "degraded" in query_lower:
            result["status_filter"] = "degraded"
        elif "healthy" in query_lower:
            result["status_filter"] = "healthy"
        elif "unhealthy" in query_lower:
            result["status_filter"] = "unhealthy"
    
    # ═════════════════════════════════════════════════════════════════════
    # STEP 7: Feature filters detection (multi-value)
    # ═════════════════════════════════════════════════════════════════════
        if any(kw in query_lower for kw in ["ssl", "secure", "encrypted"]):
            result["feature_filters"].append("ssl")
        if "https" in query_lower:
            result["feature_filters"].append("https")
        elif "http" in query_lower and "https" not in query_lower:
            result["feature_filters"].append("http")
        if "tcp" in query_lower:
            result["feature_filters"].append("tcp")
        if "udp" in query_lower:
            result["feature_filters"].append("udp")
        logger.debug(f"🔍 Query analysis result: {result}")
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
        force_refresh=force_refresh)
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
                        "response": "Please specify which load balancer (need LBCI)."}
            logger.info(f"🔍 Fetching details for: {lbci}") 
            result = await api_executor_service.get_load_balancer_details(
                lbci=lbci,
                user_id=user_id)
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
                user_id=user_id)
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
