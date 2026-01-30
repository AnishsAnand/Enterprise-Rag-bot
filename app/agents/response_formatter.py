"""
Response Formatter - Formats agent responses for better UI display.
Converts raw JSON/data structures into user-friendly, readable text.
"""

import json
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class ResponseFormatter:
    """Formats agent responses for better readability in chat UIs."""

    @staticmethod
    def format_cluster_list(data: Dict[str, Any]) -> str:
        """
        Format cluster listing response into readable text.
        Args:
            data: Cluster data from API  
        Returns:
            Formatted string for display
        """
        try:
            if not data.get("success"):
                return f"❌ Failed to retrieve clusters: {data.get('error', 'Unknown error')}"
            clusters = data.get("clusters", [])
            if not clusters:
                return "📋 No clusters found."
            total = data.get("total_clusters", len(clusters))
            endpoints = data.get("endpoints", 0)
            response = f"✅ Found **{total} clusters** across **{endpoints} data centers**\n\n"
            # Group by endpoint
            by_endpoint = {}
            for cluster in clusters:
                endpoint = cluster.get("displayNameEndpoint", "Unknown")
                if endpoint not in by_endpoint:
                    by_endpoint[endpoint] = []
                by_endpoint[endpoint].append(cluster)
            # Display clusters by endpoint
            for endpoint, endpoint_clusters in by_endpoint.items():
                response += f"### 📍 {endpoint}\n\n"
                for cluster in endpoint_clusters[:5]:  
                    name = cluster.get("clusterName", "Unknown")
                    status = cluster.get("status", "Unknown")
                    nodes = cluster.get("nodescount", "?")
                    version = cluster.get("kubernetesVersion", "Unknown")
                    status_emoji = "✅" if status.lower() == "healthy" else "⚠️"
                    response += f"{status_emoji} **{name}**\n"
                    response += f"   - Status: {status}\n"
                    response += f"   - Nodes: {nodes}\n"
                    response += f"   - K8s Version: {version}\n\n"
                if len(endpoint_clusters) > 5:
                    response += f"   ... and {len(endpoint_clusters) - 5} more clusters\n\n"
            if total > len(clusters):
                response += f"\n_Showing {len(clusters)} of {total} total clusters_"
            return response
        except Exception as e:
            logger.error(f"Error formatting cluster list: {e}")
            # Fallback to basic format
            return f"✅ Found clusters (raw data follows):\n```json\n{json.dumps(data, indent=2)}\n```"
    @staticmethod
    def format_endpoint_list(data: Dict[str, Any]) -> str:
        """
        Format endpoint (datacenter) listing.
        Args:
            data: Endpoint data 
        Returns:
            Formatted string
        """
        try:
            if not data.get("success"):
                return f"❌ Failed to retrieve datacenters: {data.get('error', 'Unknown error')}"
            endpoints = data.get("endpoints", [])
            if not endpoints:
                return "📋 No datacenters found."
            response = f"✅ Found **{len(endpoints)} data centers**:\n\n"
            for endpoint in endpoints:
                name = endpoint.get("displayName", endpoint.get("endpointName", "Unknown"))
                endpoint_id = endpoint.get("endpointId", "?")
                location = endpoint.get("location", "Unknown location")
                response += f"📍 **{name}** (ID: {endpoint_id})\n"
                response += f"   Location: {location}\n\n"
            return response
        except Exception as e:
            logger.error(f"Error formatting endpoint list: {e}")
            return str(data)
    
    @staticmethod
    def format_execution_result(operation: str, resource_type: str, result: Dict[str, Any]) -> str:
        """
        Format generic execution result.
        Args:
            operation: Operation performed (create, update, delete)
            resource_type: Type of resource
            result: Result data
        Returns:
            Formatted string
        """
        try:
            if result.get("success"):
                response = f"✅ Successfully **{operation}d** {resource_type}\n\n"
                # Add key details
                data = result.get("data", {})
                if isinstance(data, dict):
                    if "clusterId" in data:
                        response += f"**Cluster ID:** {data['clusterId']}\n"
                    if "clusterName" in data:
                        response += f"**Name:** {data['clusterName']}\n"
                    if "status" in data:
                        response += f"**Status:** {data['status']}\n"
                    # Add any message
                    if "message" in data:
                        response += f"\n{data['message']}\n"
                return response
            else:
                error = result.get("error", "Unknown error")
                return f"❌ Failed to {operation} {resource_type}: {error}"
        except Exception as e:
            logger.error(f"Error formatting execution result: {e}")
            return str(result)
    
    @staticmethod
    def format_firewall_list(data: Dict[str, Any]) -> str:
        """
        Format firewall listing response into user-friendly text.
        Args:
            data: Firewall data from API
        Returns:
            Formatted string for display
        """
        try:
            if not data.get("success"):
                return f"❌ Failed to retrieve firewalls: {data.get('error', 'Unknown error')}"
            firewalls = data.get("data", [])
            if not firewalls:
                return "🔥 No firewalls found."
            total = data.get("total", len(firewalls))
            metadata = data.get("metadata", {})
            location_filter = metadata.get("location_filter")
            # Build summary
            response = f"🔥 Found **{total} firewall(s)**"
            if location_filter:
                response += f" in **{location_filter}**"
            response += "\n\n"
            # Group by location/endpoint if available
            by_location = {}
            for fw in firewalls:
                # Extract location from endpoint info or name
                location = fw.get("_location") or fw.get("endpointName") or "Unknown Location"
                if location not in by_location:
                    by_location[location] = []
                by_location[location].append(fw)
            # Build table for each location
            for location, location_firewalls in sorted(by_location.items()):
                response += f"### 📍 {location}\n\n"
                # Create markdown table (matching UI: Name, IP, Type)
                response += "| Name | IP | Type |\n"
                response += "|------|-----|------|\n"
                for fw in location_firewalls[:20]:  # Max 20 per location
                    # Extract name - try multiple fields
                    name = ResponseFormatter._extract_firewall_name(fw)
                    # Extract IP - check multiple possible field names
                    ip = (
                        fw.get("ip") or 
                        fw.get("IP") or 
                        fw.get("managementIp") or 
                        fw.get("MANAGEMENTIP") or
                        "N/A")
                    # Handle edge cases like IP = 0 or "None"
                    if ip in [0, "0", "None", "null", None, ""]:
                        ip = "N/A"
                    # Extract type with user-friendly mapping
                    fw_type = ResponseFormatter._extract_firewall_type(fw)
                    response += f"| **{name}** | {ip} | {fw_type} |\n"
                if len(location_firewalls) > 20:
                    response += f"\n*... and {len(location_firewalls) - 20} more firewalls*\n"
                response += "\n"
            # Add helpful tip
            response += "💡 **Tip:** Ask about a specific firewall by name for more details.\n"
            return response
        except Exception as e:
            logger.error(f"Error formatting firewall list: {e}", exc_info=True)
            # Fallback - show count at minimum
            count = len(data.get("data", []))
            return f"🔥 Found **{count} firewall(s)** (formatting error: {str(e)})"
    
    @staticmethod
    def _extract_firewall_name(fw: Dict[str, Any]) -> str:
        """
        Extract user-friendly firewall name from raw API data.
        API returns:
        - displayName: 
        - department:
        - basicDetails: May contain name/label info
        """
        # Priority 1: displayName 
        name = (
            fw.get("displayName") or 
            fw.get("name") or 
            fw.get("technicalName") or
            fw.get("firewallName") or 
            fw.get("componentName"))
        # Priority 2: Check basicDetails for name
        basic_details = fw.get("basicDetails") or fw.get("BASICDETAILS") or {}
        if not name and isinstance(basic_details, dict):
            name = (
                basic_details.get("name") or 
                basic_details.get("displayName") or
                basic_details.get("firewallName") or
                basic_details.get("label"))
        # Priority 3: Get department/tenant name for context (shown in parentheses)
        department = fw.get("department") or fw.get("DEPARTMENT")
        dept_name = None
        if isinstance(department, list) and department:
            # Department is a list of dicts like [{"name": "qatest0801", "id": "6083"}]
            first_dept = department[0]
            if isinstance(first_dept, dict):
                dept_name = first_dept.get("name") or first_dept.get("departmentName")
        elif isinstance(department, dict):
            dept_name = department.get("name") or department.get("departmentName")
        elif isinstance(department, str):
            dept_name = department
        # Also check for tenant/engagement name fields
        if not dept_name:
            dept_name = fw.get("tenantName") or fw.get("engagementName")
        # Build display name matching UI format: "FirewallName (DepartmentName)"
        if name and dept_name and name != dept_name:
            return f"{name} ({dept_name})"
        elif name:
            return name
        elif dept_name:
            # If no name but have department, use department as primary
            return dept_name
        else:
            # Fallback to ID
            fw_id = fw.get("id") or fw.get("firewallId") or fw.get("circuitId") or fw.get("ENDCOMPID")
            return f"Firewall-{fw_id}" if fw_id else "Unknown"

    @staticmethod
    def _extract_firewall_type(fw: Dict[str, Any]) -> str:
        """
        Extract user-friendly firewall type.
        
        API fields (in priority order):
        - LOGO"
        - component
        - componentType: May have type info
        """
        # LOGO field is the primary source 
        logo = fw.get("LOGO") or fw.get("logo") or ""
        # component field is used by some API responses
        component = fw.get("component") or fw.get("COMPONENT") or ""
        # componentType as fallback
        component_type = fw.get("componentType") or fw.get("COMPONENTTYPE") or ""
        fw_type = fw.get("type") or fw.get("firewallType") or ""
        # Combine all type sources for checking
        logo_str = str(logo).upper() if logo else ""
        component_str = str(component).upper() if component else ""
        type_str = str(component_type or fw_type).upper()
        # Check all sources for type identification
        all_type_info = f"{logo_str} {component_str} {type_str}"
        # Map to user-friendly types (order matters - check specific first)
        if "FORTINET" in all_type_info:
            return "🟧 Fortinet"
        elif "IZO FW (F)" in all_type_info or "FW (F)" in all_type_info or "VAYU FIREWALL(F)" in all_type_info:
            return "🔵 Vayu Firewall(F)"
        elif "IZO FW (N)" in all_type_info or "FW (N)" in all_type_info or "VAYU FIREWALL(N)" in all_type_info:
            return "🟢 Vayu Firewall(N)"
        elif "IZO" in all_type_info or "VAYU" in all_type_info:
            return "🔷 Vayu Firewall"
        elif logo:
            # Return raw logo value if we couldn't map it
            return str(logo)
        elif component:
            return str(component)
        else:
            return "Unknown"

    @staticmethod
    def format_rag_response(data: Dict[str, Any]) -> str:
        """
        Format RAG agent response.
        
        Args:
            data: RAG response data
            
        Returns:
            Formatted string
        """
        try:
            answer = data.get("answer", "")
            sources = data.get("sources", [])
            confidence = data.get("confidence", 0.0)
            if not answer:
                return "I couldn't find relevant information in the documentation."
            response = answer
            # Sources section removed per user request
            # Sources are still available in the API response metadata if needed
            # Add confidence if low
            if confidence < 0.5:
                response += "\n\n_Note: I'm not very confident about this answer. You may want to verify this information._"
            return response 
        except Exception as e:
            logger.error(f"Error formatting RAG response: {e}")
            return str(data)
        
    @staticmethod
    def auto_format(response_text: str) -> str:
        """
        Automatically detect and format JSON responses.
        Args:
            response_text: Raw response text (might contain JSON)  
        Returns:
            Formatted text
        """
        try:
            # Try to parse as JSON
            data = json.loads(response_text)
            # Detect response type and format accordingly
            if isinstance(data, dict):
                # Check for cluster list
                if "clusters" in data or ("data" in data and isinstance(data.get("data"), dict) and "data" in data["data"]):
                    return ResponseFormatter.format_cluster_list(data)
                # Check for endpoints
                elif "endpoints" in data:
                    return ResponseFormatter.format_endpoint_list(data)
                # Check for RAG response
                elif "answer" in data:
                    return ResponseFormatter.format_rag_response(data)
                # Check for execution result
                elif "success" in data:
                    return f"{'✅' if data['success'] else '❌'} {data.get('message', 'Operation completed')}"
            # If not JSON or unknown format, return as-is
            return response_text
        except json.JSONDecodeError:
            # Not JSON, return as-is
            return response_text
        except Exception as e:
            logger.error(f"Error in auto_format: {e}")
            return response_text
        
    @staticmethod
    def format_load_balancer_list(data: Dict[str, Any]) -> str:
        """
        Formatter for load balancers.
        Handles:
        1. Detailed view (single LB with virtual services)
        2. Specific view (single LB basic info)
        3. General list (multiple LBs)
        PRODUCTION: Always user-friendly, never raw JSON
        """
        try:
            if not data.get("success"):
                error = data.get('error', 'Unknown error')
                return f"❌ Failed to retrieve load balancers: {error}"
        
            load_balancers = data.get("data", [])
            metadata = data.get("metadata", {})
            query_type = metadata.get("query_type", "general")
            # ═══════════════════════════════════════════════════════════
            # TYPE 1: DETAILED VIEW (with virtual services)
            # ═══════════════════════════════════════════════════════════
            if query_type == "specific_detailed" and isinstance(load_balancers, dict):
                return ResponseFormatter.format_load_balancer_detailed(load_balancers)
            # ═══════════════════════════════════════════════════════════
            # TYPE 2: SPECIFIC VIEW (single LB, basic info)
            # ═══════════════════════════════════════════════════════════
            if query_type == "specific" and len(load_balancers) == 1:
                lb = load_balancers[0]
                lb_name = lb.get("name", "Unknown")
                lbci = lb.get("lbci") or lb.get("circuitId") or "N/A"
                location = lb.get("_location", "Unknown")
                status = lb.get("status", "Unknown")
                vip = lb.get("virtual_ip") or lb.get("virtualIp") or "N/A"
                protocol = lb.get("protocol", "N/A")
                port = lb.get("port", "N/A")
                ssl_enabled = lb.get("ssl_enabled") or lb.get("sslEnabled", False)
                # Status emoji
                status_emoji = "✅" if status.lower() in ["active", "running", "healthy"] else "⚠️"
                ssl_emoji = " 🔒" if ssl_enabled else ""
                response = f"⚖️ **Load Balancer: {lb_name}**\n\n"
                response += f"{status_emoji} **Status:** {status}\n"
                response += f"📍 **Location:** {location}\n"
                if vip != "N/A":
                    vip_display = f"{vip}:{port}" if port != "N/A" else vip
                    response += f"🌐 **VIP:** {vip_display} ({protocol}{ssl_emoji})\n"
                response += f"🔑 **LBCI:** `{lbci}`\n"
                # Hint for more details
                if not metadata.get("has_details"):
                    response += f"\n💡 **Tip:** Ask for **'details for {lbci}'** or **'details for {lb_name}'** to see:\n"
                    response += f"- Full configuration\n"
                    response += f"- Virtual services (VIPs, listeners)\n"
                    response += f"- Health monitors\n"
                    response += f"- Pool members\n"
                return response
            # ═══════════════════════════════════════════════════════════
            # TYPE 3: GENERAL LIST (multiple LBs)
            # ═══════════════════════════════════════════════════════════
            if not load_balancers:
                return "⚖️ No load balancers found."
            total = metadata.get("count", len(load_balancers))
            original_count = metadata.get("original_count", total)
            filter_applied = metadata.get("filter_applied", False)
            filter_reason = metadata.get("filter_reason")
            response = f"✅ Found **{total} load balancer(s)**"
            if filter_applied and filter_reason:
                response += f" (filtered by: **{filter_reason}**)"
            if total != original_count:
                response += f" out of {original_count} total"
            response += "\n\n"
            # Group by location
            by_location = {}
            for lb in load_balancers:
                location = lb.get("_location", "Unknown")
                if location not in by_location:
                    by_location[location] = []
                by_location[location].append(lb)
            # Display by location
            for location, location_lbs in sorted(by_location.items()):
                response += f"### 📍 {location}\n\n"
                for lb in location_lbs[:10]:  
                    name = lb.get("name", lb.get("loadBalancerName", "Unknown"))
                    status = lb.get("status", "Unknown")
                    vip = lb.get("virtual_ip") or lb.get("virtualIp") or "N/A"
                    protocol = lb.get("protocol", "N/A")
                    port = lb.get("port", "N/A")
                    ssl_enabled = lb.get("ssl_enabled") or lb.get("sslEnabled", False)
                    lbci = lb.get("lbci") or lb.get("circuitId") or "N/A"
                    # Status emoji
                    status_lower = status.lower()
                    if status_lower in ["active", "running", "healthy"]:
                        status_emoji = "✅"
                    elif status_lower in ["degraded", "warning"]:
                        status_emoji = "⚠️"
                    else:
                        status_emoji = "❌"
                    ssl_emoji = " 🔒" if ssl_enabled else ""
                    response += f"{status_emoji} **{name}**{ssl_emoji}\n"
                    response += f"   - VIP: {vip} | Protocol: {protocol}:{port} | Status: {status} | LBCI: `{lbci}`\n\n"
                if len(location_lbs) > 10:
                    response += f"   ... and {len(location_lbs) - 10} more\n\n"
            # Helpful tip at the end
            response += f"\n💡 **Tip:** To see details for a specific load balancer, use:\n"
            response += f"- `details for <LBCI>` (e.g., 'details for 312798')\n"
            response += f"- `details for <name>` (e.g., 'details for {load_balancers[0].get('name', 'LB_Name')}')\n"
            return response
        except Exception as e:
            logger.error(f"❌ Error formatting LB list: {e}", exc_info=True)
            # FALLBACK
            count = len(data.get("data", []))
            return (
                f"⚖️ **Load Balancers**\n\n"
                f"Found {count} load balancer(s), but formatting failed.\n\n"
                f"⚠️ Unable to display details due to an internal error.\n"
                f"💡 Please try your query again.\n")

    @staticmethod
    def format_load_balancer_detailed(data: Dict[str, Any]) -> str:
        """
        Format detailed load balancer with virtual services.
        
        CRITICAL: This handles the output from get_details_and_format
        which includes both configuration AND virtual services.
        
        Args:
            data: Dict with structure:
                {
                    "load_balancer": {...},
                    "details": {...},  # optional
                    "virtual_services": [...],  # optional
                    "errors": {...}  # optional
                }
        
        Returns:
            User-friendly markdown string (NO raw JSON)
        """
        try:
            lb = data.get("load_balancer", {})
            details = data.get("details", {})
            virtual_services = data.get("virtual_services", [])
            errors = data.get("errors", {})
            
            # Extract basic info
            lb_name = lb.get("name", "Unknown")
            lbci = lb.get("lbci") or lb.get("circuitId") or lb.get("LBCI") or "N/A"
            status = lb.get("status", "Unknown")
            location = lb.get("_location", "Unknown")
            
            # VIP info (check multiple possible keys)
            vip = (lb.get("virtual_ip") or 
                   lb.get("virtualIp") or 
                   details.get("virtual_ip") or 
                   details.get("virtualIp") or 
                   "N/A")
            protocol = lb.get("protocol") or details.get("protocol") or "N/A"
            port = lb.get("port") or details.get("port") or "N/A"
            ssl_enabled = (lb.get("ssl_enabled") or 
                          lb.get("sslEnabled") or 
                          details.get("ssl_enabled") or 
                          details.get("sslEnabled") or 
                          False)
            # Status emoji (production-ready)
            status_lower = status.lower()
            if status_lower in ["active", "running", "healthy", "up"]:
                status_emoji = "✅"
            elif status_lower in ["degraded", "warning", "partial"]:
                status_emoji = "⚠️"
            elif status_lower in ["down", "inactive", "error", "failed"]:
                status_emoji = "❌"
            else:
                status_emoji = "❓"
            ssl_emoji = " 🔒" if ssl_enabled else ""
            # ═══════════════════════════════════════════════════════════
            # BUILD RESPONSE
            # ═══════════════════════════════════════════════════════════
            response = f"⚖️ **Load Balancer Details**\n\n"
            response += f"### {status_emoji} {lb_name}{ssl_emoji}\n\n"
            # Basic Information Section
            response += f"**Basic Information:**\n"
            response += f"- **LBCI:** `{lbci}`\n"
            response += f"- **Status:** {status}\n"
            response += f"- **Location:** {location}\n"
            if vip != "N/A":
                vip_display = f"{vip}:{port}" if port != "N/A" else vip
                response += f"- **Virtual IP:** {vip_display}"
                if protocol != "N/A":
                    response += f" ({protocol})"
                response += "\n"
            response += f"- **SSL/TLS:** {'Enabled ✓' if ssl_enabled else 'Disabled'}\n"
            # Additional details from configuration
            if details:
                algorithm = details.get("algorithm") or details.get("load_balancing_algorithm")
                if algorithm:
                    response += f"- **Algorithm:** {algorithm}\n"
                backend = details.get("backend_pool") or details.get("backendPool") or details.get("pool")
                if backend:
                    response += f"- **Backend Pool:** {backend}\n"
                timeout = details.get("timeout") or details.get("connection_timeout")
                if timeout:
                    response += f"- **Timeout:** {timeout}s\n"
            # ═══════════════════════════════════════════════════════════
            # VIRTUAL SERVICES SECTION (CRITICAL FOR PRODUCTION)
            # ═══════════════════════════════════════════════════════════
            response += f"\n### 🌐 Virtual Services"
            # Handle errors
            vs_error = errors.get("virtual_services")
            if vs_error:
                response += f" (⚠️ Error)\n\n"
                response += f"Failed to retrieve virtual services: `{vs_error}`\n"
                response += f"\n💡 **Tip:** The load balancer exists, but virtual service data is unavailable.\n"
            # No virtual services configured
            elif not virtual_services or len(virtual_services) == 0:
                response += f"\n\n"
                response += f"ℹ️ No virtual services configured for this load balancer.\n"
                response += f"\n💡 This could mean:\n"
                response += f"- Load balancer is newly created\n"
                response += f"- No listeners/VIPs have been configured yet\n"
            # Virtual services exist - format them nicely
            else:
                response += f" ({len(virtual_services)})\n\n"
                for idx, vs in enumerate(virtual_services, 1):
                    # Extract virtual service details
                    vs_name = vs.get("virtualServerName") or vs.get("name") or f"Virtual Service {idx}"
                    vip_ip = vs.get("vipIp") or vs.get("virtual_ip") or "N/A"
                    vs_port = vs.get("virtualServerport") or vs.get("port") or "N/A"
                    vs_protocol = vs.get("protocol", "N/A")
                    vs_status = vs.get("status", "Unknown")
                    algorithm = vs.get("poolAlgorithm") or vs.get("algorithm") or "N/A"
                    monitors = vs.get("monitor", [])
                    pool_path = vs.get("virtualServerPath") or vs.get("pool_path") or "N/A"
                    pool_members = vs.get("poolMembers")
                    persistence = vs.get("persistenceType")
                    # Virtual service status emoji
                    vs_status_upper = vs_status.upper()
                    if vs_status_upper == "UP":
                        vs_status_emoji = "✅"
                    elif vs_status_upper == "DOWN":
                        vs_status_emoji = "⚠️"
                    elif vs_status_upper in ["DEGRADED", "PARTIAL"]:
                        vs_status_emoji = "⚠️"
                    else:
                        vs_status_emoji = "❓"
                    # Format this virtual service
                    response += f"#### {vs_name}\n\n"
                    response += f"- **VIP:** {vip_ip}:{vs_port}\n"
                    response += f"- **Protocol:** {vs_protocol}\n"
                    response += f"- **Status:** {vs_status_emoji} {vs_status}\n"
                    response += f"- **Algorithm:** {algorithm}\n"
                    # Health monitors
                    if monitors:
                        if isinstance(monitors, list):
                            monitors_str = ", ".join(monitors)
                        else:
                            monitors_str = str(monitors)
                        response += f"- **Health Monitors:** {monitors_str}\n"
                    # Persistence
                    if persistence and persistence != "null":
                        response += f"- **Persistence:** {persistence}\n"
                    # Pool path (technical detail)
                    if pool_path != "N/A":
                        response += f"- **Pool Path:** `{pool_path}`\n"
                    # Pool members (if available)
                    if pool_members and pool_members != "null":
                        response += f"- **Pool Members:** {pool_members}\n"
                    response += "\n"
            # ═══════════════════════════════════════════════════════════
            # ERROR NOTES (if configuration details failed)
            # ═══════════════════════════════════════════════════════════
            details_error = errors.get("details")
            if details_error:
                response += f"\n⚠️ **Note:** Configuration details unavailable: `{details_error}`\n"
            # ═══════════════════════════════════════════════════════════
            # HELPFUL TIPS
            # ═══════════════════════════════════════════════════════════
            if vs_error or details_error:
                response += f"\n💡 **Troubleshooting:**\n"
                response += f"- Basic load balancer info is available\n"
                response += f"- Some API calls failed (see errors above)\n"
                response += f"- Try again later or contact support if issue persists\n"
            return response
        except Exception as e:
            logger.error(f"❌ Error formatting detailed LB: {e}", exc_info=True)
            # PRODUCTION FALLBACK: Never show raw JSON, show clean error
            lb_name = data.get("load_balancer", {}).get("name", "Unknown")
            lbci = data.get("load_balancer", {}).get("lbci", "N/A")
            return (
                f"⚖️ **Load Balancer: {lb_name}**\n\n"
                f"**LBCI:** `{lbci}`\n\n"
                f"⚠️ Unable to format detailed information due to an internal error.\n"
                f"The load balancer exists, but formatting failed.\n\n"
                f"💡 Please try your query again or contact support.\n")
    
    @staticmethod
    def auto_format(response_text: str) -> str:
        """
        Automatically detect and format JSON responses.
        Args:
            response_text: Raw response text (might contain JSON)
        Returns:
            Formatted text
        """
        try:
            # Try to parse as JSON
            data = json.loads(response_text)
            # Detect response type and format accordingly
            if isinstance(data, dict):
                # Check for cluster list
                if "clusters" in data or ("data" in data and isinstance(data.get("data"), dict) and "data" in data["data"]):
                    return ResponseFormatter.format_cluster_list(data)
                elif "metadata" in data and data.get("metadata", {}).get("resource_type") == "load_balancer":
                    return ResponseFormatter.format_load_balancer_list(data)
                # Check for firewall response
                elif "metadata" in data and data.get("metadata", {}).get("resource_type") == "firewall":
                    return ResponseFormatter.format_firewall_list(data)
                # Check for endpoints
                elif "endpoints" in data:
                    return ResponseFormatter.format_endpoint_list(data)
                # Check for RAG response
                elif "answer" in data:
                    return ResponseFormatter.format_rag_response(data)
                # Check for execution result
                elif "success" in data:
                    return f"{'✅' if data['success'] else '❌'} {data.get('message', 'Operation completed')}"
            # If not JSON or unknown format, return as-is
            return response_text
        except json.JSONDecodeError:
            # Not JSON, return as-is
            return response_text
        except Exception as e:
            logger.error(f"Error in auto_format: {e}")
            return response_text

# Create global instance
response_formatter = ResponseFormatter()
