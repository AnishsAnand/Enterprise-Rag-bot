"""
Response Formatter - Formats agent responses for better UI display.
Converts raw JSON/data structures into user-friendly, readable text.
"""

import json
import logging
from typing import Any, Dict, List

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
                
                for cluster in endpoint_clusters[:5]:  # Show first 5 per endpoint
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
        ENHANCED formatter for load balancers with better detail display.
        
        Handles both:
        - General list view (multiple LBs)
        - Detailed single LB view (with configuration + virtual services)
        """
        try:
            if not data.get("success"):
                return f"❌ Failed to retrieve load balancers: {data.get('error', 'Unknown error')}"
        
            load_balancers = data.get("data", [])
            metadata = data.get("metadata", {})
            query_type = metadata.get("query_type", "general")
        
        # ═════════════════════════════════════════════════════════════════
        # DETAILED SINGLE LB VIEW
        # ═════════════════════════════════════════════════════════════════
            if query_type == "specific" and len(load_balancers) == 1:
                lb = load_balancers[0]
                lb_name = lb.get("name", "Unknown")
                lbci = lb.get("lbci", "N/A")
                location = lb.get("_location", "Unknown")
                status = lb.get("status", "Unknown")
                vip = lb.get("virtual_ip", lb.get("virtualIp", "N/A"))
                protocol = lb.get("protocol", "N/A")
                port = lb.get("port", "N/A")
                ssl_enabled = lb.get("ssl_enabled", lb.get("sslEnabled", False))
            
            # Status emoji
                status_emoji = "✅" if status.lower() in ["active", "running", "healthy"] else "⚠️"
                ssl_emoji = "🔒" if ssl_enabled else "🔓"
            
            # Build detailed response
                response = f"⚖️ **Load Balancer Details**\n\n"
                response += f"### {status_emoji} {lb_name} {ssl_emoji}\n\n"
                response += f"**Basic Information:**\n"
                response += f"- **LBCI:** `{lbci}`\n"
                response += f"- **Status:** {status}\n"
                response += f"- **Location:** {location}\n"
                response += f"- **Virtual IP:** {vip}\n"
                response += f"- **Protocol:** {protocol} (Port {port})\n"
                response += f"- **SSL/TLS:** {'Enabled' if ssl_enabled else 'Disabled'}\n\n"
            
            # Add algorithm if available
                algorithm = lb.get("algorithm")
                if algorithm:
                    response += f"- **Algorithm:** {algorithm}\n"
            
            # Add backend pool info if available
                backend = lb.get("backend_pool", lb.get("backendPool"))
                if backend:
                    response += f"\n**Backend Pool:** {backend}\n"
            
            # Show hint for more details
                if not metadata.get("has_details"):
                    response += f"\n💡 **Tip:** Ask for 'details for {lb_name}' to see full configuration and virtual services.\n"
            
                return response
        
        # ═════════════════════════════════════════════════════════════════
        # GENERAL LIST VIEW (multiple LBs)
        # ═════════════════════════════════════════════════════════════════
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
        
        # Display load balancers by location
            for location, location_lbs in sorted(by_location.items()):
                response += f"### 📍 {location}\n\n"
            
                for lb in location_lbs[:10]:  # Show first 10 per location
                    name = lb.get("name", lb.get("loadBalancerName", "Unknown"))
                    status = lb.get("status", "Unknown")
                    vip = lb.get("virtual_ip", lb.get("virtualIp", "N/A"))
                    protocol = lb.get("protocol", "N/A")
                    port = lb.get("port", "N/A")
                    ssl_enabled = lb.get("ssl_enabled", lb.get("sslEnabled", False))
                
                # Status emoji
                    if status.lower() in ["active", "running", "healthy"]:
                        status_emoji = "✅"
                    elif status.lower() in ["degraded", "warning"]:
                        status_emoji = "⚠️"
                    else:
                        status_emoji = "❌"
                
                # SSL emoji
                    ssl_emoji = " 🔒" if ssl_enabled else ""
                
                    response += f"{status_emoji} **{name}**{ssl_emoji}\n"
                    response += f"   - VIP: {vip} | Protocol: {protocol}:{port} | Status: {status}\n\n"
            
                if len(location_lbs) > 10:
                    response += f"   ... and {len(location_lbs) - 10} more\n\n"
        
            return response
        
        except Exception as e:
            logger.error(f"Error formatting load balancer list: {e}", exc_info=True)
        # Fallback to JSON
            return f"✅ Found load balancers (raw data):\n```json\n{json.dumps(data, indent=2)[:1000]}\n```"
        
 

    @staticmethod
    def format_load_balancer_detailed(data: Dict[str, Any]) -> str:
        """
        Format detailed load balancer response with virtual services.
        
        PRODUCTION-READY: Creates user-friendly output, NOT raw JSON.
        
        Args:
            data: Dict containing:
                - load_balancer: Basic LB info
                - details: Configuration details (optional)
                - virtual_services: List of virtual services (optional)
                - errors: Any errors encountered
        
        Returns:
            User-friendly formatted string
        """
        try:
            lb = data.get("load_balancer", {})
            details = data.get("details", {})
            virtual_services = data.get("virtual_services", [])
            errors = data.get("errors", {})
            
            lb_name = lb.get("name", "Unknown")
            lbci = lb.get("lbci") or lb.get("circuitId") or "N/A"
            status = lb.get("status", "Unknown")
            location = lb.get("_location", "Unknown")
            vip = lb.get("virtual_ip") or lb.get("virtualIp", "N/A")
            protocol = lb.get("protocol", "N/A")
            port = lb.get("port", "N/A")
            ssl_enabled = lb.get("ssl_enabled") or lb.get("sslEnabled", False)
            
            # Status emoji
            if status.lower() in ["active", "running", "healthy", "up"]:
                status_emoji = "✅"
            elif status.lower() in ["degraded", "warning"]:
                status_emoji = "⚠️"
            else:
                status_emoji = "❌"
            
            ssl_emoji = "🔒" if ssl_enabled else "🔓"
            
            # Build response
            response = f"⚖️ **Load Balancer Details**\n\n"
            response += f"### {status_emoji} {lb_name} {ssl_emoji}\n\n"
            
            # Basic Information
            response += f"**Basic Information:**\n"
            response += f"- **LBCI:** `{lbci}`\n"
            response += f"- **Status:** {status}\n"
            response += f"- **Location:** {location}\n"
            
            if vip != "N/A":
                response += f"- **Virtual IP:** {vip}\n"
            
            response += f"- **Protocol:** {protocol}"
            if port != "N/A":
                response += f" (Port {port})"
            response += "\n"
            
            response += f"- **SSL/TLS:** {'Enabled' if ssl_enabled else 'Disabled'}\n"
            
            # Configuration Details (if available)
            if details:
                algorithm = details.get("algorithm")
                if algorithm:
                    response += f"- **Algorithm:** {algorithm}\n"
                
                backend = details.get("backend_pool") or details.get("backendPool")
                if backend:
                    response += f"- **Backend Pool:** {backend}\n"
            
            # Virtual Services Section (CRITICAL)
            response += f"\n### 🌐 Virtual Services"
            
            if errors.get("virtual_services"):
                response += f" (⚠️ Error)\n\n"
                response += f"Failed to retrieve virtual services: {errors['virtual_services']}\n"
            elif not virtual_services:
                response += f"\n\n"
                response += f"ℹ️ No virtual services configured\n"
            else:
                response += f" ({len(virtual_services)})\n\n"
                
                for vs in virtual_services:
                    vs_name = vs.get("virtualServerName", "Unknown")
                    vip_ip = vs.get("vipIp", "N/A")
                    vs_port = vs.get("virtualServerport", "N/A")
                    vs_protocol = vs.get("protocol", "N/A")
                    vs_status = vs.get("status", "Unknown")
                    algorithm = vs.get("poolAlgorithm", "N/A")
                    monitors = vs.get("monitor", [])
                    pool_path = vs.get("virtualServerPath", "N/A")
                    persistence = vs.get("persistenceType")
                    
                    # Status emoji
                    vs_status_emoji = "✅" if vs_status.upper() == "UP" else "⚠️"
                    
                    response += f"#### {vs_name}\n\n"
                    response += f"- **VIP:** {vip_ip}:{vs_port}\n"
                    response += f"- **Protocol:** {vs_protocol}\n"
                    response += f"- **Status:** {vs_status_emoji} {vs_status}\n"
                    response += f"- **Algorithm:** {algorithm}\n"
                    
                    if monitors:
                        monitors_str = ", ".join(monitors) if isinstance(monitors, list) else monitors
                        response += f"- **Health Monitors:** {monitors_str}\n"
                    
                    if persistence:
                        response += f"- **Persistence:** {persistence}\n"
                    
                    if pool_path != "N/A":
                        response += f"- **Pool Path:** `{pool_path}`\n"
                    
                    response += "\n"
            
            # Configuration Details Error
            if errors.get("details"):
                response += f"\n⚠️ **Note:** Configuration details unavailable: {errors['details']}\n"
            
            return response
        
        except Exception as e:
            logger.error(f"❌ Error formatting detailed LB response: {e}", exc_info=True)
            # Fallback
            return f"⚖️ Load balancer details (formatting error):\n```json\n{json.dumps(data, indent=2)[:1000]}\n```"
    
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
