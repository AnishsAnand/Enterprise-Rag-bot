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
            
            # Add sources if available
            if sources:
                response += "\n\n**Sources:**\n"
                for i, source in enumerate(sources[:3], 1):
                    title = source.get("title", source.get("url", "Unknown"))
                    response += f"{i}. {title}\n"
            
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


# Create global instance
response_formatter = ResponseFormatter()

