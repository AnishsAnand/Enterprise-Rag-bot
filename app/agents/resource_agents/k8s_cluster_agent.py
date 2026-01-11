"""
K8s Cluster Agent - Specialized agent for Kubernetes cluster operations.
Handles listing, creating, scaling, and managing Kubernetes clusters.
"""

from typing import Any, Dict, List, Optional
import logging

from app.agents.resource_agents.base_resource_agent import BaseResourceAgent
from app.services.api_executor_service import api_executor_service

logger = logging.getLogger(__name__)


class K8sClusterAgent(BaseResourceAgent):
    """
    Agent for Kubernetes cluster operations.
    
    Supported Operations:
    - list: List clusters (with intelligent filtering and formatting)
    - create: Create new cluster
    - update: Update cluster configuration
    - delete: Delete cluster
    - scale: Scale cluster nodes
    """
    
    def __init__(self):
        super().__init__(
            agent_name="K8sClusterAgent",
            agent_description=(
                "Specialized agent for Kubernetes cluster operations. "
                "Uses LLM intelligence to filter, format, and analyze cluster data."
            ),
            resource_type="k8s_cluster",
            temperature=0.2
        )
    
    def get_supported_operations(self) -> List[str]:
        """Get list of supported operations."""
        return ["list", "create", "update", "delete", "scale"]
    
    async def execute_operation(
        self,
        operation: str,
        params: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a K8s cluster operation with LLM intelligence.
        
        Args:
            operation: Operation to perform (list, create, etc.)
            params: Parameters (endpoints, filters, etc.)
            context: Context (session_id, user_query, etc.)
            
        Returns:
            Dict with success status and formatted response
        """
        try:
            logger.info(f"🚢 K8sClusterAgent executing: {operation}")
            
            if operation == "list":
                return await self._list_clusters(params, context)
            elif operation == "create":
                return await self._create_cluster(params, context)
            elif operation == "scale":
                return await self._scale_cluster(params, context)
            elif operation == "delete":
                return await self._delete_cluster(params, context)
            else:
                return {
                    "success": False,
                    "error": f"Unsupported operation: {operation}",
                    "response": f"I don't support the '{operation}' operation for Kubernetes clusters yet."
                }
                
        except Exception as e:
            logger.error(f"❌ K8sClusterAgent error: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "response": f"An error occurred while {operation}ing clusters: {str(e)}"
            }
    
    async def _list_clusters(
        self,
        params: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        List K8s clusters with intelligent filtering and formatting.
        
        Args:
            params: Parameters (endpoints, endpoint_names, filters)
            context: Context (user_query for intelligent formatting)
            
        Returns:
            Formatted cluster list
        """
        try:
            # Get endpoint IDs
            endpoint_ids = params.get("endpoints", [])
            
            # Check if user explicitly wants all endpoints
            user_query = context.get("user_query", "").lower()
            wants_all = "all" in user_query or "every" in user_query or "all endpoints" in user_query
            
            if not endpoint_ids and not wants_all:
                # No specific endpoints and user didn't ask for all - fetch all endpoints
                # This is a fallback; ideally ValidationAgent should have asked for endpoint selection
                logger.info("📍 No specific endpoints provided, fetching all available endpoints")
                datacenters = await self.get_datacenters()
                endpoint_ids = [dc.get("endpointId") for dc in datacenters if dc.get("endpointId")]
            elif not endpoint_ids and wants_all:
                # User explicitly wants all endpoints
                logger.info("📍 User requested all endpoints, fetching all")
                datacenters = await self.get_datacenters()
                endpoint_ids = [dc.get("endpointId") for dc in datacenters if dc.get("endpointId")]
            
            if not endpoint_ids:
                return {
                    "success": True,
                    "data": [],
                    "response": "No datacenters found for your engagement."
                }
            
            logger.info(f"🔍 Listing clusters for endpoints: {endpoint_ids}")
            
            # Get engagement_id (required for URL parameter)
            engagement_id = await api_executor_service.get_engagement_id()
            if not engagement_id:
                return {
                    "success": False,
                    "error": "Failed to get engagement ID",
                    "response": "Unable to retrieve engagement information."
                }
            
            # Call API (note: schema expects "endpoints" not "endpoint_ids")
            # engagement_id is a URL parameter, endpoints goes in body
            result = await api_executor_service.execute_operation(
                resource_type="k8s_cluster",
                operation="list",
                params={
                    "engagement_id": engagement_id,
                    "endpoints": endpoint_ids
                },
                user_roles=context.get("user_roles", [])
            )
            
            if not result.get("success"):
                return {
                    "success": False,
                    "error": result.get("error"),
                    "response": f"Failed to list clusters: {result.get('error')}"
                }
            
            clusters = result.get("data", [])
            logger.info(f"✅ Found {len(clusters)} clusters")
            
            # Apply intelligent filtering if user specified criteria
            user_query = context.get("user_query", "")
            filter_criteria = self._extract_filter_criteria(user_query)
            
            if filter_criteria and clusters:
                logger.info(f"🔍 Applying filter: {filter_criteria}")
                clusters = await self.filter_with_llm(clusters, filter_criteria, user_query)
                logger.info(f"✅ After filtering: {len(clusters)} clusters")
            
            # Format response with LLM
            formatted_response = await self.format_response_with_llm(
                operation="list",
                raw_data=clusters,
                user_query=user_query,
                context={"endpoint_names": params.get("endpoint_names", [])}
            )
            
            return {
                "success": True,
                "data": clusters,
                "response": formatted_response,
                "metadata": {
                    "count": len(clusters),
                    "endpoints_queried": len(endpoint_ids),
                    "resource_type": "k8s_cluster"
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error listing clusters: {str(e)}", exc_info=True)
            raise
    
    async def _create_cluster(
        self,
        params: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a new K8s cluster.
        
        Args:
            params: Cluster creation parameters
            context: Context
            
        Returns:
            Creation result
        """
        try:
            logger.info(f"🚀 Creating cluster with params: {list(params.keys())}")
            
            # Transform collected params into API payload format
            api_payload = self._build_cluster_payload(params)
            logger.info(f"📦 Transformed payload keys: {list(api_payload.keys())}")
            
            # Call API
            result = await api_executor_service.execute_operation(
                resource_type="k8s_cluster",
                operation="create",
                params=api_payload,
                user_roles=context.get("user_roles", [])
            )
            
            if not result.get("success"):
                return {
                    "success": False,
                    "error": result.get("error"),
                    "response": f"Failed to create cluster: {result.get('error')}"
                }
            
            # Format response with LLM
            user_query = context.get("user_query", "create cluster")
            formatted_response = await self.format_response_with_llm(
                operation="create",
                raw_data=result.get("data", {}),
                user_query=user_query,
                context=params
            )
            
            return {
                "success": True,
                "data": result.get("data", {}),
                "response": formatted_response
            }
            
        except Exception as e:
            logger.error(f"❌ Error creating cluster: {str(e)}", exc_info=True)
            raise
    
    def _build_cluster_payload(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform collected parameters into API payload format for customer login.
        
        For customer login, the payload structure is:
        - vmPurpose: "" (empty)
        - clusterMode: "High availability" (hardcoded)
        - dedicatedDeployment: false (hardcoded)
        - Master node: hardcoded D8 flavor
        - Worker nodes: from collected params
        
        Args:
            params: Collected parameters from handler
            
        Returns:
            API payload dict
        """
        # Extract OS details
        os_info = params.get("operatingSystem", {})
        flavor_info = params.get("flavor", {})
        
        # Build master node (hardcoded for customer)
        master_node = {
            "vmHostName": "",
            "vmFlavor": "D8",
            "skuCode": "D8.UBN",
            "nodeType": "Master",
            "replicaCount": 3,
            "maxReplicaCount": None,
            "additionalDisk": {},
            "labelsNTaints": "no"
        }
        
        # Build worker node
        worker_node = {
            "vmHostName": params.get("workerPoolName", ""),
            "vmFlavor": flavor_info.get("flavor_name", "B4"),
            "skuCode": flavor_info.get("sku_code", "B4.UBN"),
            "nodeType": "Worker",
            "replicaCount": params.get("replicaCount", 1),
            "maxReplicaCount": params.get("maxReplicas") if params.get("enableAutoscaling") else None,
            "additionalDisk": {},
            "labelsNTaints": "no"
        }
        
        # Build vmSpecificInput
        vm_specific_input = [master_node, worker_node]
        
        # Build imageDetails
        image_details = {
            "valueOSModel": os_info.get("os_model"),
            "valueOSMake": os_info.get("os_make"),
            "valueOSVersion": os_info.get("os_version"),
            "valueOSServicePack": None
        }
        
        # Build CNI driver
        cni_driver_payload = None
        if params.get("cniDriver"):
            cni_driver_payload = [{"name": params.get("cniDriver")}]
        
        # Build final payload
        payload = {
            "name": "",
            "hypervisor": os_info.get("hypervisor"),
            "purpose": "ipc",
            "vmPurpose": "",
            "imageId": os_info.get("os_id"),
            "zoneId": params.get("_zone_id"),  # Hidden param from zone selection
            "alertSuppression": True,
            "iops": 1,
            "isKdumpOrPageEnabled": "No",
            "applicationType": "Container",
            "application": "Containers",
            "vmSpecificInput": vm_specific_input,
            "clusterMode": "High availability",
            "dedicatedDeployment": False,
            "clusterName": params.get("clusterName"),
            "k8sVersion": params.get("k8sVersion"),
            "circuitId": flavor_info.get("circuit_id", "E-IPCTEAM-1602"),
            "vApp": "",
            "imageDetails": image_details
        }
        
        # Add optional networkingDriver if present
        if cni_driver_payload:
            payload["networkingDriver"] = cni_driver_payload
        
        logger.info(f"✅ Built cluster payload: {payload.get('clusterName')}")
        return payload
    
    async def _scale_cluster(
        self,
        params: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Scale cluster nodes."""
        # TODO: Implement when scaling API is available
        return {
            "success": False,
            "response": "Cluster scaling is not yet implemented."
        }
    
    async def _delete_cluster(
        self,
        params: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Delete a cluster."""
        # TODO: Implement when delete API is available
        return {
            "success": False,
            "response": "Cluster deletion is not yet implemented."
        }
    
    def _extract_filter_criteria(self, user_query: str) -> Optional[str]:
        """
        Extract filter criteria from user query.
        
        Examples:
            "list active clusters" → "active"
            "show clusters running version 1.28" → "version 1.28"
            "clusters in production environment" → "production environment"
        
        Args:
            user_query: User's original query
            
        Returns:
            Filter criteria string or None
        """
        query_lower = user_query.lower()
        
        # Common filter keywords
        filter_keywords = [
            "active", "running", "pending", "failed",
            "production", "staging", "development",
            "version", "v1.", "latest"
        ]
        
        for keyword in filter_keywords:
            if keyword in query_lower:
                # Extract surrounding context
                words = query_lower.split()
                if keyword in words:
                    idx = words.index(keyword)
                    # Get 2 words before and after for context
                    start = max(0, idx - 2)
                    end = min(len(words), idx + 3)
                    return " ".join(words[start:end])
        
        return None

