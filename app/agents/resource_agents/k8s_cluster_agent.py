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
                "Uses LLM intelligence to filter, format, and analyze cluster data."),
            resource_type="k8s_cluster",
            temperature=0.2)
    
    def get_supported_operations(self) -> List[str]:
        """Get list of supported operations."""
        return ["list", "create", "update", "delete", "scale", "read"]
    
    async def execute_operation(self,operation: str,params: Dict[str, Any],context: Dict[str, Any]) -> Dict[str, Any]:
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
            elif operation == "read":
                return await self._read_cluster(params, context)
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
                    "response": f"I don't support the '{operation}' operation for Kubernetes clusters yet."}
        except Exception as e:
            logger.error(f"❌ K8sClusterAgent error: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "response": f"An error occurred while {operation}ing clusters: {str(e)}"}
    
    async def _list_clusters(self,params: Dict[str, Any],context: Dict[str, Any]) -> Dict[str, Any]:
        """
        List K8s clusters with intelligent filtering and formatting.
        
        Supports filtering by:
        - endpoints: Datacenter/location IDs
        - businessUnits: Department/BU IDs
        - environments: Environment IDs
        - zones: Zone IDs
        
        Args:
            params: Parameters (endpoints, endpoint_names, filters, businessUnits, environments, zones)
            context: Context (user_query for intelligent formatting)
        Returns:
            Formatted cluster list
        """
        try:
            # FIRST: Extract auth_token, user_id, user_type, and engagement_id from context
            # This must happen BEFORE any API calls
            auth_token = context.get("auth_token")
            user_id = context.get("user_id")
            user_type = context.get("user_type")
            selected_engagement_id = context.get("selected_engagement_id")
            
            # Get engagement_id (required for URL parameter)
            if selected_engagement_id:
                engagement_id = selected_engagement_id
                logger.info(f"✅ Using selected engagement ID from context: {engagement_id}")
            else:
                engagement_id = await api_executor_service.get_engagement_id(
                    auth_token=auth_token,
                    user_id=user_id,
                    user_type=user_type
                )
            
            if not engagement_id:
                return {
                    "success": False,
                    "error": "Failed to get engagement ID",
                    "response": "Unable to retrieve engagement information. Please select an engagement first."
                }
            
            # Get endpoint IDs
            endpoint_ids = params.get("endpoints", [])
            
            # Get BU/Environment/Zone filters from params
            business_units = params.get("businessUnits", [])
            environments = params.get("environments", [])
            zones = params.get("zones", [])
            
            # Check if user explicitly wants all endpoints
            user_query = context.get("user_query", "").lower()
            wants_all = "all" in user_query or "every" in user_query or "all endpoints" in user_query
            
            # Check if user wants to filter by BU/Environment/Zone
            filter_request = self._detect_filter_request(user_query)
            if filter_request and not (business_units or environments or zones):
                # User asked to filter but hasn't selected options yet - show them the list
                logger.info(f"🔍 Filter request detected: {filter_request}")
                filter_options = await self._get_filter_options(filter_request, context)
                if filter_options and filter_options.get("needs_selection"):
                    # Store options in response metadata for orchestrator to save to state
                    filter_options["set_filter_state"] = True
                    filter_options["filter_options_for_state"] = filter_options.get("options", [])
                    filter_options["filter_type_for_state"] = filter_request
                    return filter_options  # Returns formatted response with selectable options
            if not endpoint_ids and not wants_all:
                # This is a fallback; ideally ValidationAgent should have asked for endpoint selection
                logger.info("📍 No specific endpoints provided, fetching all available endpoints")
                datacenters = await self.get_datacenters(
                    engagement_id=engagement_id,
                    user_roles=context.get("user_roles"),
                    auth_token=auth_token
                )
                endpoint_ids = [dc.get("endpointId") for dc in datacenters if dc.get("endpointId")]
            elif not endpoint_ids and wants_all:
                # User explicitly wants all endpoints
                logger.info("📍 User requested all endpoints, fetching all")
                datacenters = await self.get_datacenters(
                    engagement_id=engagement_id,
                    user_roles=context.get("user_roles"),
                    auth_token=auth_token
                )
                endpoint_ids = [dc.get("endpointId") for dc in datacenters if dc.get("endpointId")]
            if not endpoint_ids:
                return {
                    "success": True,
                    "data": [],
                    "response": "No datacenters found for your engagement."}
            logger.info(f"🔍 Listing clusters for endpoints: {endpoint_ids}")
            if business_units:
                logger.info(f"🏢 Filtering by BU IDs: {business_units}")
            if environments:
                logger.info(f"🌍 Filtering by Environment IDs: {environments}")
            if zones:
                logger.info(f"📍 Filtering by Zone IDs: {zones}")
            
            # Build API payload with filters
            api_payload = {
                "engagement_id": engagement_id,
                "endpoints": endpoint_ids
            }
            
            # Add optional BU/Environment/Zone filters to payload
            if business_units:
                api_payload["businessUnits"] = business_units
            if environments:
                api_payload["environments"] = environments
            if zones:
                api_payload["zones"] = zones
            
            # Call API (note: schema expects "endpoints" not "endpoint_ids")
            # engagement_id is a URL parameter, other params go in body
            result = await api_executor_service.execute_operation(
                resource_type="k8s_cluster",
                operation="list",
                params=api_payload,
                user_roles=context.get("user_roles", []),
                auth_token=context.get("auth_token")
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
            # Format response with agentic formatter (prevents hallucination)
            formatted_response = await self.format_response_agentic(
                operation="list",
                raw_data=clusters,
                user_query=user_query,
                context={"query_type": "general", "endpoint_names": params.get("endpoint_names", [])})
            return {
                "success": True,
                "data": clusters,
                "response": formatted_response,
                "metadata": {
                    "count": len(clusters),
                    "endpoints_queried": len(endpoint_ids),
                    "resource_type": "k8s_cluster"}}
        except Exception as e:
            logger.error(f"❌ Error listing clusters: {str(e)}", exc_info=True)
            raise
    
    async def _read_cluster(
        self,
        params: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Read/lookup a specific cluster and find its Zone → Environment → BU hierarchy.
        
        This performs a reverse lookup: given a cluster name, find which zone/env/BU it belongs to.
        
        Args:
            params: Parameters including cluster_name to look up
            context: Context (session_id, user_query, etc.)
            
        Returns:
            Dict with cluster details including zone/env/BU hierarchy
        """
        try:
            user_query = context.get("user_query", "")
            cluster_name = params.get("cluster_name") or self._extract_cluster_name(user_query)
            
            if not cluster_name:
                return {
                    "success": False,
                    "error": "No cluster name provided",
                    "response": "Please specify which cluster you want to look up. For example: 'Which zone is cluster my-cluster in?'"
                }
            
            # Check if user is specifically asking about firewall
            query_lower = user_query.lower()
            wants_firewall = any(kw in query_lower for kw in ["firewall", "fw", "edge gateway"])
            
            logger.info(f"🔍 Looking up cluster: {cluster_name} (wants_firewall: {wants_firewall})")
            
            # Extract auth_token and user_id from context
            auth_token = context.get("auth_token")
            user_id = context.get("user_id")
            user_type = context.get("user_type")
            selected_engagement_id = context.get("selected_engagement_id")
            
            # Step 1: Get all endpoints
            endpoints = await api_executor_service.get_endpoints(
                auth_token=auth_token,
                user_id=user_id,
                user_type=user_type,
                engagement_id=selected_engagement_id
            )
            if not endpoints:
                return {
                    "success": False,
                    "error": "Failed to get endpoints",
                    "response": "Unable to retrieve endpoint information."
                }
            
            all_endpoint_ids = [ep.get("endpointId") for ep in endpoints if ep.get("endpointId")]
            
            # Step 2: Get engagement_id
            if selected_engagement_id:
                engagement_id = selected_engagement_id
                logger.info(f"✅ Using selected engagement ID from state: {engagement_id}")
            else:
                engagement_id = await api_executor_service.get_engagement_id(
                    auth_token=auth_token,
                    user_id=user_id,
                    user_type=user_type
                )
            if not engagement_id:
                return {
                    "success": False,
                    "error": "Failed to get engagement ID",
                    "response": "Unable to retrieve engagement information."
                }
            
            # Step 3: List all clusters to find the target cluster
            api_payload = {
                "engagement_id": engagement_id,
                "endpoints": all_endpoint_ids
            }
            
            result = await api_executor_service.execute_operation(
                resource_type="k8s_cluster",
                operation="list",
                params=api_payload,
                user_roles=context.get("user_roles", []),
                auth_token=context.get("auth_token")
            )
            
            if not result.get("success"):
                return {
                    "success": False,
                    "error": result.get("error"),
                    "response": f"Failed to list clusters: {result.get('error')}"
                }
            
            all_clusters = result.get("data", [])
            
            # Find the target cluster (case-insensitive partial match)
            cluster_name_lower = cluster_name.lower()
            matching_clusters = [
                c for c in all_clusters 
                if cluster_name_lower in c.get("clusterName", "").lower()
            ]
            
            if not matching_clusters:
                return {
                    "success": False,
                    "error": f"Cluster '{cluster_name}' not found",
                    "response": f"I couldn't find a cluster named '{cluster_name}'. Please check the name and try again."
                }
            
            # If multiple matches, try exact match first
            exact_match = [c for c in matching_clusters if c.get("clusterName", "").lower() == cluster_name_lower]
            target_cluster = exact_match[0] if exact_match else matching_clusters[0]
            
            logger.info(f"✅ Found cluster: {target_cluster.get('clusterName')}")
            
            # Step 4: Get department details for hierarchy lookup
            dept_result = await api_executor_service.get_department_details(
                user_id=user_id,
                auth_token=auth_token
            )
            
            if not dept_result or not dept_result.get("success"):
                # Return basic cluster info without hierarchy
                return self._format_cluster_info_response(target_cluster, None, None)
            
            # Extract the department list from the response
            dept_list = dept_result.get("departmentList", [])
            
            if not dept_list:
                return self._format_cluster_info_response(target_cluster, None, None)
            
            # Step 5: Find the zone/env/BU hierarchy by filtering
            hierarchy = await self._find_cluster_hierarchy(
                target_cluster, 
                dept_list,  # Pass the list, not the full response dict
                engagement_id,
                context
            )
            
            # Step 6: Find firewall for this cluster's BU - ONLY if user asked for it
            firewall_info = None
            if wants_firewall and hierarchy:
                logger.info(f"🔥 User requested firewall info, looking up for BU {hierarchy.get('bu_id')}")
                firewall_info = await self._find_cluster_firewall(
                    hierarchy.get("bu_id"),
                    hierarchy.get("endpoint_id"),
                    context
                )
            
            return self._format_cluster_info_response(target_cluster, hierarchy, firewall_info, wants_firewall)
            
        except Exception as e:
            logger.error(f"❌ Error reading cluster: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "response": f"An error occurred while looking up the cluster: {str(e)}"
            }
    
    def _extract_cluster_name(self, query: str) -> Optional[str]:
        """
        Extract cluster name from user query.
        
        Handles patterns like:
        - "which zone is cluster blr-paas in?"
        - "what zone is blr-paas cluster in?"
        - "find cluster my-cluster"
        - "info about cluster test-cluster"
        """
        import re
        
        query_lower = query.lower()
        
        # Pattern 1: "cluster <name>" or "<name> cluster"
        patterns = [
            r"cluster[:\s]+([a-zA-Z0-9_-]+)",  # cluster: name or cluster name
            r"([a-zA-Z0-9_-]+)\s+cluster",      # name cluster
            r"is\s+([a-zA-Z0-9_-]+)\s+in",      # is <name> in
            r"about\s+([a-zA-Z0-9_-]+)",        # about <name>
            r"find\s+([a-zA-Z0-9_-]+)",         # find <name>
            r"lookup\s+([a-zA-Z0-9_-]+)",       # lookup <name>
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                name = match.group(1)
                # Filter out common words
                if name not in ["the", "a", "an", "my", "our", "this", "that", "which", "what", "zone", "environment", "bu"]:
                    return name
        
        return None
    
    async def _find_cluster_hierarchy(
        self,
        target_cluster: Dict[str, Any],
        dept_details: List[Dict[str, Any]],
        engagement_id: str,
        context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Find the Zone → Environment → BU hierarchy for a cluster.
        
        Strategy: For each zone in the hierarchy, check if filtering by that zone
        returns the target cluster. If yes, we found the hierarchy.
        """
        target_cluster_id = target_cluster.get("clusterId")
        target_cluster_name = target_cluster.get("clusterName", "")
        target_endpoint = target_cluster.get("displayNameEndpoint", "")
        target_endpoint_id = target_cluster.get("endpointId")  # May or may not exist
        
        logger.info(f"🔍 Finding hierarchy for cluster {target_cluster_name} (endpoint: {target_endpoint}, endpoint_id: {target_endpoint_id})")
        
        # Build a list of all zones with their hierarchy info
        zones_to_check = []
        
        for dept in dept_details:
            dept_name = dept.get("departmentName", "Unknown")
            dept_id = dept.get("departmentId")
            endpoint_id = dept.get("endpointId")
            endpoint_name = dept.get("endpointName", "Unknown")
            
            for env in dept.get("environmentList", []):
                env_name = env.get("environmentName", "Unknown")
                env_id = env.get("environmentId")
                
                for zone in env.get("zoneList", []):
                    zone_name = zone.get("zoneName", "Unknown")
                    zone_id = zone.get("zoneId")
                    
                    if zone_id and endpoint_id:
                        zones_to_check.append({
                            "zone_id": zone_id,
                            "zone_name": zone_name,
                            "environment_id": env_id,
                            "environment_name": env_name,
                            "bu_id": dept_id,
                            "bu_name": dept_name,
                            "endpoint_id": endpoint_id,
                            "endpoint_name": endpoint_name
                        })
        
        logger.info(f"📋 Found {len(zones_to_check)} total zones across all BUs")
        
        # Filter zones to only those that might match the cluster's endpoint
        # Use substring matching since endpoint names may differ (e.g., "Bengaluru" vs "Bengaluru (EP_V2_BL)")
        target_endpoint_lower = target_endpoint.lower()
        matching_endpoint_zones = []
        
        for zone_info in zones_to_check:
            zone_endpoint_lower = zone_info["endpoint_name"].lower()
            # Check if either name contains the other (handles partial matches)
            if target_endpoint_lower in zone_endpoint_lower or zone_endpoint_lower in target_endpoint_lower:
                matching_endpoint_zones.append(zone_info)
        
        logger.info(f"📋 {len(matching_endpoint_zones)} zones match endpoint '{target_endpoint}'")
        
        if not matching_endpoint_zones:
            # If no endpoint match, try all zones as fallback
            logger.warning(f"⚠️ No zones match endpoint '{target_endpoint}', checking all {len(zones_to_check)} zones")
            matching_endpoint_zones = zones_to_check
        
        # Check each zone - filter clusters by zone and see if target cluster appears
        for zone_info in matching_endpoint_zones:
            api_payload = {
                "engagement_id": engagement_id,
                "endpoints": [zone_info["endpoint_id"]],
                "zones": [zone_info["zone_id"]]
            }
            
            try:
                logger.debug(f"🔍 Checking zone: {zone_info['zone_name']} (ID: {zone_info['zone_id']})")
                
                result = await api_executor_service.execute_operation(
                    resource_type="k8s_cluster",
                    operation="list",
                    params=api_payload,
                    user_roles=context.get("user_roles", [])
                )
                
                if result.get("success"):
                    filtered_clusters = result.get("data", [])
                    
                    # Check if target cluster is in the filtered results
                    for cluster in filtered_clusters:
                        cluster_id = cluster.get("clusterId")
                        cluster_name = cluster.get("clusterName", "").lower()
                        
                        if cluster_id == target_cluster_id or cluster_name == target_cluster_name.lower():
                            logger.info(f"✅ Found hierarchy: Zone={zone_info['zone_name']}, Env={zone_info['environment_name']}, BU={zone_info['bu_name']}")
                            return zone_info
                            
            except Exception as e:
                logger.warning(f"⚠️ Error checking zone {zone_info['zone_name']}: {e}")
                continue
        
        logger.info(f"⚠️ Could not determine hierarchy for cluster {target_cluster_name} after checking {len(matching_endpoint_zones)} zones")
        return None
    
    async def _find_cluster_firewall(
        self,
        bu_id: int,
        endpoint_id: int,
        context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Find the firewall associated with a cluster's Business Unit.
        
        Firewalls are linked to BUs (departments). We fetch firewalls for the endpoint
        and find the one that matches the cluster's BU.
        
        Args:
            bu_id: Business Unit (department) ID
            endpoint_id: Endpoint ID
            context: Context with user roles
            
        Returns:
            Firewall info dict or None if not found
        """
        try:
            if not bu_id or not endpoint_id:
                return None
            
            logger.info(f"🔥 Looking for firewall for BU ID: {bu_id}, Endpoint ID: {endpoint_id}")
            
            # Extract auth_token and user_id from context
            auth_token = context.get("auth_token") if context else None
            user_id = context.get("user_id") if context else None
            
            # Use the direct list_firewalls method for better control
            result = await api_executor_service.list_firewalls(
                endpoint_ids=[endpoint_id],
                auth_token=auth_token,
                user_id=user_id
            )
            
            if not result.get("success"):
                logger.warning(f"⚠️ Failed to fetch firewalls: {result.get('error')}")
                return None
            
            firewalls = result.get("data", [])
            logger.info(f"🔥 Found {len(firewalls)} firewalls for endpoint {endpoint_id}")
            
            # Find firewall that matches the BU
            for fw in firewalls:
                # Log firewall structure for debugging
                fw_name = fw.get("displayName", "Unknown") if isinstance(fw, dict) else str(fw)[:50]
                logger.debug(f"🔍 Checking firewall: {fw_name}")
                
                if not isinstance(fw, dict):
                    logger.warning(f"⚠️ Unexpected firewall type: {type(fw)}")
                    continue
                    
                departments = fw.get("department", [])
                logger.debug(f"🔍 Firewall {fw_name} has {len(departments)} departments: {departments}")
                
                for dept in departments:
                    dept_id = dept.get("id")
                    # Compare as strings since API might return string IDs
                    if str(dept_id) == str(bu_id):
                        logger.info(f"✅ Found matching firewall: {fw.get('displayName')} for BU {dept.get('name')}")
                        return {
                            "id": fw.get("id"),
                            "display_name": fw.get("displayName"),
                            "technical_name": fw.get("technicalName"),
                            "component": fw.get("component"),
                            "component_type": fw.get("componentType"),
                            "throughput": fw.get("basicDetails", {}).get("throughput"),
                            "public_ips": fw.get("internetDetails", {}).get("publicIP"),
                            "category": fw.get("config", {}).get("category"),
                            "vdom_name": fw.get("config", {}).get("vdomName"),
                            "iks_enabled": fw.get("basicDetails", {}).get("iksEnabled"),
                            "hypervisor": fw.get("hypervisor"),
                            "department_name": dept.get("name"),
                            "department_id": dept.get("id")
                        }
            
            logger.info(f"⚠️ No firewall found matching BU ID {bu_id}")
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ Error finding firewall: {str(e)}")
            return None
    
    def _format_cluster_info_response(
        self,
        cluster: Dict[str, Any],
        hierarchy: Optional[Dict[str, Any]],
        firewall: Optional[Dict[str, Any]] = None,
        show_firewall_section: bool = False
    ) -> Dict[str, Any]:
        """Format the cluster info response with hierarchy and optional firewall details."""
        cluster_name = cluster.get("clusterName", "Unknown")
        
        response = f"## 📦 Cluster: {cluster_name}\n\n"
        
        # Basic cluster info
        response += "### Cluster Details\n"
        response += f"- **Status:** {cluster.get('status', 'Unknown')}\n"
        response += f"- **Type:** {cluster.get('type', 'Unknown')}\n"
        response += f"- **Nodes:** {cluster.get('nodescount', 'Unknown')}\n"
        response += f"- **K8s Version:** {cluster.get('kubernetesVersion', 'Unknown')}\n"
        response += f"- **Location:** {cluster.get('displayNameEndpoint', 'Unknown')}\n"
        response += f"- **Backup Enabled:** {'Yes' if cluster.get('isIksBackupEnabled') else 'No'}\n"
        response += f"- **Created:** {cluster.get('createdTime', 'Unknown')}\n"
        response += f"- **Cluster ID:** {cluster.get('clusterId', 'Unknown')}\n\n"
        
        # Hierarchy info
        if hierarchy:
            response += "### 🗂️ Organization Hierarchy\n\n"
            response += f"```\n"
            response += f"📍 Zone: {hierarchy['zone_name']}\n"
            response += f"   └── 🌍 Environment: {hierarchy['environment_name']}\n"
            response += f"       └── 🏢 Business Unit: {hierarchy['bu_name']}\n"
            response += f"           └── 📍 Location: {hierarchy['endpoint_name']}\n"
            response += f"```\n\n"
            
            response += "| Level | Name | ID |\n"
            response += "|-------|------|----|\n"
            response += f"| Zone | {hierarchy['zone_name']} | {hierarchy['zone_id']} |\n"
            response += f"| Environment | {hierarchy['environment_name']} | {hierarchy['environment_id']} |\n"
            response += f"| Business Unit | {hierarchy['bu_name']} | {hierarchy['bu_id']} |\n"
            response += f"| Location | {hierarchy['endpoint_name']} | {hierarchy['endpoint_id']} |\n"
        else:
            response += "### 🗂️ Organization Hierarchy\n\n"
            response += "_Could not determine the zone/environment/business unit hierarchy for this cluster._\n"
            response += "_The cluster may not be associated with a specific zone, or the hierarchy data is unavailable._\n"
        
        # Firewall info - ONLY shown if user specifically asked for it
        if show_firewall_section:
            if firewall:
                response += "\n### 🔥 Associated Firewall\n\n"
                response += f"- **Name:** {firewall.get('display_name', 'Unknown')}\n"
                response += f"- **Technical Name:** {firewall.get('technical_name', 'Unknown')}\n"
                response += f"- **Component:** {firewall.get('component', 'Unknown')}\n"
                response += f"- **Category:** {firewall.get('category', 'Unknown')}\n"
                response += f"- **Throughput:** {firewall.get('throughput', 'Unknown')}\n"
                response += f"- **IKS Enabled:** {firewall.get('iks_enabled', 'Unknown')}\n"
                if firewall.get('public_ips'):
                    response += f"- **Public IPs:** `{firewall.get('public_ips')}`\n"
                response += f"- **VDOM Name:** {firewall.get('vdom_name', 'Unknown')}\n"
                response += f"- **Firewall ID:** {firewall.get('id', 'Unknown')}\n"
            else:
                # User asked for firewall but none was found
                response += "\n### 🔥 Associated Firewall\n\n"
                response += "_No firewall found associated with this cluster's business unit._\n"
        
        return {
            "success": True,
            "data": {
                "cluster": cluster,
                "hierarchy": hierarchy,
                "firewall": firewall
            },
            "response": response,
            "metadata": {
                "cluster_name": cluster_name,
                "has_hierarchy": hierarchy is not None,
                "has_firewall": firewall is not None,
                "resource_type": "k8s_cluster"
            }
        }
    
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
            api_payload = self._build_cluster_payload(params)
            logger.info(f"📦 Transformed payload keys: {list(api_payload.keys())}")
            # Call API
            result = await api_executor_service.execute_operation(
                resource_type="k8s_cluster",
                operation="create",
                params=api_payload,
                user_roles=context.get("user_roles", []),
                auth_token=context.get("auth_token"))
            if not result.get("success"):
                return {
                    "success": False,
                    "error": result.get("error"),
                    "response": f"Failed to create cluster: {result.get('error')}"}
            # Format response with LLM
            user_query = context.get("user_query", "create cluster")
            formatted_response = await self.format_response_with_llm(
                operation="create",
                raw_data=result.get("data", {}),
                user_query=user_query,
                context=params)
            return {
                "success": True,
                "data": result.get("data", {}),
                "response": formatted_response}
            
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
            "labelsNTaints": "no"}
        
        # Build worker node
        worker_node = {
            "vmHostName": params.get("workerPoolName", ""),
            "vmFlavor": flavor_info.get("flavor_name", "B4"),
            "skuCode": flavor_info.get("sku_code", "B4.UBN"),
            "nodeType": "Worker",
            "replicaCount": params.get("replicaCount", 1),
            "maxReplicaCount": params.get("maxReplicas") if params.get("enableAutoscaling") else None,
            "additionalDisk": {},
            "labelsNTaints": "no"}
        # Build vmSpecificInput
        vm_specific_input = [master_node, worker_node]
        # Build imageDetails
        image_details = {
            "valueOSModel": os_info.get("os_model"),
            "valueOSMake": os_info.get("os_make"),
            "valueOSVersion": os_info.get("os_version"),
            "valueOSServicePack": None}
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
            "zoneId": params.get("_zone_id"),  
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
            "imageDetails": image_details}
        if cni_driver_payload:
            payload["networkingDriver"] = cni_driver_payload
        logger.info(f"✅ Built cluster payload: {payload.get('clusterName')}")
        return payload
    
    async def _scale_cluster(self,params: Dict[str, Any],context: Dict[str, Any]) -> Dict[str, Any]:
        """Scale cluster nodes."""
        return {
            "success": False,
            "response": "Cluster scaling is not yet implemented."}
    
    async def _delete_cluster(self,params: Dict[str, Any],context: Dict[str, Any]) -> Dict[str, Any]:
        """Delete a cluster."""
        return {
            "success": False,
            "response": "Cluster deletion is not yet implemented."}

    def _extract_filter_criteria(self, user_query: str) -> Optional[str]:
        """
        Extract filter criteria from user query.
        Args:
            user_query: User's original query
        Returns:
            Filter criteria string or None
        """
        query_lower = user_query.lower()
        filter_keywords = [
            "active", "running", "pending", "failed",
            "production", "staging", "development",
            "version", "v1.", "latest"]
        for keyword in filter_keywords:
            if keyword in query_lower:
                words = query_lower.split()
                if keyword in words:
                    idx = words.index(keyword)
                    start = max(0, idx - 2)
                    end = min(len(words), idx + 3)
                    return " ".join(words[start:end])
        return None
    
    def _detect_filter_request(self, user_query: str) -> Optional[str]:
        """
        Detect if user is asking to filter clusters by BU/Environment/Zone.
        
        Args:
            user_query: User's query (lowercase)
            
        Returns:
            Filter type: "bu", "environment", "zone", or None
        """
        query_lower = user_query.lower()
        
        # Keywords indicating filter intent
        filter_keywords = ["filter", "by", "in", "for", "from"]
        has_filter_intent = any(kw in query_lower for kw in filter_keywords)
        
        if not has_filter_intent:
            return None
        
        # Check what type of filter
        bu_keywords = ["bu", "business unit", "business units", "department", "dept"]
        env_keywords = ["environment", "environments", "env"]
        zone_keywords = ["zone", "zones", "network", "segment"]
        
        for kw in bu_keywords:
            if kw in query_lower:
                return "bu"
        
        for kw in env_keywords:
            if kw in query_lower:
                return "environment"
        
        for kw in zone_keywords:
            if kw in query_lower:
                return "zone"
        
        return None
    
    async def _get_filter_options(self, filter_type: str, context: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """
        Get available filter options (BU/Environment/Zone) for user to select.
        
        IMPORTANT: Each option includes endpoint_id so we can make API calls without
        asking the user for endpoint separately.
        
        Args:
            filter_type: "bu", "environment", or "zone"
            context: Context with auth_token, user_id, etc.
            
        Returns:
            Response dict with formatted options for user selection, or None on error
        """
        try:
            # Fetch department details
            user_id = context.get("user_id") if context else None
            auth_token = context.get("auth_token") if context else None
            dept_result = await api_executor_service.get_department_details(
                user_id=user_id,
                auth_token=auth_token
            )
            
            if not dept_result.get("success"):
                logger.warning("⚠️ Could not fetch department details for filtering")
                return None
            
            dept_list = dept_result.get("departmentList", [])
            if not dept_list:
                logger.warning("⚠️ No departments found for filtering")
                return None
            
            # Build options list based on filter type
            # CRITICAL: Every option MUST include endpoint_id for the API call
            options = []
            
            if filter_type == "bu":
                # List all Business Units
                for dept in dept_list:
                    dept_name = dept.get("departmentName", "Unknown")
                    dept_id = dept.get("departmentId")
                    endpoint_id = dept.get("endpointId")  # CRITICAL: Include endpoint
                    endpoint_name = dept.get("endpointName", "Unknown Location")
                    env_count = dept.get("noOfEnvironments", 0)
                    
                    if dept_id and endpoint_id:
                        options.append({
                            "id": dept_id,
                            "name": dept_name,
                            "endpoint_id": endpoint_id,  # For API call
                            "endpoint_name": endpoint_name,
                            "location": endpoint_name,
                            "environments": env_count
                        })
                
                # Format response
                response = "## 🏢 Available Business Units\n\n"
                response += "Select one or more to filter clusters:\n\n"
                response += "| # | Business Unit | Location | Environments |\n"
                response += "|---|--------------|----------|-------------|\n"
                
                for idx, opt in enumerate(options, 1):
                    response += f"| {idx} | {opt['name']} | {opt['location']} | {opt['environments']} |\n"
                
                response += "\n\n💡 **Reply with the name or number** of the Business Unit(s) you want to filter by."
                response += "\nExample: `1` or `qatest332` or `1, 3` for multiple selections."
                
            elif filter_type == "environment":
                # List all Environments (grouped by BU)
                for dept in dept_list:
                    dept_name = dept.get("departmentName", "Unknown")
                    endpoint_id = dept.get("endpointId")  # From parent BU
                    endpoint_name = dept.get("endpointName", "Unknown Location")
                    
                    for env in dept.get("environmentList", []):
                        env_name = env.get("environmentName", "Unknown")
                        env_id = env.get("environmentId")
                        zone_count = env.get("noOfZones", 0)
                        
                        if env_id and endpoint_id:
                            options.append({
                                "id": env_id,
                                "name": env_name,
                                "endpoint_id": endpoint_id,  # From parent BU
                                "endpoint_name": endpoint_name,
                                "bu": dept_name,
                                "location": endpoint_name,
                                "zones": zone_count
                            })
                
                # Format response
                response = "## 🌍 Available Environments\n\n"
                response += "Select one or more to filter clusters:\n\n"
                response += "| # | Environment | Business Unit | Location | Zones |\n"
                response += "|---|------------|--------------|----------|-------|\n"
                
                for idx, opt in enumerate(options, 1):
                    response += f"| {idx} | {opt['name']} | {opt['bu']} | {opt['location']} | {opt['zones']} |\n"
                
                response += "\n\n💡 **Reply with the name or number** of the Environment(s) you want to filter by."
                response += "\nExample: `2` or `production` or `1, 4` for multiple selections."
                
            elif filter_type == "zone":
                # List all Zones (grouped by Environment and BU)
                for dept in dept_list:
                    dept_name = dept.get("departmentName", "Unknown")
                    endpoint_name = dept.get("endpointName", "Unknown Location")
                    endpoint_id = dept.get("endpointId")
                    
                    for env in dept.get("environmentList", []):
                        env_name = env.get("environmentName", "Unknown")
                        env_id = env.get("environmentId")
                        
                        for zone in env.get("zoneList", []):
                            zone_name = zone.get("zoneName", "Unknown")
                            zone_id = zone.get("zoneId")
                            
                            if zone_id and endpoint_id:
                                options.append({
                                    "id": zone_id,
                                    "name": zone_name,
                                    "endpoint_id": endpoint_id,  # From parent BU
                                    "endpoint_name": endpoint_name,
                                    "environment": env_name,
                                    "environment_id": env_id,
                                    "bu": dept_name,
                                    "location": endpoint_name
                                })
                
                # Format response
                response = "## 📍 Available Zones\n\n"
                response += "Select one or more to filter clusters:\n\n"
                response += "| # | Zone | Environment | Business Unit | Location |\n"
                response += "|---|------|------------|--------------|----------|\n"
                
                for idx, opt in enumerate(options, 1):
                    response += f"| {idx} | {opt['name']} | {opt['environment']} | {opt['bu']} | {opt['location']} |\n"
                
                response += "\n\n💡 **Reply with the name or number** of the Zone(s) you want to filter by."
                response += "\nExample: `3` or `production-zone` or `2, 5` for multiple selections."
            
            else:
                return None
            
            if not options:
                return {
                    "success": True,
                    "response": f"No {filter_type}s found for your engagement.",
                    "needs_selection": False
                }
            
            logger.info(f"📋 Showing {len(options)} {filter_type} options for selection")
            
            return {
                "success": True,
                "response": response,
                "needs_selection": True,
                "filter_type": filter_type,
                "options": options,
                "metadata": {
                    "filter_type": filter_type,
                    "option_count": len(options),
                    "resource_type": "k8s_cluster",
                    "awaiting_filter_selection": True,  # Special flag for filter selection
                    "skip_endpoint_prompt": True  # Tell validation to NOT ask for endpoint
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting filter options: {str(e)}")
            return None
    
    def parse_filter_selection(
        self,
        user_input: str,
        options: List[Dict[str, Any]],
        filter_type: str
    ) -> Optional[Dict[str, Any]]:
        """
        Parse user's selection from filter options.
        
        Handles:
        - Numbers: "9" → selects option at index 8
        - Names: "qatest332" → matches by name
        - Multiple: "1, 3" or "1 and 3" → selects multiple
        
        Args:
            user_input: User's response (e.g., "9", "qatest332", "1, 3")
            options: List of options that were shown to user
            filter_type: "bu", "environment", or "zone"
            
        Returns:
            Dict with selected options info including endpoint_ids, or None if no match
        """
        if not options:
            return None
        
        user_input = user_input.strip()
        selected = []
        
        # Try to parse as comma/and separated numbers or names
        import re
        parts = re.split(r'[,;]|\band\b', user_input)
        parts = [p.strip() for p in parts if p.strip()]
        
        for part in parts:
            # Try as number first
            try:
                idx = int(part) - 1  # Convert to 0-based index
                if 0 <= idx < len(options):
                    selected.append(options[idx])
                    continue
            except ValueError:
                pass
            
            # Try as name match (case-insensitive, partial match)
            part_lower = part.lower()
            for opt in options:
                opt_name = (opt.get("name") or "").lower()
                if part_lower in opt_name or opt_name in part_lower:
                    if opt not in selected:
                        selected.append(opt)
                    break
        
        if not selected:
            logger.info(f"❌ No match found for filter selection: '{user_input}'")
            return None
        
        # Extract the filter IDs and endpoint IDs from selected options
        filter_ids = [opt["id"] for opt in selected]
        endpoint_ids = list(set(opt["endpoint_id"] for opt in selected))  # Dedupe endpoints
        endpoint_names = list(set(opt.get("endpoint_name", opt.get("location", "")) for opt in selected))
        selected_names = [opt["name"] for opt in selected]
        
        logger.info(f"✅ Parsed filter selection: {selected_names} (filter_ids={filter_ids}, endpoints={endpoint_ids})")
        
        # Determine the filter key based on filter_type
        filter_key_map = {
            "bu": "businessUnits",
            "environment": "environments",
            "zone": "zones"
        }
        filter_key = filter_key_map.get(filter_type, "businessUnits")
        
        return {
            "matched": True,
            "filter_type": filter_type,
            "filter_key": filter_key,
            "filter_ids": filter_ids,
            "endpoint_ids": endpoint_ids,
            "endpoint_names": endpoint_names,
            "selected_names": selected_names,
            "selected_options": selected
        }

