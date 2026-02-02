"""
RBAC Agent - Handles Role-Based Access Control operations.
Manages user role bindings, permissions, business units, environments, and access queries.

API Endpoints Used:
- POST /portalservice/api/v1/getAuthToken - Get authentication token
- GET /portalservice/securityservice/departments/{engagementId} - List Business Units
- GET /portalservice/securityservice/deptDetailsForEngagement/{engagementId} - BUs with Environments
- POST /paasservice/paas/getIksClusterRoleBindingInEnv/{envId} - Get user role bindings in Environment
"""

from typing import Any, Dict, List, Optional
import logging
import aiohttp
import re
from datetime import datetime

from app.agents.resource_agents.base_resource_agent import BaseResourceAgent
from app.services.api_executor_service import api_executor_service

logger = logging.getLogger(__name__)


class RBACAgent(BaseResourceAgent):
    """
    Specialized agent for RBAC (Role-Based Access Control) operations.
    
    Handles:
    - Listing Business Units (BUs) and their details
    - Listing Environments (filtered by BU or all)
    - Listing users with their cluster role bindings
    - Querying specific user's access permissions
    - Finding users with specific roles on clusters
    - Filtering by role type (admin, exceptdelete, etc.)
    """
    
    # Base URL for APIs
    BASE_URL = "https://ipcloud.tatacommunications.com"
    
    # Default engagement ID (can be fetched dynamically)
    DEFAULT_ENGAGEMENT_ID = 1602
    
    # Available role types in the system
    ROLE_TYPES = ["admin", "exceptdelete", "customer-namespace-access"]
    
    # Cache for BU and Environment data
    _bu_cache: Dict[str, Any] = {}
    _env_cache: Dict[str, Any] = {}
    _bu_env_details_cache: Dict[str, Any] = {}
    
    def __init__(self):
        super().__init__(
            agent_name="RBACAgent",
            agent_description=(
                "Handles Role-Based Access Control operations for Kubernetes clusters. "
                "Manages Business Units, Environments, user role bindings, permissions, and access queries."
            ),
            resource_type="rbac",
            temperature=0.2
        )
        logger.info("✅ RBACAgent initialized")
    
    def get_supported_operations(self) -> List[str]:
        """Get list of supported RBAC operations."""
        return [
            "list",              # List all users with their role bindings (needs env_id)
            "read",              # Get specific user's access
            "list_bu",           # List all Business Units
            "list_env",          # List all Environments (optionally filtered by BU)
            "show_env",          # Show specific environment details
            "show_bu",           # Show specific BU with its environments
        ]
    
    async def execute_operation(
        self,
        operation: str,
        params: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute RBAC operation.
        
        Args:
            operation: Operation to perform
            params: Parameters for the operation
            context: Execution context
            
        Returns:
            Dict with operation result
        """
        logger.info(f"🔐 RBACAgent executing operation: {operation}")
        logger.info(f"   Params: {params}")
        logger.info(f"   Context keys: {list(context.keys())}")
        
        try:
            if operation == "list":
                return await self._list_role_bindings(params, context)
            elif operation == "read":
                return await self._get_user_access(params, context)
            elif operation in ["list_bu", "listbu", "list_business_units"]:
                return await self._list_business_units(params, context)
            elif operation in ["list_env", "listenv", "list_environments"]:
                return await self._list_environments(params, context)
            elif operation in ["show_env", "showenv", "show_environment"]:
                return await self._show_environment_details(params, context)
            elif operation in ["show_bu", "showbu", "show_business_unit"]:
                return await self._show_bu_details(params, context)
            else:
                return {
                    "success": False,
                    "error": f"Unsupported operation: {operation}",
                    "output": f"Operation '{operation}' is not supported by RBACAgent"
                }
                
        except Exception as e:
            logger.error(f"❌ RBACAgent operation failed: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "output": f"Failed to execute RBAC operation: {str(e)}"
            }
    
    # =========================================================================
    # Authentication
    # =========================================================================
    
    async def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for API requests."""
        try:
            headers = await api_executor_service._get_auth_headers()
            headers["Accept"] = "application/json"
            
            if "Authorization" in headers:
                logger.info(f"✅ RBACAgent got auth headers (token length: {len(headers.get('Authorization', ''))})")
            else:
                logger.error("❌ RBACAgent: No Authorization header in response!")
            
            return headers
        except Exception as e:
            logger.error(f"❌ RBACAgent: Failed to get auth headers: {e}", exc_info=True)
            return {
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
    
    async def _get_engagement_id(self) -> int:
        """
        Get the IPC engagement ID for API calls.
        
        Note: We need the IPC engagement ID (e.g., 1602) for securityservice APIs,
        not the PAAS engagement ID (e.g., 1923).
        """
        try:
            # First try to get PAAS engagement ID
            paas_engagement_id = await api_executor_service.get_engagement_id()
            if paas_engagement_id:
                # Then get the IPC engagement ID from PAAS engagement
                ipc_engagement_id = await api_executor_service.get_ipc_engagement_id(
                    engagement_id=paas_engagement_id
                )
                if ipc_engagement_id:
                    logger.info(f"✅ Got IPC engagement ID: {ipc_engagement_id}")
                    return ipc_engagement_id
        except Exception as e:
            logger.warning(f"Could not fetch engagement ID: {e}")
        
        logger.info(f"⚠️ Using default engagement ID: {self.DEFAULT_ENGAGEMENT_ID}")
        return self.DEFAULT_ENGAGEMENT_ID
    
    # =========================================================================
    # Business Unit Operations
    # =========================================================================
    
    async def _fetch_business_units(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Fetch all Business Units from API.
        
        API: GET /portalservice/securityservice/departments/{engagementId}
        
        Returns:
            List of Business Unit dictionaries
        """
        cache_key = "business_units"
        if not force_refresh and cache_key in self._bu_cache:
            return self._bu_cache[cache_key]
        
        try:
            engagement_id = await self._get_engagement_id()
            url = f"{self.BASE_URL}/portalservice/securityservice/departments/{engagement_id}"
            headers = await self._get_auth_headers()
            
            logger.info(f"📋 Fetching Business Units from: {url}")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, ssl=False) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("status") == "success":
                            bus = data.get("data", {}).get("department", [])
                            self._bu_cache[cache_key] = bus
                            logger.info(f"✅ Fetched {len(bus)} Business Units")
                            return bus
                    
                    logger.error(f"❌ Failed to fetch BUs: status={response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"❌ Error fetching Business Units: {e}")
            return []
    
    async def _fetch_bu_with_environments(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Fetch Business Units with their Environments.
        
        API: GET /portalservice/securityservice/deptDetailsForEngagement/{engagementId}
        
        Returns:
            List of BUs with environmentList
        """
        cache_key = "bu_env_details"
        if not force_refresh and cache_key in self._bu_env_details_cache:
            return self._bu_env_details_cache[cache_key]
        
        try:
            engagement_id = await self._get_engagement_id()
            url = f"{self.BASE_URL}/portalservice/securityservice/deptDetailsForEngagement/{engagement_id}"
            headers = await self._get_auth_headers()
            
            logger.info(f"📋 Fetching BU details with environments from: {url}")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, ssl=False) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("status") == "success":
                            bus = data.get("data", {}).get("departmentList", [])
                            self._bu_env_details_cache[cache_key] = bus
                            logger.info(f"✅ Fetched {len(bus)} BUs with environment details")
                            return bus
                    
                    logger.error(f"❌ Failed to fetch BU details: status={response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"❌ Error fetching BU details: {e}")
            return []
    
    async def _list_business_units(
        self,
        params: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        List all Business Units.
        
        User queries:
        - "List all business units"
        - "Show me available BUs"
        - "What business units do I have access to?"
        """
        bus = await self._fetch_business_units(force_refresh=params.get("refresh", False))
        
        if not bus:
            return {
                "success": False,
                "error": "No Business Units found",
                "output": "❌ No Business Units found. Please check your permissions."
            }
        
        # Format output
        output = "## 🏢 Available Business Units\n\n"
        output += f"**Total:** {len(bus)} Business Units\n\n"
        output += "| # | Business Unit Name | ID | Location | Environments | VMs |\n"
        output += "|---|-------------------|-----|----------|--------------|-----|\n"
        
        for idx, bu in enumerate(bus, 1):
            name = bu.get("name", "N/A")
            bu_id = bu.get("id", "N/A")
            location = bu.get("endpoint", {}).get("location", "N/A")
            num_envs = bu.get("noOfEnvironments", 0)
            num_vms = bu.get("noOfVMs", 0)
            output += f"| {idx} | {name} | {bu_id} | {location} | {num_envs} | {num_vms} |\n"
        
        output += "\n---\n"
        output += "\n💡 **Next steps:**\n"
        output += "- To see environments in a BU: `Show environments in IKS_PAAS_BLR_BU`\n"
        output += "- To see role bindings: `List users with role bindings`\n"
        
        return {
            "success": True,
            "data": bus,
            "output": output,
            "metadata": {"bu_count": len(bus)}
        }
    
    async def _show_bu_details(
        self,
        params: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Show details of a specific Business Unit including its environments.
        
        User queries:
        - "Show details of BU IKS_PAAS_BLR_BU"
        - "What environments are in VCD-DELHI-BU?"
        - "Show BU 4854"
        """
        bu_name = params.get("bu_name") or params.get("business_unit_name")
        bu_id = params.get("bu_id") or params.get("business_unit_id")
        
        bus = await self._fetch_bu_with_environments()
        
        if not bus:
            return {
                "success": False,
                "error": "Could not fetch Business Unit details",
                "output": "❌ Could not fetch Business Unit details. Please try again."
            }
        
        # Find the matching BU
        target_bu = None
        
        if bu_id:
            try:
                bu_id = int(bu_id)
                target_bu = next((bu for bu in bus if bu.get("departmentId") == bu_id), None)
            except (ValueError, TypeError):
                pass
        
        if not target_bu and bu_name:
            bu_name_lower = bu_name.lower()
            target_bu = next(
                (bu for bu in bus if bu_name_lower in bu.get("departmentName", "").lower() or
                 bu_name_lower in bu.get("itemName", "").lower()),
                None
            )
        
        if not target_bu:
            # Show available BUs for selection
            return await self._prompt_bu_selection(bus, "show environments")
        
        # Format BU details
        return self._format_bu_details(target_bu)
    
    def _format_bu_details(self, bu: Dict[str, Any]) -> Dict[str, Any]:
        """Format Business Unit details with its environments."""
        bu_name = bu.get("departmentName") or bu.get("itemName", "N/A")
        bu_id = bu.get("departmentId", "N/A")
        endpoint = bu.get("endpointName", "N/A")
        envs = bu.get("environmentList", [])
        
        output = f"## 🏢 Business Unit: {bu_name}\n\n"
        output += f"**BU ID:** {bu_id}\n"
        output += f"**Location:** {endpoint}\n"
        output += f"**Total Environments:** {len(envs)}\n\n"
        
        if envs:
            output += "### 🌐 Environments\n\n"
            output += "| # | Environment Name | ID | Zones |\n"
            output += "|---|-----------------|-----|-------|\n"
            
            for idx, env in enumerate(envs, 1):
                env_name = env.get("environmentName") or env.get("itemName", "N/A")
                env_id = env.get("environmentId", "N/A")
                num_zones = env.get("noOfZones", 0)
                output += f"| {idx} | {env_name} | {env_id} | {num_zones} |\n"
            
            # Show zones for each environment
            output += "\n### 📍 Zone Details\n\n"
            for env in envs:
                env_name = env.get("environmentName", "N/A")
                zones = env.get("zoneList", [])
                if zones:
                    output += f"**{env_name}:**\n"
                    for zone in zones:
                        zone_name = zone.get("zoneName", "N/A")
                        zone_id = zone.get("zoneId", "N/A")
                        output += f"  - {zone_name} (ID: {zone_id})\n"
                    output += "\n"
        else:
            output += "No environments found in this Business Unit.\n"
        
        output += "\n---\n"
        output += f"\n💡 To see role bindings in an environment, use the environment ID:\n"
        output += f"   Example: `List users with role bindings in env {envs[0].get('environmentId') if envs else '5345'}`\n"
        
        return {
            "success": True,
            "data": bu,
            "output": output,
            "metadata": {
                "bu_id": bu_id,
                "bu_name": bu_name,
                "environment_count": len(envs)
            }
        }
    
    async def _prompt_bu_selection(
        self,
        bus: List[Dict[str, Any]],
        action: str = "view"
    ) -> Dict[str, Any]:
        """Prompt user to select a Business Unit."""
        output = f"📋 **Please select a Business Unit to {action}:**\n\n"
        
        bu_options = []
        for idx, bu in enumerate(bus, 1):
            bu_name = bu.get("departmentName") or bu.get("itemName") or bu.get("name", "Unknown")
            bu_id = bu.get("departmentId") or bu.get("id", "N/A")
            location = bu.get("endpointName") or bu.get("endpoint", {}).get("location", "")
            output += f"{idx}. **{bu_name}** (ID: {bu_id}) - {location}\n"
            bu_options.append({"name": bu_name, "id": bu_id})
        
        output += f"\n💡 Reply with the BU number or name (e.g., \"1\" or \"{bus[0].get('departmentName', 'BU-name')}\")"
        
        return {
            "success": True,
            "output": output,
            "awaiting_selection": True,
            "set_filter_state": True,
            "filter_type_for_state": "rbac_bu",
            "filter_options_for_state": bu_options,
            "metadata": {
                "awaiting_filter_selection": True,
                "filter_type": "rbac_bu"
            }
        }
    
    async def _prompt_bu_selection_with_matches(
        self,
        matching_bus: List[Dict[str, Any]],
        search_term: str,
        action: str = "view"
    ) -> Dict[str, Any]:
        """
        Prompt user to select from matching Business Units.
        
        Shows BUs that partially/fully match the user's search term.
        """
        output = f"🔍 **Found {len(matching_bus)} Business Units matching '{search_term}':**\n\n"
        output += "| # | Business Unit Name | ID | Location |\n"
        output += "|---|-------------------|----|-----------|\n"
        
        bu_options = []
        for idx, bu in enumerate(matching_bus, 1):
            bu_name = bu.get("departmentName") or bu.get("itemName") or bu.get("name", "Unknown")
            bu_id = bu.get("departmentId") or bu.get("id", "N/A")
            location = bu.get("endpointName") or bu.get("endpoint", {}).get("location", "")
            env_count = len(bu.get("environmentList", []))
            
            output += f"| {idx} | {bu_name} | {bu_id} | {location} |\n"
            bu_options.append({"name": bu_name, "id": bu_id})
        
        output += f"\n💡 **Select a Business Unit** to view its environments."
        output += f"\n   Reply with the **number** or **name** (e.g., \"1\" or \"{matching_bus[0].get('departmentName', 'BU-name')}\")"
        
        return {
            "success": True,
            "output": output,
            "awaiting_selection": True,
            "set_filter_state": True,
            "filter_type_for_state": "rbac_bu",
            "filter_options_for_state": bu_options,
            "metadata": {
                "awaiting_filter_selection": True,
                "filter_type": "rbac_bu",
                "search_term": search_term,
                "match_count": len(matching_bus)
            }
        }
    
    # =========================================================================
    # Environment Operations
    # =========================================================================
    
    async def _list_environments(
        self,
        params: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        List all Environments, optionally filtered by Business Unit.
        
        User queries:
        - "List all environments"
        - "Show environments in IKS_PAAS_BLR_BU"
        - "What environments are available in BU 4854?"
        - "What are the envs available in bu NSX AutoScale BU - DND?"
        """
        bu_name = params.get("bu_name") or params.get("business_unit_name")
        bu_id = params.get("bu_id") or params.get("business_unit_id")
        user_query = context.get("user_query", "")
        
        logger.info(f"🌐 RBAC _list_environments called:")
        logger.info(f"   params: {params}")
        logger.info(f"   user_query: '{user_query}'")
        logger.info(f"   bu_name: {bu_name}, bu_id: {bu_id}")
        
        # Try to extract bu_name from user query if not provided
        if not bu_id and not bu_name and user_query:
            extracted_bu = self._extract_bu_name_from_query(user_query)
            if extracted_bu:
                bu_name = extracted_bu
                logger.info(f"📝 Extracted BU name from query: {bu_name}")
        
        bus = await self._fetch_bu_with_environments()
        
        if not bus:
            return {
                "success": False,
                "error": "Could not fetch environment details",
                "output": "❌ Could not fetch environment details. Please try again."
            }
        
        # Filter by BU if specified
        if bu_id or bu_name:
            # If bu_id provided, try exact match first
            if bu_id:
                try:
                    bu_id = int(bu_id)
                    target_bu = next((bu for bu in bus if bu.get("departmentId") == bu_id), None)
                    if target_bu:
                        return self._format_bu_details(target_bu)
                except (ValueError, TypeError):
                    pass
            
            # If bu_name provided, find all matching BUs (partial/full match)
            if bu_name:
                bu_name_lower = bu_name.lower().strip()
                # Normalize: replace hyphens/underscores with spaces for flexible matching
                bu_name_normalized = bu_name_lower.replace('-', ' ').replace('_', ' ')
                
                matching_bus = []
                for bu in bus:
                    dept_name = bu.get("departmentName", "").lower()
                    item_name = bu.get("itemName", "").lower()
                    dept_normalized = dept_name.replace('-', ' ').replace('_', ' ')
                    item_normalized = item_name.replace('-', ' ').replace('_', ' ')
                    
                    # Check for exact match first
                    if bu_name_lower == dept_name or bu_name_lower == item_name:
                        # Exact match - return directly
                        logger.info(f"✅ Exact BU match found: {bu.get('departmentName')}")
                        return self._format_bu_details(bu)
                    
                    # Check for partial match
                    if (bu_name_lower in dept_name or bu_name_lower in item_name or
                        bu_name_normalized in dept_normalized or bu_name_normalized in item_normalized or
                        dept_name in bu_name_lower or item_name in bu_name_lower):
                        matching_bus.append(bu)
                
                # If we found matching BUs, show them for selection
                if matching_bus:
                    if len(matching_bus) == 1:
                        # Only one match - return directly
                        logger.info(f"✅ Single BU match found: {matching_bus[0].get('departmentName')}")
                        return self._format_bu_details(matching_bus[0])
                    else:
                        # Multiple matches - show list for selection
                        logger.info(f"📋 Found {len(matching_bus)} matching BUs for '{bu_name}'")
                        return await self._prompt_bu_selection_with_matches(matching_bus, bu_name, "view environments")
                else:
                    # No matches found - show all BUs for selection
                    logger.info(f"⚠️ No BU matches found for '{bu_name}', showing all BUs")
                    return await self._prompt_bu_selection(bus, "view environments")
        
        # Show all environments across all BUs
        all_envs = []
        for bu in bus:
            bu_name_str = bu.get("departmentName") or bu.get("itemName", "Unknown")
            bu_id_val = bu.get("departmentId", 0)
            for env in bu.get("environmentList", []):
                env["_bu_name"] = bu_name_str
                env["_bu_id"] = bu_id_val
                all_envs.append(env)
        
        output = "## 🌐 All Available Environments\n\n"
        output += f"**Total:** {len(all_envs)} Environments across {len(bus)} Business Units\n\n"
        output += "| # | Environment Name | Env ID | Business Unit | Zones |\n"
        output += "|---|-----------------|--------|---------------|-------|\n"
        
        for idx, env in enumerate(all_envs, 1):
            env_name = env.get("environmentName") or env.get("itemName", "N/A")
            env_id = env.get("environmentId", "N/A")
            bu_name_str = env.get("_bu_name", "N/A")
            num_zones = env.get("noOfZones", 0)
            
            # Truncate long names
            if len(env_name) > 30:
                env_name = env_name[:27] + "..."
            if len(bu_name_str) > 20:
                bu_name_str = bu_name_str[:17] + "..."
            
            output += f"| {idx} | {env_name} | {env_id} | {bu_name_str} | {num_zones} |\n"
        
        output += "\n---\n"
        output += "\n💡 **To view role bindings in an environment:**\n"
        output += "   `List users with role bindings` (then select environment)\n"
        output += "   or `List role bindings in env 5345`\n"
        
        return {
            "success": True,
            "data": all_envs,
            "output": output,
            "metadata": {"environment_count": len(all_envs), "bu_count": len(bus)}
        }
    
    async def _show_environment_details(
        self,
        params: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Show details of a specific Environment.
        
        User queries:
        - "Show environment IKS_PAAS_UAT_PROD_ENV"
        - "Details of env 5345"
        """
        env_name = params.get("env_name") or params.get("environment_name")
        env_id = params.get("env_id") or params.get("environment_id")
        
        bus = await self._fetch_bu_with_environments()
        
        # Find the environment
        target_env = None
        parent_bu = None
        
        for bu in bus:
            for env in bu.get("environmentList", []):
                if env_id:
                    try:
                        if env.get("environmentId") == int(env_id):
                            target_env = env
                            parent_bu = bu
                            break
                    except (ValueError, TypeError):
                        pass
                
                if env_name:
                    env_name_lower = env_name.lower()
                    check_name = (env.get("environmentName") or env.get("itemName", "")).lower()
                    if env_name_lower in check_name:
                        target_env = env
                        parent_bu = bu
                        break
            
            if target_env:
                break
        
        if not target_env:
            # Prompt for environment selection
            return await self._prompt_environment_selection(params, context)
        
        # Format environment details
        return self._format_environment_details(target_env, parent_bu)
    
    def _format_environment_details(
        self,
        env: Dict[str, Any],
        parent_bu: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Format environment details."""
        env_name = env.get("environmentName") or env.get("itemName", "N/A")
        env_id = env.get("environmentId", "N/A")
        zones = env.get("zoneList", [])
        bu_name = parent_bu.get("departmentName") or parent_bu.get("itemName", "N/A")
        bu_id = parent_bu.get("departmentId", "N/A")
        
        output = f"## 🌐 Environment: {env_name}\n\n"
        output += f"**Environment ID:** {env_id}\n"
        output += f"**Business Unit:** {bu_name} (ID: {bu_id})\n"
        output += f"**Location:** {parent_bu.get('endpointName', 'N/A')}\n"
        output += f"**Total Zones:** {len(zones)}\n\n"
        
        if zones:
            output += "### 📍 Zones\n\n"
            output += "| # | Zone Name | Zone ID |\n"
            output += "|---|-----------|----------|\n"
            
            for idx, zone in enumerate(zones, 1):
                zone_name = zone.get("zoneName") or zone.get("itemName", "N/A")
                zone_id = zone.get("zoneId", "N/A")
                output += f"| {idx} | {zone_name} | {zone_id} |\n"
        else:
            output += "No zones configured in this environment.\n"
        
        output += "\n---\n"
        output += f"\n💡 **To view role bindings in this environment:**\n"
        output += f"   `List users with role bindings in env {env_id}`\n"
        
        return {
            "success": True,
            "data": {"environment": env, "business_unit": parent_bu},
            "output": output,
            "metadata": {
                "env_id": env_id,
                "env_name": env_name,
                "bu_id": bu_id,
                "zone_count": len(zones)
            }
        }
    
    async def _prompt_environment_selection(
        self,
        params: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prompt user to select an environment.
        Fetches available environments and returns them as options.
        """
        try:
            logger.info("🔍 Fetching available environments for RBAC...")
            
            bus = await self._fetch_bu_with_environments(force_refresh=True)
            
            if not bus:
                return {
                    "success": False,
                    "error": "Could not fetch environments",
                    "output": "❌ Could not fetch environments. Please try again."
                }
            
            # Flatten all environments
            all_envs = []
            for bu in bus:
                for env in bu.get("environmentList", []):
                    all_envs.append({
                        "name": env.get("environmentName") or env.get("itemName", "Unknown"),
                        "id": env.get("environmentId"),
                        "bu_name": bu.get("departmentName") or bu.get("itemName", "")
                    })
            
            logger.info(f"📋 Found {len(all_envs)} environments")
            
            if not all_envs:
                return {
                    "success": False,
                    "error": "No environments found",
                    "output": "❌ No environments found in your account."
                }
            
            # Format environment options (without IDs - user selects by number or name)
            output = "📋 **Please select an environment to view role bindings:**\n\n"
            
            for idx, env in enumerate(all_envs, 1):
                output += f"{idx}. {env['name']}\n"
            
            first_env_name = all_envs[0].get('name', 'env-name')
            output += f"\n💡 Reply with the environment **number** or **name** (e.g., \"1\" or \"{first_env_name}\")"
            
            return {
                "success": True,
                "output": output,
                "awaiting_selection": True,
                "set_filter_state": True,
                "filter_type_for_state": "rbac_env",
                "filter_options_for_state": [
                    {"name": env["name"], "id": env["id"]} 
                    for env in all_envs
                ],
                "metadata": {
                    "awaiting_filter_selection": True,
                    "filter_type": "rbac_env"
                }
            }
            
        except Exception as e:
            logger.error(f"Error prompting for environment selection: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "output": f"❌ Error fetching environments: {str(e)}"
            }
    
    # =========================================================================
    # Cluster Operations
    # =========================================================================
    # Role Binding Operations
    # =========================================================================
    
    async def _list_role_bindings(
        self,
        params: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        List all user role bindings in an environment.
        
        API: POST /paasservice/paas/getIksClusterRoleBindingInEnv/{envId}
        Payload: {"user": " "}  (empty for all users)
        
        User queries:
        - "List all users with their role bindings"
        - "Show role bindings in env 5345"
        - "Who has access in IKS_PAAS_BLR_DEMO_ENV?"
        - "tell me the role bindings for the env TCL-IKS-PAAS-DEV-ENV-DND"
        """
        env_id = params.get("env_id") or params.get("environment_id")
        env_name_param = params.get("env_name") or params.get("environment_name")
        username_filter = params.get("username")
        cluster_filter = params.get("cluster_name") or params.get("cluster")
        role_filter = params.get("role")
        user_query = context.get("user_query", "")
        
        logger.info(f"🔐 RBAC _list_role_bindings called:")
        logger.info(f"   params: {params}")
        logger.info(f"   user_query: '{user_query}'")
        logger.info(f"   env_id: {env_id}, env_name_param: {env_name_param}")
        
        # CRITICAL FIX: Validate username_filter is actually a username, not a sentence
        # Valid usernames are typically short (< 50 chars) and don't contain spaces or query words
        if username_filter:
            query_words = ['list', 'show', 'tell', 'me', 'the', 'role', 'bindings', 'for', 'env', 
                          'environment', 'what', 'access', 'does', 'have', 'user', 'in', 'who', 'has']
            username_lower = username_filter.lower()
            # If it looks like a sentence (has spaces and query words), it's NOT a valid username
            if ' ' in username_filter or any(word in username_lower for word in query_words):
                logger.warning(f"⚠️ Invalid username_filter detected (looks like a query): '{username_filter}' - ignoring")
                username_filter = None
        
        # Validate role_filter - only accept valid roles
        valid_roles = ["admin", "exceptdelete", "customer-namespace-access"]
        if role_filter and role_filter.lower() not in valid_roles:
            logger.info(f"⚠️ Invalid role filter '{role_filter}' ignored (valid: {valid_roles})")
            role_filter = None
        
        # Try to extract env_name from user query if not already set
        if not env_id and not env_name_param and user_query:
            extracted = self._extract_env_name_from_query(user_query)
            if extracted:
                env_name_param = extracted
                logger.info(f"📝 Extracted env name from query: {env_name_param}")
        
        # Validate env_id is a valid number
        if env_id:
            try:
                env_id = int(env_id)
            except (ValueError, TypeError):
                logger.warning(f"⚠️ Invalid env_id received: {env_id}, will try as env_name")
                env_name_param = str(env_id)
                env_id = None
        
        # If we have env_name but not env_id, try to find the environment
        if not env_id and env_name_param:
            env_id = await self._find_env_id_by_name(env_name_param)
            if env_id:
                logger.info(f"✅ Found env_id {env_id} for name '{env_name_param}'")
        
        if not env_id:
            logger.info("📋 No valid env_id provided, fetching available environments...")
            return await self._prompt_environment_selection(params, context)
        
        try:
            url = f"{self.BASE_URL}/paasservice/paas/getIksClusterRoleBindingInEnv/{env_id}"
            headers = await self._get_auth_headers()
            
            payload = {"user": username_filter if username_filter else " "}
            
            logger.info(f"🔐 RBACAgent calling API: POST {url}")
            logger.info(f"   Payload: {payload}")
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload, ssl=False) as response:
                    logger.info(f"   Response status: {response.status}")
                    
                    if response.status == 200:
                        data = await response.json()
                        
                        if data.get("status") == "success":
                            role_bindings = data.get("data", {}).get("data", [])
                            
                            filtered_bindings = self._apply_filters(
                                role_bindings,
                                username_filter=username_filter,
                                cluster_filter=cluster_filter,
                                role_filter=role_filter
                            )
                            
                            # Get environment name for better output
                            env_name = await self._get_environment_name(env_id)
                            
                            formatted_output = self._format_role_bindings_response(
                                filtered_bindings,
                                env_id,
                                env_name,
                                username_filter,
                                cluster_filter,
                                role_filter
                            )
                            
                            return {
                                "success": True,
                                "data": filtered_bindings,
                                "output": formatted_output,
                                "metadata": {
                                    "env_id": env_id,
                                    "env_name": env_name,
                                    "total_users": len(filtered_bindings),
                                    "filters_applied": {
                                        "username": username_filter,
                                        "cluster": cluster_filter,
                                        "role": role_filter
                                    }
                                }
                            }
                        else:
                            return {
                                "success": False,
                                "error": data.get("message", "Unknown error"),
                                "output": f"Failed to fetch role bindings: {data.get('message')}"
                            }
                    else:
                        return {
                            "success": False,
                            "error": f"API returned status {response.status}",
                            "output": f"Failed to fetch role bindings. API returned status {response.status}"
                        }
                        
        except Exception as e:
            logger.error(f"Error fetching role bindings: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "output": f"Error fetching role bindings: {str(e)}"
            }
    
    async def _get_environment_name(self, env_id: int) -> str:
        """Get environment name by ID."""
        try:
            bus = await self._fetch_bu_with_environments()
            for bu in bus:
                for env in bu.get("environmentList", []):
                    if env.get("environmentId") == env_id:
                        return env.get("environmentName") or env.get("itemName", f"Environment {env_id}")
        except Exception:
            pass
        return f"Environment {env_id}"
    
    async def _find_env_id_by_name(self, env_name: str) -> Optional[int]:
        """
        Find environment ID by name (partial match supported).
        
        Args:
            env_name: Environment name to search for
            
        Returns:
            Environment ID if found, None otherwise
        """
        try:
            logger.info(f"🔍 Looking up env_id for name: '{env_name}'")
            bus = await self._fetch_bu_with_environments()
            
            if not bus:
                logger.error("❌ No BUs fetched - cannot look up environment")
                return None
            
            env_name_lower = env_name.lower().strip()
            # Also normalize hyphens and underscores
            env_name_normalized = env_name_lower.replace('-', '_').replace(' ', '_')
            
            logger.info(f"   Searching across {len(bus)} BUs...")
            
            # First try exact match (case-insensitive)
            for bu in bus:
                for env in bu.get("environmentList", []):
                    check_name = (env.get("environmentName") or env.get("itemName", "")).lower().strip()
                    check_name_normalized = check_name.replace('-', '_').replace(' ', '_')
                    
                    if check_name == env_name_lower or check_name_normalized == env_name_normalized:
                        env_id = env.get("environmentId")
                        logger.info(f"✅ EXACT match: '{env_name}' → '{check_name}' (ID: {env_id})")
                        return env_id
            
            # Then try partial match
            for bu in bus:
                for env in bu.get("environmentList", []):
                    check_name = (env.get("environmentName") or env.get("itemName", "")).lower().strip()
                    check_name_normalized = check_name.replace('-', '_').replace(' ', '_')
                    
                    if (env_name_lower in check_name or check_name in env_name_lower or
                        env_name_normalized in check_name_normalized or check_name_normalized in env_name_normalized):
                        env_id = env.get("environmentId")
                        logger.info(f"✅ Partial match: '{env_name}' → '{check_name}' (ID: {env_id})")
                        return env_id
            
            logger.warning(f"⚠️ No environment found matching '{env_name}'")
            return None
            
        except Exception as e:
            logger.error(f"Error finding env by name: {e}", exc_info=True)
            return None
    
    def _extract_bu_name_from_query(self, query: str) -> Optional[str]:
        """
        Extract business unit name from user query.
        
        Examples:
        - "What are the environments available in business unit NSX AutoScale BU - DND?" → "NSX AutoScale BU - DND"
        - "What are the envs available in bu NSX AutoScale BU - DND?" → "NSX AutoScale BU - DND"
        - "Show environments in IKS_PAAS_BLR_BU" → "IKS_PAAS_BLR_BU"
        - "List envs in VCD-DELHI-BU" → "VCD-DELHI-BU"
        """
        logger.info(f"🔍 Extracting BU name from query: '{query}'")
        
        patterns = [
            # Pattern 1: "business unit NAME" (with spaces, hyphens)
            r"business\s+unit\s+([A-Za-z][A-Za-z0-9_\-\s]+?)(?:\s*\?|\s*$)",
            # Pattern 2: "bu NAME" (with spaces, hyphens)
            r"\bbu\s+([A-Za-z][A-Za-z0-9_\-\s]+?)(?:\s*\?|\s*$)",
            # Pattern 3: "in BU_NAME" where name has underscores
            r"\bin\s+([A-Z][A-Za-z0-9_]+(?:_[A-Z0-9]+)*)\b",
            # Pattern 4: "in BU-NAME" where name has hyphens  
            r"\bin\s+([A-Z][A-Za-z0-9\-]+(?:-[A-Z0-9]+)+)\b",
            # Pattern 5: All caps with underscores ending in BU (like IKS_PAAS_BLR_BU)
            r"\b([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*_BU)\b",
            # Pattern 6: Pattern with hyphens ending in BU (like VCD-DELHI-BU)
            r"\b([A-Z][A-Z0-9]*(?:-[A-Z0-9]+)*-BU)\b",
        ]
        
        # Common words to filter out
        stop_words = {'the', 'all', 'list', 'show', 'get', 'what', 'are', 'available', 
                      'envs', 'environments', 'in', 'for', 'tell', 'me'}
        
        for i, pattern in enumerate(patterns):
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                bu_name = match.group(1).strip()
                # Remove trailing punctuation
                bu_name = bu_name.rstrip('?.,!')
                # Check it's not a stop word
                if bu_name.lower() in stop_words:
                    continue
                # Must be at least 3 chars
                if len(bu_name) >= 3:
                    logger.info(f"✅ Extracted BU name '{bu_name}' using pattern {i+1}")
                    return bu_name
        
        logger.info(f"⚠️ Could not extract BU name from query")
        return None
    
    def _extract_env_name_from_query(self, query: str) -> Optional[str]:
        """
        Extract environment name from user query.
        
        Examples:
        - "tell me the role bindings for the env TCL-IKS-PAAS-DEV-ENV-DND" → "TCL-IKS-PAAS-DEV-ENV-DND"
        - "show role bindings in IKS_PAAS_BLR_DEMO_ENV" → "IKS_PAAS_BLR_DEMO_ENV"
        - "role bindings for environment Vayu cloud demo env" → "Vayu cloud demo env"
        - "tell me the role bindings for env TCL-IKS-PAAS-UAT-PROD-ENV-DND" → "TCL-IKS-PAAS-UAT-PROD-ENV-DND"
        """
        logger.info(f"🔍 Extracting env name from query: '{query}'")
        
        patterns = [
            # Pattern 1: "env NAME" or "environment NAME" with complex names
            r"(?:for\s+(?:the\s+)?)?(?:env|environment)\s+([A-Za-z][A-Za-z0-9_\-]+(?:[_\-][A-Za-z0-9]+)*)",
            # Pattern 2: "in ENV_NAME" where name has underscores/hyphens
            r"\bin\s+([A-Z][A-Za-z0-9_\-]+(?:[_\-][A-Z0-9]+)+)",
            # Pattern 3: "bindings in/for NAME" 
            r"bindings\s+(?:in|for)\s+([A-Za-z][A-Za-z0-9_\-\s]+?)(?:\s*$|\s+(?:env|environment|for|with))",
            # Pattern 4: quoted env names
            r"[\"']([^\"']+)[\"']",
            # Pattern 5: All caps/mixed case with underscores (like IKS_PAAS_BLR_DEMO_ENV)
            r"\b([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)\b",
            # Pattern 6: All caps/mixed case with hyphens (like TCL-IKS-PAAS-DEV-ENV-DND)
            r"\b([A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+)\b",
        ]
        
        # Common words/phrases to filter out
        stop_words = {'the', 'all', 'list', 'show', 'get', 'role', 'bindings', 'users', 
                      'tell', 'me', 'what', 'are', 'for', 'in', 'with', 'access'}
        
        for i, pattern in enumerate(patterns):
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                env_name = match.group(1).strip()
                # Check it's not a stop word
                if env_name.lower() in stop_words:
                    continue
                # Remove trailing common words
                env_name = re.sub(r'\s+(role|bindings|users|list|show|tell|me)$', '', env_name, flags=re.IGNORECASE)
                env_name = env_name.strip()
                
                # Must be at least 3 chars and not a common word
                if len(env_name) >= 3 and env_name.lower() not in stop_words:
                    logger.info(f"✅ Extracted env name '{env_name}' using pattern {i+1}")
                    return env_name
        
        logger.info(f"⚠️ Could not extract env name from query")
        return None
    
    async def _get_user_access(
        self,
        params: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get specific user's access permissions.
        
        User queries:
        - "What access does user aramalin have?"
        - "Show role bindings for aramalin"
        - "Check user aramalin's permissions in env 5345"
        """
        env_id = params.get("env_id") or params.get("environment_id")
        username = params.get("username") or params.get("user")
        
        # Validate env_id
        if env_id:
            try:
                env_id = int(env_id)
            except (ValueError, TypeError):
                logger.warning(f"⚠️ Invalid env_id received: {env_id}")
                env_id = None
        
        if not env_id:
            logger.info("📋 No valid env_id for user access check")
            return await self._prompt_environment_selection(params, context)
        
        if not username:
            return {
                "success": False,
                "error": "Username is required",
                "output": "❌ Please provide a username to check access for.\n\n💡 Example: `What access does user aramalin have?`",
                "needs_params": ["username"]
            }
        
        result = await self._list_role_bindings(
            {"env_id": env_id, "username": username},
            context
        )
        
        if result.get("success") and result.get("data"):
            user_data = result["data"]
            if len(user_data) > 0:
                user_info = user_data[0]
                env_name = await self._get_environment_name(env_id)
                formatted_output = self._format_user_access_response(user_info, username, env_id, env_name)
                result["output"] = formatted_output
            else:
                result["output"] = f"No access found for user '{username}' in environment {env_id}."
        
        return result
    
    # =========================================================================
    # Formatting Helpers
    # =========================================================================
    
    def _apply_filters(
        self,
        role_bindings: List[Dict[str, Any]],
        username_filter: Optional[str] = None,
        cluster_filter: Optional[str] = None,
        role_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Apply filters to role bindings data."""
        filtered = role_bindings
        
        # Only apply username filter if it looks like a username (not common words)
        if username_filter:
            username_lower = username_filter.lower()
            # Skip common words that shouldn't be usernames
            skip_words = ['all', 'users', 'user', 'role', 'bindings', 'list', 'the', 'their', 'with']
            if username_lower not in skip_words:
                filtered = [
                    rb for rb in filtered
                    if username_lower in rb.get("username", "").lower()
                ]
        
        if cluster_filter:
            cluster_lower = cluster_filter.lower()
            filtered = [
                rb for rb in filtered
                if cluster_lower in rb.get("clusterRoles", "").lower()
            ]
        
        # Only apply role filter if it's a valid role type
        valid_roles = ["admin", "exceptdelete", "customer-namespace-access"]
        if role_filter and role_filter.lower() in valid_roles:
            role_lower = role_filter.lower()
            filtered = [
                rb for rb in filtered
                if role_lower in rb.get("clusterRoles", "").lower()
            ]
        
        return filtered
    
    def _parse_cluster_roles(self, cluster_roles_str: str) -> List[Dict[str, str]]:
        """
        Parse cluster roles string into structured list.
        
        Input: "demo-caas:admin, blr-paas:exceptdelete"
        Output: [{"cluster": "demo-caas", "role": "admin"}, ...]
        """
        if not cluster_roles_str:
            return []
        
        roles = []
        pairs = cluster_roles_str.split(", ")
        
        for pair in pairs:
            if ":" in pair:
                parts = pair.strip().split(":")
                if len(parts) == 2:
                    roles.append({
                        "cluster": parts[0].strip(),
                        "role": parts[1].strip()
                    })
        
        return roles
    
    def _format_role_bindings_response(
        self,
        role_bindings: List[Dict],
        env_id: int,
        env_name: str,
        username_filter: Optional[str],
        cluster_filter: Optional[str],
        role_filter: Optional[str]
    ) -> str:
        """Format role bindings response as markdown."""
        if not role_bindings:
            filters_text = []
            if username_filter:
                filters_text.append(f"username='{username_filter}'")
            if cluster_filter:
                filters_text.append(f"cluster='{cluster_filter}'")
            if role_filter:
                filters_text.append(f"role='{role_filter}'")
            
            filter_str = " with filters: " + ", ".join(filters_text) if filters_text else ""
            return f"No role bindings found in {env_name} (ID: {env_id}){filter_str}."
        
        output = f"## 🔐 RBAC Role Bindings\n\n"
        output += f"**Environment:** {env_name} (ID: {env_id})\n"
        
        if username_filter or cluster_filter or role_filter:
            output += "\n**Applied Filters:**\n"
            if username_filter:
                output += f"- Username: `{username_filter}`\n"
            if cluster_filter:
                output += f"- Cluster: `{cluster_filter}`\n"
            if role_filter:
                output += f"- Role: `{role_filter}`\n"
        
        output += f"\n**Total Users:** {len(role_bindings)}\n\n"
        
        # Summary table
        output += "| # | Username | Clusters & Roles | Last Sync |\n"
        output += "|---|----------|------------------|------------|\n"
        
        for idx, rb in enumerate(role_bindings, 1):
            username = rb.get("username", "N/A")
            cluster_roles = rb.get("clusterRoles", "N/A")
            last_sync = rb.get("lastSyncTime", "N/A")
            
            if len(cluster_roles) > 45:
                cluster_roles_display = cluster_roles[:42] + "..."
            else:
                cluster_roles_display = cluster_roles
            
            if last_sync and last_sync != "N/A":
                try:
                    dt = datetime.fromisoformat(last_sync.replace("Z", "+00:00"))
                    last_sync = dt.strftime("%Y-%m-%d %H:%M")
                except:
                    pass
            
            output += f"| {idx} | {username} | {cluster_roles_display} | {last_sync} |\n"
        
        # Detailed breakdown
        output += "\n---\n\n### 📋 Detailed User Access\n\n"
        
        for rb in role_bindings:
            username = rb.get("username", "N/A")
            cluster_roles_str = rb.get("clusterRoles", "")
            parsed_roles = self._parse_cluster_roles(cluster_roles_str)
            
            if parsed_roles:
                output += f"**{username}:**\n"
                
                admin_clusters = [r["cluster"] for r in parsed_roles if r["role"] == "admin"]
                exceptdelete_clusters = [r["cluster"] for r in parsed_roles if r["role"] == "exceptdelete"]
                namespace_clusters = [r["cluster"] for r in parsed_roles if r["role"] == "customer-namespace-access"]
                other_clusters = [f"{r['cluster']}:{r['role']}" for r in parsed_roles 
                                 if r["role"] not in ["admin", "exceptdelete", "customer-namespace-access"]]
                
                if admin_clusters:
                    output += f"  - 🔴 **Admin** ({len(admin_clusters)}): {', '.join(admin_clusters)}\n"
                if exceptdelete_clusters:
                    output += f"  - 🟡 **Except Delete** ({len(exceptdelete_clusters)}): {', '.join(exceptdelete_clusters)}\n"
                if namespace_clusters:
                    output += f"  - 🟢 **Namespace Access** ({len(namespace_clusters)}): {', '.join(namespace_clusters)}\n"
                if other_clusters:
                    output += f"  - 🔵 **Other**: {', '.join(other_clusters)}\n"
                
                output += "\n"
        
        return output
    
    def _format_user_access_response(
        self,
        user_info: Dict,
        username: str,
        env_id: int,
        env_name: str
    ) -> str:
        """Format single user's access as detailed markdown."""
        output = f"## 🔐 Access Details for User: `{username}`\n\n"
        output += f"**Environment:** {env_name} (ID: {env_id})\n"
        
        last_sync = user_info.get("lastSyncTime", "N/A")
        assigned_by = user_info.get("assignedBy") or "system"
        cluster_roles_str = user_info.get("clusterRoles", "")
        
        if last_sync and last_sync != "N/A":
            try:
                dt = datetime.fromisoformat(last_sync.replace("Z", "+00:00"))
                last_sync = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
            except:
                pass
        
        output += f"**Last Synced:** {last_sync}\n"
        output += f"**Assigned By:** {assigned_by}\n\n"
        
        parsed_roles = self._parse_cluster_roles(cluster_roles_str)
        
        if not parsed_roles:
            output += "No cluster roles assigned.\n"
            return output
        
        output += f"**Total Cluster Access:** {len(parsed_roles)}\n\n"
        
        admin_clusters = [r["cluster"] for r in parsed_roles if r["role"] == "admin"]
        exceptdelete_clusters = [r["cluster"] for r in parsed_roles if r["role"] == "exceptdelete"]
        namespace_clusters = [r["cluster"] for r in parsed_roles if r["role"] == "customer-namespace-access"]
        other_roles = [(r["cluster"], r["role"]) for r in parsed_roles 
                       if r["role"] not in ["admin", "exceptdelete", "customer-namespace-access"]]
        
        output += "### Role Breakdown\n\n"
        
        if admin_clusters:
            output += f"#### 🔴 Admin Access ({len(admin_clusters)} clusters)\n"
            output += "Full administrative access including create, update, and delete operations.\n\n"
            output += "| # | Cluster Name |\n"
            output += "|---|-------------|\n"
            for idx, cluster in enumerate(admin_clusters, 1):
                output += f"| {idx} | {cluster} |\n"
            output += "\n"
        
        if exceptdelete_clusters:
            output += f"#### 🟡 Except Delete Access ({len(exceptdelete_clusters)} clusters)\n"
            output += "Can perform all operations except delete.\n\n"
            output += "| # | Cluster Name |\n"
            output += "|---|-------------|\n"
            for idx, cluster in enumerate(exceptdelete_clusters, 1):
                output += f"| {idx} | {cluster} |\n"
            output += "\n"
        
        if namespace_clusters:
            output += f"#### 🟢 Customer Namespace Access ({len(namespace_clusters)} clusters)\n"
            output += "Access to customer namespaces only.\n\n"
            output += "| # | Cluster Name |\n"
            output += "|---|-------------|\n"
            for idx, cluster in enumerate(namespace_clusters, 1):
                output += f"| {idx} | {cluster} |\n"
            output += "\n"
        
        if other_roles:
            output += f"#### 🔵 Other Roles ({len(other_roles)} clusters)\n\n"
            output += "| # | Cluster Name | Role |\n"
            output += "|---|-------------|------|\n"
            for idx, (cluster, role) in enumerate(other_roles, 1):
                output += f"| {idx} | {cluster} | {role} |\n"
            output += "\n"
        
        return output
    
    # =========================================================================
    # Query Parameter Extraction
    # =========================================================================
    
    def extract_params_from_query(self, user_query: str) -> Dict[str, Any]:
        """
        Extract RBAC-related parameters from user query.
        
        Examples:
        - "Show access for user aramalin" → {"username": "aramalin"}
        - "Who has admin access to blr-paas cluster?" → {"cluster_name": "blr-paas", "role": "admin"}
        - "List environments in IKS_PAAS_BLR_BU" → {"bu_name": "IKS_PAAS_BLR_BU"}
        - "Show env 5345" → {"env_id": 5345}
        """
        params = {}
        query_lower = user_query.lower()
        
        # Extract environment ID
        env_id_patterns = [
            r"env(?:ironment)?\s*(?:id)?\s*[:\s]?\s*(\d+)",
            r"in\s+env\s+(\d+)",
            r"environment\s+(\d+)",
        ]
        
        for pattern in env_id_patterns:
            match = re.search(pattern, user_query, re.IGNORECASE)
            if match:
                params["env_id"] = int(match.group(1))
                break
        
        # Extract BU ID
        bu_id_patterns = [
            r"bu(?:siness\s*unit)?\s*(?:id)?\s*[:\s]?\s*(\d+)",
            r"in\s+bu\s+(\d+)",
            r"department\s*id?\s*(\d+)",
        ]
        
        for pattern in bu_id_patterns:
            match = re.search(pattern, user_query, re.IGNORECASE)
            if match:
                params["bu_id"] = int(match.group(1))
                break
        
        # Extract BU name
        bu_name_patterns = [
            r"(?:bu|business\s*unit)\s+([A-Z][A-Z0-9_-]+(?:\s*-\s*[A-Z0-9_-]+)*)",
            r"in\s+([A-Z][A-Z0-9_-]+(?:[-_][A-Z0-9]+)+)",
            r"environments?\s+(?:in|of|for)\s+([A-Z][A-Z0-9_-]+)",
        ]
        
        for pattern in bu_name_patterns:
            match = re.search(pattern, user_query, re.IGNORECASE)
            if match:
                params["bu_name"] = match.group(1)
                break
        
        # Extract environment name - improved patterns
        env_name_patterns = [
            # "for the env TCL-IKS-PAAS-DEV-ENV-DND"
            r"(?:for\s+(?:the\s+)?)?env(?:ironment)?\s+([A-Za-z][A-Za-z0-9_\-]+(?:[-_][A-Za-z0-9]+)*)",
            # "in IKS_PAAS_BLR_DEMO_ENV"
            r"in\s+([A-Z][A-Z0-9_-]+(?:[-_][A-Z0-9]+)+)",
            # "show TCL-IKS-PAAS-DEV-ENV-DND"
            r"show\s+([A-Z][A-Z0-9_-]+(?:[-_][A-Z0-9]+)+)",
            # Names ending with ENV or DND
            r"\b([A-Z][A-Z0-9_\-]+(?:ENV|DND))\b",
        ]
        
        for pattern in env_name_patterns:
            match = re.search(pattern, user_query, re.IGNORECASE)
            if match:
                potential_name = match.group(1)
                # Filter out common words
                if potential_name.lower() not in ['env', 'environment', 'the', 'for', 'in', 'show', 'list']:
                    params["env_name"] = potential_name
                    break
        
        # Extract username patterns
        username_patterns = [
            r"user\s+([a-zA-Z0-9._@-]+)",
            r"for\s+user\s+([a-zA-Z0-9._@-]+)",
            r"access\s+for\s+([a-zA-Z0-9._@-]+)",
            r"'([a-zA-Z0-9._@-]+)'s?\s+access",
            r"username\s+([a-zA-Z0-9._@-]+)",
            r"does\s+(?:user\s+)?([a-zA-Z0-9._@-]+)\s+have",
        ]
        
        for pattern in username_patterns:
            match = re.search(pattern, user_query, re.IGNORECASE)
            if match:
                potential_username = match.group(1)
                # Filter out common words
                if potential_username.lower() not in ['the', 'a', 'an', 'in', 'on', 'for', 'to', 'all']:
                    params["username"] = potential_username
                    break
        
        # Extract cluster name patterns
        cluster_patterns = [
            r"cluster\s+([a-zA-Z0-9-_]+)",
            r"to\s+([a-zA-Z0-9-_]+)\s+cluster",
            r"on\s+([a-zA-Z0-9-_]+)\s+cluster",
        ]
        
        for pattern in cluster_patterns:
            match = re.search(pattern, user_query, re.IGNORECASE)
            if match:
                params["cluster_name"] = match.group(1)
                break
        
        # Extract role type
        if "admin" in query_lower and "except" not in query_lower:
            params["role"] = "admin"
        elif "exceptdelete" in query_lower or "except delete" in query_lower:
            params["role"] = "exceptdelete"
        elif "namespace" in query_lower:
            params["role"] = "customer-namespace-access"
        
        return params
