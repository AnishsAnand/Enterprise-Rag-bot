"""
Context Switch Handler - Handles user requests to change/switch context entities.

Supports switching:
- Engagement (for ENG users with multiple engagements)
- Datacenter/Endpoint
- Cluster
- Firewall
- Business Unit
- Environment
- Zone

When switching context, dependent parameters are automatically cleared.
For example, switching datacenter clears cluster and firewall selections.
"""

import logging
from typing import Any, Dict, List, Optional

from app.agents.state.conversation_state import ConversationState, ConversationStatus
from app.services.api_executor_service import api_executor_service

# Import UserContextService for persistent storage
try:
    from app.services.user_context_service import user_context_service
    USER_CONTEXT_SERVICE_AVAILABLE = True
except ImportError:
    USER_CONTEXT_SERVICE_AVAILABLE = False

logger = logging.getLogger(__name__)


class ContextSwitchHandler:
    """
    Handler for context switching operations.
    
    Manages switching between different entities (engagement, datacenter, etc.)
    and handles the cascade clearing of dependent parameters.
    """
    
    # Entity dependency hierarchy - when switching an entity, clear all dependents
    ENTITY_DEPENDENCIES = {
        "engagement": ["datacenter", "cluster", "firewall", "business_unit", "environment", "zone"],
        "datacenter": ["cluster", "firewall"],
        "business_unit": ["environment", "zone"],
        "environment": ["zone"],
        "cluster": [],
        "firewall": [],
        "zone": [],
    }
    
    # Mapping of entity types to their display names
    ENTITY_DISPLAY_NAMES = {
        "engagement": "Engagement",
        "datacenter": "Datacenter",
        "cluster": "Cluster",
        "firewall": "Firewall",
        "business_unit": "Business Unit",
        "environment": "Environment",
        "zone": "Zone",
    }
    
    def __init__(self):
        """Initialize the context switch handler."""
        self.api_executor = api_executor_service
        logger.info("✅ ContextSwitchHandler initialized")
    
    async def handle_switch(
        self,
        state: ConversationState,
        entity_type: str,
        target_value: Optional[str] = None,
        auth_token: str = None
    ) -> Dict[str, Any]:
        """
        Handle a context switch request.
        
        Args:
            state: Current conversation state
            entity_type: Type of entity to switch (engagement, datacenter, etc.)
            target_value: Target value to switch to (e.g., "Mumbai", "Chennai")
            auth_token: Bearer token for API calls
            
        Returns:
            Response dict with success status and message
        """
        logger.info(f"🔄 Handling context switch: {entity_type} -> {target_value}")
        
        # Validate entity type
        if entity_type not in self.ENTITY_DEPENDENCIES:
            return {
                "success": False,
                "response": f"❌ Unknown entity type: {entity_type}. Supported types: {', '.join(self.ENTITY_DEPENDENCIES.keys())}"
            }
        
        # Route to specific handler
        handler_map = {
            "engagement": self._switch_engagement,
            "datacenter": self._switch_datacenter,
            "cluster": self._switch_cluster,
            "firewall": self._switch_firewall,
            "business_unit": self._switch_business_unit,
            "environment": self._switch_environment,
            "zone": self._switch_zone,
        }
        
        handler = handler_map.get(entity_type)
        if handler:
            return await handler(state, target_value, auth_token)
        
        return {
            "success": False,
            "response": f"❌ No handler for entity type: {entity_type}"
        }
    
    async def _switch_engagement(
        self,
        state: ConversationState,
        target_value: Optional[str],
        auth_token: str
    ) -> Dict[str, Any]:
        """
        Switch engagement for ENG users.
        
        If target_value is provided, try to match it to an engagement.
        Otherwise, show the list of available engagements for selection.
        """
        user_id = state.user_id
        
        # Get current engagement info for context
        current_engagement_id = state.selected_engagement_id
        current_engagement_name = None
        if current_engagement_id:
            session = await self.api_executor._get_user_session(user_id)
            if session and session.get("engagement_data"):
                current_engagement_name = session["engagement_data"].get("engagementName")
        
        # Fetch available engagements
        engagements = await self.api_executor.get_engagements_list(
            auth_token=auth_token,
            user_id=user_id
        )
        
        if not engagements:
            return {
                "success": False,
                "response": "❌ Failed to fetch engagements. Please try again."
            }
        
        # If only one engagement, can't switch
        if len(engagements) == 1:
            return {
                "success": False,
                "response": f"ℹ️ You only have one engagement: **{engagements[0].get('engagementName')}**. No switching needed."
            }
        
        # If target_value provided, try to match
        if target_value:
            matched = None
            for eng in engagements:
                eng_name = eng.get("engagementName", "").lower()
                if target_value.lower() in eng_name or eng_name in target_value.lower():
                    matched = eng
                    break
            
            if matched:
                # Switch to matched engagement
                new_engagement_id = matched.get("id")
                new_engagement_name = matched.get("engagementName")
                
                # Clear dependent parameters
                await self._clear_dependents(state, user_id, "engagement")
                
                # Update state
                state.selected_engagement_id = new_engagement_id
                
                # Update API executor session
                await self.api_executor.set_engagement_id(
                    user_id=user_id,
                    engagement_id=new_engagement_id,
                    engagement_data=matched,
                    save_as_default=True
                )
                
                response = f"✅ **Switched engagement**\n\n"
                if current_engagement_name:
                    response += f"📤 From: {current_engagement_name}\n"
                response += f"📥 To: **{new_engagement_name}**\n\n"
                response += "ℹ️ Your datacenter, cluster, and other selections have been cleared. Please select them again for your next operation."
                
                return {
                    "success": True,
                    "response": response,
                    "switched_to": {
                        "engagement_id": new_engagement_id,
                        "engagement_name": new_engagement_name
                    }
                }
            else:
                # No match found, show options
                pass
        
        # Show engagement selection options
        # Mark this as an explicit context-switch flow so the selection handler
        # does NOT try to "continue" with the original query as an operation.
        state.collected_params["_pending_context_switch"] = "engagement"
        state.pending_engagements = engagements
        state.status = ConversationStatus.AWAITING_ENGAGEMENT_SELECTION
        
        response = "🔄 **Select an engagement to switch to:**\n\n"
        for i, eng in enumerate(engagements, 1):
            eng_name = eng.get("engagementName", "Unknown")
            eng_id = eng.get("id")
            is_current = " *(current)*" if eng_id == current_engagement_id else ""
            response += f"{i}. **{eng_name}**{is_current}\n"
        
        response += "\n💡 Reply with the number or name of the engagement you want to switch to."
        
        return {
            "success": True,
            "response": response,
            "awaiting_selection": True,
            "options": engagements
        }
    
    async def _switch_datacenter(
        self,
        state: ConversationState,
        target_value: Optional[str],
        auth_token: str
    ) -> Dict[str, Any]:
        """
        Switch datacenter/endpoint.
        
        If target_value is provided, try to match it to a datacenter.
        Otherwise, show the list of available datacenters for selection.
        """
        user_id = state.user_id
        
        # Get current datacenter info
        current_dc_name = state.collected_params.get("_default_datacenter_name")
        current_dc_id = state.collected_params.get("_default_datacenter_id")
        
        # Fetch available endpoints
        endpoints_result = await self.api_executor.list_endpoints(
            auth_token=auth_token,
            user_id=user_id
        )
        
        if endpoints_result.get("success"):
            data = endpoints_result.get("data", {})
            endpoints = data.get("endpoints", []) if isinstance(data, dict) else data
        else:
            endpoints = None
        
        if not endpoints:
            # Check if it's an auth issue
            error_msg = endpoints_result.get("error", "") if endpoints_result else ""
            if "engagement" in error_msg.lower() or "auth" in error_msg.lower():
                return {
                    "success": False,
                    "response": "❌ Unable to fetch datacenters. Please ensure you have selected an engagement first.\n\n💡 Try: `list endpoints` or `show datacenters` to see available options."
                }
            return {
                "success": False,
                "response": "❌ Failed to fetch datacenters. Please try again."
            }
        
        # If target_value provided, try to match
        if target_value:
            matched = None
            target_lower = target_value.lower()
            for ep in endpoints:
                # The formatted endpoint has: id, name, type, region, status
                ep_name = ep.get("name", "").lower()
                ep_region = ep.get("region", "").lower()
                if (target_lower in ep_name or 
                    ep_name in target_lower or
                    target_lower in ep_region):
                    matched = ep
                    break
            
            if matched:
                # Switch to matched datacenter
                new_dc_id = matched.get("id")
                new_dc_name = matched.get("name")
                
                # Clear dependent parameters
                await self._clear_dependents(state, user_id, "datacenter")
                
                # Update state
                state.collected_params["_default_datacenter_id"] = new_dc_id
                state.collected_params["_default_datacenter_name"] = new_dc_name
                state.collected_params["_default_endpoint_ids"] = [new_dc_id]
                
                # Save to persistent storage
                if USER_CONTEXT_SERVICE_AVAILABLE:
                    user_context_service.set_datacenter(
                        user_id=user_id,
                        datacenter_id=new_dc_id,
                        datacenter_name=new_dc_name,
                        endpoint_ids=[new_dc_id],
                        save_as_default=True
                    )
                
                response = f"✅ **Switched datacenter**\n\n"
                if current_dc_name:
                    response += f"📤 From: {current_dc_name}\n"
                response += f"📥 To: **{new_dc_name}**\n\n"
                response += "ℹ️ Your cluster and firewall selections have been cleared."
                
                return {
                    "success": True,
                    "response": response,
                    "switched_to": {
                        "datacenter_id": new_dc_id,
                        "datacenter_name": new_dc_name
                    }
                }
        
        # Show datacenter selection options
        response = "🔄 **Select a datacenter to switch to:**\n\n"
        for i, ep in enumerate(endpoints, 1):
            ep_name = ep.get("name", "Unknown")
            ep_id = ep.get("id")
            ep_region = ep.get("region", "")
            region_info = f" ({ep_region})" if ep_region else ""
            is_current = " *(current)*" if ep_id == current_dc_id else ""
            response += f"{i}. **{ep_name}**{region_info}{is_current}\n"
        
        response += "\n💡 Reply with the number or name of the datacenter you want to switch to."
        
        # Store pending selection
        state.collected_params["_pending_datacenter_selection"] = endpoints
        state.status = ConversationStatus.AWAITING_SELECTION
        
        return {
            "success": True,
            "response": response,
            "awaiting_selection": True,
            "options": endpoints
        }
    
    async def _switch_cluster(
        self,
        state: ConversationState,
        target_value: Optional[str],
        auth_token: str
    ) -> Dict[str, Any]:
        """Switch cluster context."""
        user_id = state.user_id
        
        if target_value:
            # Update state
            state.collected_params["_default_cluster_name"] = target_value
            
            # Save to persistent storage
            if USER_CONTEXT_SERVICE_AVAILABLE:
                user_context_service.set_cluster(
                    user_id=user_id,
                    cluster_name=target_value,
                    save_as_default=True
                )
            
            return {
                "success": True,
                "response": f"✅ **Set default cluster to:** {target_value}"
            }
        
        return {
            "success": False,
            "response": "❌ Please specify a cluster name. Example: `switch to cluster my-cluster`"
        }
    
    async def _switch_firewall(
        self,
        state: ConversationState,
        target_value: Optional[str],
        auth_token: str
    ) -> Dict[str, Any]:
        """Switch firewall context."""
        user_id = state.user_id
        
        if target_value:
            # Update state
            state.collected_params["_default_firewall_name"] = target_value
            
            # Save to persistent storage
            if USER_CONTEXT_SERVICE_AVAILABLE:
                user_context_service.set_firewall(
                    user_id=user_id,
                    firewall_name=target_value,
                    save_as_default=True
                )
            
            return {
                "success": True,
                "response": f"✅ **Set default firewall to:** {target_value}"
            }
        
        return {
            "success": False,
            "response": "❌ Please specify a firewall name. Example: `switch to firewall my-firewall`"
        }
    
    async def _switch_business_unit(
        self,
        state: ConversationState,
        target_value: Optional[str],
        auth_token: str
    ) -> Dict[str, Any]:
        """Switch business unit context."""
        user_id = state.user_id
        
        if target_value:
            # Clear dependent parameters
            await self._clear_dependents(state, user_id, "business_unit")
            
            # Update state
            state.collected_params["_default_business_unit_name"] = target_value
            
            # Save to persistent storage
            if USER_CONTEXT_SERVICE_AVAILABLE:
                user_context_service.set_business_unit(
                    user_id=user_id,
                    bu_name=target_value,
                    save_as_default=True
                )
            
            return {
                "success": True,
                "response": f"✅ **Set default business unit to:** {target_value}\n\nℹ️ Environment and zone selections have been cleared."
            }
        
        return {
            "success": False,
            "response": "❌ Please specify a business unit name. Example: `switch to BU my-department`"
        }
    
    async def _switch_environment(
        self,
        state: ConversationState,
        target_value: Optional[str],
        auth_token: str
    ) -> Dict[str, Any]:
        """Switch environment context."""
        user_id = state.user_id
        
        if target_value:
            # Clear dependent parameters
            await self._clear_dependents(state, user_id, "environment")
            
            # Update state
            state.collected_params["_default_environment_name"] = target_value
            
            # Save to persistent storage
            if USER_CONTEXT_SERVICE_AVAILABLE:
                user_context_service.set_environment(
                    user_id=user_id,
                    env_name=target_value,
                    save_as_default=True
                )
            
            return {
                "success": True,
                "response": f"✅ **Set default environment to:** {target_value}\n\nℹ️ Zone selection has been cleared."
            }
        
        return {
            "success": False,
            "response": "❌ Please specify an environment name. Example: `switch to environment production`"
        }
    
    async def _switch_zone(
        self,
        state: ConversationState,
        target_value: Optional[str],
        auth_token: str
    ) -> Dict[str, Any]:
        """Switch zone context."""
        user_id = state.user_id
        
        if target_value:
            # Update state
            state.collected_params["_default_zone_name"] = target_value
            
            # Save to persistent storage
            if USER_CONTEXT_SERVICE_AVAILABLE:
                user_context_service.set_zone(
                    user_id=user_id,
                    zone_name=target_value,
                    save_as_default=True
                )
            
            return {
                "success": True,
                "response": f"✅ **Set default zone to:** {target_value}"
            }
        
        return {
            "success": False,
            "response": "❌ Please specify a zone name. Example: `switch to zone my-zone`"
        }
    
    async def _clear_dependents(
        self,
        state: ConversationState,
        user_id: str,
        entity_type: str
    ) -> None:
        """
        Clear dependent parameters when switching an entity.
        
        For example, switching engagement clears datacenter, cluster, etc.
        """
        dependents = self.ENTITY_DEPENDENCIES.get(entity_type, [])
        
        if not dependents:
            return
        
        logger.info(f"🧹 Clearing dependent parameters for {entity_type}: {dependents}")
        
        # Clear from state
        param_mappings = {
            # Datacenter/endpoints (per-chat + legacy defaults)
            "datacenter": [
                "endpoints",
                "endpoint_names",
                "_default_datacenter_id",
                "_default_datacenter_name",
                "_default_endpoint_ids",
                "_pending_datacenter_selection",
            ],
            # Cluster (explicit selection + extracted params)
            "cluster": ["_default_cluster_id", "_default_cluster_name", "cluster_name", "cluster_id"],
            "firewall": ["_default_firewall_id", "_default_firewall_name"],
            "business_unit": ["_default_business_unit_id", "_default_business_unit_name"],
            "environment": ["_default_environment_id", "_default_environment_name"],
            "zone": ["_default_zone_id", "_default_zone_name"],
        }
        
        for dep in dependents:
            params_to_clear = param_mappings.get(dep, [])
            for param in params_to_clear:
                if param in state.collected_params:
                    del state.collected_params[param]

        # Clear internal derived context that depends on engagement
        if entity_type == "engagement":
            state.collected_params.pop("ipc_engagement_id", None)

        # Clear list cache on any context change that affects list scope/results
        if entity_type in {"engagement", "datacenter", "business_unit", "environment", "zone"}:
            state.collected_params.pop("_last_list_cache", None)
            state.collected_params.pop("_last_list_resource_type", None)
            state.collected_params.pop("_last_list_endpoints", None)
            state.collected_params.pop("_last_list_engagement_id", None)

        # Clear any pending selection UI state when context changes
        state.pending_filter_options = None
        state.pending_filter_type = None
        
        # Clear from persistent storage
        if USER_CONTEXT_SERVICE_AVAILABLE:
            if "datacenter" in dependents:
                user_context_service.clear_datacenter(user_id)
            # Note: We don't have individual clear methods for all entities,
            # but the set methods with None values effectively clear them
    
    def get_current_context_summary(self, state: ConversationState) -> str:
        """
        Get a summary of the user's current context settings.
        """
        parts = []
        
        if state.selected_engagement_id:
            parts.append(f"Engagement ID: {state.selected_engagement_id}")
        
        dc_name = state.collected_params.get("_default_datacenter_name")
        if dc_name:
            parts.append(f"Datacenter: {dc_name}")
        
        cluster_name = state.collected_params.get("_default_cluster_name")
        if cluster_name:
            parts.append(f"Cluster: {cluster_name}")
        
        bu_name = state.collected_params.get("_default_business_unit_name")
        if bu_name:
            parts.append(f"BU: {bu_name}")
        
        env_name = state.collected_params.get("_default_environment_name")
        if env_name:
            parts.append(f"Env: {env_name}")
        
        zone_name = state.collected_params.get("_default_zone_name")
        if zone_name:
            parts.append(f"Zone: {zone_name}")
        
        if parts:
            return " | ".join(parts)
        return "No defaults set"


# Global singleton instance
context_switch_handler = ContextSwitchHandler()
