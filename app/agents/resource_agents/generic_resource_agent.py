"""
Generic Resource Agent - Handles resources without specialized agents.
Provides a fallback for endpoint, business_unit, environment, zone, etc.
"""
from typing import Any, Dict, List
import logging
from app.agents.resource_agents.base_resource_agent import BaseResourceAgent
from app.services.api_executor_service import api_executor_service
from app.services.llm_formatter_service import llm_formatter
logger = logging.getLogger(__name__)

class GenericResourceAgent(BaseResourceAgent):
    """
    Generic agent for resources without specialized handling.
    Supports: endpoint, business_unit, environment, zone, and any
    other resource types that use standard CRUD operations.
    """
    # Mapping of resource types to API service methods
    RESOURCE_METHOD_MAP = {
        "endpoint": {
            "list": "list_endpoints",
            "display_name": "Endpoints/Datacenters"
        },
        "business_unit": {
            "list": "get_business_units_list",
            "display_name": "Business Units"
        },
        "environment": {
            "list": "get_environments_list",
            "display_name": "Environments"
        },
        "zone": {
            "list": "get_zones_list",
            "display_name": "Zones"
        }}
    
    def __init__(self, resource_type: str = "generic"):
        super().__init__(
            agent_name="GenericResourceAgent",
            agent_description=(
                "Generic agent for handling resources without specialized agents. "
                "Provides standard CRUD operations with LLM formatting."
            ),
            resource_type=resource_type,
            temperature=0.2)
    
    def get_supported_operations(self) -> List[str]:
        """Get list of supported operations."""
        return ["list", "create", "update", "delete"]
    
    async def execute_operation(self,operation: str,params: Dict[str, Any],context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a generic resource operation.
        Args:
            operation: Operation to perform
            params: Parameters for the operation
            context: Context (session_id, user_query, user_roles, etc.) 
        Returns:
            Dict with success status and formatted response
        """
        try:
            resource_type = context.get("resource_type", self.resource_type)
            logger.info(f"📦 GenericResourceAgent executing: {operation} for {resource_type}")
            
            if operation == "list":
                return await self._list_resource(resource_type, params, context)
            else:
                # Generic CRUD via api_executor_service
                return await self._execute_generic_crud(resource_type, operation, params, context)
        except Exception as e:
            logger.error(f"❌ GenericResourceAgent error: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "response": f"An error occurred while {operation}ing {resource_type}: {str(e)}"}
    
    async def _list_resource(self,resource_type: str,params: Dict[str, Any],context: Dict[str, Any]) -> Dict[str, Any]:
        """
        List resources using the appropriate API method.
        Args:
            resource_type: Type of resource to list
            params: Parameters
            context: Context 
        Returns:
            Formatted list response
        """
        user_roles = context.get("user_roles", [])
        user_id = context.get("user_id")
        user_query = context.get("user_query", f"list {resource_type}")
        # Get resource config
        resource_config = self.RESOURCE_METHOD_MAP.get(resource_type)
        if resource_config:
            # Use specialized method
            method_name = resource_config.get("list")
            display_name = resource_config.get("display_name", resource_type)
            logger.info(f"📋 Using {method_name} for {display_name}")
            # Call the appropriate method
            if method_name == "list_endpoints":
                result = await api_executor_service.list_endpoints()
            elif method_name == "get_business_units_list":
                result = await api_executor_service.get_business_units_list(
                    ipc_engagement_id=None,
                    user_id=user_id
                )
            elif method_name == "get_environments_list":
                result = await api_executor_service.get_environments_list(
                    ipc_engagement_id=None,
                    user_id=user_id
                )
            elif method_name == "get_zones_list":
                result = await api_executor_service.get_zones_list(
                    ipc_engagement_id=None,
                    user_id=user_id
                )
            else:
                result = await api_executor_service.execute_operation(
                    resource_type=resource_type,
                    operation="list",
                    params=params,
                    user_roles=user_roles
                )
        else:
            # Generic execution
            display_name = resource_type.replace("_", " ").title()
            result = await api_executor_service.execute_operation(
                resource_type=resource_type,
                operation="list",
                params=params,
                user_roles=user_roles)
        if not result.get("success"):
            return {
                "success": False,
                "error": result.get("error"),
                "response": f"Failed to list {display_name}: {result.get('error')}"}
        # Extract data
        raw_data = result.get("data", result.get(resource_type + "s", []))
        # Format response with LLM
        formatted_response = await llm_formatter.format_response(
            resource_type=resource_type,
            operation="list",
            raw_data=raw_data,
            user_query=user_query)
        return {
            "success": True,
            "data": raw_data,
            "response": formatted_response,
            "metadata": {
                "resource_type": resource_type,
                "count": len(raw_data) if isinstance(raw_data, list) else 1} }
    
    async def _execute_generic_crud(self,resource_type: str,operation: str,params: Dict[str, Any],context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute generic CRUD operation.
        Args:
            resource_type: Type of resource
            operation: Operation (create, update, delete)
            params: Parameters
            context: Context
        Returns:
            Operation result
        """
        user_roles = context.get("user_roles", [])
        user_id = context.get("user_id")
        result = await api_executor_service.execute_operation(
            resource_type=resource_type,
            operation=operation,
            params=params,
            user_roles=user_roles,
            user_id=user_id)
        if not result.get("success"):
            return {
                "success": False,
                "error": result.get("error"),
                "response": f"Failed to {operation} {resource_type}: {result.get('error')}"}
        # Format success response
        display_name = resource_type.replace("_", " ").title()
        operation_verb = {
            "create": "created",
            "update": "updated",
            "delete": "deleted"}.get(operation, operation + "ed")
        return {
            "success": True,
            "data": result.get("data", {}),
            "response": f"✅ Successfully {operation_verb} {display_name}.",
            "metadata": {
                "resource_type": resource_type,
                "operation": operation} }