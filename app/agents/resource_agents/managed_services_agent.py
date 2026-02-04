"""
Managed Services Agent - Handles all PaaS managed services.
"""

from typing import Any, Dict, List, Optional
import logging

from app.agents.resource_agents.base_resource_agent import BaseResourceAgent
from app.services.api_executor_service import api_executor_service

logger = logging.getLogger(__name__)


class ManagedServicesAgent(BaseResourceAgent):
    """
    Agent for managed PaaS services.
    
    Handles multiple service types with intelligent formatting for each:
    - Kafka (IKSKafka)
    - GitLab (IKSGitlab)
    - Jenkins (IKSJenkins)
    - PostgreSQL (IKSPostgres)
    - DocumentDB (IKSDocumentDB)
    - Container Registry (IKSContainerRegistry)
    """
    SERVICE_TYPE_MAP = {
        "kafka": "IKSKafka",
        "gitlab": "IKSGitlab",
        "jenkins": "IKSJenkins",
        "postgres": "IKSPostgres",
        "postgresql": "IKSPostgres",
        "documentdb": "IKSDocumentDB",
        "container_registry": "IKSContainerRegistry",
        "registry": "IKSContainerRegistry"
    }
    SERVICE_DISPLAY_NAMES = {
        "IKSKafka": "Apache Kafka",
        "IKSGitlab": "GitLab SCM",
        "IKSJenkins": "Jenkins CI/CD",
        "IKSPostgres": "PostgreSQL Database",
        "IKSDocumentDB": "DocumentDB (MongoDB)",
        "IKSContainerRegistry": "Container Registry"
    }
    
    def __init__(self):
        super().__init__(
            agent_name="ManagedServicesAgent",
            agent_description=(
                "Specialized agent for managed PaaS services including Kafka, "
                "GitLab, Jenkins, PostgreSQL, DocumentDB, and Container Registry. "
                "Uses LLM intelligence to format service-specific responses."
            ),
            resource_type="managed_service",
            temperature=0.2
        )
    
    def get_supported_operations(self) -> List[str]:
        """Get list of supported operations."""
        return ["list", "create", "delete"]  
    
    async def execute_operation(
        self,
        operation: str,
        params: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a managed service operation.
        
        Args:
            operation: Operation to perform (list, create, delete)
            params: Parameters including service_type_hint
            context: Context (session_id, user_query, etc.)
            
        Returns:
            Dict with success status and formatted response
        """
        try:
            # Determine which service type we're dealing with
            service_type_hint = params.get("service_type_hint", "")
            resource_type = context.get("resource_type", "")
            
            logger.info(f"📦 ManagedServicesAgent executing: {operation} for {resource_type or service_type_hint}")
            
            if operation == "list":
                return await self._list_service(params, context)
            elif operation == "create":
                return await self._create_service(params, context)
            elif operation == "delete":
                return await self._delete_service(params, context)
            else:
                return {
                    "success": False,
                    "error": f"Unsupported operation: {operation}",
                    "response": f"I don't support the '{operation}' operation for managed services yet."
                }
                
        except Exception as e:
            logger.error(f"❌ ManagedServicesAgent error: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "response": f"An error occurred while {operation}ing managed service: {str(e)}"
            }
    
    async def _list_service(
        self,
        params: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        List managed services with intelligent formatting.
        
        Args:
            params: Parameters (endpoints, service_type_hint)
            context: Context (user_query, resource_type)
            
        Returns:
            Formatted service list
        """
        try:
            # Determine service type
            resource_type = context.get("resource_type", "")
            service_type = self.SERVICE_TYPE_MAP.get(resource_type)
            
            if not service_type:
                # Try to infer from user query
                user_query = context.get("user_query", "").lower()
                for res_type, svc_type in self.SERVICE_TYPE_MAP.items():
                    if res_type in user_query:
                        service_type = svc_type
                        break
            
            if not service_type:
                return {
                    "success": False,
                    "error": "Could not determine service type",
                    "response": "I couldn't determine which managed service you're asking about. Please specify: kafka, gitlab, jenkins, postgres, documentdb, or container registry."
                }
            
            service_display_name = self.SERVICE_DISPLAY_NAMES.get(service_type, service_type)
            logger.info(f"📋 Listing {service_display_name} services")
            
            # Get user roles and auth from context - MUST be done FIRST
            user_roles = context.get("user_roles", [])
            auth_token = context.get("auth_token")
            user_id = context.get("user_id")
            user_type = context.get("user_type")
            selected_engagement_id = context.get("selected_engagement_id")
            
            logger.info(f"🔐 Managed services listing with auth_token: {'✓' if auth_token else '✗'}, engagement: {selected_engagement_id}")
            
            # Get endpoint IDs
            endpoint_ids = params.get("endpoints", [])
            
            if not endpoint_ids:
                # List all
                datacenters = await self.get_datacenters(user_roles=user_roles, auth_token=auth_token, user_id=user_id, user_type=user_type)
                endpoint_ids = [dc.get("endpointId") for dc in datacenters if dc.get("endpointId")]
            
            if not endpoint_ids:
                return {
                    "success": True,
                    "data": [],
                    "response": f"No datacenters found to query {service_display_name} services."
                }
            
            logger.info(f"🔍 Querying endpoints: {endpoint_ids}")
            
            # Get IPC engagement ID if we have a selected engagement
            ipc_engagement_id = None
            if selected_engagement_id:
                ipc_engagement_id = await api_executor_service.get_ipc_engagement_id(
                    engagement_id=selected_engagement_id,
                    auth_token=auth_token,
                    user_id=user_id
                )
                logger.info(f"✅ Using IPC engagement ID: {ipc_engagement_id} (from selected: {selected_engagement_id})")
            
            # Call API
            result = await api_executor_service.list_managed_services(
                service_type=service_type,
                endpoint_ids=endpoint_ids,
                ipc_engagement_id=ipc_engagement_id,
                auth_token=auth_token,
                user_id=user_id
            )
            
            if not result.get("success"):
                return {
                    "success": False,
                    "error": result.get("error"),
                    "response": f"Failed to list {service_display_name} services: {result.get('error')}"
                }
            
            services = result.get("data", [])
            
            # Handle nested data
            if isinstance(services, dict) and "data" in services:
                services = services.get("data", [])
            
            logger.info(f"✅ Found {len(services)} {service_display_name} service(s)")
            
            # Format response with agentic formatter (prevents hallucination)
            user_query = context.get("user_query", "")
            formatted_response = await self.format_response_agentic(
                operation="list",
                raw_data=services,
                user_query=user_query,
                context={
                    "query_type": "general",
                    "service_type": service_type,
                    "service_display_name": service_display_name,
                    "endpoint_names": params.get("endpoint_names", [])
                }
            )
            
            return {
                "success": True,
                "data": services,
                "response": formatted_response,
                "metadata": {
                    "count": len(services),
                    "service_type": service_type,
                    "service_display_name": service_display_name,
                    "endpoints_queried": len(endpoint_ids)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error listing managed services: {str(e)}", exc_info=True)
            raise
    
    async def _create_service(
        self,
        params: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a managed service."""
        # TODO: Implement when creation APIs are available
        return {
            "success": False,
            "response": "Managed service creation is not yet implemented."
        }
    
    async def _delete_service(
        self,
        params: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Delete a managed service."""
        # TODO: Implement when delete APIs are available
        return {
            "success": False,
            "response": "Managed service deletion is not yet implemented."
        }

