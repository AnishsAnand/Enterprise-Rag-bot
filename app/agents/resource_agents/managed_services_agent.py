"""
Managed Services Agent - Handles all PaaS managed services.
Includes: Kafka, GitLab, Jenkins, PostgreSQL, DocumentDB, Container Registry.
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
    
    # Mapping of resource types to service types
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
    
    # Display names for each service
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
        return ["list", "create", "delete"]  # Most managed services support these
    
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
            
            # Get user roles from context for permissions
            user_roles = context.get("user_roles", [])
            
            # Get endpoint IDs
            endpoint_ids = params.get("endpoints", [])
            
            if not endpoint_ids:
                # List all
                datacenters = await self.get_datacenters(user_roles=user_roles)
                endpoint_ids = [dc.get("endpointId") for dc in datacenters if dc.get("endpointId")]
            
            if not endpoint_ids:
                return {
                    "success": True,
                    "data": [],
                    "response": f"No datacenters found to query {service_display_name} services."
                }
            
            logger.info(f"🔍 Querying endpoints: {endpoint_ids}")
            
            # Call API
            result = await api_executor_service.list_managed_services(
                service_type=service_type,
                endpoint_ids=endpoint_ids
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
            
            # Format response with LLM
            user_query = context.get("user_query", "")
            formatted_response = await self.format_response_with_llm(
                operation="list",
                raw_data=services,
                user_query=user_query,
                context={
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
    
    def _build_formatting_prompt(
        self,
        operation: str,
        raw_data: Any,
        user_query: str,
        context: Dict[str, Any] = None
    ) -> str:
        """
        Build managed service-specific formatting prompt.
        
        Overrides base class to provide service-specific formatting.
        """
        if operation == "list":
            return self._build_list_formatting_prompt(raw_data, user_query, context)
        else:
            return super()._build_formatting_prompt(operation, raw_data, user_query, context)
    
    def _build_list_formatting_prompt(
        self,
        services: List[Dict[str, Any]],
        user_query: str,
        context: Dict[str, Any]
    ) -> str:
        """Build prompt for listing managed services."""
        import json
        
        service_type = context.get("service_type", "")
        service_display_name = context.get("service_display_name", "Service")
        
        # Service-specific field mappings
        key_fields = self._get_service_key_fields(service_type)
        
        return f"""You are a cloud infrastructure assistant. Format the {service_display_name} listing for the user.

**User's Query:** {user_query}

**Services Found:**
```json
{json.dumps(services, indent=2)}
```

**Service Type:** {service_display_name}

**Instructions:**
1. Start with a summary: "Found X {service_display_name} Service(s)"
2. If no services found, say "No {service_display_name} services found" and suggest creating one
3. Present services in a table format with these key fields:
   {key_fields}
4. Use emojis for status:
   - ✅ for Active/Running/Ready
   - ⚠️ for Pending/Provisioning/Warning
   - ❌ for Failed/Error/Down
5. Highlight important information:
   - **Bold** service names
   - Show versions clearly
   - Display URLs/endpoints prominently
6. Group by location if multiple locations
7. Add service-specific insights:
   {self._get_service_insights_instructions(service_type)}
8. Include helpful next steps

**Format Example:**
```
## ✅ Found 1 {service_display_name} Service

### Chennai-AMB

| Service Name | Status | Version | URL | Location |
|--------------|--------|---------|-----|----------|
| **my-service** | ✅ Active | 2.11.0 | https://... | Chennai-AMB |

**Service Details:**
- **Cluster:** aistdh200cl01
- **Namespace:** ms-iksconta-svc-name
- **Resources:** CPU: 16 cores, Memory: 32 GiB

💡 **Next Steps:** To view detailed configuration, ask me about "my-service details"
```

Do NOT include raw JSON. Provide a user-friendly, formatted response."""
    
    def _get_service_key_fields(self, service_type: str) -> str:
        """Get key fields to display for each service type."""
        field_map = {
            "IKSKafka": "- Service Name\n   - Status\n   - Version\n   - Kafka URL\n   - Location/Cluster",
            "IKSGitlab": "- Service Name\n   - Status\n   - Version\n   - GitLab URL\n   - Location/Cluster",
            "IKSJenkins": "- Service Name\n   - Status\n   - Version\n   - Jenkins URL\n   - Location/Cluster\n   - Plugins Installed",
            "IKSPostgres": "- Service Name\n   - Status\n   - Version\n   - Storage Size\n   - Location/Cluster\n   - Replicas",
            "IKSDocumentDB": "- Service Name\n   - Status\n   - Version\n   - Storage Size\n   - Location/Cluster\n   - Replicas",
            "IKSContainerRegistry": "- Service Name\n   - Status\n   - Version\n   - Registry URL\n   - Location/Cluster\n   - Storage"
        }
        return field_map.get(service_type, "- Service Name\n   - Status\n   - Version\n   - Location")
    
    def _get_service_insights_instructions(self, service_type: str) -> str:
        """Get service-specific insights to highlight."""
        insights_map = {
            "IKSKafka": "- Mention topic count if available\n   - Highlight broker configuration\n   - Note replication factor",
            "IKSGitlab": "- Mention number of projects/repos\n   - Highlight CI/CD runner status\n   - Note storage usage",
            "IKSJenkins": "- List installed plugins\n   - Mention active jobs\n   - Note executor count",
            "IKSPostgres": "- Highlight database size\n   - Mention replication status\n   - Note backup schedule if available",
            "IKSDocumentDB": "- Highlight database size\n   - Mention replica set configuration\n   - Note sharding if enabled",
            "IKSContainerRegistry": "- Mention storage used\n   - Note number of images/repos\n   - Highlight access URL format"
        }
        return insights_map.get(service_type, "- Highlight unique service characteristics")

