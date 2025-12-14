"""
Function Calling Service - Modern AI agent pattern using OpenAI function calling.
This service defines tools/functions that the LLM can call to perform actions.
"""

from typing import Any, Dict, List, Optional, Callable
import logging
import json
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class FunctionParameter:
    """Function parameter definition."""
    name: str
    type: str  # "string", "integer", "array", "object", "boolean"
    description: str
    required: bool = True
    items: Optional[Dict[str, Any]] = None  # For array types
    properties: Optional[Dict[str, Any]] = None  # For object types
    enum: Optional[List[Any]] = None  # For constrained values


@dataclass
class FunctionDefinition:
    """OpenAI-compatible function definition."""
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Optional[Callable] = None  # The actual function to execute
    
    def to_openai_tool(self) -> Dict[str, Any]:
        """Convert to OpenAI tool format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }


class FunctionCallingService:
    """
    Service for managing function calling (tool use) with LLMs.
    Provides a registry of available functions and handles execution.
    """
    
    def __init__(self):
        self.functions: Dict[str, FunctionDefinition] = {}
        self._register_builtin_functions()
        logger.info("✅ FunctionCallingService initialized")
    
    def _register_builtin_functions(self):
        """Register built-in functions for cluster operations."""
        
        # Function 1: List K8s Clusters
        self.register_function(
            FunctionDefinition(
                name="list_k8s_clusters",
                description=(
                    "List all Kubernetes clusters in specified datacenter locations. "
                    "Use this when user wants to see, view, count, or list clusters. "
                    "Automatically resolves location names (like 'Delhi', 'Mumbai') to endpoint IDs."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "location_names": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Names of datacenter locations (e.g., ['Delhi', 'Mumbai', 'Chennai']). "
                                "If user doesn't specify, use empty array to get all available locations first."
                            )
                        }
                    },
                    "required": []  # location_names is optional
                },
                handler=self._list_k8s_clusters_handler
            )
        )
        
        # Function 2: Get Available Datacenters
        self.register_function(
            FunctionDefinition(
                name="get_datacenters",
                description=(
                    "Get list of available datacenter locations (endpoints). "
                    "Use this first if user doesn't specify a location, or to show available options."
                ),
                parameters={
                    "type": "object",
                    "properties": {},
                    "required": []
                },
                handler=self._get_datacenters_handler
            )
        )
        
        # Function 3: Create K8s Cluster
        self.register_function(
            FunctionDefinition(
                name="create_k8s_cluster",
                description=(
                    "Create a new Kubernetes cluster. "
                    "Requires cluster name, datacenter location, and cluster size."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "cluster_name": {
                            "type": "string",
                            "description": "Name for the new cluster (e.g., 'prod-cluster-01')"
                        },
                        "location_name": {
                            "type": "string",
                            "description": "Datacenter location name (e.g., 'Delhi', 'Mumbai')"
                        },
                        "cluster_size": {
                            "type": "string",
                            "description": "Cluster size tier",
                            "enum": ["small", "medium", "large"]
                        }
                    },
                    "required": ["cluster_name", "location_name", "cluster_size"]
                },
                handler=self._create_k8s_cluster_handler
            )
        )
        
        # Function 4: List Virtual Machines
        self.register_function(
            FunctionDefinition(
                name="list_vms",
                description=(
                    "List all virtual machines (VMs) / instances / servers. "
                    "Can filter by endpoint, zone, or department."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "endpoint_filter": {
                            "type": "string",
                            "description": "Filter VMs by endpoint/datacenter name (optional)"
                        },
                        "zone_filter": {
                            "type": "string",
                            "description": "Filter VMs by zone/VLAN name (optional)"
                        },
                        "department_filter": {
                            "type": "string",
                            "description": "Filter VMs by department/team name (optional)"
                        }
                    },
                    "required": []
                },
                handler=self._list_vms_handler
            )
        )
        
        # Function 5: List Firewalls
        self.register_function(
            FunctionDefinition(
                name="list_firewalls",
                description=(
                    "List all firewalls / network security rules. "
                    "Can specify location(s) or list all."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "location_names": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Datacenter location names to query (optional, defaults to all)"
                        }
                    },
                    "required": []
                },
                handler=self._list_firewalls_handler
            )
        )
        
        # Function 6: List Kafka Services
        self.register_function(
            FunctionDefinition(
                name="list_kafka",
                description=(
                    "List all Apache Kafka messaging services. "
                    "Can specify location(s) or list all."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "location_names": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Datacenter location names (optional)"
                        }
                    },
                    "required": []
                },
                handler=self._list_kafka_handler
            )
        )
        
        # Function 7: List GitLab Services
        self.register_function(
            FunctionDefinition(
                name="list_gitlab",
                description=(
                    "List all GitLab SCM services. "
                    "Can specify location(s) or list all."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "location_names": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Datacenter location names (optional)"
                        }
                    },
                    "required": []
                },
                handler=self._list_gitlab_handler
            )
        )
        
        # Function 8: List Container Registry Services
        self.register_function(
            FunctionDefinition(
                name="list_registry",
                description=(
                    "List all container registry services (Docker registry). "
                    "Can specify location(s) or list all."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "location_names": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Datacenter location names (optional)"
                        }
                    },
                    "required": []
                },
                handler=self._list_registry_handler
            )
        )
        
        # Function 9: List Jenkins Services
        self.register_function(
            FunctionDefinition(
                name="list_jenkins",
                description=(
                    "List all Jenkins CI/CD services. "
                    "Can specify location(s) or list all."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "location_names": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Datacenter location names (optional)"
                        }
                    },
                    "required": []
                },
                handler=self._list_jenkins_handler
            )
        )
        
        # Function 10: List PostgreSQL Services
        self.register_function(
            FunctionDefinition(
                name="list_postgresql",
                description=(
                    "List all PostgreSQL relational database services. "
                    "Can specify location(s) or list all."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "location_names": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Datacenter location names (optional)"
                        }
                    },
                    "required": []
                },
                handler=self._list_postgresql_handler
            )
        )
        
        # Function 11: List DocumentDB Services
        self.register_function(
            FunctionDefinition(
                name="list_documentdb",
                description=(
                    "List all DocumentDB (MongoDB-compatible) NoSQL database services. "
                    "Can specify location(s) or list all."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "location_names": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Datacenter location names (optional)"
                        }
                    },
                    "required": []
                },
                handler=self._list_documentdb_handler
            )
        )
        
        # Function 12: List ALL Managed Services (Comprehensive)
        self.register_function(
            FunctionDefinition(
                name="list_all_managed_services",
                description=(
                    "List ALL managed services across all types (clusters, VMs, firewalls, Kafka, GitLab, "
                    "Jenkins, PostgreSQL, DocumentDB, Container Registry). Use this when user asks for "
                    "'all services' or 'all managed services' in a location. This is more efficient than "
                    "calling individual list functions."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "location_names": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Datacenter location names (optional). If empty, lists all."
                        }
                    },
                    "required": []
                },
                handler=self._list_all_managed_services_handler
            )
        )
        
        logger.info(f"📦 Registered {len(self.functions)} built-in functions")
    
    def register_function(self, func_def: FunctionDefinition):
        """Register a function for LLM to call."""
        self.functions[func_def.name] = func_def
        logger.debug(f"✅ Registered function: {func_def.name}")
    
    def get_tools_for_llm(self) -> List[Dict[str, Any]]:
        """Get all functions in OpenAI tools format."""
        return [func.to_openai_tool() for func in self.functions.values()]
    
    def get_function_schemas_text(self) -> str:
        """Get human-readable function descriptions for system prompts."""
        schemas = []
        for func in self.functions.values():
            params_desc = []
            props = func.parameters.get("properties", {})
            required = func.parameters.get("required", [])
            
            for param_name, param_info in props.items():
                req_marker = " (required)" if param_name in required else " (optional)"
                params_desc.append(
                    f"  - {param_name}: {param_info.get('type')}{req_marker} - {param_info.get('description', '')}"
                )
            
            schema_text = f"**{func.name}**\n{func.description}\nParameters:\n" + "\n".join(params_desc)
            schemas.append(schema_text)
        
        return "\n\n".join(schemas)
    
    async def execute_function(
        self,
        function_name: str,
        arguments: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a function call from the LLM.
        
        Args:
            function_name: Name of the function to call
            arguments: Function arguments as dict
            context: Optional context (user_id, session_id, etc.)
            
        Returns:
            Dict with function execution result
        """
        if function_name not in self.functions:
            error_msg = f"Unknown function: {function_name}"
            logger.error(f"❌ {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "available_functions": list(self.functions.keys())
            }
        
        func_def = self.functions[function_name]
        
        if not func_def.handler:
            error_msg = f"Function {function_name} has no handler"
            logger.error(f"❌ {error_msg}")
            return {
                "success": False,
                "error": error_msg
            }
        
        try:
            logger.info(f"🔧 Executing function: {function_name} with args: {json.dumps(arguments, indent=2)}")
            
            # Call the handler with arguments and context
            result = await func_def.handler(arguments, context or {})
            
            logger.info(f"✅ Function {function_name} executed successfully")
            return result
            
        except Exception as e:
            error_msg = f"Function execution failed: {str(e)}"
            logger.error(f"❌ {error_msg}", exc_info=True)
            return {
                "success": False,
                "error": error_msg
            }
    
    # ==================== Function Handlers ====================
    
    async def _list_k8s_clusters_handler(
        self,
        arguments: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handler for list_k8s_clusters function.
        Implements intelligent location resolution.
        """
        from app.services.api_executor_service import api_executor_service
        
        location_names = arguments.get("location_names", [])
        
        try:
            # Step 0: Get engagement ID (required for all API calls)
            logger.info("🔑 Fetching engagement ID...")
            engagement_result = await api_executor_service.execute_operation(
                resource_type="engagement",
                operation="get",
                params={},
                user_roles=context.get("user_roles", [])
            )
            
            if not engagement_result.get("success"):
                return {
                    "success": False,
                    "error": "Failed to fetch engagement ID",
                    "details": engagement_result.get("error")
                }
            
            engagement_data = engagement_result.get("data", [])
            logger.info(f"📊 Engagement data type: {type(engagement_data)}, content: {engagement_data}")
            
            if not engagement_data:
                return {
                    "success": False,
                    "error": "No engagement data found"
                }
            
            # Handle nested response structure: API returns {"data": {"data": [...]}}
            # The outer "data" is returned by execute_operation
            # The inner "data" contains the actual list of engagements
            if isinstance(engagement_data, dict) and "data" in engagement_data:
                # Nested structure: extract inner data
                engagement_list = engagement_data.get("data", [])
                logger.info(f"🔍 Found nested 'data' key. Inner data type: {type(engagement_list)}, length: {len(engagement_list) if isinstance(engagement_list, list) else 'N/A'}")
                
                if isinstance(engagement_list, list) and len(engagement_list) > 0:
                    engagement_id = engagement_list[0].get("id")
                    logger.info(f"🔍 Extracted from nested structure. Keys: {list(engagement_list[0].keys()) if isinstance(engagement_list[0], dict) else 'not a dict'}")
                else:
                    return {
                        "success": False,
                        "error": f"Unexpected inner data format: {type(engagement_list)}"
                    }
            elif isinstance(engagement_data, dict):
                # Direct dict (no nesting)
                engagement_id = engagement_data.get("id")
                logger.info(f"🔍 Dict format - looking for 'id' field. Keys available: {list(engagement_data.keys())}")
            elif isinstance(engagement_data, list) and len(engagement_data) > 0:
                # Direct list
                engagement_id = engagement_data[0].get("id")
                logger.info(f"🔍 List format - first item keys: {list(engagement_data[0].keys()) if isinstance(engagement_data[0], dict) else 'not a dict'}")
            else:
                return {
                    "success": False,
                    "error": f"Unexpected engagement data format: {type(engagement_data)}"
                }
            
            if not engagement_id:
                logger.error(f"❌ Could not extract engagement_id from data: {engagement_data}")
                return {
                    "success": False,
                    "error": "Engagement ID not found in response",
                    "debug_data": str(engagement_data)[:500]  # First 500 chars for debugging
                }
            
            logger.info(f"✅ Got engagement ID: {engagement_id}")
            
            # Step 1: Get available datacenters
            logger.info("📍 Fetching available datacenters...")
            datacenters_result = await api_executor_service.execute_operation(
                resource_type="endpoint",
                operation="list",
                params={"engagement_id": engagement_id},  # ✅ Now passing engagement_id!
                user_roles=context.get("user_roles", [])
            )
            
            logger.info(f"📍 Datacenters result: success={datacenters_result.get('success')}, has_data={bool(datacenters_result.get('data'))}, error={datacenters_result.get('error', 'None')}")
            
            if not datacenters_result.get("success"):
                return {
                    "success": False,
                    "error": "Failed to fetch datacenters",
                    "details": datacenters_result.get("error")
                }
            
            available_datacenters = datacenters_result.get("data", [])
            logger.info(f"📍 Found {len(available_datacenters) if isinstance(available_datacenters, list) else 'N/A (not a list)'} datacenters")
            logger.info(f"📊 Datacenters raw data type: {type(available_datacenters)}")
            logger.info(f"📊 Datacenters raw content: {available_datacenters}")
            
            # Handle nested response structure (similar to engagement API)
            if isinstance(available_datacenters, dict) and "data" in available_datacenters:
                available_datacenters = available_datacenters.get("data", [])
                logger.info(f"✅ Extracted inner 'data' from datacenters response. Type: {type(available_datacenters)}, Length: {len(available_datacenters) if isinstance(available_datacenters, list) else 'N/A'}")
            
            if available_datacenters and isinstance(available_datacenters, list):
                logger.info(f"📊 First datacenter type: {type(available_datacenters[0])}, value: {available_datacenters[0]}")
            
            if not available_datacenters:
                return {
                    "success": False,
                    "error": "No datacenters found",
                    "details": "The API returned an empty list of datacenters."
                }
            
            # Step 2: Resolve location names to endpoint IDs
            endpoint_ids = []
            
            if location_names:
                # User specified locations - resolve them
                for loc_name in location_names:
                    matched = False
                    for dc in available_datacenters:
                        # Handle both dict and string formats
                        if isinstance(dc, dict):
                            dc_name = dc.get("endpointDisplayName", "").lower()
                            if loc_name.lower() in dc_name or dc_name in loc_name.lower():
                                endpoint_ids.append(dc.get("endpointId"))
                                matched = True
                                logger.info(f"✅ Matched '{loc_name}' to endpoint {dc.get('endpointId')} ({dc_name})")
                                break
                        elif isinstance(dc, str):
                            # API returned string format - log and skip
                            logger.warning(f"⚠️ Datacenter is a string, not dict: {dc}")
                    
                    if not matched:
                        logger.warning(f"⚠️ Could not match location '{loc_name}'")
            else:
                # No locations specified - use all available
                for dc in available_datacenters:
                    if isinstance(dc, dict):
                        ep_id = dc.get("endpointId")
                        if ep_id:
                            endpoint_ids.append(ep_id)
                    elif isinstance(dc, str):
                        logger.warning(f"⚠️ Datacenter is a string, not dict: {dc}")
                logger.info(f"📍 No locations specified, using all {len(endpoint_ids)} datacenters")
            
            if not endpoint_ids:
                logger.warning(f"⚠️ No endpoint IDs found! Available datacenters: {len(available_datacenters)}")
                return {
                    "success": False,
                    "error": "No valid endpoint IDs found",
                    "available_datacenters": [
                        {"id": dc.get("endpointId"), "name": dc.get("endpointDisplayName")}
                        for dc in available_datacenters
                    ]
                }
            
            # Step 3: List clusters for resolved endpoints
            logger.info(f"🔍 Listing clusters for endpoints: {endpoint_ids}")
            clusters_result = await api_executor_service.execute_operation(
                resource_type="k8s_cluster",
                operation="list",
                params={
                    "engagement_id": engagement_id,  # ✅ Required in URL path
                    "endpoints": endpoint_ids
                },
                user_roles=context.get("user_roles", [])
            )
            
            logger.info(f"🔍 Clusters result: success={clusters_result.get('success')}, has_data={bool(clusters_result.get('data'))}, error={clusters_result.get('error', 'None')}")
            
            if not clusters_result.get("success"):
                return {
                    "success": False,
                    "error": "Failed to list clusters",
                    "details": clusters_result.get("error")
                }
            
            clusters = clusters_result.get("data", [])
            
            return {
                "success": True,
                "clusters": clusters,
                "total_count": len(clusters),
                "datacenters_queried": [
                    {"id": dc.get("endpointId"), "name": dc.get("endpointDisplayName")}
                    for dc in available_datacenters
                    if dc.get("endpointId") in endpoint_ids
                ],
                "message": f"Found {len(clusters)} cluster(s) across {len(endpoint_ids)} datacenter(s)"
            }
            
        except Exception as e:
            logger.error(f"❌ Error in list_k8s_clusters_handler: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _get_datacenters_handler(
        self,
        arguments: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handler for get_datacenters function."""
        from app.services.api_executor_service import api_executor_service
        
        try:
            # Step 0: Get engagement ID first
            logger.info("🔑 Fetching engagement ID...")
            engagement_result = await api_executor_service.execute_operation(
                resource_type="engagement",
                operation="get",
                params={},
                user_roles=context.get("user_roles", [])
            )
            
            if not engagement_result.get("success"):
                return {
                    "success": False,
                    "error": "Failed to fetch engagement ID",
                    "details": engagement_result.get("error")
                }
            
            engagement_data = engagement_result.get("data", [])
            logger.info(f"📊 Engagement data type: {type(engagement_data)}, content: {engagement_data}")
            
            if not engagement_data:
                return {
                    "success": False,
                    "error": "No engagement data found"
                }
            
            # Handle nested response structure: API returns {"data": {"data": [...]}}
            if isinstance(engagement_data, dict) and "data" in engagement_data:
                engagement_list = engagement_data.get("data", [])
                logger.info(f"🔍 Found nested 'data' key. Inner data type: {type(engagement_list)}, length: {len(engagement_list) if isinstance(engagement_list, list) else 'N/A'}")
                
                if isinstance(engagement_list, list) and len(engagement_list) > 0:
                    engagement_id = engagement_list[0].get("id")
                    logger.info(f"🔍 Extracted from nested structure. Keys: {list(engagement_list[0].keys()) if isinstance(engagement_list[0], dict) else 'not a dict'}")
                else:
                    return {
                        "success": False,
                        "error": f"Unexpected inner data format: {type(engagement_list)}"
                    }
            elif isinstance(engagement_data, dict):
                engagement_id = engagement_data.get("id")
                logger.info(f"🔍 Dict format - looking for 'id' field. Keys available: {list(engagement_data.keys())}")
            elif isinstance(engagement_data, list) and len(engagement_data) > 0:
                engagement_id = engagement_data[0].get("id")
                logger.info(f"🔍 List format - first item keys: {list(engagement_data[0].keys()) if isinstance(engagement_data[0], dict) else 'not a dict'}")
            else:
                return {
                    "success": False,
                    "error": f"Unexpected engagement data format: {type(engagement_data)}"
                }
            
            if not engagement_id:
                logger.error(f"❌ Could not extract engagement_id from data: {engagement_data}")
                return {
                    "success": False,
                    "error": "Engagement ID not found in response",
                    "debug_data": str(engagement_data)[:500]
                }
            
            logger.info(f"✅ Got engagement ID: {engagement_id}")
            
            # Step 1: List datacenters with engagement_id
            result = await api_executor_service.execute_operation(
                resource_type="endpoint",
                operation="list",
                params={"engagement_id": engagement_id},  # ✅ Now passing engagement_id!
                user_roles=context.get("user_roles", [])
            )
            
            if result.get("success"):
                datacenters = result.get("data", [])
                logger.info(f"📊 Datacenters raw type: {type(datacenters)}")
                
                # Handle nested response structure (same as engagement API)
                if isinstance(datacenters, dict) and "data" in datacenters:
                    datacenters = datacenters.get("data", [])
                    logger.info(f"✅ Extracted inner 'data' from datacenters. Length: {len(datacenters)}")
                
                if not isinstance(datacenters, list):
                    return {
                        "success": False,
                        "error": f"Unexpected datacenters format: {type(datacenters)}"
                    }
                
                return {
                    "success": True,
                    "datacenters": [
                        {
                            "id": dc.get("endpointId") if isinstance(dc, dict) else None,
                            "name": dc.get("endpointDisplayName") if isinstance(dc, dict) else str(dc),
                            "region": dc.get("region", "Unknown") if isinstance(dc, dict) else "Unknown"
                        }
                        for dc in datacenters
                        if isinstance(dc, dict)  # Only process dict items
                    ],
                    "total_count": len(datacenters),
                    "message": f"Found {len(datacenters)} available datacenter(s)"
                }
            else:
                return {
                    "success": False,
                    "error": result.get("error", "Unknown error")
                }
                
        except Exception as e:
            logger.error(f"❌ Error in get_datacenters_handler: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _create_k8s_cluster_handler(
        self,
        arguments: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handler for create_k8s_cluster function."""
        from app.services.api_executor_service import api_executor_service
        
        cluster_name = arguments.get("cluster_name")
        location_name = arguments.get("location_name")
        cluster_size = arguments.get("cluster_size")
        
        try:
            # Step 0: Get engagement ID
            logger.info("🔑 Fetching engagement ID...")
            engagement_result = await api_executor_service.execute_operation(
                resource_type="engagement",
                operation="get",
                params={},
                user_roles=context.get("user_roles", [])
            )
            
            if not engagement_result.get("success"):
                return {
                    "success": False,
                    "error": "Failed to fetch engagement ID"
                }
            
            engagement_data = engagement_result.get("data", [])
            logger.info(f"📊 Engagement data type: {type(engagement_data)}, content: {engagement_data}")
            
            if not engagement_data:
                return {
                    "success": False,
                    "error": "No engagement data found"
                }
            
            # Handle nested response structure: API returns {"data": {"data": [...]}}
            if isinstance(engagement_data, dict) and "data" in engagement_data:
                engagement_list = engagement_data.get("data", [])
                logger.info(f"🔍 Found nested 'data' key. Inner data type: {type(engagement_list)}, length: {len(engagement_list) if isinstance(engagement_list, list) else 'N/A'}")
                
                if isinstance(engagement_list, list) and len(engagement_list) > 0:
                    engagement_id = engagement_list[0].get("id")
                    logger.info(f"🔍 Extracted from nested structure. Keys: {list(engagement_list[0].keys()) if isinstance(engagement_list[0], dict) else 'not a dict'}")
                else:
                    return {
                        "success": False,
                        "error": f"Unexpected inner data format: {type(engagement_list)}"
                    }
            elif isinstance(engagement_data, dict):
                engagement_id = engagement_data.get("id")
                logger.info(f"🔍 Dict format - looking for 'id' field. Keys available: {list(engagement_data.keys())}")
            elif isinstance(engagement_data, list) and len(engagement_data) > 0:
                engagement_id = engagement_data[0].get("id")
                logger.info(f"🔍 List format - first item keys: {list(engagement_data[0].keys()) if isinstance(engagement_data[0], dict) else 'not a dict'}")
            else:
                return {
                    "success": False,
                    "error": f"Unexpected engagement data format: {type(engagement_data)}"
                }
            
            if not engagement_id:
                logger.error(f"❌ Could not extract engagement_id from data: {engagement_data}")
                return {
                    "success": False,
                    "error": "Engagement ID not found in response",
                    "debug_data": str(engagement_data)[:500]
                }
            
            logger.info(f"✅ Got engagement ID: {engagement_id}")
            
            # Resolve location to endpoint ID
            datacenters_result = await api_executor_service.execute_operation(
                resource_type="endpoint",
                operation="list",
                params={"engagement_id": engagement_id},  # ✅ Pass engagement_id
                user_roles=context.get("user_roles", [])
            )
            
            if not datacenters_result.get("success"):
                return {
                    "success": False,
                    "error": "Failed to fetch datacenters"
                }
            
            available_datacenters = datacenters_result.get("data", [])
            endpoint_id = None
            
            for dc in available_datacenters:
                dc_name = dc.get("endpointDisplayName", "").lower()
                if location_name.lower() in dc_name:
                    endpoint_id = dc.get("endpointId")
                    break
            
            if not endpoint_id:
                return {
                    "success": False,
                    "error": f"Could not find datacenter matching '{location_name}'",
                    "available_locations": [dc.get("endpointDisplayName") for dc in available_datacenters]
                }
            
            # Create cluster
            create_result = await api_executor_service.execute_operation(
                resource_type="k8s_cluster",
                operation="create",
                params={
                    "Cluster Name": cluster_name,
                    "endpoints": [endpoint_id],
                    "size": cluster_size
                },
                user_roles=context.get("user_roles", [])
            )
            
            return create_result
            
        except Exception as e:
            logger.error(f"❌ Error in create_k8s_cluster_handler: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }


    # ==================== NEW HANDLERS FOR ALL RESOURCES ====================
    
    async def _list_vms_handler(
        self,
        arguments: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handler for list_vms function."""
        from app.services.api_executor_service import api_executor_service
        
        endpoint_filter = arguments.get("endpoint_filter")
        zone_filter = arguments.get("zone_filter")
        department_filter = arguments.get("department_filter")
        
        try:
            logger.info("🖥️ Listing virtual machines...")
            result = await api_executor_service.list_vms(
                endpoint_filter=endpoint_filter,
                zone_filter=zone_filter,
                department_filter=department_filter
            )
            
            if result.get("success"):
                vms = result.get("data", [])
                return {
                    "success": True,
                    "vms": vms,
                    "total_count": len(vms),
                    "filters_applied": {
                        "endpoint": endpoint_filter,
                        "zone": zone_filter,
                        "department": department_filter
                    },
                    "message": f"Found {len(vms)} virtual machine(s)"
                }
            else:
                return {
                    "success": False,
                    "error": result.get("error", "Unknown error listing VMs")
                }
                
        except Exception as e:
            logger.error(f"❌ Error in list_vms_handler: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _list_firewalls_handler(
        self,
        arguments: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handler for list_firewalls function."""
        from app.services.api_executor_service import api_executor_service
        
        location_names = arguments.get("location_names", [])
        
        try:
            # Get engagement ID and datacenters (same pattern as clusters)
            logger.info("🔑 Fetching engagement ID...")
            engagement_result = await api_executor_service.execute_operation(
                resource_type="engagement",
                operation="get",
                params={},
                user_roles=context.get("user_roles", [])
            )
            
            if not engagement_result.get("success"):
                return {
                    "success": False,
                    "error": "Failed to fetch engagement ID"
                }
            
            engagement_data = engagement_result.get("data", {})
            if isinstance(engagement_data, dict) and "data" in engagement_data:
                engagement_id = engagement_data["data"][0].get("id")
            else:
                return {"success": False, "error": "Could not extract engagement ID"}
            
            # Get datacenters
            logger.info("📍 Fetching available datacenters...")
            datacenters_result = await api_executor_service.execute_operation(
                resource_type="endpoint",
                operation="list",
                params={"engagement_id": engagement_id},
                user_roles=context.get("user_roles", [])
            )
            
            if not datacenters_result.get("success"):
                return {"success": False, "error": "Failed to fetch datacenters"}
            
            available_datacenters = datacenters_result.get("data", {})
            if isinstance(available_datacenters, dict) and "data" in available_datacenters:
                available_datacenters = available_datacenters["data"]
            
            # Resolve endpoint IDs
            endpoint_ids = []
            if location_names:
                for loc_name in location_names:
                    for dc in available_datacenters:
                        if isinstance(dc, dict) and loc_name.lower() in dc.get("endpointDisplayName", "").lower():
                            endpoint_ids.append(dc.get("endpointId"))
                            break
            else:
                endpoint_ids = [dc.get("endpointId") for dc in available_datacenters if isinstance(dc, dict)]
            
            # List firewalls
            logger.info(f"🔥 Listing firewalls for endpoints: {endpoint_ids}")
            result = await api_executor_service.list_firewalls(
                endpoint_ids=endpoint_ids
            )
            
            if result.get("success"):
                firewalls = result.get("data", [])
                return {
                    "success": True,
                    "firewalls": firewalls,
                    "total_count": len(firewalls),
                    "message": f"Found {len(firewalls)} firewall(s)"
                }
            else:
                return {
                    "success": False,
                    "error": result.get("error", "Unknown error")
                }
                
        except Exception as e:
            logger.error(f"❌ Error in list_firewalls_handler: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _list_managed_service_handler(
        self,
        service_type: str,
        location_names: List[str],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generic handler for all managed services (Kafka, GitLab, Registry, Jenkins, PostgreSQL, DocumentDB).
        """
        from app.services.api_executor_service import api_executor_service
        
        try:
            # Get engagement ID
            logger.info(f"🔑 Fetching engagement ID for {service_type}...")
            engagement_result = await api_executor_service.execute_operation(
                resource_type="engagement",
                operation="get",
                params={},
                user_roles=context.get("user_roles", [])
            )
            
            if not engagement_result.get("success"):
                return {"success": False, "error": "Failed to fetch engagement ID"}
            
            engagement_data = engagement_result.get("data", {})
            if isinstance(engagement_data, dict) and "data" in engagement_data:
                engagement_id = engagement_data["data"][0].get("id")
            else:
                return {"success": False, "error": "Could not extract engagement ID"}
            
            # Get datacenters
            logger.info("📍 Fetching available datacenters...")
            datacenters_result = await api_executor_service.execute_operation(
                resource_type="endpoint",
                operation="list",
                params={"engagement_id": engagement_id},
                user_roles=context.get("user_roles", [])
            )
            
            if not datacenters_result.get("success"):
                return {"success": False, "error": "Failed to fetch datacenters"}
            
            available_datacenters = datacenters_result.get("data", {})
            if isinstance(available_datacenters, dict) and "data" in available_datacenters:
                available_datacenters = available_datacenters["data"]
            
            # Resolve endpoint IDs
            endpoint_ids = []
            if location_names:
                for loc_name in location_names:
                    for dc in available_datacenters:
                        if isinstance(dc, dict) and loc_name.lower() in dc.get("endpointDisplayName", "").lower():
                            endpoint_ids.append(dc.get("endpointId"))
                            break
            else:
                endpoint_ids = [dc.get("endpointId") for dc in available_datacenters if isinstance(dc, dict)]
            
            # Get IPC engagement ID
            logger.info("🔄 Converting to IPC engagement ID...")
            ipc_result = await api_executor_service.get_ipc_engagement_id(engagement_id)
            if not ipc_result:
                return {"success": False, "error": "Failed to get IPC engagement ID"}
            
            # List managed service
            logger.info(f"📦 Listing {service_type} services...")
            result = await api_executor_service.list_managed_services(
                service_type=service_type,
                endpoint_ids=endpoint_ids,
                ipc_engagement_id=ipc_result
            )
            
            if result.get("success"):
                services = result.get("data", [])
                return {
                    "success": True,
                    "services": services,
                    "service_type": service_type,
                    "total_count": len(services),
                    "message": f"Found {len(services)} {service_type} service(s)"
                }
            else:
                return {
                    "success": False,
                    "error": result.get("error", "Unknown error")
                }
                
        except Exception as e:
            logger.error(f"❌ Error in list_{service_type}_handler: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _list_kafka_handler(self, arguments: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Handler for list_kafka function."""
        return await self._list_managed_service_handler("IKSKafka", arguments.get("location_names", []), context)
    
    async def _list_gitlab_handler(self, arguments: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Handler for list_gitlab function."""
        return await self._list_managed_service_handler("IKSGitlab", arguments.get("location_names", []), context)
    
    async def _list_registry_handler(self, arguments: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Handler for list_registry function."""
        return await self._list_managed_service_handler("IKSContainerRegistry", arguments.get("location_names", []), context)
    
    async def _list_jenkins_handler(self, arguments: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Handler for list_jenkins function."""
        return await self._list_managed_service_handler("IKSJenkins", arguments.get("location_names", []), context)
    
    async def _list_postgresql_handler(self, arguments: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Handler for list_postgresql function."""
        return await self._list_managed_service_handler("IKSPostgreSQL", arguments.get("location_names", []), context)
    
    async def _list_documentdb_handler(self, arguments: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Handler for list_documentdb function."""
        return await self._list_managed_service_handler("IKSDocumentDB", arguments.get("location_names", []), context)
    
    async def _list_all_managed_services_handler(
        self,
        arguments: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handler for list_all_managed_services function.
        Queries all service types and aggregates results.
        """
        from app.services.api_executor_service import api_executor_service
        
        try:
            location_names = arguments.get("location_names", [])
            logger.info(f"🌐 Listing ALL managed services for locations: {location_names or 'all'}")
            
            # Step 1: Get engagement ID
            logger.info("🔑 Fetching engagement ID...")
            engagement_result = await api_executor_service.execute_operation(
                resource_type="engagement",
                operation="get",
                params={},
                user_roles=context.get("user_roles", [])
            )
            
            if not engagement_result.get("success"):
                return {
                    "success": False,
                    "error": "Failed to fetch engagement ID"
                }
            
            # Extract engagement data
            engagement_data = engagement_result.get("data", {})
            if isinstance(engagement_data, dict) and "data" in engagement_data:
                engagement_list = engagement_data.get("data", [])
                if isinstance(engagement_list, list) and len(engagement_list) > 0:
                    engagement_id = engagement_list[0].get("id")
                else:
                    return {"success": False, "error": "No engagement found"}
            else:
                return {"success": False, "error": "Invalid engagement data"}
            
            # Step 2: Get datacenters
            logger.info("📍 Fetching available datacenters...")
            datacenters_result = await api_executor_service.execute_operation(
                resource_type="endpoint",
                operation="list",
                params={"engagement_id": engagement_id},
                user_roles=context.get("user_roles", [])
            )
            
            if not datacenters_result.get("success"):
                return {"success": False, "error": "Failed to fetch datacenters"}
            
            available_datacenters = datacenters_result.get("data", [])
            if isinstance(available_datacenters, dict) and "data" in available_datacenters:
                available_datacenters = available_datacenters.get("data", [])
            
            # Step 3: Filter by location if specified
            if location_names:
                endpoint_ids = []
                for loc in location_names:
                    loc_lower = loc.lower()
                    for dc in available_datacenters:
                        dc_name = dc.get("endpointDisplayName", "").lower()
                        if loc_lower in dc_name or dc_name in loc_lower:
                            endpoint_ids.append(dc.get("endpointId"))
                            logger.info(f"✅ Matched '{loc}' to endpoint {dc.get('endpointId')} ({dc.get('endpointDisplayName')})")
                            break
            else:
                endpoint_ids = [dc.get("endpointId") for dc in available_datacenters]
            
            if not endpoint_ids:
                return {
                    "success": True,
                    "data": [],
                    "summary": f"No datacenters found matching: {location_names}"
                }
            
            # Step 4: Query ALL service types
            logger.info(f"🔍 Querying all service types for endpoints: {endpoint_ids}")
            
            all_services = {
                "clusters": [],
                "vms": [],
                "firewalls": [],
                "kafka": [],
                "gitlab": [],
                "jenkins": [],
                "postgresql": [],
                "documentdb": [],
                "registry": []
            }
            
            # Query K8s clusters
            try:
                clusters_result = await api_executor_service.execute_operation(
                    resource_type="k8s_cluster",
                    operation="list",
                    params={"endpoint_ids": endpoint_ids},
                    user_roles=context.get("user_roles", [])
                )
                if clusters_result.get("success"):
                    all_services["clusters"] = clusters_result.get("data", [])
            except Exception as e:
                logger.warning(f"⚠️ Failed to query clusters: {e}")
            
            # Query VMs
            try:
                vm_result = await api_executor_service.list_vms(
                    endpoint_filter=location_names[0] if location_names else None
                )
                if vm_result.get("success"):
                    all_services["vms"] = vm_result.get("data", [])
            except Exception as e:
                logger.warning(f"⚠️ Failed to query VMs: {e}")
            
            # Query Firewalls
            try:
                fw_result = await api_executor_service.list_firewalls(endpoint_ids=endpoint_ids)
                if fw_result.get("success"):
                    all_services["firewalls"] = fw_result.get("data", [])
            except Exception as e:
                logger.warning(f"⚠️ Failed to query firewalls: {e}")
            
            # Query Managed Services (Kafka, GitLab, Jenkins, etc.)
            service_types = [
                ("kafka", "IKSKafka"),
                ("gitlab", "IKSGitlab"),
                ("jenkins", "IKSJenkins"),
                ("postgresql", "IKSPostgres"),
                ("documentdb", "IKSDocumentDB"),
                ("registry", "IKSContainerRegistry")
            ]
            
            for service_name, service_type in service_types:
                try:
                    result = await api_executor_service.list_managed_services(
                        service_type=service_type,
                        endpoint_ids=endpoint_ids
                    )
                    if result.get("success"):
                        services = result.get("data", [])
                        # Handle nested data structure
                        if isinstance(services, dict) and "data" in services:
                            services = services.get("data", [])
                        all_services[service_name] = services
                except Exception as e:
                    logger.warning(f"⚠️ Failed to query {service_name}: {e}")
            
            # Step 5: Build summary
            total_count = sum(len(v) for v in all_services.values())
            summary_parts = []
            for service_type, items in all_services.items():
                if items:
                    summary_parts.append(f"{len(items)} {service_type}")
            
            summary = f"Found {total_count} total services: " + ", ".join(summary_parts) if summary_parts else "No services found"
            
            logger.info(f"✅ {summary}")
            
            return {
                "success": True,
                "data": all_services,
                "summary": summary,
                "total_count": total_count,
                "locations_queried": [
                    {"id": dc.get("endpointId"), "name": dc.get("endpointDisplayName")}
                    for dc in available_datacenters
                    if dc.get("endpointId") in endpoint_ids
                ]
            }
            
        except Exception as e:
            logger.error(f"❌ Error in list_all_managed_services_handler: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }


# Global instance
function_calling_service = FunctionCallingService()

