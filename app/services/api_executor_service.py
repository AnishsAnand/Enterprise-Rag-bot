"""
API Executor Service for performing CRUD operations on managed services.
Handles API calls to external services (K8s, Firewall, etc.) with validation and error handling.
"""

from typing import Any, Dict, List, Optional
import httpx
import logging
import json
import os
import re
from datetime import datetime, timedelta
import asyncio

logger = logging.getLogger(__name__)


class APIExecutorService:
    """
    Service for executing CRUD operations via API calls.
    Handles authentication, request formatting, and response parsing.
    """
    
    def __init__(self, resource_schema_path: str = None):
        """
        Initialize API executor service.
        
        Args:
            resource_schema_path: Path to resource schema JSON file
        """
        self.resource_schema_path = resource_schema_path or os.path.join(
            os.path.dirname(__file__), "../config/resource_schema.json"
        )
        self.resource_schema: Dict[str, Any] = {}
        self.http_client: Optional[httpx.AsyncClient] = None
        
        # Load resource schema
        self._load_resource_schema()
        
        # API configuration
        self.api_timeout = float(os.getenv("API_EXECUTOR_TIMEOUT", "30"))
        self.max_retries = int(os.getenv("API_EXECUTOR_MAX_RETRIES", "3"))
        
        # Token management
        self.auth_token: Optional[str] = None
        self.token_expires_at: Optional[datetime] = None
        self.token_lock = asyncio.Lock()  # Prevent concurrent token refreshes
        
        # Auth API configuration
        self.auth_url = os.getenv(
            "API_AUTH_URL",
            "https://ipcloud.tatacommunications.com/portalservice/api/v1/getAuthToken"
        )
        self.auth_email = os.getenv("API_AUTH_EMAIL", "")
        self.auth_password = os.getenv("API_AUTH_PASSWORD", "")
        
        # Engagement caching - store engagement ID per email
        self.cached_engagement: Optional[Dict[str, Any]] = None
        self.engagement_cache_time: Optional[datetime] = None
        self.engagement_cache_duration = timedelta(hours=1)  # Cache for 1 hour
        
        logger.info("✅ APIExecutorService initialized")
    
    def _load_resource_schema(self) -> None:
        """Load resource schema from JSON file."""
        try:
            with open(self.resource_schema_path, 'r') as f:
                self.resource_schema = json.load(f)
            logger.info(f"✅ Loaded resource schema with {len(self.resource_schema.get('resources', {}))} resources")
        except FileNotFoundError:
            logger.warning(f"⚠️ Resource schema not found at {self.resource_schema_path}")
            self.resource_schema = {"resources": {}}
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse resource schema: {str(e)}")
            self.resource_schema = {"resources": {}}
    
    async def _fetch_auth_token(self) -> Optional[str]:
        """
        Fetch authentication token from the auth API.
        
        Returns:
            Bearer token string or None if failed
        """
        if not self.auth_email or not self.auth_password:
            logger.error("❌ API_AUTH_EMAIL or API_AUTH_PASSWORD not configured")
            return None
        
        try:
            client = await self._get_http_client()
            
            auth_payload = {
                "email": self.auth_email,
                "password": self.auth_password
            }
            
            logger.info(f"🔑 Fetching auth token from {self.auth_url}")
            
            response = await client.post(
                self.auth_url,
                json=auth_payload,
                headers={"Content-Type": "application/json"},
                timeout=10.0
            )
            
            response.raise_for_status()
            data = response.json()
            
            # Extract token from response
            # Tata Communications API returns: {"statusCode": 200, "accessToken": "..."}
            token = (
                data.get("accessToken") or 
                data.get("access_token") or 
                data.get("token") or 
                data.get("authToken")
            )
            
            if token:
                logger.info(f"✅ Successfully fetched auth token (token length: {len(token)})")
                logger.debug(f"Token starts with: {token[:50]}...")
                return token
            else:
                logger.error(f"❌ Token not found in response. Keys available: {list(data.keys())}")
                return None
                
        except httpx.HTTPStatusError as e:
            logger.error(f"❌ Auth API returned error {e.response.status_code}: {e.response.text}")
            return None
        except Exception as e:
            logger.error(f"❌ Failed to fetch auth token: {str(e)}")
            return None
    
    async def _ensure_valid_token(self) -> bool:
        """
        Ensure we have a valid authentication token.
        Fetches a new token if expired or missing.
        
        Returns:
            True if valid token available, False otherwise
        """
        async with self.token_lock:
            # Check if token is still valid (with 5-minute buffer)
            if self.auth_token and self.token_expires_at:
                if datetime.utcnow() < (self.token_expires_at - timedelta(minutes=5)):
                    logger.debug("✅ Using cached auth token")
                    return True
            
            # Fetch new token
            logger.info("🔄 Refreshing auth token...")
            new_token = await self._fetch_auth_token()
            
            if new_token:
                self.auth_token = new_token
                # Token is valid for 10 minutes, cache for 8 minutes to allow buffer
                self.token_expires_at = datetime.utcnow() + timedelta(minutes=8)
                logger.info("✅ Auth token refreshed successfully")
                return True
            else:
                logger.error("❌ Failed to refresh auth token")
                return False
    
    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self.http_client is None:
            self.http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.api_timeout),
                follow_redirects=True
            )
        return self.http_client
    
    async def close(self) -> None:
        """Close HTTP client."""
        if self.http_client:
            await self.http_client.aclose()
            self.http_client = None
    
    async def get_engagement_id(self, force_refresh: bool = False) -> Optional[int]:
        """
        Get engagement ID for the authenticated user.
        Caches the result to avoid repeated API calls.
        
        Args:
            force_refresh: Force fetch even if cached
            
        Returns:
            Engagement ID or None if failed
        """
        # Check cache first
        if not force_refresh and self.cached_engagement and self.engagement_cache_time:
            if datetime.utcnow() < (self.engagement_cache_time + self.engagement_cache_duration):
                logger.debug(f"✅ Using cached engagement ID: {self.cached_engagement.get('id')}")
                return self.cached_engagement.get("id")
        
        # Fetch engagement details
        logger.info("🔍 Fetching engagement details from API...")
        result = await self.execute_operation(
            resource_type="engagement",
            operation="get",
            params={},
            user_roles=None  # Auth token is enough
        )
        
        if result.get("success") and result.get("data"):
            data = result["data"]
            # API returns {"status": "success", "data": [{...}]}
            if isinstance(data, dict) and "data" in data:
                engagements = data["data"]
                if engagements and len(engagements) > 0:
                    engagement = engagements[0]  # Use first engagement
                    self.cached_engagement = engagement
                    self.engagement_cache_time = datetime.utcnow()
                    
                    logger.info(f"✅ Cached engagement: {engagement.get('engagementName')} (ID: {engagement.get('id')})")
                    return engagement.get("id")
        
        logger.error("❌ Failed to fetch engagement ID")
        return None
    
    async def get_endpoints(self, engagement_id: int = None) -> Optional[List[Dict[str, Any]]]:
        """
        Get available endpoints (data centers) for an engagement.
        
        Args:
            engagement_id: Engagement ID (fetches if not provided)
            
        Returns:
            List of endpoint dicts or None if failed
        """
        # Get engagement ID if not provided
        if engagement_id is None:
            engagement_id = await self.get_engagement_id()
            if not engagement_id:
                return None
        
        logger.info(f"🔍 Fetching endpoints for engagement {engagement_id}...")
        result = await self.execute_operation(
            resource_type="endpoint",
            operation="list",
            params={"engagement_id": engagement_id},
            user_roles=None
        )
        
        if result.get("success") and result.get("data"):
            data = result["data"]
            # API returns {"status": "success", "data": [{...}]}
            if isinstance(data, dict) and "data" in data:
                endpoints = data["data"]
                logger.info(f"✅ Found {len(endpoints)} endpoints")
                return endpoints
        
        logger.error("❌ Failed to fetch endpoints")
        return None
    
    async def list_clusters(
        self,
        endpoint_ids: List[int] = None,
        engagement_id: int = None
    ) -> Dict[str, Any]:
        """
        List Kubernetes clusters for given endpoints.
        This is the main workflow method that handles the multi-step process.
        
        Args:
            endpoint_ids: List of endpoint IDs to query (fetches all if not provided)
            engagement_id: Engagement ID (fetches if not provided)
            
        Returns:
            Dict with cluster list or error
        """
        try:
            # Step 1: Get engagement ID
            if engagement_id is None:
                engagement_id = await self.get_engagement_id()
                if not engagement_id:
                    return {
                        "success": False,
                        "error": "Failed to fetch engagement ID",
                        "step": "get_engagement"
                    }
            
            # Step 2: Get endpoints if not provided
            if endpoint_ids is None:
                endpoints = await self.get_endpoints(engagement_id)
                if not endpoints:
                    return {
                        "success": False,
                        "error": "Failed to fetch endpoints",
                        "step": "get_endpoints"
                    }
                
                # Use all endpoint IDs
                endpoint_ids = [ep["endpointId"] for ep in endpoints]
                logger.info(f"📍 Using all {len(endpoint_ids)} endpoints: {endpoint_ids}")
            
            # Step 3: Fetch cluster list
            logger.info(f"📋 Fetching clusters for engagement {engagement_id} with endpoints {endpoint_ids}")
            result = await self.execute_operation(
                resource_type="k8s_cluster",
                operation="list",
                params={
                    "engagement_id": engagement_id,
                    "endpoints": endpoint_ids
                },
                user_roles=None
            )
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to list clusters: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_resource_config(self, resource_type: str) -> Optional[Dict[str, Any]]:
        """
        Get configuration for a resource type.
        
        Args:
            resource_type: Type of resource (k8s_cluster, firewall, etc.)
            
        Returns:
            Resource configuration dict or None if not found
        """
        return self.resource_schema.get("resources", {}).get(resource_type)
    
    def get_operation_config(
        self,
        resource_type: str,
        operation: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get configuration for a specific operation on a resource.
        
        Args:
            resource_type: Type of resource
            operation: Operation name (create, read, update, delete, list)
            
        Returns:
            Operation configuration dict or None if not found
        """
        resource_config = self.get_resource_config(resource_type)
        if not resource_config:
            return None
        
        api_endpoint = resource_config.get("api_endpoints", {}).get(operation)
        parameters = resource_config.get("parameters", {}).get(operation, {})
        permissions = resource_config.get("permissions", {}).get(operation, [])
        
        if api_endpoint:
            return {
                "endpoint": api_endpoint,
                "parameters": parameters,
                "permissions": permissions
            }
        return None
    
    def validate_parameters(
        self,
        resource_type: str,
        operation: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate parameters for an operation.
        
        Args:
            resource_type: Type of resource
            operation: Operation name
            params: Parameters to validate
            
        Returns:
            Dict with 'valid' bool and 'errors' list
        """
        operation_config = self.get_operation_config(resource_type, operation)
        if not operation_config:
            return {
                "valid": False,
                "errors": [f"Unknown resource type or operation: {resource_type}.{operation}"]
            }
        
        param_config = operation_config.get("parameters", {})
        required_params = param_config.get("required", [])
        validation_rules = param_config.get("validation", {})
        
        errors = []
        
        # Check required parameters
        for required_param in required_params:
            if required_param not in params or params[required_param] is None:
                errors.append(f"Missing required parameter: {required_param}")
        
        # Validate parameter values
        for param_name, param_value in params.items():
            if param_name in validation_rules:
                rules = validation_rules[param_name]
                param_errors = self._validate_param_value(param_name, param_value, rules)
                errors.extend(param_errors)
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
    
    def _validate_param_value(
        self,
        param_name: str,
        param_value: Any,
        rules: Dict[str, Any]
    ) -> List[str]:
        """
        Validate a single parameter value against rules.
        
        Args:
            param_name: Parameter name
            param_value: Parameter value
            rules: Validation rules
            
        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        
        # Type validation
        expected_type = rules.get("type")
        if expected_type == "string" and not isinstance(param_value, str):
            errors.append(f"{param_name} must be a string")
        elif expected_type == "integer" and not isinstance(param_value, int):
            errors.append(f"{param_name} must be an integer")
        elif expected_type == "number" and not isinstance(param_value, (int, float)):
            errors.append(f"{param_name} must be a number")
        elif expected_type == "boolean" and not isinstance(param_value, bool):
            errors.append(f"{param_name} must be a boolean")
        
        # String validations
        if isinstance(param_value, str):
            if "min_length" in rules and len(param_value) < rules["min_length"]:
                errors.append(f"{param_name} must be at least {rules['min_length']} characters")
            if "max_length" in rules and len(param_value) > rules["max_length"]:
                errors.append(f"{param_name} must be at most {rules['max_length']} characters")
            if "pattern" in rules:
                import re
                if not re.match(rules["pattern"], param_value):
                    errors.append(f"{param_name} does not match required pattern")
        
        # Numeric validations
        if isinstance(param_value, (int, float)):
            if "min" in rules and param_value < rules["min"]:
                errors.append(f"{param_name} must be at least {rules['min']}")
            if "max" in rules and param_value > rules["max"]:
                errors.append(f"{param_name} must be at most {rules['max']}")
        
        # Enum validation
        if "values" in rules and param_value not in rules["values"]:
            errors.append(f"{param_name} must be one of: {', '.join(map(str, rules['values']))}")
        
        return errors
    
    def check_permissions(
        self,
        resource_type: str,
        operation: str,
        user_roles: List[str]
    ) -> bool:
        """
        Check if user has permission to perform operation.
        
        Args:
            resource_type: Type of resource
            operation: Operation name
            user_roles: List of user's roles
            
        Returns:
            True if user has permission, False otherwise
        """
        operation_config = self.get_operation_config(resource_type, operation)
        if not operation_config:
            return False
        
        required_permissions = operation_config.get("permissions", [])
        
        # Check if user has any of the required roles
        has_permission = any(role in required_permissions for role in user_roles)
        
        if not has_permission:
            logger.warning(
                f"⚠️ Permission denied: user roles {user_roles} do not match "
                f"required permissions {required_permissions} for {resource_type}.{operation}"
            )
        
        return has_permission
    
    async def execute_operation(
        self,
        resource_type: str,
        operation: str,
        params: Dict[str, Any],
        user_roles: List[str] = None,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Execute a CRUD operation on a resource.
        
        Args:
            resource_type: Type of resource
            operation: Operation name (create, read, update, delete, list)
            params: Operation parameters
            user_roles: User's roles for permission checking
            dry_run: If True, validate but don't execute
            
        Returns:
            Dict with execution result
        """
        start_time = datetime.utcnow()
        
        try:
            # Get operation configuration
            operation_config = self.get_operation_config(resource_type, operation)
            if not operation_config:
                return {
                    "success": False,
                    "error": f"Unknown resource type or operation: {resource_type}.{operation}",
                    "timestamp": start_time.isoformat()
                }
            
            # Check permissions
            if user_roles is not None:
                if not self.check_permissions(resource_type, operation, user_roles):
                    return {
                        "success": False,
                        "error": f"Permission denied for {operation} on {resource_type}",
                        "required_permissions": operation_config.get("permissions", []),
                        "timestamp": start_time.isoformat()
                    }
            
            # Validate parameters
            validation_result = self.validate_parameters(resource_type, operation, params)
            if not validation_result["valid"]:
                return {
                    "success": False,
                    "error": "Parameter validation failed",
                    "validation_errors": validation_result["errors"],
                    "timestamp": start_time.isoformat()
                }
            
            # Dry run - return success without executing
            if dry_run:
                return {
                    "success": True,
                    "dry_run": True,
                    "message": f"Validation successful for {operation} on {resource_type}",
                    "params": params,
                    "timestamp": start_time.isoformat()
                }
            
            # Execute API call
            endpoint_config = operation_config["endpoint"]
            result = await self._make_api_call(endpoint_config, params)
            
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            return {
                "success": result.get("success", True),
                "data": result.get("data"),
                "error": result.get("error"),
                "resource_type": resource_type,
                "operation": operation,
                "duration_seconds": duration,
                "timestamp": end_time.isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to execute {operation} on {resource_type}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "resource_type": resource_type,
                "operation": operation,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _make_api_call(
        self,
        endpoint_config: Dict[str, Any],
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Make the actual API call with automatic token refresh.
        
        Args:
            endpoint_config: Endpoint configuration (method, url, etc.)
            params: Request parameters
            
        Returns:
            API response dict
        """
        # Ensure we have a valid token before making the call
        token_valid = await self._ensure_valid_token()
        if not token_valid:
            logger.error("❌ Cannot make API call: No valid auth token")
            return {
                "success": False,
                "error": "Authentication failed: Unable to obtain valid token"
            }
        
        method = endpoint_config.get("method", "GET").upper()
        url = endpoint_config.get("url", "")
        
        # Separate path parameters from body/query parameters
        path_params = {}
        body_params = {}
        
        for param_name, param_value in params.items():
            # If parameter is in URL template, it's a path parameter
            if f"{{{param_name}}}" in url:
                path_params[param_name] = param_value
                url = url.replace(f"{{{param_name}}}", str(param_value))
            else:
                # Otherwise it goes in body/query
                body_params[param_name] = param_value
        
        # Get HTTP client
        client = await self._get_http_client()
        
        # Prepare request
        headers = endpoint_config.get("headers", {})
        headers.setdefault("Content-Type", "application/json")
        
        # Add authentication with dynamically fetched token
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
            logger.debug("✅ Using dynamically fetched auth token")
        else:
            logger.warning("⚠️ No auth token available for API call")
        
        try:
            logger.info(f"🌐 API Call: {method} {url}")
            if body_params:
                logger.debug(f"📦 Request body: {json.dumps(body_params, indent=2)}")
            
            # Make request based on method
            if method == "GET":
                response = await client.get(url, headers=headers, params=body_params)
            elif method == "POST":
                response = await client.post(url, headers=headers, json=body_params)
            elif method == "PUT":
                response = await client.put(url, headers=headers, json=body_params)
            elif method == "PATCH":
                response = await client.patch(url, headers=headers, json=body_params)
            elif method == "DELETE":
                response = await client.delete(url, headers=headers, params=body_params)
            else:
                return {
                    "success": False,
                    "error": f"Unsupported HTTP method: {method}"
                }
            
            # Parse response
            response.raise_for_status()
            
            try:
                data = response.json()
            except json.JSONDecodeError:
                data = {"raw_response": response.text}
            
            logger.info(f"✅ API Call successful: {method} {url} (status {response.status_code})")
            
            return {
                "success": True,
                "data": data,
                "status_code": response.status_code
            }
            
        except httpx.HTTPStatusError as e:
            logger.error(f"❌ API Call failed: {method} {url} - {str(e)}")
            try:
                error_data = e.response.json()
            except:
                error_data = {"message": e.response.text}
            
            return {
                "success": False,
                "error": f"HTTP {e.response.status_code}: {error_data}",
                "status_code": e.response.status_code
            }
        
        except httpx.RequestError as e:
            logger.error(f"❌ API Request error: {method} {url} - {str(e)}")
            return {
                "success": False,
                "error": f"Request failed: {str(e)}"
            }
    
    async def check_cluster_name_available(self, cluster_name: str) -> Dict[str, Any]:
        """
        Check if cluster name is available.
        
        Args:
            cluster_name: Name to check
            
        Returns:
            Dict with availability status
        """
        logger.info(f"🔍 Checking cluster name availability: {cluster_name}")
        
        # TODO: Replace with real API call
        # result = await self.execute_operation(...)
        
        # Sample response: Empty {} means available, data present means taken
        # For now, simulate all names are available
        return {
            "success": True,
            "available": True,
            "message": f"Cluster name '{cluster_name}' is available"
        }
    
    async def get_iks_images_and_datacenters(self, engagement_id: int) -> Dict[str, Any]:
        """
        Get IKS images with datacenter information.
        
        Returns dict with:
        - datacenters: List of unique data centers
        - images: All images grouped by datacenter
        """
        logger.info(f"🖼️ Fetching IKS images for engagement {engagement_id}")
        
        # TODO: Replace with real API call
        # url = f"https://ipcloud.tatacommunications.com/paasservice/paas/{engagement_id}/iks/images/version"
        
        # Sample response structure
        sample_data = {
            "status": "success",
            "data": {
                "vks-enabledImages": [
                    {
                        "ImageName": "ubuntu-2204--IKS-AUG25--v1.27.16",
                        "endpoint": "EP_V2_DEL",
                        "endpointId": 11,
                        "endpointName": "Delhi",
                        "id": 43280
                    },
                    {
                        "ImageName": "ubuntu-2204--IKS-AUG25--v1.30.14",
                        "endpoint": "EP_V2_BLR",
                        "endpointId": 12,
                        "endpointName": "Bengaluru",
                        "id": 46792
                    }
                ],
                "all-images": [
                    {
                        "ImageName": "UBUNTU24.04_STD_IKS_01AUG2025-v1.31.13",
                        "endpoint": "EP_V2_MUMBKC",
                        "endpointId": 162,
                        "endpointName": "Mumbai-BKC",
                        "id": 47582
                    }
                ]
            }
        }
        
        # Extract unique datacenters
        all_images = []
        for category, images in sample_data["data"].items():
            all_images.extend(images)
        
        # Get unique datacenters
        datacenters = {}
        for img in all_images:
            dc_id = img["endpointId"]
            if dc_id not in datacenters:
                datacenters[dc_id] = {
                    "id": dc_id,
                    "name": img["endpointName"],
                    "endpoint": img["endpoint"]
                }
        
        return {
            "success": True,
            "datacenters": list(datacenters.values()),
            "images": all_images
        }
    
    async def get_k8s_versions_for_datacenter(self, datacenter_id: int, all_images: List[Dict]) -> List[str]:
        """
        Extract unique k8s versions for a specific datacenter.
        
        Args:
            datacenter_id: Datacenter endpoint ID
            all_images: List of all images
            
        Returns:
            List of k8s versions (sorted, latest first)
        """
        # Filter images by datacenter
        dc_images = [img for img in all_images if img["endpointId"] == datacenter_id]
        
        # Extract versions
        versions = set()
        for img in dc_images:
            match = re.search(r'v\d+\.\d+\.\d+', img["ImageName"])
            if match:
                versions.add(match.group(0))
        
        # Sort semantically (latest first)
        sorted_versions = sorted(list(versions), key=lambda v: [int(x) for x in v[1:].split('.')], reverse=True)
        
        logger.info(f"📦 Found {len(sorted_versions)} k8s versions for datacenter {datacenter_id}")
        return sorted_versions
    
    async def get_network_drivers(self, endpoint_id: int, k8s_version: str) -> Dict[str, Any]:
        """
        Get CNI drivers for datacenter + k8s version.
        
        Args:
            endpoint_id: Datacenter ID
            k8s_version: Kubernetes version
            
        Returns:
            Dict with CNI drivers list
        """
        logger.info(f"🌐 Fetching CNI drivers for endpoint {endpoint_id}, k8s {k8s_version}")
        
        # TODO: Replace with real API call
        # url = f"https://ipcloud.tatacommunications.com/paasservice/paas/getNetworkList/{endpoint_id}/{k8s_version}/APP"
        
        # Sample response
        sample_drivers = {
            "status": "success",
            "data": {
                "data": [
                    "calico-v3.25.1",
                    "cilium-ebpf-v1.16.4",
                    "cilium-iptables-v1.16.4"
                ]
            }
        }
        
        return {
            "success": True,
            "drivers": sample_drivers["data"]["data"]
        }
    
    async def get_environments_and_business_units(self, engagement_id: int) -> Dict[str, Any]:
        """
        Get environments and business units for engagement.
        
        Args:
            engagement_id: Engagement ID
            
        Returns:
            Dict with business units and environments
        """
        logger.info(f"🏢 Fetching environments for engagement {engagement_id}")
        
        # TODO: Replace with real API call
        # url = f"https://ipcloud.tatacommunications.com/portalservice/environment/getEnvironmentListPerEngagement/{engagement_id}"
        
        # Sample response
        sample_data = {
            "status": "success",
            "data": {
                "environments": [
                    {
                        "id": 1,
                        "name": "Production",
                        "department": "Engineering",
                        "departmentId": 101,
                        "endpointName": "EP_V2_DEL"
                    },
                    {
                        "id": 2,
                        "name": "Development",
                        "department": "Engineering",
                        "departmentId": 101,
                        "endpointName": "EP_V2_DEL"
                    },
                    {
                        "id": 3,
                        "name": "Staging",
                        "department": "QA",
                        "departmentId": 102,
                        "endpointName": "EP_V2_BLR"
                    }
                ]
            }
        }
        
        environments = sample_data["data"]["environments"]
        
        # Extract unique business units
        business_units = {}
        for env in environments:
            bu_id = env["departmentId"]
            if bu_id not in business_units:
                business_units[bu_id] = {
                    "id": bu_id,
                    "name": env["department"]
                }
        
        return {
            "success": True,
            "business_units": list(business_units.values()),
            "environments": environments
        }
    
    async def get_zones_list(self, engagement_id: int) -> Dict[str, Any]:
        """
        Get zones for engagement.
        
        Args:
            engagement_id: Engagement ID
            
        Returns:
            Dict with zones list
        """
        logger.info(f"🗺️ Fetching zones for engagement {engagement_id}")
        
        # TODO: Replace with real API call
        # url = f"https://ipcloud.tatacommunications.com/portalservice/zone/getZoneList/{engagement_id}"
        
        # Sample response
        sample_data = {
            "status": "success",
            "data": [
                {
                    "zoneId": 16710,
                    "zoneName": "zone-prod-01",
                    "departmentName": "Engineering",
                    "environmentName": "Production"
                },
                {
                    "zoneId": 16711,
                    "zoneName": "zone-dev-01",
                    "departmentName": "Engineering",
                    "environmentName": "Development"
                },
                {
                    "zoneId": 16712,
                    "zoneName": "zone-qa-01",
                    "departmentName": "QA",
                    "environmentName": "Staging"
                }
            ]
        }
        
        return {
            "success": True,
            "zones": sample_data["data"]
        }
    
    async def get_os_images(self, zone_id: int, circuit_id: str, k8s_version: str) -> Dict[str, Any]:
        """
        Get OS images for zone, filtered by k8s version.
        
        Args:
            zone_id: Zone ID
            circuit_id: Circuit ID
            k8s_version: Kubernetes version to filter by
            
        Returns:
            Dict with OS options
        """
        logger.info(f"💿 Fetching OS images for zone {zone_id}, k8s {k8s_version}")
        
        # TODO: Replace with real API call
        # url = f"https://ipcloud.tatacommunications.com/portalservice/configservice/ppuEnabledImages/{zone_id}?circuitId={circuit_id}&isDeployment=false"
        
        # Sample response
        sample_data = {
            "status": "success",
            "data": {
                "image": {
                    "options": [
                        {
                            "id": 43280,
                            "label": "UBUNTU22.04_STD_IKS_01AUG2025-v1.27.16",
                            "osMake": "Ubuntu",
                            "osModel": "ubuntu",
                            "osVersion": "22.04 LTS",
                            "hypervisor": "VCD_ESXI"
                        },
                        {
                            "id": 47582,
                            "label": "UBUNTU24.04_STD_IKS_01AUG2025-v1.27.16",
                            "osMake": "Ubuntu",
                            "osModel": "ubuntu",
                            "osVersion": "24.04 LTS",
                            "hypervisor": "VCD_ESXI"
                        }
                    ]
                }
            }
        }
        
        # Filter by k8s version
        images = sample_data["data"]["image"]["options"]
        filtered = [img for img in images if k8s_version in img["label"]]
        
        # Group by osMake + osVersion
        grouped = {}
        for img in filtered:
            key = f"{img['osMake']} {img['osVersion']}"
            if key not in grouped:
                grouped[key] = {
                    "display_name": key,
                    "os_id": img["id"],
                    "os_make": img["osMake"],
                    "os_model": img["osModel"],
                    "os_version": img["osVersion"],
                    "hypervisor": img["hypervisor"],
                    "images": []
                }
            grouped[key]["images"].append(img)
        
        return {
            "success": True,
            "os_options": list(grouped.values())
        }
    
    async def get_flavors(self, zone_id: int, circuit_id: str, os_model: str, node_type: str = None) -> Dict[str, Any]:
        """
        Get compute flavors for zone, filtered by OS and optionally node type.
        
        Args:
            zone_id: Zone ID
            circuit_id: Circuit ID
            os_model: OS model (e.g., "ubuntu")
            node_type: Node type to filter (generalPurpose, computeOptimized, memoryOptimized)
            
        Returns:
            Dict with flavor options
        """
        logger.info(f"💻 Fetching flavors for zone {zone_id}, OS {os_model}, node type {node_type}")
        
        # TODO: Replace with real API call
        # url = f"https://ipcloud.tatacommunications.com/portalservice/configservice/ppuEnabledFlavors/{zone_id}?isDeployment=false&circuitId={circuit_id}"
        
        # Sample response
        sample_data = {
            "status": "success",
            "data": {
                "flavor": [
                    {
                        "artifactId": 3234,
                        "FlavorName": "B4",
                        "skuCode": "B4.UBN",
                        "vCpu": 2,
                        "vRam": 4096,
                        "vDisk": 50,
                        "osModel": "ubuntu",
                        "applicationType": "Container",
                        "flavorCategory": "generalPurpose",
                        "label": "B4_ubuntu_container"
                    },
                    {
                        "artifactId": 3235,
                        "FlavorName": "C8",
                        "skuCode": "C8.UBN",
                        "vCpu": 4,
                        "vRam": 8192,
                        "vDisk": 100,
                        "osModel": "ubuntu",
                        "applicationType": "Container",
                        "flavorCategory": "computeOptimized",
                        "label": "C8_ubuntu_container"
                    },
                    {
                        "artifactId": 3236,
                        "FlavorName": "M16",
                        "skuCode": "M16.UBN",
                        "vCpu": 4,
                        "vRam": 16384,
                        "vDisk": 100,
                        "osModel": "ubuntu",
                        "applicationType": "Container",
                        "flavorCategory": "memoryOptimized",
                        "label": "M16_ubuntu_container"
                    }
                ]
            }
        }
        
        # Filter by OS model and application type
        flavors = sample_data["data"]["flavor"]
        filtered = [f for f in flavors if f["osModel"] == os_model and f["applicationType"] == "Container"]
        
        # Further filter by node type if provided
        if node_type:
            filtered = [f for f in filtered if f["flavorCategory"] == node_type]
        
        # Extract unique node types for the first query
        node_types = list(set([f["flavorCategory"] for f in filtered]))
        
        # Format flavors
        formatted_flavors = []
        for flavor in filtered:
            formatted_flavors.append({
                "id": flavor["artifactId"],
                "name": f"{flavor['vCpu']} vCPU / {flavor['vRam'] // 1024} GB RAM / {flavor['vDisk']} GB Storage",
                "flavor_name": flavor["FlavorName"],
                "sku_code": flavor["skuCode"],
                "vcpu": flavor["vCpu"],
                "vram_gb": flavor["vRam"] // 1024,
                "disk_gb": flavor["vDisk"],
                "node_type": flavor["flavorCategory"]
            })
        
        return {
            "success": True,
            "node_types": node_types,
            "flavors": formatted_flavors
        }
    
    async def get_circuit_id(self, engagement_id: int) -> Optional[str]:
        """
        Get circuit ID (copfId) for engagement.
        
        Args:
            engagement_id: Engagement ID
            
        Returns:
            Circuit ID string or default
        """
        logger.info(f"🔌 Fetching circuit ID for engagement {engagement_id}")
        
        # TODO: Implement if there's a specific API endpoint
        # For now, return default from createcluster.ts line 110
        return "E-IPCTEAM-1602"
    
    def __repr__(self) -> str:
        resource_count = len(self.resource_schema.get("resources", {}))
        return f"<APIExecutorService(resources={resource_count})>"


# Global instance
api_executor_service = APIExecutorService()

