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

# Import user credentials service
try:
    from app.services.user_credentials_service import user_credentials_service
except ImportError:
    user_credentials_service = None
    logger.warning("⚠️ UserCredentialsService not available, will use env vars only")


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
        
        # Token management - per user (since different users may have different credentials)
        self.user_tokens: Dict[str, Dict[str, Any]] = {}  # {user_id: {"token": str, "expires_at": datetime}}
        self.token_lock = asyncio.Lock()  # Prevent concurrent token refreshes
        
        # Auth API configuration
        self.auth_url = os.getenv(
            "API_AUTH_URL",
            "https://ipcloud.tatacommunications.com/portalservice/api/v1/getAuthToken"
        )
        # Keep env vars as fallback for backward compatibility
        self.default_auth_email = os.getenv("API_AUTH_EMAIL", "")
        self.default_auth_password = os.getenv("API_AUTH_PASSWORD", "")
        
        # Per-user session storage for engagement IDs and other frequently accessed data
        # Structure: {user_id: {"paas_engagement_id": int, "ipc_engagement_id": int, 
        #                       "engagement_data": dict, "cached_at": datetime, "endpoints": list}}
        self.user_sessions: Dict[str, Dict[str, Any]] = {}
        self.session_cache_duration = timedelta(hours=24)  # Cache for 24 hours (they rarely change)
        self.session_lock = asyncio.Lock()  # Prevent concurrent session updates
        
        # Legacy engagement caching - kept for backward compatibility
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
    
    def _get_user_id_from_email(self, email: str = None) -> str:
        """
        Get user ID from email for session storage.
        
        Args:
            email: User email (uses default if not provided)
            
        Returns:
            User ID (email or 'default')
        """
        if email:
            return email
        if self.default_auth_email:
            return self.default_auth_email
        return "default"
    
    async def _get_user_session(self, user_id: str = None, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get user session data from cache.
        
        Args:
            user_id: User ID (email)
            force_refresh: Force refresh the session
            
        Returns:
            User session dict or empty dict if not cached or expired
        """
        if not user_id:
            user_id = self._get_user_id_from_email()
        
        async with self.session_lock:
            session = self.user_sessions.get(user_id, {})
            
            if not session or force_refresh:
                return {}
            
            # Check if session is expired
            cached_at = session.get("cached_at")
            if cached_at and datetime.utcnow() < (cached_at + self.session_cache_duration):
                logger.debug(f"✅ Using cached session for user: {user_id}")
                return session
            else:
                logger.debug(f"⏰ Session expired for user: {user_id}")
                return {}
    
    async def _update_user_session(self, user_id: str = None, **kwargs) -> None:
        """
        Update user session data in cache.
        
        Args:
            user_id: User ID (email)
            **kwargs: Session data to update (paas_engagement_id, ipc_engagement_id, etc.)
        """
        if not user_id:
            user_id = self._get_user_id_from_email()
        
        async with self.session_lock:
            if user_id not in self.user_sessions:
                self.user_sessions[user_id] = {
                    "cached_at": datetime.utcnow()
                }
            
            # Update session data
            self.user_sessions[user_id].update(kwargs)
            self.user_sessions[user_id]["cached_at"] = datetime.utcnow()
            
            logger.debug(f"💾 Updated session for user: {user_id} with keys: {list(kwargs.keys())}")
    
    async def _clear_user_session(self, user_id: str = None) -> None:
        """
        Clear user session data from cache.
        
        Args:
            user_id: User ID (email), clears all if None
        """
        async with self.session_lock:
            if user_id:
                if user_id in self.user_sessions:
                    del self.user_sessions[user_id]
                    logger.info(f"🗑️ Cleared session for user: {user_id}")
            else:
                self.user_sessions.clear()
                logger.info("🗑️ Cleared all user sessions")
    
    async def _fetch_auth_token(
        self,
        auth_email: str = None,
        auth_password: str = None
    ) -> Optional[str]:
        """
        Fetch authentication token from the auth API.
        
        Args:
            auth_email: Email for authentication (uses default from env if not provided)
            auth_password: Password for authentication (uses default from env if not provided)
        
        Returns:
            Bearer token string or None if failed
        """
        # Use provided credentials or fall back to env vars
        email = auth_email or self.default_auth_email
        password = auth_password or self.default_auth_password
        
        if not email or not password:
            logger.error("❌ API credentials not configured (email or password missing)")
            return None
        
        try:
            client = await self._get_http_client()
            
            auth_payload = {
                "email": email,
                "password": password
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
            
            # Check for error response
            status_code = data.get("statusCode")
            if status_code == 500:
                error_msg = data.get("accessToken", "Unknown error")
                logger.error(f"❌ Auth API returned error: {error_msg}")
                return None
            
            # Extract token from response
            # Tata Communications API returns: {"statusCode": 200, "accessToken": "..."}
            token = (
                data.get("accessToken") or 
                data.get("access_token") or 
                data.get("token") or 
                data.get("authToken")
            )
            
            if token and token != "Failed to generate token after retries":
                logger.info(f"✅ Successfully fetched auth token (token length: {len(token)})")
                logger.debug(f"Token starts with: {token[:50]}...")
                return token
            else:
                logger.error(f"❌ Token not found in response or failed. Response: {data}")
                return None
                
        except httpx.HTTPStatusError as e:
            logger.error(f"❌ Auth API returned error {e.response.status_code}: {e.response.text}")
            return None
        except Exception as e:
            logger.error(f"❌ Failed to fetch auth token: {str(e)}")
            return None
    
    async def _ensure_valid_token(
        self,
        user_id: str = None,
        auth_email: str = None,
        auth_password: str = None
    ) -> bool:
        """
        Ensure we have a valid authentication token for a user.
        Fetches a new token if expired or missing.
        
        Args:
            user_id: User identifier (for per-user token caching)
            auth_email: Email for authentication (uses default from env if not provided)
            auth_password: Password for authentication (uses default from env if not provided)
        
        Returns:
            True if valid token available, False otherwise
        """
        # Use default user_id if not provided
        cache_key = user_id or "default"
        
        async with self.token_lock:
            # Check if token is still valid (with 5-minute buffer)
            user_token_data = self.user_tokens.get(cache_key)
            if user_token_data:
                token = user_token_data.get("token")
                expires_at = user_token_data.get("expires_at")
                if token and expires_at:
                    if datetime.utcnow() < (expires_at - timedelta(minutes=5)):
                        logger.debug(f"✅ Using cached auth token for user: {cache_key}")
                        return True
            
            # Fetch new token
            logger.info(f"🔄 Refreshing auth token for user: {cache_key}")
            new_token = await self._fetch_auth_token(auth_email, auth_password)
            
            if new_token:
                # Store token per user
                self.user_tokens[cache_key] = {
                    "token": new_token,
                    "expires_at": datetime.utcnow() + timedelta(minutes=8)  # Cache for 8 minutes
                }
                logger.info(f"✅ Auth token refreshed successfully for user: {cache_key}")
                return True
            else:
                logger.error(f"❌ Failed to refresh auth token for user: {cache_key}")
                return False
    
    async def _get_or_refresh_token(
        self,
        user_id: str = None,
        auth_email: str = None,
        auth_password: str = None
    ) -> Optional[str]:
        """
        Get a valid authentication token, refreshing if necessary.
        
        Args:
            user_id: User identifier (for per-user token caching)
            auth_email: Email for authentication (uses default from env if not provided)
            auth_password: Password for authentication (uses default from env if not provided)
        
        Returns:
            Valid token string or None if unable to get token
        """
        cache_key = user_id or "default"
        
        # Ensure we have a valid token
        if await self._ensure_valid_token(user_id, auth_email, auth_password):
            # Return the token
            user_token_data = self.user_tokens.get(cache_key, {})
            token = user_token_data.get("token")
            if token:
                logger.debug(f"✅ Returning valid token for user: {cache_key}")
                return token
        
        logger.error(f"❌ No valid token available for user: {cache_key}")
        return None
    
    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self.http_client is None:
            self.http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.api_timeout),
                follow_redirects=True
            )
        return self.http_client
    
    def _get_user_credentials(self, user_id: str = None) -> Optional[Dict[str, str]]:
        """
        Get API credentials for a user.
        First tries to get from database by email, then by username, then falls back to env vars.
        
        Args:
            user_id: User identifier (can be email or username)
            
        Returns:
            Dict with 'email' and 'password' keys, or None if not found
        """
        if user_credentials_service and user_id:
            # First try to get by email (most common case from OpenWebUI X-User-Email header)
            credentials = user_credentials_service.get_credentials_by_email(user_id)
            if credentials:
                logger.info(f"✅ Got credentials by email for: {user_id}")
                return credentials
            
            # Then try by username
            credentials = user_credentials_service.get_user_credentials(user_id)
            if credentials:
                logger.info(f"✅ Got credentials by username for: {user_id}")
                return credentials
        
        # Fall back to env vars if user_id not provided or credentials not found
        if self.default_auth_email and self.default_auth_password:
            if user_id:
                logger.info(f"⚠️ No stored credentials for {user_id}, using default from env")
            return {
                "email": self.default_auth_email,
                "password": self.default_auth_password
            }
        
        logger.warning(f"❌ No credentials found for user: {user_id}")
        return None
    
    async def _get_auth_headers(
        self,
        user_id: str = None,
        auth_email: str = None,
        auth_password: str = None
    ) -> Dict[str, str]:
        """
        Get authentication headers with current token for a user.
        
        Args:
            user_id: User identifier (for per-user token caching)
            auth_email: Email for authentication (uses default from env if not provided)
            auth_password: Password for authentication (uses default from env if not provided)
        
        Returns:
            Dictionary of headers including authorization
        """
        headers = {
            "Content-Type": "application/json"
        }
        
        # Ensure we have a valid token
        await self._ensure_valid_token(user_id, auth_email, auth_password)
        
        # Get token for this user
        cache_key = user_id or "default"
        user_token_data = self.user_tokens.get(cache_key, {})
        token = user_token_data.get("token")
        
        if token:
            headers["Authorization"] = f"Bearer {token}"
            logger.debug(f"✅ Using dynamically fetched auth token for user: {cache_key}")
        else:
            logger.warning(f"⚠️ No auth token available for API call (user: {cache_key})")
        
        return headers
    
    async def close(self) -> None:
        """Close HTTP client."""
        if self.http_client:
            await self.http_client.aclose()
            self.http_client = None
    
    async def get_engagement_id(self, force_refresh: bool = False, user_id: str = None) -> Optional[int]:
        """
        Get engagement ID for the authenticated user.
        Uses per-user session storage to avoid repeated API calls.
        
        Args:
            force_refresh: Force fetch even if cached
            user_id: User ID (email) for session lookup
            
        Returns:
            PAAS Engagement ID or None if failed
        """
        if not user_id:
            user_id = self._get_user_id_from_email()
        
        # Check user session cache first
        if not force_refresh:
            session = await self._get_user_session(user_id)
            if session and "paas_engagement_id" in session:
                paas_id = session["paas_engagement_id"]
                logger.debug(f"✅ Using cached PAAS engagement ID from session: {paas_id}")
                return paas_id
        
        # Fetch engagement details from API
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
                    paas_engagement_id = engagement.get("id")
                    
                    # Update user session with engagement data
                    await self._update_user_session(
                        user_id=user_id,
                        paas_engagement_id=paas_engagement_id,
                        engagement_data=engagement
                    )
                    
                    # Also update legacy cache for backward compatibility
                    self.cached_engagement = engagement
                    self.engagement_cache_time = datetime.utcnow()
                    
                    logger.info(f"✅ Cached PAAS engagement: {engagement.get('engagementName')} (ID: {paas_engagement_id})")
                    return paas_engagement_id
        
        logger.error("❌ Failed to fetch engagement ID")
        return None
    
    async def get_ipc_engagement_id(self, engagement_id: int = None, user_id: str = None, force_refresh: bool = False) -> Optional[int]:
        """
        Convert PAAS engagement ID to IPC engagement ID.
        Uses per-user session storage to avoid repeated API calls.
        
        Args:
            engagement_id: PAAS Engagement ID (fetches if not provided)
            user_id: User ID (email) for session lookup
            force_refresh: Force fetch even if cached
            
        Returns:
            IPC Engagement ID or None if failed
        """
        if not user_id:
            user_id = self._get_user_id_from_email()
        
        # Check user session cache first
        if not force_refresh:
            session = await self._get_user_session(user_id)
            if session and "ipc_engagement_id" in session:
                ipc_id = session["ipc_engagement_id"]
                logger.debug(f"✅ Using cached IPC engagement ID from session: {ipc_id}")
                return ipc_id
        
        # Get PAAS engagement ID if not provided
        if engagement_id is None:
            engagement_id = await self.get_engagement_id(user_id=user_id)
            if not engagement_id:
                return None
        
        logger.info(f"🔄 Converting PAAS engagement {engagement_id} to IPC engagement ID...")
        
        result = await self.execute_operation(
            resource_type="k8s_cluster",
            operation="get_ipc_engagement",
            params={"engagement_id": engagement_id},
            user_roles=None
        )
        
        if result.get("success") and result.get("data"):
            data = result["data"]
            # Response: {"status": "success", "data": {"ipc_eng": "...", "ipc_engid": 1602}}
            if data.get("status") == "success" and data.get("data"):
                ipc_engid = data["data"].get("ipc_engid")
                if ipc_engid:
                    # Update user session with IPC engagement ID
                    await self._update_user_session(
                        user_id=user_id,
                        ipc_engagement_id=ipc_engid,
                        paas_engagement_id=engagement_id
                    )
                    
                    logger.info(f"✅ Cached IPC engagement ID: {ipc_engid}")
                    return ipc_engid
        
        logger.error("❌ Failed to get IPC engagement ID")
        return None
    
    async def get_endpoints(self, engagement_id: int = None, user_id: str = None, force_refresh: bool = False) -> Optional[List[Dict[str, Any]]]:
        """
        Get available endpoints (data centers) for an engagement.
        Uses per-user session storage to avoid repeated API calls.
        
        Args:
            engagement_id: Engagement ID (fetches if not provided)
            user_id: User ID (email) for session lookup
            force_refresh: Force fetch even if cached
            
        Returns:
            List of endpoint dicts or None if failed
        """
        if not user_id:
            user_id = self._get_user_id_from_email()
        
        # Check user session cache first
        if not force_refresh:
            session = await self._get_user_session(user_id)
            if session and "endpoints" in session:
                endpoints = session["endpoints"]
                logger.debug(f"✅ Using cached endpoints from session ({len(endpoints)} endpoints)")
                return endpoints
        
        # Get engagement ID if not provided
        if engagement_id is None:
            engagement_id = await self.get_engagement_id(user_id=user_id)
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
                
                # Update user session with endpoints
                await self._update_user_session(
                    user_id=user_id,
                    endpoints=endpoints
                )
                
                logger.info(f"✅ Cached {len(endpoints)} endpoints")
                return endpoints
        
        logger.error("❌ Failed to fetch endpoints")
        return None
    
    async def list_endpoints(
        self,
        engagement_id: int = None
    ) -> Dict[str, Any]:
        """
        List available endpoints (datacenters/locations) for the user's engagement.
        This is the main workflow method for endpoint listing.
        
        Args:
            engagement_id: Engagement ID (fetches if not provided)
            
        Returns:
            Dict with endpoint list or error
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
            
            # Step 2: Fetch endpoints
            logger.info(f"📍 Fetching endpoints for engagement {engagement_id}")
            endpoints = await self.get_endpoints(engagement_id)
            
            if endpoints:
                # Format the response nicely
                formatted_endpoints = []
                for ep in endpoints:
                    formatted_endpoints.append({
                        "id": ep.get("endpointId"),
                        "name": ep.get("endpointDisplayName"),
                        "type": ep.get("endpointType", ""),
                        "region": ep.get("region", ""),
                        "status": ep.get("status", "active")
                    })
                
                return {
                    "success": True,
                    "data": {
                        "endpoints": formatted_endpoints,
                        "total": len(formatted_endpoints),
                        "engagement_id": engagement_id
                    },
                    "message": f"Found {len(formatted_endpoints)} available endpoints/datacenters"
                }
            else:
                return {
                    "success": False,
                    "error": "No endpoints found for this engagement",
                    "step": "get_endpoints"
                }
            
        except Exception as e:
            logger.error(f"❌ Failed to list endpoints: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
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
    
    async def list_managed_services(
        self,
        service_type: str,
        endpoint_ids: List[int] = None,
        ipc_engagement_id: int = None
    ) -> Dict[str, Any]:
        """
        List managed services (Kafka, GitLab, etc.) for given endpoints.
        This is the main workflow method that handles the multi-step process.
        
        Args:
            service_type: Service type to list (e.g., "IKSKafka", "IKSGitlab")
            endpoint_ids: List of endpoint IDs to query (fetches all if not provided)
            ipc_engagement_id: IPC Engagement ID (fetches and converts if not provided)
            
        Returns:
            Dict with service list or error
        """
        try:
            # Step 1: Get PAAS engagement ID (if needed for endpoints)
            paas_engagement_id = None
            if endpoint_ids is None:
                paas_engagement_id = await self.get_engagement_id()
                if not paas_engagement_id:
                    return {
                        "success": False,
                        "error": "Failed to fetch PAAS engagement ID",
                        "step": "get_engagement"
                    }
            
            # Step 2: Get IPC engagement ID (required for managed services API)
            if ipc_engagement_id is None:
                if paas_engagement_id is None:
                    paas_engagement_id = await self.get_engagement_id()
                
                ipc_engagement_id = await self.get_ipc_engagement_id(paas_engagement_id)
                if not ipc_engagement_id:
                    return {
                        "success": False,
                        "error": "Failed to convert PAAS engagement to IPC engagement ID",
                        "step": "get_ipc_engagement"
                    }
                logger.info(f"🔄 Converted PAAS engagement {paas_engagement_id} to IPC engagement {ipc_engagement_id}")
            
            # Step 3: Get endpoints if not provided
            if endpoint_ids is None:
                if paas_engagement_id is None:
                    paas_engagement_id = await self.get_engagement_id()
                
                endpoints = await self.get_endpoints(paas_engagement_id)
                if not endpoints:
                    return {
                        "success": False,
                        "error": "Failed to fetch endpoints",
                        "step": "get_endpoints"
                    }
                
                # Use all endpoint IDs
                endpoint_ids = [ep["endpointId"] for ep in endpoints]
                logger.info(f"📍 Using all {len(endpoint_ids)} endpoints: {endpoint_ids}")
            
            # Step 4: Fetch managed services list
            logger.info(f"📋 Fetching {service_type} services for IPC engagement {ipc_engagement_id} with endpoints {endpoint_ids}")
            
            # Build the API URL
            url = f"https://ipcloud.tatacommunications.com/paasservice/api/v1/paas/listManagedServices/{service_type}"
            
            # Build the payload
            payload = {
                "engagementId": ipc_engagement_id,
                "endpoints": endpoint_ids,
                "serviceType": service_type
            }
            
            # Make the API call
            client = await self._get_http_client()
            headers = await self._get_auth_headers()
            
            logger.info(f"🌐 POST {url}")
            logger.debug(f"📦 Payload: {json.dumps(payload, indent=2)}")
            
            response = await client.post(
                url,
                json=payload,
                headers=headers,
                timeout=self.api_timeout
            )
            
            if response.status_code == 200:
                response_data = response.json()
                
                # API returns nested structure: {"data": {"data": [...]}}
                # Extract the inner data array
                outer_data = response_data.get("data", {})
                if isinstance(outer_data, dict):
                    services = outer_data.get("data", [])
                else:
                    # Fallback: if data is already a list
                    services = outer_data if isinstance(outer_data, list) else []
                
                logger.info(f"✅ Found {len(services)} {service_type} services")
                
                # Ensure services is a list
                if not isinstance(services, list):
                    logger.warning(f"⚠️ Expected list but got {type(services)}, wrapping in list")
                    services = [services] if services else []
                
                return {
                    "success": True,
                    "data": services,
                    "total": len(services),
                    "service_type": service_type,
                    "ipc_engagement_id": ipc_engagement_id,
                    "endpoints": endpoint_ids,
                    "message": f"Found {len(services)} {service_type} services",
                    "raw_response": response_data  # Include raw response for debugging
                }
            else:
                error_msg = f"API returned status {response.status_code}"
                try:
                    error_data = response.json()
                    error_msg = error_data.get("message", error_msg)
                except:
                    pass
                
                logger.error(f"❌ Failed to fetch {service_type} services: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "status_code": response.status_code
                }
            
        except Exception as e:
            logger.error(f"❌ Failed to list {service_type} services: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def list_kafka(
        self,
        endpoint_ids: List[int] = None,
        ipc_engagement_id: int = None
    ) -> Dict[str, Any]:
        """
        List Kafka managed services.
        Convenience wrapper around list_managed_services.
        
        Args:
            endpoint_ids: List of endpoint IDs to query
            ipc_engagement_id: IPC Engagement ID
            
        Returns:
            Dict with Kafka service list or error
        """
        return await self.list_managed_services(
            service_type="IKSKafka",
            endpoint_ids=endpoint_ids,
            ipc_engagement_id=ipc_engagement_id
        )
    
    async def list_gitlab(
        self,
        endpoint_ids: List[int] = None,
        ipc_engagement_id: int = None
    ) -> Dict[str, Any]:
        """
        List GitLab managed services.
        Convenience wrapper around list_managed_services.
        
        Args:
            endpoint_ids: List of endpoint IDs to query
            ipc_engagement_id: IPC Engagement ID
            
        Returns:
            Dict with GitLab service list or error
        """
        return await self.list_managed_services(
            service_type="IKSGitlab",
            endpoint_ids=endpoint_ids,
            ipc_engagement_id=ipc_engagement_id
        )
    
    async def list_container_registry(
        self,
        endpoint_ids: List[int] = None,
        ipc_engagement_id: int = None
    ) -> Dict[str, Any]:
        """
        List Container Registry managed services.
        Convenience wrapper around list_managed_services.
        
        Args:
            endpoint_ids: List of endpoint IDs to query
            ipc_engagement_id: IPC Engagement ID
            
        Returns:
            Dict with Container Registry service list or error
        """
        return await self.list_managed_services(
            service_type="IKSContainerRegistry",
            endpoint_ids=endpoint_ids,
            ipc_engagement_id=ipc_engagement_id
        )
    
    async def list_jenkins(
        self,
        endpoint_ids: List[int] = None,
        ipc_engagement_id: int = None
    ) -> Dict[str, Any]:
        """
        List Jenkins managed services.
        Convenience wrapper around list_managed_services.
        
        Args:
            endpoint_ids: List of endpoint IDs to query
            ipc_engagement_id: IPC Engagement ID
            
        Returns:
            Dict with Jenkins service list or error
        """
        return await self.list_managed_services(
            service_type="IKSJenkins",
            endpoint_ids=endpoint_ids,
            ipc_engagement_id=ipc_engagement_id
        )
    
    async def list_postgres(
        self,
        endpoint_ids: List[int] = None,
        ipc_engagement_id: int = None
    ) -> Dict[str, Any]:
        """
        List PostgreSQL managed services.
        Convenience wrapper around list_managed_services.
        
        Args:
            endpoint_ids: List of endpoint IDs to query
            ipc_engagement_id: IPC Engagement ID
            
        Returns:
            Dict with PostgreSQL service list or error
        """
        return await self.list_managed_services(
            service_type="IKSPostgres",
            endpoint_ids=endpoint_ids,
            ipc_engagement_id=ipc_engagement_id
        )
    
    async def list_documentdb(
        self,
        endpoint_ids: List[int] = None,
        ipc_engagement_id: int = None
    ) -> Dict[str, Any]:
        """
        List DocumentDB managed services.
        Convenience wrapper around list_managed_services.
        
        Args:
            endpoint_ids: List of endpoint IDs to query
            ipc_engagement_id: IPC Engagement ID
            
        Returns:
            Dict with DocumentDB service list or error
        """
        return await self.list_managed_services(
            service_type="IKSDocumentDB",
            endpoint_ids=endpoint_ids,
            ipc_engagement_id=ipc_engagement_id
        )
    
    async def list_vms(
        self,
        ipc_engagement_id: int = None,
        endpoint_filter: str = None,
        zone_filter: str = None,
        department_filter: str = None
    ) -> Dict[str, Any]:
        """
        List all virtual machines for the engagement.
        
        Args:
            ipc_engagement_id: IPC Engagement ID (will be fetched if not provided)
            endpoint_filter: Optional endpoint name to filter VMs
            zone_filter: Optional zone name to filter VMs
            department_filter: Optional department name to filter VMs
            
        Returns:
            Dict with VM list or error
        """
        import time
        start_time = time.time()
        
        try:
            # Get IPC engagement ID if not provided
            if not ipc_engagement_id:
                logger.info("🔄 Fetching IPC engagement ID for VM listing...")
                ipc_engagement_id = await self.get_ipc_engagement_id()
                if not ipc_engagement_id:
                    return {
                        "success": False,
                        "error": "Could not retrieve IPC engagement ID"
                    }
                logger.info(f"✅ Got IPC engagement ID: {ipc_engagement_id}")
            
            # Build URL
            url = f"https://ipcloud.tatacommunications.com/portalservice/instances/vmlist/{ipc_engagement_id}"
            
            logger.info(f"📡 Calling VM list API: GET {url}")
            
            # Get auth headers (will use default/env credentials if user_id not provided)
            headers = await self._get_auth_headers()
            
            # Get HTTP client
            client = await self._get_http_client()
            
            # Make GET request
            response = await client.get(
                url,
                headers=headers,
                timeout=30.0
            )
            
            logger.info(f"📥 VM API response: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract VM list
                vm_list = data.get("data", {}).get("vmList", [])
                last_synced = data.get("data", {}).get("lastSyncedAt", "N/A")
                
                logger.info(f"✅ Found {len(vm_list)} VMs (last synced: {last_synced})")
                
                # Apply filters if provided
                filtered_vms = vm_list
                
                if endpoint_filter:
                    filtered_vms = [
                        vm for vm in filtered_vms
                        if endpoint_filter.lower() in vm.get("virtualMachine", {}).get("endpoint", {}).get("endpointName", "").lower()
                    ]
                    logger.info(f"🔍 Filtered by endpoint '{endpoint_filter}': {len(filtered_vms)} VMs")
                
                if zone_filter:
                    filtered_vms = [
                        vm for vm in filtered_vms
                        if zone_filter.lower() in vm.get("virtualMachine", {}).get("zone", {}).get("zoneName", "").lower()
                    ]
                    logger.info(f"🔍 Filtered by zone '{zone_filter}': {len(filtered_vms)} VMs")
                
                if department_filter:
                    filtered_vms = [
                        vm for vm in filtered_vms
                        if department_filter.lower() in vm.get("virtualMachine", {}).get("department", {}).get("departmentName", "").lower()
                    ]
                    logger.info(f"🔍 Filtered by department '{department_filter}': {len(filtered_vms)} VMs")
                
                duration = time.time() - start_time
                
                return {
                    "success": True,
                    "data": filtered_vms,
                    "total": len(filtered_vms),
                    "total_unfiltered": len(vm_list),
                    "last_synced": last_synced,
                    "ipc_engagement_id": ipc_engagement_id,
                    "filters_applied": {
                        "endpoint": endpoint_filter,
                        "zone": zone_filter,
                        "department": department_filter
                    },
                    "duration_seconds": duration,
                    "message": f"Found {len(filtered_vms)} VMs"
                }
            else:
                error_msg = f"API returned status {response.status_code}"
                try:
                    error_data = response.json()
                    error_msg = error_data.get("message", error_msg)
                except:
                    pass
                
                logger.error(f"❌ VM list API failed: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "status_code": response.status_code
                }
        
        except Exception as e:
            logger.error(f"❌ Exception in list_vms: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def list_firewalls(
        self,
        endpoint_ids: List[int] = None,
        ipc_engagement_id: int = None,
        variant: str = ""
    ) -> Dict[str, Any]:
        """
        List firewalls across multiple endpoints.
        
        Args:
            endpoint_ids: List of endpoint IDs to query
            ipc_engagement_id: IPC Engagement ID (will be fetched if not provided)
            variant: Firewall variant filter (default: empty string)
            
        Returns:
            Dict with firewall list or error
        """
        import time
        start_time = time.time()
        
        try:
            # Get IPC engagement ID if not provided
            if not ipc_engagement_id:
                logger.info("🔄 Fetching IPC engagement ID for firewall listing...")
                ipc_engagement_id = await self.get_ipc_engagement_id()
                if not ipc_engagement_id:
                    return {
                        "success": False,
                        "error": "Could not retrieve IPC engagement ID"
                    }
                logger.info(f"✅ Got IPC engagement ID: {ipc_engagement_id}")
            
            # Get endpoints if not provided
            if not endpoint_ids:
                logger.info("🔄 Fetching all endpoints...")
                endpoints_result = await self.list_endpoints()
                if not endpoints_result.get("success"):
                    return {
                        "success": False,
                        "error": "Could not fetch endpoints"
                    }
                available_endpoints = endpoints_result.get("data", {}).get("endpoints", [])
                endpoint_ids = [ep.get("id") for ep in available_endpoints if ep.get("id")]
                logger.info(f"✅ Found {len(endpoint_ids)} endpoints")
            
            # Query each endpoint
            all_firewalls = []
            endpoint_results = {}
            
            url = "https://ipcloud.tatacommunications.com/networkservice/firewallconfig/details"
            headers = await self._get_auth_headers()
            
            # Get HTTP client
            client = await self._get_http_client()
            
            for endpoint_id in endpoint_ids:
                try:
                    payload = {
                        "engagementId": ipc_engagement_id,
                        "endpointId": endpoint_id,
                        "variant": variant
                    }
                    
                    logger.info(f"📡 Querying firewalls for endpoint {endpoint_id}...")
                    
                    response = await client.post(
                        url,
                        json=payload,
                        headers=headers,
                        timeout=30.0
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        firewalls = data.get("data", [])
                        
                        logger.info(f"✅ Endpoint {endpoint_id}: Found {len(firewalls)} firewalls")
                        
                        # Add endpoint info to each firewall
                        for fw in firewalls:
                            fw["_queried_endpoint_id"] = endpoint_id
                        
                        all_firewalls.extend(firewalls)
                        endpoint_results[endpoint_id] = {
                            "success": True,
                            "count": len(firewalls)
                        }
                    else:
                        logger.warning(f"⚠️ Endpoint {endpoint_id}: API returned {response.status_code}")
                        endpoint_results[endpoint_id] = {
                            "success": False,
                            "error": f"Status {response.status_code}"
                        }
                
                except Exception as e:
                    logger.error(f"❌ Endpoint {endpoint_id}: {e}")
                    endpoint_results[endpoint_id] = {
                        "success": False,
                        "error": str(e)
                    }
            
            duration = time.time() - start_time
            
            logger.info(f"✅ Total firewalls found: {len(all_firewalls)} across {len(endpoint_ids)} endpoints")
            
            return {
                "success": True,
                "data": all_firewalls,
                "total": len(all_firewalls),
                "endpoints_queried": endpoint_ids,
                "endpoint_results": endpoint_results,
                "ipc_engagement_id": ipc_engagement_id,
                "variant": variant,
                "duration_seconds": duration,
                "message": f"Found {len(all_firewalls)} firewalls"
            }
        
        except Exception as e:
            logger.error(f"❌ Exception in list_firewalls: {e}")
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
        dry_run: bool = False,
        user_id: str = None,
        auth_email: str = None,
        auth_password: str = None
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
        
        # Get user credentials if user_id is provided and credentials not explicitly passed
        if user_id and not auth_email and not auth_password:
            credentials = self._get_user_credentials(user_id)
            if credentials:
                auth_email = credentials.get("email")
                auth_password = credentials.get("password")
                logger.info(f"✅ Retrieved API credentials for user: {user_id}")
            else:
                logger.warning(f"⚠️ No API credentials found for user: {user_id}, using default/env")
        
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
                    # Check if this is a read-only user trying to perform actions
                    is_read_only = user_roles == ["viewer"]
                    is_write_operation = operation in ["create", "update", "delete", "provision"]
                    
                    if is_read_only and is_write_operation:
                        # Provide enrollment information for unauthorized users
                        return {
                            "success": False,
                            "error": "Unauthorized",
                            "message": "You don't have permission to perform this action.",
                            "enrollment_info": {
                                "title": "Want to perform actions?",
                                "description": "Enroll for full access to create and manage cloud resources.",
                                "enrollment_url": "https://cloud.tatacommunications.com/enroll",
                                "contact": "support@tatacommunications.com",
                                "sso_login": "Sign in with Tata Communications for full access"
                            },
                            "required_permissions": operation_config.get("permissions", []),
                            "your_permissions": user_roles,
                            "timestamp": start_time.isoformat()
                        }
                    else:
                        # Generic permission denied
                        return {
                            "success": False,
                            "error": f"Permission denied for {operation} on {resource_type}",
                            "required_permissions": operation_config.get("permissions", []),
                            "your_permissions": user_roles,
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
            result = await self._make_api_call(
                endpoint_config,
                params,
                user_id=user_id,
                auth_email=auth_email,
                auth_password=auth_password
            )
            
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
        params: Dict[str, Any],
        user_id: str = None,
        auth_email: str = None,
        auth_password: str = None
    ) -> Dict[str, Any]:
        """
        Make the actual API call with automatic token refresh.
        Supports both regular JSON and SSE streaming responses.
        
        Args:
            endpoint_config: Endpoint configuration (method, url, etc.)
            params: Request parameters
            user_id: User identifier (for per-user token caching)
            auth_email: Email for authentication (uses default from env if not provided)
            auth_password: Password for authentication (uses default from env if not provided)
            
        Returns:
            API response dict
        """
        # Ensure we have a valid token before making the call
        token_valid = await self._ensure_valid_token(user_id, auth_email, auth_password)
        if not token_valid:
            logger.error(f"❌ Cannot make API call: No valid auth token (user: {user_id or 'default'})")
            return {
                "success": False,
                "error": "Authentication failed: Unable to obtain valid token"
            }
        
        method = endpoint_config.get("method", "GET").upper()
        url = endpoint_config.get("url", "")
        is_streaming = endpoint_config.get("streaming", False)
        
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
        
        # Get auth headers with user-specific token
        auth_headers = await self._get_auth_headers(user_id, auth_email, auth_password)
        headers.update(auth_headers)
        
        try:
            logger.info(f"🌐 API Call: {method} {url}")
            if body_params:
                logger.debug(f"📦 Request body: {json.dumps(body_params, indent=2)}")
            
            # Handle SSE streaming response
            if is_streaming:
                return await self._handle_streaming_response(client, method, url, headers, body_params)
            
            # Make request based on method (regular JSON response)
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
    
    async def _handle_streaming_response(
        self,
        client: httpx.AsyncClient,
        method: str,
        url: str,
        headers: Dict[str, str],
        body_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle Server-Sent Events (SSE) streaming response for cluster listing.
        
        Args:
            client: HTTP client
            method: HTTP method
            url: API URL
            headers: Request headers
            body_params: Request body parameters
            
        Returns:
            Dict with parsed cluster data from all endpoints
        """
        logger.info("🌊 Handling SSE streaming response")
        
        all_clusters = []
        endpoint_data = {}
        errors = {}
        
        try:
            async with client.stream(method, url, headers=headers, json=body_params) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    line = line.strip()
                    
                    # Skip empty lines and keepalive pings
                    if not line or line == ":ping":
                        continue
                    
                    # Parse SSE format
                    if line.startswith("event:"):
                        event_type = line.split(":", 1)[1].strip()
                        
                        if event_type == "complete":
                            logger.info("✅ Stream completed")
                            break
                            
                        elif event_type == "endpoint":
                            # Next line should be id, then data
                            continue
                    
                    elif line.startswith("id:"):
                        endpoint_id = line.split(":", 1)[1].strip()
                        
                    elif line.startswith("data:"):
                        data_str = line.split(":", 1)[1].strip()
                        
                        # Skip "done" messages
                        if data_str == "done":
                            continue
                        
                        try:
                            # Fix malformed JSON: datetime values without quotes
                            # Pattern: "createdTime":2025-04-08 10:08:38.0
                            # Should be: "createdTime":"2025-04-08 10:08:38.0"
                            import re
                            data_str = re.sub(
                                r'("createdTime":)(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d)',
                                r'\1"\2"',
                                data_str
                            )
                            
                            endpoint_info = json.loads(data_str)
                            endpoint_id = endpoint_info.get("endpointId")
                            endpoint_name = endpoint_info.get("endpointName")
                            clusters = endpoint_info.get("clusters", [])
                            error = endpoint_info.get("error")
                            
                            if error:
                                logger.warning(f"⚠️ Endpoint {endpoint_name} ({endpoint_id}): {error}")
                                errors[endpoint_name] = error
                            else:
                                logger.info(f"📊 Endpoint {endpoint_name} ({endpoint_id}): {len(clusters)} clusters")
                            
                            # Store endpoint data
                            endpoint_data[endpoint_id] = {
                                "endpoint_id": endpoint_id,
                                "endpoint_name": endpoint_name,
                                "clusters": clusters,
                                "error": error
                            }
                            
                            # Collect all clusters
                            all_clusters.extend(clusters)
                            
                        except json.JSONDecodeError as e:
                            logger.error(f"❌ Failed to parse SSE data: {e}")
                            logger.debug(f"Problematic data: {data_str[:200]}...")
                            continue
            
            logger.info(f"✅ SSE streaming complete: {len(all_clusters)} total clusters from {len(endpoint_data)} endpoints")
            
            return {
                "success": True,
                "data": all_clusters,
                "endpoint_data": endpoint_data,
                "errors": errors if errors else None,
                "status_code": 200
            }
            
        except Exception as e:
            logger.error(f"❌ SSE streaming error: {str(e)}")
            return {
                "success": False,
                "error": f"Streaming failed: {str(e)}"
            }
    
    async def check_cluster_name_available(self, cluster_name: str) -> Dict[str, Any]:
        """
        Check if cluster name is available using resource_schema.json configuration.
        
        API Response format:
        - Name TAKEN: {"status": "success", "data": {"clusterName": "xyz", "clusterId": 123}, ...}
        - Name AVAILABLE: {"status": "success", "data": {}, ...}
        
        Args:
            cluster_name: Name to check
            
        Returns:
            Dict with availability status
        """
        logger.info(f"🔍 Checking cluster name availability: {cluster_name}")
        
        # Use the schema-based execute_operation method
        result = await self.execute_operation(
            resource_type="k8s_cluster",
            operation="check_cluster_name",
            params={"clusterName": cluster_name},
            user_roles=None
        )
        
        if result.get("success"):
            # The API response is wrapped: result["data"] contains the full API response
            # API response format: {"status": "success", "data": {...}, "message": "OK", "responseCode": 200}
            api_response = result.get("data", {})
            
            # Get the nested "data" field from the API response
            # - If data is empty {} → name is AVAILABLE
            # - If data has clusterName/clusterId → name is TAKEN
            inner_data = api_response.get("data", {})
            
            # Name is available if inner_data is empty
            is_available = not inner_data or inner_data == {}
            
            if is_available:
                logger.info(f"✅ Cluster name '{cluster_name}' is AVAILABLE")
            else:
                existing_cluster = inner_data.get("clusterName", cluster_name)
                existing_id = inner_data.get("clusterId", "unknown")
                logger.info(f"❌ Cluster name '{cluster_name}' is TAKEN (existing: {existing_cluster}, ID: {existing_id})")
            
            return {
                "success": True,
                "available": is_available,
                "message": f"Cluster name '{cluster_name}' is {'available' if is_available else 'already taken'}",
                "existing_cluster": inner_data if not is_available else None
            }
        
        # API call failed
        logger.error(f"❌ Failed to check cluster name availability: {result.get('error')}")
        return {
            "success": False,
            "available": False,
            "error": result.get("error", "Failed to verify cluster name availability"),
            "message": "Unable to check cluster name availability at this time"
        }
    
    async def get_iks_images_and_datacenters(self, engagement_id: int) -> Dict[str, Any]:
        """
        Get IKS images with datacenter information using resource_schema.json configuration.
        
        Flow:
        1. Convert PAAS engagement_id to IPC engagement_id
        2. Call getTemplatesByEngagement with ipc_engagement_id
        3. Parse response to extract datacenters and images
        
        Returns dict with:
        - datacenters: List of unique data centers (from endpointName)
        - images: All images (contains ImageName with K8s version)
        """
        logger.info(f"🖼️ Fetching IKS images for engagement {engagement_id}")
        
        # Step 1: Get IPC engagement ID
        ipc_engagement_id = await self.get_ipc_engagement_id(engagement_id)
        if not ipc_engagement_id:
            logger.error("❌ Failed to get IPC engagement ID")
            return {
                "success": False,
                "error": "Failed to get IPC engagement ID",
                "datacenters": [],
                "images": []
            }
        
        # Step 2: Call get_iks_images with IPC engagement ID
        logger.info(f"📡 Calling getTemplatesByEngagement with IPC engagement ID: {ipc_engagement_id}")
        result = await self.execute_operation(
            resource_type="k8s_cluster",
            operation="get_iks_images",
            params={"ipc_engagement_id": ipc_engagement_id},
            user_roles=None
        )
        
        if result.get("success") and result.get("data"):
            api_data = result["data"]
            
            # Parse API response
            # Response: {"status": "success", "data": {"all-images": [...]}}
            if api_data.get("status") == "success" and api_data.get("data"):
                all_images = []
                data = api_data["data"]
                
                # Collect images from all categories (e.g., "all-images", "vks-enabledImages")
                for category, images in data.items():
                    if isinstance(images, list):
                        all_images.extend(images)
                
                # Get unique datacenters from endpointName and endpointId
                datacenters = {}
                for img in all_images:
                    dc_id = img.get("endpointId")
                    if dc_id and dc_id not in datacenters:
                        datacenters[dc_id] = {
                            "id": dc_id,
                            "name": img.get("endpointName", f"DC-{dc_id}"),
                            "endpoint": img.get("endpoint", "")
                        }
                
                logger.info(f"✅ Found {len(datacenters)} datacenters, {len(all_images)} images from API")
                return {
                    "success": True,
                    "datacenters": list(datacenters.values()),
                    "images": all_images
                }
        
        # API failed
        logger.error("❌ Failed to fetch IKS images from API")
        return {
            "success": False,
            "error": "Failed to fetch IKS images from API",
            "datacenters": [],
            "images": []
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
        Get CNI drivers for datacenter + k8s version using resource_schema.json configuration.
        
        Args:
            endpoint_id: Datacenter ID
            k8s_version: Kubernetes version
            
        Returns:
            Dict with CNI drivers list
        """
        logger.info(f"🌐 Fetching CNI drivers for endpoint {endpoint_id}, k8s {k8s_version}")
        
        # Use the schema-based execute_operation method
        result = await self.execute_operation(
            resource_type="k8s_cluster",
            operation="get_network_list",
            params={
                "endpointId": endpoint_id,
                "k8sVersion": k8s_version
            },
            user_roles=None
        )
        
        if result.get("success") and result.get("data"):
            api_data = result["data"]
            
            # Parse response - API returns {"status": "success", "data": {"data": [...]}}
            if api_data.get("status") == "success" and api_data.get("data"):
                drivers = api_data["data"].get("data", [])
                logger.info(f"✅ Found {len(drivers)} CNI drivers from API")
                return {
                    "success": True,
                    "drivers": drivers
                }
        
        # API failed
        logger.error("❌ Failed to fetch CNI drivers from API")
        return {
            "success": False,
            "error": "Failed to fetch CNI driver data from API",
            "drivers": []
        }
    
    async def get_environments_and_business_units(self, engagement_id: int) -> Dict[str, Any]:
        """
        Get environments and business units for engagement using resource_schema.json configuration.
        
        Args:
            engagement_id: Engagement ID
            
        Returns:
            Dict with business units and environments
        """
        logger.info(f"🏢 Fetching environments for engagement {engagement_id}")
        
        # Use the schema-based execute_operation method
        result = await self.execute_operation(
            resource_type="k8s_cluster",
            operation="get_environments",
            params={"engagement_id": engagement_id},
            user_roles=None
        )
        
        if result.get("success") and result.get("data"):
            api_data = result["data"]
            
            # Parse response
            if api_data.get("status") == "success" and api_data.get("data"):
                environments = api_data["data"].get("environments", [])
                
                # Extract unique business units
                business_units = {}
                for env in environments:
                    bu_id = env.get("departmentId")
                    if bu_id and bu_id not in business_units:
                        business_units[bu_id] = {
                            "id": bu_id,
                            "name": env.get("department", f"BU-{bu_id}")
                        }
                
                logger.info(f"✅ Found {len(business_units)} business units, {len(environments)} environments from API")
                return {
                    "success": True,
                    "business_units": list(business_units.values()),
                    "environments": environments
                }
        
        # API failed
        logger.error("❌ Failed to fetch environments from API")
        return {
            "success": False,
            "error": "Failed to fetch environment data from API",
            "business_units": [],
            "environments": []
        }
    
    async def get_zones_list(self, engagement_id: int) -> Dict[str, Any]:
        """
        Get zones for engagement using resource_schema.json configuration.
        
        Args:
            engagement_id: Engagement ID
            
        Returns:
            Dict with zones list
        """
        logger.info(f"🗺️ Fetching zones for engagement {engagement_id}")
        
        # Use the schema-based execute_operation method
        result = await self.execute_operation(
            resource_type="k8s_cluster",
            operation="get_zones",
            params={"engagement_id": engagement_id},
            user_roles=None
        )
        
        if result.get("success") and result.get("data"):
            api_data = result["data"]
            
            # Parse response
            if isinstance(api_data, dict) and api_data.get("data"):
                zones = api_data["data"]
                logger.info(f"✅ Found {len(zones)} zones from API")
                return {
                    "success": True,
                    "zones": zones
                }
        
        # API failed
        logger.error("❌ Failed to fetch zones from API")
        return {
            "success": False,
            "error": "Failed to fetch zone data from API",
            "zones": []
        }
    
    async def get_os_images(self, zone_id: int, circuit_id: str, k8s_version: str) -> Dict[str, Any]:
        """
        Get OS images for zone, filtered by k8s version using resource_schema.json configuration.
        
        Args:
            zone_id: Zone ID
            circuit_id: Circuit ID
            k8s_version: Kubernetes version to filter by
            
        Returns:
            Dict with OS options
        """
        logger.info(f"💿 Fetching OS images for zone {zone_id}, k8s {k8s_version}")
        
        # Use the schema-based execute_operation method
        result = await self.execute_operation(
            resource_type="k8s_cluster",
            operation="get_os_images",
            params={
                "zoneId": zone_id,
                "circuitId": circuit_id
            },
            user_roles=None
        )
        
        if result.get("success") and result.get("data"):
            api_data = result["data"]
            
            # Parse response
            if api_data.get("status") == "success" and api_data.get("data"):
                images = api_data["data"].get("image", {}).get("options", [])
                
                # Filter by k8s version
                filtered = [img for img in images if k8s_version in img.get("label", "")]
                
                # Group by osMake + osVersion
                grouped = {}
                for img in filtered:
                    key = f"{img.get('osMake', '')} {img.get('osVersion', '')}"
                    if key not in grouped:
                        grouped[key] = {
                            "display_name": key,
                            "os_id": img.get("id"),
                            "os_make": img.get("osMake"),
                            "os_model": img.get("osModel"),
                            "os_version": img.get("osVersion"),
                            "hypervisor": img.get("hypervisor"),
                            "images": []
                        }
                    grouped[key]["images"].append(img)
                
                logger.info(f"✅ Found {len(grouped)} OS options from API")
                return {
                    "success": True,
                    "os_options": list(grouped.values())
                }
        
        # API failed
        logger.error("❌ Failed to fetch OS images from API")
        return {
            "success": False,
            "error": "Failed to fetch OS image data from API",
            "os_options": []
        }
    
    async def get_flavors(self, zone_id: int, circuit_id: str, os_model: str, node_type: str = None) -> Dict[str, Any]:
        """
        Get compute flavors for zone, filtered by OS and optionally node type using resource_schema.json configuration.
        
        Args:
            zone_id: Zone ID
            circuit_id: Circuit ID
            os_model: OS model (e.g., "ubuntu")
            node_type: Node type to filter (generalPurpose, computeOptimized, memoryOptimized)
            
        Returns:
            Dict with flavor options
        """
        logger.info(f"💻 Fetching flavors for zone {zone_id}, OS {os_model}, node type {node_type}")
        
        # Use the schema-based execute_operation method
        result = await self.execute_operation(
            resource_type="k8s_cluster",
            operation="get_flavors",
            params={
                "zoneId": zone_id,
                "circuitId": circuit_id
            },
            user_roles=None
        )
        
        if result.get("success") and result.get("data"):
            api_data = result["data"]
            
            # Parse response
            if api_data.get("status") == "success" and api_data.get("data"):
                flavors = api_data["data"].get("flavor", [])
                
                # Filter by OS model and application type
                filtered = [f for f in flavors if f.get("osModel") == os_model and f.get("applicationType") == "Container"]
                
                # Further filter by node type if provided
                if node_type:
                    filtered = [f for f in filtered if f.get("flavorCategory") == node_type]
                
                # Extract unique node types for the first query
                node_types = list(set([f.get("flavorCategory") for f in filtered if f.get("flavorCategory")]))
                
                # Format flavors
                formatted_flavors = []
                for flavor in filtered:
                    formatted_flavors.append({
                        "id": flavor.get("artifactId"),
                        "name": f"{flavor.get('vCpu')} vCPU / {flavor.get('vRam', 0) // 1024} GB RAM / {flavor.get('vDisk')} GB Storage",
                        "flavor_name": flavor.get("FlavorName"),
                        "sku_code": flavor.get("skuCode"),
                        "vcpu": flavor.get("vCpu"),
                        "vram_gb": flavor.get("vRam", 0) // 1024,
                        "disk_gb": flavor.get("vDisk"),
                        "node_type": flavor.get("flavorCategory")
                    })
                
                logger.info(f"✅ Found {len(node_types)} node types, {len(formatted_flavors)} flavors from API")
                return {
                    "success": True,
                    "node_types": node_types,
                    "flavors": formatted_flavors
                }
        
        # API failed
        logger.error("❌ Failed to fetch flavors from API")
        return {
            "success": False,
            "error": "Failed to fetch flavor data from API",
            "node_types": [],
            "flavors": []
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
    
    async def get_business_units_list(self, ipc_engagement_id: int = None, user_id: str = None, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get business units (departments) listing for engagement.
        Uses per-user session storage to avoid repeated API calls.
        
        Args:
            ipc_engagement_id: IPC Engagement ID (will be fetched if not provided)
            user_id: User ID (email) for session lookup
            force_refresh: Force fetch even if cached
            
        Returns:
            Dict with business units data including zones, environments, and VMs count
        """
        try:
            if not user_id:
                user_id = self._get_user_id_from_email()
            
            # Check user session cache first
            if not force_refresh:
                session = await self._get_user_session(user_id)
                if session and "business_units" in session:
                    bu_data = session["business_units"]
                    cached_depts = bu_data.get("department", []) if bu_data else []
                    logger.info(f"📋 Using cached business units from session ({len(cached_depts)} BUs)")
                    return {
                        "success": True,
                        "data": bu_data,
                        "engagement": bu_data.get("engagement"),
                        "departments": cached_depts,
                        "ipc_engagement_id": session.get("ipc_engagement_id")
                    }
            else:
                logger.info(f"🔄 Force refresh requested, bypassing cache")
            
            # Get IPC engagement ID if not provided
            if not ipc_engagement_id:
                ipc_engagement_id = await self.get_ipc_engagement_id(user_id=user_id)
                if not ipc_engagement_id:
                    return {
                        "success": False,
                        "error": "Failed to get IPC engagement ID",
                        "data": None
                    }
                
                logger.info(f"✅ Got IPC engagement ID: {ipc_engagement_id}")
            
            url = f"https://ipcloud.tatacommunications.com/portalservice/securityservice/departments/{ipc_engagement_id}"
            logger.info(f"🏢 Fetching business units from: {url} (IPC engagement ID: {ipc_engagement_id})")
            
            # Get auth token
            token = await self._get_or_refresh_token()
            if not token:
                return {
                    "success": False,
                    "error": "Failed to get authentication token",
                    "data": None
                }
            
            # Make API call
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            
            async with httpx.AsyncClient(timeout=self.api_timeout) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                
                data = response.json()
                logger.info(f"🏢 Raw API response status: {data.get('status')}")
                
                if data.get("status") == "success":
                    departments = data.get("data", {}).get("department", [])
                    engagement_info = data.get("data", {}).get("engagement", {})
                    bu_data = data.get("data")
                    
                    logger.info(f"🏢 API returned {len(departments)} departments for engagement: {engagement_info}")
                    
                    # Log first few departments for debugging
                    if departments:
                        for dept in departments[:3]:
                            logger.info(f"🏢 Sample dept: {dept.get('name')} (ID: {dept.get('id')}, endpoint: {dept.get('endpoint')})")
                    
                    # Update user session with business units data
                    await self._update_user_session(
                        user_id=user_id,
                        business_units=bu_data
                    )
                    
                    logger.info(f"✅ Cached {len(departments)} business units for engagement '{engagement_info.get('name')}'")
                    
                    return {
                        "success": True,
                        "data": bu_data,
                        "engagement": engagement_info,
                        "departments": departments,
                        "ipc_engagement_id": ipc_engagement_id
                    }
                else:
                    logger.error(f"❌ API returned error: {data}")
                    return {
                        "success": False,
                        "error": data.get("message", "Unknown error"),
                        "data": None
                    }
                    
        except httpx.HTTPStatusError as e:
            logger.error(f"❌ HTTP error fetching business units: {e}")
            return {
                "success": False,
                "error": f"HTTP {e.response.status_code}: {str(e)}",
                "data": None
            }
        except Exception as e:
            logger.error(f"❌ Error fetching business units: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "data": None
            }
    
    async def get_environments_list(self, ipc_engagement_id: int = None, user_id: str = None, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get environments listing per engagement.
        Uses per-user session storage to avoid repeated API calls.
        
        Args:
            ipc_engagement_id: IPC Engagement ID (will be fetched if not provided)
            user_id: User ID (email) for session lookup
            force_refresh: Force fetch even if cached
            
        Returns:
            Dict with environments data
        """
        try:
            if not user_id:
                user_id = self._get_user_id_from_email()
            
            # Check user session cache first
            if not force_refresh:
                session = await self._get_user_session(user_id)
                if session and "environments_list" in session:
                    environments = session["environments_list"]
                    logger.debug(f"✅ Using cached environments from session ({len(environments)} environments)")
                    return {
                        "success": True,
                        "data": environments,
                        "environments": environments,
                        "ipc_engagement_id": session.get("ipc_engagement_id")
                    }
            
            # Get IPC engagement ID if not provided
            if not ipc_engagement_id:
                ipc_engagement_id = await self.get_ipc_engagement_id(user_id=user_id)
                if not ipc_engagement_id:
                    return {
                        "success": False,
                        "error": "Failed to get IPC engagement ID",
                        "data": None
                    }
                
                logger.info(f"✅ Got IPC engagement ID: {ipc_engagement_id}")
            
            url = f"https://ipcloud.tatacommunications.com/portalservice/securityservice/environmentsperengagement/{ipc_engagement_id}"
            logger.info(f"🌍 Fetching environments from: {url}")
            
            # Get auth token
            token = await self._get_or_refresh_token()
            if not token:
                return {
                    "success": False,
                    "error": "Failed to get authentication token",
                    "data": None
                }
            
            # Make API call
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            
            async with httpx.AsyncClient(timeout=self.api_timeout) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                
                data = response.json()
                
                if data.get("status") == "success":
                    environments = data.get("data", [])
                    
                    # Update user session with environments data
                    # Note: Using key "environments_list" to avoid conflict with "environments" in business_units
                    await self._update_user_session(
                        user_id=user_id,
                        environments_list=environments
                    )
                    
                    logger.info(f"✅ Cached {len(environments)} environments")
                    
                    return {
                        "success": True,
                        "data": environments,
                        "environments": environments,
                        "ipc_engagement_id": ipc_engagement_id
                    }
                else:
                    logger.error(f"❌ API returned error: {data}")
                    return {
                        "success": False,
                        "error": data.get("message", "Unknown error"),
                        "data": None
                    }
                    
        except httpx.HTTPStatusError as e:
            logger.error(f"❌ HTTP error fetching environments: {e}")
            return {
                "success": False,
                "error": f"HTTP {e.response.status_code}: {str(e)}",
                "data": None
            }
        except Exception as e:
            logger.error(f"❌ Error fetching environments: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "data": None
            }
    
    async def get_zones_list(self, ipc_engagement_id: int = None, user_id: str = None, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get zones (network segments/VLANs) listing for engagement.
        Uses per-user session storage to avoid repeated API calls.
        
        Args:
            ipc_engagement_id: IPC Engagement ID (will be fetched if not provided)
            user_id: User ID (email) for session lookup
            force_refresh: Force fetch even if cached
            
        Returns:
            Dict with zones data including CIDR, hypervisors, status, and associated environments
        """
        try:
            if not user_id:
                user_id = self._get_user_id_from_email()
            
            # Check user session cache first
            if not force_refresh:
                session = await self._get_user_session(user_id)
                if session and "zones_list" in session:
                    zones = session["zones_list"]
                    logger.debug(f"✅ Using cached zones from session ({len(zones)} zones)")
                    return {
                        "success": True,
                        "data": zones,
                        "zones": zones,
                        "ipc_engagement_id": session.get("ipc_engagement_id")
                    }
            
            # Get IPC engagement ID if not provided
            if not ipc_engagement_id:
                ipc_engagement_id = await self.get_ipc_engagement_id(user_id=user_id)
                if not ipc_engagement_id:
                    return {
                        "success": False,
                        "error": "Failed to get IPC engagement ID",
                        "data": None
                    }
                
                logger.info(f"✅ Got IPC engagement ID: {ipc_engagement_id}")
            
            url = f"https://ipcloud.tatacommunications.com/portalservice/api/v1/{ipc_engagement_id}/zonelist"
            logger.info(f"🌐 Fetching zones from: {url}")
            
            # Get auth token
            token = await self._get_or_refresh_token()
            if not token:
                return {
                    "success": False,
                    "error": "Failed to get authentication token",
                    "data": None
                }
            
            # Make API call
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            
            async with httpx.AsyncClient(timeout=self.api_timeout) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                
                data = response.json()
                
                if data.get("status") == "success":
                    zones = data.get("data", [])
                    
                    # Update user session with zones data
                    await self._update_user_session(
                        user_id=user_id,
                        zones_list=zones
                    )
                    
                    logger.info(f"✅ Cached {len(zones)} zones")
                    
                    return {
                        "success": True,
                        "data": zones,
                        "zones": zones,
                        "ipc_engagement_id": ipc_engagement_id
                    }
                else:
                    logger.error(f"❌ API returned error: {data}")
                    return {
                        "success": False,
                        "error": data.get("message", "Unknown error"),
                        "data": None
                    }
                    
        except httpx.HTTPStatusError as e:
            logger.error(f"❌ HTTP error fetching zones: {e}")
            return {
                "success": False,
                "error": f"HTTP {e.response.status_code}: {str(e)}",
                "data": None
            }
        except Exception as e:
            logger.error(f"❌ Error fetching zones: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "data": None
            }
    
    def __repr__(self) -> str:
        resource_count = len(self.resource_schema.get("resources", {}))
        return f"<APIExecutorService(resources={resource_count})>"


# Global instance
api_executor_service = APIExecutorService()

