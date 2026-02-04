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
# Auth is now handled by UI - tokens passed via Authorization header from Keycloak

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
            os.path.dirname(__file__), "../config/resource_schema.json")
        self.resource_schema: Dict[str, Any] = {}
        self.http_client: Optional[httpx.AsyncClient] = None
        self._load_resource_schema()
        # API configuration
        self.api_timeout = float(os.getenv("API_EXECUTOR_TIMEOUT", "30"))
        self.max_retries = int(os.getenv("API_EXECUTOR_MAX_RETRIES", "3"))
        # Token management - per user (since different users may have different credentials)
        self.user_tokens: Dict[str, Dict[str, Any]] = {} 
        self.token_lock = asyncio.Lock()  
        # Auth API configuration
        self.auth_url = os.getenv(
            "API_AUTH_URL",
            "https://ipcloud.tatacommunications.com/portalservice/api/v1/getAuthToken")
        # Keep env vars as fallback for backward compatibility
        self.default_auth_email = os.getenv("API_AUTH_EMAIL", "")
        self.default_auth_password = os.getenv("API_AUTH_PASSWORD", "")
        # Development mode: skip authentication if explicitly disabled
        self.auth_enabled = os.getenv("API_AUTH_ENABLED", "true").lower() == "true"
        # Per-user session storage for engagement IDs and other frequently accessed data
        self.user_sessions: Dict[str, Dict[str, Any]] = {}
        self.session_cache_duration = timedelta(hours=24)  
        self.session_lock = asyncio.Lock()  # Prevent concurrent session updates
        # Legacy engagement caching - kept for backward compatibility
        self.cached_engagement: Optional[Dict[str, Any]] = None
        self.engagement_cache_time: Optional[datetime] = None
        self.engagement_cache_duration = timedelta(hours=1)  
        logger.info("✅ APIExecutorService initialized")

    def _load_resource_schema(self) -> None:
        """Load resource schema from JSON file."""
        try:
            with open(self.resource_schema_path,"r", encoding="utf-8") as f:
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
    
    async def _fetch_auth_token(self,auth_email: str = None,auth_password: str = None) -> Optional[str]:
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
            if self.auth_enabled:
                logger.error("❌ API credentials not configured (email or password missing)")
            else:
                logger.debug("ℹ️ API credentials not configured (auth disabled)")
            return None
        try:
            client = await self._get_http_client()
            
            auth_payload = {
                "email": email,
                "password": password}
            logger.info(f"🔑 Fetching auth token from {self.auth_url}")
            response = await client.post(
                self.auth_url,
                json=auth_payload,
                headers={"Content-Type": "application/json"},
                timeout=10.0)
            response.raise_for_status()
            data = response.json()
            # Check for error response
            status_code = data.get("statusCode")
            if status_code == 500:
                error_msg = data.get("accessToken", "Unknown error")
                logger.error(f"❌ Auth API returned error: {error_msg}")
                return None
            # Extract token from response
            token = (
                data.get("accessToken") or 
                data.get("access_token") or 
                data.get("token") or 
                data.get("authToken"))
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
        
    async def _ensure_valid_token(self,user_id: str = None,auth_email: str = None,auth_password: str = None) -> bool:
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
        # Skip authentication if disabled (development mode)
        if not self.auth_enabled:
            logger.warning("⚠️ Authentication is disabled (development mode)")
            return True
            
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
                    "expires_at": datetime.utcnow() + timedelta(minutes=8)  }
                logger.info(f"✅ Auth token refreshed successfully for user: {cache_key}")
                return True
            else:
                logger.error(f"❌ Failed to refresh auth token for user: {cache_key}")
                return False
    
    async def _get_or_refresh_token(self,user_id: str = None,auth_email: str = None,auth_password: str = None) -> Optional[str]:
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
                follow_redirects=True)
        return self.http_client
    
    def _get_user_credentials(self, user_id: str = None) -> Optional[Dict[str, str]]:
        """
        Get API credentials for fallback (service-to-service calls).
        Auth is now primarily handled via tokens passed from UI.
        Args:
            user_id: User identifier (for logging)
        Returns:
            Dict with 'email' and 'password' keys from env vars, or None if not configured
        """
        # Fall back to env vars for service-to-service calls (e.g., background jobs)
        if self.default_auth_email and self.default_auth_password:
            logger.debug(f"Using default credentials from env (user: {user_id or 'service'})")
            return {
                "email": self.default_auth_email,
                "password": self.default_auth_password}
        logger.warning(f"❌ No credentials available (user: {user_id})")
        return None
    
    async def _get_auth_headers(self,user_id: str = None,auth_email: str = None,auth_password: str = None,auth_token: str = None) -> Dict[str, str]:
        """
        Get authentication headers with current token for a user.
        Args:
            user_id: User identifier (for per-user token caching)
            auth_email: Email for authentication (uses default from env if not provided)
            auth_password: Password for authentication (uses default from env if not provided)
            auth_token: Bearer token from UI (Keycloak) - takes precedence over email/password
        Returns:
            Dictionary of headers including authorization
        """
        headers = {
            "Content-Type": "application/json"}
        
        # Skip token logic if auth is disabled
        if not self.auth_enabled:
            logger.debug("⚠️ Auth disabled - returning basic headers")
            return headers
        
        # If auth_token provided (from Keycloak/UI), use it directly
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
            logger.debug("✅ Using Bearer token from UI (Keycloak)")
            return headers
            
        # Otherwise, fetch token using email/password (legacy flow)
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

    async def get_engagements_list(self, auth_token: str = None, user_id: str = None) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch list of all available engagements for the user.
        
        Args:
            auth_token: Bearer token from UI (Keycloak)
            user_id: User ID for session lookup
            
        Returns:
            List of engagement dicts with id, name, etc. or None if failed
        """
        logger.info("🔍 Fetching engagements list from API...")
        result = await self.execute_operation(
            resource_type="engagement",
            operation="get",
            params={},
            user_roles=None,
            auth_token=auth_token)
        
        if result.get("success") and result.get("data"):
            data = result["data"]
            if isinstance(data, dict) and "data" in data:
                engagements = data["data"]
                logger.info(f"✅ Found {len(engagements)} engagements")
                return engagements
        
        logger.error("❌ Failed to fetch engagements list")
        return None
    
    async def set_engagement_id(self, user_id: str, engagement_id: int, engagement_data: Dict = None) -> bool:
        """
        Set/update the selected engagement ID for a user session.
        Called when ENG user selects an engagement.
        
        Args:
            user_id: User ID for session
            engagement_id: Selected engagement ID
            engagement_data: Full engagement data dict (optional)
            
        Returns:
            True if successful
        """
        await self._update_user_session(
            user_id=user_id,
            paas_engagement_id=engagement_id,
            engagement_data=engagement_data
        )
        logger.info(f"✅ Set engagement ID {engagement_id} for user {user_id}")
        return True
    
    async def get_engagement_id(self, force_refresh: bool = False, user_id: str = None, auth_token: str = None, user_type: str = None) -> Optional[int]:
        """
        Get engagement ID for the authenticated user.
        Uses per-user session storage to avoid repeated API calls.
        
        For CUS (Customer): Auto-selects the single engagement returned by API
        For ENG (Engineer): Checks session for selected engagement, returns None if not selected
        
        Args:
            force_refresh: Force fetch even if cached
            user_id: User ID (email) for session lookup
            auth_token: Bearer token from UI (Keycloak)
            user_type: User type from header (ENG or CUS)
        Returns:
            PAAS Engagement ID or None if selection needed (for ENG) or failed
        """
        if not user_id:
            user_id = self._get_user_id_from_email()
        
        # Check user session cache first (works for both ENG and CUS once selected)
        if not force_refresh:
            session = await self._get_user_session(user_id)
            if session and "paas_engagement_id" in session:
                paas_id = session["paas_engagement_id"]
                logger.debug(f"✅ Using cached PAAS engagement ID from session: {paas_id}")
                return paas_id
        
        # Fetch engagements from API
        logger.info(f"🔍 Fetching engagement details from API (user_type: {user_type})...")
        engagements = await self.get_engagements_list(auth_token=auth_token, user_id=user_id)
        
        if not engagements:
            logger.error("❌ Failed to fetch engagement ID - no engagements returned")
            return None
        
        # CUS (Customer): Auto-select the single engagement
        if user_type == "CUS" or len(engagements) == 1:
            engagement = engagements[0]
            paas_engagement_id = engagement.get("id")
            
            await self._update_user_session(
                user_id=user_id,
                paas_engagement_id=paas_engagement_id,
                engagement_data=engagement
            )
            # Also update legacy cache for backward compatibility
            self.cached_engagement = engagement
            self.engagement_cache_time = datetime.utcnow()
            
            logger.info(f"✅ Auto-selected engagement: {engagement.get('engagementName')} (ID: {paas_engagement_id})")
            return paas_engagement_id
        
        # ENG (Engineer) with multiple engagements: Need to prompt for selection
        # Cache the engagements so we don't need to fetch them again
        self._pending_engagements = engagements
        self._pending_engagements_user = user_id
        logger.info(f"🔄 ENG user has {len(engagements)} engagements - selection required")
        return None  # Caller should handle engagement selection flow
    
    def get_cached_pending_engagements(self, user_id: str = None) -> Optional[List[Dict[str, Any]]]:
        """Get cached pending engagements if available (to avoid double API call)."""
        if hasattr(self, '_pending_engagements') and self._pending_engagements:
            # Check if it's for the same user
            if not user_id or getattr(self, '_pending_engagements_user', None) == user_id:
                engagements = self._pending_engagements
                # Clear cache after retrieval
                self._pending_engagements = None
                self._pending_engagements_user = None
                return engagements
        return None
    
    async def get_engagement_selection_prompt(self, auth_token: str = None, user_id: str = None) -> Optional[Dict[str, Any]]:
        """
        Get engagement selection prompt for ENG users with multiple engagements.
        
        Args:
            auth_token: Bearer token from UI
            user_id: User ID
            
        Returns:
            Dict with engagements list and formatted prompt, or None if failed/not needed
        """
        engagements = await self.get_engagements_list(auth_token=auth_token, user_id=user_id)
        
        if not engagements:
            return None
        
        if len(engagements) == 1:
            # Only one engagement, no selection needed
            return None
        
        # Format engagements for display
        options = []
        for i, eng in enumerate(engagements, 1):
            options.append({
                "index": i,
                "id": eng.get("id"),
                "name": eng.get("engagementName") or eng.get("name"),
                "description": eng.get("description", "")
            })
        
        prompt = "Please select an engagement to work with:\n\n"
        for opt in options:
            prompt += f"**{opt['index']}. {opt['name']}** (ID: {opt['id']})\n"
        prompt += "\nYou can say the number, name, or ID. You can also change this later by saying 'switch engagement'."
        
        return {
            "engagements": engagements,
            "options": options,
            "prompt": prompt,
            "needs_selection": True
        }
    
    async def get_ipc_engagement_id(self, engagement_id: int = None, user_id: str = None, force_refresh: bool = False, auth_token: str = None) -> Optional[int]:
        """
        Convert PAAS engagement ID to IPC engagement ID.
        Uses per-user session storage to avoid repeated API calls.
        Args:
            engagement_id: PAAS Engagement ID (fetches if not provided)
            user_id: User ID (email) for session lookup
            force_refresh: Force fetch even if cached
            auth_token: Bearer token from UI (Keycloak)
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
            engagement_id = await self.get_engagement_id(user_id=user_id, auth_token=auth_token)
            if not engagement_id:
                return None
        logger.info(f"🔄 Converting PAAS engagement {engagement_id} to IPC engagement ID...")
        result = await self.execute_operation(
            resource_type="k8s_cluster",
            operation="get_ipc_engagement",
            params={"engagement_id": engagement_id},
            user_roles=None,
            auth_token=auth_token
        )
        if result.get("success") and result.get("data"):
            data = result["data"]
            if data.get("status") == "success" and data.get("data"):
                ipc_engid = data["data"].get("ipc_engid")
                if ipc_engid:
                    # Update user session with IPC engagement ID
                    await self._update_user_session(
                        user_id=user_id,
                        ipc_engagement_id=ipc_engid,
                        paas_engagement_id=engagement_id)
                    
                    logger.info(f"✅ Cached IPC engagement ID: {ipc_engid}")
                    return ipc_engid
        logger.error("❌ Failed to get IPC engagement ID")
        return None
    async def get_endpoints(self, engagement_id: int = None, user_id: str = None, force_refresh: bool = False, auth_token: str = None, user_type: str = None) -> Optional[List[Dict[str, Any]]]:
        """
        Get available endpoints (data centers) for an engagement.
        Uses per-user session storage to avoid repeated API calls.
        Args:
            engagement_id: Engagement ID (fetches if not provided)
            user_id: User ID (email) for session lookup
            force_refresh: Force fetch even if cached
            auth_token: Bearer token from UI (Keycloak)
            user_type: User type (ENG or CUS) for engagement selection logic
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
            # First check if there's a cached engagement in session
            session = await self._get_user_session(user_id)
            if session and "paas_engagement_id" in session:
                engagement_id = session["paas_engagement_id"]
                logger.info(f"✅ Using engagement ID from session cache: {engagement_id}")
            else:
                engagement_id = await self.get_engagement_id(user_id=user_id, auth_token=auth_token, user_type=user_type)
                if not engagement_id:
                    # For ENG users with multiple engagements, return None to trigger selection flow
                    logger.info(f"🔄 No engagement ID - selection may be required (user_type: {user_type})")
                    return None
        else:
            logger.info(f"✅ Using provided engagement ID: {engagement_id}")
        
        logger.info(f"🔍 Fetching endpoints for engagement {engagement_id}...")
        result = await self.execute_operation(
            resource_type="endpoint",
            operation="list",
            params={"engagement_id": engagement_id},
            user_roles=None,
            auth_token=auth_token)
        if result.get("success") and result.get("data"):
            data = result["data"]
            # API returns {"status": "success", "data": [{...}]}
            if isinstance(data, dict) and "data" in data:
                endpoints = data["data"]
                # Update user session with endpoints
                await self._update_user_session(
                    user_id=user_id,
                    endpoints=endpoints)
                logger.info(f"✅ Cached {len(endpoints)} endpoints")
                return endpoints
        logger.error("❌ Failed to fetch endpoints")
        return None

    async def list_endpoints(self, engagement_id: int = None, auth_token: str = None, user_id: str = None) -> Dict[str, Any]:
        """
        List available endpoints (datacenters/locations) for the user's engagement.
        This is the main workflow method for endpoint listing.
        Args:
            engagement_id: Engagement ID (fetches if not provided)
            auth_token: Bearer token from UI for API authentication
            user_id: User ID for session lookup
        Returns:
            Dict with endpoint list or error
        """
        try:
            # Step 1: Get engagement ID
            if engagement_id is None:
                engagement_id = await self.get_engagement_id(auth_token=auth_token, user_id=user_id)
                if not engagement_id:
                    return {
                        "success": False,
                        "error": "Failed to fetch engagement ID",
                        "step": "get_engagement"}
            # Step 2: Fetch endpoints
            logger.info(f"📍 Fetching endpoints for engagement {engagement_id}")
            endpoints = await self.get_endpoints(
                engagement_id=engagement_id,
                user_id=user_id,
                auth_token=auth_token
            )
            if endpoints:
                # Format the response nicely
                formatted_endpoints = []
                for ep in endpoints:
                    formatted_endpoints.append({
                        "id": ep.get("endpointId"),
                        "name": ep.get("endpointDisplayName"),
                        "type": ep.get("endpointType", ""),
                        "region": ep.get("region", ""),
                        "status": ep.get("status", "active")})
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
                    "step": "get_endpoints"}
        except Exception as e:
            logger.error(f"❌ Failed to list endpoints: {str(e)}")
            return {
                "success": False,
                "error": str(e)}
    
    async def list_clusters(self, endpoint_ids: List[int] = None, engagement_id: int = None, auth_token: str = None, user_id: str = None) -> Dict[str, Any]:
        """
        List Kubernetes clusters for given endpoints.
        This is the main workflow method that handles the multi-step process.
        Args:
            endpoint_ids: List of endpoint IDs to query (fetches all if not provided)
            engagement_id: Engagement ID (fetches if not provided)
            auth_token: Bearer token from UI for API authentication
            user_id: User ID for session lookup
        Returns:
            Dict with cluster list or error
        """
        try:
            # Step 1: Get engagement ID
            if engagement_id is None:
                engagement_id = await self.get_engagement_id(auth_token=auth_token, user_id=user_id)
                if not engagement_id:
                    return {
                        "success": False,
                        "error": "Failed to fetch engagement ID",
                        "step": "get_engagement"
                    }
            
            # Step 2: Get endpoints if not provided
            if endpoint_ids is None:
                endpoints = await self.get_endpoints(engagement_id, auth_token=auth_token, user_id=user_id)
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
    
    async def list_managed_services(self, service_type: str, endpoint_ids: List[int] = None, ipc_engagement_id: int = None, auth_token: str = None, user_id: str = None) -> Dict[str, Any]:
        """
        List managed services (Kafka, GitLab, etc.) for given endpoints.
        This is the main workflow method that handles the multi-step process.
        Args:
            service_type: Service type to list (e.g., "IKSKafka", "IKSGitlab")
            endpoint_ids: List of endpoint IDs to query (fetches all if not provided)
            ipc_engagement_id: IPC Engagement ID (fetches and converts if not provided)
            auth_token: Bearer token from UI for API authentication
            user_id: User ID for session lookup
        Returns:
            Dict with service list or error
        """
        try:
            # Step 1: Get PAAS engagement ID (if needed for endpoints)
            paas_engagement_id = None
            if endpoint_ids is None:
                paas_engagement_id = await self.get_engagement_id(auth_token=auth_token, user_id=user_id)
                if not paas_engagement_id:
                    return {
                        "success": False,
                        "error": "Failed to fetch PAAS engagement ID",
                        "step": "get_engagement"}
            # Step 2: Get IPC engagement ID (required for managed services API)
            if ipc_engagement_id is None:
                if paas_engagement_id is None:
                    paas_engagement_id = await self.get_engagement_id(auth_token=auth_token, user_id=user_id)
                
                ipc_engagement_id = await self.get_ipc_engagement_id(paas_engagement_id, auth_token=auth_token, user_id=user_id)
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
                    paas_engagement_id = await self.get_engagement_id(auth_token=auth_token, user_id=user_id)
                endpoints = await self.get_endpoints(paas_engagement_id, auth_token=auth_token, user_id=user_id)
                if not endpoints:
                    return {
                        "success": False,
                        "error": "Failed to fetch endpoints",
                        "step": "get_endpoints"}
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
                "serviceType": service_type}
            client = await self._get_http_client()
            headers = await self._get_auth_headers(user_id=user_id, auth_token=auth_token)
            logger.info(f"🌐 POST {url}")
            logger.debug(f"📦 Payload: {json.dumps(payload, indent=2)}")
            response = await client.post(
                url,
                json=payload,
                headers=headers,
                timeout=self.api_timeout)
            if response.status_code == 200:
                response_data = response.json()
                # API returns nested structure
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
                    "raw_response": response_data  }
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
                    "status_code": response.status_code}
        except Exception as e:
            logger.error(f"❌ Failed to list {service_type} services: {str(e)}")
            return {
                "success": False,
                "error": str(e)}
    
    async def list_kafka(self,endpoint_ids: List[int] = None,ipc_engagement_id: int = None) -> Dict[str, Any]:
        """
        List Kafka managed services.
        Convenience wrapper around list_managed_services.
        Args:
            endpoint_ids: List of endpoint IDs to query
            ipc_engagement_id: IPC Engagement ID 
        Returns:
            Dict with Kafka service list or error
        """
        return await self.list_managed_services(service_type="IKSKafka",endpoint_ids=endpoint_ids,ipc_engagement_id=ipc_engagement_id)
    
    async def list_gitlab(self,endpoint_ids: List[int] = None,ipc_engagement_id: int = None) -> Dict[str, Any]:
        """
        List GitLab managed services.
        Convenience wrapper around list_managed_services.
        Args:
            endpoint_ids: List of endpoint IDs to query
            ipc_engagement_id: IPC Engagement ID
        Returns:
            Dict with GitLab service list or error
        """
        return await self.list_managed_services(service_type="IKSGitlab",endpoint_ids=endpoint_ids,ipc_engagement_id=ipc_engagement_id)
    
    async def list_container_registry(self,endpoint_ids: List[int] = None,ipc_engagement_id: int = None) -> Dict[str, Any]:
        """
        List Container Registry managed services.
        Convenience wrapper around list_managed_services.
        Args:
            endpoint_ids: List of endpoint IDs to query
            ipc_engagement_id: IPC Engagement ID   
        Returns:
            Dict with Container Registry service list or error
        """
        return await self.list_managed_services(service_type="IKSContainerRegistry",endpoint_ids=endpoint_ids,ipc_engagement_id=ipc_engagement_id)
    
    async def list_jenkins(self,endpoint_ids: List[int] = None,ipc_engagement_id: int = None) -> Dict[str, Any]:
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
    
    async def list_postgres(self,endpoint_ids: List[int] = None,ipc_engagement_id: int = None) -> Dict[str, Any]:
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
            ipc_engagement_id=ipc_engagement_id)
    
    async def list_documentdb(self,endpoint_ids: List[int] = None,ipc_engagement_id: int = None) -> Dict[str, Any]:
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
            ipc_engagement_id=ipc_engagement_id)
    
    async def list_vms(self, ipc_engagement_id: int = None, endpoint_filter: str = None, zone_filter: str = None, department_filter: str = None, auth_token: str = None, user_id: str = None) -> Dict[str, Any]:
        """
        List all virtual machines for the engagement.
        Args:
            ipc_engagement_id: IPC Engagement ID (will be fetched if not provided)
            endpoint_filter: Filter by endpoint
            zone_filter: Filter by zone
            department_filter: Filter by department
            auth_token: Bearer token from UI for API authentication
            user_id: User ID for session lookup
        Returns:
            Dict with VM list or error
        """
        import time
        start_time = time.time()
        try:
            # Get IPC engagement ID if not provided
            if not ipc_engagement_id:
                logger.info("🔄 Fetching IPC engagement ID for VM listing...")
                ipc_engagement_id = await self.get_ipc_engagement_id(user_id=user_id, auth_token=auth_token)
                if not ipc_engagement_id:
                    return {
                        "success": False,
                        "error": "Could not retrieve IPC engagement ID"
                    }
                logger.info(f"✅ Got IPC engagement ID: {ipc_engagement_id}")
            # Build URL
            url = f"https://ipcloud.tatacommunications.com/portalservice/instances/vmlist/{ipc_engagement_id}"
            logger.info(f"📡 Calling VM list API: GET {url}")
            # Get auth headers - prefer passed token, fallback to refresh
            token = auth_token or await self._get_or_refresh_token(user_id)
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            } if token else await self._get_auth_headers(user_id=user_id, auth_token=auth_token)
            # Get HTTP client
            client = await self._get_http_client()
            # Make GET request
            response = await client.get(url,headers=headers,timeout=30.0)
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
                        if endpoint_filter.lower() in vm.get("virtualMachine", {}).get("endpoint", {}).get("endpointName", "").lower()]
                    logger.info(f"🔍 Filtered by endpoint '{endpoint_filter}': {len(filtered_vms)} VMs")
                if zone_filter:
                    filtered_vms = [
                        vm for vm in filtered_vms
                        if zone_filter.lower() in vm.get("virtualMachine", {}).get("zone", {}).get("zoneName", "").lower()]
                    logger.info(f"🔍 Filtered by zone '{zone_filter}': {len(filtered_vms)} VMs")
                if department_filter:
                    filtered_vms = [
                        vm for vm in filtered_vms
                        if department_filter.lower() in vm.get("virtualMachine", {}).get("department", {}).get("departmentName", "").lower()]
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
    
    async def list_firewalls(self, endpoint_ids: List[int] = None, ipc_engagement_id: int = None, variant: str = "", auth_token: str = None, user_id: str = None) -> Dict[str, Any]:
        """
        List firewalls across multiple endpoints.
        Args:
            endpoint_ids: List of endpoint IDs to query
            ipc_engagement_id: IPC Engagement ID (will be fetched if not provided)
            variant: Firewall variant filter (default: empty string)
            auth_token: Bearer token from UI for API authentication
            user_id: User ID for session lookup
        Returns:
            Dict with firewall list or error
        """
        import time
        start_time = time.time()
        try:
            # Get IPC engagement ID if not provided
            if not ipc_engagement_id:
                logger.info("🔄 Fetching IPC engagement ID for firewall listing...")
                ipc_engagement_id = await self.get_ipc_engagement_id(user_id=user_id, auth_token=auth_token)
                if not ipc_engagement_id:
                    return {
                        "success": False,
                        "error": "Could not retrieve IPC engagement ID"}
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
            headers = await self._get_auth_headers(user_id=user_id, auth_token=auth_token)
            
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
        
    async def list_load_balancers(
    self,
    ipc_engagement_id: int = None,
    user_id: str = None,
    force_refresh: bool = False,
    auth_token: str = None
) -> Dict[str, Any]:

        import time
        start_time = time.time()
    
        try:
            if not user_id:
                user_id = self._get_user_id_from_email()
        
            logger.info(f"⚖️ Listing load balancers for user: {user_id}")
        
        # Check user session cache first
            if not force_refresh:
                session = await self._get_user_session(user_id)
                if session and "load_balancers" in session:
                    cached_lbs = session["load_balancers"]
                    logger.info(f"✅ Using cached load balancers ({len(cached_lbs)} LBs)")
                    return {
                    "success": True,
                    "data": cached_lbs,
                    "total": len(cached_lbs),
                    "ipc_engagement_id": session.get("ipc_engagement_id"),
                    "cached": True,
                    "duration_seconds": time.time() - start_time
                }
        
        # Get IPC engagement ID if not provided
            if not ipc_engagement_id:
                ipc_engagement_id = await self.get_ipc_engagement_id(user_id=user_id)
                if not ipc_engagement_id:
                    return {
                    "success": False,
                    "error": "Could not retrieve IPC engagement ID"
                }
                logger.info(f"✅ Got IPC engagement ID: {ipc_engagement_id}")
        
        # CRITICAL: Build URL with IPC engagement ID in path
            url = f"https://ipcloud.tatacommunications.com/networkservice/loadbalancer/list/loadbalancers/{ipc_engagement_id}"
        
            logger.info(f"📡 API Call: GET {url}")
            logger.info(f"🔑 Using IPC engagement ID: {ipc_engagement_id}")
        
        # Get auth headers
            headers = await self._get_auth_headers(user_id=user_id, auth_token=auth_token)
        
        # Get HTTP client
            client = await self._get_http_client()
        
        # Make GET request (no payload needed)
            response = await client.get(
            url,
            headers=headers,
            timeout=30.0
        )
        
            logger.info(f"📥 Load balancer API response: {response.status_code}")
        
            if response.status_code == 200:
                data = response.json()
            
            # Handle the nested response structure
            # API returns: {"status": "success" | "failed", "data": [...] | null}
                api_status = data.get("status", "").lower()
            
                if api_status == "success":
                # Success case - load balancers exist
                    load_balancers = data.get("data", [])
                
                # Handle nested data structure if present
                    if isinstance(load_balancers, dict) and "data" in load_balancers:
                        load_balancers = load_balancers["data"]
                
                # Ensure it's a list
                    if not isinstance(load_balancers, list):
                        load_balancers = [load_balancers] if load_balancers else []
                
                # Cache in user session
                    await self._update_user_session(
                    user_id=user_id,
                    load_balancers=load_balancers
                    )
                
                    logger.info(f"✅ Found {len(load_balancers)} load balancer(s)")
                
                    duration = time.time() - start_time
                
                    return {
                    "success": True,
                    "data": load_balancers,
                    "total": len(load_balancers),
                    "ipc_engagement_id": ipc_engagement_id,
                    "cached": False,
                    "duration_seconds": duration,
                    "message": f"Found {len(load_balancers)} load balancer(s)"
                }
            
                elif api_status == "failed":
                # "failed" status with HTTP 200 means NO load balancers
                # This is NOT an error - it's a valid empty result
                    logger.info(f"ℹ️ No load balancers found (status=failed)")
                
                # Cache empty result
                    await self._update_user_session(
                    user_id=user_id,
                    load_balancers=[]
                    )
                
                    duration = time.time() - start_time
                
                    return {
                    "success": True,
                    "data": [],
                    "total": 0,
                    "ipc_engagement_id": ipc_engagement_id,
                    "cached": False,
                    "duration_seconds": duration,
                    "message": "No load balancers found"
                    }
            
                else:
                # Unknown status - treat as error
                    error_msg = data.get("message", f"Unknown API status: {api_status}")
                    logger.error(f"❌ Load balancer API returned unexpected status: {api_status}")
                    return {
                    "success": False,
                    "error": error_msg,
                    "status_code": response.status_code,
                    "ipc_engagement_id": ipc_engagement_id
                }
        
            elif response.status_code == 404:
            # 404 means no load balancers (NOT an error)
                logger.info(f"ℹ️ No load balancers found (404)")
            
            # Cache empty result
                await self._update_user_session(
                user_id=user_id,
                load_balancers=[]
                )
            
                return {
                "success": True,
                "data": [],
                "total": 0,
                "ipc_engagement_id": ipc_engagement_id,
                "cached": False,
                "message": "No load balancers found"
            }
        
            else:
            # Handle other error codes
                error_msg = f"API returned status {response.status_code}"
                try:
                    error_data = response.json()
                    error_msg = error_data.get("message", error_msg)
                except:
                    error_msg = response.text if response.text else error_msg
            
                logger.error(f"❌ Load balancer API failed: {error_msg}")
                return {
                "success": False,
                "error": error_msg,
                "status_code": response.status_code,
                "ipc_engagement_id": ipc_engagement_id
            }
    
        except Exception as e:
            logger.error(f"❌ Exception in list_load_balancers: {e}", exc_info=True)
            return {
            "success": False,
            "error": str(e),
            "ipc_engagement_id": ipc_engagement_id}

    async def get_load_balancer_details(self, lbci: str, user_id: str = None, auth_token: str = None) -> Dict[str, Any]:
        import time
        start_time = time.time()
        try:
            if not user_id:
                user_id = self._get_user_id_from_email()
            if not lbci:
                return {
                "success": False,
                "error": "LBCI (Load Balancer Circuit ID) is required"}
            logger.info(f"🔍 Fetching details for load balancer: {lbci}")
        # Build URL
            url = f"https://ipcloud.tatacommunications.com/networkservice/loadbalancer/getDetails/{lbci}"
            logger.info(f"📡 API Call: GET {url}")
        # Get auth headers
            headers = await self._get_auth_headers(user_id=user_id, auth_token=auth_token)
        # Get HTTP client
            client = await self._get_http_client()
        # Make GET request
            response = await client.get(url,headers=headers,timeout=30.0)
            logger.info(f"📥 LB Details API response: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
            # API response format: {"status": "success", "data": {...}}
                if data.get("status") == "success":
                    details = data.get("data", {})
                    logger.info(f"✅ Retrieved details for load balancer: {lbci}")
                    duration = time.time() - start_time
                    return {
                    "success": True,
                    "data": details,
                    "lbci": lbci,
                    "duration_seconds": duration,
                    "message": f"Retrieved details for {lbci}"
                    }
                else:
                    error_msg = data.get("message", "Failed to get load balancer details")
                    logger.error(f"❌ API returned error: {error_msg}")
                    return {
                    "success": False,
                    "error": error_msg,
                    "lbci": lbci}
            elif response.status_code == 404:
                logger.warning(f"⚠️ Load balancer not found: {lbci}")
                return {
                "success": False,
                "error": f"Load balancer not found: {lbci}",
                "status_code": 404,
                "lbci": lbci}
            else:
                error_msg = f"API returned status {response.status_code}"
                try:
                    error_data = response.json()
                    error_msg = error_data.get("message", error_msg)
                except:
                    error_msg = response.text if response.text else error_msg
                logger.error(f"❌ LB Details API failed: {error_msg}")
                return {
                "success": False,
                "error": error_msg,
                "status_code": response.status_code,
                "lbci": lbci}
        except Exception as e:
            logger.error(f"❌ Exception in get_load_balancer_details: {e}", exc_info=True)
            return {
            "success": False,
            "error": str(e),
            "lbci": lbci}
        
    async def get_load_balancer_virtual_services(self, lbci: str, user_id: str = None, auth_token: str = None) -> Dict[str, Any]:
        import time
        start_time = time.time()
        try:
            if not user_id:
                user_id = self._get_user_id_from_email()
        
            if not lbci:
                return {
                "success": False,
                "error": "LBCI (Load Balancer Circuit ID) is required"}
            logger.info(f"🌐 Fetching virtual services for load balancer: {lbci}")
        # Build URL
            url = f"https://ipcloud.tatacommunications.com/networkservice/loadbalancer/list/virtualservices/{lbci}"
            logger.info(f"📡 API Call: GET {url}")
        # Get auth headers
            headers = await self._get_auth_headers(user_id=user_id, auth_token=auth_token)
        # Get HTTP client
            client = await self._get_http_client()
        # Make GET request
            response = await client.get(url,headers=headers,timeout=30.0)
            logger.info(f"📥 Virtual Services API response: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
            # API response format: {"status": "success", "data": [...]}
                if data.get("status") == "success":
                    virtual_services = data.get("data", [])
                # Ensure it's a list
                    if not isinstance(virtual_services, list):
                        virtual_services = [virtual_services] if virtual_services else []
                    logger.info(f"✅ Retrieved {len(virtual_services)} virtual service(s) for {lbci}")
                    duration = time.time() - start_time
                    return {
                    "success": True,
                    "data": virtual_services,
                    "total": len(virtual_services),
                    "lbci": lbci,
                    "duration_seconds": duration,
                    "message": f"Found {len(virtual_services)} virtual service(s)"}
                else:
                    error_msg = data.get("message", "Failed to get virtual services")
                    logger.error(f"❌ API returned error: {error_msg}")
                    return {
                    "success": False,
                    "error": error_msg,
                    "lbci": lbci
                    }
            elif response.status_code == 404:
                logger.info(f"ℹ️ No virtual services found for {lbci}")
                return {
                "success": True,
                "data": [],
                "total": 0,
                "lbci": lbci,
                "message": "No virtual services found"}
            else:
                error_msg = f"API returned status {response.status_code}"
                try:
                    error_data = response.json()
                    error_msg = error_data.get("message", error_msg)
                except:
                    error_msg = response.text if response.text else error_msg
                logger.error(f"❌ Virtual Services API failed: {error_msg}")
                return {
                "success": False,
                "error": error_msg,
                "status_code": response.status_code,
                "lbci": lbci}
        except Exception as e:
            logger.error(f"❌ Exception in get_load_balancer_virtual_services: {e}", exc_info=True)
            return {
            "success": False,
            "error": str(e),
            "lbci": lbci}
        
    def get_resource_config(self, resource_type: str) -> Optional[Dict[str, Any]]:
        """
        Get configuration for a resource type.
        Args:
            resource_type: Type of resource (k8s_cluster, firewall, etc.)     
        Returns:
            Resource configuration dict or None if not found
        """
        return self.resource_schema.get("resources", {}).get(resource_type)

    def get_operation_config(self,resource_type: str,operation: str) -> Optional[Dict[str, Any]]:
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
                "permissions": permissions}
        return None
    
    def validate_parameters(self,resource_type: str,operation: str,params: Dict[str, Any]) -> Dict[str, Any]:
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
                "errors": [f"Unknown resource type or operation: {resource_type}.{operation}"]}
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
            "errors": errors}
    
    def _validate_param_value(self,param_name: str,param_value: Any,rules: Dict[str, Any]) -> List[str]:
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
    
    def check_permissions(self,resource_type: str,operation: str,user_roles: List[str]) -> bool:
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
                f"required permissions {required_permissions} for {resource_type}.{operation}")
        return has_permission
    
    async def execute_operation(self,
        resource_type: str,operation: str,
        params: Dict[str, Any],
        user_roles: List[str] = None,
        dry_run: bool = False,
        user_id: str = None,
        auth_email: str = None,
        auth_password: str = None,
        auth_token: str = None) -> Dict[str, Any]:
        """
        Execute a CRUD operation on a resource.
        Args:
            resource_type: Type of resource
            operation: Operation name (create, read, update, delete, list)
            params: Operation parameters
            user_roles: User's roles for permission checking
            dry_run: If True, validate but don't execute
            user_id: User identifier
            auth_email: Email for authentication (legacy)
            auth_password: Password for authentication (legacy)
            auth_token: Bearer token from UI (Keycloak) - takes precedence
        Returns:
            Dict with execution result
        """
        start_time = datetime.utcnow()
        if user_id and not auth_email and not auth_password:
            credentials = self._get_user_credentials(user_id)
            if credentials:
                auth_email = credentials.get("email")
                auth_password = credentials.get("password")
                logger.info(f"✅ Retrieved API credentials for user: {user_id}")
            else:
                logger.warning(f"⚠️ No API credentials found for user: {user_id}, using default/env")
        
        try:
            operation_config = self.get_operation_config(resource_type, operation)
            if not operation_config:
                return {
                    "success": False,
                    "error": f"Unknown resource type or operation: {resource_type}.{operation}",
                    "timestamp": start_time.isoformat()}
            
            if user_roles is not None:
                if not self.check_permissions(resource_type, operation, user_roles):
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
                            "timestamp": start_time.isoformat()}
                    else:
                        return {
                            "success": False,
                            "error": f"Permission denied for {operation} on {resource_type}",
                            "required_permissions": operation_config.get("permissions", []),
                            "your_permissions": user_roles,
                            "timestamp": start_time.isoformat()}
            # Validate parameters
            validation_result = self.validate_parameters(resource_type, operation, params)
            if not validation_result["valid"]:
                return {
                    "success": False,
                    "error": "Parameter validation failed",
                    "validation_errors": validation_result["errors"],
                    "timestamp": start_time.isoformat()}
            # Dry run - return success without executing
            if dry_run:
                return {
                    "success": True,
                    "dry_run": True,
                    "message": f"Validation successful for {operation} on {resource_type}",
                    "params": params,
                    "timestamp": start_time.isoformat()}
            # Execute API call
            endpoint_config = operation_config["endpoint"]
            result = await self._make_api_call(
                endpoint_config,
                params,
                user_id=user_id,
                auth_email=auth_email,
                auth_password=auth_password,
                auth_token=auth_token)
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            return {
                "success": result.get("success", True),
                "data": result.get("data"),
                "error": result.get("error"),
                "resource_type": resource_type,
                "operation": operation,
                "duration_seconds": duration,
                "timestamp": end_time.isoformat()}
        except Exception as e:
            logger.error(f"❌ Failed to execute {operation} on {resource_type}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "resource_type": resource_type,
                "operation": operation,
                "timestamp": datetime.utcnow().isoformat()}
    
    async def _make_api_call(
        self,
        endpoint_config: Dict[str, Any],
        params: Dict[str, Any],
        user_id: str = None,
        auth_email: str = None,
        auth_password: str = None,
        auth_token: str = None
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
            auth_token: Bearer token from UI (Keycloak) - takes precedence over email/password
        Returns:
            API response dict
        """
        # DEBUG: Log auth_token status
        logger.debug(f"🔐 _make_api_call auth_token: {'PROVIDED' if auth_token else 'NONE'}")
        
        # Skip token validation if auth_token provided from UI or auth is disabled
        if not auth_token:
            token_valid = await self._ensure_valid_token(user_id, auth_email, auth_password)
            if not token_valid:
                logger.error(f"❌ Cannot make API call: No valid auth token (user: {user_id or 'default'})")
                return {
                    "success": False,
                    "error": "Authentication failed: Unable to obtain valid token"}
        method = endpoint_config.get("method", "GET").upper()
        url = endpoint_config.get("url", "")
        is_streaming = endpoint_config.get("streaming", False)
        path_params = {}
        body_params = {}
        for param_name, param_value in params.items():
            # If parameter is in URL template, it's a path parameter
            if f"{{{param_name}}}" in url:
                path_params[param_name] = param_value
                url = url.replace(f"{{{param_name}}}", str(param_value))
            else:
                body_params[param_name] = param_value
        # Get HTTP client
        client = await self._get_http_client()
        headers = endpoint_config.get("headers", {})
        headers.setdefault("Content-Type", "application/json")
        auth_headers = await self._get_auth_headers(user_id, auth_email, auth_password, auth_token)
        headers.update(auth_headers)
        
        try:
            logger.info(f"🌐 API Call: {method} {url}")
            if body_params:
                logger.debug(f"📦 Request body: {json.dumps(body_params, indent=2)}")
            # Handle SSE streaming response
            if is_streaming:
                return await self._handle_streaming_response(client, method, url, headers, body_params)
            if method == "GET":
                # IMPORTANT: httpx replaces URL query params when params= is passed (even if empty!)
                if body_params:
                    # If URL already has query params, we need to merge them
                    if "?" in url:
                        from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
                        parsed = urlparse(url)
                        existing_params = parse_qs(parsed.query)
                        # Flatten single-value lists from parse_qs
                        existing_params = {k: v[0] if len(v) == 1 else v for k, v in existing_params.items()}
                        # Merge with body_params (body_params takes precedence)
                        merged_params = {**existing_params, **body_params}
                        # Rebuild URL without query string
                        url_without_query = urlunparse(parsed._replace(query=''))
                        response = await client.get(url_without_query, headers=headers, params=merged_params)
                    else:
                        response = await client.get(url, headers=headers, params=body_params)
                else:
                    # No additional params - use URL as-is (preserves existing query params)
                    response = await client.get(url, headers=headers)
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
                    "error": f"Unsupported HTTP method: {method}"}
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
                "status_code": e.response.status_code}
        except httpx.RequestError as e:
            logger.error(f"❌ API Request error: {method} {url} - {str(e)}")
            return {
                "success": False,
                "error": f"Request failed: {str(e)}"}
    
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
                    if not line or line == ":ping":
                        continue
                    if line.startswith("event:"):
                        event_type = line.split(":", 1)[1].strip()
                        if event_type == "complete":
                            logger.info("✅ Stream completed")
                            break
                        elif event_type == "endpoint":
                            continue
                    
                    elif line.startswith("id:"):
                        endpoint_id = line.split(":", 1)[1].strip()
                        
                    elif line.startswith("data:"):
                        data_str = line.split(":", 1)[1].strip()
                        if data_str == "done":
                            continue
                        
                        try:
                            # Pattern: "createdTime":2025-04-08 10:08:38.0
                            # Should be: "createdTime":"2025-04-08 10:08:38.0"
                            import re
                            data_str = re.sub(
                                r'("createdTime":)(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d)',
                                r'\1"\2"',
                                data_str)
                            parsed_data = json.loads(data_str)
                            
                            # Handle both dict format (with endpoint metadata) and list format (direct cluster array)
                            if isinstance(parsed_data, list):
                                # API returned a direct list of clusters (no endpoint wrapper)
                                logger.debug(f"📊 Received direct cluster list: {len(parsed_data)} clusters")
                                all_clusters.extend(parsed_data)
                                # Store in endpoint_data with a generic key
                                if "direct" not in endpoint_data:
                                    endpoint_data["direct"] = {
                                        "endpoint_id": "direct",
                                        "endpoint_name": "Direct Response",
                                        "clusters": [],
                                        "error": None
                                    }
                                endpoint_data["direct"]["clusters"].extend(parsed_data)
                                continue
                            
                            # Standard dict format with endpoint metadata
                            endpoint_info = parsed_data
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
                "status_code": 200}
            
        except Exception as e:
            logger.error(f"❌ SSE streaming error: {str(e)}")
            return {
                "success": False,
                "error": f"Streaming failed: {str(e)}"}
    
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
            user_roles=None)
        if result.get("success"):
            # The API response is wrapped: result["data"] contains the full API response
            api_response = result.get("data", {})
            # Get the nested "data" field from the API response
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
                "existing_cluster": inner_data if not is_available else None}
    
        logger.error(f"❌ Failed to check cluster name availability: {result.get('error')}")
        return {
            "success": False,
            "available": False,
            "error": result.get("error", "Failed to verify cluster name availability"),
            "message": "Unable to check cluster name availability at this time"}
    
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
                "images": []}
        # Step 2: Call get_iks_images with IPC engagement ID
        logger.info(f"📡 Calling getTemplatesByEngagement with IPC engagement ID: {ipc_engagement_id}")
        result = await self.execute_operation(
            resource_type="k8s_cluster",
            operation="get_iks_images",
            params={"ipc_engagement_id": ipc_engagement_id},
            user_roles=None)
        if result.get("success") and result.get("data"):
            api_data = result["data"]
            # Parse API response
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
                            "endpoint": img.get("endpoint", "")}
                logger.info(f"✅ Found {len(datacenters)} datacenters, {len(all_images)} images from API")
                return {
                    "success": True,
                    "datacenters": list(datacenters.values()),
                    "images": all_images}
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
                    "drivers": drivers}

        logger.error("❌ Failed to fetch CNI drivers from API")
        return {
            "success": False,
            "error": "Failed to fetch CNI driver data from API",
            "drivers": []}
    
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
            user_roles=None)
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
                            "name": env.get("department", f"BU-{bu_id}")}
                logger.info(f"✅ Found {len(business_units)} business units, {len(environments)} environments from API")
                return {
                    "success": True,
                    "business_units": list(business_units.values()),
                    "environments": environments}
        logger.error("❌ Failed to fetch environments from API")
        return {
            "success": False,
            "error": "Failed to fetch environment data from API",
            "business_units": [],
            "environments": []}
    
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
            user_roles=None)
        if result.get("success") and result.get("data"):
            api_data = result["data"]
            # Parse response
            if isinstance(api_data, dict) and api_data.get("data"):
                zones = api_data["data"]
                logger.info(f"✅ Found {len(zones)} zones from API")
                return {
                    "success": True,
                    "zones": zones}

        logger.error("❌ Failed to fetch zones from API")
        return {
            "success": False,
            "error": "Failed to fetch zone data from API",
            "zones": []}

    async def get_os_images(self, zone_id: int, circuit_id: str, k8s_version: str, auth_token: str = None, user_id: str = None) -> Dict[str, Any]:
        """
        Get OS images (templates) for zone, filtered by k8s version.
        Uses the templates API: /uat-portalservice/configservice/templates/{zoneId}?type=Container
        Filters by k8s version in the label/ImageName field.
        Returns distinct options grouped by osMake + osVersion.
        Args:
            zone_id: Zone ID
            circuit_id: Circuit ID (not used by new API, kept for compatibility)
            k8s_version: Kubernetes version to filter by (e.g., "v1.30.9")
            auth_token: Bearer token from UI for API authentication
            user_id: User ID for session lookup
        Returns:
            Dict with OS options
        """
        logger.info(f"💿 Fetching OS templates for zone {zone_id}, k8s {k8s_version}")
        try:
            # Get auth token - prefer passed token
            token = auth_token or await self._get_or_refresh_token(user_id)
            if not token:
                return {"success": False, "error": "Failed to get auth token", "os_options": []}
            # Call the templates API directly
            url = f"https://ipcloud.tatacommunications.com/uat-portalservice/configservice/templates/{zone_id}?type=Container"
            logger.info(f"🌐 API Call: GET {url}")
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"}

            async with httpx.AsyncClient(timeout=self.api_timeout) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                api_data = response.json()
            # Parse response
            if api_data.get("status") == "success" and api_data.get("data"):
                images = api_data["data"].get("image", {}).get("options", [])
                logger.info(f"📦 Found {len(images)} total OS images from API")
                # Filter by k8s version - check both label and ImageName fields
                version_patterns = [k8s_version]
                if k8s_version.startswith("v"):
                    version_patterns.append(k8s_version[1:])  
                filtered = []
                for img in images:
                    label = img.get("label", "") or ""
                    image_name = img.get("ImageName", "") or ""
                    # Check if any version pattern matches
                    if any(pattern in label or pattern in image_name for pattern in version_patterns):
                        filtered.append(img)
                logger.info(f"🎯 Filtered to {len(filtered)} images matching k8s version {k8s_version}")
                # Group by osMake + osVersion (distinct display names)
                grouped = {}
                for img in filtered:
                    os_make = img.get('osMake', 'Unknown')
                    os_version = img.get('osVersion', '')
                    key = f"{os_make} {os_version}".strip()
                    if key not in grouped:
                        grouped[key] = {
                            "display_name": key,
                            "os_id": img.get("id"),
                            "os_make": os_make,
                            "os_model": img.get("osModel"),
                            "os_version": os_version,
                            "hypervisor": img.get("hypervisor"),
                            "image_id": img.get("IMAGEID"),
                            "image_name": img.get("ImageName"),
                            "images": []  }
                    grouped[key]["images"].append(img)
                logger.info(f"✅ Found {len(grouped)} distinct OS options: {list(grouped.keys())}")
                return {
                    "success": True,
                    "os_options": list(grouped.values())}
            else:
                logger.error(f"❌ API returned error: {api_data}")
                return {"success": False, "error": "API returned error", "os_options": []}     
        except httpx.HTTPStatusError as e:
            logger.error(f"❌ HTTP error fetching OS images: {e}")
            return {"success": False, "error": f"HTTP {e.response.status_code}", "os_options": []}
        except Exception as e:
            logger.error(f"❌ Error fetching OS images: {str(e)}")
            return {"success": False, "error": str(e), "os_options": []}
        
    async def get_flavors(self, zone_id: int, os_model: str = None, node_type: str = None, k8s_version: str = None, auth_token: str = None, user_id: str = None) -> Dict[str, Any]:
        """
        Get compute flavors for zone using the flavordetails API.
        API: GET /uat-portalservice/configservice/flavordetails/{zoneId}
        Args:
            zone_id: Zone ID
            os_model: OS model filter
            node_type: Node type filter
            k8s_version: K8s version filter
            auth_token: Bearer token from UI for API authentication
            user_id: User ID for session lookup
        Returns:
            Dict with node_types (unique flavorCategory values) and formatted flavors
        """
        logger.info(f"💻 Fetching flavors for zone {zone_id}, OS filter: {os_model}, node type filter: {node_type}, k8s version: {k8s_version}")
        
        try:
            # Get auth token - prefer passed token
            token = auth_token or await self._get_or_refresh_token(user_id)
            if not token:
                return {"success": False, "error": "Failed to get auth token", "node_types": [], "flavors": []}
            # Call the flavordetails API directly
            url = f"https://ipcloud.tatacommunications.com/uat-portalservice/configservice/flavordetails/{zone_id}"
            logger.info(f"🌐 API Call: GET {url}")
            
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"}
            
            async with httpx.AsyncClient(timeout=self.api_timeout) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                api_data = response.json()
            # Parse response
            if api_data.get("status") == "success" and api_data.get("data"):
                all_flavors = api_data["data"].get("flavor", [])
                logger.info(f"📦 Found {len(all_flavors)} total flavors from API")
                # Debug: Log sample flavor to see structure
                if all_flavors:
                    sample = all_flavors[0]
                    logger.info(f"🔍 Sample flavor: applicationType={sample.get('applicationType')}, osModel={sample.get('osModel')}, flavorCategory={sample.get('flavorCategory')}")
                    # Log all unique applicationTypes and flavorCategories for debugging
                    app_types = set(f.get("applicationType", "N/A") for f in all_flavors)
                    flavor_cats = set(f.get("flavorCategory", "N/A") for f in all_flavors)
                    logger.info(f"🔍 All applicationType values: {app_types}")
                    logger.info(f"🔍 All flavorCategory values: {flavor_cats}")
                # STRICT Filter by applicationType = "Container" (exact match)
                container_flavors = [f for f in all_flavors if f.get("applicationType") == "Container"]
                logger.info(f"🎯 Found {len(container_flavors)} container flavors (applicationType=Container)")
                # No k8s version filtering for flavors - flavors are compute configs, not tied to k8s version
                # Filter by OS model if provided (case-insensitive partial match)
                if os_model and container_flavors:
                    os_model_lower = os_model.lower()
                    # Try partial match 
                    filtered_by_os = [f for f in container_flavors 
                                      if os_model_lower in f.get("osModel", "").lower() 
                                      or f.get("osModel", "").lower() in os_model_lower]
                    if filtered_by_os:
                        container_flavors = filtered_by_os
                        logger.info(f"🎯 Filtered to {len(container_flavors)} flavors matching OS '{os_model}'")
                    else:
                        logger.info(f"⚠️ No OS match for '{os_model}', keeping all {len(container_flavors)} container flavors")
                logger.info(f"🎯 Total flavors after filtering: {len(container_flavors)}")
                # Extract unique node types (flavorCategory) - use raw values
                node_types_set = set()
                for f in container_flavors:
                    cat = f.get("flavorCategory")
                    if cat:
                        node_types_set.add(cat)
                node_types = list(node_types_set)
                logger.info(f"📋 Found {len(node_types)} unique node types: {node_types}")
                # Further filter by node type if provided
                if node_type:
                    container_flavors = [f for f in container_flavors if f.get("flavorCategory") == node_type]
                    logger.info(f"🎯 Filtered to {len(container_flavors)} flavors for node type '{node_type}'")
                # Format flavors with display name like "8 vCPU / 32 GB RAM / 100 GB Storage"
                formatted_flavors = []
                for flavor in container_flavors:
                    vcpu = flavor.get("vCpu", 0)
                    vram_mb = flavor.get("vRam", 0)
                    vram_gb = vram_mb // 1024 if vram_mb else 0
                    vdisk = flavor.get("vDisk", 0)
                    formatted_flavors.append({
                        "id": flavor.get("artifactId"),
                        "name": f"{vcpu} vCPU / {vram_gb} GB RAM / {vdisk} GB Storage",
                        "display_name": flavor.get("display_name", flavor.get("FlavorName")),
                        "flavor_name": flavor.get("FlavorName"),
                        "sku_code": flavor.get("skuCode"),
                        "circuit_id": flavor.get("circuitId"),
                        "vcpu": vcpu,
                        "vram_gb": vram_gb,
                        "vram_mb": vram_mb,
                        "disk_gb": vdisk,
                        "node_type": flavor.get("flavorCategory"),
                        "storage_type": flavor.get("storageType"),
                        "os_model": flavor.get("osModel")})
                logger.info(f"✅ Returning {len(node_types)} node types, {len(formatted_flavors)} formatted flavors")
                return {
                    "success": True,
                    "node_types": node_types,
                    "flavors": formatted_flavors,
                    "all_flavors": container_flavors}
            else:
                logger.error(f"❌ API returned error: {api_data}")
                return {"success": False, "error": "API returned error", "node_types": [], "flavors": []}
        except httpx.HTTPStatusError as e:
            logger.error(f"❌ HTTP error fetching flavors: {e}")
            return {"success": False, "error": f"HTTP {e.response.status_code}", "node_types": [], "flavors": []}
        except Exception as e:
            logger.error(f"❌ Error fetching flavors: {str(e)}")
            return {"success": False, "error": str(e), "node_types": [], "flavors": []}
    
    async def get_circuit_id(self, engagement_id: int) -> Optional[str]:
        """
        Get circuit ID (copfId) for engagement.
        Args:
            engagement_id: Engagement ID
        Returns:
            Circuit ID string or default
        """
        logger.info(f"🔌 Fetching circuit ID for engagement {engagement_id}")
        return "E-IPCTEAM-1602"
    
    async def get_business_units_list(self, ipc_engagement_id: int = None, user_id: str = None, force_refresh: bool = False, auth_token: str = None) -> Dict[str, Any]:
        """
        Get business units (departments) listing for engagement.
        Uses per-user session storage to avoid repeated API calls.
        Args:
            ipc_engagement_id: IPC Engagement ID (will be fetched if not provided)
            user_id: User ID (email) for session lookup
            force_refresh: Force fetch even if cached
            auth_token: Bearer token from UI for API authentication
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
                        "ipc_engagement_id": session.get("ipc_engagement_id")}
            else:
                logger.info(f"🔄 Force refresh requested, bypassing cache")
            # Get IPC engagement ID if not provided
            if not ipc_engagement_id:
                ipc_engagement_id = await self.get_ipc_engagement_id(user_id=user_id, auth_token=auth_token)
                if not ipc_engagement_id:
                    return {
                        "success": False,
                        "error": "Failed to get IPC engagement ID",
                        "data": None}
                logger.info(f"✅ Got IPC engagement ID: {ipc_engagement_id}")
            url = f"https://ipcloud.tatacommunications.com/portalservice/securityservice/departments/{ipc_engagement_id}"
            logger.info(f"🏢 Fetching business units from: {url} (IPC engagement ID: {ipc_engagement_id})")
            # Get auth token - prefer passed token
            token = auth_token or await self._get_or_refresh_token(user_id)
            if not token:
                return {
                    "success": False,
                    "error": "Failed to get authentication token",
                    "data": None}
            # Make API call
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"}
    
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
                        business_units=bu_data)
                    logger.info(f"✅ Cached {len(departments)} business units for engagement '{engagement_info.get('name')}'")
                    return {
                        "success": True,
                        "data": bu_data,
                        "engagement": engagement_info,
                        "departments": departments,
                        "ipc_engagement_id": ipc_engagement_id}
                else:
                    logger.error(f"❌ API returned error: {data}")
                    return {
                        "success": False,
                        "error": data.get("message", "Unknown error"),
                        "data": None}
        except httpx.HTTPStatusError as e:
            logger.error(f"❌ HTTP error fetching business units: {e}")
            return {
                "success": False,
                "error": f"HTTP {e.response.status_code}: {str(e)}",
                "data": None}
        except Exception as e:
            logger.error(f"❌ Error fetching business units: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "data": None}

    async def get_department_details(self, ipc_engagement_id: int = None, user_id: str = None, force_refresh: bool = False, auth_token: str = None) -> Dict[str, Any]:
        """
        Get full department details including nested environments and zones.
        This API returns hierarchical data:
        - departmentList (Business Units)
            - environmentList (for each BU)
                - zoneList (for each Environment)
        Args:
            ipc_engagement_id: IPC Engagement ID (will be fetched if not provided)
            user_id: User ID (email) for session lookup
            force_refresh: Force fetch even if cached
            auth_token: Bearer token from UI for API authentication
        Returns:
            Dict with full department hierarchy including environments and zones
        """
        try:
            if not user_id:
                user_id = self._get_user_id_from_email()
            if not force_refresh:
                session = await self._get_user_session(user_id)
                if session and "department_details" in session:
                    dept_data = session["department_details"]
                    dept_list = dept_data.get("departmentList", [])
                    logger.info(f"📋 Using cached department details ({len(dept_list)} departments)")
                    return {
                        "success": True,
                        "data": dept_data,
                        "departmentList": dept_list,
                        "ipc_engagement_id": session.get("ipc_engagement_id")}
            # Get IPC engagement ID if not provided
            if not ipc_engagement_id:
                ipc_engagement_id = await self.get_ipc_engagement_id(user_id=user_id, auth_token=auth_token)
                if not ipc_engagement_id:
                    return {
                        "success": False,
                        "error": "Failed to get IPC engagement ID",
                        "data": None}
            url = f"https://ipcloud.tatacommunications.com/portalservice/securityservice/deptDetailsForEngagement/{ipc_engagement_id}"
            logger.info(f"🏢 Fetching department details from: {url}")
            # Get auth token - prefer passed token, fallback to refresh
            token = auth_token or await self._get_or_refresh_token()
            if not token:
                return {
                    "success": False,
                    "error": "Failed to get authentication token",
                    "data": None}
            # Make API call
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"}
            
            async with httpx.AsyncClient(timeout=self.api_timeout) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()
                if data.get("status") == "success":
                    dept_data = data.get("data", {})
                    dept_list = dept_data.get("departmentList", [])
                    # Update user session with department details
                    await self._update_user_session(
                        user_id=user_id,
                        department_details=dept_data)
                    logger.info(f"✅ Cached {len(dept_list)} departments with nested environments/zones")
                    return {
                        "success": True,
                        "data": dept_data,
                        "departmentList": dept_list,
                        "ipc_engagement_id": ipc_engagement_id}
                else:
                    logger.error(f"❌ API returned error: {data}")
                    return {
                        "success": False,
                        "error": data.get("message", "Unknown error"),
                        "data": None}
        except httpx.HTTPStatusError as e:
            logger.error(f"❌ HTTP error fetching department details: {e}")
            return {
                "success": False,
                "error": f"HTTP {e.response.status_code}: {str(e)}",
                "data": None}
        except Exception as e:
            logger.error(f"❌ Error fetching department details: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "data": None}
    
    async def get_environments_list(self, ipc_engagement_id: int = None, user_id: str = None, force_refresh: bool = False, auth_token: str = None) -> Dict[str, Any]:
        """
        Get environments listing per engagement.
        Uses per-user session storage to avoid repeated API calls.
        Args:
            ipc_engagement_id: IPC Engagement ID (will be fetched if not provided)
            user_id: User ID (email) for session lookup
            force_refresh: Force fetch even if cached
            auth_token: Bearer token from UI for API authentication
        Returns:
            Dict with environments data
        """
        try:
            if not user_id:
                user_id = self._get_user_id_from_email()
            if not force_refresh:
                session = await self._get_user_session(user_id)
                if session and "environments_list" in session:
                    environments = session["environments_list"]
                    logger.debug(f"✅ Using cached environments from session ({len(environments)} environments)")
                    return {
                        "success": True,
                        "data": environments,
                        "environments": environments,
                        "ipc_engagement_id": session.get("ipc_engagement_id")}
            
            if not ipc_engagement_id:
                ipc_engagement_id = await self.get_ipc_engagement_id(user_id=user_id, auth_token=auth_token)
                if not ipc_engagement_id:
                    return {
                        "success": False,
                        "error": "Failed to get IPC engagement ID",
                        "data": None}
                logger.info(f"✅ Got IPC engagement ID: {ipc_engagement_id}")
            url = f"https://ipcloud.tatacommunications.com/portalservice/securityservice/environmentsperengagement/{ipc_engagement_id}"
            logger.info(f"🌍 Fetching environments from: {url}")
            # Get auth token - prefer passed token
            token = auth_token or await self._get_or_refresh_token(user_id)
            if not token:
                return {
                    "success": False,
                    "error": "Failed to get authentication token",
                    "data": None}
            # Make API call
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"}
            async with httpx.AsyncClient(timeout=self.api_timeout) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()
                if data.get("status") == "success":
                    environments = data.get("data", [])
                    # Update user session with environments data
                    await self._update_user_session(user_id=user_id,environments_list=environments)
                    logger.info(f"✅ Cached {len(environments)} environments")
                    return {
                        "success": True,
                        "data": environments,
                        "environments": environments,
                        "ipc_engagement_id": ipc_engagement_id}
                else:
                    logger.error(f"❌ API returned error: {data}")
                    return {"success": False,"error": data.get("message", "Unknown error"),"data": None}   
        except httpx.HTTPStatusError as e:
            logger.error(f"❌ HTTP error fetching environments: {e}")
            return {
                "success": False,
                "error": f"HTTP {e.response.status_code}: {str(e)}",
                "data": None}
        except Exception as e:
            logger.error(f"❌ Error fetching environments: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "data": None}
        
    async def get_zones_list(self, ipc_engagement_id: int = None, user_id: str = None, force_refresh: bool = False, auth_token: str = None) -> Dict[str, Any]:
        """
        Get zones (network segments/VLANs) listing for engagement.
        Uses per-user session storage to avoid repeated API calls.
        Args:
            ipc_engagement_id: IPC Engagement ID (will be fetched if not provided)
            user_id: User ID (email) for session lookup
            force_refresh: Force fetch even if cached
            auth_token: Bearer token from UI for API authentication
        Returns:
            Dict with zones data including CIDR, hypervisors, status, and associated environments
        """
        try:
            if not user_id:
                user_id = self._get_user_id_from_email()
            if not force_refresh:
                session = await self._get_user_session(user_id)
                if session and "zones_list" in session:
                    zones = session["zones_list"]
                    logger.debug(f"✅ Using cached zones from session ({len(zones)} zones)")
                    return {
                        "success": True,
                        "data": zones,
                        "zones": zones,
                        "ipc_engagement_id": session.get("ipc_engagement_id")}

            if not ipc_engagement_id:
                ipc_engagement_id = await self.get_ipc_engagement_id(user_id=user_id, auth_token=auth_token)
                if not ipc_engagement_id:
                    return {
                        "success": False,
                        "error": "Failed to get IPC engagement ID",
                        "data": None}
                logger.info(f"✅ Got IPC engagement ID: {ipc_engagement_id}")

            url = f"https://ipcloud.tatacommunications.com/portalservice/api/v1/{ipc_engagement_id}/zonelist"
            logger.info(f"🌐 Fetching zones from: {url}")
            # Get auth token - prefer passed token
            token = auth_token or await self._get_or_refresh_token(user_id)
            if not token:
                return {
                    "success": False,
                    "error": "Failed to get authentication token",
                    "data": None}
            
            # Make API call
            headers = {"Authorization": f"Bearer {token}","Content-Type": "application/json"}
            
            async with httpx.AsyncClient(timeout=self.api_timeout) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()
                if data.get("status") == "success":
                    zones = data.get("data", [])
                    # Update user session with zones data
                    await self._update_user_session(user_id=user_id,zones_list=zones)
                    logger.info(f"✅ Cached {len(zones)} zones")
                    return {
                        "success": True,
                        "data": zones,
                        "zones": zones,
                        "ipc_engagement_id": ipc_engagement_id}
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
                "data": None}
        except Exception as e:
            logger.error(f"❌ Error fetching zones: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "data": None}
    def __repr__(self) -> str:
        resource_count = len(self.resource_schema.get("resources", {}))
        return f"<APIExecutorService(resources={resource_count})>"
    
# Global instance
api_executor_service = APIExecutorService()
