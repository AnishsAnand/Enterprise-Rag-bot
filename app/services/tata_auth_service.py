"""
Tata Communications Authentication Service
Validates users against Tata's auth API to determine access level
"""

import httpx
import logging
from typing import Dict, Optional, List
import jwt
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class TataAuthService:
    """
    Service to authenticate users via Tata Communications API
    and determine their access level
    """
    
    def __init__(self):
        self.auth_url = "https://ipcloud.tatacommunications.com/portalservice/api/v1/getAuthToken"
        self.token_cache: Dict[str, Dict] = {}  # Cache validated tokens
        self.cache_duration = timedelta(minutes=30)  
    
    async def validate_user(self, email: str, password: str) -> Dict[str, any]:
        """
        Validate user credentials against Tata Communications API
        Args:
            email: User's email
            password: User's password
        Returns:
            {
                "valid": True/False,
                "access_level": "full" or "read_only",
                "user_info": {...},
                "token": "..." (if valid)
            }
        """
        try:
            # Call Tata Communications auth API
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    self.auth_url,
                    json={"email": email, "password": password},
                    headers={"Content-Type": "application/json"})
                data = response.json()
                # Check response
                if data.get("statusCode") == 200 and data.get("accessToken"):
                    # Valid Tata Communications user
                    access_token = data["accessToken"]
                    # Decode JWT to get user info (without verification for now)
                    try:
                        user_info = jwt.decode(
                            access_token,
                            options={"verify_signature": False})
                        # Extract user details
                        user_email = user_info.get("email", email)
                        user_name = user_info.get("name", "")
                        roles = user_info.get("realm_access", {}).get("roles", [])
                        # Cache the token
                        self.token_cache[email] = {
                            "token": access_token,
                            "user_info": user_info,
                            "cached_at": datetime.now()}
                        logger.info(f"✅ Valid Tata user: {user_email} | Roles: {roles}")
                        return {
                            "valid": True,
                            "access_level": "full",  # Tata users get full access
                            "user_info": {
                                "email": user_email,
                                "name": user_name,
                                "roles": roles,
                                "provider": "tata"
                            },
                            "token": access_token
                        }
                    except Exception as e:
                        logger.error(f"Failed to decode JWT: {e}")
                        # Even if decode fails, if we got 200 + token, user is valid
                        return {
                            "valid": True,
                            "access_level": "full",
                            "user_info": {
                                "email": email,
                                "name": email.split("@")[0],
                                "roles": [],
                                "provider": "tata"
                            },
                            "token": access_token
                        }
                elif data.get("statusCode") == 500:
                    # Invalid credentials or not a Tata user
                    logger.info(f"❌ Invalid Tata credentials for: {email}")
                    return {
                        "valid": False,
                        "access_level": "read_only",  # Not a Tata user, read-only access
                        "user_info": {
                            "email": email,
                            "name": email.split("@")[0],
                            "roles": [],
                            "provider": "local"
                        },
                        "token": None
                    }
                else:
                    # Unexpected response
                    logger.warning(f"⚠️ Unexpected auth response: {data}")
                    return {
                        "valid": False,
                        "access_level": "read_only",
                        "user_info": {"email": email, "provider": "local"},
                        "token": None}
        
        except httpx.TimeoutException:
            logger.error(f"⏱️ Tata auth API timeout for {email}")
            # On timeout, allow read-only access (graceful degradation)
            return {
                "valid": False,
                "access_level": "read_only",
                "user_info": {"email": email, "provider": "local"},
                "token": None,
                "error": "Auth service timeout"}
        
        except Exception as e:
            logger.error(f"❌ Error validating user {email}: {e}")
            return {
                "valid": False,
                "access_level": "read_only",
                "user_info": {"email": email, "provider": "local"},
                "token": None,
                "error": str(e)}
    
    def get_user_roles(self, access_level: str) -> List[str]:
        """
        Convert access level to user roles for permission checking
        Args:
            access_level: "full" or "read_only"
        Returns:
            List of roles for permission system
        """
        if access_level == "full":
            # Tata users get full access
            return ["admin", "developer", "viewer"]
        else:
            # Non-Tata users get read-only access
            return ["viewer"]
    
    def is_cached(self, email: str) -> bool:
        """Check if user's token is cached and still valid"""
        if email not in self.token_cache:
            return False
        cached = self.token_cache[email]
        cached_at = cached.get("cached_at")
        
        if not cached_at:
            return False
        # Check if cache is still valid (within cache_duration)
        if datetime.now() - cached_at > self.cache_duration:
            # Cache expired
            del self.token_cache[email]
            return False
        return True
    
    def get_cached_token(self, email: str) -> Optional[str]:
        """Get cached token for user"""
        if self.is_cached(email):
            return self.token_cache[email].get("token")
        return None

# Global instance
tata_auth_service = TataAuthService()

