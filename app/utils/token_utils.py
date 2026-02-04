"""
Token utilities for extracting user info from Keycloak JWT tokens.
Tokens are passed from the UI in the Authorization header.
"""

import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
import jwt  # PyJWT library

logger = logging.getLogger(__name__)


@dataclass
class TokenUser:
    """User info extracted from Keycloak JWT token."""
    user_id: str  # sub claim (UUID)
    email: str
    name: str
    username: str  # preferred_username
    roles: list
    raw_token: str  # Original token for pass-through to downstream APIs


def decode_keycloak_token(token: str, verify: bool = False) -> Optional[Dict[str, Any]]:
    """
    Decode a Keycloak JWT token.
    
    Args:
        token: JWT token string
        verify: Whether to verify the signature (requires public key)
        
    Returns:
        Decoded token payload or None if invalid
    """
    try:
        # Decode without verification for now
        # In production, you can verify using Keycloak's public key
        payload = jwt.decode(
            token,
            options={"verify_signature": verify},
            algorithms=["RS256", "HS256"]
        )
        logger.debug(f"✅ Successfully decoded token. Claims: {list(payload.keys())}")
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("❌ Token has expired")
        return None
    except jwt.InvalidTokenError as e:
        logger.warning(f"❌ Invalid token: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ Error decoding token: {e}", exc_info=True)
        return None


def extract_user_from_token(authorization_header: Optional[str]) -> Optional[TokenUser]:
    """
    Extract user info from Authorization header containing Keycloak Bearer token.
    
    Args:
        authorization_header: The Authorization header value (e.g., "Bearer eyJ...")
        
    Returns:
        TokenUser object with user info, or None if invalid/missing
    """
    if not authorization_header:
        logger.debug("No Authorization header provided")
        return None
    
    if not authorization_header.startswith("Bearer "):
        logger.warning("Authorization header is not a Bearer token")
        return None
    
    token = authorization_header[7:].strip()  # Remove "Bearer " prefix
    
    if not token:
        logger.warning("Empty token after Bearer prefix")
        return None
    
    payload = decode_keycloak_token(token)
    if not payload:
        logger.warning("❌ Failed to decode token payload")
        return None
    
    # Extract user info from Keycloak token claims
    user_id = payload.get("sub", "")
    email = payload.get("email", "")
    name = payload.get("name", "")
    username = payload.get("preferred_username", email)
    
    # Get roles from realm_access
    realm_access = payload.get("realm_access", {})
    roles = realm_access.get("roles", [])
    
    if not user_id:
        logger.error("❌ Token missing 'sub' claim - cannot identify user uniquely. Available claims: " + str(list(payload.keys())))
        return None
    
    # Log extracted user info for debugging user isolation
    logger.info(f"✅ Extracted user from token: sub={user_id[:20]}..., email={email or 'N/A'}, username={username or 'N/A'}")
    
    return TokenUser(
        user_id=user_id,
        email=email,
        name=name,
        username=username,
        roles=roles,
        raw_token=token
    )


def get_user_id_from_request(
    authorization: Optional[str] = None,
    x_user_id: Optional[str] = None,
    x_user_email: Optional[str] = None,
    default: str = "default_user"
) -> str:
    """
    Get user ID from request headers with fallbacks.
    
    Priority:
    1. Authorization Bearer token (Keycloak JWT)
    2. X-User-Id header
    3. X-User-Email header
    4. Default value
    
    Args:
        authorization: Authorization header value
        x_user_id: X-User-Id header value
        x_user_email: X-User-Email header value
        default: Default user ID if none found
        
    Returns:
        User ID string
    """
    # Try Authorization header first (Keycloak token)
    if authorization:
        user = extract_user_from_token(authorization)
        if user:
            # CRITICAL: Use email as primary user_id, but fallback to sub (UUID) if email missing
            # This ensures unique user identification even if email is not in token
            if user.email:
                logger.debug(f"Using email as user_id: {user.email}")
                return user.email
            elif user.username:
                logger.debug(f"Using username as user_id (email missing): {user.username}")
                return user.username
            elif user.user_id:
                logger.debug(f"Using sub claim as user_id (email/username missing): {user.user_id}")
                return user.user_id
            else:
                logger.warning("Token extracted but no user identifier found (email/username/sub)")
        else:
            logger.warning("Failed to extract user from Authorization token - token may be invalid or expired")
    
    # Fallback to explicit headers
    if x_user_id:
        logger.debug(f"Using X-User-Id header: {x_user_id}")
        return x_user_id
    
    if x_user_email:
        logger.debug(f"Using X-User-Email header: {x_user_email}")
        return x_user_email
    
    # CRITICAL: Only use default if NO authentication info is available
    # This prevents user isolation issues
    logger.warning(f"No user identification found - using default: {default}. This may cause user isolation issues!")
    return default


def get_token_from_request(authorization: Optional[str] = None) -> Optional[str]:
    """
    Extract raw token from Authorization header for pass-through to downstream APIs.
    
    Args:
        authorization: Authorization header value
        
    Returns:
        Raw token string or None
    """
    if not authorization or not authorization.startswith("Bearer "):
        return None
    return authorization[7:].strip()
