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
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("Token has expired")
        return None
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid token: {e}")
        return None
    except Exception as e:
        logger.error(f"Error decoding token: {e}")
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
        logger.warning("Token missing 'sub' claim")
        return None
    
    logger.debug(f"Extracted user from token: {username} ({email})")
    
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
            # Use email as user_id for consistency with chat storage
            return user.email or user.username or user.user_id
    
    # Fallback to explicit headers
    if x_user_id:
        return x_user_id
    
    if x_user_email:
        return x_user_email
    
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
