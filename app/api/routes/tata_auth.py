"""
Tata Communications Authentication Endpoint
Allows WebUI to validate users against Tata's auth API

SINGLE SOURCE OF TRUTH:
- Tata Auth API response determines access level
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, EmailStr
from typing import Optional
import logging
from app.services.tata_auth_service import tata_auth_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/tata-auth", tags=["Tata Authentication"])

class TataAuthRequest(BaseModel):
    """Request model for Tata authentication"""
    email: EmailStr
    password: str

class TataAuthResponse(BaseModel):
    """Response model for Tata authentication"""
    success: bool
    access_level: str  
    message: str
    user_info: Optional[dict] = None
    token: Optional[str] = None

@router.post("/validate", response_model=TataAuthResponse)
async def validate_tata_user(request: TataAuthRequest):
    """
    Validate user credentials against Tata Communications API
    Returns:
        - success: True if validation succeeded
        - access_level: "full" for Tata users, "read_only" for others
        - message: Human-readable message
        - user_info: User details if valid
        - token: Tata access token if valid Tata user
    """
    try:
        # Validate against Tata API
        result = await tata_auth_service.validate_user(
            email=request.email,
            password=request.password)
        if result["valid"]:
            # Valid Tata Communications user
            return TataAuthResponse(
                success=True,
                access_level="full",
                message="Authenticated as Tata Communications user. Full access granted.",
                user_info=result["user_info"],
                token=result["token"])
        else:
            # Not a Tata user (or invalid credentials)
            # Still allow access but read-only
            return TataAuthResponse(
                success=True,  
                access_level="read_only",
                message="Not a Tata Communications user. Read-only access granted (RAG docs only, no actions).",
                user_info=result["user_info"],
                token=None)
    except Exception as e:
        logger.error(f"Error in Tata auth validation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Authentication service error: {str(e)}")

@router.get("/check-email/{email}")
async def check_email_domain(email: str):
    """
    Quick check if email is from Tata Communications domain
    (doesn't validate credentials, just checks domain)
    """
    is_tata = email.endswith("@tatacommunications.com") or \
              email.endswith("@tatacommunications.onmicrosoft.com")
    
    return {"email": email,"is_tata_domain": is_tata,"expected_access": "full" if is_tata else "read_only"}