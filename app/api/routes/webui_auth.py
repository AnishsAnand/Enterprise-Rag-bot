"""
Custom Auth Endpoint for OpenWebUI Integration
Validates users via Tata Auth API on login and stores credentials
"""

from fastapi import APIRouter, HTTPException, status, Response
from pydantic import BaseModel, EmailStr
from typing import Optional
import logging
from passlib.context import CryptContext
from datetime import datetime, timedelta
from jose import jwt
from app.services.tata_auth_service import tata_auth_service
from app.services.user_credentials_service import user_credentials_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/openwebui-auth", tags=["OpenWebUI Auth"])
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
# JWT settings (should match your main auth)
SECRET_KEY = "cBbPN3Sa8Yu_mtVrJCozcPpnE0FDXDCSwWZKY-Opw30"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

class LoginRequest(BaseModel):
    """Login request with email and password"""
    email: EmailStr
    password: str

class LoginResponse(BaseModel):
    """Login response with token and access level"""
    success: bool
    token: str
    access_level: str 
    user_info: dict
    message: str

@router.post("/login", response_model=LoginResponse)
async def login_with_tata_validation(request: LoginRequest, response: Response):
    """
    Login endpoint that validates against Tata Auth API
    Flow:
    1. User provides email + password
    2. Call Tata Auth API to validate
    3. If API returns 200 + token → Full access
    4. If API returns 500 or error → Read-only access
    5. Return JWT token with access level embedded
    """
    try:
        # Store credentials for future API calls
        # This allows API executor to use user's credentials instead of env vars
        stored = user_credentials_service.store_credentials(email=request.email,password=request.password)
        if stored:
            logger.info(f"✅ Stored credentials for user: {request.email}")
        else:
            logger.warning(f"⚠️ Could not store credentials for: {request.email}")
        # Validate against Tata Auth API (SINGLE SOURCE OF TRUTH)
        validation_result = await tata_auth_service.validate_user(email=request.email,password=request.password)
        # Determine access level based on Tata API response
        if validation_result["valid"] and validation_result["access_level"] == "full":
            # Valid Tata user - Full access
            access_level = "full"
            user_roles = ["admin", "developer", "viewer"]
            message = "Authenticated as Tata Communications user. Full access granted."
            tata_validated = True
            logger.info(f"✅ Tata Auth Success: {request.email} | Full Access")
        else:
            # Not a Tata user or invalid credentials - Read-only access
            access_level = "read_only"
            user_roles = ["viewer"]
            message = "Not a Tata Communications user. Read-only access granted (RAG docs only, no actions)."
            tata_validated = False
            logger.info(f"ℹ️ Non-Tata User: {request.email} | Read-Only Access")
        # Create JWT token with access level embedded
        token_data = {
            "sub": request.email,
            "email": request.email,
            "roles": user_roles,
            "access_level": access_level,
            "tata_validated": tata_validated,
            "exp": datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)}
        token = jwt.encode(token_data, SECRET_KEY, algorithm=ALGORITHM)
        # Set custom header for downstream services
        response.headers["X-Tata-Validated"] = str(tata_validated)
        response.headers["X-Access-Level"] = access_level
        return LoginResponse(success=True,token=token,access_level=access_level,user_info=validation_result.get("user_info", {"email": request.email}),message=message)
    except Exception as e:
        logger.error(f"❌ Login error for {request.email}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Authentication error: {str(e)}")

@router.post("/validate-token")
async def validate_token(token: str):
    """
    Validate a JWT token and return access level
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return {
            "valid": True,
            "email": payload.get("email"),
            "access_level": payload.get("access_level"),
            "tata_validated": payload.get("tata_validated"),
            "roles": payload.get("roles")}
    except Exception as e:
        logger.error(f"❌ Token validation error: {e}")
        return {
            "valid": False,
            "error": str(e)}
    
class SetupCredentialsRequest(BaseModel):
    """Request to store API credentials for future use"""
    email: EmailStr
    password: str

class SetupCredentialsResponse(BaseModel):
    """Response from credentials setup"""
    success: bool
    message: str
    tata_validated: bool
    access_level: str

@router.post("/setup-credentials", response_model=SetupCredentialsResponse)
async def setup_credentials(request: SetupCredentialsRequest):
    """
    Store user credentials for API access.
    This endpoint allows users to set up their Tata API credentials after signing up in WebUI.
    Flow:
    1. User signs up in WebUI
    2. User calls this endpoint with their email/password
    3. Credentials are stored (encrypted) for future API calls
    4. Credentials are validated against Tata Auth API to determine access level
    """
    try:
        # Store credentials for future API calls
        stored = user_credentials_service.store_credentials(email=request.email,password=request.password)
        if not stored:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to store credentials")
        logger.info(f"✅ Credentials stored for: {request.email}")
        # Validate against Tata Auth API to determine access level
        validation_result = await tata_auth_service.validate_user(email=request.email,password=request.password)
        tata_validated = validation_result.get("valid", False) and validation_result.get("access_level") == "full"
        access_level = "full" if tata_validated else "read_only"
        if tata_validated:
            message = f"Credentials stored and validated. Full access granted for {request.email}."
        else:
            message = f"Credentials stored. Read-only access (RAG docs only) for {request.email}."
        return SetupCredentialsResponse(success=True,message=message,tata_validated=tata_validated,access_level=access_level)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error setting up credentials: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error setting up credentials: {str(e)}")
    
@router.get("/check-credentials/{email}")
async def check_credentials(email: str):
    """
    Check if credentials are stored for a user.
    """
    credentials = user_credentials_service.get_credentials_by_email(email)
    return {
        "email": email,
        "has_credentials": credentials is not None,
        "message": "Credentials found" if credentials else "No credentials stored. Please call /setup-credentials first."
    }