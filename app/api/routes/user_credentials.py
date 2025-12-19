"""
API endpoints for managing user API credentials.
Allows users to set their Tata Communications API credentials.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr
from typing import Optional
from app.api.routes.auth import get_current_user, User as AuthUser
from app.services.user_credentials_service import user_credentials_service
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/user", tags=["User Credentials"])


class UpdateCredentialsRequest(BaseModel):
    """Request model for updating API credentials."""
    api_auth_email: EmailStr
    api_auth_password: str


class CredentialsResponse(BaseModel):
    """Response model for credentials operations."""
    success: bool
    message: str
    has_credentials: Optional[bool] = None


@router.put("/credentials", response_model=CredentialsResponse)
async def update_credentials(
    request: UpdateCredentialsRequest,
    current_user: AuthUser = Depends(get_current_user)
):
    """
    Update API credentials for the current user.
    
    Args:
        request: Credentials update request
        current_user: Current authenticated user
        
    Returns:
        Success status and message
    """
    try:
        success = user_credentials_service.update_user_credentials(
            username=current_user.username,
            api_auth_email=request.api_auth_email,
            api_auth_password=request.api_auth_password
        )
        
        if success:
            logger.info(f"✅ Updated API credentials for user: {current_user.username}")
            return CredentialsResponse(
                success=True,
                message="API credentials updated successfully",
                has_credentials=True
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to update credentials"
            )
            
    except Exception as e:
        logger.error(f"❌ Error updating credentials: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating credentials: {str(e)}"
        )


@router.get("/credentials", response_model=CredentialsResponse)
async def get_credentials_status(
    current_user: AuthUser = Depends(get_current_user)
):
    """
    Check if the current user has API credentials configured.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        Whether credentials are configured
    """
    try:
        credentials = user_credentials_service.get_user_credentials(current_user.username)
        has_credentials = credentials is not None
        
        return CredentialsResponse(
            success=True,
            message="Credentials status retrieved",
            has_credentials=has_credentials
        )
        
    except Exception as e:
        logger.error(f"❌ Error checking credentials: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error checking credentials: {str(e)}"
        )


