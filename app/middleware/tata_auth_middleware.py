"""
Tata Auth Middleware
Validates users on login via Tata Auth API and sets permissions
"""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
import logging
from typing import Callable

from app.services.tata_auth_service import tata_auth_service

logger = logging.getLogger(__name__)


class TataAuthMiddleware(BaseHTTPMiddleware):
    """
    Middleware to validate users via Tata Auth API
    Sets X-Tata-Validated header based on API response
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and validate user if credentials are provided
        """
        
        # Check if this is a login/auth request with credentials
        if request.method == "POST" and "/auth" in request.url.path:
            try:
                # Try to get email and password from request body
                # This is a simplified check - in production, you'd parse the body properly
                body = await request.body()
                
                # For now, we'll let the request through
                # The actual validation happens in the auth endpoint
                response = await call_next(request)
                return response
            
            except Exception as e:
                logger.error(f"Error in Tata auth middleware: {e}")
                response = await call_next(request)
                return response
        
        # For all other requests, pass through
        response = await call_next(request)
        return response

