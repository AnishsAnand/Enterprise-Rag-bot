"""
Lightweight auth helpers for API routes.
"""

from typing import List, Optional

from fastapi import Header, HTTPException, status
from jose import JWTError, jwt
from pydantic import BaseModel, EmailStr

from app.api.routes.webui_auth import ALGORITHM, SECRET_KEY


class User(BaseModel):
    """Authenticated user context."""
    username: str
    email: Optional[EmailStr] = None
    roles: List[str] = []


def _parse_bearer_token(authorization: Optional[str]) -> str:
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
        )
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header",
        )
    return authorization.split(" ", 1)[1].strip()


def get_current_user(authorization: Optional[str] = Header(None)) -> User:
    """
    Decode JWT token from Authorization header and return user context.
    """
    token = _parse_bearer_token(authorization)
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        ) from exc

    email = payload.get("email") or payload.get("sub")
    roles = payload.get("roles") or []
    username = email or "user"
    return User(username=username, email=email, roles=roles)
