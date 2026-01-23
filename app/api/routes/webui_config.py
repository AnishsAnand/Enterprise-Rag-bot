"""
WebUI-compatible Config API Routes
Provides OpenWebUI-style /api/config endpoint

This endpoint returns application configuration that the frontend uses
to determine available features, default settings, etc.
"""

import os
import logging
from typing import Optional, Dict, Any, List

from fastapi import APIRouter, Request, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.config import settings

log = logging.getLogger(__name__)

router = APIRouter(tags=["config"])


# ==================== Configuration Models ====================

class OAuthProviders(BaseModel):
    providers: Dict[str, str] = {}


class FeaturesConfig(BaseModel):
    auth: bool = True
    auth_trusted_header: bool = False
    enable_signup: bool = True
    enable_login_form: bool = True
    enable_api_keys: bool = True
    enable_websocket: bool = True
    enable_channels: bool = True
    enable_notes: bool = True
    enable_folders: bool = True
    enable_community_sharing: bool = True
    enable_message_rating: bool = True
    enable_web_search: bool = True
    enable_code_execution: bool = False
    enable_code_interpreter: bool = False
    enable_image_generation: bool = False
    enable_autocomplete_generation: bool = True
    enable_memories: bool = True
    enable_direct_connections: bool = False
    folder_max_file_count: int = 100


class AudioConfig(BaseModel):
    tts: Dict[str, Any] = {
        "engine": "",
        "voice": "",
        "split_on": "punctuation"
    }
    stt: Dict[str, Any] = {
        "engine": ""
    }


class CodeConfig(BaseModel):
    engine: str = ""


class FileConfig(BaseModel):
    max_size: int = 100 * 1024 * 1024  # 100MB
    max_count: int = 10
    image_compression: Dict[str, int] = {
        "width": 1920,
        "height": 1080
    }


class UIConfig(BaseModel):
    pending_user_overlay_title: str = ""
    pending_user_overlay_content: str = ""
    response_watermark: str = ""


class ConfigResponse(BaseModel):
    status: bool = True
    name: str = "Enterprise RAG Bot"
    version: str = "2.0.0"
    default_locale: str = "en-US"
    oauth: OAuthProviders = OAuthProviders()
    features: FeaturesConfig = FeaturesConfig()
    default_models: str = ""
    default_pinned_models: List[str] = []
    default_prompt_suggestions: List[Dict[str, Any]] = []
    user_count: int = 0
    code: CodeConfig = CodeConfig()
    audio: AudioConfig = AudioConfig()
    file: FileConfig = FileConfig()
    permissions: Dict[str, Any] = {}
    ui: UIConfig = UIConfig()
    onboarding: bool = False


# ==================== Config Endpoint ====================

@router.get("/api/config")
async def get_app_config(
    request: Request,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get application configuration.
    
    This endpoint returns the same structure as OpenWebUI's /api/config
    so that frontends can determine available features and settings.
    
    The response varies based on whether a user is authenticated:
    - Unauthenticated: Basic config + onboarding status
    - Authenticated: Full config including features, permissions, etc.
    """
    
    # Check if user is authenticated
    user = None
    user_id = None
    
    # Try to get user from various auth methods
    auth_header = request.headers.get("Authorization")
    user_id_header = request.headers.get("X-User-Id")
    
    if user_id_header:
        user_id = user_id_header
        user = {"id": user_id, "role": "user"}  # Simplified for now
    elif auth_header and auth_header.startswith("Bearer "):
        # In production: decode JWT token
        user = {"id": "default_user", "role": "user"}
    
    # Count users (for onboarding detection)
    try:
        from app.models.database_models import User
        user_count = db.query(User).count()
    except Exception:
        user_count = 1  # Assume at least one user exists
    
    onboarding = user is None and user_count == 0
    
    # Base config (always returned)
    base_config = {
        "status": True,
        "name": os.getenv("APP_NAME", "Enterprise RAG Bot"),
        "version": "2.0.0",
        "default_locale": os.getenv("DEFAULT_LOCALE", "en-US"),
        "oauth": {
            "providers": {}  # Add OAuth providers here if configured
        },
        "features": {
            "auth": True,
            "auth_trusted_header": False,
            "enable_signup": os.getenv("ENABLE_SIGNUP", "true").lower() == "true",
            "enable_login_form": True,
            "enable_api_keys": True,
            "enable_websocket": True,
        },
    }
    
    if onboarding:
        base_config["onboarding"] = True
        return base_config
    
    # Extended config for authenticated users
    if user is not None:
        base_config["features"].update({
            "enable_channels": True,
            "enable_notes": True,
            "enable_folders": True,
            "folder_max_file_count": 100,
            "enable_community_sharing": True,
            "enable_message_rating": True,
            "enable_web_search": os.getenv("ENABLE_WEB_SEARCH", "true").lower() == "true",
            "enable_code_execution": os.getenv("ENABLE_CODE_EXECUTION", "false").lower() == "true",
            "enable_code_interpreter": os.getenv("ENABLE_CODE_INTERPRETER", "false").lower() == "true",
            "enable_image_generation": os.getenv("ENABLE_IMAGE_GENERATION", "false").lower() == "true",
            "enable_autocomplete_generation": True,
            "enable_memories": True,
            "enable_direct_connections": False,
            "enable_user_webhooks": False,
            "enable_user_status": True,
            "enable_admin_export": user.get("role") == "admin",
            "enable_admin_chat_access": user.get("role") == "admin",
        })
        
        # Add user-specific config
        base_config.update({
            "default_models": os.getenv("DEFAULT_MODELS", ""),
            "default_pinned_models": [],
            "default_prompt_suggestions": _get_default_prompt_suggestions(),
            "user_count": user_count,
            "code": {
                "engine": os.getenv("CODE_EXECUTION_ENGINE", ""),
            },
            "audio": {
                "tts": {
                    "engine": os.getenv("TTS_ENGINE", ""),
                    "voice": os.getenv("TTS_VOICE", ""),
                    "split_on": "punctuation",
                },
                "stt": {
                    "engine": os.getenv("STT_ENGINE", ""),
                },
            },
            "file": {
                "max_size": int(os.getenv("FILE_MAX_SIZE", "104857600")),  # 100MB
                "max_count": int(os.getenv("FILE_MAX_COUNT", "10")),
                "image_compression": {
                    "width": 1920,
                    "height": 1080,
                },
            },
            "permissions": _get_user_permissions(user.get("role", "user")),
            "ui": {
                "pending_user_overlay_title": "",
                "pending_user_overlay_content": "",
                "response_watermark": "",
            },
        })
        
        # Admin-only fields
        if user.get("role") == "admin":
            base_config["active_entries"] = user_count
    
    return base_config


def _get_default_prompt_suggestions() -> List[Dict[str, Any]]:
    """Get default prompt suggestions for the UI"""
    return [
        {
            "title": "Help me write",
            "content": "Help me write a professional email about..."
        },
        {
            "title": "Explain a concept",
            "content": "Explain the concept of..."
        },
        {
            "title": "Summarize text",
            "content": "Summarize the following text:"
        },
        {
            "title": "Code assistance",
            "content": "Help me write code to..."
        },
    ]


def _get_user_permissions(role: str) -> Dict[str, Any]:
    """Get permissions based on user role"""
    base_permissions = {
        "chat": {
            "delete": True,
            "edit": True,
            "share": True,
        },
        "workspace": {
            "models": False,
            "knowledge": True,
            "prompts": True,
            "tools": False,
        },
    }
    
    if role == "admin":
        base_permissions["workspace"]["models"] = True
        base_permissions["workspace"]["tools"] = True
    
    return base_permissions


# ==================== Additional Config Endpoints ====================

@router.get("/api/version")
async def get_app_version():
    """Get application version"""
    return {
        "version": "2.0.0",
        "deployment_id": os.getenv("DEPLOYMENT_ID", "local"),
    }


@router.get("/api/changelog")
async def get_app_changelog():
    """Get application changelog"""
    return {
        "2.0.0": {
            "date": "2024-01-01",
            "changes": [
                "Initial release of Enterprise RAG Bot",
                "OpenWebUI-compatible API",
                "Chat persistence with PostgreSQL",
            ]
        }
    }


@router.get("/manifest.json")
async def get_manifest():
    """PWA manifest"""
    app_name = os.getenv("APP_NAME", "Enterprise RAG Bot")
    return {
        "name": app_name,
        "short_name": app_name,
        "description": f"{app_name} - AI-powered knowledge assistant",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#343541",
        "theme_color": "#343541",
        "icons": [
            {
                "src": "/static/logo.png",
                "type": "image/png",
                "sizes": "500x500",
                "purpose": "any"
            }
        ]
    }
