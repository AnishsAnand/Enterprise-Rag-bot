"""
Chat Persistence API for PORTAL Frontend Integration

This module provides the chat history and persistence endpoints that frontends
like Open WebUI expect. It handles:
- Chat session management (create, list, get, delete)
- Message history within chats
- Pinned/archived chats
- Model listing (alias endpoint)

These endpoints are required for frontends that expect a full chat backend.
"""

from fastapi import APIRouter, HTTPException, Query, Header, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import logging

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Chat Persistence"])

# ============================================================================
# In-Memory Storage (Replace with database in production)
# ============================================================================

# Simple in-memory storage for chat sessions
# In production, replace with database tables
_chat_sessions: Dict[str, Dict[str, Any]] = {}
_chat_messages: Dict[str, List[Dict[str, Any]]] = {}


# ============================================================================
# Pydantic Models
# ============================================================================

class ChatMessage(BaseModel):
    """Individual chat message"""
    role: str = Field(..., description="Message role: user, assistant, system")
    content: str = Field(..., description="Message content")
    timestamp: Optional[str] = None
    id: Optional[str] = None


class ChatSession(BaseModel):
    """Chat session/conversation"""
    id: str
    title: str
    created_at: str
    updated_at: str
    messages: List[ChatMessage] = []
    pinned: bool = False
    archived: bool = False
    model: Optional[str] = None
    tags: List[str] = []


class CreateChatRequest(BaseModel):
    """Request to create a new chat"""
    title: Optional[str] = Field(None, description="Chat title (auto-generated if not provided)")
    model: Optional[str] = Field(None, description="Model to use for this chat")
    messages: Optional[List[ChatMessage]] = Field(default_factory=list)


class UpdateChatRequest(BaseModel):
    """Request to update a chat"""
    title: Optional[str] = None
    pinned: Optional[bool] = None
    archived: Optional[bool] = None
    tags: Optional[List[str]] = None


class AddMessageRequest(BaseModel):
    """Request to add a message to a chat"""
    role: str
    content: str


class ModelInfo(BaseModel):
    """Model information"""
    id: str
    object: str = "model"
    created: int
    owned_by: str = "Tata Communications"
    name: Optional[str] = None


class ModelListResponse(BaseModel):
    """List of available models"""
    object: str = "list"
    data: List[ModelInfo]


# ============================================================================
# Helper Functions
# ============================================================================

def generate_chat_id() -> str:
    """Generate unique chat ID"""
    return f"chat_{uuid.uuid4().hex[:12]}"


def generate_message_id() -> str:
    """Generate unique message ID"""
    return f"msg_{uuid.uuid4().hex[:8]}"


def generate_title_from_message(content: str) -> str:
    """Generate a chat title from the first message"""
    # Take first 50 chars of the message as title
    title = content[:50].strip()
    if len(content) > 50:
        title += "..."
    return title or "New Chat"


def get_current_timestamp() -> str:
    """Get current timestamp in ISO format"""
    return datetime.utcnow().isoformat() + "Z"


# ============================================================================
# Chat Endpoints
# ============================================================================

@router.get("/api/v1/chats/")
@router.get("/api/v1/chats")
async def list_chats(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    archived: bool = Query(False)
):
    """
    List all chat sessions.
    
    Returns paginated list of chats, sorted by most recent first.
    """
    try:
        # Filter and sort chats
        chats = [
            chat for chat in _chat_sessions.values()
            if chat.get("archived", False) == archived
        ]
        
        # Sort by updated_at descending
        chats.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        
        # Paginate
        paginated = chats[skip:skip + limit]
        
        return {
            "chats": paginated,
            "total": len(chats),
            "skip": skip,
            "limit": limit
        }
    except Exception as e:
        logger.exception(f"Failed to list chats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/v1/chats/pinned")
async def list_pinned_chats():
    """
    List all pinned chat sessions.
    """
    try:
        pinned = [
            chat for chat in _chat_sessions.values()
            if chat.get("pinned", False) and not chat.get("archived", False)
        ]
        pinned.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        
        return {"chats": pinned, "total": len(pinned)}
    except Exception as e:
        logger.exception(f"Failed to list pinned chats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/v1/chats/new")
async def create_chat(request: CreateChatRequest = None):
    """
    Create a new chat session.
    
    Returns the newly created chat with its ID.
    """
    try:
        if request is None:
            request = CreateChatRequest()
            
        chat_id = generate_chat_id()
        now = get_current_timestamp()
        
        # Generate title from first message if not provided
        title = request.title
        if not title and request.messages:
            first_user_msg = next(
                (m for m in request.messages if m.role == "user"),
                None
            )
            if first_user_msg:
                title = generate_title_from_message(first_user_msg.content)
        
        title = title or "New Chat"
        
        # Create chat session
        chat = {
            "id": chat_id,
            "title": title,
            "created_at": now,
            "updated_at": now,
            "pinned": False,
            "archived": False,
            "model": request.model or "Vayu Maya",
            "tags": [],
            "message_count": len(request.messages) if request.messages else 0
        }
        
        _chat_sessions[chat_id] = chat
        
        # Store messages if provided
        if request.messages:
            _chat_messages[chat_id] = [
                {
                    "id": generate_message_id(),
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp or now
                }
                for msg in request.messages
            ]
        else:
            _chat_messages[chat_id] = []
        
        logger.info(f"Created new chat: {chat_id}")
        
        return {
            "id": chat_id,
            "chat": chat,
            "messages": _chat_messages.get(chat_id, [])
        }
    except Exception as e:
        logger.exception(f"Failed to create chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/v1/chats/{chat_id}")
async def get_chat(chat_id: str):
    """
    Get a specific chat session with its messages.
    """
    try:
        if chat_id not in _chat_sessions:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        chat = _chat_sessions[chat_id]
        messages = _chat_messages.get(chat_id, [])
        
        return {
            "chat": chat,
            "messages": messages
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to get chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/api/v1/chats/{chat_id}")
@router.patch("/api/v1/chats/{chat_id}")
async def update_chat(chat_id: str, request: UpdateChatRequest):
    """
    Update a chat session (title, pinned status, etc.)
    """
    try:
        if chat_id not in _chat_sessions:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        chat = _chat_sessions[chat_id]
        
        if request.title is not None:
            chat["title"] = request.title
        if request.pinned is not None:
            chat["pinned"] = request.pinned
        if request.archived is not None:
            chat["archived"] = request.archived
        if request.tags is not None:
            chat["tags"] = request.tags
        
        chat["updated_at"] = get_current_timestamp()
        
        logger.info(f"Updated chat: {chat_id}")
        
        return {"chat": chat}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to update chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/v1/chats/{chat_id}")
async def delete_chat(chat_id: str):
    """
    Delete a chat session and all its messages.
    """
    try:
        if chat_id not in _chat_sessions:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        del _chat_sessions[chat_id]
        if chat_id in _chat_messages:
            del _chat_messages[chat_id]
        
        logger.info(f"Deleted chat: {chat_id}")
        
        return {"success": True, "deleted_id": chat_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to delete chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Message Endpoints
# ============================================================================

@router.get("/api/v1/chats/{chat_id}/messages")
async def get_chat_messages(
    chat_id: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500)
):
    """
    Get messages for a specific chat.
    """
    try:
        if chat_id not in _chat_sessions:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        messages = _chat_messages.get(chat_id, [])
        paginated = messages[skip:skip + limit]
        
        return {
            "messages": paginated,
            "total": len(messages),
            "skip": skip,
            "limit": limit
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to get messages: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/v1/chats/{chat_id}/messages")
async def add_message(chat_id: str, request: AddMessageRequest):
    """
    Add a message to a chat.
    """
    try:
        if chat_id not in _chat_sessions:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        now = get_current_timestamp()
        
        message = {
            "id": generate_message_id(),
            "role": request.role,
            "content": request.content,
            "timestamp": now
        }
        
        if chat_id not in _chat_messages:
            _chat_messages[chat_id] = []
        
        _chat_messages[chat_id].append(message)
        
        # Update chat metadata
        chat = _chat_sessions[chat_id]
        chat["updated_at"] = now
        chat["message_count"] = len(_chat_messages[chat_id])
        
        # Update title if this is the first user message
        if request.role == "user" and chat.get("title") == "New Chat":
            chat["title"] = generate_title_from_message(request.content)
        
        return {"message": message}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to add message: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Model Endpoints (Aliases for compatibility)
# ============================================================================

@router.get("/api/models")
async def list_models_alias():
    """
    List available models (alias without /v1 prefix).
    
    This endpoint exists for compatibility with frontends that
    expect /api/models instead of /api/v1/models.
    """
    now = int(datetime.utcnow().timestamp())
    
    return ModelListResponse(
        object="list",
        data=[
            ModelInfo(
                id="Vayu Maya",
                created=now,
                owned_by="Tata Communications",
                name="Vayu Maya RAG"
            ),
            ModelInfo(
                id="Vayu Maya-v1",
                created=now,
                owned_by="Tata Communications",
                name="Vayu Maya RAG v1"
            )
        ]
    )


@router.get("/api/v1/models")
async def list_models_v1():
    """
    List available models (v1 endpoint).
    """
    return await list_models_alias()


# ============================================================================
# Tags Endpoints
# ============================================================================

@router.get("/api/v1/chats/tags")
async def list_all_tags():
    """
    List all unique tags across all chats.
    """
    try:
        all_tags = set()
        for chat in _chat_sessions.values():
            all_tags.update(chat.get("tags", []))
        
        return {"tags": sorted(list(all_tags))}
    except Exception as e:
        logger.exception(f"Failed to list tags: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/v1/chats/search")
async def search_chats(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(20, ge=1, le=100)
):
    """
    Search chats by title or message content.
    """
    try:
        query_lower = q.lower()
        results = []
        
        for chat_id, chat in _chat_sessions.items():
            # Search in title
            if query_lower in chat.get("title", "").lower():
                results.append(chat)
                continue
            
            # Search in messages
            messages = _chat_messages.get(chat_id, [])
            for msg in messages:
                if query_lower in msg.get("content", "").lower():
                    results.append(chat)
                    break
        
        # Sort by relevance (title matches first) and recency
        results.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        
        return {
            "results": results[:limit],
            "total": len(results),
            "query": q
        }
    except Exception as e:
        logger.exception(f"Failed to search chats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Bulk Operations
# ============================================================================

@router.delete("/api/v1/chats/all")
async def delete_all_chats():
    """
    Delete all chat sessions (use with caution).
    """
    try:
        count = len(_chat_sessions)
        _chat_sessions.clear()
        _chat_messages.clear()
        
        logger.warning(f"Deleted all {count} chats")
        
        return {"success": True, "deleted_count": count}
    except Exception as e:
        logger.exception(f"Failed to delete all chats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/v1/chats/archive-all")
async def archive_all_chats():
    """
    Archive all chat sessions.
    """
    try:
        count = 0
        now = get_current_timestamp()
        
        for chat in _chat_sessions.values():
            if not chat.get("archived", False):
                chat["archived"] = True
                chat["updated_at"] = now
                count += 1
        
        logger.info(f"Archived {count} chats")
        
        return {"success": True, "archived_count": count}
    except Exception as e:
        logger.exception(f"Failed to archive chats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


