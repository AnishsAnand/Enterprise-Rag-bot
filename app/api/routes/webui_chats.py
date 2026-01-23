"""
WebUI-compatible Chat API Routes
Provides OpenWebUI-style endpoints for chat persistence

Routes:
- GET  /api/v1/chats/?page=1 - List chats (paginated)
- POST /api/v1/chats/new - Create new chat
- GET  /api/v1/chats/{id} - Get chat by ID
- POST /api/v1/chats/{id} - Update chat
- DELETE /api/v1/chats/{id} - Delete chat
- POST /api/v1/chats/{id}/pin - Toggle pin
- POST /api/v1/chats/{id}/archive - Toggle archive
- GET  /api/v1/chats/search - Search chats
- GET  /api/v1/chats/pinned - Get pinned chats
- GET  /api/v1/chats/archived - Get archived chats
- GET  /api/v1/chats/all/tags - Get all user tags
"""

import logging
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.core.database import get_db
from app.models.chat_models import (
    ChatForm, ChatResponse, ChatTitleIdResponse, ChatModel,
    ChatListResponse, TagModel, MessageForm
)
from app.services.chat_service import chat_service

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/chats", tags=["chats"])


# ==================== Helper: Get Current User ====================

def get_current_user_id(request: Request) -> str:
    """
    Extract user ID from request.
    For now, uses a header or defaults to 'default_user'.
    
    In production, this should validate JWT tokens.
    """
    # Try to get from header (OpenWebUI style)
    user_id = request.headers.get("X-User-Id")
    if user_id:
        return user_id
    
    # Try authorization header
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        # In production: decode JWT and extract user_id
        # For now, use a placeholder
        pass
    
    # Default user for development
    return "default_user"


# ==================== Chat List Endpoints ====================

@router.get("/", response_model=List[ChatTitleIdResponse])
@router.get("/list", response_model=List[ChatTitleIdResponse])
async def get_chat_list(
    request: Request,
    page: Optional[int] = None,
    include_pinned: Optional[bool] = False,
    include_folders: Optional[bool] = False,
    db: Session = Depends(get_db)
):
    """
    Get paginated list of chats for current user.
    
    - **page**: Page number (1-indexed), 60 items per page
    - **include_pinned**: Include pinned chats in results
    - **include_folders**: Include chats in folders
    """
    user_id = get_current_user_id(request)
    
    try:
        if page is not None:
            limit = 60
            skip = (page - 1) * limit
            
            return chat_service.get_chat_title_id_list_by_user_id(
                user_id=user_id,
                include_folders=include_folders,
                include_pinned=include_pinned,
                skip=skip,
                limit=limit,
                db=db
            )
        else:
            return chat_service.get_chat_title_id_list_by_user_id(
                user_id=user_id,
                include_folders=include_folders,
                include_pinned=include_pinned,
                db=db
            )
    except Exception as e:
        log.exception(f"Error getting chat list: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to retrieve chats"
        )


@router.get("/search", response_model=List[ChatTitleIdResponse])
async def search_chats(
    request: Request,
    text: str,
    page: Optional[int] = 1,
    db: Session = Depends(get_db)
):
    """Search chats by title"""
    user_id = get_current_user_id(request)
    
    limit = 60
    skip = (page - 1) * limit
    
    return chat_service.search_chats(
        user_id=user_id,
        search_text=text,
        skip=skip,
        limit=limit,
        db=db
    )


@router.get("/pinned", response_model=List[ChatTitleIdResponse])
async def get_pinned_chats(
    request: Request,
    db: Session = Depends(get_db)
):
    """Get all pinned chats for current user"""
    user_id = get_current_user_id(request)
    
    chats = chat_service.get_pinned_chats_by_user_id(user_id, db)
    return [
        ChatTitleIdResponse(
            id=chat.id,
            title=chat.title,
            updated_at=chat.updated_at,
            created_at=chat.created_at,
        )
        for chat in chats
    ]


@router.get("/archived", response_model=List[ChatTitleIdResponse])
async def get_archived_chats(
    request: Request,
    page: Optional[int] = 1,
    db: Session = Depends(get_db)
):
    """Get archived chats for current user"""
    user_id = get_current_user_id(request)
    
    limit = 60
    skip = (page - 1) * limit
    
    chats = chat_service.get_archived_chats_by_user_id(user_id, skip, limit, db)
    return [
        ChatTitleIdResponse(
            id=chat.id,
            title=chat.title,
            updated_at=chat.updated_at,
            created_at=chat.created_at,
        )
        for chat in chats
    ]


@router.get("/all/tags", response_model=List[TagModel])
async def get_all_tags(
    request: Request,
    db: Session = Depends(get_db)
):
    """Get all tags for current user"""
    user_id = get_current_user_id(request)
    return chat_service.get_user_tags(user_id, db)


# ==================== Single Chat Endpoints ====================

@router.post("/new", response_model=Optional[ChatResponse])
async def create_new_chat(
    request: Request,
    form_data: ChatForm,
    db: Session = Depends(get_db)
):
    """Create a new chat"""
    user_id = get_current_user_id(request)
    
    try:
        chat = chat_service.insert_new_chat(user_id, form_data, db)
        if chat:
            return ChatResponse(**chat.model_dump())
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to create chat"
        )
    except Exception as e:
        log.exception(f"Error creating chat: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to create chat"
        )


@router.get("/{chat_id}", response_model=Optional[ChatResponse])
async def get_chat_by_id(
    chat_id: str,
    request: Request,
    db: Session = Depends(get_db)
):
    """Get a specific chat by ID"""
    user_id = get_current_user_id(request)
    
    chat = chat_service.get_chat_by_id_and_user_id(chat_id, user_id, db)
    if chat:
        return ChatResponse(**chat.model_dump())
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Chat not found"
    )


@router.post("/{chat_id}", response_model=Optional[ChatResponse])
async def update_chat_by_id(
    chat_id: str,
    form_data: ChatForm,
    request: Request,
    db: Session = Depends(get_db)
):
    """Update a chat by ID"""
    user_id = get_current_user_id(request)
    
    # Verify ownership
    existing = chat_service.get_chat_by_id_and_user_id(chat_id, user_id, db)
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat not found"
        )
    
    # Merge existing chat with new data
    updated_chat_data = {**existing.chat, **form_data.chat}
    chat = chat_service.update_chat_by_id(chat_id, updated_chat_data, db)
    
    if chat:
        return ChatResponse(**chat.model_dump())
    
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Failed to update chat"
    )


@router.delete("/{chat_id}", response_model=bool)
async def delete_chat_by_id(
    chat_id: str,
    request: Request,
    db: Session = Depends(get_db)
):
    """Delete a chat by ID"""
    user_id = get_current_user_id(request)
    
    result = chat_service.delete_chat_by_id_and_user_id(chat_id, user_id, db)
    return result


@router.delete("/", response_model=bool)
async def delete_all_chats(
    request: Request,
    db: Session = Depends(get_db)
):
    """Delete all chats for current user"""
    user_id = get_current_user_id(request)
    return chat_service.delete_all_chats_by_user_id(user_id, db)


# ==================== Chat Actions ====================

@router.post("/{chat_id}/pin", response_model=Optional[ChatResponse])
async def toggle_pin_chat(
    chat_id: str,
    request: Request,
    db: Session = Depends(get_db)
):
    """Toggle pin status of a chat"""
    user_id = get_current_user_id(request)
    
    # Verify ownership
    existing = chat_service.get_chat_by_id_and_user_id(chat_id, user_id, db)
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat not found"
        )
    
    chat = chat_service.toggle_chat_pinned_by_id(chat_id, db)
    if chat:
        return ChatResponse(**chat.model_dump())
    
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Failed to toggle pin"
    )


@router.post("/{chat_id}/archive", response_model=Optional[ChatResponse])
async def toggle_archive_chat(
    chat_id: str,
    request: Request,
    db: Session = Depends(get_db)
):
    """Toggle archive status of a chat"""
    user_id = get_current_user_id(request)
    
    # Verify ownership
    existing = chat_service.get_chat_by_id_and_user_id(chat_id, user_id, db)
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat not found"
        )
    
    chat = chat_service.toggle_chat_archive_by_id(chat_id, db)
    if chat:
        return ChatResponse(**chat.model_dump())
    
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Failed to toggle archive"
    )


@router.post("/archive/all", response_model=bool)
async def archive_all_chats(
    request: Request,
    db: Session = Depends(get_db)
):
    """Archive all chats for current user"""
    user_id = get_current_user_id(request)
    return chat_service.archive_all_chats_by_user_id(user_id, db)


# ==================== Message Operations ====================

@router.post("/{chat_id}/messages/{message_id}", response_model=Optional[ChatResponse])
async def update_chat_message(
    chat_id: str,
    message_id: str,
    form_data: MessageForm,
    request: Request,
    db: Session = Depends(get_db)
):
    """Update a specific message in a chat"""
    user_id = get_current_user_id(request)
    
    # Verify ownership
    existing = chat_service.get_chat_by_id_and_user_id(chat_id, user_id, db)
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat not found"
        )
    
    chat = chat_service.upsert_message_to_chat(
        chat_id,
        message_id,
        {"content": form_data.content},
        db
    )
    
    if chat:
        return ChatResponse(**chat.model_dump())
    
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Failed to update message"
    )


# ==================== Tag Operations ====================

class TagForm(BaseModel):
    name: str


@router.get("/{chat_id}/tags", response_model=List[TagModel])
async def get_chat_tags(
    chat_id: str,
    request: Request,
    db: Session = Depends(get_db)
):
    """Get tags for a specific chat"""
    user_id = get_current_user_id(request)
    
    chat = chat_service.get_chat_by_id_and_user_id(chat_id, user_id, db)
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat not found"
        )
    
    tag_ids = chat.meta.get("tags", [])
    all_tags = chat_service.get_user_tags(user_id, db)
    
    return [tag for tag in all_tags if tag.id in tag_ids]


@router.post("/{chat_id}/tags", response_model=List[TagModel])
async def add_chat_tag(
    chat_id: str,
    form_data: TagForm,
    request: Request,
    db: Session = Depends(get_db)
):
    """Add a tag to a chat"""
    user_id = get_current_user_id(request)
    
    if form_data.name.lower() == "none":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tag name cannot be 'None'"
        )
    
    chat = chat_service.add_chat_tag(chat_id, user_id, form_data.name, db)
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat not found"
        )
    
    tag_ids = chat.meta.get("tags", [])
    all_tags = chat_service.get_user_tags(user_id, db)
    
    return [tag for tag in all_tags if tag.id in tag_ids]


@router.delete("/{chat_id}/tags", response_model=List[TagModel])
async def remove_chat_tag(
    chat_id: str,
    form_data: TagForm,
    request: Request,
    db: Session = Depends(get_db)
):
    """Remove a tag from a chat"""
    user_id = get_current_user_id(request)
    
    chat_service.remove_chat_tag(chat_id, user_id, form_data.name, db)
    
    chat = chat_service.get_chat_by_id_and_user_id(chat_id, user_id, db)
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat not found"
        )
    
    tag_ids = chat.meta.get("tags", [])
    all_tags = chat_service.get_user_tags(user_id, db)
    
    return [tag for tag in all_tags if tag.id in tag_ids]


# ==================== Folder Operations ====================

class FolderForm(BaseModel):
    folder_id: Optional[str] = None


@router.post("/{chat_id}/folder", response_model=Optional[ChatResponse])
async def update_chat_folder(
    chat_id: str,
    form_data: FolderForm,
    request: Request,
    db: Session = Depends(get_db)
):
    """Move a chat to a folder"""
    user_id = get_current_user_id(request)
    
    chat = chat_service.update_chat_folder_id(chat_id, user_id, form_data.folder_id, db)
    if chat:
        return ChatResponse(**chat.model_dump())
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Chat not found"
    )
