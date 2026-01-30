"""
Chat Service - Database operations for chat persistence
Equivalent to OpenWebUI's Chats table operations
"""

import uuid
import time
import logging
from typing import Optional, List, Dict, Any

from sqlalchemy import or_, and_, text
from sqlalchemy.orm import Session

from app.models.chat_models import (
    Chat, Tag, Folder, ChatFile,
    ChatModel, ChatTitleIdResponse, ChatForm, ChatImportForm,
    ChatListResponse, TagModel, FolderModel
)

log = logging.getLogger(__name__)


def _extract_title_from_messages(chat_data: Dict[str, Any], max_length: int = 50) -> Optional[str]:
    """
    Extract a title from the first user message in the chat.
    Returns None if no suitable title can be extracted.
    """
    try:
        history = chat_data.get("history", {})
        messages = history.get("messages", {})
        
        if not messages:
            return None
        
        # Find the first user message (earliest by looking at parentId chain)
        first_user_message = None
        for msg_id, msg in messages.items():
            if msg.get("role") == "user" and msg.get("parentId") is None:
                first_user_message = msg
                break
        
        # If not found by parentId, just get any user message
        if not first_user_message:
            for msg_id, msg in messages.items():
                if msg.get("role") == "user":
                    first_user_message = msg
                    break
        
        if not first_user_message:
            return None
        
        content = first_user_message.get("content", "").strip()
        if not content:
            return None
        
        # Clean up and truncate
        # Remove markdown, code blocks, etc.
        title = content.split("\n")[0].strip()  # Take first line
        
        # Truncate if too long
        if len(title) > max_length:
            title = title[:max_length - 3].rsplit(" ", 1)[0] + "..."
        
        return title if title else None
    except Exception as e:
        log.warning(f"Error extracting title from messages: {e}")
        return None


class ChatService:
    """Service class for chat CRUD operations"""

    # ==================== Create Operations ====================

    def insert_new_chat(
        self,
        user_id: str,
        form_data: ChatForm,
        db: Session
    ) -> Optional[ChatModel]:
        """Create a new chat"""
        try:
            chat_id = str(uuid.uuid4())
            now = int(time.time())
            
            # Get title: explicit > auto-generated from messages > default
            title = form_data.chat.get("title", "New Chat")
            if title == "New Chat":
                auto_title = _extract_title_from_messages(form_data.chat)
                if auto_title:
                    title = auto_title
                    # Update the chat data to reflect the generated title
                    form_data.chat["title"] = title
            
            chat = Chat(
                id=chat_id,
                user_id=user_id,
                title=title,
                chat=form_data.chat,
                folder_id=form_data.folder_id,
                created_at=now,
                updated_at=now,
                meta={},
                archived=False,
                pinned=False,
            )
            
            db.add(chat)
            db.commit()
            db.refresh(chat)
            
            return ChatModel.model_validate(chat)
        except Exception as e:
            log.exception(f"Error creating chat: {e}")
            db.rollback()
            return None

    def upsert_chat_with_id(
        self,
        chat_id: str,
        user_id: str,
        chat_data: Dict[str, Any],
        folder_id: Optional[str] = None,
        db: Session = None
    ) -> Optional[ChatModel]:
        """
        Create a chat with a specific ID (upsert behavior).
        Used when OpenWebUI frontend generates its own chat IDs.
        """
        try:
            now = int(time.time())
            
            # Get title: explicit > auto-generated from messages > default
            title = chat_data.get("title", "New Chat")
            if title == "New Chat":
                auto_title = _extract_title_from_messages(chat_data)
                if auto_title:
                    title = auto_title
                    chat_data["title"] = title
            
            chat = Chat(
                id=chat_id,  # Use the provided ID instead of generating UUID
                user_id=user_id,
                title=title,
                chat=chat_data,
                folder_id=folder_id,
                created_at=now,
                updated_at=now,
                meta={},
                archived=False,
                pinned=False,
            )
            
            db.add(chat)
            db.commit()
            db.refresh(chat)
            
            log.info(f"Created chat with custom ID: {chat_id}")
            return ChatModel.model_validate(chat)
        except Exception as e:
            log.exception(f"Error creating chat with ID {chat_id}: {e}")
            db.rollback()
            return None

    def import_chats(
        self,
        user_id: str,
        chat_import_forms: List[ChatImportForm],
        db: Session
    ) -> List[ChatModel]:
        """Import multiple chats"""
        try:
            chats = []
            now = int(time.time())
            
            for form_data in chat_import_forms:
                chat_id = str(uuid.uuid4())
                title = form_data.chat.get("title", "New Chat")
                
                chat = Chat(
                    id=chat_id,
                    user_id=user_id,
                    title=title,
                    chat=form_data.chat,
                    meta=form_data.meta or {},
                    pinned=form_data.pinned or False,
                    folder_id=form_data.folder_id,
                    created_at=form_data.created_at or now,
                    updated_at=form_data.updated_at or now,
                )
                chats.append(chat)
            
            db.add_all(chats)
            db.commit()
            
            return [ChatModel.model_validate(chat) for chat in chats]
        except Exception as e:
            log.exception(f"Error importing chats: {e}")
            db.rollback()
            return []

    # ==================== Read Operations ====================

    def get_chat_by_id(
        self,
        chat_id: str,
        db: Session
    ) -> Optional[ChatModel]:
        """Get a chat by ID"""
        try:
            chat = db.query(Chat).filter(Chat.id == chat_id).first()
            if chat:
                return ChatModel.model_validate(chat)
            return None
        except Exception as e:
            log.exception(f"Error getting chat: {e}")
            return None

    def get_chat_by_id_and_user_id(
        self,
        chat_id: str,
        user_id: str,
        db: Session
    ) -> Optional[ChatModel]:
        """Get a chat by ID and user ID"""
        try:
            chat = db.query(Chat).filter(
                Chat.id == chat_id,
                Chat.user_id == user_id
            ).first()
            if chat:
                return ChatModel.model_validate(chat)
            return None
        except Exception as e:
            log.exception(f"Error getting chat: {e}")
            return None

    def get_chat_title_id_list_by_user_id(
        self,
        user_id: str,
        include_archived: bool = False,
        include_folders: bool = False,
        include_pinned: bool = False,
        skip: Optional[int] = None,
        limit: Optional[int] = None,
        db: Session = None
    ) -> List[ChatTitleIdResponse]:
        """Get paginated list of chat titles and IDs for a user"""
        try:
            query = db.query(Chat).filter(Chat.user_id == user_id)
            
            if not include_folders:
                query = query.filter(Chat.folder_id == None)
            
            if not include_pinned:
                query = query.filter(or_(Chat.pinned == False, Chat.pinned == None))
            
            if not include_archived:
                query = query.filter(Chat.archived == False)
            
            query = query.order_by(Chat.updated_at.desc())
            
            if skip is not None:
                query = query.offset(skip)
            if limit is not None:
                query = query.limit(limit)
            
            chats = query.all()
            
            return [
                ChatTitleIdResponse(
                    id=chat.id,
                    title=chat.title,
                    updated_at=chat.updated_at,
                    created_at=chat.created_at,
                )
                for chat in chats
            ]
        except Exception as e:
            log.exception(f"Error getting chat list: {e}")
            return []

    def get_chats_by_user_id(
        self,
        user_id: str,
        skip: Optional[int] = None,
        limit: Optional[int] = None,
        db: Session = None
    ) -> ChatListResponse:
        """Get all chats for a user with pagination"""
        try:
            query = db.query(Chat).filter(Chat.user_id == user_id)
            query = query.order_by(Chat.updated_at.desc())
            
            total = query.count()
            
            if skip is not None:
                query = query.offset(skip)
            if limit is not None:
                query = query.limit(limit)
            
            chats = query.all()
            
            return ChatListResponse(
                items=[
                    ChatTitleIdResponse(
                        id=chat.id,
                        title=chat.title,
                        updated_at=chat.updated_at,
                        created_at=chat.created_at,
                    )
                    for chat in chats
                ],
                total=total,
                page=(skip // limit) + 1 if skip and limit else 1,
                page_size=limit or total,
            )
        except Exception as e:
            log.exception(f"Error getting chats: {e}")
            return ChatListResponse(items=[], total=0, page=1, page_size=0)

    def get_pinned_chats_by_user_id(
        self,
        user_id: str,
        db: Session
    ) -> List[ChatModel]:
        """Get pinned chats for a user"""
        try:
            chats = db.query(Chat).filter(
                Chat.user_id == user_id,
                Chat.pinned == True,
                Chat.archived == False
            ).order_by(Chat.updated_at.desc()).all()
            
            return [ChatModel.model_validate(chat) for chat in chats]
        except Exception as e:
            log.exception(f"Error getting pinned chats: {e}")
            return []

    def get_archived_chats_by_user_id(
        self,
        user_id: str,
        skip: int = 0,
        limit: int = 60,
        db: Session = None
    ) -> List[ChatModel]:
        """Get archived chats for a user"""
        try:
            chats = db.query(Chat).filter(
                Chat.user_id == user_id,
                Chat.archived == True
            ).order_by(Chat.updated_at.desc()).offset(skip).limit(limit).all()
            
            return [ChatModel.model_validate(chat) for chat in chats]
        except Exception as e:
            log.exception(f"Error getting archived chats: {e}")
            return []

    def search_chats(
        self,
        user_id: str,
        search_text: str,
        skip: int = 0,
        limit: int = 60,
        db: Session = None
    ) -> List[ChatTitleIdResponse]:
        """Search chats by title"""
        try:
            search_pattern = f"%{search_text.lower()}%"
            
            chats = db.query(Chat).filter(
                Chat.user_id == user_id,
                Chat.archived == False,
                Chat.title.ilike(search_pattern)
            ).order_by(Chat.updated_at.desc()).offset(skip).limit(limit).all()
            
            return [
                ChatTitleIdResponse(
                    id=chat.id,
                    title=chat.title,
                    updated_at=chat.updated_at,
                    created_at=chat.created_at,
                )
                for chat in chats
            ]
        except Exception as e:
            log.exception(f"Error searching chats: {e}")
            return []

    # ==================== Update Operations ====================

    def update_chat_by_id(
        self,
        chat_id: str,
        chat_data: Dict[str, Any],
        db: Session
    ) -> Optional[ChatModel]:
        """Update a chat by ID"""
        try:
            chat = db.query(Chat).filter(Chat.id == chat_id).first()
            if not chat:
                return None
            
            chat.chat = chat_data
            
            # Title priority:
            # 1. Explicit title in chat_data
            # 2. Auto-generate from first user message if still "New Chat"
            # 3. Keep existing title
            explicit_title = chat_data.get("title")
            if explicit_title and explicit_title != "New Chat":
                chat.title = explicit_title
            elif chat.title == "New Chat" or not chat.title:
                # Try to auto-generate from first user message
                auto_title = _extract_title_from_messages(chat_data)
                if auto_title:
                    chat.title = auto_title
                    # Also update the title in chat_data for consistency
                    chat_data["title"] = auto_title
                    chat.chat = chat_data
                else:
                    chat.title = explicit_title or chat.title or "New Chat"
            
            chat.updated_at = int(time.time())
            
            db.commit()
            db.refresh(chat)
            
            return ChatModel.model_validate(chat)
        except Exception as e:
            log.exception(f"Error updating chat: {e}")
            db.rollback()
            return None

    def toggle_chat_pinned_by_id(
        self,
        chat_id: str,
        db: Session
    ) -> Optional[ChatModel]:
        """Toggle pinned status of a chat"""
        try:
            chat = db.query(Chat).filter(Chat.id == chat_id).first()
            if not chat:
                return None
            
            chat.pinned = not chat.pinned
            chat.updated_at = int(time.time())
            
            db.commit()
            db.refresh(chat)
            
            return ChatModel.model_validate(chat)
        except Exception as e:
            log.exception(f"Error toggling pin: {e}")
            db.rollback()
            return None

    def toggle_chat_archive_by_id(
        self,
        chat_id: str,
        db: Session
    ) -> Optional[ChatModel]:
        """Toggle archived status of a chat"""
        try:
            chat = db.query(Chat).filter(Chat.id == chat_id).first()
            if not chat:
                return None
            
            chat.archived = not chat.archived
            chat.folder_id = None  # Remove from folder when archived
            chat.updated_at = int(time.time())
            
            db.commit()
            db.refresh(chat)
            
            return ChatModel.model_validate(chat)
        except Exception as e:
            log.exception(f"Error toggling archive: {e}")
            db.rollback()
            return None

    def update_chat_folder_id(
        self,
        chat_id: str,
        user_id: str,
        folder_id: Optional[str],
        db: Session
    ) -> Optional[ChatModel]:
        """Update chat's folder"""
        try:
            chat = db.query(Chat).filter(
                Chat.id == chat_id,
                Chat.user_id == user_id
            ).first()
            if not chat:
                return None
            
            chat.folder_id = folder_id
            chat.pinned = False
            chat.updated_at = int(time.time())
            
            db.commit()
            db.refresh(chat)
            
            return ChatModel.model_validate(chat)
        except Exception as e:
            log.exception(f"Error updating folder: {e}")
            db.rollback()
            return None

    def upsert_message_to_chat(
        self,
        chat_id: str,
        message_id: str,
        message: Dict[str, Any],
        db: Session
    ) -> Optional[ChatModel]:
        """Add or update a message in a chat"""
        try:
            chat = db.query(Chat).filter(Chat.id == chat_id).first()
            if not chat:
                return None
            
            chat_data = chat.chat or {}
            history = chat_data.get("history", {"messages": {}, "currentId": None})
            
            if message_id in history.get("messages", {}):
                history["messages"][message_id] = {
                    **history["messages"][message_id],
                    **message,
                }
            else:
                history["messages"][message_id] = message
            
            history["currentId"] = message_id
            chat_data["history"] = history
            
            chat.chat = chat_data
            chat.updated_at = int(time.time())
            
            db.commit()
            db.refresh(chat)
            
            return ChatModel.model_validate(chat)
        except Exception as e:
            log.exception(f"Error upserting message: {e}")
            db.rollback()
            return None

    # ==================== Delete Operations ====================

    def delete_chat_by_id(
        self,
        chat_id: str,
        db: Session
    ) -> bool:
        """Delete a chat by ID"""
        try:
            result = db.query(Chat).filter(Chat.id == chat_id).delete()
            db.commit()
            return result > 0
        except Exception as e:
            log.exception(f"Error deleting chat: {e}")
            db.rollback()
            return False

    def delete_chat_by_id_and_user_id(
        self,
        chat_id: str,
        user_id: str,
        db: Session
    ) -> bool:
        """Delete a chat by ID and user ID"""
        try:
            result = db.query(Chat).filter(
                Chat.id == chat_id,
                Chat.user_id == user_id
            ).delete()
            db.commit()
            return result > 0
        except Exception as e:
            log.exception(f"Error deleting chat: {e}")
            db.rollback()
            return False

    def delete_all_chats_by_user_id(
        self,
        user_id: str,
        db: Session
    ) -> bool:
        """Delete all chats for a user"""
        try:
            db.query(Chat).filter(Chat.user_id == user_id).delete()
            db.commit()
            return True
        except Exception as e:
            log.exception(f"Error deleting chats: {e}")
            db.rollback()
            return False

    def archive_all_chats_by_user_id(
        self,
        user_id: str,
        db: Session
    ) -> bool:
        """Archive all chats for a user"""
        try:
            db.query(Chat).filter(Chat.user_id == user_id).update({"archived": True})
            db.commit()
            return True
        except Exception as e:
            log.exception(f"Error archiving chats: {e}")
            db.rollback()
            return False

    def unarchive_all_chats_by_user_id(
        self,
        user_id: str,
        db: Session
    ) -> bool:
        """Unarchive all chats for a user"""
        try:
            db.query(Chat).filter(
                Chat.user_id == user_id,
                Chat.archived == True
            ).update({"archived": False})
            db.commit()
            return True
        except Exception as e:
            log.exception(f"Error unarchiving chats: {e}")
            db.rollback()
            return False

    # ==================== Tag Operations ====================

    def add_chat_tag(
        self,
        chat_id: str,
        user_id: str,
        tag_name: str,
        db: Session
    ) -> Optional[ChatModel]:
        """Add a tag to a chat"""
        try:
            tag_id = tag_name.replace(" ", "_").lower()
            
            # Ensure tag exists
            tag = db.query(Tag).filter(
                Tag.id == tag_id,
                Tag.user_id == user_id
            ).first()
            
            if not tag:
                tag = Tag(
                    id=tag_id,
                    name=tag_name,
                    user_id=user_id,
                    created_at=int(time.time())
                )
                db.add(tag)
            
            # Add tag to chat
            chat = db.query(Chat).filter(Chat.id == chat_id).first()
            if not chat:
                return None
            
            meta = chat.meta or {}
            tags = meta.get("tags", [])
            
            if tag_id not in tags:
                tags.append(tag_id)
                meta["tags"] = tags
                chat.meta = meta
            
            db.commit()
            db.refresh(chat)
            
            return ChatModel.model_validate(chat)
        except Exception as e:
            log.exception(f"Error adding tag: {e}")
            db.rollback()
            return None

    def remove_chat_tag(
        self,
        chat_id: str,
        user_id: str,
        tag_name: str,
        db: Session
    ) -> bool:
        """Remove a tag from a chat"""
        try:
            tag_id = tag_name.replace(" ", "_").lower()
            
            chat = db.query(Chat).filter(Chat.id == chat_id).first()
            if not chat:
                return False
            
            meta = chat.meta or {}
            tags = meta.get("tags", [])
            
            if tag_id in tags:
                tags.remove(tag_id)
                meta["tags"] = tags
                chat.meta = meta
            
            db.commit()
            return True
        except Exception as e:
            log.exception(f"Error removing tag: {e}")
            db.rollback()
            return False

    def get_user_tags(
        self,
        user_id: str,
        db: Session
    ) -> List[TagModel]:
        """Get all tags for a user"""
        try:
            tags = db.query(Tag).filter(Tag.user_id == user_id).all()
            return [TagModel.model_validate(tag) for tag in tags]
        except Exception as e:
            log.exception(f"Error getting tags: {e}")
            return []


# Singleton instance
chat_service = ChatService()
