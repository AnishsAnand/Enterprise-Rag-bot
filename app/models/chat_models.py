"""
Chat Models for Enterprise RAG Bot
Equivalent to OpenWebUI's chat persistence layer

Database Schema compatible with OpenWebUI-style chat storage
"""

import uuid
import time
from typing import Optional, List, Dict, Any

from sqlalchemy import (
    Column,
    String,
    Text,
    BigInteger,
    Boolean,
    JSON,
    Index,
    ForeignKey,
)
from pydantic import BaseModel, ConfigDict
from app.models.database_models import Base


# ===================== SQLAlchemy Models =====================

class Chat(Base):
    """
    Chat model - stores conversation sessions.
    Compatible with OpenWebUI's chat schema.
    
    The `chat` JSON field stores:
    - history: { messages: {...}, currentId: str }
    - title: str
    - models: list[str]
    - etc.
    """
    __tablename__ = "chats"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(255), nullable=False, index=True)
    title = Column(Text, nullable=False, default="New Chat")
    
    # JSON field containing messages and conversation history
    # Structure: { "history": { "messages": {...}, "currentId": "..." }, "title": "...", "models": [...] }
    chat = Column(JSON, nullable=False, default=dict)
    
    # Timestamps as epoch integers (OpenWebUI style)
    created_at = Column(BigInteger, nullable=False, default=lambda: int(time.time()))
    updated_at = Column(BigInteger, nullable=False, default=lambda: int(time.time()))
    
    # Sharing
    share_id = Column(Text, unique=True, nullable=True)
    
    # Organization
    archived = Column(Boolean, default=False, nullable=False)
    pinned = Column(Boolean, default=False, nullable=True)
    
    # Metadata (tags, etc.)
    meta = Column(JSON, default=dict, nullable=False)
    folder_id = Column(Text, nullable=True)

    __table_args__ = (
        Index("idx_chat_user_id", "user_id"),
        Index("idx_chat_folder_id", "folder_id"),
        Index("idx_chat_user_pinned", "user_id", "pinned"),
        Index("idx_chat_user_archived", "user_id", "archived"),
        Index("idx_chat_updated_user", "updated_at", "user_id"),
    )

    def __repr__(self):
        return f"<Chat(id={self.id}, title='{self.title[:30]}...', user_id={self.user_id})>"


class ChatFile(Base):
    """
    Tracks files attached to chat messages.
    """
    __tablename__ = "chat_files"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(255), nullable=False)
    chat_id = Column(String(36), ForeignKey("chats.id", ondelete="CASCADE"), nullable=False, index=True)
    message_id = Column(String(255), nullable=True)
    file_id = Column(String(255), nullable=False)
    
    created_at = Column(BigInteger, nullable=False, default=lambda: int(time.time()))
    updated_at = Column(BigInteger, nullable=False, default=lambda: int(time.time()))

    __table_args__ = (
        Index("idx_chat_file_chat_id", "chat_id"),
    )


class Tag(Base):
    """
    User tags for organizing chats.
    """
    __tablename__ = "tags"

    id = Column(String(255), primary_key=True)  # tag_name normalized
    name = Column(String(255), nullable=False)
    user_id = Column(String(255), nullable=False, index=True)
    
    created_at = Column(BigInteger, nullable=False, default=lambda: int(time.time()))

    __table_args__ = (
        Index("idx_tag_user_id", "user_id"),
    )


class Folder(Base):
    """
    Folders for organizing chats.
    """
    __tablename__ = "folders"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    user_id = Column(String(255), nullable=False, index=True)
    parent_id = Column(String(36), nullable=True)
    
    created_at = Column(BigInteger, nullable=False, default=lambda: int(time.time()))
    updated_at = Column(BigInteger, nullable=False, default=lambda: int(time.time()))

    __table_args__ = (
        Index("idx_folder_user_id", "user_id"),
        Index("idx_folder_parent_id", "parent_id"),
    )


# ===================== Pydantic Models (API Schemas) =====================

class ChatForm(BaseModel):
    """Form for creating/updating a chat"""
    chat: Dict[str, Any]
    folder_id: Optional[str] = None


class ChatImportForm(ChatForm):
    """Form for importing chats"""
    meta: Optional[Dict[str, Any]] = {}
    pinned: Optional[bool] = False
    created_at: Optional[int] = None
    updated_at: Optional[int] = None


class ChatTitleIdResponse(BaseModel):
    """Lightweight chat response with just id and title"""
    id: str
    title: str
    updated_at: int
    created_at: int

    model_config = ConfigDict(from_attributes=True)


class ChatResponse(BaseModel):
    """Full chat response"""
    id: str
    user_id: str
    title: str
    chat: Dict[str, Any]
    created_at: int
    updated_at: int
    share_id: Optional[str] = None
    archived: bool = False
    pinned: Optional[bool] = False
    meta: Dict[str, Any] = {}
    folder_id: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class ChatModel(BaseModel):
    """Internal chat model for data transfer"""
    id: str
    user_id: str
    title: str
    chat: Dict[str, Any]
    created_at: int
    updated_at: int
    share_id: Optional[str] = None
    archived: bool = False
    pinned: Optional[bool] = False
    meta: Dict[str, Any] = {}
    folder_id: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class TagModel(BaseModel):
    """Tag response model"""
    id: str
    name: str
    user_id: str
    created_at: int

    model_config = ConfigDict(from_attributes=True)


class FolderModel(BaseModel):
    """Folder response model"""
    id: str
    name: str
    user_id: str
    parent_id: Optional[str] = None
    created_at: int
    updated_at: int

    model_config = ConfigDict(from_attributes=True)


class MessageForm(BaseModel):
    """Form for updating a message"""
    content: str


class ChatListResponse(BaseModel):
    """Paginated chat list response"""
    items: List[ChatTitleIdResponse]
    total: int
    page: int
    page_size: int
