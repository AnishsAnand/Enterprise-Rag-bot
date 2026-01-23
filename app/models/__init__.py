from app.models.database_models import *
from app.models.chat_models import Chat, ChatFile, Tag, Folder

# Explicit export for safety
__all__ = [
    # Base
    "Base",
    # User models
    "User",
    "UserSession",
    "UserRole",
    # Document models
    "Document",
    "DocumentChunk",
    "DocumentStatus",
    # RAG models
    "RAGQuery",
    "KnowledgeBase",
    # Chat models (OpenWebUI-compatible)
    "Chat",
    "ChatFile",
    "Tag",
    "Folder",
    # Audit
    "AuditLog",
]
