

from datetime import datetime, timezone
import enum
from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    Float,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Enum as SQLEnum,
    JSON,
)
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.sql import func

# ===================== Base =====================

class Base(DeclarativeBase):
    """Base class for all models"""
    pass

# ===================== Enums =====================

class UserRole(str, enum.Enum):
    """User role enumeration - must match database CHECK constraint"""
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"
    EDITOR = "editor"  

class DocumentStatus(str, enum.Enum):
    """Document processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

# ===================== User Model =====================

class User(Base):
    """
    User model with complete schema.
    
    CRITICAL FIELDS FOR AUTH:
    - username: unique identifier for login
    - email: unique email address
    - hashed_password: bcrypt hashed password (NOT plain text!)
    - role: user permission level
    - is_active: account enabled/disabled
    - updated_at: MUST exist in database (this was the bug!)
    
    Default credentials: username=admin, password=admin123
    """
    __tablename__ = "users"

    # Primary Key
    id = Column(Integer, primary_key=True, autoincrement=True)
    # Authentication & Identity (CRITICAL - indexed for login performance)
    username = Column(String(255), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    # Profile Information
    full_name = Column(String(255), nullable=True)
    avatar_url = Column(String(500), nullable=True)
    bio = Column(Text, nullable=True)
    # User Role & Permissions
    role = Column(
        SQLEnum(UserRole, name="user_role", native_enum=False, create_constraint=False),
        default=UserRole.USER,
        nullable=False,
        index=True)
    # Account Status
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    is_verified = Column(Boolean, default=False, nullable=False)
    # User Preferences
    theme = Column(String(50), nullable=True, default="light")
    language = Column(String(10), nullable=True, default="en")
    timezone = Column(String(50), nullable=True, default="UTC")
    # Notification Settings
    notifications_enabled = Column(Boolean, default=True, nullable=False)
    email_notifications = Column(Boolean, default=False, nullable=False)
    # Security & Login Tracking
    last_login = Column(DateTime(timezone=True), nullable=True)
    login_count = Column(Integer, default=0, nullable=False)
    failed_login_attempts = Column(Integer, default=0, nullable=False)
    locked_until = Column(DateTime(timezone=True), nullable=True)
    # Timestamps - CRITICAL: Both must exist!
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False)
    # Relationships
    sessions = relationship(
        "UserSession",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="dynamic"
    )
    documents = relationship(
        "Document",
        back_populates="owner",
        cascade="all, delete-orphan",
        lazy="dynamic"
    )
    rag_queries = relationship(
        "RAGQuery",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="dynamic"
    )
    knowledge_bases = relationship(
        "KnowledgeBase",
        back_populates="owner",
        cascade="all, delete-orphan",
        lazy="dynamic"
    )
    
    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', role='{self.role.value}')>"
    
    def to_dict(self, include_sensitive=False):
        """
        Convert user to dictionary.
        
        Args:
            include_sensitive: If True, include email and other sensitive data
        
        Returns:
            dict: User data (password ALWAYS excluded)
        """
        base_data = {
            "id": self.id,
            "username": self.username,
            "full_name": self.full_name,
            "avatar_url": self.avatar_url,
            "role": self.role.value if isinstance(self.role, UserRole) else self.role,
            "theme": self.theme,
            "language": self.language,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None}
        
        if include_sensitive:
            base_data.update({
                "email": self.email,
                "bio": self.bio,
                "timezone": self.timezone,
                "notifications_enabled": self.notifications_enabled,
                "email_notifications": self.email_notifications,
                "is_verified": self.is_verified,
                "last_login": self.last_login.isoformat() if self.last_login else None,
                "login_count": self.login_count,
                "updated_at": self.updated_at.isoformat() if self.updated_at else None})

        return base_data
    
    @property
    def is_locked(self) -> bool:
        """Check if account is currently locked due to failed login attempts"""
        if not self.locked_until:
            return False
        return datetime.now(timezone.utc) < self.locked_until
    
    def increment_failed_login(self, max_attempts: int = 5, lock_duration_minutes: int = 30):
        """
        Increment failed login attempts and lock account if threshold exceeded.
        Args:
            max_attempts: Maximum allowed failed attempts before locking
            lock_duration_minutes: How long to lock the account
        """
        from datetime import timedelta
        self.failed_login_attempts += 1
        if self.failed_login_attempts >= max_attempts:
            self.locked_until = datetime.now(timezone.utc) + timedelta(minutes=lock_duration_minutes)
    
    def reset_failed_login(self):
        """Reset failed login attempts and unlock account"""
        self.failed_login_attempts = 0
        self.locked_until = None
    
    def record_login(self):
        """Record successful login - updates last_login, login_count, resets failed attempts"""
        self.last_login = datetime.now(timezone.utc)
        self.login_count = (self.login_count or 0) + 1
        self.reset_failed_login()

# ===================== UserSession =====================

class UserSession(Base):
    """
    User session tracking for security and audit purposes.
    Tracks active sessions, IP addresses, and device info.
    """
    __tablename__ = "user_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    session_token = Column(String(500), unique=True, nullable=False, index=True)
    # Session metadata
    ip_address = Column(String(45), nullable=True) 
    user_agent = Column(String(500), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    last_activity = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    # Relationships
    user = relationship("User", back_populates="sessions")
    
    def __repr__(self):
        return f"<UserSession(id={self.id}, user_id={self.user_id}, active={self.is_active})>"
    
    @property
    def is_expired(self) -> bool:
        """Check if session has expired"""
        return datetime.now(timezone.utc) >= self.expires_at

# ===================== Document =====================

class Document(Base):
    """
    Document metadata and processing status.
    Represents uploaded files or scraped web content.
    """
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    owner_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    # Document metadata
    title = Column(String(500), nullable=False, index=True)
    description = Column(Text, nullable=True)
    file_path = Column(String(500), nullable=True)
    file_size = Column(Integer, nullable=True)
    file_type = Column(String(50), nullable=True)
    # Source information
    source_url = Column(String(500), nullable=True, index=True)
    source_type = Column(String(50), nullable=True)
    # Processing status
    status = Column(
        SQLEnum(DocumentStatus, name="document_status", native_enum=False, create_constraint=False),
        default=DocumentStatus.PENDING,
        nullable=False,
        index=True
    )
    processing_error = Column(Text, nullable=True)
    # Vector database tracking
    vector_ids = Column(JSON, nullable=True)
    chunk_count = Column(Integer, default=0, nullable=False)
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    processed_at = Column(DateTime(timezone=True), nullable=True)
    # Relationships
    owner = relationship("User", back_populates="documents")
    chunks = relationship(
        "DocumentChunk",
        back_populates="document",
        cascade="all, delete-orphan",
        lazy="dynamic")

    def __repr__(self):
        return f"<Document(id={self.id}, title='{self.title[:30]}...', status={self.status.value})>"

# ===================== DocumentChunk =====================

class DocumentChunk(Base):
    """
    Individual text chunks from documents.
    Each chunk has its own embedding in the vector database.
    """
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    # Chunk content
    chunk_text = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    # Vector database tracking
    vector_id = Column(String(100), unique=True, nullable=True, index=True)
    embedding_generated = Column(Boolean, default=False, nullable=False)
    # Search metadata
    relevance_score = Column(Float, nullable=True)
    keywords = Column(JSON, nullable=True)
    # Timestamp
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    # Relationships
    document = relationship("Document", back_populates="chunks")
    
    def __repr__(self):
        return f"<DocumentChunk(id={self.id}, doc_id={self.document_id}, index={self.chunk_index})>"

# ===================== RAGQuery =====================

class RAGQuery(Base):
    """
    RAG query tracking for analytics and improvement.
    Stores user queries, responses, and quality metrics.
    """
    __tablename__ = "rag_queries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    # Query details
    query_text = Column(Text, nullable=False)
    session_id = Column(String(100), nullable=True, index=True)
    # Response details
    retrieved_chunks = Column(Integer, default=0, nullable=False)
    response_text = Column(Text, nullable=True)
    response_sources = Column(JSON, nullable=True)
    # Performance metrics
    query_latency_ms = Column(Float, nullable=True)
    relevance_score = Column(Float, nullable=True)
    # User feedback
    user_rating = Column(Integer, nullable=True)
    user_feedback = Column(Text, nullable=True)
    # Timestamp
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    # Relationships
    user = relationship("User", back_populates="rag_queries")
    
    def __repr__(self):
        return f"<RAGQuery(id={self.id}, query='{self.query_text[:50]}...')>"

# ===================== KnowledgeBase =====================

class KnowledgeBase(Base):
    """
    Knowledge base collection grouping related documents.
    Allows users to organize documents into separate collections.
    """
    __tablename__ = "knowledge_bases"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    owner_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    # Settings
    is_public = Column(Boolean, default=False, nullable=False)
    auto_train = Column(Boolean, default=True, nullable=False)
    # Statistics
    total_documents = Column(Integer, default=0, nullable=False)
    total_chunks = Column(Integer, default=0, nullable=False)
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    # Relationships
    owner = relationship("User", back_populates="knowledge_bases")
    
    def __repr__(self):
        return f"<KnowledgeBase(id={self.id}, name='{self.name}', docs={self.total_documents})>"

# ===================== AuditLog =====================

class AuditLog(Base):
    """
    Audit log for security and compliance.
    Tracks all user actions for security monitoring.
    """
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    # Action details
    action = Column(String(100), nullable=False, index=True)
    resource_type = Column(String(50), nullable=False)
    resource_id = Column(String(100), nullable=True)
    # Additional context
    details = Column(JSON, nullable=True)
    ip_address = Column(String(45), nullable=True)
    status = Column(String(20), nullable=False, index=True)
    error_message = Column(Text, nullable=True)
    # Timestamp
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    # Table-level indexes for query performance
    __table_args__ = (
        Index("idx_audit_user_action", "user_id", "action"),
        Index("idx_audit_created_status", "created_at", "status"),
    )
    def __repr__(self):
        return f"<AuditLog(id={self.id}, action='{self.action}', status='{self.status}')>"

# ===================== Backwards Compatibility Alias =====================

DBUser = User