# app/core/config.py - PRODUCTION SETTINGS (Pydantic v2 Compatible)
import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict
class Settings(BaseSettings):
    """
    PRODUCTION-GRADE Configuration Settings
    All fields properly typed for Pydantic v2 compatibility
    """
    
    # =========================
    # AI Service Configuration
    # =========================
    OPENROUTER_API_KEY: Optional[str] = Field(
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY"),
        description="OpenRouter API key for LLM access"
    )
    VOYAGE_API_KEY: Optional[str] = Field(
        default_factory=lambda: os.getenv("VOYAGE_API_KEY"),
        description="Voyage AI API key for embeddings"
    )
    GROK_API_KEY: Optional[str] = Field(
        default_factory=lambda: os.getenv("GROK_API_KEY"),
        description="Grok API key"
    )
    OLLAMA_BASE_URL: str = Field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        description="Ollama service base URL"
    )
    
    # =========================
    # Database Configuration
    # =========================
    DATABASE_URL: str = Field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL",
            "postgresql://ragbot:ragbot_secret_2024@localhost:5435/ragbot_sessions"
        ),
        description="PostgreSQL connection URL for sessions"
    )
    
    # =========================
    # PostgreSQL Configuration (Vector Store)
    # =========================
    POSTGRES_HOST: str = Field(
        default_factory=lambda: os.getenv("POSTGRES_HOST", "localhost"),
        description="PostgreSQL host"
    )
    POSTGRES_PORT: int = Field(
        default_factory=lambda: int(os.getenv("POSTGRES_PORT", "5432")),
        description="PostgreSQL port"
    )
    POSTGRES_USER: str = Field(
        default_factory=lambda: os.getenv("POSTGRES_USER", "ragbot"),
        description="PostgreSQL user"
    )
    POSTGRES_PASSWORD: str = Field(
        default_factory=lambda: os.getenv("POSTGRES_PASSWORD", "ragbot_secret_2024"),
        description="PostgreSQL password"
    )
    POSTGRES_DB: str = Field(
        default_factory=lambda: os.getenv("POSTGRES_DB", "enterprise_rag"),
        description="PostgreSQL database name"
    )
    POSTGRES_TABLE: str = Field(
        default_factory=lambda: os.getenv("POSTGRES_TABLE", "enterprise_rag"),
        description="PostgreSQL table name for documents"
    )
    
    # =========================
    # Storage Configuration
    # =========================
    CHROMA_PERSIST_DIRECTORY: str = Field(
        default_factory=lambda: os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db"),
        description="ChromaDB persistence directory"
    )
    UPLOAD_DIRECTORY: str = Field(
        default_factory=lambda: os.getenv("UPLOAD_DIRECTORY", "./uploads"),
        description="File upload directory"
    )
    OUTPUT_DIRECTORY: str = Field(
        default_factory=lambda: os.getenv("OUTPUT_DIRECTORY", "./outputs"),
        description="Output files directory"
    )
    
    # =========================
    # OCR Configuration
    # =========================
    DOTS_OCR_API_KEY: str = Field(
        default_factory=lambda: os.getenv("DOTS_OCR_API_KEY", ""),
        description="DOTS OCR API key"
    )
    OCR_ENABLED: bool = Field(
        default_factory=lambda: os.getenv("OCR_ENABLED", "true").lower() == "true",
        description="Enable OCR processing"
    )
    OCR_MAX_IMAGES_PER_PAGE: int = Field(
        default_factory=lambda: int(os.getenv("OCR_MAX_IMAGES_PER_PAGE", "5")),
        description="Max images to OCR per page"
    )
    OCR_TIMEOUT_SECONDS: int = Field(
        default_factory=lambda: int(os.getenv("OCR_TIMEOUT_SECONDS", "30")),
        description="OCR operation timeout"
    )
    
    # =========================
    # Scraper Configuration
    # =========================
    SCRAPER_MAX_CONCURRENT: int = Field(
        default_factory=lambda: int(os.getenv("SCRAPER_MAX_CONCURRENT", "3")),
        description="Max concurrent scraping operations"
    )
    SCRAPER_DELAY_SECONDS: float = Field(
        default_factory=lambda: float(os.getenv("SCRAPER_DELAY_SECONDS", "1.0")),
        description="Delay between scraping requests"
    )
    
    # =========================
    # Session Configuration
    # =========================
    SESSION_PERSISTENCE_ENABLED: bool = Field(
        default_factory=lambda: os.getenv("SESSION_PERSISTENCE_ENABLED", "true").lower() == "true",
        description="Enable session persistence"
    )
    SESSION_TTL_HOURS: int = Field(
        default_factory=lambda: int(os.getenv("SESSION_TTL_HOURS", "24")),
        description="Session time-to-live in hours"
    )
    
    # =========================
    # Embedding Configuration
    # =========================
    EMBEDDING_DIMENSION: int = Field(
        default_factory=lambda: int(os.getenv("EMBEDDING_DIMENSION", "4096")),
        description="Embedding vector dimension"
    )
    EMBEDDING_MODEL: str = Field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "openai"),
        description="Embedding model to use"
    )
    
    # =========================
    # RAG Configuration
    # =========================
    MIN_RELEVANCE_THRESHOLD: float = Field(
    default=0.08,
    description="Minimum relevance score threshold"
)
    MAX_CHUNKS_RETURN: int = Field(
        default_factory=lambda: int(os.getenv("MAX_CHUNKS_RETURN", "12")),
        description="Maximum chunks to return"
    )
    ENABLE_QUERY_EXPANSION: bool = Field(
        default_factory=lambda: os.getenv("ENABLE_QUERY_EXPANSION", "true").lower() == "true",
        description="Enable query expansion"
    )
    ENABLE_SEMANTIC_RERANK: bool = Field(
        default_factory=lambda: os.getenv("ENABLE_SEMANTIC_RERANK", "true").lower() == "true",
        description="Enable semantic reranking"
    )
    
    # =========================
    # Security Configuration
    # =========================
    SECRET_KEY: str = Field(
        default_factory=lambda: os.getenv("SECRET_KEY", "your-secret-key-change-in-production"),
        description="JWT secret key"
    )
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default_factory=lambda: int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60")),
        description="Access token expiration time"
    )
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(
        default_factory=lambda: int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7")),
        description="Refresh token expiration time"
    )
    
    # =========================
    # API Configuration
    # =========================
    ALLOWED_ORIGINS: str = Field(
        default_factory=lambda: os.getenv(
            "ALLOWED_ORIGINS",
            "http://localhost:3000,http://localhost:4200,http://localhost:4201"
        ),
        description="CORS allowed origins"
    )
    API_HOST: str = Field(
        default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"),
        description="API server host"
    )
    API_PORT: int = Field(
        default_factory=lambda: int(os.getenv("API_PORT", "8000")),
        description="API server port"
    )
    
    # =========================
    # Python Path
    # =========================
    PYTHONPATH: str = Field(
        default_factory=lambda: os.getenv("PYTHONPATH", "./app"),
        description="Python path"
    )
    
    model_config = SettingsConfigDict(
    env_file=".env",
    env_file_encoding="utf-8",
    case_sensitive=True,
    extra="allow"
)

# Global settings instance
settings = Settings()