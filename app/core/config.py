# app/core/config.py
# ✅ PRODUCTION-READY: Universal Configuration with Cross-Platform Compatibility
# ✅ OPTIMIZED: Adaptive settings based on runtime environment
# ✅ PYDANTIC V2 COMPATIBLE: Proper typing and validation

import os
import platform
import tempfile
from typing import Optional, List
from pathlib import Path
from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Production-grade configuration with universal platform compatibility.
    
    NEW FEATURES:
    - Auto-detection of runtime environment (Docker, K8s, bare metal, cloud)
    - Adaptive paths and resource limits
    - Enhanced validation and defaults
    - Comprehensive configuration for all services
    """
    
    # =========================
    # ENVIRONMENT DETECTION (computed at runtime)
    # =========================
    
    @computed_field
    @property
    def platform_system(self) -> str:
        """Detect operating system."""
        return platform.system()
    
    @computed_field
    @property
    def is_docker(self) -> bool:
        """Detect if running in Docker."""
        return os.path.exists('/.dockerenv')
    
    @computed_field
    @property
    def is_kubernetes(self) -> bool:
        """Detect if running in Kubernetes."""
        return os.path.exists('/var/run/secrets/kubernetes.io')
    
    @computed_field
    @property
    def is_cloud(self) -> bool:
        """Detect if running in cloud environment."""
        return bool(
            os.getenv('AWS_EXECUTION_ENV') or 
            os.getenv('WEBSITE_INSTANCE_ID') or 
            os.getenv('K_SERVICE')
        )
    
    @computed_field
    @property
    def environment_type(self) -> str:
        """Determine environment type."""
        if self.is_kubernetes:
            return "kubernetes"
        elif self.is_docker:
            return "docker"
        elif self.is_cloud:
            return "cloud"
        else:
            return "bare_metal"
    
    # =========================
    # ADAPTIVE PATHS
    # =========================
    
    @computed_field
    @property
    def data_dir(self) -> Path:
        """Adaptive data directory based on environment."""
        if self.is_docker or self.is_kubernetes:
            return Path("/app/data")
        elif self.platform_system == "Windows":
            return Path(os.getenv("APPDATA", "C:\\ProgramData")) / "vayu_maya"
        else:
            return Path.home() / ".vayu_maya"
    
    @computed_field
    @property
    def temp_dir(self) -> Path:
        """Adaptive temporary directory."""
        if self.is_docker or self.is_kubernetes:
            return Path("/tmp/vayu_maya")
        elif self.platform_system == "Windows":
            return Path(os.getenv("TEMP", "C:\\Temp")) / "vayu_maya"
        else:
            return Path(tempfile.gettempdir()) / "vayu_maya"
    
    # =========================
    # AI SERVICE CONFIGURATION
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
    # DATABASE CONFIGURATION (Sessions)
    # =========================
    DATABASE_URL: str = Field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL",
            "postgresql://ragbot:ragbot_secret_2024@localhost:5435/ragbot_sessions"
        ),
        description="PostgreSQL connection URL for sessions"
    )
    
    # =========================
    # POSTGRESQL CONFIGURATION (Vector Store)
    # =========================
    
    @computed_field
    @property
    def postgres_host_adaptive(self) -> str:
        """Adaptive PostgreSQL host based on environment."""
        explicit = os.getenv("POSTGRES_HOST")
        if explicit:
            return explicit
        elif self.is_kubernetes or self.is_docker:
            return "postgres"  # Service name in containers
        else:
            return "localhost"  # Bare metal default
    
    POSTGRES_HOST: Optional[str] = Field(
        default=None,
        description="PostgreSQL host (auto-detected if not set)"
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
        default_factory=lambda: os.getenv("POSTGRES_DB", "ragbot_db"),
        description="PostgreSQL database name"
    )
    POSTGRES_TABLE: str = Field(
        default_factory=lambda: os.getenv("POSTGRES_TABLE", "enterprise_rag"),
        description="PostgreSQL table name for documents"
    )
    
    # PostgreSQL Pool Configuration
    POSTGRES_POOL_MIN: int = Field(
        default_factory=lambda: int(os.getenv("POSTGRES_POOL_MIN", "2")),
        description="Minimum pool size"
    )
    POSTGRES_POOL_MAX: int = Field(
        default_factory=lambda: int(os.getenv("POSTGRES_POOL_MAX", "10")),
        description="Maximum pool size"
    )
    
    # PostgreSQL Search Configuration
    POSTGRES_MIN_RELEVANCE: float = Field(
        default_factory=lambda: float(os.getenv("POSTGRES_MIN_RELEVANCE", "0.08")),
        description="Minimum relevance threshold for search results"
    )
    POSTGRES_MAX_INITIAL_RESULTS: int = Field(
        default_factory=lambda: int(os.getenv("POSTGRES_MAX_INITIAL_RESULTS", "200")),
        description="Maximum initial results before filtering"
    )
    POSTGRES_RERANK_TOP_K: int = Field(
        default_factory=lambda: int(os.getenv("POSTGRES_RERANK_TOP_K", "100")),
        description="Top K results to rerank"
    )
    
    
    # =========================
    # STORAGE CONFIGURATION
    # =========================
    
    UPLOAD_DIRECTORY: str = Field(
        default_factory=lambda: os.getenv("UPLOAD_DIRECTORY", "./uploads"),
        description="File upload directory"
    )
    OUTPUT_DIRECTORY: str = Field(
        default_factory=lambda: os.getenv("OUTPUT_DIRECTORY", "./outputs"),
        description="Output files directory"
    )
    
    # =========================
    # OCR CONFIGURATION
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
    # SCRAPER CONFIGURATION 
    # =========================
    
    @computed_field
    @property
    def scraper_max_concurrent_adaptive(self) -> int:
        """Adaptive concurrent scraping based on environment."""
        explicit = os.getenv("SCRAPER_MAX_CONCURRENT")
        if explicit:
            return int(explicit)
        elif self.is_kubernetes or self.is_docker:
            return 3  # Conservative for containers
        else:
            return 10  # Aggressive for bare metal
    
    SCRAPER_MAX_CONCURRENT: Optional[int] = Field(
        default=None,
        description="Max concurrent scraping operations (auto-detected if not set)"
    )
    SCRAPER_DELAY_SECONDS: float = Field(
        default_factory=lambda: float(os.getenv("SCRAPER_DELAY_SECONDS", "0.5")),
        description="Delay between scraping requests"
    )
    SCRAPER_TIMEOUT_SECONDS: int = Field(
        default_factory=lambda: int(os.getenv("SCRAPER_TIMEOUT_SECONDS", "30")),
        description="Scraping request timeout"
    )
    SCRAPER_MAX_RETRIES: int = Field(
        default_factory=lambda: int(os.getenv("SCRAPER_MAX_RETRIES", "3")),
        description="Maximum retry attempts for failed requests"
    )
    
    # =========================
    # SESSION CONFIGURATION
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
    # EMBEDDING CONFIGURATION
    # =========================
    EMBEDDING_DIMENSION: int = Field(
        default_factory=lambda: int(os.getenv("EMBEDDING_DIMENSION", "4096")),
        description="Embedding vector dimension"
    )
    EMBEDDING_MODEL: str = Field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "openai"),
        description="Embedding model to use (openai, voyage, ollama)"
    )
    EMBEDDING_BATCH_SIZE: int = Field(
        default_factory=lambda: int(os.getenv("EMBEDDING_BATCH_SIZE", "100")),
        description="Batch size for embedding generation"
    )
    
    # =========================
    # RAG CONFIGURATION 
    # =========================
    MIN_RELEVANCE_THRESHOLD: float = Field(
        default=0.08,
        description="Minimum relevance score threshold"
    )
    MAX_CHUNKS_RETURN: int = Field(
        default_factory=lambda: int(os.getenv("MAX_CHUNKS_RETURN", "12")),
        description="Maximum chunks to return in RAG responses"
    )
    ENABLE_QUERY_EXPANSION: bool = Field(
        default_factory=lambda: os.getenv("ENABLE_QUERY_EXPANSION", "true").lower() == "true",
        description="Enable query expansion for better recall"
    )
    ENABLE_SEMANTIC_RERANK: bool = Field(
        default_factory=lambda: os.getenv("ENABLE_SEMANTIC_RERANK", "true").lower() == "true",
        description="Enable semantic reranking for better relevance"
    )
    ENABLE_HYBRID_SEARCH: bool = Field(
        default_factory=lambda: os.getenv("ENABLE_HYBRID_SEARCH", "true").lower() == "true",
        description="Enable hybrid vector + full-text search"
    )
    HYBRID_VECTOR_WEIGHT: float = Field(
        default_factory=lambda: float(os.getenv("HYBRID_VECTOR_WEIGHT", "0.7")),
        description="Weight for vector similarity in hybrid search"
    )
    HYBRID_FTS_WEIGHT: float = Field(
        default_factory=lambda: float(os.getenv("HYBRID_FTS_WEIGHT", "0.3")),
        description="Weight for full-text search in hybrid search"
    )
    
    # =========================
    # CACHE CONFIGURATION
    # =========================
    QUERY_CACHE_ENABLED: bool = Field(
        default_factory=lambda: os.getenv("QUERY_CACHE_ENABLED", "true").lower() == "true",
        description="Enable query result caching"
    )
    QUERY_CACHE_TTL: int = Field(
        default_factory=lambda: int(os.getenv("QUERY_CACHE_TTL", "3600")),
        description="Query cache TTL in seconds"
    )
    QUERY_CACHE_SIZE: int = Field(
        default_factory=lambda: int(os.getenv("QUERY_CACHE_SIZE", "100")),
        description="Maximum number of cached queries"
    )
    
    # =========================
    # SECURITY CONFIGURATION
    # =========================
    SECRET_KEY: str = Field(
        default_factory=lambda: os.getenv("SECRET_KEY", "your-secret-key-change-in-production-" + os.urandom(16).hex()),
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
    # API CONFIGURATION
    # =========================
    
    @computed_field
    @property
    def allowed_origins_list(self) -> List[str]:
        """Parse allowed origins from environment."""
        origins_str = os.getenv(
            "ALLOWED_ORIGINS",
            "http://localhost:3000,http://localhost:4200,http://localhost:4201"
        )
        return [origin.strip() for origin in origins_str.split(",")]
    
    ALLOWED_ORIGINS: str = Field(
        default_factory=lambda: os.getenv(
            "ALLOWED_ORIGINS",
            "http://localhost:3000,http://localhost:4200,http://localhost:4201"
        ),
        description="CORS allowed origins (comma-separated)"
    )
    API_HOST: str = Field(
        default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"),
        description="API server host"
    )
    API_PORT: int = Field(
        default_factory=lambda: int(os.getenv("API_PORT", "8000")),
        description="API server port"
    )
    
    @computed_field
    @property
    def uvicorn_workers_adaptive(self) -> int:
        """Adaptive worker count based on environment."""
        explicit = os.getenv("UVICORN_WORKERS")
        if explicit:
            return int(explicit)
        elif self.is_kubernetes:
            return 1  # K8s handles scaling
        else:
            import multiprocessing
            return multiprocessing.cpu_count()
    
    UVICORN_WORKERS: Optional[int] = Field(
        default=None,
        description="Number of Uvicorn workers (auto-detected if not set)"
    )
    
    # =========================
    # LOGGING CONFIGURATION
    # =========================
    LOG_LEVEL: str = Field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"),
        description="Logging level"
    )
    POSTGRES_LOG_LEVEL: str = Field(
        default_factory=lambda: os.getenv("POSTGRES_LOG_LEVEL", "INFO"),
        description="PostgreSQL service log level"
    )
    
    # =========================
    # PYTHON PATH
    # =========================
    PYTHONPATH: str = Field(
        default_factory=lambda: os.getenv("PYTHONPATH", "./app"),
        description="Python path"
    )
    
<<<<<<< HEAD
=======
    # =========================
    # POST-INITIALIZATION
    # =========================
    
    def model_post_init(self, __context):
        """Post-initialization processing."""
        # Set adaptive POSTGRES_HOST if not explicitly set
        if self.POSTGRES_HOST is None:
            self.POSTGRES_HOST = self.postgres_host_adaptive
        
        # Set adaptive REDIS_HOST if not explicitly set
        if self.REDIS_HOST is None:
            self.REDIS_HOST = self.redis_host_adaptive
        
        # Set adaptive SCRAPER_MAX_CONCURRENT if not explicitly set
        if self.SCRAPER_MAX_CONCURRENT is None:
            self.SCRAPER_MAX_CONCURRENT = self.scraper_max_concurrent_adaptive
        
        # Set adaptive UVICORN_WORKERS if not explicitly set
        if self.UVICORN_WORKERS is None:
            self.UVICORN_WORKERS = self.uvicorn_workers_adaptive
        
        # Build REDIS_URL if not set
        if not self.REDIS_URL:
            if self.REDIS_PASSWORD:
                self.REDIS_URL = (
                    f"redis://:{self.REDIS_PASSWORD}@"
                    f"{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
                )
            else:
                self.REDIS_URL = (
                    f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
                )
        
        # Ensure data directories exist
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self.temp_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            import logging
            logging.warning(f"⚠️ Could not create directories: {e}")
    
    # =========================
    # PYDANTIC CONFIGURATION
    # =========================
    
>>>>>>> f861d172c4f76def4655980406c89e76eb8288a1
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="allow"  # Allow extra fields for flexibility
    )


# ============================================================================
# GLOBAL SETTINGS INSTANCE
# ============================================================================
settings = Settings()

# Log configuration summary
import logging
logger = logging.getLogger(__name__)
logger.info(f"🌍 Configuration loaded:")
logger.info(f"   - Environment: {settings.environment_type}")
logger.info(f"   - Platform: {settings.platform_system}")
logger.info(f"   - Data Dir: {settings.data_dir}")
logger.info(f"   - Postgres Host: {settings.POSTGRES_HOST}")
logger.info(f"   - Redis Host: {settings.REDIS_HOST}")
logger.info(f"   - Scraper Concurrent: {settings.SCRAPER_MAX_CONCURRENT}")
logger.info(f"   - Uvicorn Workers: {settings.UVICORN_WORKERS}")