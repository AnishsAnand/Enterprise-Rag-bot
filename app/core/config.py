import os
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OPENROUTER_API_KEY: Optional[str] = os.getenv("OPENROUTER_API_KEY")
    VOYAGE_API_KEY: Optional[str] = os.getenv("VOYAGE_API_KEY")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    GROK_API_KEY: Optional[str] = os.getenv("GROK_API_KEY")
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./ragbot.db")
    REDIS_URL: Optional[str] = os.getenv("REDIS_URL")
    CHROMA_PERSIST_DIRECTORY: str = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    UPLOAD_DIRECTORY: str = os.getenv("UPLOAD_DIRECTORY", "./uploads")
    OUTPUT_DIRECTORY: str = os.getenv("OUTPUT_DIRECTORY", "./outputs")
    pythonpath: str  = os.getenv("PYTHONPATH", "./app")
    
    DOTS_OCR_API_KEY: str = os.getenv("DOTS_OCR_API_KEY", "")
    OCR_ENABLED: bool = os.getenv("OCR_ENABLED", "true").lower() == "true"
    OCR_MAX_IMAGES_PER_PAGE: int = int(os.getenv("OCR_MAX_IMAGES_PER_PAGE", "5"))
    OCR_TIMEOUT_SECONDS: int = int(os.getenv("OCR_TIMEOUT_SECONDS", "30"))
    
    SCRAPER_MAX_CONCURRENT: int = int(os.getenv("SCRAPER_MAX_CONCURRENT", "3"))
    SCRAPER_DELAY_SECONDS: float = float(os.getenv("SCRAPER_DELAY_SECONDS", "1.0"))

    MILVUS_HOST: str = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT: int = int(os.getenv("MILVUS_PORT", "19530"))
    MILVUS_USER: Optional[str] = os.getenv("MILVUS_USER", "")
    MILVUS_PASSWORD: Optional[str] = os.getenv("MILVUS_PASSWORD", "")
    MILVUS_COLLECTION: str = os.getenv("MILVUS_COLLECTION", "enterprise_rag")


    class Config:
        env_file = ".env"
        extra = "allow"  

settings = Settings()
