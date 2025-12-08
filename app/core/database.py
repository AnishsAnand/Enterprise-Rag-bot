from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# Use PostgreSQL for session persistence (Memori)
# Docker container: ragbot-postgres on port 5435
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://ragbot:ragbot_secret_2024@localhost:5435/ragbot_sessions"
)

# Configure engine based on database type
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False}
    )
else:
    # PostgreSQL with connection pooling
    engine = create_engine(
        DATABASE_URL,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True  # Verify connections are alive
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

async def init_db():
    """Initialize database"""
    from app.models import User
    try:
        Base.metadata.create_all(bind=engine)
        # Determine database type for accurate logging
        db_type = "PostgreSQL" if DATABASE_URL.startswith("postgresql") else "SQLite"
        # Mask password in log output
        safe_url = DATABASE_URL
        if "@" in DATABASE_URL:
            parts = DATABASE_URL.split("@")
            prefix = parts[0].split("://")[0] + "://*****"
            safe_url = prefix + "@" + parts[1]
        print(f"✅ Database initialized successfully with {db_type}: {safe_url}")
    except Exception as e:
        print(f"❌ Database initialization error: {e}")
        print("Continuing without database...")

def get_db():
    try:
        db = SessionLocal()
        yield db
    except Exception as e:
        print(f"Database connection error: {e}")
        yield None
    finally:
        try:
            if 'db' in locals():
                db.close()
        except:
            pass
