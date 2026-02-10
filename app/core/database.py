import os
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from typing import Generator
from app.models.database_models import Base

logger = logging.getLogger(__name__)
# Database configuration with proper fallback
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://ragbot:ragbot_secret_2024@localhost:5432/ragbot_db")
# Create engine with connection pooling
try:
    # Try to connect to PostgreSQL
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,  # Verify connections before using
        pool_size=5,
        max_overflow=10,
        echo=False, )
    logger.info(f"✅ Database engine created: {DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else 'SQLite'}")
except Exception as e:
    logger.warning(f"⚠️ Failed to create PostgreSQL engine: {e}")
    logger.warning("⚠️ Falling back to SQLite for user authentication")
    # Fallback to SQLite for auth (vector search will be disabled)
    DATABASE_URL = "sqlite:///./enterprise_rag.db"
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool)
    logger.info("✅ Using SQLite fallback database")
# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False,autoflush=False,bind=engine)

async def init_db() -> None:
    """Initialize database - create all tables and apply schema fixes."""
    try:
        # Test connection first
        with SessionLocal() as session:
            session.execute(text("SELECT 1"))
        logger.info("✅ Database connection test successful")
        # Apply schema migrations if needed
        if "postgresql" in DATABASE_URL.lower():
            await _apply_schema_fixes()
        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables created/verified")
    except Exception as e:
        logger.error(f"❌ Database initialization error: {e}")
        logger.error("Continuing with limited functionality...")
        # Don't raise - allow app to start in degraded mode

async def _apply_schema_fixes() -> None:
    """Apply schema fixes for missing columns."""
    try:
        with SessionLocal() as session:
            # Check if full_name column exists
            result = session.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name='users' AND column_name='full_name'
                """))
            if not result.fetchone():
                logger.info("🔧 Adding missing 'full_name' column to users table...")
                session.execute(text("""
                    ALTER TABLE users 
                    ADD COLUMN full_name VARCHAR(255)
                    """))
                session.commit()
                logger.info("✅ Added full_name column")
            # Check other potentially missing columns
            missing_columns = {
                'avatar_url': 'VARCHAR(500)',
                'bio': 'TEXT',
                'theme': "VARCHAR(50) DEFAULT 'light'",
                'language': "VARCHAR(10) DEFAULT 'en'",
                'timezone': "VARCHAR(50) DEFAULT 'UTC'",
                'notifications_enabled': 'BOOLEAN DEFAULT TRUE',
                'email_notifications': 'BOOLEAN DEFAULT FALSE',
                'last_login': 'TIMESTAMP',
                'login_count': 'INTEGER DEFAULT 0',
                'failed_login_attempts': 'INTEGER DEFAULT 0',
                'locked_until': 'TIMESTAMP'}
            
            for col_name, col_type in missing_columns.items():
                result = session.execute(text(f"""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name='users' AND column_name='{col_name}'
                """))
                if not result.fetchone():
                    logger.info(f"🔧 Adding missing '{col_name}' column...")
                    try:
                        session.execute(text(f"""
                            ALTER TABLE users 
                            ADD COLUMN {col_name} {col_type}
                        """))
                        session.commit()
                        logger.info(f"✅ Added {col_name} column")
                    except Exception as e:
                        logger.debug(f"Column {col_name} may already exist or have different type: {e}")
                        session.rollback()
            logger.info("✅ Schema fixes applied successfully")
    except Exception as e:
        logger.warning(f"⚠️ Schema fix attempt failed (this is okay if tables don't exist yet): {e}")

def get_db() -> Generator[Session, None, None]:
    """
    Dependency for getting database session.
    Usage in FastAPI routes: db: Session = Depends(get_db)
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def check_db_connection() -> bool:
    """Check if database connection is working."""
    try:
        with SessionLocal() as session:
            session.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"❌ Database connection check failed: {e}")
        return False