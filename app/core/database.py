from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "sqlite:///./ragbot.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

async def init_db():
    """Initialize database"""
    from app.models import User
    try:
        Base.metadata.create_all(bind=engine)
        print(f"Database initialized successfully with SQLite: {DATABASE_URL}")
    except Exception as e:
        print(f"Database initialization error: {e}")
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
