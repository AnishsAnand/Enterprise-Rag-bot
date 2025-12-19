from sqlalchemy import Column, Integer, String
from app.core.database import Base

class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    role = Column(String, default='user')
    # API credentials for Tata Communications API
    api_auth_email = Column(String, nullable=True)
    api_auth_password = Column(String, nullable=True)
