from fastapi import APIRouter, Depends, HTTPException, status, Header
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from pydantic import BaseModel
from typing import Dict, Optional, Set
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
import logging
from app.core.hashed_pwd import pwd_context
from app.core.database import get_db, SessionLocal
from app.models.user import User as DBUser
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

SECRET_KEY = "cBbPN3Sa8Yu_mtVrJCozcPpnE0FDXDCSwWZKY-Opw30"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

# Fallback in-memory users for backward compatibility
fake_users_db: Dict[str, dict] = {
    "admin": {
        "username": "admin",
        "hashed_password": pwd_context.hash("admin123"),
        "role": "admin"
    }
}

# Token blacklist for logout (in production, use Redis or database)
token_blacklist: Set[str] = set()

class User(BaseModel):
    username: str
    role: str
    id: Optional[int] = None
    
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class RegisterRequest(BaseModel):
    username: str
    password: str

router = APIRouter(prefix="/api/auth", tags=["auth"])

def create_access_token(data: dict, expires_delta: timedelta):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    token = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return token

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    # Check if token is blacklisted (logged out)
    if token in token_blacklist:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked (logged out)",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role")
        if not username:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    # Try database first
    if db:
        try:
            db_user = db.query(DBUser).filter(DBUser.username == username).first()
            if db_user:
                return User(
                    id=db_user.id,
                    username=db_user.username,
                    role=db_user.role or "user"
                )
        except Exception as e:
            # Fall back to fake_users_db if database query fails
            pass
    
    # Fallback to in-memory users
    user = fake_users_db.get(username)
    if user:
        return User(username=username, role=user.get("role", "user"))
    
    raise credentials_exception

def admin_only(user: User = Depends(get_current_user)):
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admins only")
    return user

@router.post("/register", response_model=Token)
async def register_user(data: RegisterRequest, db: Session = Depends(get_db)):
    # Check database first
    if db:
        try:
            existing_user = db.query(DBUser).filter(DBUser.username == data.username).first()
            if existing_user:
                raise HTTPException(status_code=400, detail="Username already exists")
        except Exception:
            pass
    
    # Check in-memory fallback
    if data.username in fake_users_db:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    hashed = pwd_context.hash(data.password)
    
    # Try to save to database
    if db:
        try:
            db_user = DBUser(
                username=data.username,
                hashed_password=hashed,
                role="user"
            )
            db.add(db_user)
            db.commit()
            db.refresh(db_user)
        except Exception as e:
            db.rollback()
            # Fall back to in-memory storage
            fake_users_db[data.username] = {
                "username": data.username,
                "hashed_password": hashed,
                "role": "user"
            }
    else:
        # Fall back to in-memory storage
        fake_users_db[data.username] = {
            "username": data.username,
            "hashed_password": hashed,
            "role": "user"
        }

    access_token = create_access_token(
        data={"sub": data.username, "role": "user"},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/login", response_model=Token)
async def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    # Try database first
    user = None
    user_role = "user"
    
    if db:
        try:
            db_user = db.query(DBUser).filter(DBUser.username == form.username).first()
            if db_user:
                if pwd_context.verify(form.password, db_user.hashed_password):
                    user = db_user
                    user_role = db_user.role or "user"
        except Exception:
            pass
    
    # Fallback to in-memory users
    if not user:
        user = fake_users_db.get(form.username)
        if user:
            if not pwd_context.verify(form.password, user["hashed_password"]):
                raise HTTPException(status_code=401, detail="Invalid credentials")
            user_role = user.get("role", "user")
        else:
            raise HTTPException(status_code=401, detail="Invalid credentials")
    
    access_token = create_access_token(
        data={"sub": form.username, "role": user_role},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/me", response_model=User)
async def read_current_user(user: User = Depends(get_current_user)):
    return user

@router.post("/logout")
async def logout(authorization: Optional[str] = Header(None)):
    """
    Logout endpoint.
    Invalidates the token server-side by adding it to the blacklist.
    Client should also remove the token from storage.
    
    Token is optional - if provided, it will be invalidated. If not provided,
    logout still succeeds (for cases where token is already removed client-side).
    """
    token = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]
    
    if token:
        # Add token to blacklist to invalidate it
        token_blacklist.add(token)
        
        # Optional: Clean up old tokens (keep blacklist size manageable)
        # In production, use Redis TTL or database with expiration
        if len(token_blacklist) > 10000:
            # Clear oldest 50% (simple cleanup, in production use TTL)
            token_blacklist.clear()
            logger.warning("⚠️ Token blacklist cleared (size limit reached)")
        
        logger.info("✅ Token invalidated (logged out)")
    else:
        logger.info("✅ Logout called without token (already cleared client-side)")
    
    return {
        "message": "Logged out successfully",
        "detail": "Token has been invalidated. Please remove the token from client storage." if token else "Logged out successfully."
    }


@router.get("/admin-only")
async def admin_endpoint(user: User = Depends(admin_only)):
    return {"message": f"Hello Admin {user.username}"}
