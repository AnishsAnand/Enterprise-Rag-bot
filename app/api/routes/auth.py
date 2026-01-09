from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.database_models import User as DBUser
import os

# ----------------------------------------------------------------
# Security / Auth Setup
# ----------------------------------------------------------------
SECRET_KEY = os.getenv("SECRET_KEY", "cBbPN3Sa8Yu_mtVrJCozcPpnE0FDXDCSwWZKY-Opw30")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60))

DEFAULT_ADMIN_USER = os.getenv("DEFAULT_ADMIN_USER", "admin")
DEFAULT_ADMIN_PASSWORD = os.getenv("DEFAULT_ADMIN_PASSWORD", "admin123")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

# ----------------------------------------------------------------
# Schemas
# ----------------------------------------------------------------
class User(BaseModel):
    username: str
    role: str
    id: Optional[int] = None

class Token(BaseModel):
    access_token: str
    token_type: str
    user: User

class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str

# ----------------------------------------------------------------
# Router
# ----------------------------------------------------------------
router = APIRouter(prefix="/api/auth", tags=["auth"])

# ----------------------------------------------------------------
# Utility Functions
# ----------------------------------------------------------------
def create_access_token(data: dict, expires_delta: timedelta):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        role = payload.get("role")
        user_id = payload.get("id")

        if not username or not role:
            raise HTTPException(status_code=401, detail="Invalid token")

    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    db_user = db.query(DBUser).filter(DBUser.id == user_id).first()

    if not db_user or not db_user.is_active:
        raise HTTPException(status_code=401, detail="Inactive user")

    return User(username=username, role=role, id=db_user.id)

def admin_only(user: User = Depends(get_current_user)):
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user

# ----------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------
@router.post("/register", response_model=Token)
async def register_user(data: RegisterRequest, db: Session = Depends(get_db)):
    existing_user = db.query(DBUser).filter(
        (DBUser.username == data.username) | (DBUser.email == data.email)
    ).first()

    if existing_user:
        raise HTTPException(status_code=400, detail="User already exists")

    new_user = DBUser(
        username=data.username,
        email=data.email,
        hashed_password=pwd_context.hash(data.password),
        role=DBUser.role.type.enum_class.USER,
        is_active=True,
        is_verified=True,
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    access_token = create_access_token(
        data={
            "sub": new_user.username,
            "role": new_user.role.value,
            "id": new_user.id,
        },
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {"username": new_user.username, "role": new_user.role.value, "id": new_user.id},
    }

@router.post("/login", response_model=Token)
async def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """
    Login endpoint. Checks DB first, then falls back to default admin if no user found.
    """
    user = db.query(DBUser).filter(DBUser.username == form.username).first()

    # If user exists in DB
    if user:
        if not pwd_context.verify(form.password, user.hashed_password):
            raise HTTPException(status_code=401, detail="Invalid username or password")
        if not user.is_active:
            raise HTTPException(status_code=403, detail="Account disabled")
        user.last_login = datetime.utcnow()
        user.login_count += 1
        db.commit()
        access_token = create_access_token(
            data={"sub": user.username, "role": user.role.value, "id": user.id},
            expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
        )
        return {"access_token": access_token, "token_type": "bearer", "user": {"username": user.username, "role": user.role.value, "id": user.id}}

    # Fallback to default admin login
    if form.username == DEFAULT_ADMIN_USER and form.password == DEFAULT_ADMIN_PASSWORD:
        access_token = create_access_token(
            data={"sub": DEFAULT_ADMIN_USER, "role": "ADMIN", "id": 0},
            expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
        )
        return {"access_token": access_token, "token_type": "bearer", "user": {"username": DEFAULT_ADMIN_USER, "role": "ADMIN", "id": 0}}

    # Login failed
    raise HTTPException(status_code=401, detail="Invalid username or password")

@router.get("/me", response_model=User)
async def read_current_user(user: User = Depends(get_current_user)):
    return user

@router.get("/admin-only")
async def admin_endpoint(user: User = Depends(admin_only)):
    return {"message": f"Hello Admin {user.username}", "user_id": user.id}
