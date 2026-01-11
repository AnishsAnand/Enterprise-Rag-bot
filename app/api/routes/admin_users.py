from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime, timedelta

from app.core.database import get_db
from app.models.database_models import User
from app.api.routes.auth import get_current_active_user

router = APIRouter()

class UserListResponse(BaseModel):
    id: str
    email: str
    name: str
    avatar: Optional[str]
    provider: Optional[str]
    role: str
    is_active: bool
    is_verified: bool
    login_count: int
    created_at: datetime
    last_login: datetime
    last_activity: datetime

class UserStatsResponse(BaseModel):
    total_users: int
    verified_users: int
    active_users: int
    new_users_today: int
    new_users_this_week: int
    new_users_this_month: int
    oauth_users: int
    email_users: int

class UserUpdateRequest(BaseModel):
    name: Optional[str] = None
    role: Optional[str] = None
    is_active: Optional[bool] = None

async def require_admin(current_user: User = Depends(get_current_active_user)):
    if current_user.role != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user

@router.get("/users", response_model=dict)
async def get_users(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    search: Optional[str] = Query(None),
    role: Optional[str] = Query(None),
    provider: Optional[str] = Query(None),
    is_active: Optional[bool] = Query(None),
    is_verified: Optional[bool] = Query(None),
    sort_by: str = Query("created_at"),
    sort_order: str = Query("desc"),
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    
    query = db.query(User)
    
    if search:
        query = query.filter(
            (User.name.ilike(f"%{search}%")) | 
            (User.email.ilike(f"%{search}%"))
        )
    
    if role:
        query = query.filter(User.role == role)
    
    if provider:
        query = query.filter(User.provider == provider)
    
    if is_active is not None:
        query = query.filter(User.is_active == is_active)
    
    if is_verified is not None:
        query = query.filter(User.is_verified == is_verified)
    
    if sort_order == "desc":
        if sort_by == "created_at":
            query = query.order_by(desc(User.created_at))
        elif sort_by == "last_login":
            query = query.order_by(desc(User.last_login))
        elif sort_by == "name":
            query = query.order_by(desc(User.name))
        elif sort_by == "email":
            query = query.order_by(desc(User.email))
        elif sort_by == "login_count":
            query = query.order_by(desc(User.login_count))
    else:
        if sort_by == "created_at":
            query = query.order_by(User.created_at)
        elif sort_by == "last_login":
            query = query.order_by(User.last_login)
        elif sort_by == "name":
            query = query.order_by(User.name)
        elif sort_by == "email":
            query = query.order_by(User.email)
        elif sort_by == "login_count":
            query = query.order_by(User.login_count)
    
    total_count = query.count()
    
    offset = (page - 1) * limit
    users = query.offset(offset).limit(limit).all()
    
    user_list = []
    for user in users:
        user_list.append(UserListResponse(
            id=user.id,
            email=user.email,
            name=user.name,
            avatar=user.avatar,
            provider=user.provider,
            role=user.role,
            is_active=user.is_active,
            is_verified=user.is_verified,
            login_count=user.login_count,
            created_at=user.created_at,
            last_login=user.last_login,
            last_activity=user.last_activity
        ))
    
    return {
        "users": user_list,
        "total_count": total_count,
        "page": page,
        "limit": limit,
        "total_pages": (total_count + limit - 1) // limit
    }

@router.get("/users/stats", response_model=UserStatsResponse)
async def get_user_stats(
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    
    now = datetime.utcnow()
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_ago = today - timedelta(days=7)
    month_ago = today - timedelta(days=30)
    
    total_users = db.query(User).count()
    verified_users = db.query(User).filter(User.is_verified == True).count()
    active_users = db.query(User).filter(User.is_active == True).count()
    
    new_users_today = db.query(User).filter(User.created_at >= today).count()
    new_users_this_week = db.query(User).filter(User.created_at >= week_ago).count()
    new_users_this_month = db.query(User).filter(User.created_at >= month_ago).count()
    
    oauth_users = db.query(User).filter(User.provider.in_(['google', 'github'])).count()
    email_users = db.query(User).filter(User.provider == 'email').count()
    
    return UserStatsResponse(
        total_users=total_users,
        verified_users=verified_users,
        active_users=active_users,
        new_users_today=new_users_today,
        new_users_this_week=new_users_this_week,
        new_users_this_month=new_users_this_month,
        oauth_users=oauth_users,
        email_users=email_users
    )

@router.get("/users/{user_id}")
async def get_user_details(
    user_id: str,
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {
        "id": user.id,
        "email": user.email,
        "name": user.name,
        "avatar": user.avatar,
        "provider": user.provider,
        "provider_id": user.provider_id,
        "role": user.role,
        "is_active": user.is_active,
        "is_verified": user.is_verified,
        "theme": user.theme,
        "language": user.language,
        "timezone": user.timezone,
        "notifications_enabled": user.notifications_enabled,
        "email_notifications": user.email_notifications,
        "two_factor_enabled": user.two_factor_enabled,
        "bio": user.bio,
        "location": user.location,
        "website": user.website,
        "company": user.company,
        "job_title": user.job_title,
        "login_count": user.login_count,
        "last_activity": user.last_activity,
        "preferences": user.preferences,
        "created_at": user.created_at,
        "updated_at": user.updated_at,
        "last_login": user.last_login
    }

@router.put("/users/{user_id}")
async def update_user(
    user_id: str,
    update_data: UserUpdateRequest,
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if user.id == admin_user.id and update_data.role and update_data.role != 'admin':
        raise HTTPException(status_code=400, detail="Cannot change your own admin role")
    
    if update_data.name is not None:
        user.name = update_data.name
    
    if update_data.role is not None:
        if update_data.role not in ['user', 'admin']:
            raise HTTPException(status_code=400, detail="Invalid role")
        user.role = update_data.role
    
    if update_data.is_active is not None:
        if user.id == admin_user.id and not update_data.is_active:
            raise HTTPException(status_code=400, detail="Cannot deactivate your own account")
        user.is_active = update_data.is_active
    
    user.updated_at = datetime.utcnow()
    db.commit()
    
    return {"message": "User updated successfully"}

@router.delete("/users/{user_id}")
async def delete_user(
    user_id: str,
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if user.id == admin_user.id:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")
    
    db.delete(user)
    db.commit()
    
    return {"message": "User deleted successfully"}

@router.post("/users/{user_id}/toggle-status")
async def toggle_user_status(
    user_id: str,
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if user.id == admin_user.id:
        raise HTTPException(status_code=400, detail="Cannot change your own status")
    
    user.is_active = not user.is_active
    user.updated_at = datetime.utcnow()
    db.commit()
    
    status = "activated" if user.is_active else "deactivated"
    return {"message": f"User {status} successfully"}

@router.get("/users/activity/recent")
async def get_recent_user_activity(
    limit: int = Query(50, ge=1, le=100),
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    
    recent_users = db.query(User).order_by(desc(User.last_activity)).limit(limit).all()
    
    activity_list = []
    for user in recent_users:
        activity_list.append({
            "id": user.id,
            "name": user.name,
            "email": user.email,
            "avatar": user.avatar,
            "last_activity": user.last_activity,
            "last_login": user.last_login,
            "login_count": user.login_count
        })
    
    return {"recent_activity": activity_list}
