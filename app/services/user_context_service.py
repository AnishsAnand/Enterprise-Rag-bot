"""
User Context Service for persistent user preferences and entity selections.

Provides CRUD operations for user context preferences that persist across sessions.
This enables users to set defaults for engagement, datacenter, cluster, etc.
that are automatically loaded when they start a new conversation.

Key features:
- Persistent storage in PostgreSQL
- Automatic loading of user defaults on session start
- Support for changing/switching context entities
- Context inheritance across sessions
"""

import logging
from typing import Any, Dict, Optional
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import settings
from app.models.database_models import UserContextPreferences, Base

logger = logging.getLogger(__name__)


class UserContextService:
    """
    Service for managing user context preferences.
    
    Handles persistent storage of user's default selections for:
    - Engagement ID
    - Datacenter/Endpoint
    - Cluster
    - Firewall
    - Business Unit
    - Environment
    - Zone
    """
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern to ensure single database connection."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the service with database connection."""
        if self._initialized:
            return
            
        self.database_url = settings.DATABASE_URL
        
        # Create engine with appropriate settings
        if self.database_url.startswith("sqlite"):
            self.engine = create_engine(
                self.database_url,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool
            )
        else:
            self.engine = create_engine(
                self.database_url,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True
            )
        
        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        # Create tables if they don't exist
        self._ensure_tables()
        
        self._initialized = True
        logger.info("✅ UserContextService initialized")
    
    def _ensure_tables(self) -> None:
        """Ensure the user_context_preferences table exists."""
        try:
            Base.metadata.create_all(bind=self.engine, tables=[UserContextPreferences.__table__])
            logger.info("✅ user_context_preferences table ready")
        except SQLAlchemyError as e:
            logger.error(f"❌ Failed to create tables: {str(e)}")
    
    def _get_session(self) -> Session:
        """Get a database session."""
        return self.SessionLocal()
    
    # ==================== CRUD Operations ====================
    
    def get_user_context(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user's context preferences.
        
        Args:
            user_id: User identifier (email)
            
        Returns:
            Dictionary of user preferences or None if not found
        """
        session = self._get_session()
        try:
            record = session.query(UserContextPreferences).filter(
                UserContextPreferences.user_id == user_id
            ).first()
            
            if record:
                logger.debug(f"✅ Loaded context for user: {user_id}")
                return record.to_dict()
            
            logger.debug(f"ℹ️ No context found for user: {user_id}")
            return None
            
        except SQLAlchemyError as e:
            logger.error(f"❌ Error getting user context: {str(e)}")
            return None
        finally:
            session.close()
    
    def create_or_update_context(
        self,
        user_id: str,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Create or update user's context preferences.
        
        Args:
            user_id: User identifier (email)
            **kwargs: Context fields to update (e.g., default_engagement_id=123)
            
        Returns:
            Updated context dictionary or None on error
        """
        session = self._get_session()
        try:
            record = session.query(UserContextPreferences).filter(
                UserContextPreferences.user_id == user_id
            ).first()
            
            if record:
                # Update existing record
                for key, value in kwargs.items():
                    if hasattr(record, key):
                        setattr(record, key, value)
                record.updated_at = datetime.utcnow()
                logger.info(f"📝 Updated context for user: {user_id} with keys: {list(kwargs.keys())}")
            else:
                # Create new record
                record = UserContextPreferences(user_id=user_id, **kwargs)
                session.add(record)
                logger.info(f"✨ Created context for user: {user_id}")
            
            session.commit()
            session.refresh(record)
            return record.to_dict()
            
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"❌ Error updating user context: {str(e)}")
            return None
        finally:
            session.close()
    
    def delete_user_context(self, user_id: str) -> bool:
        """
        Delete user's context preferences.
        
        Args:
            user_id: User identifier (email)
            
        Returns:
            True if deleted, False otherwise
        """
        session = self._get_session()
        try:
            result = session.query(UserContextPreferences).filter(
                UserContextPreferences.user_id == user_id
            ).delete()
            
            session.commit()
            
            if result > 0:
                logger.info(f"🗑️ Deleted context for user: {user_id}")
                return True
            return False
            
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"❌ Error deleting user context: {str(e)}")
            return False
        finally:
            session.close()
    
    # ==================== Entity-Specific Updates ====================
    
    def set_engagement(
        self,
        user_id: str,
        engagement_id: int,
        engagement_name: str = None,
        ipc_engagement_id: int = None,
        save_as_default: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Set user's engagement context.
        
        Args:
            user_id: User identifier
            engagement_id: PAAS engagement ID
            engagement_name: Human-readable engagement name
            ipc_engagement_id: IPC engagement ID (if available)
            save_as_default: Whether to persist as default
            
        Returns:
            Updated context or None on error
        """
        if not save_as_default:
            # Return without persisting (session-only)
            return {
                "engagement": {
                    "id": engagement_id,
                    "name": engagement_name,
                    "ipc_id": ipc_engagement_id
                }
            }
        
        return self.create_or_update_context(
            user_id=user_id,
            default_engagement_id=engagement_id,
            default_engagement_name=engagement_name,
            default_ipc_engagement_id=ipc_engagement_id
        )
    
    def set_datacenter(
        self,
        user_id: str,
        datacenter_id: int,
        datacenter_name: str = None,
        endpoint_ids: list = None,
        save_as_default: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Set user's datacenter/endpoint context.
        
        Args:
            user_id: User identifier
            datacenter_id: Datacenter ID
            datacenter_name: Human-readable datacenter name
            endpoint_ids: List of endpoint IDs for this datacenter
            save_as_default: Whether to persist as default
            
        Returns:
            Updated context or None on error
        """
        if not save_as_default:
            return {
                "datacenter": {
                    "id": datacenter_id,
                    "name": datacenter_name,
                    "endpoint_ids": endpoint_ids or []
                }
            }
        
        return self.create_or_update_context(
            user_id=user_id,
            default_datacenter_id=datacenter_id,
            default_datacenter_name=datacenter_name,
            default_endpoint_ids=endpoint_ids
        )
    
    def set_cluster(
        self,
        user_id: str,
        cluster_id: int = None,
        cluster_name: str = None,
        save_as_default: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Set user's cluster context."""
        if not save_as_default:
            return {"cluster": {"id": cluster_id, "name": cluster_name}}
        
        return self.create_or_update_context(
            user_id=user_id,
            default_cluster_id=cluster_id,
            default_cluster_name=cluster_name
        )
    
    def set_firewall(
        self,
        user_id: str,
        firewall_id: int = None,
        firewall_name: str = None,
        save_as_default: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Set user's firewall context."""
        if not save_as_default:
            return {"firewall": {"id": firewall_id, "name": firewall_name}}
        
        return self.create_or_update_context(
            user_id=user_id,
            default_firewall_id=firewall_id,
            default_firewall_name=firewall_name
        )
    
    def set_business_unit(
        self,
        user_id: str,
        bu_id: int = None,
        bu_name: str = None,
        save_as_default: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Set user's business unit context."""
        if not save_as_default:
            return {"business_unit": {"id": bu_id, "name": bu_name}}
        
        return self.create_or_update_context(
            user_id=user_id,
            default_business_unit_id=bu_id,
            default_business_unit_name=bu_name
        )
    
    def set_environment(
        self,
        user_id: str,
        env_id: int = None,
        env_name: str = None,
        save_as_default: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Set user's environment context."""
        if not save_as_default:
            return {"environment": {"id": env_id, "name": env_name}}
        
        return self.create_or_update_context(
            user_id=user_id,
            default_environment_id=env_id,
            default_environment_name=env_name
        )
    
    def set_zone(
        self,
        user_id: str,
        zone_id: int = None,
        zone_name: str = None,
        save_as_default: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Set user's zone context."""
        if not save_as_default:
            return {"zone": {"id": zone_id, "name": zone_name}}
        
        return self.create_or_update_context(
            user_id=user_id,
            default_zone_id=zone_id,
            default_zone_name=zone_name
        )
    
    # ==================== Context Clearing ====================
    
    def clear_engagement(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Clear user's engagement context (and dependents)."""
        return self.create_or_update_context(
            user_id=user_id,
            default_engagement_id=None,
            default_engagement_name=None,
            default_ipc_engagement_id=None,
            # Clear dependents
            default_datacenter_id=None,
            default_datacenter_name=None,
            default_endpoint_ids=None,
            default_cluster_id=None,
            default_cluster_name=None,
            default_firewall_id=None,
            default_firewall_name=None,
            default_business_unit_id=None,
            default_business_unit_name=None,
            default_environment_id=None,
            default_environment_name=None,
            default_zone_id=None,
            default_zone_name=None
        )
    
    def clear_datacenter(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Clear user's datacenter context (and dependents)."""
        return self.create_or_update_context(
            user_id=user_id,
            default_datacenter_id=None,
            default_datacenter_name=None,
            default_endpoint_ids=None,
            # Clear dependents (cluster, firewall are datacenter-specific)
            default_cluster_id=None,
            default_cluster_name=None,
            default_firewall_id=None,
            default_firewall_name=None
        )
    
    def clear_all_context(self, user_id: str) -> bool:
        """Clear all context for a user."""
        return self.delete_user_context(user_id)
    
    # ==================== Utility Methods ====================
    
    def get_context_summary(self, user_id: str) -> str:
        """
        Get a human-readable summary of user's current context.
        
        Args:
            user_id: User identifier
            
        Returns:
            Summary string like "Engagement: ABC | Datacenter: Mumbai | ..."
        """
        session = self._get_session()
        try:
            record = session.query(UserContextPreferences).filter(
                UserContextPreferences.user_id == user_id
            ).first()
            
            if record:
                return record.get_context_summary()
            return "No defaults set"
            
        except SQLAlchemyError as e:
            logger.error(f"❌ Error getting context summary: {str(e)}")
            return "Error loading context"
        finally:
            session.close()
    
    def has_engagement(self, user_id: str) -> bool:
        """Check if user has a default engagement set."""
        context = self.get_user_context(user_id)
        if context:
            return context.get("engagement", {}).get("id") is not None
        return False
    
    def has_datacenter(self, user_id: str) -> bool:
        """Check if user has a default datacenter set."""
        context = self.get_user_context(user_id)
        if context:
            return context.get("datacenter", {}).get("id") is not None
        return False
    
    def get_default_engagement_id(self, user_id: str) -> Optional[int]:
        """Get user's default engagement ID."""
        context = self.get_user_context(user_id)
        if context:
            return context.get("engagement", {}).get("id")
        return None
    
    def get_default_datacenter_id(self, user_id: str) -> Optional[int]:
        """Get user's default datacenter ID."""
        context = self.get_user_context(user_id)
        if context:
            return context.get("datacenter", {}).get("id")
        return None
    
    def get_default_endpoint_ids(self, user_id: str) -> Optional[list]:
        """Get user's default endpoint IDs."""
        context = self.get_user_context(user_id)
        if context:
            return context.get("datacenter", {}).get("endpoint_ids")
        return None


# Global singleton instance
user_context_service = UserContextService()
