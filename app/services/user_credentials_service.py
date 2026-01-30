"""
Service for retrieving user API credentials from the database.
Handles fetching API authentication credentials (email/password) for users.
"""

from typing import Optional, Dict, Any
from sqlalchemy.orm import Session
from app.core.database import SessionLocal
from app.models.database_models import User
from cryptography.fernet import Fernet
import os
import base64
import logging

logger = logging.getLogger(__name__)

# Encryption key for storing passwords securely
# In production, use a proper key management system
def _get_encryption_key() -> bytes:
    """Get or generate encryption key for password storage."""
    key = os.getenv("CREDENTIALS_ENCRYPTION_KEY")
    if not key:
        # Generate a key (store this in env in production!)
        key = Fernet.generate_key().decode()
        logger.warning("⚠️ Using generated encryption key. Set CREDENTIALS_ENCRYPTION_KEY in .env for production!")
    else:
        key = key.encode() if isinstance(key, str) else key
    return key

def _encrypt_password(password: str) -> str:
    """Encrypt a password before storing."""
    try:
        f = Fernet(_get_encryption_key())
        encrypted = f.encrypt(password.encode())
        return encrypted.decode()
    except Exception as e:
        logger.error(f"❌ Encryption failed: {e}")
        # Fallback: base64 encoding (not secure, but better than plain text)
        return base64.b64encode(password.encode()).decode()

def _decrypt_password(encrypted_password: str) -> str:
    """Decrypt a stored password."""
    try:
        f = Fernet(_get_encryption_key())
        decrypted = f.decrypt(encrypted_password.encode())
        return decrypted.decode()
    except Exception as e:
        logger.error(f"❌ Decryption failed: {e}")
        # Fallback: try base64 decode
        try:
            return base64.b64decode(encrypted_password.encode()).decode()
        except:
            return encrypted_password  # Return as-is if decryption fails


class UserCredentialsService:
    """
    Service for managing user API credentials.
    """
    
    @staticmethod
    def get_user_credentials(username: str) -> Optional[Dict[str, str]]:
        """
        Get API credentials for a user from the database.
        
        Args:
            username: Username to look up (can be username or email)
            
        Returns:
            Dict with 'email' and 'password' keys, or None if not found
        """
        db: Session = SessionLocal()
        try:
            # Try to find user by username first, then by email (api_auth_email)
            user = db.query(User).filter(User.username == username).first()
            
            if not user:
                # Try to find by api_auth_email (for WebUI users where email is passed)
                user = db.query(User).filter(User.api_auth_email == username).first()
            
            if not user:
                logger.warning(f"⚠️ User not found: {username}")
                return None
            
            if not user.api_auth_email or not user.api_auth_password:
                logger.warning(f"⚠️ API credentials not configured for user: {username}")
                return None
            
            logger.info(f"✅ Retrieved API credentials for user: {username}")
            # Decrypt password before returning
            decrypted_password = _decrypt_password(user.api_auth_password)
            return {
                "email": user.api_auth_email,
                "password": decrypted_password
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get user credentials: {str(e)}")
            return None
        finally:
            db.close()
    
    @staticmethod
    def get_credentials_by_email(email: str) -> Optional[Dict[str, str]]:
        """
        Get API credentials for a user by their email address.
        This is used when WebUI sends X-User-Email header.
        
        Args:
            email: Email address to look up
            
        Returns:
            Dict with 'email' and 'password' keys, or None if not found
        """
        db: Session = SessionLocal()
        try:
            # Find user by api_auth_email
            user = db.query(User).filter(User.api_auth_email == email).first()
            
            if not user:
                logger.warning(f"⚠️ No stored credentials for email: {email}")
                return None
            
            if not user.api_auth_password:
                logger.warning(f"⚠️ Password not stored for email: {email}")
                return None
            
            logger.info(f"✅ Retrieved API credentials by email: {email}")
            decrypted_password = _decrypt_password(user.api_auth_password)
            return {
                "email": user.api_auth_email,
                "password": decrypted_password
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get credentials by email: {str(e)}")
            return None
        finally:
            db.close()
    
    @staticmethod
    def store_credentials(email: str, password: str) -> bool:
        """
        Store API credentials for a user. Creates a new user if doesn't exist.
        This is called when user provides their credentials via the setup endpoint.
        
        Args:
            email: User's email (used as both username and api_auth_email)
            password: User's password for Tata Auth API
            
        Returns:
            True if successful, False otherwise
        """
        db: Session = SessionLocal()
        try:
            # Check if user exists by email
            user = db.query(User).filter(User.api_auth_email == email).first()
            
            if not user:
                # Create new user with email as username
                from passlib.context import CryptContext
                pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
                
                user = User(
                    username=email,
                    hashed_password=pwd_context.hash(password),  # Hash for local auth
                    role="user",
                    api_auth_email=email,
                    api_auth_password=_encrypt_password(password)  # Encrypt for API auth
                )
                db.add(user)
                logger.info(f"✅ Created new user with credentials: {email}")
            else:
                # Update existing user's API credentials
                user.api_auth_password = _encrypt_password(password)
                logger.info(f"✅ Updated API credentials for: {email}")
            
            db.commit()
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"❌ Failed to store credentials: {str(e)}")
            return False
        finally:
            db.close()
    
    @staticmethod
    def update_user_credentials(
        username: str,
        api_auth_email: str,
        api_auth_password: str
    ) -> bool:
        """
        Update API credentials for a user.
        
        Args:
            username: Username to update
            api_auth_email: API authentication email
            api_auth_password: API authentication password
            
        Returns:
            True if successful, False otherwise
        """
        db: Session = SessionLocal()
        try:
            user = db.query(User).filter(User.username == username).first()
            
            if not user:
                logger.error(f"❌ User not found: {username}")
                return False
            
            user.api_auth_email = api_auth_email
            # Encrypt password before storing
            user.api_auth_password = _encrypt_password(api_auth_password)
            
            db.commit()
            logger.info(f"✅ Updated API credentials for user: {username}")
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"❌ Failed to update user credentials: {str(e)}")
            return False
        finally:
            db.close()
    
    @staticmethod
    def get_credentials_from_env() -> Optional[Dict[str, str]]:
        """
        Fallback: Get credentials from environment variables.
        This is used when user credentials are not available in the database.
        
        Returns:
            Dict with 'email' and 'password' keys, or None if not found
        """
        import os
        email = os.getenv("API_AUTH_EMAIL", "")
        password = os.getenv("API_AUTH_PASSWORD", "")
        
        if email and password:
            logger.info("✅ Using API credentials from environment variables")
            return {"email": email, "password": password}
        
        logger.warning("⚠️ API credentials not found in environment variables")
        return None


# Global instance
user_credentials_service = UserCredentialsService()

