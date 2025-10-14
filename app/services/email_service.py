from fastapi_mail import FastMail, MessageSchema, ConnectionConfig, MessageType
from pydantic import EmailStr
from typing import List
import os
from jinja2 import Environment, FileSystemLoader
from app.core.config import settings

class EmailService:
    def __init__(self):
        self.conf = ConnectionConfig(
            MAIL_USERNAME=os.getenv("MAIL_USERNAME", ""),
            MAIL_PASSWORD=os.getenv("MAIL_PASSWORD", ""),
            MAIL_FROM=os.getenv("MAIL_FROM", "noreply@enterpriseragbot.com"),
            MAIL_PORT=int(os.getenv("MAIL_PORT", "587")),
            MAIL_SERVER=os.getenv("MAIL_SERVER", "smtp.gmail.com"),
            MAIL_FROM_NAME=os.getenv("MAIL_FROM_NAME", "Enterprise RAG Bot"),
            MAIL_STARTTLS=True,
            MAIL_SSL_TLS=False,
            USE_CREDENTIALS=True,
            VALIDATE_CERTS=True,
            TEMPLATE_FOLDER="app/templates/email"
        )
        self.template_env = Environment(
            loader=FileSystemLoader("app/templates/email")
        )
        
        self.fastmail = FastMail(self.conf)
    
    async def send_verification_email(self, email: EmailStr, name: str, verification_token: str):
        """Send email verification email"""
        verification_url = f"{os.getenv('FRONTEND_URL', 'http://localhost:4200')}/verify-email?token={verification_token}"
        
        template = self.template_env.get_template("verification.html")
        html_content = template.render(
            name=name,
            verification_url=verification_url,
            app_name="Enterprise RAG Bot"
        )
        
        message = MessageSchema(
            subject="Verify Your Email - Enterprise RAG Bot",
            recipients=[email],
            body=html_content,
            subtype=MessageType.html
        )
        
        try:
            await self.fastmail.send_message(message)
            return True
        except Exception as e:
            print(f"Failed to send verification email: {e}")
            return False
    
    async def send_password_reset_email(self, email: EmailStr, name: str, reset_token: str):
        """Send password reset email"""
        reset_url = f"{os.getenv('FRONTEND_URL', 'http://localhost:4200')}/reset-password?token={reset_token}"
        
        template = self.template_env.get_template("password_reset.html")
        html_content = template.render(
            name=name,
            reset_url=reset_url,
            app_name="Enterprise RAG Bot"
        )
        
        message = MessageSchema(
            subject="Reset Your Password - Enterprise RAG Bot",
            recipients=[email],
            body=html_content,
            subtype=MessageType.html
        )
        
        try:
            await self.fastmail.send_message(message)
            return True
        except Exception as e:
            print(f"Failed to send password reset email: {e}")
            return False
    
    async def send_welcome_email(self, email: EmailStr, name: str):
        """Send welcome email to new users"""
        template = self.template_env.get_template("welcome.html")
        html_content = template.render(
            name=name,
            app_name="Enterprise RAG Bot",
            dashboard_url=f"{os.getenv('FRONTEND_URL', 'http://localhost:4200')}/dashboard"
        )
        
        message = MessageSchema(
            subject="Welcome to Enterprise RAG Bot!",
            recipients=[email],
            body=html_content,
            subtype=MessageType.html
        )
        
        try:
            await self.fastmail.send_message(message)
            return True
        except Exception as e:
            print(f"Failed to send welcome email: {e}")
            return False

email_service = EmailService()
