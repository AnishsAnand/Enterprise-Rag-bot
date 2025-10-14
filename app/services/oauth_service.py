from authlib.integrations.starlette_client import OAuth
from starlette.config import Config
from starlette.requests import Request
import os

config = Config('.env')
oauth = OAuth(config)

oauth.register(
    name='google',
    client_id=os.getenv('GOOGLE_CLIENT_ID'),
    client_secret=os.getenv('GOOGLE_CLIENT_SECRET'),
    server_metadata_url='https://accounts.google.com/.well-known/openid_configuration',
    client_kwargs={
        'scope': 'openid email profile'
    }
)


oauth.register(
    name='github',
    client_id=os.getenv('GITHUB_CLIENT_ID'),
    client_secret=os.getenv('GITHUB_CLIENT_SECRET'),
    access_token_url='https://github.com/login/oauth/access_token',
    authorize_url='https://github.com/login/oauth/authorize',
    api_base_url='https://api.github.com/',
    client_kwargs={'scope': 'user:email'},
)

class OAuthService:
    def __init__(self):
        self.oauth = oauth
    
    async def get_google_auth_url(self, request: Request, redirect_uri: str):
        """Generate Google OAuth authorization URL"""
        return await self.oauth.google.authorize_redirect(request, redirect_uri)
    
    async def get_github_auth_url(self, request: Request, redirect_uri: str):
        """Generate GitHub OAuth authorization URL"""
        return await self.oauth.github.authorize_redirect(request, redirect_uri)
    
    async def get_google_user_info(self, request: Request):
        """Get user info from Google after OAuth callback"""
        try:
            token = await self.oauth.google.authorize_access_token(request)
            user_info = token.get('userinfo')
            if user_info:
                return {
                    'provider_id': user_info['sub'],
                    'email': user_info['email'],
                    'name': user_info['name'],
                    'avatar': user_info.get('picture'),
                    'provider': 'google'
                }
        except Exception as e:
            print(f"Google OAuth error: {e}")
            return None
    
    async def get_github_user_info(self, request: Request):
        """Get user info from GitHub after OAuth callback"""
        try:
            token = await self.oauth.github.authorize_access_token(request)
            resp = await self.oauth.github.get('user', token=token)
            user_data = resp.json()
            email_resp = await self.oauth.github.get('user/emails', token=token)
            emails = email_resp.json()
            primary_email = next((email['email'] for email in emails if email['primary']), None)
            
            if user_data and primary_email:
                return {
                    'provider_id': str(user_data['id']),
                    'email': primary_email,
                    'name': user_data.get('name') or user_data.get('login'),
                    'avatar': user_data.get('avatar_url'),
                    'provider': 'github'
                }
        except Exception as e:
            print(f"GitHub OAuth error: {e}")
            return None

oauth_service = OAuthService()
