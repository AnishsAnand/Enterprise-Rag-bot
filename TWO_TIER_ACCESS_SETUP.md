# Two-Tier Access System Setup

## Requirements

### Tier 1: Public Users (Self-Registration)
- ✅ Can create their own accounts
- ✅ Can access RAG documents
- ✅ Can ask questions about products
- ❌ **Cannot perform actions** (create clusters, etc.)
- If they try actions → Show "Unauthorized - Please enroll for full access"

### Tier 2: Authorized Users (SSO via Tata Communications)
- ✅ Everything Tier 1 can do
- ✅ **Can perform actions** (create clusters, manage resources)
- ✅ Authenticated via: `https://idp.tatacommunications.com/auth/realms/master/protocol/openid-connect/auth`

## Implementation Steps

### Step 1: Enable User Signup in OpenWebUI

```bash
# Add these environment variables
ENABLE_SIGNUP=true                    # Allow users to register
ENABLE_OAUTH_SIGNUP=true              # Enable SSO signup
DEFAULT_USER_ROLE=user                # New users get 'user' role
```

### Step 2: Configure Tata Communications SSO

```bash
# OAuth/OIDC Configuration
OAUTH_CLIENT_ID=<your-client-id>
OAUTH_CLIENT_SECRET=<your-client-secret>
OPENID_PROVIDER_URL=https://idp.tatacommunications.com/auth/realms/master/.well-known/openid-configuration
OAUTH_PROVIDER_NAME=Tata Communications
OPENID_REDIRECT_URI=http://localhost:3000/oauth/oidc/callback

# Group/Role Management
ENABLE_OAUTH_ROLE_MANAGEMENT=true
OAUTH_ROLES_CLAIM=roles              # Extract roles from SSO token
OAUTH_ALLOWED_ROLES=admin,developer  # Roles that can perform actions
```

### Step 3: Update Backend Permission Logic

The backend needs to check user authentication level:

```python
# In app/routers/openai_compatible.py

def get_user_roles(request: Request) -> List[str]:
    """
    Extract user roles from request.
    - SSO users: Get roles from JWT token
    - Regular users: Default to 'user' role
    """
    auth_header = request.headers.get("Authorization", "")
    
    if auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        try:
            # Decode JWT and extract roles
            payload = jwt.decode(token, options={"verify_signature": False})
            roles = payload.get("roles", [])
            
            # If user has SSO roles, they can perform actions
            if any(role in ["admin", "developer"] for role in roles):
                return ["admin", "developer", "viewer"]
            else:
                # Regular user - read-only access
                return ["viewer"]  # Can read docs, cannot perform actions
        except:
            return ["viewer"]  # Default to read-only
    
    return ["viewer"]  # No auth = read-only
```

### Step 4: Customize Error Messages

When unauthorized users try to perform actions:

```python
# In app/services/api_executor_service.py

def check_permissions(user_roles: List[str], required_permissions: List[str]) -> bool:
    """Check if user has required permissions"""
    if not any(role in required_permissions for role in user_roles):
        raise HTTPException(
            status_code=403,
            detail={
                "error": "Unauthorized",
                "message": "You don't have permission to perform this action.",
                "enrollment_info": {
                    "title": "Want to perform actions?",
                    "description": "Enroll for full access to create and manage cloud resources.",
                    "enrollment_url": "https://your-enrollment-page.com",
                    "contact": "support@tatacommunications.com"
                }
            }
        )
    return True
```

## Docker Configuration

```yaml
# misc/docker/docker-compose.openwebui.yml
services:
  open-webui:
    environment:
      # Authentication
      - WEBUI_AUTH=true
      - ENABLE_SIGNUP=true  # ← Allow self-registration
      - ENABLE_OAUTH_SIGNUP=true  # ← Enable SSO
      
      # SSO Configuration (Tata Communications)
      - OAUTH_CLIENT_ID=${OAUTH_CLIENT_ID}
      - OAUTH_CLIENT_SECRET=${OAUTH_CLIENT_SECRET}
      - OPENID_PROVIDER_URL=https://idp.tatacommunications.com/auth/realms/master/.well-known/openid-configuration
      - OAUTH_PROVIDER_NAME=Tata Communications
      - OPENID_REDIRECT_URI=http://localhost:3000/oauth/oidc/callback
      
      # Role Management
      - ENABLE_OAUTH_ROLE_MANAGEMENT=true
      - OAUTH_ROLES_CLAIM=roles
      - DEFAULT_USER_ROLE=user
      
      # Permissions
      - ENABLE_API_KEY=false  # Disable API keys for regular users
```

## User Experience Flow

### For Public Users (Self-Registration):

1. **Visit**: `http://localhost:3000/`
2. **See**: Two options:
   - "Sign in with Tata Communications" (SSO)
   - "Create Account" (Email/Password)
3. **Choose**: "Create Account"
4. **Access**:
   - ✅ Can chat with AI
   - ✅ Can ask about products
   - ✅ Can read documentation
5. **Try Action**: "Create a Kubernetes cluster"
6. **Result**: 
   ```
   ❌ Unauthorized
   
   You don't have permission to perform this action.
   
   Want to perform actions?
   Enroll for full access to create and manage cloud resources.
   
   Contact: support@tatacommunications.com
   Enrollment: https://your-enrollment-page.com
   ```

### For Authorized Users (SSO):

1. **Visit**: `http://localhost:3000/`
2. **Click**: "Sign in with Tata Communications"
3. **Redirect**: To `https://idp.tatacommunications.com/auth/realms/master/protocol/openid-connect/auth`
4. **Login**: With Tata Communications credentials
5. **Access**:
   - ✅ Can chat with AI
   - ✅ Can ask about products
   - ✅ Can read documentation
   - ✅ **Can perform actions** (create clusters, etc.)

## Implementation Checklist

- [ ] Get OAuth credentials from Tata Communications
- [ ] Update `.env` file with SSO configuration
- [ ] Restart OpenWebUI with new environment variables
- [ ] Update backend permission logic
- [ ] Customize unauthorized error messages
- [ ] Test self-registration flow
- [ ] Test SSO login flow
- [ ] Test permission enforcement
- [ ] Create enrollment page/process

## Next Steps

1. **Get SSO Credentials**: Contact Tata Communications IT to get:
   - OAuth Client ID
   - OAuth Client Secret
   - Confirm redirect URI

2. **Update Configuration**: I'll help you update the docker-compose file

3. **Test**: Verify both user types work correctly

