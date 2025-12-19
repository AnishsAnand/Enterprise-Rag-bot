# ✅ Two-Tier Access System - Implementation Summary

## What's Been Implemented

### 1. ✅ User Signup Enabled
OpenWebUI now allows users to create their own accounts:
- Visit `http://localhost:3000/`
- Users can self-register with email/password
- `ENABLE_SIGNUP=true` is configured

### 2. ✅ Role-Based Permissions
The backend now distinguishes between two types of users:

#### Tier 1: Regular Users (Self-Registered)
- **Role**: `viewer`
- **Can do**:
  - ✅ Chat with AI
  - ✅ Ask questions about products
  - ✅ Read RAG documents
  - ✅ List available resources (read-only)
- **Cannot do**:
  - ❌ Create clusters
  - ❌ Delete resources
  - ❌ Perform any write operations

#### Tier 2: Authorized Users (SSO or Admin)
- **Role**: `admin`, `developer`
- **Can do**:
  - ✅ Everything Tier 1 can do
  - ✅ Create clusters
  - ✅ Delete resources
  - ✅ Perform all write operations

### 3. ✅ Custom Error Messages
When unauthorized users try to perform actions, they get:

```json
{
  "success": false,
  "error": "Unauthorized",
  "message": "You don't have permission to perform this action.",
  "enrollment_info": {
    "title": "Want to perform actions?",
    "description": "Enroll for full access to create and manage cloud resources.",
    "enrollment_url": "https://cloud.tatacommunications.com/enroll",
    "contact": "support@tatacommunications.com",
    "sso_login": "Sign in with Tata Communications for full access"
  }
}
```

## What's Next: SSO Integration

To complete the two-tier system, you need to integrate Tata Communications SSO:

### Step 1: Get OAuth Credentials
Contact Tata Communications IT to obtain:
1. **Client ID**
2. **Client Secret**
3. **Confirm Redirect URI**: `http://localhost:3000/oauth/oidc/callback`

### Step 2: Update OpenWebUI Configuration

Stop the current container and start with SSO configuration:

```bash
# Stop current container
docker stop enterprise-rag-openwebui
docker rm enterprise-rag-openwebui

# Start with SSO enabled
docker run -d \
  --name enterprise-rag-openwebui \
  -p 3000:8080 \
  -e WEBUI_AUTH=true \
  -e ENABLE_SIGNUP=true \
  -e WEBUI_SECRET_KEY="secure-webui-secret-key-for-sessions-2024" \
  \
  -e OPENAI_API_BASE_URL="http://host.docker.internal:8001/api/v1" \
  -e OPENAI_API_KEY="secure-openwebui-api-key-2024" \
  \
  -e ENABLE_RAG=true \
  -e ENABLE_OLLAMA_API=false \
  -e ENABLE_OPENAI_API=true \
  -e DEFAULT_MODELS="Vayu Maya" \
  -e DEFAULT_USER_ROLE=user \
  -e WEBUI_NAME="Vayu Maya - AI Cloud Assistant" \
  \
  -e ENABLE_OAUTH_SIGNUP=true \
  -e OAUTH_CLIENT_ID="<YOUR_CLIENT_ID>" \
  -e OAUTH_CLIENT_SECRET="<YOUR_CLIENT_SECRET>" \
  -e OPENID_PROVIDER_URL="https://idp.tatacommunications.com/auth/realms/master/.well-known/openid-configuration" \
  -e OAUTH_PROVIDER_NAME="Tata Communications" \
  -e OPENID_REDIRECT_URI="http://localhost:3000/oauth/oidc/callback" \
  -e ENABLE_OAUTH_ROLE_MANAGEMENT=true \
  -e OAUTH_ROLES_CLAIM="roles" \
  \
  --add-host=host.docker.internal:host-gateway \
  -v openwebui-data:/app/backend/data \
  ghcr.io/open-webui/open-webui:main
```

### Step 3: Test the System

#### Test 1: Regular User (Self-Registration)
1. Visit `http://localhost:3000/`
2. Click "Create Account"
3. Register with email/password
4. Try to create a cluster
5. **Expected**: Get enrollment message

#### Test 2: SSO User (Tata Communications)
1. Visit `http://localhost:3000/`
2. Click "Sign in with Tata Communications"
3. Login with Tata Communications credentials
4. Try to create a cluster
5. **Expected**: Successfully create cluster

## Current Status

| Feature | Status | Notes |
|---------|--------|-------|
| User Signup | ✅ Enabled | Users can self-register |
| Role-Based Permissions | ✅ Implemented | viewer vs admin/developer |
| Custom Error Messages | ✅ Implemented | Shows enrollment info |
| Model Access for All | ✅ Enabled | All users can chat |
| SSO Integration | ⏳ Pending | Need OAuth credentials |

## Files Modified

1. **`app/routers/openai_compatible.py`**
   - Added role detection based on auth provider
   - SSO users get full access
   - Regular users get viewer access

2. **`app/services/api_executor_service.py`**
   - Enhanced permission denied messages
   - Added enrollment information
   - Distinguishes between read-only and write operations

3. **OpenWebUI Configuration**
   - `ENABLE_SIGNUP=true` - Allow self-registration
   - `DEFAULT_USER_ROLE=user` - New users are regular users
   - Ready for SSO integration

## Next Action Required

**Contact Tata Communications IT** to get:
- OAuth Client ID
- OAuth Client Secret

Once you have these, I'll help you update the configuration and test the complete two-tier system.

## Testing Checklist

- [x] OpenWebUI running on port 3000
- [x] Signup enabled
- [x] Role-based permissions implemented
- [x] Custom error messages added
- [ ] SSO credentials obtained
- [ ] SSO configuration added
- [ ] Test regular user flow
- [ ] Test SSO user flow
- [ ] Test permission enforcement

