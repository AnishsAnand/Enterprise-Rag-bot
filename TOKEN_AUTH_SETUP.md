# 🔑 Dynamic Token Authentication Setup

## Overview

The API Executor Service now automatically fetches and refreshes authentication tokens from the Tata Communications auth API before making any API calls.

## 🔧 Configuration

Add the following environment variables to your `.env` file:

```bash
# ===== API Authentication Configuration =====

# Auth API endpoint (fetches bearer token)
API_AUTH_URL=https://ipcloud.tatacommunications.com/portalservice/api/v1/getAuthToken

# Auth credentials
API_AUTH_EMAIL=izo_cloud_admin@tatacommunications.onmicrosoft.com
API_AUTH_PASSWORD=Tata@1234

# API Executor configuration
API_EXECUTOR_TIMEOUT=30
API_EXECUTOR_MAX_RETRIES=3
```

## 🔄 How It Works

### 1. **Automatic Token Refresh**
- Token is fetched automatically before EVERY API call
- Token is cached with a 1-hour expiry (configurable)
- Refreshed automatically when expired (with 5-minute buffer)
- Thread-safe with async locks to prevent concurrent refreshes

### 2. **Token Lifecycle**

```
┌─────────────────────────────────────────────────────────┐
│ User Initiates Action (e.g., "create k8s cluster")     │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ Execution Agent → API Executor Service                 │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ _ensure_valid_token() called                           │
│ • Check if cached token is still valid                 │
│ • If expired/missing → fetch new token                 │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼ (if token expired/missing)
┌─────────────────────────────────────────────────────────┐
│ _fetch_auth_token()                                    │
│ POST https://ipcloud.tatacommunications.com/           │
│      portalservice/api/v1/getAuthToken                 │
│ Body: {                                                │
│   "email": "izo_cloud_admin@...",                      │
│   "password": "Tata@1234"                              │
│ }                                                      │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ Token cached with expiry time (1 hour from now)       │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ _make_api_call()                                       │
│ Authorization: Bearer {fresh_token}                    │
│ POST https://ipcloud.tatacommunications.com/          │
│      paasservice/api/v1/iks                           │
└─────────────────────────────────────────────────────────┘
```

### 3. **Token Validation**
- Tokens are checked before expiry (5-minute buffer)
- If token expires in < 5 minutes → refresh immediately
- Prevents API calls from failing due to expired tokens

## 📝 Code Flow

### File: `app/services/api_executor_service.py`

#### Key Methods:

1. **`_fetch_auth_token()`** (Lines ~70-109)
   - Makes POST request to auth API
   - Extracts token from response
   - Returns bearer token string

2. **`_ensure_valid_token()`** (Lines ~111-140)
   - Checks token validity with 5-minute buffer
   - Fetches new token if expired
   - Thread-safe with `asyncio.Lock`

3. **`_make_api_call()`** (Lines ~431+)
   - Calls `_ensure_valid_token()` first
   - Uses `self.auth_token` instead of env variable
   - Automatically includes fresh token in Authorization header

## 🧪 Testing

### Test Token Fetch Manually:

```bash
curl -X POST https://ipcloud.tatacommunications.com/portalservice/api/v1/getAuthToken \
  -H "Content-Type: application/json" \
  -d '{
    "email": "izo_cloud_admin@tatacommunications.onmicrosoft.com",
    "password": "Tata@1234"
  }'
```

**Expected Response:**
```json
{
    "statusCode": 200,
    "accessToken": "eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICI..."
}
```

### Test Through Agent System:

```bash
curl -X POST http://localhost:8000/api/agent/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "create k8s cluster named test-cluster",
    "user_id": "admin",
    "user_roles": ["admin"]
  }'
```

Watch the logs for:
```
🔑 Fetching auth token from https://ipcloud.tatacommunications.com/...
✅ Successfully fetched auth token
✅ Auth token refreshed successfully
✅ Using dynamically fetched auth token
🌐 API Call: POST https://ipcloud.tatacommunications.com/paasservice/api/v1/iks
```

## 🔐 Security Notes

1. **Never commit credentials to git**
   - Use `.env` file (already in `.gitignore`)
   - Use environment variables in production

2. **Token Storage**
   - Token stored in memory only (not persisted to disk)
   - Cleared when service restarts
   - Automatically refreshed on expiry

3. **Credential Rotation**
   - Update `API_AUTH_EMAIL` and `API_AUTH_PASSWORD` in `.env`
   - Restart service to apply changes
   - No code changes needed

## 🎯 Token Response Format

The Tata Communications auth API returns:

```json
{
    "statusCode": 200,
    "accessToken": "eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICI..."
}
```

The service extracts the token from the `accessToken` field. It also supports fallback to:
- `access_token`
- `token`
- `authToken`

**Token Format**: JWT (JSON Web Token) with 3 parts separated by dots:
- Header (algorithm, type)
- Payload (user info, expiry, permissions)
- Signature (cryptographic verification)

**Example Token** (truncated):
```
eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3NjM3MDUxNTksImlhdCI...
```

## ⏱️ Token Expiry Configuration

Default: **1 hour** (configurable in code)

To change, edit line ~136 in `api_executor_service.py`:

```python
# Current: 1 hour
self.token_expires_at = datetime.utcnow() + timedelta(hours=1)

# Change to 30 minutes:
self.token_expires_at = datetime.utcnow() + timedelta(minutes=30)

# Change to 2 hours:
self.token_expires_at = datetime.utcnow() + timedelta(hours=2)
```

## 📊 Logging

Token operations are logged at different levels:

| Level | Message | When |
|-------|---------|------|
| INFO | `🔑 Fetching auth token` | New token being fetched |
| INFO | `✅ Successfully fetched auth token` | Token fetch succeeded |
| INFO | `✅ Auth token refreshed successfully` | Token cached and ready |
| DEBUG | `✅ Using cached auth token` | Using existing valid token |
| ERROR | `❌ Auth API returned error` | Auth API call failed |
| ERROR | `❌ Failed to fetch auth token` | Network/parsing error |
| WARNING | `⚠️ No auth token available` | Missing credentials |

## 🚀 Deployment Checklist

- [ ] Set `API_AUTH_EMAIL` in environment
- [ ] Set `API_AUTH_PASSWORD` in environment
- [ ] Verify `API_AUTH_URL` is correct
- [ ] Test token fetch with credentials
- [ ] Monitor logs for successful token refresh
- [ ] Test API call with auto-refreshed token

## 🔍 Troubleshooting

### Problem: "API_AUTH_EMAIL or API_AUTH_PASSWORD not configured"
**Solution**: Add credentials to `.env` file

### Problem: "Token not found in response"
**Solution**: Check auth API response format and update token extraction logic

### Problem: "Auth API returned error 401"
**Solution**: Verify email and password are correct

### Problem: "Failed to refresh auth token"
**Solution**: 
1. Check network connectivity to auth API
2. Verify auth URL is correct
3. Test credentials manually with curl

---

## ✅ Migration from Static Token

**Before** (static token in .env):
```bash
API_AUTH_TOKEN=static_bearer_token_here
```

**After** (dynamic token):
```bash
API_AUTH_URL=https://ipcloud.tatacommunications.com/portalservice/api/v1/getAuthToken
API_AUTH_EMAIL=izo_cloud_admin@tatacommunications.onmicrosoft.com
API_AUTH_PASSWORD=Tata@1234
```

**Changes**:
- ✅ Tokens refresh automatically
- ✅ No manual token updates needed
- ✅ Tokens never expire mid-operation
- ✅ Thread-safe concurrent operations

---

**Status**: ✅ **Production Ready**

Last Updated: 2025-11-10

