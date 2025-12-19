# ✅ Single Source of Truth - Tata Auth API Only

## The Simple Rule

**ONLY the Tata Auth API determines access level. Nothing else.**

```
Tata Auth API Response → Access Level
```

## How It Works

### Single Source of Truth:

```
User Login
    ↓
Call: https://ipcloud.tatacommunications.com/portalservice/api/v1/getAuthToken
    ↓
Response?
    ├─ statusCode: 200 + accessToken → ✅ FULL ACCESS
    └─ statusCode: 500 or error      → ⚠️ READ-ONLY ACCESS
```

### No Other Checks:
- ❌ No email domain checking
- ❌ No manual approval
- ❌ No SSO
- ✅ **ONLY Tata Auth API response matters**

## Access Levels

| Tata API Response | Access Level | Permissions |
|-------------------|--------------|-------------|
| **200 + token** | **Full** | ✅ Chat<br>✅ RAG docs<br>✅ Create clusters<br>✅ All actions |
| **500 or error** | **Read-Only** | ✅ Chat<br>✅ RAG docs<br>❌ No actions |

## Implementation

### 1. Login Flow

```python
# User logs in with email + password
POST /api/openwebui-auth/login
{
    "email": "user@example.com",
    "password": "password123"
}

# Backend calls Tata Auth API
response = call_tata_auth_api(email, password)

# Check response (SINGLE SOURCE OF TRUTH)
if response.statusCode == 200 and response.accessToken:
    # Valid Tata user
    access_level = "full"
    roles = ["admin", "developer", "viewer"]
else:
    # Not a Tata user
    access_level = "read_only"
    roles = ["viewer"]

# Return JWT with access level embedded
return {
    "token": jwt_token,
    "access_level": access_level,
    "message": "..."
}
```

### 2. Permission Check

```python
# When user tries to perform an action
user_roles = get_roles_from_token(jwt_token)

if "admin" in user_roles or "developer" in user_roles:
    # Full access - allow action
    perform_action()
else:
    # Read-only - deny action
    return {
        "error": "Unauthorized",
        "message": "You don't have permission to perform this action.",
        "enrollment_info": {
            "title": "Want to perform actions?",
            "description": "Sign in with your Tata Communications credentials for full access.",
            "contact": "support@tatacommunications.com"
        }
    }
```

## API Endpoints

### 1. Login with Tata Validation

```bash
POST /api/openwebui-auth/login
Content-Type: application/json

{
    "email": "user@example.com",
    "password": "password123"
}

# Response:
{
    "success": true,
    "token": "eyJhbGciOiJIUzI1NiIs...",
    "access_level": "full",  # or "read_only"
    "user_info": {
        "email": "user@example.com",
        "name": "User Name",
        "provider": "tata"  # or "local"
    },
    "message": "Authenticated as Tata Communications user. Full access granted."
}
```

### 2. Validate Existing Token

```bash
POST /api/openwebui-auth/validate-token
Content-Type: application/json

{
    "token": "eyJhbGciOiJIUzI1NiIs..."
}

# Response:
{
    "valid": true,
    "email": "user@example.com",
    "access_level": "full",
    "tata_validated": true,
    "roles": ["admin", "developer", "viewer"]
}
```

### 3. Direct Tata API Validation (for testing)

```bash
POST /api/tata-auth/validate
Content-Type: application/json

{
    "email": "user@tatacommunications.com",
    "password": "password123"
}

# Response:
{
    "success": true,
    "access_level": "full",
    "message": "Authenticated as Tata Communications user. Full access granted.",
    "user_info": {...},
    "token": "eyJhbGciOiJSUzI1NiIs..."  # Tata's token
}
```

## Testing

### Test 1: Valid Tata User

```bash
curl -X POST http://localhost:8001/api/openwebui-auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "izo_cloud_admin@tatacommunications.onmicrosoft.com",
    "password": "correct_password"
  }'

# Expected:
{
  "success": true,
  "access_level": "full",  ✅
  "message": "Authenticated as Tata Communications user. Full access granted."
}
```

### Test 2: Invalid/Non-Tata User

```bash
curl -X POST http://localhost:8001/api/openwebui-auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "someone@gmail.com",
    "password": "anypassword"
  }'

# Expected:
{
  "success": true,
  "access_level": "read_only",  ⚠️
  "message": "Not a Tata Communications user. Read-only access granted (RAG docs only, no actions)."
}
```

## User Experience

### Scenario 1: Tata Employee

1. **Opens OpenWebUI**: `http://localhost:3000/`
2. **Enters credentials**:
   - Email: `employee@tatacommunications.com`
   - Password: (Tata portal password)
3. **Backend calls Tata Auth API**
4. **API returns**: `statusCode: 200` + token
5. **Result**: ✅ **Full access granted**
6. **Can perform**: Create clusters, deploy resources, all actions

### Scenario 2: Public User

1. **Opens OpenWebUI**: `http://localhost:3000/`
2. **Enters credentials**:
   - Email: `user@gmail.com`
   - Password: (any password)
3. **Backend calls Tata Auth API**
4. **API returns**: `statusCode: 500` (invalid)
5. **Result**: ⚠️ **Read-only access granted**
6. **Can do**: Chat, read RAG docs
7. **Cannot do**: Perform actions
8. **If tries action**: Gets enrollment message

## Code Changes

### Files Modified:

1. **`app/routers/openai_compatible.py`** (line ~260)
   - Removed email domain check
   - Now checks `X-Tata-Validated` header only
   - Single source of truth

2. **`app/api/routes/openwebui_auth.py`** (NEW)
   - Custom login endpoint
   - Calls Tata Auth API
   - Returns JWT with access level

3. **`app/services/tata_auth_service.py`**
   - Service to call Tata Auth API
   - Returns validation result
   - Caches tokens

## Benefits

✅ **Simple**: One rule - Tata API response determines access
✅ **Secure**: Uses Tata's existing authentication
✅ **Automatic**: No manual checks or approvals
✅ **Fast**: Token caching for performance
✅ **Clear**: Easy to understand and maintain

## No More:

❌ Email domain checking
❌ Manual user approval
❌ Complex SSO setup
❌ Multiple sources of truth

## Just:

✅ **Tata Auth API response = Access level**

That's it!

## Monitoring

```bash
# Watch authentication logs
tail -f /tmp/user_main.log | grep -E "(Tata|Access)"

# You'll see:
# ✅ Tata Auth Success: user@tata.com | Full Access
# ℹ️ Non-Tata User: user@gmail.com | Read-Only Access
# 👤 Tata Validated User: user@tata.com | Full Access
# 👤 Regular User: user@gmail.com | Read-Only Access (Not Tata validated)
```

## Summary

| Component | Implementation |
|-----------|----------------|
| **Source of Truth** | Tata Auth API only |
| **Full Access** | API returns 200 + token |
| **Read-Only** | API returns 500 or error |
| **No Other Checks** | Email domain ignored |
| **Login Endpoint** | `/api/openwebui-auth/login` |
| **Validation** | `/api/tata-auth/validate` |

**Simple. Clear. One source of truth. Done! ✅**

