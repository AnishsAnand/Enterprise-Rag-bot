# ✅ Token Authentication - Verification Guide

## 🎯 Actual API Response Format

The Tata Communications auth API returns:

```json
{
    "statusCode": 200,
    "accessToken": "eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICIzZzBSOUo3X0VWVWtsdEY4V2FUZ3kyMXZLZ1pHckg2QWJ0c3ZfbjVfcVpjIn0.eyJleHAiOjE3NjM3MDUxNTksImlhdCI6MTc2MzcwNDU1OSwianRpIjoiMjBiYmJhMzQtM2UwYi00NzYwLWJiMzgtZjdmNTNjZTI1ZTdkIiwiaXNzIjoiaHR0cHM6Ly9pZHAudGF0YWNvbW11bmljYXRpb25zLmNvbS9hdXRoL3JlYWxtcy9tYXN0ZXIiLCJhdWQiOiJhY2NvdW50Iiwic3ViIjoiOWM1OGY2NTYtMzhjMS00YjUyLTllMjYtZmY2MzAxMjRkNjQwIiwidHlwIjoiQmVhcmVyIiwiYXpwIjoib3B0aW11cyIsInNlc3Npb25fc3RhdGUiOiJiZmUyNzYyMS1iYjE4LTQ2MDMtODEwMS1kM2JkMjUwZTliYzUiLCJyZWFsbV9hY2Nlc3MiOnsicm9sZXMiOlsiZGVmYXVsdC1yb2xlcy1tYXN0ZXIiLCJvZmZsaW5lX2FjY2VzcyIsInVtYV9hdXRob3JpemF0aW9uIl19LCJyZXNvdXJjZV9hY2Nlc3MiOnsiYWNjb3VudCI6eyJyb2xlcyI6WyJtYW5hZ2UtYWNjb3VudCIsIm1hbmFnZS1hY2NvdW50LWxpbmtzIiwidmlldy1wcm9maWxlIl19fSwic2NvcGUiOiJwcm9maWxlIGVtYWlsIiwic2lkIjoiYmZlMjc2MjEtYmIxOC00NjAzLTgxMDEtZDNiZDI1MGU5YmM1IiwiZW1haWxfdmVyaWZpZWQiOnRydWUsIm5hbWUiOiJJWk9DbG91ZCBBZG1pbiIsInByZWZlcnJlZF91c2VybmFtZSI6Iml6b19jbG91ZF9hZG1pbkB0YXRhY29tbXVuaWNhdGlvbnMub25taWNyb3NvZnQuY29tIiwiZ2l2ZW5fbmFtZSI6IklaT0Nsb3VkIiwiZmFtaWx5X25hbWUiOiJBZG1pbiIsImVtYWlsIjoiaXpvX2Nsb3VkX2FkbWluQHRhdGFjb21tdW5pY2F0aW9ucy5vbm1pY3Jvc29mdC5jb20ifQ.id4Oon6JYSRNGhrQkTdEzCVbg-aJb4bLowLzRoytl8_-FS5HGA14HAEOzW9AG9zAg75Hgx6IG7jVW3KEJ19lOs5OFZExNunYghmMVBklLCm-L9Xwc8i_V_fTSYUAf450e72zWF-aq-4FuCUw7Xqyx-yzEoFdFk3Y6DnrJ0EonlwhDzO248DQLZF8Z1nVG_T7h4Tro0b_fTWlp1-odwxAwo8NXYqTQRxaL2ZJv8usmEncOOu67VublSjLvN5Csm3Vg-v9TUoAJ8P3OlmcgUdWw3UpjiF44xMGqGlWGRfxq2rlno3op6VV7vfTwFMtoqoCKbpbpMKGB32CK63377zAfw"
}
```

## 🔍 Token Details

### Token Structure (JWT)

The token is a **JSON Web Token (JWT)** with 3 parts separated by dots (`.`):

```
eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9     ← Header
.
eyJleHAiOjE3NjM3MDUxNTksImlhdCI...      ← Payload (user data)
.
id4Oon6JYSRNGhrQkTdEzCVbg-aJb4b...        ← Signature
```

### Decoded Payload

The token contains:

| Field | Value | Meaning |
|-------|-------|---------|
| `exp` | 1763705159 | Expires: Thu Nov 10 2025 (10 minutes from issue) |
| `iat` | 1763704559 | Issued at: Thu Nov 10 2025 |
| `email` | `izo_cloud_admin@tata...` | User email |
| `name` | "IZOCloud Admin" | User name |
| `typ` | "Bearer" | Token type |

**Token Expiry**: **10 minutes** (not 1 hour as initially configured)

## ⚠️ Important: Token Expiry Adjustment Needed

The actual token expires in **10 minutes**, not 1 hour. We need to adjust the code!

### Current Setting (WRONG):
```python
# Line ~136 in api_executor_service.py
self.token_expires_at = datetime.utcnow() + timedelta(hours=1)  # ❌ TOO LONG
```

### Correct Setting (NEEDS UPDATE):
```python
# Should be:
self.token_expires_at = datetime.utcnow() + timedelta(minutes=10)  # ✅ CORRECT
# Or be conservative:
self.token_expires_at = datetime.utcnow() + timedelta(minutes=8)   # ✅ SAFE (2-min buffer)
```

## 🔧 Quick Fix Required

Update the token expiry in `app/services/api_executor_service.py`:

```python
# Find line ~136:
self.token_expires_at = datetime.utcnow() + timedelta(hours=1)

# Change to:
self.token_expires_at = datetime.utcnow() + timedelta(minutes=8)  # 8 minutes (safe with 2-min buffer)
```

**Why 8 minutes?**
- Actual token expiry: 10 minutes
- Safety buffer: 2 minutes
- Refresh trigger: 5 minutes before expiry (built-in)
- Effective refresh: At 3 minutes remaining

## 🧪 Verification Steps

### Step 1: Manual Token Fetch

```bash
curl -X POST https://ipcloud.tatacommunications.com/portalservice/api/v1/getAuthToken \
  -H "Content-Type: application/json" \
  -d '{
    "email": "izo_cloud_admin@tatacommunications.onmicrosoft.com",
    "password": "Tata@1234"
  }' | jq .
```

**Expected**:
```json
{
  "statusCode": 200,
  "accessToken": "eyJhbGci..."
}
```

### Step 2: Decode Token (Optional)

Visit: https://jwt.io

Paste the `accessToken` value to see:
- Header: Algorithm and token type
- Payload: User info, expiry time, permissions
- Signature: Cryptographic verification

### Step 3: Run Test Script

```bash
cd /home/unixlogin/vayuMaya/Enterprise-Rag-bot
source .venv/bin/activate
python test_token_auth.py
```

**Expected Output**:
```
🔑 Fetching auth token from https://ipcloud.tatacommunications.com/...
✅ Successfully fetched auth token (token length: 1234)
✅ Token format: Valid JWT (3 parts)
   - Header length: 123
   - Payload length: 789
   - Signature length: 234
✅ Token validation successful
Token cached until: 2025-11-10 13:22:00
Token will be refreshed in: ~8.0 minutes
🎉 All tests PASSED!
```

## 📊 Token Lifecycle (Corrected)

| Time | Status | Action |
|------|--------|--------|
| T+0m | 🔑 Token fetched | Fresh token obtained |
| T+3m | ✅ Valid | Using cached token |
| T+5m | 🔄 Refresh triggered (5-min buffer) | Auto-refresh starts |
| T+6m | ✅ New token cached | Using new token |
| T+8m | ⚠️ Approaching expiry | Would have expired at T+10m |

## 🔐 Code Changes Required

### File: `app/services/api_executor_service.py`

**Line ~136 - Update token expiry:**

```python
# BEFORE:
self.token_expires_at = datetime.utcnow() + timedelta(hours=1)

# AFTER:
self.token_expires_at = datetime.utcnow() + timedelta(minutes=8)
```

**Line ~103 - Token extraction (ALREADY CORRECT):**

```python
# Correctly extracts accessToken
token = (
    data.get("accessToken") or      # ✅ Primary (Tata API)
    data.get("access_token") or     # ✅ Fallback
    data.get("token") or            # ✅ Fallback
    data.get("authToken")           # ✅ Fallback
)
```

## ✅ Verification Checklist

- [ ] Updated token expiry to 8 minutes (line ~136)
- [ ] Code extracts `accessToken` from response (line ~103)
- [ ] Test script runs successfully
- [ ] Manual curl test returns valid token
- [ ] Server logs show token refresh every ~3-5 minutes
- [ ] API calls include `Authorization: Bearer {token}` header
- [ ] No authentication errors in production

## 🚨 Critical Note

**Token expiry is 10 minutes, NOT 1 hour!**

If you don't update the code, tokens will be cached for 1 hour but expire after 10 minutes, causing API calls to fail with 401 errors after 10 minutes.

**Action Required**: Update `timedelta(hours=1)` to `timedelta(minutes=8)`

---

**Status**: ⚠️ **Requires Code Update**

**Priority**: **HIGH** - Fix token expiry setting

**Files to Update**: 
1. `app/services/api_executor_service.py` (line ~136)

---

Last Updated: 2025-11-10

