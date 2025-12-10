# 🚀 Quick Setup: Dynamic Token Authentication

## ✅ What Was Implemented

Your API Executor Service now **automatically fetches and refreshes** authentication tokens before making any API calls to the Tata Communications platform.

## 📝 Setup Instructions (3 Steps)

### Step 1: Add Environment Variables

Add these lines to your `.env` file:

```bash
# API Authentication (Dynamic Token)
API_AUTH_URL=https://ipcloud.tatacommunications.com/portalservice/api/v1/getAuthToken
API_AUTH_EMAIL=izo_cloud_admin@tatacommunications.onmicrosoft.com
API_AUTH_PASSWORD=Tata@1234
```

**Note**: The old `API_AUTH_TOKEN` variable is NO LONGER NEEDED. Remove it if present.

### Step 2: Test Token Fetch

```bash
cd /home/unixlogin/vayuMaya/Enterprise-Rag-bot
source .venv/bin/activate
python test_token_auth.py
```

**Expected Output**:
```
✅ SUCCESS! Token fetched successfully
✅ Token validation successful
✅ Token caching working (same token used)
🎉 All tests PASSED!
```

### Step 3: Restart Your Application

```bash
# Kill existing server
sudo pkill -9 -f uvicorn

# Start fresh
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 🎯 Verification

### Test End-to-End:

```bash
curl -X POST http://localhost:8000/api/agent/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "create k8s cluster named test-cluster",
    "user_id": "admin",
    "user_roles": ["admin"]
  }'
```

### Check Logs for Token Activity:

```bash
tail -f /tmp/ragbot_new.log | grep -E "(🔑|✅|❌)" | grep -i token
```

You should see:
- `🔑 Fetching auth token`
- `✅ Successfully fetched auth token`
- `✅ Using dynamically fetched auth token`

## 🔄 How It Works

```
Before Every API Call:
┌────────────────────────────────────┐
│ 1. Check if token is expired       │
│    (with 5-minute safety buffer)   │
└────────────┬───────────────────────┘
             │
             ▼ (if expired)
┌────────────────────────────────────┐
│ 2. Fetch new token from:           │
│    POST /portalservice/.../        │
│         getAuthToken               │
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│ 3. Cache token for 1 hour          │
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│ 4. Make API call with fresh token  │
│    Authorization: Bearer {token}   │
└────────────────────────────────────┘
```

## 📊 Key Features

✅ **Automatic Refresh**: Tokens refresh before expiry (no failed API calls)  
✅ **Thread-Safe**: Concurrent requests won't cause duplicate token fetches  
✅ **Caching**: Tokens cached for 1 hour (configurable)  
✅ **Error Handling**: Graceful fallback if token fetch fails  
✅ **Logging**: Full visibility into token lifecycle  

## 🔐 Security

- ✅ Credentials stored in `.env` (gitignored)
- ✅ Tokens stored in memory only (cleared on restart)
- ✅ No hardcoded credentials in code
- ✅ Bearer tokens included in every API request

## 📁 Files Modified

| File | Changes |
|------|---------|
| `app/services/api_executor_service.py` | Added token management logic |
| `TOKEN_AUTH_SETUP.md` | Comprehensive documentation |
| `test_token_auth.py` | Testing script |

## 🐛 Troubleshooting

### Problem: "API_AUTH_EMAIL or API_AUTH_PASSWORD not configured"

**Fix**: Add credentials to `.env`:
```bash
echo 'API_AUTH_EMAIL=izo_cloud_admin@tatacommunications.onmicrosoft.com' >> .env
echo 'API_AUTH_PASSWORD=Tata@1234' >> .env
```

### Problem: Test shows "FAILED to fetch token"

**Check**:
1. Network connectivity: `ping ipcloud.tatacommunications.com`
2. Credentials are correct
3. Auth URL is accessible

### Problem: API calls still fail with 401

**Check logs for**:
- `✅ Successfully fetched auth token` (token fetch worked)
- `❌ Auth API returned error` (credentials wrong)
- `⚠️ No auth token available` (token fetch disabled)

## 🎓 Usage Examples

### Example 1: Create Cluster (with auto token refresh)

```bash
# The system will automatically:
# 1. Check token validity
# 2. Refresh if needed
# 3. Make API call with fresh token

curl -X POST http://localhost:8000/api/agent/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "create k8s cluster",
    "user_id": "admin",
    "user_roles": ["admin"]
  }'
```

### Example 2: Manual Token Test

```bash
cd /home/unixlogin/vayuMaya/Enterprise-Rag-bot
source .venv/bin/activate
python -c "
import asyncio
from app.services.api_executor_service import api_executor_service

async def test():
    token = await api_executor_service._fetch_auth_token()
    print(f'Token: {token[:50]}...' if token else 'Failed')

asyncio.run(test())
"
```

## ✨ Benefits Over Static Token

| Feature | Static Token | Dynamic Token |
|---------|--------------|---------------|
| Manual updates | ❌ Required | ✅ Automatic |
| Expiry handling | ❌ Manual refresh | ✅ Auto-refresh |
| Concurrent safety | ⚠️ Race conditions | ✅ Thread-safe |
| Security | ⚠️ Long-lived tokens | ✅ Short-lived |
| Maintenance | ❌ High | ✅ Zero |

## 📞 Support

- **Full Documentation**: See `TOKEN_AUTH_SETUP.md`
- **Test Script**: Run `python test_token_auth.py`
- **Logs**: Check `/tmp/ragbot_new.log` for token activity

---

**Status**: ✅ **Ready to Use**

Last Updated: 2025-11-10

