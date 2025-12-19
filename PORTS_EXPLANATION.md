# OpenWebUI Ports Explanation

## Why Multiple Ports?

You're seeing multiple ports because of different attempts to fix the authentication issue:

### Port History:

1. **Port 3000** - Original docker-compose configuration
   - Defined in `misc/docker/docker-compose.openwebui.yml`
   - Maps container port 8080 to host port 3000
   - `ports: - "3000:8080"`

2. **Port 57695** - Your current running instance
   - This was the port you were using when you reported the issue
   - Likely started manually or from a different configuration

3. **Port 8080** - My temporary fix attempt
   - I used `--network host` which exposed the container's internal port 8080 directly
   - This was to quickly test if authentication was working

## Current Situation

Right now, OpenWebUI should be running on **port 3000** (the standard configuration).

## Recommended: Use Port 3000

```bash
# Access OpenWebUI at:
http://localhost:3000/

# This is the standard port configured in docker-compose
```

## Why the Login Failed (Screenshot Issue)

From the logs, I can see:
```
authenticate_user: cdemopoc@gmail.com
POST /api/v1/auths/signin HTTP/1.1 400
```

The login failed with HTTP 400 because:
1. **User doesn't exist yet** - You need to create an account first
2. **Wrong credentials** - If the user exists, the password is incorrect

## Solution

### Option 1: Create a New Account (Recommended)
1. Go to `http://localhost:3000/`
2. Click "Sign Up" or "Create Account"
3. Enter:
   - Email: `cdemopoc@gmail.com`
   - Password: `Ipc@1234` (or your preferred password)
   - Name: Your Name
4. Click "Create Account"
5. **First user becomes admin automatically**

### Option 2: Use Existing Account
If you already created an account with different credentials:
1. Go to `http://localhost:3000/`
2. Use the email/password you registered with
3. Example from logs: `kathir.m1@tatacommunications.com` successfully logged in

## Port Cleanup

To avoid confusion, let's standardize on **port 3000**:

```bash
# Stop any running OpenWebUI containers
docker stop enterprise-rag-openwebui
docker rm enterprise-rag-openwebui

# Start OpenWebUI on port 3000 (standard)
docker run -d \
  --name enterprise-rag-openwebui \
  -p 3000:8080 \
  -e WEBUI_AUTH=true \
  -e WEBUI_SECRET_KEY="secure-webui-secret-key-for-sessions-2024" \
  -e OPENAI_API_BASE_URL="http://host.docker.internal:8001/api/v1" \
  -e OPENAI_API_KEY="secure-openwebui-api-key-2024" \
  -e ENABLE_RAG=true \
  -e ENABLE_OLLAMA_API=false \
  -e ENABLE_OPENAI_API=true \
  -e DEFAULT_MODELS="Vayu Maya" \
  -e WEBUI_NAME="Vayu Maya - AI Cloud Assistant" \
  --add-host=host.docker.internal:host-gateway \
  -v openwebui-data:/app/backend/data \
  ghcr.io/open-webui/open-webui:main
```

## Summary

| Port | Purpose | Status |
|------|---------|--------|
| 3000 | Standard OpenWebUI port | ✅ Recommended |
| 8080 | Container internal port | ⚠️ Only with --network host |
| 57695 | Your previous instance | ❌ Should be stopped |

## Next Steps

1. **Access**: `http://localhost:3000/`
2. **Create Account**: Sign up with your email
3. **Test Login**: Use the credentials you just created
4. **Test Logout**: Should work properly now with `WEBUI_AUTH=true`

---

**Note**: The authentication error you saw was because you were trying to log in with credentials that don't exist in OpenWebUI's database. You need to create an account first!

