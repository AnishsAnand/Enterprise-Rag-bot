# ✅ OpenWebUI Setup Complete - Models Available!

## Issue Fixed

**Problem**: "No results found" when searching for models  
**Cause**: OpenWebUI was trying to connect to `http://enterprise-rag-bot:8001` (Docker network) but backend was running on host  
**Solution**: Updated OpenWebUI to use `http://host.docker.internal:8001/api/v1`

## Current Status

| Component | Status | Details |
|-----------|--------|---------|
| Backend | ✅ Running | Port 8001 |
| OpenWebUI | ✅ Running | Port 3000 |
| Models Available | ✅ Yes | 2 models |
| Connection | ✅ Working | host.docker.internal |

## Available Models

1. **Vayu Maya** - Main AI assistant
2. **Vayu Maya v2** - Alternative version

## How to Use

### Step 1: Access OpenWebUI

Open your browser: **http://localhost:3000/**

### Step 2: Login/Signup

- **First time?** Create an account (first user becomes admin)
- **Returning?** Login with your credentials

### Step 3: Select Model

1. Click **"Select a model"** dropdown (top left)
2. You should now see:
   - ✅ Vayu Maya
   - ✅ Vayu Maya v2
3. Select **Vayu Maya**

### Step 4: Start Chatting!

Try these prompts:
- "What can you help me with?"
- "Show me available cloud resources"
- "How do I create a cluster?"

## Access Levels (Based on Tata Auth API)

### Tata Communications Users:
- ✅ Full access to all features
- ✅ Can perform actions (create clusters, deploy, etc.)
- ✅ Access to RAG documentation

### Public Users:
- ✅ Access to RAG documentation
- ✅ Can ask questions and learn
- ❌ Cannot perform actions
- 💡 See enrollment message if they try actions

## Troubleshooting

### If models still don't appear:

1. **Refresh the page** (Ctrl+F5 or Cmd+Shift+R)

2. **Check backend is running**:
   ```bash
   curl http://localhost:8001/api/v1/models
   # Should return: {"object":"list","data":[...]}
   ```

3. **Check OpenWebUI logs**:
   ```bash
   docker logs enterprise-rag-openwebui
   ```

4. **Restart OpenWebUI**:
   ```bash
   docker restart enterprise-rag-openwebui
   ```

### If you need to reconfigure:

```bash
# Stop and remove current container
docker stop enterprise-rag-openwebui
docker rm enterprise-rag-openwebui

# Start with new configuration
docker run -d \
  --name enterprise-rag-openwebui \
  -p 3000:8080 \
  -e OPENAI_API_BASE_URL=http://host.docker.internal:8001/api/v1 \
  -e OPENAI_API_KEY=not-needed \
  -e ENABLE_OLLAMA_API=false \
  -e ENABLE_OPENAI_API=true \
  -e WEBUI_AUTH=true \
  -e WEBUI_NAME="Vayu Maya - AI Cloud Assistant" \
  -e DEFAULT_USER_ROLE=user \
  -v open-webui-data:/app/backend/data \
  --add-host=host.docker.internal:host-gateway \
  ghcr.io/open-webui/open-webui:main
```

## Configuration Details

### OpenWebUI Environment Variables:

| Variable | Value | Purpose |
|----------|-------|---------|
| OPENAI_API_BASE_URL | http://host.docker.internal:8001/api/v1 | Connect to backend |
| OPENAI_API_KEY | not-needed | Placeholder (not used) |
| ENABLE_OPENAI_API | true | Enable OpenAI-compatible API |
| ENABLE_OLLAMA_API | false | Disable Ollama |
| WEBUI_AUTH | true | Enable authentication |
| DEFAULT_USER_ROLE | user | Default role for new users |
| WEBUI_NAME | Vayu Maya - AI Cloud Assistant | Custom branding |

### Backend Configuration:

- **Port**: 8001
- **API Endpoint**: `/api/v1`
- **Models Endpoint**: `/api/v1/models`
- **Chat Endpoint**: `/api/v1/chat/completions`

## Testing Connection

### From Host:
```bash
curl http://localhost:8001/api/v1/models
```

### From Docker Container:
```bash
docker exec enterprise-rag-openwebui curl http://host.docker.internal:8001/api/v1/models
```

Both should return:
```json
{
  "object": "list",
  "data": [
    {"id": "Vayu Maya", "object": "model", ...},
    {"id": "Vayu Maya v2", "object": "model", ...}
  ]
}
```

## Admin Settings

### To Enable Signup for Other Users:

1. Login as admin (first user)
2. Go to **Admin Panel** (gear icon)
3. Navigate to **Settings** → **General**
4. Enable **"Enable Signup"**
5. Save changes

### To Customize Interface:

1. Go to **Admin Panel** → **Settings** → **Interface**
2. Customize:
   - App name
   - Logo
   - Colors
   - Landing page text
3. Save changes

See `CUSTOMIZE_OPENWEBUI.md` for detailed customization guide.

## Quick Commands

```bash
# Check if backend is running
ps aux | grep "uvicorn.*8001"

# Check if OpenWebUI is running
docker ps | grep openwebui

# View backend logs
tail -f /tmp/user_main.log

# View OpenWebUI logs
docker logs -f enterprise-rag-openwebui

# Restart OpenWebUI
docker restart enterprise-rag-openwebui

# Restart backend
cd /home/unixlogin/Vayu/Enterprise-Rag-bot
sudo lsof -ti:8001 | xargs sudo kill -9
source .venv/bin/activate
nohup python -m uvicorn app.user_main:app --host 0.0.0.0 --port 8001 > /tmp/user_main.log 2>&1 &
```

## Summary

✅ **Backend**: Running on port 8001  
✅ **OpenWebUI**: Running on port 3000  
✅ **Models**: 2 models available (Vayu Maya, Vayu Maya v2)  
✅ **Connection**: Working via host.docker.internal  
✅ **Authentication**: Enabled with Tata Auth API integration  
✅ **Access Control**: Two-tier (full vs read-only)  

**You're all set! Go to http://localhost:3000/ and start chatting! 🚀**

