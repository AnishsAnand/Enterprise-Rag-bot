# Quick Start: Open WebUI Integration

This guide gets you up and running with Open WebUI + Enterprise RAG Bot in under 15 minutes.

## 🚀 Quick Installation

### Prerequisites

- Docker and Docker Compose installed
- Your Enterprise RAG Bot running (or ready to run)
- 4GB+ RAM available

### Step 1: Setup Environment (2 minutes)

```bash
cd /home/unixlogin/vayuMaya/Enterprise-Rag-bot

# Create environment file
cp env.openwebui.template .env

# Generate secure keys
echo "WEBUI_SECRET_KEY=$(openssl rand -hex 32)" >> .env
echo "JWT_SECRET_KEY=$(openssl rand -hex 32)" >> .env

# Add your OpenAI key
nano .env  # Edit and add your OPENAI_API_KEY
```

### Step 2: Update Your Backend (3 minutes)

Add the OpenAI-compatible router to your main app:

```bash
# Edit app/main.py
nano app/main.py
```

Add these lines:

```python
# Import the router
from app.routers import openai_compatible

# Include the router (add this after your other routers)
app.include_router(openai_compatible.router)
```

### Step 3: Start Everything (5 minutes)

```bash
# Start all services with Open WebUI
docker-compose -f docker-compose.openwebui.yml up -d

# Check if services are running
docker-compose -f docker-compose.openwebui.yml ps

# View logs
docker-compose -f docker-compose.openwebui.yml logs -f open-webui
```

### Step 4: Access Open WebUI (5 minutes)

1. **Open your browser**: http://localhost:3000

2. **Create your account**:
   - Email: your-email@example.com
   - Password: (choose a strong password)
   - Name: Your Name

3. **Select Model**:
   - Click on the model dropdown
   - Select "enterprise-rag-bot"

4. **Start Chatting**:
   - Type: "Hello! Can you help me create a Kubernetes cluster?"
   - Watch your Enterprise RAG Bot respond through the beautiful Open WebUI interface!

## 🎯 What You Get

After following these steps, you'll have:

✅ **Modern Chat UI** - Professional interface like ChatGPT  
✅ **Your RAG Bot** - Running with full capabilities  
✅ **User Management** - Multi-user support with authentication  
✅ **Chat History** - All conversations saved automatically  
✅ **Document Upload** - RAG with file upload support  
✅ **Vector Database** - Milvus for semantic search  
✅ **Persistence** - PostgreSQL for data storage  

## 🧪 Test the Integration

### Test 1: Simple Chat

```
You: Hello!
Bot: Hello! I'm your Enterprise RAG Assistant. How can I help you today?
```

### Test 2: Cluster Creation

```
You: I need to create a new Kubernetes cluster
Bot: I'll help you create a cluster. Let me guide you through the process...
```

### Test 3: Document Search

1. Click the "+" icon
2. Upload a PDF or text file
3. Ask: "What does this document say about X?"

## 📊 Service URLs

| Service | URL | Purpose |
|---------|-----|---------|
| Open WebUI | http://localhost:3000 | Main chat interface |
| RAG Bot API | http://localhost:8000 | Backend API |
| API Docs | http://localhost:8000/docs | API documentation |
| MinIO Console | http://localhost:9001 | Object storage UI |
| PostgreSQL | localhost:5432 | Database |
| Redis | localhost:6379 | Cache |
| Milvus | localhost:19530 | Vector DB |

## 🔧 Troubleshooting

### "Service not found" error

```bash
# Restart services
docker-compose -f docker-compose.openwebui.yml restart
```

### "Cannot connect to backend"

```bash
# Check if RAG Bot is running
curl http://localhost:8000/api/v1/models

# Should return: {"object":"list","data":[...]}
```

### "Authentication failed"

```bash
# Check environment variables
docker-compose -f docker-compose.openwebui.yml config

# Verify OPENWEBUI_API_KEY is set
```

### View detailed logs

```bash
# All services
docker-compose -f docker-compose.openwebui.yml logs -f

# Specific service
docker-compose -f docker-compose.openwebui.yml logs -f open-webui
docker-compose -f docker-compose.openwebui.yml logs -f enterprise-rag-bot
```

## 🛑 Stop Services

```bash
# Stop all services
docker-compose -f docker-compose.openwebui.yml down

# Stop and remove volumes (CAUTION: deletes all data)
docker-compose -f docker-compose.openwebui.yml down -v
```

## 📚 Next Steps

1. **Customize the UI**: 
   - Admin Settings → Interface
   - Change name, logo, colors

2. **Add Users**:
   - Admin Settings → Users
   - Create accounts for your team

3. **Configure RAG**:
   - Settings → Documents
   - Upload your knowledge base

4. **Set up Analytics**:
   - Enable Langfuse in .env
   - Track usage and costs

5. **Production Deployment**:
   - Read OPENWEBUI_INTEGRATION.md
   - Set up SSL/TLS
   - Configure backups

## 🎉 Success Checklist

- [ ] Services running (check with `docker-compose ps`)
- [ ] Open WebUI accessible at http://localhost:3000
- [ ] Account created and logged in
- [ ] Model "enterprise-rag-bot" available
- [ ] Can send messages and get responses
- [ ] Chat history being saved
- [ ] Ready to add more users!

---

**Need Help?**  
Check the full integration guide: `OPENWEBUI_INTEGRATION.md`

**Enjoying Open WebUI?**  
⭐ Star the repo: https://github.com/open-webui/open-webui

