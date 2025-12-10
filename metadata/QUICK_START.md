# Quick Start Guide - Multi-Agent System

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies (5 minutes)

```bash
cd /home/unixlogin/vayuMaya/Enterprise-Rag-bot

# Fix virtual environment permissions
sudo chown -R $USER:$USER .venv/

# Activate and install
source .venv/bin/activate
pip install langchain==0.1.0 langchain-openai==0.0.2 langchain-community==0.0.10 langgraph==0.0.20

# Verify installation
python test_agent_system.py
```

### Step 2: Configure API Endpoints (30 minutes)

Edit `app/config/resource_schema.json` with your actual APIs:

```json
{
  "resources": {
    "k8s_cluster": {
      "operations": ["create", "update", "delete", "list"],
      "api_endpoints": {
        "create": {
          "method": "POST",
          "url": "https://YOUR-ACTUAL-API.com/v1/clusters"
        }
      },
      "parameters": {
        "create": {
          "required": ["name", "region", "version"],
          "validation": {
            "name": {"type": "string", "min_length": 3}
          }
        }
      },
      "permissions": {
        "create": ["admin", "developer"]
      }
    }
  }
}
```

### Step 3: Start & Test (5 minutes)

```bash
# Add API token to .env
echo "API_AUTH_TOKEN=your-token-here" >> .env

# Start the application
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Test in another terminal
curl -X POST http://localhost:8000/api/agent/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Create a new Kubernetes cluster named prod-cluster",
    "user_id": "test_user",
    "user_roles": ["admin"]
  }'
```

---

## 📚 Key Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/agent/chat` | POST | Main conversational interface |
| `/api/agent/conversation/{id}` | GET | Get conversation status |
| `/api/agent/stats` | GET | System statistics |
| `/docs` | GET | Interactive API documentation |

---

## 💡 Example Conversations

### Create Resource
```
User: "Create a new Kubernetes cluster"
Bot: "I'll help you create a cluster. I need: Cluster Name, Region, Version..."
User: "Name it prod-01, version 1.28, in us-east-1"
Bot: "Perfect! Shall I proceed?"
User: "Yes"
Bot: "✅ Successfully created cluster prod-01!"
```

### Ask Question
```
User: "How do I scale a cluster?"
Bot: "Based on our documentation, here's how to scale..."
```

---

## 🐛 Troubleshooting

### "Module not found: langchain_openai"
```bash
source .venv/bin/activate
pip install langchain-openai langchain-community langgraph
```

### "Permission denied" during pip install
```bash
sudo chown -R $USER:$USER .venv/
```

### "API execution failed"
Check:
1. API endpoints in `resource_schema.json`
2. `API_AUTH_TOKEN` in `.env`
3. API service is accessible

---

## 📖 Full Documentation

- **Setup Guide**: `AGENT_SYSTEM_SETUP.md`
- **Implementation Summary**: `IMPLEMENTATION_SUMMARY.md`
- **Agent Documentation**: `app/agents/README.md`
- **API Docs**: http://localhost:8000/docs

---

## ✅ Checklist

- [ ] Install LangChain dependencies
- [ ] Configure resource schema with actual APIs
- [ ] Set API_AUTH_TOKEN in .env
- [ ] Run test_agent_system.py (all tests pass)
- [ ] Start application
- [ ] Test with curl
- [ ] Test with Postman/browser
- [ ] Deploy to production

---

**Ready to go!** 🎉

