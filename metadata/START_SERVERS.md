# Quick Start Guide - Running Servers

## ✅ **Currently Running**

```bash
✅ Port 8001: User Backend (user_main.py) - RUNNING
```

## 🎯 **What You Can Test Right Now**

### **Option 1: Use the Widget**
Open your browser: **http://localhost:4201**

Try these queries in the chat widget:
- "list the clusters that are available"
- "show me all k8s clusters"  
- "what clusters do we have?"
- "list clusters in Mumbai"

**Expected Result:** Beautiful formatted list of 63 clusters across 5 data centers!

### **Option 2: Direct API Test**
```bash
curl -X POST http://localhost:8001/api/chat/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "list all clusters",
    "max_results": 5,
    "include_images": false
  }'
```

## 🚀 **Starting Servers Manually**

### **User Backend (Port 8001) - For Widget**
```bash
cd /home/unixlogin/vayuMaya/Enterprise-Rag-bot
source .venv/bin/activate

# Start user backend
python -m uvicorn app.user_main:app --host 0.0.0.0 --port 8001
```

**Or in background:**
```bash
nohup python -m uvicorn app.user_main:app --host 0.0.0.0 --port 8001 > /tmp/user_main.log 2>&1 &
```

### **Admin Backend (Port 8000) - For Admin/API**
```bash
cd /home/unixlogin/vayuMaya/Enterprise-Rag-bot
source .venv/bin/activate

# Start admin backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Or in background:**
```bash
nohup python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 > /tmp/main.log 2>&1 &
```

## 🛑 **Stopping Servers**

### **Stop All**
```bash
sudo pkill -9 -f "uvicorn.*app"
```

### **Stop Specific**
```bash
# Stop user backend (8001)
sudo lsof -ti:8001 | xargs sudo kill -9

# Stop admin backend (8000)
sudo lsof -ti:8000 | xargs sudo kill -9
```

## 🔍 **Checking Status**

### **See Running Servers**
```bash
ps aux | grep uvicorn | grep -v grep
```

### **Check Ports**
```bash
sudo lsof -i:8000
sudo lsof -i:8001
sudo lsof -i:4201
```

### **View Logs**
```bash
# User backend logs
tail -f /tmp/user_main.log

# Admin backend logs  
tail -f /tmp/main.log
```

## 📊 **Health Checks**

### **User Backend (8001)**
```bash
curl http://localhost:8001/health
```

**Expected:**
```json
{
  "status": "healthy",
  "service": "user_chat",
  "documents_available": 170,
  "version": "1.0.0"
}
```

### **Admin Backend (8000)**
```bash
curl http://localhost:8000/health
```

**Expected:**
```json
{
  "status": "healthy",
  "services": {
    "milvus": {"status": "active"},
    "ai_services": {"embedding": "operational"}
  }
}
```

## 🎯 **Architecture**

```
┌─────────────────┐
│  Widget (4201)  │  ← User Interface
└────────┬────────┘
         │
         ↓ /api/chat/query
┌─────────────────┐
│ User API (8001) │  ← user_main.py (CURRENTLY RUNNING ✅)
└────────┬────────┘
         │
         ↓ widget_query()
┌─────────────────┐
│ Main API (8000) │  ← app.main.py (Optional for admin)
└─────────────────┘
```

## 🐛 **Troubleshooting**

### **Port Already in Use**
```bash
# Kill process on port
sudo lsof -ti:8001 | xargs sudo kill -9

# Then restart
python -m uvicorn app.user_main:app --host 0.0.0.0 --port 8001
```

### **Module Import Errors**
```bash
# Make sure venv is activated
source .venv/bin/activate

# Check Python version
python --version  # Should be 3.10+

# Reinstall if needed
pip install -r requirements.txt
```

### **Permission Denied on .venv**
```bash
sudo chown -R unixlogin:users .venv/
```

### **Widget Shows RAG Response Instead of Clusters**
1. Check user backend is running: `curl http://localhost:8001/health`
2. Check widget is calling correct URL (should be :8001 not :8000)
3. Restart user backend: `sudo lsof -ti:8001 | xargs sudo kill -9 && <start command>`

## ✅ **Verification Checklist**

Before using the widget:
- [ ] User backend running on port 8001
- [ ] Health check returns "healthy"
- [ ] Direct API test returns cluster data
- [ ] Widget loads at http://localhost:4201
- [ ] Environment variables set (GROK_API_KEY, API_AUTH_EMAIL, API_AUTH_PASSWORD)

## 🎉 **Success!**

When everything is working, you should see:
- ✅ Widget responds instantly
- ✅ Shows 63 Kubernetes clusters
- ✅ Grouped by 5 data centers
- ✅ Includes node counts and K8s versions
- ✅ Formatted with emojis and structure

---

**Need Help?** Check:
- `ARCHITECTURE_AND_FLOW.md` - Complete architecture
- `CLUSTER_LISTING_GUIDE.md` - User guide
- `WIDGET_INTEGRATION_STATUS.md` - Integration status

