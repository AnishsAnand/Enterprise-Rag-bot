# Quick Start: Cluster Listing

## 🚀 Get Started in 3 Steps

### Step 1: Environment Setup

Make sure `.env` file has authentication credentials:

```bash
# Check if credentials are set
grep "API_AUTH_EMAIL" .env
grep "API_AUTH_PASSWORD" .env
```

If not present, add them:

```bash
echo "API_AUTH_EMAIL=your-email@example.com" >> .env
echo "API_AUTH_PASSWORD=your-password" >> .env
```

### Step 2: Start the Server

```bash
cd /home/unixlogin/vayuMaya/Enterprise-Rag-bot
source .venv/bin/activate
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Wait for: `✅ Application startup complete`

### Step 3: Test Cluster Listing

**Option A: Via Test Script**
```bash
python test_cluster_list.py
```

**Option B: Via Agent Chat API**
```bash
curl -X POST http://localhost:8000/api/agent/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Show me all clusters",
    "session_id": "test-user-123"
  }'
```

**Option C: Programmatic**
```python
from app.services.api_executor_service import api_executor_service
import asyncio

async def test():
    result = await api_executor_service.list_clusters()
    print(result)

asyncio.run(test())
```

## 🎯 Common User Queries

Try these natural language queries via the chat API:

```bash
# List all clusters
curl -X POST http://localhost:8000/api/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Show me all clusters", "session_id": "user-1"}'

# List clusters in specific location
curl -X POST http://localhost:8000/api/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "List clusters in Mumbai and Delhi", "session_id": "user-1"}'

# Get cluster count
curl -X POST http://localhost:8000/api/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How many k8s clusters do we have?", "session_id": "user-1"}'

# Check available locations
curl -X POST http://localhost:8000/api/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What data centers are available?", "session_id": "user-1"}'
```

## 📊 Expected Output

### Test Script Output
```
🚀 Starting Cluster Listing Workflow Tests
✅ Successfully fetched engagement ID: 1923
✅ Successfully fetched 5 endpoints
✅ Successfully fetched 63 clusters
✅ All tests PASSED!
```

### Agent Chat Response
```json
{
  "response": "I found 63 Kubernetes clusters across 5 data centers:\n\n📍 **Bengaluru**: 17 clusters\n📍 **Chennai-AMB**: 21 clusters (AI-enabled)\n📍 **Cressex**: 4 clusters\n📍 **Delhi**: 13 clusters\n📍 **Mumbai-BKC**: 8 clusters\n\nWould you like details about any specific location?",
  "session_id": "user-1",
  "agent": "execution"
}
```

## 🔧 Troubleshooting

### Problem: "API_AUTH_EMAIL not configured"

**Solution:**
```bash
# Set credentials in .env
echo "API_AUTH_EMAIL=your-email@example.com" >> .env
echo "API_AUTH_PASSWORD=your-password" >> .env
```

### Problem: "Failed to fetch engagement ID"

**Solution:** Check your credentials are correct:
```bash
# Test token fetch manually
python test_token_auth.py
```

### Problem: "Connection refused"

**Solution:** Server not running, start it:
```bash
source .venv/bin/activate
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Problem: "No clusters found"

**Solution:** Check endpoint IDs are correct:
```bash
# Run test to see available endpoints
python test_cluster_list.py
```

## 📝 Quick Reference

### Available Endpoints (Example)
- Delhi: ID `11`
- Bengaluru: ID `12`
- Cressex: ID `14`
- Mumbai-BKC: ID `162`
- Chennai-AMB: ID `204` (AI-enabled)

### Cluster Statuses
- **Healthy**: Cluster is running normally
- **Draft**: Cluster is being provisioned
- **Error**: Cluster has issues

### API Response Fields
- `clusterId`: Unique cluster identifier
- `clusterName`: Human-readable name
- `nodescount`: Number of nodes
- `kubernetesVersion`: K8s version (e.g., v1.26.15)
- `status`: Health status
- `displayNameEndpoint`: Location name
- `type`: MGMT or APP
- `isIksBackupEnabled`: Backup enabled (true/false)

## 🎯 What You Can Ask

### Listing Queries
- "Show me all clusters"
- "List k8s clusters"
- "What clusters are available?"
- "How many clusters do we have?"

### Location-Specific
- "Show clusters in Mumbai"
- "List clusters in Delhi and Bengaluru"
- "What's in Chennai?"

### Information Queries
- "What data centers do we have?"
- "Which locations have AI cloud enabled?"
- "Show me healthy clusters"

## 🚀 Production Deployment

### Health Check
```bash
curl http://localhost:8000/health
```

Should return:
```json
{
  "status": "healthy",
  "services": {
    "milvus": {"status": "active"},
    "ai_services": {"embedding": "operational"}
  }
}
```

### Load Testing
```bash
# Run multiple concurrent requests
for i in {1..10}; do
  curl -X POST http://localhost:8000/api/agent/chat \
    -H "Content-Type: application/json" \
    -d "{\"message\": \"Show me all clusters\", \"session_id\": \"load-test-$i\"}" &
done
wait
```

## 📚 Additional Resources

- **Full Documentation**: `CLUSTER_LISTING_GUIDE.md`
- **Implementation Details**: `IMPLEMENTATION_SUMMARY.md`
- **Token Auth Setup**: `TOKEN_AUTH_SETUP.md`
- **Test Scripts**: `test_cluster_list.py`, `test_token_auth.py`

## 💡 Tips

1. **Cache Optimization**: First query takes 2-3 seconds, subsequent queries <1 second
2. **Token Management**: Tokens auto-refresh, no manual intervention needed
3. **Error Handling**: All errors are gracefully handled with helpful messages
4. **Logging**: Check logs for detailed execution flow

## ✅ Verification Checklist

Before using in production:

- [ ] Environment variables configured
- [ ] Server starts without errors
- [ ] Health check returns "healthy"
- [ ] Test script passes all tests
- [ ] Agent chat returns cluster list
- [ ] Token authentication working
- [ ] Engagement ID cached correctly

---

**Need Help?** Check logs or run test scripts for detailed diagnostics.

**Quick Test**: `python test_cluster_list.py` - Should complete in ~5 seconds with all tests passing.

