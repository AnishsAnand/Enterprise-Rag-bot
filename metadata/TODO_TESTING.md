# Testing TODO - Managed Services

## 🧪 Pending Tests (Require API Credentials)

The following tests need to be run with valid API credentials to verify the integration:

### 1. Test Kafka Service Listing

**Query**: `"list kafka services"`

**Expected Flow**:
```
1. IntentAgent detects: resource=kafka, operation=list
2. ValidationAgent collects endpoints
3. ExecutionAgent:
   - Gets PAAS engagement_id
   - Converts to IPC engagement_id
   - Calls list_kafka(endpoints=[...])
   - Formats response
```

**Expected Response**:
```
✅ Found X Kafka Services
Queried Y endpoints

📍 Delhi
✅ kafka-prod-01 | Running | v3.4.0 | Cluster: prod-k8s-01
...
```

**API Call**:
```bash
POST https://ipcloud.tatacommunications.com/paasservice/api/v1/paas/listManagedServices/IKSKafka
{
  "engagementId": 1602,  // IPC engagement ID
  "endpoints": [11, 12, 14, 162, 204],
  "serviceType": "IKSKafka"
}
```

---

### 2. Test GitLab Service Listing

**Query**: `"show me gitlab services"`

**Expected Flow**:
```
Same as Kafka, but with:
- resource=gitlab
- serviceType=IKSGitlab
```

**Expected Response**:
```
✅ Found X GitLab Services
Queried Y endpoints

📍 Mumbai-BKC
✅ gitlab-enterprise | Running | v16.5.0
   Cluster: prod-k8s-mumbai
   URL: https://gitlab.enterprise.local
...
```

**API Call**:
```bash
POST https://ipcloud.tatacommunications.com/paasservice/api/v1/paas/listManagedServices/IKSGitlab
{
  "engagementId": 1602,
  "endpoints": [11, 12, 14, 162, 204],
  "serviceType": "IKSGitlab"
}
```

---

### 3. Test with Specific Endpoints

**Query**: `"list kafka in delhi and mumbai"`

**Expected**:
- ValidationAgent extracts "delhi" and "mumbai"
- Matches to endpoint IDs [11, 162]
- API called with only those 2 endpoints

---

### 4. Test IPC Engagement ID Conversion

**Verify**:
```
1. PAAS engagement API returns engagement_id (e.g., 1923)
2. get_ipc_engagement API converts 1923 → 1602
3. Managed services API called with 1602 (IPC ID)
```

**Log Output Should Show**:
```
🔄 Converted PAAS engagement 1923 to IPC engagement 1602
📋 Fetching IKSKafka services for IPC engagement 1602
```

---

## ✅ What's Already Verified (Without API)

✅ **Code compiles** - No syntax errors  
✅ **Schema is valid JSON** - resource_schema.json parses correctly  
✅ **Methods exist** - list_kafka(), list_gitlab(), list_managed_services() defined  
✅ **Intent detection patterns** - System prompt includes kafka/gitlab examples  
✅ **Execution routing** - ExecutionAgent has handlers for kafka/gitlab  
✅ **Formatting logic** - _format_success_message() handles kafka/gitlab  
✅ **Extensible pattern** - Clear path for adding new service types  

---

## 🔐 Prerequisites for Testing

1. **Valid API credentials** in `.env`:
   ```
   API_AUTH_EMAIL=your-email@example.com
   API_AUTH_PASSWORD=your-password
   ```

2. **Valid engagement** with managed services deployed

3. **Docker services running**:
   ```bash
   cd /home/unixlogin/Vayu/Enterprise-Rag-bot
   sudo docker-compose up -d
   ```

4. **Access OpenWebUI** at http://localhost:3000

---

## 🚀 How to Test

### Option 1: Via OpenWebUI (Recommended)

```bash
# 1. Start services
cd /home/unixlogin/Vayu/Enterprise-Rag-bot
sudo docker-compose up -d

# 2. Open http://localhost:3000 in browser

# 3. Try queries:
"list kafka services"
"show me kafka in delhi"
"what gitlab services do we have?"
"list gitlab in mumbai and chennai"
```

### Option 2: Via cURL (Direct API)

```bash
# Test through agent chat API
curl -X POST http://localhost:8001/api/agent/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "list kafka services",
    "user_id": "test-user",
    "user_roles": ["admin"]
  }'
```

### Option 3: Via Python Script

```python
import asyncio
from app.services.api_executor_service import api_executor_service

async def test_kafka():
    result = await api_executor_service.list_kafka()
    print(result)

asyncio.run(test_kafka())
```

---

## 📊 Success Criteria

For each test, verify:

✅ **Intent detection works** - IntentAgent correctly identifies resource_type  
✅ **Endpoint collection works** - ValidationAgent asks for/extracts endpoints  
✅ **IPC conversion works** - PAAS ID correctly converted to IPC ID  
✅ **API call succeeds** - Returns 200 with service data  
✅ **Response formatting** - Beautiful, readable output with emojis  
✅ **Error handling** - Graceful failures if API is down or returns errors  

---

## 🐛 Common Issues to Watch For

1. **IPC engagement ID missing/wrong**
   - Symptom: API returns 401 or 404
   - Fix: Verify get_ipc_engagement API is working

2. **Endpoint IDs incorrect**
   - Symptom: API returns empty array
   - Fix: Check endpoint name-to-ID conversion

3. **Service type mismatch**
   - Symptom: API returns empty array
   - Fix: Ensure serviceType matches exactly (case-sensitive)

4. **Auth token expired**
   - Symptom: 401 Unauthorized
   - Fix: Token should auto-refresh, check API_AUTH_EMAIL/PASSWORD

---

## 📝 Testing Checklist

- [ ] Test Kafka listing with all endpoints
- [ ] Test Kafka listing with specific endpoint (delhi)
- [ ] Test Kafka listing with multiple endpoints (delhi, mumbai)
- [ ] Test GitLab listing with all endpoints
- [ ] Test GitLab listing with specific endpoint
- [ ] Test GitLab listing with multiple endpoints
- [ ] Verify IPC engagement ID conversion in logs
- [ ] Verify beautiful formatting in response
- [ ] Test error handling (invalid endpoint, API down)
- [ ] Test with no services deployed (empty response)

---

## 🎯 Next Steps After Testing

Once tests pass:

1. ✅ Mark TODOs 7 & 8 as completed
2. 📸 Take screenshots of successful responses
3. 📊 Document any API quirks discovered
4. ➕ Consider adding more managed service types
5. 🔒 Add permission checks if needed

---

**Created**: 2025-12-11  
**Status**: ⏳ Awaiting API credentials for testing  
**Priority**: High (should test before production use)
