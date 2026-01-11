# Embedding API Error Analysis

## Date: December 12, 2025

---

## Overview

During backend startup, the system encounters multiple errors related to embeddings and database connectivity. This document provides a detailed analysis of each issue, their impact, and solutions.

---

## Issue #1: PostgreSQL Connection Failure

### Error Messages
```
Line 966: ⚠️ Failed to initialize persistent storage: 
  (psycopg2.OperationalError) could not translate host name "postgres" to address: Name or service not known
  
Line 984: ❌ Database initialization error: 
  (psycopg2.OperationalError) could not translate host name "postgres" to address: Name or service not known
```

### Root Cause
The backend is running on the **host machine** (not in Docker), but it's configured to connect to PostgreSQL using the Docker hostname `"postgres"`.

**Configuration Issue:**
```python
# Current DATABASE_URL in .env or config
DATABASE_URL=postgresql://user:pass@postgres:5432/dbname
                                    ^^^^^^
                                    Docker hostname (not resolvable from host)
```

### Impact
- ✅ **Application continues**: Falls back to in-memory session storage
- ❌ **No persistent sessions**: Conversation history lost on restart
- ❌ **No Memori feature**: Session persistence disabled

### Solution

**Option 1: Update DATABASE_URL for Host Network**
```bash
# In .env file, change:
DATABASE_URL=postgresql://ragbot:ragbot_secret_2024@localhost:5435/ragbot_sessions
#                                                    ^^^^^^^^^
#                                                    localhost + external port
```

**Option 2: Run Backend in Docker**
```bash
# Start the full Docker stack
sudo docker-compose up -d
```

**Option 3: Start PostgreSQL Container with Host Network**
```bash
# Make PostgreSQL accessible on host
sudo docker run -d \
  --name ragbot-postgres \
  --network host \
  -e POSTGRES_USER=ragbot \
  -e POSTGRES_PASSWORD=ragbot_secret_2024 \
  -e POSTGRES_DB=ragbot_sessions \
  postgres:16-alpine
  
# Then use:
DATABASE_URL=postgresql://ragbot:ragbot_secret_2024@localhost:5432/ragbot_sessions
```

---

## Issue #2: Embedding API Failures (Critical)

### Error Sequence

#### Step 1: SDK Embedding Timeout
```
Line 995: WARNING: Batch embedding timeout on attempt 1
Line 996: WARNING: ⚠️ SDK embedding attempt 1 failed
Line 997: WARNING: Batch embedding timeout on attempt 2
Line 998: WARNING: ⚠️ SDK embedding attempt 2 failed
Line 999: WARNING: ⚠️ All SDK attempts failed, falling back to HTTP
```

**Analysis:**
- SDK client is not responding within timeout period
- Attempted twice, both failed
- System falls back to HTTP API

#### Step 2: HTTP Fallback - Model Not Available
```
Line 1000: INFO: Attempting HTTP embedding with model: Qwen/Qwen3-Embedding-8B
Line 1001-1002: HTTP Request: POST .../v1/embeddings "HTTP/1.1 500 Internal Server Error"
Line 1003: ERROR: ❌ HTTP embedding failed for index 0: HTTP 500
  Error: "litellm.InternalServerError: OpenAIException - Connection error.. 
         Received Model Group=Qwen/Qwen3-Embedding-8B
         Available Model Group Fallbacks=None"
```

**Analysis:**
- Model `Qwen/Qwen3-Embedding-8B` is configured but not available
- API returns 500 Internal Server Error
- No fallback models available for this model group

#### Step 3: Backup Model - Invalid Model Name
```
Line 1004: WARNING: ⚠️ Primary model produced zeros, trying hosted fallback: openai/gpt-oss-20b-embedding
Line 1005-1006: HTTP Request: POST .../v1/embeddings "HTTP/1.1 400 Bad Request"
Line 1007: ERROR: ❌ HTTP embedding failed for index 0: HTTP 400
  Error: "/embeddings: Invalid model name passed in model=openai/gpt-oss-20b-embedding. 
         Call `/v1/models` to view available models for your key."
```

**Analysis:**
- Fallback model `openai/gpt-oss-20b-embedding` doesn't exist
- API returns 400 Bad Request
- Suggests calling `/v1/models` to see available models

### Root Causes

1. **Wrong Embedding Model Configured**
   - `Qwen/Qwen3-Embedding-8B` is not available on your API endpoint
   - Backup model `openai/gpt-oss-20b-embedding` also doesn't exist

2. **SDK Configuration Issues**
   - Timeout is too short for the API response time
   - SDK might not be properly initialized

3. **API Endpoint Issues**
   - The embedding service might be down
   - Connection errors to the upstream model

### Impact

**Does NOT Affect:**
- ✅ Cluster listing operations
- ✅ Kafka listing operations
- ✅ GitLab listing operations
- ✅ Any direct API CRUD operations
- ✅ Agent routing and orchestration
- ✅ Parameter validation and collection

**Does Affect:**
- ❌ RAG (document search) operations
- ❌ Semantic search in knowledge base
- ❌ Document embedding for new content
- ❌ Vector similarity searches

### Solutions

#### Solution 1: Find Available Embedding Models (Recommended)

**Step 1: Check available models**
```bash
curl -X GET https://api.ai-cloud.cloudlyte.com/v1/models \
  -H "Authorization: Bearer YOUR_API_KEY" \
  | jq '.data[] | select(.id | contains("embed"))'
```

**Step 2: Update configuration**
```python
# In app/config/settings.py or .env
EMBEDDING_MODEL = "text-embedding-ada-002"  # Use actual available model
EMBEDDING_FALLBACK = "voyage-2"  # Use actual available fallback
```

#### Solution 2: Increase Timeout

**File:** `app/services/ai_service.py`

```python
# Find the embedding timeout configuration
EMBEDDING_TIMEOUT = 10  # Current (too short)
EMBEDDING_TIMEOUT = 30  # Recommended (longer)

# Or in .env
EMBEDDING_API_TIMEOUT=30
```

#### Solution 3: Use Different Embedding Provider

If the current API is unreliable, consider:

**Option A: Voyage AI**
```python
# Already have VOYAGE_API_KEY in .env
EMBEDDING_PROVIDER = "voyage"
EMBEDDING_MODEL = "voyage-2"
```

**Option B: OpenAI Direct**
```python
EMBEDDING_PROVIDER = "openai"
EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_API_KEY = "your-key"
```

**Option C: Local Embeddings**
```python
# Use sentence-transformers locally (no API needed)
pip install sentence-transformers
EMBEDDING_PROVIDER = "local"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
```

#### Solution 4: Disable Embedding Health Check

If you're not using RAG features, disable the startup check:

**File:** `app/user_main.py`

```python
# Find the startup health check
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("✅ Database initialized")
    logger.info("✅ Milvus connected successfully")
    
    # Comment out or skip embedding validation
    # await validate_ai_service()  # Skip this
    
    logger.info("✅ AI services operational")
```

---

## Issue #3: Pydantic Configuration Warning

### Warning Message
```
Line 972-974: UserWarning: Valid config keys have changed in V2:
* 'schema_extra' has been renamed to 'json_schema_extra'
```

### Root Cause
Code uses Pydantic v1 syntax (`schema_extra`) but running Pydantic v2.

### Impact
- ⚠️ Warning only, not breaking
- Works but should be updated for future compatibility

### Solution

Find and replace in codebase:

```bash
# Search for old syntax
grep -r "schema_extra" app/

# Update to new syntax
# Change from:
class Config:
    schema_extra = {...}

# To:
model_config = ConfigDict(json_schema_extra={...})
```

---

## Issue #4: Missing Static Files

### Warning Messages
```
Line 979: WARNING: ⚠️ User frontend not found at .../dist/user-frontend
Line 980: INFO: ⚠️ Widget static not mounted (folder .../widget_static not found)
Line 981: INFO: ⚠️ CDN static not mounted (folder .../widget_static not found)
```

### Root Cause
Frontend build files not present when running backend on host.

### Impact
- ✅ Backend API works fine
- ❌ Frontend UI not served from backend
- ✅ OpenWebUI works independently (port 3000)

### Solution

**If you need the Angular frontends:**

```bash
cd /home/unixlogin/Vayu/Enterprise-Rag-bot

# Build admin frontend
cd angular-frontend
npm install
npm run build
cd ..

# Build user frontend
cd user-frontend
npm install
npm run build
cd ..
```

**If you don't need them:**
- Ignore these warnings
- Use OpenWebUI on port 3000 (already working)

---

## Recommended Action Plan

### Priority 1: Fix Embedding Model (If using RAG)

1. **Check available models:**
   ```bash
   curl https://api.ai-cloud.cloudlyte.com/v1/models | jq '.data[].id' | grep -i embed
   ```

2. **Update config with valid model:**
   ```bash
   # In .env
   EMBEDDING_MODEL=<model-from-step-1>
   ```

3. **Restart backend:**
   ```bash
   # Backend will auto-reload if using --reload flag
   # Or manually restart
   ```

### Priority 2: Fix PostgreSQL Connection (For session persistence)

1. **Update DATABASE_URL:**
   ```bash
   # In .env
   DATABASE_URL=postgresql://ragbot:ragbot_secret_2024@localhost:5435/ragbot_sessions
   ```

2. **Verify PostgreSQL is running:**
   ```bash
   sudo docker ps | grep postgres
   ```

3. **Test connection:**
   ```bash
   psql postgresql://ragbot:ragbot_secret_2024@localhost:5435/ragbot_sessions -c "SELECT 1;"
   ```

### Priority 3: Update Pydantic Syntax (Technical debt)

1. **Find all instances:**
   ```bash
   grep -r "schema_extra" app/
   ```

2. **Update to Pydantic v2 syntax**

### Priority 4: Frontend (Optional)

Only if you need the Angular UIs built.

---

## Current Working State

### ✅ What's Working
- Backend API responding (port 8001)
- OpenWebUI connected (port 3000)
- Agent orchestration
- Kafka listing operations
- GitLab listing operations
- Cluster listing operations
- All CRUD operations via direct API calls
- In-memory session management

### ❌ What's Not Working
- Persistent session storage (PostgreSQL)
- Embedding generation (RAG features)
- Vector search operations
- Document semantic search

### ⚠️ What's Working with Warnings
- Application starts successfully
- Falls back gracefully to in-memory storage
- Operations continue despite embedding errors

---

## Testing After Fixes

### Test PostgreSQL Connection
```bash
# Check if backend can connect
tail -f logs/backend.log | grep -i "database"
# Should see: ✅ Database connected successfully
```

### Test Embedding API
```bash
# Check if embeddings work
tail -f logs/backend.log | grep -i "embedding"
# Should see: ✅ Embedding service initialized
```

### Test Overall Health
```bash
# Hit health endpoint
curl http://localhost:8001/health
# Should return: {"status": "healthy", "database": "connected", "embeddings": "available"}
```

---

## Conclusion

**Current State:**
- Your main features (Kafka/GitLab listing) are **working perfectly**
- Embedding and PostgreSQL issues are **isolated** and don't affect operations

**Immediate Action Required:**
- **None** - Your operations work fine as-is

**Recommended Actions:**
1. Fix embedding model configuration (if you plan to use RAG)
2. Fix PostgreSQL connection (if you need persistent sessions)
3. Update Pydantic syntax (technical debt, non-urgent)

**Decision:**
- If you're **only doing resource operations** (Kafka, GitLab, Clusters): ✅ **No fixes needed**
- If you plan to use **RAG features**: ⚠️ **Fix embedding configuration**
- If you want **persistent sessions**: ⚠️ **Fix PostgreSQL connection**


