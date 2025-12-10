# Multi-Agent System Setup Guide

## 🎉 Implementation Complete!

I've successfully implemented a **LangChain-based multi-agent system** for your Enterprise RAG Bot. The system enables conversational CRUD operations on cloud resources with intelligent parameter collection and validation.

## 📦 What's Been Implemented

### 1. **Core Agent Architecture**
- ✅ `BaseAgent` - Abstract base class for all agents
- ✅ `OrchestratorAgent` - Main coordinator that routes requests
- ✅ `IntentAgent` - Detects user intent and extracts parameters
- ✅ `ValidationAgent` - Validates parameters and collects missing info
- ✅ `ExecutionAgent` - Executes CRUD operations via API
- ✅ `RAGAgent` - Answers questions using documentation

### 2. **State Management**
- ✅ `ConversationState` - Tracks conversation flow and parameters
- ✅ `ConversationStateManager` - Manages multiple conversation sessions
- ✅ Multi-turn parameter collection
- ✅ Session persistence and cleanup

### 3. **Services**
- ✅ `APIExecutorService` - Executes CRUD operations on resources
- ✅ Parameter validation against schema
- ✅ Permission-based access control
- ✅ API call handling with retries

### 4. **API Integration**
- ✅ New `/api/agent/chat` endpoint for conversational interface
- ✅ Session management endpoints
- ✅ Statistics and health check endpoints
- ✅ Integrated with existing FastAPI app

### 5. **Configuration**
- ✅ Resource schema system (`app/config/resource_schema.json`)
- ✅ Updated `requirements.txt` with LangChain dependencies
- ✅ Comprehensive documentation and README

## 📁 File Structure

```
app/
├── agents/
│   ├── __init__.py                    # Agent exports
│   ├── README.md                      # Comprehensive documentation
│   ├── base_agent.py                  # Base agent class
│   ├── orchestrator_agent.py          # Main coordinator
│   ├── intent_agent.py                # Intent detection
│   ├── validation_agent.py            # Parameter validation
│   ├── execution_agent.py             # Operation execution
│   ├── rag_agent.py                   # RAG-based Q&A
│   ├── agent_manager.py               # Central manager
│   ├── state/
│   │   ├── __init__.py
│   │   └── conversation_state.py      # State management
│   ├── tools/
│   │   └── __init__.py
│   └── prompts/
│       └── __init__.py
├── services/
│   └── api_executor_service.py        # API execution service
├── api/
│   └── routes/
│       └── agent_chat.py              # Agent API endpoints
├── config/
│   └── resource_schema.json           # Resource definitions
└── main.py                            # Updated with agent routes
```

## 🚀 Installation Steps

### 1. Install Dependencies

The virtual environment has permission issues. You need to install the packages with proper permissions:

```bash
cd /home/unixlogin/vayuMaya/Enterprise-Rag-bot

# Option 1: Fix permissions and install
sudo chown -R $USER:$USER .venv/
source .venv/bin/activate
pip install langchain==0.1.0 langchain-openai==0.0.2 langchain-community==0.0.10 langgraph==0.0.20

# Option 2: Recreate virtual environment
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Add to your `.env` file:

```bash
# LangChain/LLM Configuration (already configured)
OPENAI_API_KEY=your-api-key
CHAT_MODEL=openai/gpt-oss-120b
GROK_BASE_URL=https://api.ai-cloud.cloudlyte.com/v1

# API Executor Configuration
API_EXECUTOR_TIMEOUT=30
API_EXECUTOR_MAX_RETRIES=3
API_AUTH_TOKEN=your-api-token-for-crud-operations

# Vector Database (already configured)
VECTOR_DB=milvus
MILVUS_HOST=localhost
MILVUS_PORT=19530
```

### 3. Update Resource Schema

Edit `app/config/resource_schema.json` to add your actual API endpoints and parameters. The file currently has a template for K8s clusters.

### 4. Start the Application

```bash
# Activate virtual environment
source .venv/bin/activate

# Run the application
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## 🧪 Testing the System

### Test 1: Simple Intent Detection

```bash
curl -X POST http://localhost:8000/api/agent/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Create a new Kubernetes cluster",
    "user_id": "test_user",
    "user_roles": ["admin"]
  }'
```

Expected response: The agent will ask for required parameters.

### Test 2: Multi-turn Conversation

```bash
# First message
SESSION_ID=$(curl -s -X POST http://localhost:8000/api/agent/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Create a cluster named prod-cluster",
    "user_id": "test_user",
    "user_roles": ["admin"]
  }' | jq -r '.session_id')

# Second message (same session)
curl -X POST http://localhost:8000/api/agent/chat \
  -H "Content-Type: application/json" \
  -d "{
    \"message\": \"Use version 1.28 in us-east-1\",
    \"session_id\": \"$SESSION_ID\",
    \"user_id\": \"test_user\",
    \"user_roles\": [\"admin\"]
  }"
```

### Test 3: RAG Question

```bash
curl -X POST http://localhost:8000/api/agent/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "How do I configure a Kubernetes cluster?",
    "user_id": "test_user",
    "user_roles": ["viewer"]
  }'
```

### Test 4: Get Conversation Status

```bash
curl http://localhost:8000/api/agent/conversation/$SESSION_ID
```

### Test 5: Agent Statistics

```bash
curl http://localhost:8000/api/agent/stats
```

## 🎯 Usage Examples

### Example 1: Creating a K8s Cluster

```
User: "Create a new Kubernetes cluster"
Bot: "I'll help you create a Kubernetes cluster. I need some information:
     - Cluster Name
     - Data Center location
     - Kubernetes version
     - Control Plane Type
     ..."

User: "Name it prod-cluster-01, version 1.28, in us-east-1"
Bot: "Great! I've collected:
     - Cluster Name: prod-cluster-01
     - Kubernetes version: 1.28
     - Data Center: us-east-1
     I still need: Control Plane Type, Business Unit..."

User: "Standard control plane, Engineering BU"
Bot: "Perfect! I have all the information. Shall I proceed?"

User: "Yes"
Bot: "✅ Successfully created Kubernetes cluster!
     - Name: prod-cluster-01
     - Status: Provisioning
     - Cluster ID: cls-abc123"
```

### Example 2: Asking Questions

```
User: "How do I scale a cluster?"
Bot: "Based on our documentation, here's how to scale a cluster:
     1. Identify the node pool to scale
     2. Use the update API with new node count
     3. Monitor the scaling progress
     ..."
```

## 🔧 Customization

### Adding New Resources

1. **Edit `app/config/resource_schema.json`:**

```json
{
  "resources": {
    "firewall": {
      "operations": ["create", "read", "update", "delete", "list"],
      "api_endpoints": {
        "create": {
          "method": "POST",
          "url": "https://your-api.com/v1/firewalls"
        }
      },
      "parameters": {
        "create": {
          "required": ["name", "rules"],
          "optional": ["description"],
          "validation": {
            "name": {
              "type": "string",
              "min_length": 3,
              "max_length": 50
            }
          }
        }
      },
      "permissions": {
        "create": ["admin"],
        "delete": ["admin"]
      }
    }
  }
}
```

2. **Test the new resource:**

```bash
curl -X POST http://localhost:8000/api/agent/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Create a firewall rule",
    "user_id": "test_user",
    "user_roles": ["admin"]
  }'
```

### Customizing Agent Behavior

Edit the system prompts in each agent file:
- `app/agents/intent_agent.py` - Intent detection behavior
- `app/agents/validation_agent.py` - Validation messages
- `app/agents/execution_agent.py` - Success/error messages
- `app/agents/rag_agent.py` - Documentation responses

## 📊 Monitoring

The system provides comprehensive logging:

```python
# Check logs
tail -f logs/app.log | grep -E "(Agent|Intent|Validation|Execution)"
```

Key log patterns:
- `🎭 Orchestrating request` - New request received
- `🔀 Routing to` - Agent routing decision
- `🎯 IntentAgent analyzing` - Intent detection
- `✅ ValidationAgent processing` - Parameter validation
- `⚡ ExecutionAgent executing` - Operation execution
- `📚 RAGAgent answering` - RAG query

## 🐛 Troubleshooting

### Issue: "Module not found: langchain_openai"

**Solution:**
```bash
source .venv/bin/activate
pip install langchain-openai langchain-community langgraph
```

### Issue: "Permission denied" during installation

**Solution:**
```bash
sudo chown -R $USER:$USER .venv/
# or recreate the virtual environment
```

### Issue: "Vector service not configured"

**Solution:**
Ensure Milvus is running and initialized:
```bash
# Check Milvus status
sudo docker ps | grep milvus

# Check if Milvus service is initialized in app
curl http://localhost:8000/api/agent/health
```

### Issue: "API execution failed"

**Solution:**
1. Verify API endpoints in `resource_schema.json`
2. Check `API_AUTH_TOKEN` in `.env`
3. Test API endpoint directly with curl
4. Review API executor logs

## 📚 Documentation

- **Agent System README**: `app/agents/README.md`
- **API Documentation**: http://localhost:8000/docs (after starting the app)
- **Resource Schema**: `app/config/resource_schema.json`

## 🎓 Architecture Overview

```
User Request
     ↓
OrchestratorAgent (decides routing)
     ↓
┌────┴────┬─────────┬──────────┬─────────┐
│         │         │          │         │
Intent  Validation Execution  RAG    (Specialized Agents)
Agent    Agent      Agent     Agent
     ↓
API Executor Service
     ↓
External APIs (K8s, Firewall, etc.)
```

## 🚀 Next Steps

1. **Install Dependencies** (see Installation Steps above)
2. **Configure Resource Schema** with your actual APIs
3. **Set Environment Variables** for API authentication
4. **Test the System** with the provided curl commands
5. **Customize Agent Prompts** for your use case
6. **Add More Resources** as needed
7. **Monitor and Iterate** based on user feedback

## 💡 Key Features

✅ **Conversational Interface** - Natural language CRUD operations  
✅ **Multi-turn Conversations** - Collects parameters across multiple messages  
✅ **Intent Detection** - Automatically understands what user wants  
✅ **Parameter Validation** - Schema-based validation with helpful error messages  
✅ **Permission Management** - Role-based access control  
✅ **RAG Integration** - Documentation-based Q&A  
✅ **Session Management** - Persistent conversation state  
✅ **Error Handling** - Graceful error handling with user-friendly messages  
✅ **Extensible** - Easy to add new resources and agents  

## 📞 Support

For questions or issues:
1. Check the comprehensive README: `app/agents/README.md`
2. Review the API docs: http://localhost:8000/docs
3. Check logs for detailed error messages
4. Test individual components with curl commands

---

**Status**: ✅ Implementation Complete - Ready for Testing!

**Note**: The system is fully implemented and integrated. You just need to:
1. Fix the virtual environment permissions
2. Install the LangChain dependencies
3. Configure your actual API endpoints in the resource schema
4. Start testing!

