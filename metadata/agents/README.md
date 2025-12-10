# Multi-Agent System for Enterprise RAG Bot

## Overview

This is a **LangChain-based multi-agent system** that enables conversational CRUD operations on cloud resources. The system uses specialized agents that collaborate to understand user intent, collect parameters, validate inputs, and execute operations.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Request                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                 OrchestratorAgent                            │
│  (Routes requests to specialized agents)                     │
└──────┬────────────┬────────────┬────────────┬───────────────┘
       │            │            │            │
       ▼            ▼            ▼            ▼
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│  Intent  │ │Validation│ │Execution │ │   RAG    │
│  Agent   │ │  Agent   │ │  Agent   │ │  Agent   │
└──────────┘ └──────────┘ └──────────┘ └──────────┘
     │            │            │            │
     └────────────┴────────────┴────────────┘
                       │
                       ▼
              ┌────────────────┐
              │ API Executor   │
              │   Service      │
              └────────────────┘
```

## Agents

### 1. **OrchestratorAgent**
- **Role**: Main coordinator
- **Responsibilities**:
  - Routes user requests to appropriate agents
  - Manages conversation flow
  - Coordinates between agents
  - Synthesizes responses

### 2. **IntentAgent**
- **Role**: Intent detection and parameter extraction
- **Responsibilities**:
  - Identifies resource type (k8s_cluster, firewall, etc.)
  - Detects operation (create, update, delete, list)
  - Extracts parameters from natural language
  - Determines required vs optional parameters

### 3. **ValidationAgent**
- **Role**: Parameter validation and collection
- **Responsibilities**:
  - Validates parameters against schema
  - Collects missing required parameters
  - Asks clarifying questions
  - Ensures data quality

### 4. **ExecutionAgent**
- **Role**: Operation execution
- **Responsibilities**:
  - Executes validated CRUD operations
  - Makes API calls
  - Handles success/error responses
  - Provides user-friendly feedback

### 5. **RAGAgent**
- **Role**: Documentation Q&A
- **Responsibilities**:
  - Answers questions using vector database
  - Retrieves relevant documentation
  - Generates context-aware responses
  - Cites sources

## Features

### ✅ Conversational Parameter Collection
- Multi-turn conversations to collect all required parameters
- Intelligent prompting for missing information
- Contextual understanding of user responses

### ✅ Intent Detection
- Automatically detects what the user wants to do
- Supports multiple resource types and operations
- Handles ambiguous requests with clarification

### ✅ Parameter Validation
- Schema-based validation
- Type checking, range validation, regex patterns
- User-friendly error messages

### ✅ Permission Management
- Role-based access control
- Operation-level permissions
- Graceful permission denial with helpful messages

### ✅ Execution & Feedback
- Actual API execution for CRUD operations
- Clear success/error messages
- Actionable feedback and next steps

### ✅ RAG Integration
- Seamless integration with existing vector database
- Documentation-based answers
- Fallback to general knowledge when needed

## Usage

### API Endpoint

```bash
POST /api/agent/chat
```

**Request:**
```json
{
  "message": "Create a new Kubernetes cluster named prod-cluster",
  "session_id": "optional-session-id",
  "user_id": "user123",
  "user_roles": ["admin", "developer"]
}
```

**Response:**
```json
{
  "success": true,
  "response": "I'll help you create a Kubernetes cluster named 'prod-cluster'. I need a few more details:\n- Data Center location\n- Kubernetes version\n- Control Plane Type\n...",
  "session_id": "abc-123-def",
  "routing": "intent",
  "metadata": {
    "duration_seconds": 0.45,
    "timestamp": "2025-11-07T10:30:00Z"
  }
}
```

### Example Conversations

#### Creating a K8s Cluster

```
User: Create a new Kubernetes cluster
Bot: I'll help you create a Kubernetes cluster. I need some information:
     - Cluster Name
     - Data Center location
     - Kubernetes version
     - Control Plane Type
     You can provide these all at once or one at a time.

User: Name it prod-cluster-01, use version 1.28, in us-east-1
Bot: Great! I've collected:
     - Cluster Name: prod-cluster-01
     - Kubernetes version: 1.28
     - Data Center: us-east-1
     
     I still need:
     - Control Plane Type
     - Business Unit
     - Environments
     ...

User: Standard control plane, BU is Engineering, env is production
Bot: Perfect! I have all the information needed to create the cluster.
     
     **Summary:**
     - Cluster Name: prod-cluster-01
     - Data Center: us-east-1
     - Kubernetes version: 1.28
     - Control Plane Type: Standard
     - Business Unit: Engineering
     - Environment: production
     
     Shall I proceed with this operation?

User: Yes, proceed
Bot: ✅ Successfully created Kubernetes cluster!
     
     **Details:**
     - Name: prod-cluster-01
     - Status: Provisioning
     - Cluster ID: cls-abc123
     
     💡 Your cluster is now being provisioned. It should be ready in about 10-15 minutes.
```

#### Asking Questions

```
User: How do I configure a load balancer?
Bot: Based on our documentation, here's how to configure a load balancer:
     
     1. **Choose your load balancer type:**
        - Application Load Balancer (Layer 7)
        - Network Load Balancer (Layer 4)
     
     2. **Configure listeners:**
        - Define ports and protocols
        - Set up SSL certificates if needed
     
     3. **Set up target groups:**
        - Define health check parameters
        - Add backend instances
     
     **Related topics:** SSL certificates, Health checks, Auto-scaling
```

## Configuration

### Resource Schema

Define your resources in `app/config/resource_schema.json`:

```json
{
  "resources": {
    "k8s_cluster": {
      "operations": ["create", "update", "delete", "list"],
      "api_endpoints": {
        "create": {
          "method": "POST",
          "url": "https://api.example.com/v1/clusters"
        }
      },
      "parameters": {
        "create": {
          "required": ["name", "region", "version"],
          "optional": ["node_count", "tags"],
          "validation": {
            "name": {
              "type": "string",
              "pattern": "^[a-z0-9-]+$",
              "min_length": 3,
              "max_length": 63
            }
          }
        }
      },
      "permissions": {
        "create": ["admin", "developer"],
        "delete": ["admin"]
      }
    }
  }
}
```

### Environment Variables

```bash
# LangChain/LLM Configuration
OPENAI_API_KEY=your-api-key
CHAT_MODEL=openai/gpt-oss-120b
GROK_BASE_URL=https://api.ai-cloud.cloudlyte.com/v1

# API Executor Configuration
API_EXECUTOR_TIMEOUT=30
API_EXECUTOR_MAX_RETRIES=3
API_AUTH_TOKEN=your-api-token

# Vector Database (for RAG)
VECTOR_DB=milvus
MILVUS_HOST=localhost
MILVUS_PORT=19530
```

## State Management

The system maintains conversation state across multiple turns:

- **Session Management**: Each conversation has a unique session ID
- **Parameter Tracking**: Tracks collected, missing, and invalid parameters
- **Conversation History**: Maintains full conversation context
- **Agent Handoffs**: Records which agent handled each step

### Conversation States

1. `INITIATED` - Conversation started
2. `COLLECTING_PARAMS` - Collecting required parameters
3. `VALIDATING` - Validating collected parameters
4. `READY_TO_EXECUTE` - All parameters collected and valid
5. `EXECUTING` - Operation in progress
6. `COMPLETED` - Operation completed successfully
7. `FAILED` - Operation failed
8. `CANCELLED` - User cancelled

## API Reference

### Chat Endpoint
- `POST /api/agent/chat` - Process user message

### Conversation Management
- `GET /api/agent/conversation/{session_id}` - Get conversation status
- `DELETE /api/agent/conversation/{session_id}` - Reset conversation

### System Management
- `GET /api/agent/stats` - Get agent system statistics
- `POST /api/agent/cleanup` - Clean up old sessions
- `GET /api/agent/health` - Health check

## Development

### Adding New Resources

1. **Update resource schema** (`app/config/resource_schema.json`)
2. **Add API endpoints** for the resource
3. **Define validation rules** for parameters
4. **Set permissions** for operations

### Adding New Agents

1. **Extend BaseAgent** class
2. **Implement required methods**:
   - `get_system_prompt()`
   - `get_tools()`
3. **Register with AgentManager**
4. **Wire to OrchestratorAgent**

### Testing

```python
# Test intent detection
from app.agents import get_agent_manager

manager = get_agent_manager()
result = await manager.process_request(
    user_input="Create a cluster named test",
    session_id="test-session",
    user_id="test-user",
    user_roles=["admin"]
)
print(result["response"])
```

## Monitoring

The system provides comprehensive logging:

```
🎭 Orchestrating request for session abc-123
🔀 Routing to IntentAgent: Create a cluster...
🎯 IntentAgent analyzing: Create a cluster...
✅ Intent detected: create k8s_cluster | Required params: 5
📨 OrchestratorAgent -> ValidationAgent: Collecting parameters
✅ ValidationAgent processing: prod-cluster...
⚡ ExecutionAgent executing operation...
🚀 Executing create on k8s_cluster with params: ['name', 'region']
✅ Execution successful: create k8s_cluster
```

## Troubleshooting

### Agent Not Responding
- Check if LangChain dependencies are installed
- Verify OPENAI_API_KEY is set
- Check GROK_BASE_URL is accessible

### Parameter Validation Failing
- Review resource schema validation rules
- Check parameter types and formats
- Verify required parameters are defined

### API Execution Failing
- Verify API endpoints in resource schema
- Check API_AUTH_TOKEN is set
- Review API service logs

## Future Enhancements

- [ ] Multi-agent collaboration for complex tasks
- [ ] Agent learning from user feedback
- [ ] Advanced NER for parameter extraction
- [ ] Workflow orchestration for multi-step operations
- [ ] Agent performance analytics
- [ ] Custom agent plugins

## License

Part of Enterprise RAG Bot - See main project LICENSE

