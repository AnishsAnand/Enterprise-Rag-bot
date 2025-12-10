# Enterprise RAG Bot - Project Overview

## 🎯 What is This Project?

Enterprise RAG Bot is an intelligent conversational AI system that combines Retrieval-Augmented Generation (RAG) with a multi-agent architecture to help users manage enterprise resources through natural language interactions.

## 🏗️ Core Components

### 1. **Backend (FastAPI)**
- **Location**: `app/`
- **Purpose**: REST API server handling all business logic
- **Key Features**:
  - Multi-agent routing system
  - Vector database integration (Milvus)
  - LLM integration for chat and embeddings
  - Resource operation handling
  - Authentication and session management

### 2. **Agent System**
- **Location**: `app/agents/`
- **Purpose**: Specialized agents for different operations
- **Agents**:
  - **Cluster Agent**: Handles cluster creation, listing, management
  - **Resource Agent**: General resource operations
  - **RAG Agent**: Pure retrieval-augmented generation
- **Documentation**: [agents/README.md](./agents/README.md)

### 3. **Frontend (React)**
- **Location**: `user-frontend/`
- **Purpose**: User interface for interacting with the bot
- **Features**:
  - Chat widget integration
  - Resource visualization
  - Authentication UI
- **Documentation**: [frontend/README.md](./frontend/README.md)

### 4. **OpenWebUI Integration**
- **Purpose**: Modern chat interface alternative
- **Features**:
  - Session management
  - Multi-user support
  - Rich UI components
- **Documentation**: [OPENWEBUI_README.md](./OPENWEBUI_README.md)

### 5. **Vector Database (Milvus)**
- **Purpose**: Semantic search for documentation and resources
- **Features**:
  - Embedding storage
  - Similarity search
  - Hybrid search with keyword filtering

## 🔄 How It Works

### Query Flow

```
User Query
    ↓
Intent Detection (LLM)
    ↓
Agent Router
    ↓
┌─────────────┬─────────────┬─────────────┐
│   Cluster   │  Resource   │     RAG     │
│    Agent    │    Agent    │    Agent    │
└─────────────┴─────────────┴─────────────┘
    ↓
Vector Search (Milvus)
    ↓
Context Retrieval
    ↓
LLM Response Generation
    ↓
Formatted Response to User
```

### Key Processes

1. **Intent Detection**: Determines if query is about resources or general chat
2. **Agent Routing**: Routes to appropriate specialized agent
3. **Vector Search**: Finds relevant documentation/context
4. **Response Generation**: LLM generates contextual response
5. **Action Execution**: Performs operations if needed (create cluster, etc.)

## 🔑 Key Features

### 1. Intelligent Query Understanding
- Keyword-based detection
- LLM-powered intent analysis
- Confidence scoring
- Query correction

### 2. Multi-Agent Architecture
- Specialized agents for different domains
- Dynamic routing based on query intent
- Fallback mechanisms

### 3. RAG (Retrieval-Augmented Generation)
- Semantic search across documentation
- Context-aware responses
- Hybrid search (vector + keyword)
- Diversity filtering for better results

### 4. Resource Operations
- Cluster creation and management
- Resource listing and querying
- Schema-based validation
- Operation tracking

### 5. Session Management
- User authentication
- Session persistence
- Multi-user support
- Token-based security

## 📊 Technology Stack

### Backend
- **Framework**: FastAPI (Python)
- **LLM Integration**: OpenAI-compatible APIs
- **Vector DB**: Milvus
- **Embedding Models**: Qwen/Qwen3-Embedding-8B
- **Chat Models**: openai/gpt-oss-120b

### Frontend
- **Framework**: React + TypeScript
- **UI Library**: Custom components
- **State Management**: React hooks
- **API Client**: Axios/Fetch

### Infrastructure
- **Deployment**: Uvicorn (ASGI server)
- **Authentication**: JWT tokens
- **Configuration**: JSON schemas
- **Logging**: Python logging module

## 🎨 Architecture Highlights

### 1. Modular Design
- Clear separation of concerns
- Service-based architecture
- Easy to extend and maintain

### 2. Scalable
- Async/await patterns
- Connection pooling
- Efficient vector search

### 3. Robust Error Handling
- Fallback mechanisms
- Retry logic
- Graceful degradation

### 4. Configurable
- JSON-based configuration
- Environment variables
- Dynamic model selection

## 🔐 Security Features

- Token-based authentication
- Session validation
- API key management
- Secure credential storage

## 📈 Current Status

- ✅ Core RAG functionality
- ✅ Multi-agent system
- ✅ Cluster operations
- ✅ OpenWebUI integration
- ✅ Session management
- ✅ Vector search optimization
- 🔄 Ongoing improvements to embedding reliability

## 🚀 Getting Started

1. **Quick Start**: [QUICK_START.md](./QUICK_START.md)
2. **Architecture Deep Dive**: [ARCHITECTURE.md](./ARCHITECTURE.md)
3. **Agent System**: [agents/README.md](./agents/README.md)
4. **Server Setup**: [START_SERVERS.md](./START_SERVERS.md)

## 📝 Common Use Cases

1. **Cluster Management**
   - "Create a new cluster with 3 nodes"
   - "List all my clusters"
   - "Show cluster configuration"

2. **Resource Queries**
   - "How do I enable firewall?"
   - "What are the networking options?"
   - "Show me storage configurations"

3. **Documentation Search**
   - "How does authentication work?"
   - "What are the API endpoints?"
   - "Explain the agent architecture"

## 🐛 Known Issues & Limitations

- Embedding service occasional timeouts (fallback mechanisms in place)
- Some model compatibility issues (documented in logs)
- See [TESTING_STATUS.md](./TESTING_STATUS.md) for current test coverage

## 🤝 Contributing

For contribution guidelines and development setup:
- [AGENT_SYSTEM_SETUP.md](./AGENT_SYSTEM_SETUP.md)
- [REFACTORING_SUMMARY.md](./REFACTORING_SUMMARY.md)

## 📞 Support

Check the documentation index: [INDEX.md](./INDEX.md)

---

*This overview provides a high-level understanding of the project. For detailed information, please refer to the specific documentation files in this metadata folder.*

