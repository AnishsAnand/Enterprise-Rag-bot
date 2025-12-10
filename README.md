# Enterprise RAG Bot

An intelligent RAG (Retrieval-Augmented Generation) bot with multi-agent architecture for enterprise resource management.

> 📋 **Project Organization**: See [PROJECT_ORGANIZATION.md](./PROJECT_ORGANIZATION.md) for complete directory structure and organization details.

## 📚 Documentation

All project documentation has been organized in the **[`metadata/`](./metadata/)** folder for easy navigation.

**Start here**: [metadata/INDEX.md](./metadata/INDEX.md) - Complete documentation index

### Quick Links

- 🚀 [Quick Start Guide](./metadata/QUICK_START.md)
- 🏗️ [Architecture Overview](./metadata/ARCHITECTURE.md)
- 🤖 [Agent System Documentation](./metadata/agents/README.md)
- 🔌 [OpenWebUI Integration](./metadata/OPENWEBUI_README.md)
- 🔐 [Authentication Setup](./metadata/TOKEN_AUTH_SETUP.md)

## 🎯 Project Overview

This is an enterprise-grade RAG bot that combines:
- **Multi-Agent System**: Specialized agents for different operations
- **Vector Search**: Milvus-based semantic search
- **Resource Management**: Cluster creation, listing, and management
- **OpenWebUI Integration**: Modern chat interface
- **Authentication**: Secure token-based authentication

## 🚀 Quick Start

```bash
# Start the backend server
cd /home/unixlogin/vayuMaya/Enterprise-Rag-bot
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Start the frontend (in another terminal)
cd user-frontend
npm start
```

For detailed instructions, see [START_SERVERS.md](./metadata/START_SERVERS.md)

## 📂 Project Structure

```
Enterprise-Rag-bot/
├── app/                    # Backend application
│   ├── agents/            # Agent system
│   ├── api/               # API routes
│   ├── config/            # Configuration files
│   ├── services/          # Core services
│   └── main.py            # Application entry point
├── user-frontend/         # React frontend
├── metadata/              # 📚 All documentation
│   ├── INDEX.md          # Documentation index
│   ├── agents/           # Agent documentation
│   └── frontend/         # Frontend documentation
├── tests/                 # 🧪 All test files
│   ├── test_*.py         # Python tests
│   └── test_*.sh         # Shell script tests
├── misc/                  # 🔧 Miscellaneous files
│   ├── docker/           # Docker configurations
│   ├── config/           # Configuration files
│   └── scripts/          # Utility scripts
└── README.md             # This file
```

## 🛠️ Technology Stack

- **Backend**: FastAPI, Python
- **Frontend**: React, TypeScript
- **Vector DB**: Milvus
- **LLM Integration**: OpenAI-compatible APIs
- **UI**: OpenWebUI integration

## 📖 For New Contributors

1. Read the [Quick Start Guide](./metadata/QUICK_START.md)
2. Understand the [Architecture](./metadata/ARCHITECTURE.md)
3. Explore the [Agent System](./metadata/agents/README.md)
4. Check [Recent Updates](./metadata/UPDATES_NOV24_2025.md)

## 🤝 Contributing

Please refer to the documentation in the [`metadata/`](./metadata/) folder for contribution guidelines and project architecture details.

## 📝 License

[Add your license information here]

---

**Note**: All detailed documentation, guides, and architecture documents are located in the [`metadata/`](./metadata/) folder. Please check the [INDEX.md](./metadata/INDEX.md) for a complete catalog.

