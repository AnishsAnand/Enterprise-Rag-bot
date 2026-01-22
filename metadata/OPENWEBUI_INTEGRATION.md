# Open WebUI Integration Guide

## Overview

This guide explains how to integrate Open WebUI with the Enterprise RAG Bot to provide a professional, feature-rich chat interface.

## What is Open WebUI?

Open WebUI is an extensible, feature-rich, and user-friendly self-hosted AI interface that supports:
- Multiple LLM providers (Ollama, OpenAI, custom APIs)
- RAG (document upload, knowledge bases)
- Multi-user authentication
- Chat history and conversation management
- Custom pipelines and function calling
- Analytics and monitoring

**Official Repository**: https://github.com/open-webui/open-webui

---

## Architecture Integration

```
┌─────────────────────────────────────────────────────────────┐
│                        User Interface                        │
│  ┌──────────────────────┐       ┌────────────────────────┐  │
│  │   Open WebUI         │       │  Your Angular Frontend │  │
│  │  (Chat Interface)    │       │  (Admin/Monitoring)    │  │
│  └──────────┬───────────┘       └───────────┬────────────┘  │
└─────────────┼───────────────────────────────┼───────────────┘
              │                               │
              ├───────────────────────────────┤
              │                               │
┌─────────────▼───────────────────────────────▼───────────────┐
│              Open WebUI Backend (Pipelines)                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Custom Pipeline: Forward to Enterprise RAG Bot      │   │
│  │  - Route requests to FastAPI backend                 │   │
│  │  - Transform responses                               │   │
│  │  - Handle streaming                                  │   │
│  └────────────────────────┬─────────────────────────────┘   │
└───────────────────────────┼─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│          Your Enterprise RAG Bot (FastAPI Backend)           │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  /api/agent/chat                                     │   │
│  │  /api/agent/conversation/{id}                        │   │
│  │  /api/rag/query                                      │   │
│  └────────────────────────┬─────────────────────────────┘   │
│                           │                                  │
│  ┌────────────────────────▼─────────────────────────────┐   │
│  │  Multi-Agent System (LangChain + LangGraph)          │   │
│  │  - Intent Classifier                                 │   │
│  │  - Cluster Creation Handler                          │   │
│  │  - Document Search Handler                           │   │
│  │  - Validation Agent                                  │   │
│  └────────────────────────┬─────────────────────────────┘   │
└───────────────────────────┼─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                    Data & AI Layers                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                 │
│  │ ChromaDB │  │  Milvus  │  │PostgreSQL│                 │
│  └──────────┘  └──────────┘  └──────────┘                 │
└──────────────────────────────────────────────────────────────┘
```

---

## Installation Methods

### Method 1: Docker with Separate Services (Recommended)

This keeps Open WebUI and your RAG bot as separate services.

**Step 1**: Create `docker-compose.openwebui.yml`

```yaml
version: '3.8'

services:
  # Open WebUI Service
  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    container_name: enterprise-rag-openwebui
    ports:
      - "3000:8080"
    environment:
      - OPENAI_API_BASE_URL=http://enterprise-rag-bot:8000/api/v1
      - OPENAI_API_KEY=your-api-key-here
      - ENABLE_RAG=true
      - ENABLE_OLLAMA_API=false
      - WEBUI_AUTH=true
      - WEBUI_SECRET_KEY=your-secret-key-change-this
      - DEFAULT_MODELS=enterprise-rag-bot
    volumes:
      - open-webui-data:/app/backend/data
    restart: always
    depends_on:
      - enterprise-rag-bot
    networks:
      - rag-network

  # Your Existing RAG Bot
  enterprise-rag-bot:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: enterprise-rag-bot
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./app:/app/app
      - ./uploads:/app/uploads
    restart: always
    networks:
      - rag-network

volumes:
  open-webui-data:

networks:
  rag-network:
    driver: bridge
```

**Step 2**: Start services

```bash
cd /home/unixlogin/vayuMaya/Enterprise-Rag-bot
docker-compose -f docker-compose.openwebui.yml up -d
```

**Step 3**: Access Open WebUI at `http://localhost:3000`

---

### Method 2: Open WebUI with Custom Pipeline

Create a custom pipeline to integrate with your backend.

**Step 1**: Install Open WebUI

```bash
docker run -d \
  -p 3000:8080 \
  --add-host=host.docker.internal:host-gateway \
  -v open-webui:/app/backend/data \
  --name open-webui \
  --restart always \
  ghcr.io/open-webui/open-webui:main
```

**Step 2**: Create Custom Pipeline

Create `pipelines/enterprise_rag_pipeline.py`:

```python
"""
Title: Enterprise RAG Bot Pipeline
Author: Your Name
Date: 2025-12-08
Version: 1.0
License: MIT
Description: Custom pipeline to integrate Open WebUI with Enterprise RAG Bot
"""

from typing import List, Union, Generator, Iterator
from pydantic import BaseModel
import requests
import json
import os


class Pipeline:
    """Pipeline to connect Open WebUI to Enterprise RAG Bot."""

    class Valves(BaseModel):
        """Configuration for the pipeline."""
        ENTERPRISE_RAG_URL: str = "http://localhost:8000"
        API_KEY: str = ""
        ENABLE_STREAMING: bool = True
        MAX_TOKENS: int = 4096
        TEMPERATURE: float = 0.7

    def __init__(self):
        self.name = "Enterprise RAG Bot"
        self.valves = self.Valves()

    async def on_startup(self):
        """Called when the pipeline starts."""
        print(f"Pipeline {self.name} initialized")
        print(f"Connecting to: {self.valves.ENTERPRISE_RAG_URL}")

    async def on_shutdown(self):
        """Called when the pipeline shuts down."""
        print(f"Pipeline {self.name} shutting down")

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        """
        Main pipeline function that processes chat messages.
        
        Args:
            user_message: The latest user message
            model_id: The model identifier
            messages: Full conversation history
            body: Additional parameters from Open WebUI
        
        Returns:
            Response from the Enterprise RAG Bot
        """
        
        # Extract user info
        user_id = body.get("user", {}).get("id", "anonymous")
        user_roles = body.get("user", {}).get("roles", ["user"])
        
        # Prepare request payload for Enterprise RAG Bot
        payload = {
            "message": user_message,
            "user_id": user_id,
            "user_roles": user_roles,
            "conversation_history": messages[:-1],  # Exclude current message
            "metadata": {
                "model": model_id,
                "temperature": body.get("temperature", self.valves.TEMPERATURE),
                "max_tokens": body.get("max_tokens", self.valves.MAX_TOKENS),
            }
        }
        
        # Make request to Enterprise RAG Bot
        headers = {
            "Content-Type": "application/json",
        }
        
        if self.valves.API_KEY:
            headers["Authorization"] = f"Bearer {self.valves.API_KEY}"
        
        try:
            response = requests.post(
                f"{self.valves.ENTERPRISE_RAG_URL}/api/agent/chat",
                json=payload,
                headers=headers,
                stream=self.valves.ENABLE_STREAMING,
                timeout=120
            )
            
            response.raise_for_status()
            
            if self.valves.ENABLE_STREAMING:
                return self._stream_response(response)
            else:
                result = response.json()
                return result.get("response", "No response from agent")
                
        except requests.exceptions.RequestException as e:
            return f"Error connecting to Enterprise RAG Bot: {str(e)}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"

    def _stream_response(self, response) -> Generator:
        """Stream the response from the backend."""
        for chunk in response.iter_lines():
            if chunk:
                try:
                    data = json.loads(chunk.decode('utf-8'))
                    if "content" in data:
                        yield data["content"]
                except json.JSONDecodeError:
                    continue
```

**Step 3**: Install the pipeline in Open WebUI

1. Go to `http://localhost:3000`
2. Navigate to **Settings** → **Admin Settings** → **Pipelines**
3. Upload `enterprise_rag_pipeline.py`
4. Configure the valves:
   - `ENTERPRISE_RAG_URL`: `http://host.docker.internal:8000`
   - `API_KEY`: Your API token

---

## Backend Modifications

To make your Enterprise RAG Bot compatible with Open WebUI, you need to add OpenAI-compatible endpoints.

### Create OpenAI-Compatible API

Create `app/routers/openai_compatible.py`:

```python
"""
OpenAI-compatible API endpoints for Open WebUI integration.
"""

from fastapi import APIRouter, HTTPException, Header, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import time
import uuid
from app.services.agent_service import AgentService
from app.core.auth import verify_api_key

router = APIRouter(prefix="/api/v1", tags=["OpenAI Compatible"])


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 4096
    stream: Optional[bool] = False
    user: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Optional[Dict[str, int]] = None


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "enterprise-rag-bot"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


@router.get("/models")
async def list_models(
    authorization: Optional[str] = Header(None)
) -> ModelListResponse:
    """List available models (OpenAI-compatible endpoint)."""
    
    return ModelListResponse(
        data=[
            ModelInfo(
                id="enterprise-rag-bot",
                created=int(time.time()),
                owned_by="enterprise"
            )
        ]
    )


@router.post("/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    authorization: Optional[str] = Header(None),
    agent_service: AgentService = Depends()
) -> ChatCompletionResponse:
    """
    OpenAI-compatible chat completions endpoint.
    This allows Open WebUI to communicate with the Enterprise RAG Bot.
    """
    
    # Verify API key if provided
    if authorization:
        token = authorization.replace("Bearer ", "")
        # Add your token verification logic here
    
    # Extract the latest user message
    user_message = request.messages[-1].content if request.messages else ""
    
    # Extract conversation history
    conversation_history = [
        {"role": msg.role, "content": msg.content}
        for msg in request.messages[:-1]
    ]
    
    try:
        # Call your agent service
        result = await agent_service.process_message(
            message=user_message,
            user_id=request.user or "openwebui_user",
            user_roles=["user"],
            conversation_history=conversation_history
        )
        
        # Format response in OpenAI format
        response_content = result.get("response", "No response generated")
        
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=request.model,
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_content
                    },
                    "finish_reason": "stop"
                }
            ],
            usage={
                "prompt_tokens": len(user_message.split()),
                "completion_tokens": len(response_content.split()),
                "total_tokens": len(user_message.split()) + len(response_content.split())
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Update `app/main.py`

Add the OpenAI-compatible router:

```python
from app.routers import openai_compatible

# Include the router
app.include_router(openai_compatible.router)
```

---

## Configuration

### Open WebUI Configuration

Set these environment variables for Open WebUI:

```bash
# In docker-compose or .env
OPENAI_API_BASE_URL=http://enterprise-rag-bot:8000/api/v1
OPENAI_API_KEY=your-api-key-here

# Enable features
ENABLE_RAG=true
ENABLE_WEB_SEARCH=false
ENABLE_OLLAMA_API=false

# Authentication
WEBUI_AUTH=true
WEBUI_SECRET_KEY=generate-a-secure-random-key

# Default model
DEFAULT_MODELS=enterprise-rag-bot
```

### Enterprise RAG Bot Configuration

Add to your `.env`:

```bash
# Open WebUI Integration
ALLOW_OPENAI_COMPATIBLE_API=true
OPENWEBUI_API_KEY=your-secure-api-key
CORS_ORIGINS=http://localhost:3000,https://your-openwebui-domain.com
```

---

## Testing the Integration

### Test 1: Check Models Endpoint

```bash
curl http://localhost:8000/api/v1/models \
  -H "Authorization: Bearer your-api-key"
```

Expected response:
```json
{
  "object": "list",
  "data": [
    {
      "id": "enterprise-rag-bot",
      "object": "model",
      "created": 1733675432,
      "owned_by": "enterprise-rag-bot"
    }
  ]
}
```

### Test 2: Chat Completion

```bash
curl http://localhost:8000/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "model": "enterprise-rag-bot",
    "messages": [
      {"role": "user", "content": "Create a Kubernetes cluster"}
    ]
  }'
```

### Test 3: Open WebUI Integration

1. Open `http://localhost:3000`
2. Create an account
3. Go to **Settings** → **Connections**
4. Verify "enterprise-rag-bot" appears in model list
5. Start chatting!

---

## Advanced Features

### 1. Custom RAG Document Upload

Open WebUI supports document uploads. To integrate with your RAG system:

```python
# Add endpoint in app/routers/openai_compatible.py

@router.post("/embeddings")
async def create_embeddings(files: List[UploadFile]):
    """Handle document uploads from Open WebUI."""
    # Process files and store in ChromaDB/Milvus
    pass
```

### 2. Function Calling

Enable your agents to call functions through Open WebUI:

```python
# Define tools/functions
tools = [
    {
        "type": "function",
        "function": {
            "name": "create_cluster",
            "description": "Create a new Kubernetes cluster",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "region": {"type": "string"},
                    "version": {"type": "string"}
                },
                "required": ["name", "region"]
            }
        }
    }
]
```

### 3. User Analytics

Track usage with Open WebUI's built-in analytics or integrate Langfuse:

```bash
# In Open WebUI .env
ENABLE_LANGFUSE=true
LANGFUSE_PUBLIC_KEY=your-key
LANGFUSE_SECRET_KEY=your-secret
```

---

## Benefits of Using Open WebUI

### For End Users:
✅ Beautiful, modern chat interface  
✅ Chat history and search  
✅ Document uploads (RAG)  
✅ Voice input support  
✅ Mobile-friendly  
✅ Multi-language support  

### For Developers:
✅ No frontend development needed  
✅ Focus on backend AI logic  
✅ Built-in user authentication  
✅ API monitoring and analytics  
✅ Easy to customize and extend  

### For Administrators:
✅ User management and RBAC  
✅ Usage monitoring  
✅ Cost tracking  
✅ Rate limiting  
✅ Audit logs  

---

## Deployment Options

### Option 1: Same Server

```
┌────────────────────────────────┐
│  Server (localhost)            │
│  ├─ Open WebUI :3000          │
│  ├─ RAG Bot API :8000         │
│  ├─ PostgreSQL :5432          │
│  └─ Milvus :19530             │
└────────────────────────────────┘
```

### Option 2: Separate Services

```
┌──────────────────┐      ┌──────────────────┐
│  Open WebUI      │      │  RAG Bot API     │
│  (Frontend)      │─────▶│  (Backend)       │
│  :3000           │      │  :8000           │
└──────────────────┘      └──────────────────┘
```

### Option 3: Production with Load Balancer

```
                   ┌──────────────┐
                   │   Nginx LB   │
                   └───────┬──────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
┌───────▼────────┐ ┌───────▼────────┐ ┌──────▼─────────┐
│ Open WebUI 1   │ │ Open WebUI 2   │ │ RAG Bot API    │
└────────────────┘ └────────────────┘ └────────────────┘
```

---

## Troubleshooting

### Issue: "Model not found"

**Solution**: Check that your OpenAI-compatible API is responding:
```bash
curl http://localhost:8000/api/v1/models
```

### Issue: "Connection refused"

**Solution**: Ensure both services are running and can communicate:
```bash
# Check RAG Bot
curl http://localhost:8000/health

# Check Open WebUI
curl http://localhost:3000
```

### Issue: "Unauthorized"

**Solution**: Verify API key configuration:
- Check `OPENAI_API_KEY` in Open WebUI
- Check token verification in your backend

### Issue: "Slow responses"

**Solution**: Enable streaming:
```python
# In ChatCompletionRequest
stream: bool = True
```

---

## Migration Plan

### Phase 1: Setup (Week 1)
- [ ] Install Open WebUI
- [ ] Add OpenAI-compatible endpoints
- [ ] Test basic chat functionality
- [ ] Configure authentication

### Phase 2: Integration (Week 2)
- [ ] Integrate RAG document upload
- [ ] Connect to existing ChromaDB/Milvus
- [ ] Test multi-agent workflows
- [ ] Set up user roles

### Phase 3: Enhancement (Week 3)
- [ ] Custom branding
- [ ] Analytics integration
- [ ] Performance optimization
- [ ] User training

### Phase 4: Production (Week 4)
- [ ] Load testing
- [ ] Security audit
- [ ] Backup procedures
- [ ] Monitoring setup
- [ ] Go live!

---

## Resources

- **Open WebUI Docs**: https://docs.openwebui.com
- **GitHub**: https://github.com/open-webui/open-webui
- **Discord Community**: https://discord.gg/5rJgQTnV4s
- **Pipelines Guide**: https://docs.openwebui.com/pipelines

---

## Conclusion

Open WebUI provides a production-ready frontend for your Enterprise RAG Bot with minimal integration effort. It saves months of frontend development while providing features like:

- ✨ Modern UI/UX
- 👥 User management
- 📚 RAG support
- 📊 Analytics
- 🔧 Extensibility

**Next Steps:**
1. Install Open WebUI using Docker
2. Add OpenAI-compatible endpoints to your backend
3. Test the integration
4. Customize and deploy!

---

**Questions?** Check the troubleshooting section or consult the Open WebUI documentation.

