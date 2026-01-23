# WebUI-Compatible API Documentation

This document describes the OpenWebUI-compatible API layer implemented for Vayu Maya. These APIs replace the OpenWebUI backend while maintaining frontend compatibility.

> **Important:** As of the latest update, the in-memory `chat_persistence.py` router has been removed.
> All chat operations now use the PostgreSQL-backed `webui_chats.py` router for persistent storage.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Database Models](#database-models)
4. [API Endpoints](#api-endpoints)
5. [Authentication](#authentication)
6. [Usage Examples](#usage-examples)
7. [Database Schema](#database-schema)
8. [Configuration](#configuration)

---

## Overview

The WebUI-compatible API provides a chat persistence layer that mirrors OpenWebUI's API structure. This allows:

- **Chat persistence**: Store and retrieve conversation history
- **Organization**: Pin, archive, tag, and folder-organize chats
- **Search**: Full-text search across chat titles
- **Configuration**: Dynamic app configuration for frontends

### Key Features

| Feature | Description |
|---------|-------------|
| Chat CRUD | Create, read, update, delete conversations |
| Pagination | 60 items per page with page-based navigation |
| Pinning | Pin important chats to the top |
| Archiving | Archive old chats without deleting |
| Tagging | Organize chats with custom tags |
| Folders | Group chats into folders |
| Search | Search chats by title |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend                              │
│                   (Angular / React / etc.)                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      API Layer (FastAPI)                     │
│  ┌─────────────────┐  ┌─────────────────┐                   │
│  │ /api/config     │  │ /api/v1/chats   │                   │
│  │ webui_config.py │  │ webui_chats.py  │                   │
│  └─────────────────┘  └─────────────────┘                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Service Layer                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              chat_service.py                         │    │
│  │  - CRUD operations                                   │    │
│  │  - Pagination & filtering                            │    │
│  │  - Tag & folder management                           │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Database Layer                            │
│  ┌──────────┐  ┌──────────┐  ┌──────┐  ┌─────────┐         │
│  │  chats   │  │chat_files│  │ tags │  │ folders │         │
│  └──────────┘  └──────────┘  └──────┘  └─────────┘         │
│                     PostgreSQL                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Database Models

### Location: `app/models/chat_models.py`

### 1. Chat Model

The main chat storage table, compatible with OpenWebUI's schema.

```python
class Chat(Base):
    __tablename__ = "chats"
    
    id = Column(String(36), primary_key=True)      # UUID
    user_id = Column(String(255), nullable=False)  # Owner
    title = Column(Text, nullable=False)           # Chat title
    chat = Column(JSON, nullable=False)            # Messages & history
    created_at = Column(BigInteger)                # Epoch timestamp
    updated_at = Column(BigInteger)                # Epoch timestamp
    share_id = Column(Text, unique=True)           # Sharing link
    archived = Column(Boolean, default=False)      # Archive status
    pinned = Column(Boolean, default=False)        # Pin status
    meta = Column(JSON, default={})                # Tags & metadata
    folder_id = Column(Text, nullable=True)        # Folder reference
```

#### Chat JSON Structure

The `chat` column stores conversation data in this format:

```json
{
  "title": "Chat Title",
  "history": {
    "messages": {
      "msg-uuid-1": {
        "id": "msg-uuid-1",
        "role": "user",
        "content": "Hello!",
        "timestamp": 1704067200
      },
      "msg-uuid-2": {
        "id": "msg-uuid-2",
        "role": "assistant",
        "content": "Hi! How can I help?",
        "timestamp": 1704067201,
        "model": "vayu-maya"
      }
    },
    "currentId": "msg-uuid-2"
  },
  "models": ["vayu-maya"]
}
```

### 2. ChatFile Model

Tracks file attachments in chat messages.

```python
class ChatFile(Base):
    __tablename__ = "chat_files"
    
    id = Column(String(36), primary_key=True)
    user_id = Column(String(255), nullable=False)
    chat_id = Column(String(36), ForeignKey("chats.id"))
    message_id = Column(String(255), nullable=True)
    file_id = Column(String(255), nullable=False)
    created_at = Column(BigInteger)
    updated_at = Column(BigInteger)
```

### 3. Tag Model

User-defined tags for organizing chats.

```python
class Tag(Base):
    __tablename__ = "tags"
    
    id = Column(String(255), primary_key=True)    # Normalized: "my_tag"
    name = Column(String(255), nullable=False)    # Display: "My Tag"
    user_id = Column(String(255), nullable=False)
    created_at = Column(BigInteger)
```

### 4. Folder Model

Folder hierarchy for chat organization.

```python
class Folder(Base):
    __tablename__ = "folders"
    
    id = Column(String(36), primary_key=True)
    name = Column(String(255), nullable=False)
    user_id = Column(String(255), nullable=False)
    parent_id = Column(String(36), nullable=True)  # For nesting
    created_at = Column(BigInteger)
    updated_at = Column(BigInteger)
```

---

## API Endpoints

### Configuration API

**Location:** `app/api/routes/webui_config.py`

#### GET /api/config

Returns application configuration for the frontend.

**Response (Unauthenticated):**
```json
{
  "status": true,
  "name": "Vayu Maya",
  "version": "2.0.0",
  "default_locale": "en-US",
  "oauth": { "providers": {} },
  "features": {
    "auth": true,
    "enable_signup": true,
    "enable_login_form": true,
    "enable_websocket": true
  },
  "onboarding": true
}
```

**Response (Authenticated):**
```json
{
  "status": true,
  "name": "Vayu Maya",
  "version": "2.0.0",
  "features": {
    "enable_channels": true,
    "enable_folders": true,
    "enable_web_search": true,
    ...
  },
  "default_models": "",
  "default_prompt_suggestions": [
    { "title": "List my Kubernetes clusters", "content": "Show me all Kubernetes clusters" },
    { "title": "Create a new VM", "content": "I want to create a new virtual machine" },
    ...
  ],
  "permissions": { ... },
  "file": { "max_size": 104857600, "max_count": 10 }
}
```

---

### Chat API

**Location:** `app/api/routes/webui_chats.py`

All endpoints require user identification via `X-User-Id` header or JWT token.

#### List Chats

```http
GET /api/v1/chats/?page=1&include_pinned=false&include_folders=false
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| page | int | null | Page number (1-indexed), 60 items per page |
| include_pinned | bool | false | Include pinned chats in results |
| include_folders | bool | false | Include chats that are in folders |

**Response:**
```json
[
  {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "title": "Kubernetes Cluster Setup",
    "updated_at": 1704067200,
    "created_at": 1704060000
  }
]
```

#### Create Chat

```http
POST /api/v1/chats/new
Content-Type: application/json

{
  "chat": {
    "title": "New Conversation",
    "history": {
      "messages": {},
      "currentId": null
    }
  },
  "folder_id": null
}
```

**Response:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "user_id": "user123",
  "title": "New Conversation",
  "chat": { ... },
  "created_at": 1704067200,
  "updated_at": 1704067200,
  "archived": false,
  "pinned": false,
  "meta": {},
  "folder_id": null
}
```

#### Get Chat by ID

```http
GET /api/v1/chats/{chat_id}
```

#### Update Chat

```http
POST /api/v1/chats/{chat_id}
Content-Type: application/json

{
  "chat": {
    "title": "Updated Title",
    "history": { ... }
  }
}
```

#### Delete Chat

```http
DELETE /api/v1/chats/{chat_id}
```

#### Toggle Pin

```http
POST /api/v1/chats/{chat_id}/pin
```

#### Toggle Archive

```http
POST /api/v1/chats/{chat_id}/archive
```

#### Search Chats

```http
GET /api/v1/chats/search?text=kubernetes&page=1
```

#### Get Pinned Chats

```http
GET /api/v1/chats/pinned
```

#### Get Archived Chats

```http
GET /api/v1/chats/archived?page=1
```

#### Get All Tags

```http
GET /api/v1/chats/all/tags
```

#### Manage Chat Tags

```http
# Get tags for a chat
GET /api/v1/chats/{chat_id}/tags

# Add tag to chat
POST /api/v1/chats/{chat_id}/tags
Content-Type: application/json
{ "name": "important" }

# Remove tag from chat
DELETE /api/v1/chats/{chat_id}/tags
Content-Type: application/json
{ "name": "important" }
```

#### Move Chat to Folder

```http
POST /api/v1/chats/{chat_id}/folder
Content-Type: application/json
{ "folder_id": "folder-uuid" }
```

#### Update Message

```http
POST /api/v1/chats/{chat_id}/messages/{message_id}
Content-Type: application/json
{ "content": "Updated message content" }
```

---

## Authentication

### Current Implementation

The API uses header-based user identification:

```http
X-User-Id: user123
```

Or Bearer token (JWT):

```http
Authorization: Bearer <token>
```

### User Extraction Logic

```python
def get_current_user_id(request: Request) -> str:
    # 1. Check X-User-Id header
    user_id = request.headers.get("X-User-Id")
    if user_id:
        return user_id
    
    # 2. Check Authorization header (JWT)
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        # Decode JWT and extract user_id
        pass
    
    # 3. Default user for development
    return "default_user"
```

---

## Usage Examples

### cURL Examples

```bash
# Get app configuration
curl http://localhost:8000/api/config

# List chats (page 1)
curl -H "X-User-Id: user123" \
  "http://localhost:8000/api/v1/chats/?page=1"

# Create a new chat
curl -X POST http://localhost:8000/api/v1/chats/new \
  -H "Content-Type: application/json" \
  -H "X-User-Id: user123" \
  -d '{
    "chat": {
      "title": "My New Chat",
      "history": {"messages": {}, "currentId": null}
    }
  }'

# Get specific chat
curl -H "X-User-Id: user123" \
  "http://localhost:8000/api/v1/chats/550e8400-e29b-41d4-a716-446655440000"

# Pin a chat
curl -X POST -H "X-User-Id: user123" \
  "http://localhost:8000/api/v1/chats/550e8400-e29b-41d4-a716-446655440000/pin"

# Search chats
curl -H "X-User-Id: user123" \
  "http://localhost:8000/api/v1/chats/search?text=kubernetes"

# Add tag to chat
curl -X POST http://localhost:8000/api/v1/chats/{id}/tags \
  -H "Content-Type: application/json" \
  -H "X-User-Id: user123" \
  -d '{"name": "important"}'
```

### Python Examples

```python
import requests

BASE_URL = "http://localhost:8000"
HEADERS = {"X-User-Id": "user123"}

# Get config
config = requests.get(f"{BASE_URL}/api/config").json()
print(f"App Name: {config['name']}")

# List chats
chats = requests.get(
    f"{BASE_URL}/api/v1/chats/",
    headers=HEADERS,
    params={"page": 1}
).json()

# Create chat
new_chat = requests.post(
    f"{BASE_URL}/api/v1/chats/new",
    headers=HEADERS,
    json={
        "chat": {
            "title": "Python Test Chat",
            "history": {"messages": {}, "currentId": None}
        }
    }
).json()

print(f"Created chat: {new_chat['id']}")
```

---

## Database Schema

### SQL Migration Script

**Location:** `migrations/001_create_chat_tables.sql`

```sql
-- Chats table
CREATE TABLE IF NOT EXISTS chats (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    title TEXT NOT NULL DEFAULT 'New Chat',
    chat JSONB NOT NULL DEFAULT '{}',
    created_at BIGINT NOT NULL,
    updated_at BIGINT NOT NULL,
    share_id TEXT UNIQUE,
    archived BOOLEAN NOT NULL DEFAULT FALSE,
    pinned BOOLEAN DEFAULT FALSE,
    meta JSONB NOT NULL DEFAULT '{}',
    folder_id TEXT
);

-- Indexes
CREATE INDEX idx_chat_user_id ON chats(user_id);
CREATE INDEX idx_chat_folder_id ON chats(folder_id);
CREATE INDEX idx_chat_user_pinned ON chats(user_id, pinned);
CREATE INDEX idx_chat_user_archived ON chats(user_id, archived);
CREATE INDEX idx_chat_updated_user ON chats(updated_at, user_id);

-- Tags table
CREATE TABLE IF NOT EXISTS tags (
    id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    created_at BIGINT NOT NULL
);

-- Folders table
CREATE TABLE IF NOT EXISTS folders (
    id VARCHAR(36) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    parent_id VARCHAR(36),
    created_at BIGINT NOT NULL,
    updated_at BIGINT NOT NULL
);

-- Chat files table
CREATE TABLE IF NOT EXISTS chat_files (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    chat_id VARCHAR(36) REFERENCES chats(id) ON DELETE CASCADE,
    message_id VARCHAR(255),
    file_id VARCHAR(255) NOT NULL,
    created_at BIGINT NOT NULL,
    updated_at BIGINT NOT NULL
);
```

### Database Commands

```bash
# Access PostgreSQL via Docker
docker exec -it enterprise-rag-postgres psql -U ragbot -d enterprise_rag

# List all tables
docker exec enterprise-rag-postgres psql -U ragbot -d enterprise_rag -c "\dt"

# Describe chats table
docker exec enterprise-rag-postgres psql -U ragbot -d enterprise_rag -c "\d chats"

# Query chats
docker exec enterprise-rag-postgres psql -U ragbot -d enterprise_rag \
  -c "SELECT id, title, user_id, created_at FROM chats LIMIT 10;"

# Run migration manually
docker exec -i enterprise-rag-postgres psql -U ragbot -d enterprise_rag \
  < migrations/001_create_chat_tables.sql
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_NAME` | Vayu Maya | Application name shown in config |
| `DEFAULT_LOCALE` | en-US | Default locale |
| `ENABLE_SIGNUP` | true | Allow new user registration |
| `ENABLE_WEB_SEARCH` | true | Enable web search feature |
| `ENABLE_CODE_EXECUTION` | false | Enable code execution |
| `FILE_MAX_SIZE` | 104857600 | Max file size (100MB) |
| `FILE_MAX_COUNT` | 10 | Max files per upload |

### Feature Flags

All features can be toggled via environment variables:

```bash
# .env file
ENABLE_WEB_SEARCH=true
ENABLE_CODE_EXECUTION=false
ENABLE_IMAGE_GENERATION=false
ENABLE_MEMORIES=true
```

---

## File Structure

```
app/
├── api/
│   └── routes/
│       ├── webui_config.py    # /api/config endpoint
│       └── webui_chats.py     # /api/v1/chats/* endpoints
├── models/
│   ├── __init__.py            # Model exports
│   └── chat_models.py         # SQLAlchemy & Pydantic models
├── services/
│   └── chat_service.py        # Business logic & DB operations
└── main.py                    # Router registration

migrations/
└── 001_create_chat_tables.sql # Database migration
```

---

## Comparison with OpenWebUI

| Feature | OpenWebUI | Vayu Maya |
|---------|-----------|-----------|
| Chat storage | SQLite/PostgreSQL | PostgreSQL |
| Message format | JSON blob | JSON blob (compatible) |
| Pagination | 60/page | 60/page |
| Timestamps | Unix epoch (int) | Unix epoch (int) |
| User auth | JWT + API Keys | Header + JWT |
| Tags | ✅ | ✅ |
| Folders | ✅ | ✅ |
| Pinning | ✅ | ✅ |
| Archiving | ✅ | ✅ |
| Sharing | ✅ | ✅ (share_id) |
| Search | Full-text | Title search |

---

## Future Enhancements

1. **Full-text search** - Search within message content
2. **Sharing** - Generate shareable links for chats
3. **Export** - Export chats as JSON/Markdown
4. **Bulk operations** - Delete/archive multiple chats
5. **Folder CRUD API** - Full folder management endpoints

---

*Last updated: January 2024*
