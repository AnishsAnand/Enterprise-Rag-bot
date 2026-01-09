# Enterprise RAG Bot - Production Deployment Guide

## Overview

This comprehensive guide walks through deploying the Enterprise RAG Bot as a production service with proper PostgreSQL persistence, RAG accuracy improvements, and Open WebUI integration.

## Prerequisites

- Docker & Docker Compose (latest versions)
- 4GB+ RAM available
- 10GB+ disk space
- Modern Linux kernel (5.1+) for optimal performance
- Environment file with proper secrets

## Quick Start

### 1. Prepare Environment

```bash
# Clone the repository
git clone <your-repo>
cd enterprise-rag-bot

# Copy and configure environment
cp .env.example .env
```

Edit `.env` with your configuration:

```env
# Critical - Change these values!
POSTGRES_PASSWORD=your-secure-password
REDIS_PASSWORD=your-redis-password
SECRET_KEY=your-32-char-secret-key-here
OPENROUTER_API_KEY=your-api-key
OPENWEBUI_API_KEY=your-webui-api-key
WEBUI_SECRET_KEY=your-webui-secret

# Optional - Customize these as needed
ENV=production
LOG_LEVEL=info
MIN_RELEVANCE_THRESHOLD=0.5
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:4200
```

### 2. Start Services

```bash
# Start all services (PostgreSQL, Redis, RAG Bot, Open WebUI, Nginx)
docker-compose -f docker-compose.prod.yml up -d

# Verify all services are running
docker-compose -f docker-compose.prod.yml ps

# Check logs
docker-compose -f docker-compose.prod.yml logs -f enterprise-rag-bot
```

### 3. Verify Deployment

```bash
# Check health status
curl http://localhost:8000/health

# Access Open WebUI
curl http://localhost:3000

# Check API documentation
curl http://localhost:8000/docs
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Nginx (Reverse Proxy)                 │
│                  (Port 80/443, SSL Termination)              │
└────────────┬──────────────────────────┬──────────────────────┘
             │                          │
      ┌──────▼──────┐         ┌─────────▼─────────┐
      │  Open WebUI │         │  RAG Bot Backend  │
      │  (Port 3000)│         │  (Port 8000/8001) │
      └──────┬──────┘         └────┬──────────┬───┘
             │                     │          │
      ┌──────▼────────────────┬────▼──────┬───▼─────────┐
      │   PostgreSQL + pgvector│   Redis    │  AI Services│
      │   (Vector Database)   │  (Cache)   │ (Embedding) │
      └───────────────────────┴────────────┴─────────────┘
```

## Key Components

### PostgreSQL with pgvector
- **Purpose**: Vector storage and semantic search
- **Port**: 5432 (internal), 5433 (host)
- **Data**: Persisted in `postgres_data` volume
- **Extensions**: pgvector (HNSW indexing)

### Redis
- **Purpose**: Session cache, rate limiting
- **Port**: 6379 (internal only)
- **Data**: Persisted in `redis_data` volume
- **Memory**: 256MB max with LRU eviction

### RAG Bot Backend
- **Purpose**: Main application (authentication, RAG search, API)
- **Ports**: 8000 (admin), 8001 (user)
- **Features**:
  - Database-backed authentication
  - RAG search with semantic reranking
  - OpenAI-compatible API
  - Health checks & metrics

### Open WebUI
- **Purpose**: Chat interface for RAG system
- **Port**: 3000
- **Integration**: Connects to RAG Bot via OpenAI API

### Angular Admin (Optional)
- **Purpose**: Admin dashboard for document management
- **Port**: 4200
- **Features**: Upload, train, manage knowledge base

## Production Considerations

### Security

1. **HTTPS/SSL**
   ```bash
   # Generate self-signed certificate
   openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
     -keyout ssl/key.pem -out ssl/cert.pem
   ```

2. **Environment Variables**
   - Never commit `.env` to version control
   - Use strong passwords (32+ characters)
   - Rotate API keys regularly

3. **Database Access**
   - PostgreSQL only accessible via internal Docker network
   - Use strong passwords and connection pooling
   - Enable SSL for external connections

### Scalability

1. **Horizontal Scaling**
   ```yaml
   # In docker-compose.prod.yml, increase replicas:
   enterprise-rag-bot:
     deploy:
       replicas: 3
   ```

2. **Database Optimization**
   - Monitor pgvector index performance
   - Adjust connection pooling based on load
   - Use read replicas for high traffic

3. **Caching Strategy**
   - Redis for session management
   - Application-level caching for embeddings
   - CDN for static assets

### Monitoring & Logging

1. **Health Checks**
   ```bash
   # Liveness check
   curl http://localhost:8000/health/live

   # Readiness check
   curl http://localhost:8000/health/ready

   # Full status
   curl http://localhost:8000/health
   ```

2. **Logging**
   - All logs output to stdout (Docker capture)
   - Log rotation configured in docker-compose
   - Max 50MB per file, 5 files retained

3. **Metrics**
   ```bash
   curl http://localhost:8000/metrics
   ```

## Database Management

### Backup

```bash
# Backup PostgreSQL
docker-compose exec postgres pg_dump -U ragbot -d ragbot_db > backup.sql

# Restore from backup
docker-compose exec -T postgres psql -U ragbot -d ragbot_db < backup.sql
```

### Schema Migrations

```bash
# The database schema is created automatically on startup
# Existing migrations are in /scripts/postgres-init.sql

# Manual migration (if needed)
docker-compose exec postgres psql -U ragbot -d ragbot_db < migration.sql
```

### Monitoring Queries

```bash
# Connect to PostgreSQL
docker-compose exec postgres psql -U ragbot -d ragbot_db

# Check users
SELECT id, username, email, role FROM users;

# Check documents
SELECT id, title, status, chunk_count FROM documents;

# Check RAG queries
SELECT id, query_text, user_id, retrieved_chunks FROM rag_queries ORDER BY created_at DESC;

# Check vector storage
SELECT COUNT(*) FROM vector_store.documents;

# Exit
\q
```

## Authentication

### Login Endpoints

**Register:**
```bash
curl -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "myuser",
    "email": "user@example.com",
    "password": "securepass123",
    "full_name": "My Name"
  }'
```

**Login:**
```bash
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "myuser",
    "password": "securepass123"
  }'
```

**Response:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "token_type": "bearer",
  "user": {
    "id": 1,
    "username": "myuser",
    "email": "user@example.com",
    "role": "user",
    "is_active": true,
    "is_verified": false
  }
}
```

## RAG Search

### Basic Search

```bash
curl -X POST http://localhost:8000/api/rag/search \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do I create a Kubernetes cluster?"
  }'
```

**Response includes:**
- Relevant documents from knowledge base
- Confidence scores for each result
- Relevance reasoning
- Search quality assessment
- No generic fallback responses

### Upload Documents

```bash
# Single document
curl -X POST http://localhost:8000/api/rag/upload \
  -H "Authorization: Bearer <access_token>" \
  -F "file=@document.pdf"

# Bulk documents
for file in *.pdf; do
  curl -X POST http://localhost:8000/api/rag/upload \
    -H "Authorization: Bearer <access_token>" \
    -F "file=@$file"
done
```

### Train Knowledge Base

```bash
curl -X POST http://localhost:8000/api/rag/train \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json"
```

## Open WebUI Integration

### Configuration

Open WebUI automatically connects to the RAG Bot via:
- **API Base**: `http://enterprise-rag-bot:8000/api/v1`
- **API Key**: Configured in `.env` as `OPENWEBUI_API_KEY`

### Features

1. **Chat Interface**: Talk to the RAG system
2. **Document Upload**: Add training data
3. **Conversation History**: Persisted queries and responses
4. **User Management**: Role-based access

## Troubleshooting

### Service Won't Start

```bash
# Check logs
docker-compose logs enterprise-rag-bot

# Verify services
docker-compose ps

# Check network
docker network ls
docker inspect enterprise-rag-bot_rag-network
```

### PostgreSQL Connection Issues

```bash
# Test connection from RAG bot container
docker-compose exec enterprise-rag-bot \
  psql postgresql://ragbot:password@postgres:5432/ragbot_db

# Check PostgreSQL logs
docker-compose logs postgres
```

### Slow Search Results

```bash
# Check index status
docker-compose exec postgres psql -U ragbot -d ragbot_db \
  -c "SELECT * FROM pg_stat_user_indexes WHERE schemaname='vector_store';"

# Reindex if needed
docker-compose exec postgres psql -U ragbot -d ragbot_db \
  -c "REINDEX INDEX idx_documents_embedding;"
```

### Out of Memory

```bash
# Check resource usage
docker stats

# Increase limits in docker-compose.prod.yml:
deploy:
  resources:
    limits:
      memory: 4G
      cpus: '2'
```

## Maintenance

### Regular Tasks

- **Daily**: Monitor logs, check health
- **Weekly**: Backup database
- **Monthly**: Review analytics, optimize indexes
- **Quarterly**: Security updates, dependency patches

### Updates

```bash
# Pull latest code
git pull origin main

# Rebuild images
docker-compose build --no-cache

# Restart services
docker-compose down
docker-compose up -d
```

## Performance Tuning

### PostgreSQL

In docker-compose, adjust:
```yaml
environment:
  POSTGRES_INIT_ARGS: "-c shared_buffers=512MB -c effective_cache_size=2GB"
```

### Redis

```yaml
environment:
  REDIS_MAXMEMORY: "512mb"
  REDIS_MAXMEMORY_POLICY: "allkeys-lru"
```

### RAG Parameters

In `.env`:
```env
MIN_RELEVANCE_THRESHOLD=0.5        # Raise for stricter filtering
ENABLE_QUERY_EXPANSION=true         # Slower, more accurate
ENABLE_SEMANTIC_RERANK=true         # Extra processing time
MAX_CHUNKS_RETURN=10                # More context = more tokens
```

## Support & Issues

For issues:
1. Check logs: `docker-compose logs`
2. Verify health: `curl http://localhost:8000/health`
3. Review PostgreSQL: Connect and query
4. Check memory/disk: `docker stats` and `df -h`

---

**Version**: 2.0.0  
**Last Updated**: 2024  
**Maintained By**: Tata Communications
```

This comprehensive deployment guide covers everything needed to run the production system including all the improvements made.
