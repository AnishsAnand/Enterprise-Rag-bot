# Agentic Copilot Scripts

Phase 1 + 2 of the RAG-driven agentic migration.

## Quick Test (after ingest)

```bash
# 1. Ingest API specs (once)
python3 -m app.scripts.ingest_api_specs

# 2. Test RAG search only (no LLM)
python3 -m app.scripts.test_rag_intent --rag-only

# 3. Test full intent flow (RAG + LLM)
python3 -m app.scripts.test_rag_intent "list clusters"
python3 -m app.scripts.test_rag_intent "show load balancers"
```

## Scripts

### 1. Convert resource_schema.json to RAG chunks

```bash
# Generate markdown files
python3 -m app.scripts.convert_schema_to_rag --write-files

# Output as JSON
python3 -m app.scripts.convert_schema_to_rag --json

# Custom paths
python3 -m app.scripts.convert_schema_to_rag --schema-path /path/to/schema.json --output-dir ./output
```

Output: 33 API spec chunks (one per resource+operation) in markdown format.

### 2. Ingest API specs into RAG

Requires PostgreSQL with `enterprise_rag` table and embeddings configured. Run from project root with dependencies installed (e.g. `pip install -r requirements.txt`).

```bash
# Ingest (requires POSTGRES_* env vars)
python3 -m app.scripts.ingest_api_specs

# With custom schema path
python3 -m app.scripts.ingest_api_specs --schema-path /path/to/resource_schema.json
```

Documents are stored with `source="api_spec"` for filtered retrieval.

### 3. Retrain RAG (clear + bulk scrape + ingest)

Full pipeline: clear knowledge base, optionally bulk-scrape URLs, ingest API specs + md files.

```bash
# Full retrain: clear + API specs + md files from metadata/api_spec_chunks
python3 -m app.scripts.retrain_rag

# With bulk scrape (scrape URLs, then ingest API specs + md)
python3 -m app.scripts.retrain_rag --base-url https://docs.example.com

# Scrape only (no API/md ingest)
python3 -m app.scripts.retrain_rag --base-url https://docs.example.com --scrape-only

# No clear (append to existing)
python3 -m app.scripts.retrain_rag --no-clear

# Hit backend APIs (server must be running)
python3 -m app.scripts.retrain_rag --api-base http://localhost:8000 --base-url https://docs.example.com
```

Or hit the APIs directly:

```bash
# Clear knowledge base (DELETE)
curl -X DELETE http://localhost:8000/api/rag-widget/widget/clear-knowledge

# Bulk scrape (POST, runs in background)
curl -X POST http://localhost:8000/api/rag-widget/widget/bulk-scrape \
  -H "Content-Type: application/json" \
  -d '{"base_url":"https://docs.example.com","max_depth":8,"max_urls":500,"auto_store":true}'
```

### 4. Retrieval (in code)

```python
from app.services.postgres_service import postgres_service

# Search only API specs
results = await postgres_service.search_api_specs("list kubernetes clusters", n_results=5)

# Or use search_documents with source filter
results = await postgres_service.search_documents(
    "list clusters",
    n_results=10,
    source_filter="api_spec",
)
```

## Phase 1 Checklist

- [x] Script to convert `resource_schema.json` → markdown chunks
- [x] Ingest into RAG with `source="api_spec"`
- [x] `search_api_specs()` and `source_filter` in `search_documents()`

## Phase 2 Checklist

- [x] IntentAgent queries RAG via `search_api_specs()` before LLM
- [x] RAG context included in intent prompt
- [x] Params parsed from RAG when matching chunk found; schema fallback
- [x] `api_spec` stored in ConversationState for downstream agents

## Phase 3 Checklist

- [x] `resource_schema.json` removed from `app/config/`
- [x] Schema backup at `metadata/resource_schema_backup.json` for convert/ingest
- [x] APIExecutorService uses RAG via `get_operation_config_async()` for execution
- [x] IntentAgent uses static resource list when schema empty

## Full App Test

Start the app and use the chat UI or API:

```bash
# Start backend (adjust for your setup)
uvicorn app.main:app --reload

# Or via docker-compose
docker-compose up -d
```

Then try: "list clusters", "show me load balancers", "list endpoints". Check logs for:
- `📚 RAG retrieved N API spec chunks for intent`
- `✅ Intent enriched from RAG` (when RAG provides params)
