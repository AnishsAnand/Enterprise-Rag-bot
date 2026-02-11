# Agentic Copilot Scripts

Phase 1 of the RAG-driven agentic migration.

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

### 3. Retrieval (in code)

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

## Next: Phase 2

IntentAgent will query RAG for API specs before/during intent detection.
See `metadata/AGENTIC_COPILOT_BRAINSTORM.md`.
