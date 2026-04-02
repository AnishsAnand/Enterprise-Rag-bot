# Manual smoke tests (`misc/tests`)

This folder holds **optional shell scripts** to hit the running FastAPI app. It is **not** a full automated test suite.

Legacy **Python** tests (old multi-agent stack, cluster/endpoint/model scripts, persistence tests, etc.) have been **removed** as the product now uses **Google ADK** (`app/adk/`). Add new automated tests under `app/` or your CI layout if needed.

---

## Contents

| Script | Purpose |
|--------|--------|
| **`test_backend.sh`** | Quick checks against **`POST /api/widget/query`** (expects backend on **port 8001**). Uses `jq` on the JSON field **`answer`**. |
| **`test_backend_standalone.sh`** | Starts **`uvicorn`** on **8001** from the repo root, then probes **`/health`**, **`/api/v1/models`**, **`/api/v1/chat/completions`**. |
| **`test_openai_endpoints.sh`** | Exercises **OpenWebUI-compatible** routes: **`/health`**, **`/api/v1/models`**, **`/api/v1/chat/completions`** (non-streaming, RAG-style question, streaming sample). |

All scripts assume **`BASE_URL=http://localhost:8001`** unless you edit the file.

---

## Prerequisites

- Backend running (or let **`test_backend_standalone.sh`** start it).
- **`curl`**, **`jq`** on `PATH`.
- For chat/widget calls: valid env / auth as required by your deployment.

---

## Run (from repository root)

```bash
chmod +x misc/tests/*.sh

# Widget / ADK path (server must already listen on 8001)
./misc/tests/test_backend.sh

# OpenAI-compatible API checks
./misc/tests/test_openai_endpoints.sh

# Standalone: start server + smoke test (uses repo root detected from script location)
./misc/tests/test_backend_standalone.sh
```

---

## Related documentation

- [`metadata/`](../metadata/) — project docs
- [Architecture diagram](../metadata/ARCHITECTURE_DIAGRAM.md)
- [Database schema](../metadata/DATABASE_SCHEMA.md)

For automated coverage of the current stack, prefer tests that call **`get_adk_runner()`** or HTTP endpoints under **`/api`**, not the removed legacy agent modules.

---

*Questions: see [`metadata/`](../metadata/) and [`app/`](../app/).*
