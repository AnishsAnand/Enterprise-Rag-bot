# Enterprise RAG Bot — Complete Architecture

```
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║                            ANGULAR FRONTEND (port 4200)                                 ║
║                                                                                         ║
║   User types message → POST /api/widget/query                                           ║
║   Headers: Authorization: Bearer <keycloak-token>, usertype: ENG|CUS                    ║
╚══════════════════════════════════╤═══════════════════════════════════════════════════════╝
                                   │
                                   │ HTTP POST (JSON)
                                   ▼
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║  DOCKER COMPOSE NETWORK (rag-network-kv)                                                ║
║                                                                                         ║
║  ┌─────────────┐  ┌──────────┐  ┌──────────────┐  ┌─────────────┐                      ║
║  │ PostgreSQL  │  │  Redis   │  │  Prometheus  │  │   Grafana   │                      ║
║  │   pgvector  │  │          │  │              │  │             │                      ║
║  │  :5400      │  │  :6378   │  │  :9091       │  │  :3001      │                      ║
║  │             │  │          │  │              │  │             │                      ║
║  │ 67 API spec │  │ Response │  │ Scrapes      │  │ Visualizes  │                      ║
║  │ documents   │  │ caching  │  │ /metrics     │  │ metrics     │                      ║
║  │ Sessions    │  │          │  │              │  │             │                      ║
║  │ Chat hist.  │  │          │  │              │  │             │                      ║
║  └──────┬──────┘  └────┬─────┘  └──────────────┘  └─────────────┘                      ║
║         │              │                                                                ║
║         │              │                                                                ║
║  ┌──────▼──────────────▼────────────────────────────────────────────────────────────┐   ║
║  │                                                                                  │   ║
║  │   BACKEND CONTAINER  (backend-kv :8004 → 8000)                                   │   ║
║  │                                                                                  │   ║
║  │ ╔════════════════════════════════════════════════════════════════════════════════╗ │   ║
║  │ ║  LAYER 1 — FastAPI Routes (app/main.py)                                      ║ │   ║
║  │ ║                                                                               ║ │   ║
║  │ ║  /api/widget/query ─────── Main chat (→ ADK)                                  ║ │   ║
║  │ ║  /api/agent/chat ───────── Alt chat  (→ ADK)                                  ║ │   ║
║  │ ║  /v1/chat/completions ──── OpenAI compat (→ ADK)                              ║ │   ║
║  │ ║  /api/widget/scrape ────── Web scraping (→ PostgreSQL)                        ║ │   ║
║  │ ║  /api/widget/upload-file ─ File upload  (→ PostgreSQL)                        ║ │   ║
║  │ ║  /health ───────────────── Liveness check                                     ║ │   ║
║  │ ╚════════════════╤═══════════════════════════════════════════════════════════════╝ │   ║
║  │                  │                                                                │   ║
║  │                  │ Extract: auth_token, user_type, session_id                     │   ║
║  │                  │ (40 lines — zero logic, just pass-through)                     │   ║
║  │                  ▼                                                                │   ║
║  │ ╔════════════════════════════════════════════════════════════════════════════════╗ │   ║
║  │ ║  LAYER 2 — ADK RUNNER (app/adk/runner.py)          ★ THE SINGLE BRAIN ★      ║ │   ║
║  │ ║                                                                               ║ │   ║
║  │ ║  ┌─────────────────────────────────────────────────────────────────────────┐  ║ │   ║
║  │ ║  │  QUERY CLASSIFIER (two-tier regex system)                               │  ║ │   ║
║  │ ║  │                                                                         │  ║ │   ║
║  │ ║  │  Priority 1:  "1", "2", "3"           ─────── engagement_selection      │  ║ │   ║
║  │ ║  │  Priority 2:  "hello", "hi"           ─────── greeting                  │  ║ │   ║
║  │ ║  │  Priority 3:  "what can you do"       ─────── greeting (capabilities)   │  ║ │   ║
║  │ ║  │  Priority 4:  "my engagements"        ─────── engagement                │  ║ │   ║
║  │ ║  │  Priority 5:  "how to" / "metrics"    ─────── rag  (STRONG: always)     │  ║ │   ║
║  │ ║  │  Priority 6:  "pod" / "log" (no verb) ─────── rag  (WEAK: if no verb)  │  ║ │   ║
║  │ ║  │  Priority 7:  "show clusters"         ─────── resource_list_k8s_*       │  ║ │   ║
║  │ ║  │  Priority 8:  everything else         ─────── rag  (fallback)           │  ║ │   ║
║  │ ║  └───────┬───────────┬──────────────┬─────────────────────┬────────────────┘  ║ │   ║
║  │ ║          │           │              │                     │                    ║ │   ║
║  │ ║          ▼           ▼              ▼                     ▼                    ║ │   ║
║  │ ║  ┌────────────┐ ┌──────────┐ ┌────────────┐  ┌─────────────────────────────┐  ║ │   ║
║  │ ║  │  GREETING  │ │ENGAGEMENT│ │  RESOURCE   │  │           RAG               │  ║ │   ║
║  │ ║  │  Handler   │ │ Handler  │ │  Handler    │  │         Handler             │  ║ │   ║
║  │ ║  │            │ │          │ │             │  │                             │  ║ │   ║
║  │ ║  │ Send to    │ │ List     │ │ 1. Check    │  │ 1. Vector search pgvector  │  ║ │   ║
║  │ ║  │ LLM via    │ │ user's   │ │    engage-  │  │ 2. Filter by relevance     │  ║ │   ║
║  │ ║  │ ADK Agent  │ │ engage-  │ │    ment     │  │ 3. Extract sources (URL,   │  ║ │   ║
║  │ ║  │            │ │ ments    │ │ 2. If none: │  │    title, score, preview)  │  ║ │   ║
║  │ ║  │ Returns:   │ │          │ │    prompt   │  │ 4. Extract images          │  ║ │   ║
║  │ ║  │ "Hello!    │ │ Or show  │ │    user to  │  │ 5. Build numbered prompt   │  ║ │   ║
║  │ ║  │ I'm Vayu   │ │ current  │ │    select   │  │    [Source 1] [Source 2]   │  ║ │   ║
║  │ ║  │ Maya..."   │ │ active   │ │ 3. Dispatch │  │ 6. Send to LLM via ADK    │  ║ │   ║
║  │ ║  │            │ │ one      │ │    tool fn  │  │ 7. Return answer +         │  ║ │   ║
║  │ ║  │            │ │          │ │ 4. Format   │  │    sources + images +      │  ║ │   ║
║  │ ║  │            │ │ Or save  │ │    result   │  │    confidence + follow-ups │  ║ │   ║
║  │ ║  │            │ │ user's   │ │    as       │  │                             │  ║ │   ║
║  │ ║  │            │ │ choice   │ │    Markdown │  │                             │  ║ │   ║
║  │ ║  └────────────┘ └──────────┘ └──────┬─────┘  └─────────────────────────────┘  ║ │   ║
║  │ ╚═════════════════════════════════════╪══════════════════════════════════════════╝ │   ║
║  │                                       │                                           │   ║
║  │                                       │ dispatch tool by name                     │   ║
║  │                                       ▼                                           │   ║
║  │ ╔════════════════════════════════════════════════════════════════════════════════╗ │   ║
║  │ ║  LAYER 3 — ADK TOOLS (app/adk/tools.py)    15 tool functions                 ║ │   ║
║  │ ║                                                                               ║ │   ║
║  │ ║  search_knowledge_base()     get_engagements()      select_engagement()       ║ │   ║
║  │ ║  list_k8s_clusters()         get_cluster_details()  get_cluster_report()      ║ │   ║
║  │ ║  check_cluster_name()        list_vms()             list_firewalls()          ║ │   ║
║  │ ║  list_load_balancers()       list_managed_services()                          ║ │   ║
║  │ ║  list_zones()                list_environments()    list_business_units()     ║ │   ║
║  │ ║                                                                               ║ │   ║
║  │ ║  Each tool: builds context (auth_token, engagement_id, user_type)             ║ │   ║
║  │ ║             then delegates to the right Resource Agent                        ║ │   ║
║  │ ╚════════════════════════════╤═══════════════════════════════════════════════════╝ │   ║
║  │                              │                                                    │   ║
║  │                              ▼                                                    │   ║
║  │ ╔════════════════════════════════════════════════════════════════════════════════╗ │   ║
║  │ ║  LAYER 4 — RESOURCE AGENTS (app/agents/resource_agents/)                      ║ │   ║
║  │ ║                                                                               ║ │   ║
║  │ ║  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  ║ │   ║
║  │ ║  │   K8sAgent   │  │   VMAgent    │  │ NetworkAgent │  │ ManagedService   │  ║ │   ║
║  │ ║  │              │  │              │  │              │  │     Agent        │  ║ │   ║
║  │ ║  │ • clusters   │  │ • list VMs   │  │ • firewalls  │  │                  │  ║ │   ║
║  │ ║  │ • zones      │  │ • VM details │  │ • load       │  │ • Kafka          │  ║ │   ║
║  │ ║  │ • envs       │  │              │  │   balancers  │  │ • PostgreSQL     │  ║ │   ║
║  │ ║  │ • bus        │  │              │  │              │  │ • DocumentDB     │  ║ │   ║
║  │ ║  │ • namespaces │  │              │  │              │  │ • GitLab         │  ║ │   ║
║  │ ║  │ • metrics    │  │              │  │              │  │ • Jenkins        │  ║ │   ║
║  │ ║  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │ • Registry      │  ║ │   ║
║  │ ║         │                 │                 │          └────────┬─────────┘  ║ │   ║
║  │ ║         └─────────────────┴────────┬────────┴───────────────────┘            ║ │   ║
║  │ ║                                    │                                         ║ │   ║
║  │ ║         Each Resource Agent:       │                                         ║ │   ║
║  │ ║         1. Read API spec from pgvector                                       ║ │   ║
║  │ ║         2. Resolve URL placeholders ({BASE_URL_PAAS_SERVICE} → real URL)     ║ │   ║
║  │ ║         3. Make HTTP call with Bearer token                                  ║ │   ║
║  │ ║         4. Return structured data                                            ║ │   ║
║  │ ╚════════════════════════════╤═══════════════════════════════════════════════════╝ │   ║
║  │                              │                                                    │   ║
║  │                              ▼                                                    │   ║
║  │ ╔════════════════════════════════════════════════════════════════════════════════╗ │   ║
║  │ ║  LAYER 5 — SERVICES (app/services/)                                           ║ │   ║
║  │ ║                                                                               ║ │   ║
║  │ ║  ┌─────────────────┐  ┌───────────────────┐  ┌───────────────────────────┐   ║ │   ║
║  │ ║  │  ai_service.py  │  │postgres_service.py│  │ api_executor_service.py   │   ║ │   ║
║  │ ║  │                 │  │                   │  │                           │   ║ │   ║
║  │ ║  │ • Embeddings    │  │ • Store documents │  │ • Keycloak auth           │   ║ │   ║
║  │ ║  │   (4096-dim)    │  │ • Vector search   │  │ • URL resolution          │   ║ │   ║
║  │ ║  │ • Text gen      │  │ • Session CRUD    │  │ • HTTP calls to Tata APIs │   ║ │   ║
║  │ ║  │ • Enhanced RAG  │  │ • Stats           │  │ • Engagement management   │   ║ │   ║
║  │ ║  └────────┬────────┘  └─────────┬─────────┘  └─────────────┬─────────────┘   ║ │   ║
║  │ ╚═══════════╪═════════════════════╪═══════════════════════════╪═════════════════╝ │   ║
║  └─────────────┼─────────────────────┼───────────────────────────┼───────────────────┘   ║
║                │                     │                           │                       ║
╚════════════════╪═════════════════════╪═══════════════════════════╪═══════════════════════╝
                 │                     │                           │
                 ▼                     ▼                           ▼
┌────────────────────────┐  ┌──────────────────┐  ┌──────────────────────────────────────┐
│   Tata vLLM Endpoint   │  │   PostgreSQL     │  │      Tata Communications APIs       │
│                        │  │   (internal)     │  │                                      │
│  Qwen2.5-Coder-14B    │  │                  │  │  ┌──────────┐  ┌──────────────────┐  │
│  -Instruct             │  │  Already shown   │  │  │ Keycloak │  │  PaaS Service    │  │
│                        │  │  above (pgvector)│  │  │ (Auth)   │  │  (K8s, VMs)      │  │
│  models.cloudservices  │  │                  │  │  └──────────┘  └──────────────────┘  │
│  .tatacommunications   │  │                  │  │  ┌──────────┐  ┌──────────────────┐  │
│  .com/v1               │  │                  │  │  │ Portal   │  │ Managed Services │  │
│                        │  │                  │  │  │ Service  │  │ (Kafka, PG,      │  │
│  Used for:             │  │                  │  │  │ (Names,  │  │  DocDB, GitLab,  │  │
│  • Chat responses      │  │                  │  │  │  Checks) │  │  Jenkins, Reg.)  │  │
│  • RAG synthesis       │  │                  │  │  └──────────┘  └──────────────────┘  │
│  • Embeddings          │  │                  │  │                                      │
└────────────────────────┘  └──────────────────┘  └──────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════════════════
  EXAMPLE FLOWS
═══════════════════════════════════════════════════════════════════════════════════════════

  FLOW 1: "Show me all Kubernetes clusters"
  ─────────────────────────────────────────
  Frontend ──► rag_widget.py ──► ADK Runner
                                    │
                                    ├── classify → resource_list_k8s_clusters
                                    ├── pre-check engagement → found? continue : prompt
                                    ├── dispatch → tools.list_k8s_clusters()
                                    ├── K8sAgent reads API spec from pgvector
                                    ├── K8sAgent calls Tata PaaS API
                                    ├── Format as Markdown table
                                    └── Return {answer, followUps}

  FLOW 2: "How do I create a namespace?"
  ──────────────────────────────────────
  Frontend ──► rag_widget.py ──► ADK Runner
                                    │
                                    ├── classify → rag (STRONG: "how do I")
                                    ├── search_knowledge_base() → pgvector
                                    ├── 8 results returned
                                    ├── Filter by relevance, extract sources & images
                                    ├── Build prompt: "[Source 1]...[Source 6]..."
                                    ├── Send to LLM via ADK Agent
                                    └── Return {answer, sources[8], images, confidence}

  FLOW 3: "Show me my engagements" → "2" (select)
  ────────────────────────────────────────────────
  Frontend ──► ADK Runner ── classify → engagement
                                │
                                ├── api_executor.get_engagements_list()
                                ├── Return table: | # | Name | ID |
                                └── follow_ups: ["Select engagement 1", ...]

  Frontend ──► ADK Runner ── classify → engagement_selection
                                │
                                ├── Parse "2" → select engagements[1]
                                ├── api_executor.set_engagement_id()
                                └── "You are now working with **Engagement X**"
```
