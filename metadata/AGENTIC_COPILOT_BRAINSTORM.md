# Agentic Copilot: Brainstorm & Design Document

> **Phase 1–3 Complete**: RAG is source of truth. `resource_schema.json` removed from config.
> Schema backup at `metadata/resource_schema_backup.json` for convert/ingest scripts only.
> APIExecutorService loads operation config from RAG at runtime via `get_operation_config_async()`.

## Executive Summary

This document brainstorms how to transform the Enterprise RAG Bot from a **chatbot/querying engine** (with static `resource_schema.json`) into a **truly agentic copilot** where:
- APIs are discovered dynamically via RAG
- Agents infer params, workflows, and steps from retrieved context
- The system is extensible without code changes (add APIs by adding docs)
- Proactive, context-aware assistance (copilot behavior)

---

## Part 1: Current Limitations (Why It's Not Truly Agentic)

| Aspect | Current State | Agentic Vision |
|--------|---------------|----------------|
| **API Discovery** | Hardcoded in `resource_schema.json` | RAG-retrieved; agents discover APIs dynamically |
| **Extensibility** | Add API = edit JSON + redeploy | Add API = ingest doc to RAG |
| **Intent** | IntentAgent knows resources from schema | IntentAgent queries RAG: "What can I do with X?" |
| **Workflows** | Static workflow steps in JSON | RAG describes workflows; LLM parses & executes |
| **Params** | Schema-driven validation | RAG describes params; LLM extracts & validates |
| **Proactivity** | Reactive only | Suggests next steps, anticipates needs |

---

## Part 2: RAG as the Single Source of Truth for APIs

### 2.1 Document Format for API Knowledge

To replace `resource_schema.json`, RAG documents must encode:

```
# API: Kubernetes Cluster - List

**Resource:** k8s_cluster
**Operation:** list
**Aliases:** clusters, kubernetes, k8s

## Endpoint
- **Method:** POST
- **URL:** https://ipcloud.tatacommunications.com/paasservice/paas/clusters/list
- **Auth:** Bearer token required

## Parameters
| Param | Required | Type | Description |
|-------|----------|------|-------------|
| endpoints | Yes | int[] | Data center IDs from list_endpoints |
| businessUnits | No | int[] | Filter by BU |
| environments | No | int[] | Filter by environment |
| zones | No | int[] | Filter by zone |

## Workflow (Prerequisites)
1. get_engagement → engagement_id
2. list_endpoints → endpoints (user selects)
3. [Optional] get_business_units / get_environments / get_zones for filters

## Response Mapping
- clusters: data
- cluster_ids: data[*].id
```

**Why this format?**
- **Chunkable**: Each resource+operation can be a separate chunk for precise retrieval
- **LLM-parseable**: Natural structure for extraction
- **Extensible**: Add new APIs by writing docs, not code

### 2.2 Structured API Schema in RAG (Alternative: OpenAPI-Style)

Alternatively, use a machine-readable format that RAG can still retrieve:

```yaml
# metadata: type=api_spec, resource=k8s_cluster, operation=list
resource: k8s_cluster
operation: list
api:
  method: POST
  url: "{{BASE_URL}}/paas/clusters/list"
  body:
    endpoints: int[]  # required
    businessUnits: int[]  # optional
workflow:
  - get_engagement
  - list_endpoints
  - list_clusters
```

The RAG retrieves this; an LLM formats it for downstream agents.

---

## Part 3: Agent Flow Redesign

### 3.1 New Flow: RAG-First Intent

```
User: "List clusters in Mumbai"
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  Orchestrator: Route to RAG-first Intent (not pure Intent)   │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  RAG Query: "APIs for listing kubernetes clusters"           │
│  Returns: Chunks with k8s_cluster list API, params, workflow │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  LLM Extraction:                                              │
│  - resource_type: k8s_cluster                                 │
│  - operation: list                                            │
│  - params: { endpoints: [resolve Mumbai] }                    │
│  - workflow: [get_engagement, list_endpoints, list_clusters]   │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  ValidationAgent: Uses RAG-retrieved param schema for validation│
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  ExecutionAgent: Uses RAG-retrieved URL/body/headers to call   │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Key Changes Per Agent

| Agent | Current | Agentic |
|-------|---------|---------|
| **IntentAgent** | Reads `resource_schema` → `_get_resources_info()` | **Queries RAG** with user input → retrieves API specs → LLM extracts intent + params + workflow |
| **ValidationAgent** | Uses schema for required/optional params | **Uses RAG-retrieved param spec** for validation |
| **ExecutionAgent** | `api_executor_service` loads schema for URLs | **Uses RAG-retrieved API spec** for URL, method, body |
| **RAGAgent** | Only for docs Q&A | **Unified**: both docs AND API specs; routing decides usage |

### 3.3 Hybrid Approach (Phased Migration)

**Phase 1**: RAG augments schema
- Keep `resource_schema.json` as fallback
- Add API docs to RAG; IntentAgent queries RAG first, falls back to schema
- Compare RAG-derived vs schema-derived results for validation

**Phase 2**: RAG primary
- Remove schema from IntentAgent
- APIExecutorService gets API config from **context passed by IntentAgent** (which got it from RAG)
- Schema only used for offline validation during migration

**Phase 3**: Schema removed
- Delete `resource_schema.json`
- All API config flows: RAG → IntentAgent → downstream agents

---

## Part 4: LLM Format & Param Extraction

### 4.1 Canonical Output Format from RAG+LLM

IntentAgent (or a new **RAG-Intent fusion agent**) should output:

```json
{
  "intent_detected": true,
  "resource_type": "k8s_cluster",
  "operation": "list",
  "api_spec": {
    "method": "POST",
    "url": "https://.../paas/clusters/list",
    "required_params": ["endpoints"],
    "optional_params": ["businessUnits", "environments", "zones"]
  },
  "workflow": [
    {"step": 1, "action": "get_engagement", "outputs": ["engagement_id"]},
    {"step": 2, "action": "list_endpoints", "outputs": ["endpoints"]},
    {"step": 3, "action": "list_clusters", "inputs": ["endpoints"]}
  ],
  "extracted_params": {"location_hint": "Mumbai"},
  "missing_params": ["endpoints"],
  "confidence": 0.92
}
```

Agents consume this structured output—no more schema lookups.

### 4.2 Dynamic Param Resolution

- **Location names** (Mumbai, Delhi) → RAG or a **lookup step** resolves to endpoint IDs
- **User-provided values** → ValidationAgent checks against RAG-retrieved types
- **Defaults** → RAG can include "defaults" in API docs; LLM applies when user doesn't specify

---

## Part 5: Training RAG for Agentic Behavior

### 5.1 Document Ingestion Strategy

1. **Convert `resource_schema.json` → RAG documents** (one-time migration)
   - Each `resource.operation` → one chunk
   - Include: URL, params, workflow, aliases, response_mapping
   - Ingest into `enterprise_rag` table

2. **Add API documentation** (ongoing)
   - OpenAPI files, internal API docs
   - Chunk with metadata: `{"type": "api_spec", "resource": "..."}`

3. **Metadata for filtering**
   - `source=api_spec` vs `source=documentation`
   - Enables: "Only retrieve API specs" vs "Only retrieve how-to docs"

### 5.2 Retrieval Augmentation

- **Query expansion**: "List clusters" → "k8s_cluster list API parameters workflow"
- **Hybrid search**: Vector + keyword (e.g., `resource_type:k8s_cluster`)
- **Re-ranking**: Prefer chunks with `api_spec` type when agent flow is operational

### 5.3 Few-Shot in System Prompt

Train the IntentAgent (or RAG-Intent) with examples:

```
Example: User says "Show clusters in Delhi"
RAG returns: k8s_cluster list API with workflow
LLM extracts: resource=k8s_cluster, operation=list, location=Delhi → needs endpoint resolution
Output: workflow includes list_endpoints; ValidationAgent will resolve "Delhi" to endpoint IDs
```

---

## Part 6: Copilot Features (Beyond Agentic Execution)

### 6.1 Proactive Suggestions

| Feature | Description |
|---------|-------------|
| **Next-step suggestions** | After "List clusters" → "Would you like to create a cluster or get details for one?" |
| **Context-aware prompts** | User viewing cluster X → "Scale this cluster? Add a node pool?" |
| **Shortcut learning** | "You often list clusters in Mumbai—use this shortcut?" |

### 6.2 Task Decomposition

- User: "Set up a new production cluster with 3 master nodes and 5 workers"
- Copilot: Breaks into steps, shows plan, executes with confirmations
- **Plan → Confirm → Execute** flow

### 6.3 Observability & Control

- **Explain before acting**: "I'll call these 3 APIs: 1) get_engagement 2) list_endpoints 3) list_clusters. Proceed?"
- **Undo/rollback**: "Revert the last cluster creation" (if supported by APIs)
- **Step-through mode**: Execute one step at a time with user approval

### 6.4 Memory & Personalization

- **Conversation state** (already exists) → extend with user preferences
- **Frequent actions**: "You usually filter by BU TATA—apply that?"
- **Recent resources**: "Last time you worked with cluster blr-paas—continue there?"

### 6.5 Multi-Turn Reasoning

- **Clarification chains**: "Which Mumbai? DC1 or DC2?" → User answers → Continue
- **Dependency resolution**: "To list clusters, I need your data center. Which ones do you have access to?" → Show endpoints → User picks

### 6.6 Tool Use (Agents as Tools)

- **Reflexive tool use**: Agent can call "list_endpoints" as a tool to help the user choose
- **Structured outputs**: Return tables, not just text—for UI rendering (cards, tables, forms)

---

## Part 7: Implementation Roadmap

### Phase 1: RAG API Documents (2–3 weeks)
1. Script to convert `resource_schema.json` → markdown/YAML chunks
2. Ingest into RAG with metadata `type=api_spec`
3. Add retrieval path that filters by `type=api_spec`

### Phase 2: RAG-Driven Intent (2–3 weeks)
1. IntentAgent: Add RAG query before/during intent detection
2. LLM prompt: "Given this RAG context, extract intent, params, workflow"
3. Pass `api_spec` in context to ValidationAgent and ExecutionAgent
4. ExecutionAgent: Use `api_spec` from context instead of schema (when available)

### Phase 3: Remove Schema Dependency (1–2 weeks)
1. APIExecutorService: Add `execute_from_spec(api_spec, params)` that doesn't use schema
2. Migrate all call sites to pass `api_spec` from context
3. Deprecate `resource_schema.json`

### Phase 4: Copilot Enhancements (ongoing)
1. Next-step suggestions in Orchestrator
2. Plan-and-confirm flow for create/update/delete
3. Proactive prompts based on conversation state

---

## Part 8: Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| RAG retrieval misses API spec | Hybrid: RAG first, schema fallback during migration; improve chunking |
| LLM hallucinates params | ValidationAgent strict type check; confidence threshold |
| Latency (RAG + LLM) | Cache frequent API specs; async RAG retrieval |
| New APIs not in RAG | CI pipeline: New OpenAPI → auto-ingest to RAG |

---

## Part 9: Summary of Recommendations

1. **Remove `resource_schema.json`** by first converting it to RAG documents, then making RAG the source for API specs.
2. **RAG-first Intent**: IntentAgent queries RAG for "APIs for X" and uses LLM to extract structured `api_spec` + `workflow` + `params`.
3. **Unified RAG**: Same knowledge base for docs and API specs; metadata differentiates them.
4. **Structured output**: Define a canonical JSON format for intent + api_spec + workflow that all agents consume.
5. **Copilot features**: Add proactive suggestions, plan-confirm-execute, and context-aware shortcuts.
6. **Phased migration**: RAG augments schema → RAG primary → schema removed.

---

## Appendix: Example RAG Chunk (Full)

```markdown
# API Specification: List Kubernetes Clusters

**Resource:** k8s_cluster
**Operation:** list
**Aliases:** clusters, kubernetes, k8s, list clusters, show clusters

## Endpoint
- Method: POST
- URL: https://ipcloud.tatacommunications.com/paasservice/paas/clusters/list
- Auth: Bearer token (from Keycloak)

## Required Parameters
- `endpoints` (int[]): Data center IDs. Obtain from list_endpoints API.

## Optional Parameters
- `businessUnits` (int[]): Filter by business unit IDs
- `environments` (int[]): Filter by environment IDs  
- `zones` (int[]): Filter by zone IDs

## Workflow Steps
1. get_engagement → yields engagement_id
2. list_endpoints(engagement_id) → yields endpoints; user selects
3. [Optional] get_business_units(ipc_engagement_id) for BU filter
4. list_clusters(endpoints, ...) → display result

## Response
- clusters: list of cluster objects
- Use display_format: table with columns [name, status, zone, endpoint]
```
