# Agentic Metrics API Documentation

## Overview

The Agentic Metrics API provides endpoints for evaluating AI agent performance using three key metrics:

1. **Task Adherence (40% weight)** - How well the agent followed the user's instructions
2. **Tool Call Accuracy (30% weight)** - Whether the agent used the correct tools with proper parameters
3. **Intent Resolution (30% weight)** - How accurately the agent understood and resolved the user's intent

These metrics are based on [Microsoft's Agentic AI Evaluation Framework](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/evaluating-agentic-ai-systems-a-deep-dive-into-agentic-metrics/4403923).

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           API Layer                                      │
│                    /api/v1/metrics/*                                     │
│              (app/routers/agentic_metrics.py)                           │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      Service Layer                                       │
│              AgenticMetricsEvaluator                                     │
│         (app/services/agentic_metrics_service.py)                       │
│                                                                          │
│  • Trace Management (start, record, complete)                           │
│  • LLM-based Evaluation                                                  │
│  • Score Calculation                                                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
┌──────────────────────────────┐    ┌──────────────────────────────┐
│      AI Service              │    │    Persistence Layer          │
│   (LLM Evaluation)           │    │      (PostgreSQL)             │
│  app/services/ai_service.py  │    │  agentic_metrics_persistence  │
└──────────────────────────────┘    └──────────────────────────────┘
```

## Base URL

```
http://localhost:8001/api/v1/metrics
```

## Endpoints

### 1. Health Check

Check if the agentic metrics service is healthy.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "service": "agentic_metrics",
  "evaluations_count": 5,
  "database": {
    "status": "connected",
    "evaluations_stored": 42
  }
}
```

**cURL Example:**
```bash
curl -X GET "http://localhost:8001/api/v1/metrics/health"
```

---

### 2. Evaluate Session by ID

Evaluate an agent session that has been previously traced.

**Endpoint:** `POST /evaluate/session`

**Request Body:**
```json
{
  "session_id": "openwebui_abc123def456"
}
```

**Response:**
```json
{
  "session_id": "openwebui_abc123def456",
  "agent_name": "OrchestratorAgent",
  "task_adherence": {
    "score": 0.95,
    "reasoning": "The agent correctly listed all clusters as requested..."
  },
  "tool_call_accuracy": {
    "score": 0.90,
    "reasoning": "The agent used the correct tool with appropriate parameters..."
  },
  "intent_resolution": {
    "score": 0.92,
    "reasoning": "The agent accurately understood the user's intent..."
  },
  "overall_score": 0.926,
  "timestamp": "2026-01-08T10:30:00Z",
  "metadata": {}
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8001/api/v1/metrics/evaluate/session" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "openwebui_abc123def456"}'
```

**Error Response (404):**
```json
{
  "detail": "No trace found for session openwebui_abc123def456"
}
```

---

### 3. Manual Evaluation

Evaluate an agent interaction without a pre-existing trace. Useful for testing, historical analysis, or A/B testing.

**Endpoint:** `POST /evaluate/manual`

**Request Body:**
```json
{
  "user_query": "List all Kubernetes clusters in Delhi",
  "agent_response": "Found 2 clusters in Delhi:\n1. prod-01 (running)\n2. dev-02 (stopped)",
  "tool_calls": [
    {
      "tool_name": "list_k8s_clusters",
      "tool_args": {"datacenter": "Delhi"},
      "tool_result": [
        {"name": "prod-01", "status": "running"},
        {"name": "dev-02", "status": "stopped"}
      ],
      "success": true
    }
  ],
  "detected_intent": "list_k8s_cluster",
  "resource_type": "k8s_cluster",
  "operation": "list"
}
```

**Response:**
```json
{
  "task_adherence": {
    "score": 0.95,
    "reasoning": "The agent correctly addressed the user's request..."
  },
  "tool_call_accuracy": {
    "score": 0.90,
    "reasoning": "Appropriate tool was selected..."
  },
  "intent_resolution": {
    "score": 0.92,
    "reasoning": "Intent was correctly identified..."
  },
  "overall_score": 0.926,
  "evaluation_timestamp": "2026-01-08T10:30:00Z"
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8001/api/v1/metrics/evaluate/manual" \
  -H "Content-Type: application/json" \
  -d '{
    "user_query": "List all Kubernetes clusters in Delhi",
    "agent_response": "Found 2 clusters in Delhi: prod-01 (running), dev-02 (stopped)",
    "tool_calls": [
      {
        "tool_name": "list_k8s_clusters",
        "tool_args": {"datacenter": "Delhi"},
        "tool_result": [{"name": "prod-01"}, {"name": "dev-02"}],
        "success": true
      }
    ],
    "detected_intent": "list_k8s_cluster",
    "resource_type": "k8s_cluster",
    "operation": "list"
  }'
```

---

### 4. Batch Evaluation

Evaluate multiple sessions at once.

**Endpoint:** `POST /evaluate/batch`

**Request Body:**
```json
{
  "session_ids": [
    "session_001",
    "session_002",
    "session_003"
  ]
}
```

**Response:**
```json
{
  "evaluated": 2,
  "not_found": 1,
  "results": [
    {
      "session_id": "session_001",
      "agent_name": "OrchestratorAgent",
      "task_adherence": 0.95,
      "tool_call_accuracy": 0.90,
      "intent_resolution": 0.92,
      "overall_score": 0.926
    },
    {
      "session_id": "session_002",
      "agent_name": "OrchestratorAgent",
      "task_adherence": 0.88,
      "tool_call_accuracy": 0.85,
      "intent_resolution": 0.90,
      "overall_score": 0.878
    }
  ],
  "missing_sessions": ["session_003"]
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8001/api/v1/metrics/evaluate/batch" \
  -H "Content-Type: application/json" \
  -d '{"session_ids": ["session_001", "session_002", "session_003"]}'
```

---

### 5. Get Evaluation Summary

Get aggregate statistics across all evaluations.

**Endpoint:** `GET /summary`

**Response:**
```json
{
  "total_evaluations": 25,
  "average_scores": {
    "task_adherence": 0.89,
    "tool_call_accuracy": 0.85,
    "intent_resolution": 0.91,
    "overall": 0.883
  },
  "score_distribution": {
    "excellent": 10,
    "good": 8,
    "acceptable": 5,
    "poor": 2,
    "failed": 0
  },
  "by_agent": {
    "OrchestratorAgent": {
      "count": 25,
      "avg_score": 0.883
    }
  },
  "by_operation": {
    "list": {"count": 15, "avg_score": 0.90},
    "create": {"count": 7, "avg_score": 0.85},
    "delete": {"count": 3, "avg_score": 0.88}
  }
}
```

**cURL Example:**
```bash
curl -X GET "http://localhost:8001/api/v1/metrics/summary"
```

---

### 6. Get Trace by Session ID

Retrieve the recorded trace for a specific session.

**Endpoint:** `GET /trace/{session_id}`

**Response:**
```json
{
  "session_id": "openwebui_abc123",
  "user_query": "List clusters in Delhi",
  "agent_name": "OrchestratorAgent",
  "intent_detected": "list_k8s_cluster",
  "resource_type": "k8s_cluster",
  "operation": "list",
  "tool_calls": [
    {
      "tool_name": "list_k8s_clusters",
      "tool_args": {"datacenter": "Delhi"},
      "tool_result": [...],
      "timestamp": "2026-01-08T10:30:00Z",
      "success": true
    }
  ],
  "final_response": "Found 2 clusters...",
  "success": true,
  "start_time": "2026-01-08T10:29:55Z",
  "end_time": "2026-01-08T10:30:00Z"
}
```

**cURL Example:**
```bash
curl -X GET "http://localhost:8001/api/v1/metrics/trace/openwebui_abc123"
```

---

### 7. Export Results

Export all evaluation results in JSON or JSONL format.

**Endpoint:** `GET /export`

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `format` | string | `json` | Export format: `json` or `jsonl` |

**Response:**
```json
{
  "format": "json",
  "data": "[{\"session_id\": \"...\", ...}, ...]"
}
```

**cURL Examples:**
```bash
# Export as JSON
curl -X GET "http://localhost:8001/api/v1/metrics/export?format=json"

# Export as JSONL
curl -X GET "http://localhost:8001/api/v1/metrics/export?format=jsonl"
```

---

### 8. Get Evaluation History

Retrieve historical evaluations from the PostgreSQL database with pagination and filtering.

**Endpoint:** `GET /history`

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | int | 50 | Maximum records to return (1-500) |
| `offset` | int | 0 | Number of records to skip |
| `agent_name` | string | null | Filter by agent name |
| `operation` | string | null | Filter by operation type |
| `min_score` | float | null | Minimum overall score (0-1) |

**Response:**
```json
{
  "count": 10,
  "limit": 50,
  "offset": 0,
  "evaluations": [
    {
      "session_id": "openwebui_abc123",
      "agent_name": "OrchestratorAgent",
      "overall_score": 0.926,
      "timestamp": "2026-01-08T10:30:00Z",
      ...
    }
  ]
}
```

**cURL Examples:**
```bash
# Get first 10 evaluations
curl -X GET "http://localhost:8001/api/v1/metrics/history?limit=10"

# Filter by agent name
curl -X GET "http://localhost:8001/api/v1/metrics/history?agent_name=OrchestratorAgent"

# Filter by minimum score
curl -X GET "http://localhost:8001/api/v1/metrics/history?min_score=0.8"

# Pagination
curl -X GET "http://localhost:8001/api/v1/metrics/history?limit=10&offset=20"
```

---

### 9. Get Specific Evaluation from History

Retrieve a specific evaluation by session ID from the database.

**Endpoint:** `GET /history/{session_id}`

**Response:**
```json
{
  "session_id": "openwebui_abc123",
  "agent_name": "OrchestratorAgent",
  "task_adherence": 0.95,
  "task_adherence_reasoning": "...",
  "tool_call_accuracy": 0.90,
  "tool_call_accuracy_reasoning": "...",
  "intent_resolution": 0.92,
  "intent_resolution_reasoning": "...",
  "overall_score": 0.926,
  "timestamp": "2026-01-08T10:30:00Z"
}
```

**cURL Example:**
```bash
curl -X GET "http://localhost:8001/api/v1/metrics/history/openwebui_abc123"
```

---

### 10. Clear All Results

Clear all stored traces and evaluation results. **Use with caution!**

**Endpoint:** `DELETE /clear`

**Response:**
```json
{
  "message": "All traces and evaluation results cleared"
}
```

**cURL Example:**
```bash
curl -X DELETE "http://localhost:8001/api/v1/metrics/clear"
```

---

## Score Interpretation

| Score Range | Rating | Description |
|-------------|--------|-------------|
| 0.90 - 1.00 | Excellent | Agent performed exceptionally well |
| 0.75 - 0.89 | Good | Agent performed well with minor issues |
| 0.60 - 0.74 | Acceptable | Agent completed task but with notable issues |
| 0.40 - 0.59 | Poor | Agent struggled significantly |
| 0.00 - 0.39 | Failed | Agent failed to complete the task |

## How Tracing Works

Traces are automatically created when agents execute. The flow is:

1. **Start Trace** - When an agent begins processing a user query
2. **Record Intent** - When the intent is detected
3. **Record Tool Calls** - Each tool invocation is recorded
4. **Complete Trace** - When the agent finishes with final response

```python
# Example: How traces are created internally
evaluator.start_trace(
    session_id="openwebui_abc123",
    user_query="List clusters in Delhi",
    agent_name="OrchestratorAgent"
)

evaluator.record_intent("openwebui_abc123", "list_k8s_cluster", "k8s_cluster", "list")

evaluator.record_tool_call(
    "openwebui_abc123",
    "list_k8s_clusters",
    {"datacenter": "Delhi"},
    [{"name": "prod-01"}, {"name": "dev-02"}],
    True
)

evaluator.complete_trace(
    "openwebui_abc123",
    "Found 2 clusters in Delhi...",
    True
)
```

## Integration with Prometheus

The agentic metrics service also exposes metrics to Prometheus for visualization in Grafana:

- `vayu_agent_sessions_total` - Total agent sessions by agent name and status
- `vayu_agent_execution_duration_seconds` - Agent execution duration histogram
- `vayu_agent_active_sessions` - Currently active agent sessions

These metrics are exposed at `/metrics` endpoint and scraped by Prometheus.

## File Locations

| Component | File Path |
|-----------|-----------|
| API Router | `app/routers/agentic_metrics.py` |
| Service Layer | `app/services/agentic_metrics_service.py` |
| Persistence | `app/services/agentic_metrics_persistence.py` |
| Prometheus Metrics | `app/services/prometheus_metrics.py` |
| Test Script | `tests/test_agentic_metrics_api.py` |

## Running Tests

```bash
# Ensure the backend is running
cd Enterprise-Rag-bot
python -m uvicorn app.user_main:app --host 0.0.0.0 --port 8001 --reload

# In another terminal, run the test script
python tests/test_agentic_metrics_api.py
```

## Error Handling

All endpoints return standard HTTP status codes:

| Status Code | Description |
|-------------|-------------|
| 200 | Success |
| 404 | Resource not found (e.g., session/trace doesn't exist) |
| 422 | Validation error (invalid request body) |
| 500 | Internal server error |

Error responses follow this format:
```json
{
  "detail": "Error message describing what went wrong"
}
```

