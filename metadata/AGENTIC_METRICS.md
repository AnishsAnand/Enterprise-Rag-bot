# 📊 Agentic Metrics - Agent Evaluation System

This document describes the agentic metrics implementation for evaluating AI agent performance, based on the [Azure AI Evaluation library](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/evaluating-agentic-ai-systems-a-deep-dive-into-agentic-metrics/4403923).

## Overview

As AI agents become more sophisticated with planning, tool use, and autonomous decision-making, we need equally sophisticated evaluation methods. This implementation provides three key metrics:

| Metric | Description | Weight |
|--------|-------------|--------|
| **Task Adherence** | Did the agent answer the right question? | 40% |
| **Tool Call Accuracy** | Did the agent use tools correctly? | 30% |
| **Intent Resolution** | Did the agent understand the user's goal? | 30% |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Query                                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  OrchestratorAgent                          │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │        AgenticMetricsEvaluator                        │   │
│  │  • start_trace(session_id, query, agent)              │   │
│  │  • record_intent(intent, resource_type, operation)    │   │
│  │  • record_tool_call(tool_name, args, result)          │   │
│  │  • complete_trace(response, success)                  │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   LLM Evaluation                             │
│                                                              │
│  ┌────────────────┐ ┌─────────────────┐ ┌────────────────┐  │
│  │ Task Adherence │ │ Tool Call       │ │ Intent         │  │
│  │ Evaluator      │ │ Accuracy        │ │ Resolution     │  │
│  │                │ │ Evaluator       │ │ Evaluator      │  │
│  └────────────────┘ └─────────────────┘ └────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  EvaluationResult                            │
│                                                              │
│  {                                                           │
│    "task_adherence": 0.85,                                   │
│    "tool_call_accuracy": 0.92,                               │
│    "intent_resolution": 0.88,                                │
│    "overall_score": 0.88                                     │
│  }                                                           │
└─────────────────────────────────────────────────────────────┘
```

## The Three Metrics

### 1. Task Adherence 📋

**Question: Is the agent answering the right question?**

Task Adherence evaluates how well the agent's final response satisfies the original user request. It assesses:

- **Relevance** (0-3): Is the response on topic?
- **Completeness** (0-3): Does it fully address the request?
- **Alignment** (0-2): Does it match user expectations?
- **Clarity** (0-2): Is the response clear and actionable?

**Example:**
```
User Query: "List clusters in Delhi"
Agent Response: "Here are luxury hotels in Delhi..."
Task Adherence: ❌ POOR (0.2) - Wrong topic entirely
```

```
User Query: "List clusters in Delhi"  
Agent Response: "Found 3 clusters in Delhi: prod-01, dev-02, test-03"
Task Adherence: ✅ EXCELLENT (0.95) - Directly addresses the request
```

### 2. Tool Call Accuracy 🔧

**Question: Is the agent using tools correctly?**

This metric focuses on procedural accuracy when invoking tools:

- **Tool Selection** (0-3): Were the right tools chosen?
- **Argument Accuracy** (0-3): Were arguments correct and well-formatted?
- **Logical Order** (0-2): Were tools called in a sensible sequence?
- **Efficiency** (0-2): Were unnecessary tool calls avoided?

**Example:**
```
User Query: "Create cluster named prod-cluster in Delhi"
Tool Called: list_k8s_clusters(location=["Delhi"])
Tool Call Accuracy: ❌ POOR (0.3) - Wrong tool selected
```

```
User Query: "Create cluster named prod-cluster in Delhi"
Tool Called: create_k8s_cluster(name="prod-cluster", datacenter="Delhi")
Tool Call Accuracy: ✅ EXCELLENT (0.95) - Correct tool and arguments
```

### 3. Intent Resolution 🎯

**Question: Did the agent understand the user's goal?**

Intent Resolution assesses whether the agent's initial actions reflect correct understanding:

- **Intent Detection** (0-4): Did the agent correctly identify what the user wants?
- **Goal Understanding** (0-3): Did it understand the underlying need?
- **Context Awareness** (0-3): Did it pick up on implied requirements?

**Example:**
```
User Query: "What clusters do we have?"
Detected Intent: documentation_query
Intent Resolution: ❌ POOR (0.3) - Should be resource_operation
```

```
User Query: "What clusters do we have?"
Detected Intent: list_k8s_cluster
Intent Resolution: ✅ EXCELLENT (0.95) - Correctly understood as resource query
```

## Usage

### Automatic Tracing (Built into BaseAgent)

Metrics are automatically captured when `ENABLE_AGENTIC_METRICS=true` (default):

```python
# Metrics are captured automatically in execute()
result = await agent.execute(
    input_text="List clusters in Delhi",
    context={"session_id": "session_123"}
)

# Evaluate the execution
evaluation = await agent.evaluate_execution("session_123")
print(f"Overall Score: {evaluation['overall_score']}")
```

### Manual Recording

For custom tool calls or intent recording:

```python
from app.services.agentic_metrics_service import agentic_metrics_evaluator

# Start a trace
evaluator = agentic_metrics_evaluator
trace = evaluator.start_trace(
    session_id="session_123",
    user_query="List clusters in Delhi",
    agent_name="OrchestratorAgent"
)

# Record intent
evaluator.record_intent(
    session_id="session_123",
    intent="list_k8s_cluster",
    resource_type="k8s_cluster",
    operation="list"
)

# Record tool calls
evaluator.record_tool_call(
    session_id="session_123",
    tool_name="list_k8s_clusters",
    tool_args={"location_names": ["Delhi"]},
    tool_result=[{"name": "prod-01", "status": "running"}],
    success=True
)

# Complete the trace
evaluator.complete_trace(
    session_id="session_123",
    final_response="Found 1 cluster in Delhi...",
    success=True
)

# Evaluate
result = await evaluator.evaluate_session("session_123")
print(f"Task Adherence: {result.task_adherence}")
print(f"Tool Accuracy: {result.tool_call_accuracy}")
print(f"Intent Resolution: {result.intent_resolution}")
```

## API Endpoints

### Evaluate a Session
```bash
POST /api/v1/metrics/evaluate/session
{
    "session_id": "session_123"
}
```

### Evaluate Manually (Without Trace)
```bash
POST /api/v1/metrics/evaluate/manual
{
    "user_query": "List clusters in Delhi",
    "agent_response": "Found 3 clusters in Delhi...",
    "detected_intent": "list_k8s_cluster",
    "resource_type": "k8s_cluster",
    "operation": "list"
}
```

### Batch Evaluate
```bash
POST /api/v1/metrics/evaluate/batch
{
    "session_ids": ["session_001", "session_002", "session_003"]
}
```

### Get Summary Statistics
```bash
GET /api/v1/metrics/summary
```

Response:
```json
{
    "total_evaluations": 150,
    "average_scores": {
        "task_adherence": 0.85,
        "tool_call_accuracy": 0.88,
        "intent_resolution": 0.82,
        "overall": 0.85
    },
    "score_distribution": {
        "excellent": 45,
        "good": 72,
        "acceptable": 28,
        "poor": 4,
        "failed": 1
    }
}
```

### Export Results
```bash
GET /api/v1/metrics/export?format=json
GET /api/v1/metrics/export?format=jsonl
```

### Get a Trace
```bash
GET /api/v1/metrics/trace/{session_id}
```

## Configuration

Set these environment variables:

```bash
# Enable/disable agentic metrics (default: true)
ENABLE_AGENTIC_METRICS=true

# For Azure AI Foundry integration (optional)
# AZURE_AI_EVALUATION_PROJECT=your-project-id
```

## Score Interpretation

| Score Range | Rating | Meaning |
|-------------|--------|---------|
| 0.9 - 1.0 | Excellent | Agent performed optimally |
| 0.7 - 0.89 | Good | Minor issues, overall successful |
| 0.5 - 0.69 | Acceptable | Works but has notable gaps |
| 0.3 - 0.49 | Poor | Significant problems |
| 0.0 - 0.29 | Failed | Critical failure |

## Testing

Run the test suite:

```bash
cd /home/unixlogin/Vayu/Enterprise-Rag-bot
python3 tests/test_agentic_metrics.py
```

## Future Enhancements

1. **Azure AI Foundry Integration**: Export evaluations to Azure AI Foundry for visualization
2. **Custom Evaluators**: Add domain-specific evaluation criteria
3. **Real-time Monitoring**: Dashboard for live agent performance monitoring
4. **Regression Testing**: Automated tests against expected metric baselines

## References

- [Evaluating Agentic AI Systems: A Deep Dive into Agentic Metrics](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/evaluating-agentic-ai-systems-a-deep-dive-into-agentic-metrics/4403923)
- [Azure AI Evaluation Library](https://learn.microsoft.com/en-us/azure/ai-studio/concepts/evaluation)
- [AgenticEvals GitHub Repository](https://github.com/Azure-Samples/AgenticEvals)

