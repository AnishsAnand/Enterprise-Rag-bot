# 🔍 Filter & Refinement Feature Implementation

## Overview

Implemented intelligent filter/refinement detection that allows users to filter previous query results without re-querying the API.

## Problem Statement

**Before:**
- User: "cluster list in bengaluru" → ✅ Shows 7 clusters
- User: "filter by version below 1.30" → ❌ Bot asks for location again, treats as new query

**After:**
- User: "cluster list in bengaluru" → ✅ Shows 7 clusters
- User: "filter by version below 1.30" → ✅ Filters the 7 clusters without re-querying

## Key Changes

### 1. Filter Detection in Orchestrator (`orchestrator_agent.py`)

**Location:** `_decide_routing()` method (lines 310-343)

**Added Logic:**
```python
# Detect filter keywords
filter_keywords = [
    "filter", "show only", "just show", "only show",
    "version below", "version above", "version less than", 
    "from the above", "from that result", "exclude", "without"
]

is_filter_request = any(keyword in user_input.lower() for keyword in filter_keywords)
has_previous_result = (
    state.execution_result is not None 
    and state.execution_result.get("data") 
    and len(state.execution_result.get("data", [])) > 0
)

if is_filter_request and has_previous_result:
    return {
        "route": "filter",
        "reason": "User wants to filter/refine previous execution results",
        "filter_query": user_input
    }
```

### 2. Filter Route Handler (`orchestrator_agent.py`)

**Location:** `_execute_routing()` method (lines 527-688)

**Two Approaches:**

#### Approach A: Using Resource Agent's `filter_with_llm()` method
- Preferred approach
- Uses the specialized resource agent for intelligent filtering
- Leverages existing LLM-powered filtering logic

#### Approach B: Fallback LLM-based filtering
- Used when no resource agent is available
- Extracts filter criteria using LLM
- Applies filter manually
- Formats response with LLM

**Flow:**
```
1. Get previous execution result from state
2. Identify resource type (k8s_cluster, kafka, etc.)
3. Get appropriate resource agent from execution_agent.resource_agent_map
4. Call resource_agent.filter_with_llm(data, user_input, resource_type)
5. Format filtered results with resource_agent.format_response_with_llm()
6. Update state.execution_result with filtered data
7. Return formatted response
```

### 3. Execution Result Storage

**Already Implemented:** `ConversationState.execution_result` field stores the last successful execution result.

**Updated by:** 
- Line 628 in orchestrator: `state.set_execution_result(exec_result.get("execution_result", {}))`
- Line 666 in orchestrator: `state.set_execution_result(exec_result.get("execution_result", {}))`

## Supported Filter Patterns

The system now recognizes these filter patterns:

1. **"filter by X"** - e.g., "filter by version below 1.30"
2. **"show only X"** - e.g., "show only healthy clusters"
3. **"just show X"** - e.g., "just show clusters in production"
4. **"version below/above X"** - e.g., "version below 1.30"
5. **"from the above/that"** - e.g., "from the above, show only version 1.28"
6. **"exclude/without X"** - e.g., "exclude test clusters"

## Benefits

### ✅ Performance
- **No API re-calls**: Filters are applied to cached data
- **Faster response**: No need to re-fetch from external APIs
- **Reduced load**: Less load on backend services

### ✅ User Experience
- **Natural conversation**: Users can refine results naturally
- **Context preservation**: System remembers previous query context
- **No repetition**: Users don't need to repeat location/parameters

### ✅ Intelligence
- **LLM-powered**: Uses LLM to understand filter criteria
- **Flexible patterns**: Recognizes various ways of expressing filters
- **Smart formatting**: Results are formatted intelligently by LLM

## Example Conversations

### Example 1: Version Filter
```
User: cluster list in bengaluru
Bot: ✅ Found 7 clusters...
     - dataplattestcls02 (v1.29.12)
     - testrd04 (v1.30.9)
     - blr-paas (v1.26.15)
     ...

User: filter by version below 1.30
Bot: ✅ Found 4 clusters with version below 1.30:
     - dataplattestcls02 (v1.29.12)
     - blr-paas (v1.26.15)
     - test11 (v1.28.15)
     - aicloud-iks11 (v1.29.12)
```

### Example 2: Status Filter
```
User: show all kafka services
Bot: ✅ Found 12 Kafka services...

User: show only the healthy ones
Bot: ✅ Found 10 healthy Kafka services:
     ...
```

### Example 3: Name Filter
```
User: list clusters in all locations
Bot: ✅ Found 17 clusters across 9 locations...

User: just show production clusters
Bot: ✅ Found 3 production clusters:
     ...
```

## Technical Implementation Details

### Filter Criteria Extraction (Fallback Method)

When no resource agent is available, the system uses LLM to extract structured filter criteria:

```python
{
    "filter_field": "k8sVersion",      # Field to filter on
    "filter_operator": "less_than",    # Comparison operator
    "filter_value": "1.30"             # Value to compare
}
```

**Supported Operators:**
- `less_than` - For numeric/version comparisons
- `greater_than` - For numeric/version comparisons
- `equals` - For exact matches
- `contains` - For substring matches

### Resource Agent Integration

Resource agents (K8sClusterAgent, ManagedServicesAgent, etc.) implement:

1. **`filter_with_llm(data, filter_criteria, resource_type)`**
   - Uses LLM to intelligently filter data
   - Understands domain-specific filtering (e.g., version comparisons)
   - Returns filtered list

2. **`format_response_with_llm(operation, raw_data, user_query, context)`**
   - Formats filtered results into natural language
   - Includes relevant insights and statistics
   - Uses tables/lists for better readability

## Testing

### Test Case 1: Version Filter
```
1. Query: "cluster list in bengaluru"
2. Expected: 7 clusters shown
3. Query: "filter by version below 1.30"
4. Expected: Filtered list (4 clusters with version < 1.30)
5. Verify: No API call made, data filtered from cache
```

### Test Case 2: Multiple Filters
```
1. Query: "list all managed services"
2. Query: "show only kafka"
3. Query: "in mumbai only"
4. Expected: Each filter narrows down the previous result
```

### Test Case 3: Invalid Filter
```
1. Query: "cluster list in bengaluru"
2. Query: "filter by xyz"
3. Expected: Bot attempts to understand and either:
   - Applies a reasonable filter
   - Asks for clarification
```

## Future Enhancements

### 1. Multi-Field Filters
- Support: "version below 1.30 AND status is healthy"
- Requires: Parse multiple filter conditions

### 2. Sorting
- Support: "sort by version descending"
- Requires: Add sort_with_llm() method

### 3. Aggregations
- Support: "count by location"
- Requires: Add aggregation logic

### 4. Filter History
- Support: "undo last filter" or "show original results"
- Requires: Stack of filter states

### 5. Save Filters
- Support: "save this filter as 'production-clusters'"
- Requires: Filter template storage

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     User Input                               │
│        "filter by version below 1.30"                        │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│               OrchestratorAgent                              │
│  • Detects filter keywords                                   │
│  • Checks for previous execution_result                      │
│  • Routes to "filter" handler                                │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│           Filter Route Handler                               │
│  1. Get previous_data from state.execution_result            │
│  2. Get resource_type from state                             │
│  3. Get resource_agent from execution_agent.resource_map     │
└───────────────────────┬─────────────────────────────────────┘
                        │
                ┌───────┴────────┐
                ▼                ▼
    ┌─────────────────┐  ┌─────────────────┐
    │ Resource Agent  │  │ Fallback LLM    │
    │ (Preferred)     │  │ (No agent)      │
    │                 │  │                 │
    │ filter_with_llm │  │ Extract criteria│
    │       +         │  │ Apply manually  │
    │ format_response │  │ Format response │
    └────────┬────────┘  └────────┬────────┘
             │                    │
             └────────┬───────────┘
                      ▼
        ┌─────────────────────────┐
        │   Update state with     │
        │   filtered results      │
        └────────┬────────────────┘
                 │
                 ▼
        ┌─────────────────────────┐
        │   Return formatted      │
        │   response to user      │
        └─────────────────────────┘
```

## Code Locations

| Component | File | Lines |
|-----------|------|-------|
| Filter Detection | `app/agents/orchestrator_agent.py` | 319-343 |
| Filter Route Handler | `app/agents/orchestrator_agent.py` | 527-688 |
| Execution Result Storage | `app/agents/state/conversation_state.py` | 266-281 |
| Resource Agent Base | `app/agents/resource_agents/base_resource_agent.py` | - |

## Conclusion

This feature significantly enhances the conversational experience by allowing users to naturally refine and filter results without repeating context or making unnecessary API calls. The LLM-powered approach ensures flexibility and intelligence in understanding various filter expressions.

---
**Status:** ✅ Implemented
**Date:** Dec 15, 2025
**Version:** 1.0

