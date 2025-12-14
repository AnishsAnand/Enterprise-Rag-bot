# Multi-Resource List Handling Bug Fix

**Date:** December 15, 2025  
**Issue:** `unhashable type: 'list'` error when IntentAgent returns resource_type as a list

---

## Problem Description

When users requested multiple resources in a single query (e.g., "show gitlab and kafka"), the IntentAgent would sometimes return `resource_type` as a **list** instead of a string:

```json
{
  "intent_detected": true,
  "resource_type": ["gitlab", "kafka"],  // ❌ LIST instead of STRING
  "operation": "list",
  ...
}
```

This caused a Python error:
```
ERROR: unhashable type: 'list'
```

The error occurred because:
1. Lists cannot be used as dictionary keys or in sets (unhashable)
2. The `api_executor_service.get_operation_config()` expected a string
3. The conversation state tried to use the list in hashable contexts

---

## Root Cause

The IntentAgent's LLM sometimes returns multi-resource requests as:
- ✅ `"resource_type": "gitlab,kafka"` (string) - worked fine
- ❌ `"resource_type": ["gitlab", "kafka"]` (list) - caused crash
- ⚠️ `"resource_type": null` with ambiguities - triggered extraction path

---

## Solution Implemented

### 1. **IntentAgent Fix** (`intent_agent.py`)

Added list-to-string conversion when looking up operation config:

```python
# Handle multi-resource: if resource_type is a list, convert to string
# For param lookup, use the first resource type
if isinstance(resource_type, list):
    logger.info(f"🔧 Multi-resource detected: {resource_type}")
    # For parameter schema, use the first resource
    lookup_resource_type = resource_type[0] if resource_type else None
else:
    lookup_resource_type = resource_type

if lookup_resource_type and operation:
    operation_config = api_executor_service.get_operation_config(
        lookup_resource_type, operation
    )
```

**Why first resource?** All managed services (gitlab, kafka, jenkins, etc.) have the same parameter schema for `list` operations (just `endpoints`), so we can use any one resource type to look up the parameters.

### 2. **Orchestrator Fix** (`orchestrator_agent.py`)

Added list-to-string conversion when setting conversation state:

```python
# Handle resource_type: convert list to comma-separated string
resource_type = intent_data.get("resource_type")
if isinstance(resource_type, list):
    resource_type = ",".join(resource_type)
    logger.info(f"🔧 Converted resource_type list to string: {resource_type}")

state.set_intent(
    resource_type=resource_type,  # Now guaranteed to be a string
    operation=intent_data.get("operation"),
    ...
)
```

This ensures `state.resource_type` is **always a string**, even if the IntentAgent returns a list.

---

## Three Execution Paths (All Working)

### Path 1: Direct String ✅
```
IntentAgent → "gitlab,kafka" (string)
→ Orchestrator → state.set_intent(resource_type="gitlab,kafka")
→ ExecutionAgent → Multi-resource detection → Parallel execution
```

### Path 2: List Conversion ✅
```
IntentAgent → ["gitlab", "kafka"] (list)
→ Orchestrator → Convert to "gitlab,kafka" (string)
→ state.set_intent(resource_type="gitlab,kafka")
→ ExecutionAgent → Multi-resource detection → Parallel execution
```

### Path 3: Extraction from Ambiguities ✅
```
IntentAgent → null with ambiguity "User requested: gitlab and kafka"
→ Orchestrator → Extract "gitlab,kafka" from ambiguity text
→ state.set_intent(resource_type="gitlab,kafka")
→ ExecutionAgent → Multi-resource detection → Parallel execution
```

---

## Testing Results

**Test Query:** `show kafka and container registry in all locations`

**Before Fix:**
```
ERROR: unhashable type: 'list'
Response: Failed to detect intent: unhashable type: 'list'
```

**After Fix:**
```
✅ Found 3 resources across 2 types
- Container Registry: 1
- Apache Kafka: 2

(Beautifully formatted table with all services)
```

**Logs Confirm:**
```
INFO: 🔧 Converted resource_type list to string: kafka,container_registry
INFO: ✅ Multi-resource confirmed: kafka,container_registry, skipping clarification
INFO: 🔀 Executing 2 resource operations in parallel
INFO: ✅ kafka completed successfully
INFO: ✅ container_registry completed successfully
```

---

## Files Modified

1. **`app/agents/intent_agent.py`**
   - Lines 366-383: Added list handling for `get_operation_config` lookup
   
2. **`app/agents/orchestrator_agent.py`**
   - Lines 752-766: Added list-to-string conversion before `state.set_intent()`

---

## Impact

✅ **Multi-resource queries now work reliably** regardless of IntentAgent's output format  
✅ **No more "unhashable type" errors**  
✅ **Parallel execution works for all multi-resource requests**  
✅ **Permissions properly passed through** (fixed in previous session)

---

## Example Working Queries

- ✅ `show gitlab and kafka`
- ✅ `list kafka and container registry in all endpoints`
- ✅ `show jenkins, postgres and documentdb in bengaluru`
- ✅ `gitlab and kafka in all datacenters`

All queries now execute in parallel and return combined, formatted results! 🚀

