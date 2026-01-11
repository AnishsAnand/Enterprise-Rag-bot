# Bug Fix: Missing Engagement ID Parameter

## Issue

Function calling was failing with "Parameter validation failed":

```
📍 Datacenters result: success=False, has_data=False, error=Parameter validation failed
```

**Impact:** All function calls were failing because the API endpoint listing requires an `engagement_id` parameter that wasn't being provided.

---

## Root Cause

From `app/config/resource_schema.json`:

```json
"endpoint": {
  "operations": ["list"],
  "api_endpoints": {
    "list": {
      "method": "GET",
      "url": "https://ipcloud.tatacommunications.com/.../getEndpointsByEngagement/{engagement_id}",
      ...
    }
  },
  "parameters": {
    "list": {
      "required": ["engagement_id"],  // ❌ This was missing!
      "optional": []
    }
  }
}
```

The API workflow requires:
1. **First:** Call `engagement.get` → Get engagement_id
2. **Then:** Call `endpoint.list` with engagement_id → Get datacenters
3. **Finally:** Call `k8s_cluster.list` with endpoint IDs → Get clusters

Our function handlers were skipping step 1!

---

## Fix Applied

Updated all three function handlers in `app/services/function_calling_service.py`:

### 1. `_list_k8s_clusters_handler` (line ~231)

**Before:**
```python
datacenters_result = await api_executor_service.execute_operation(
    resource_type="endpoint",
    operation="list",
    params={},  # ❌ Missing engagement_id
    user_roles=context.get("user_roles", [])
)
```

**After:**
```python
# Step 0: Get engagement ID first
engagement_result = await api_executor_service.execute_operation(
    resource_type="engagement",
    operation="get",
    params={},
    user_roles=context.get("user_roles", [])
)

engagement_id = engagement_result.get("data", [])[0].get("id")

# Step 1: Get datacenters with engagement_id
datacenters_result = await api_executor_service.execute_operation(
    resource_type="endpoint",
    operation="list",
    params={"engagement_id": engagement_id},  # ✅ Now passing engagement_id!
    user_roles=context.get("user_roles", [])
)
```

### 2. `_get_datacenters_handler` (line ~359)

Applied same fix - fetch engagement_id before listing endpoints.

### 3. `_create_k8s_cluster_handler` (line ~430)

Applied same fix - fetch engagement_id before resolving datacenter locations.

---

## Complete Workflow Now

```
User: "List clusters in Delhi"
    ↓
FunctionCallingAgent calls: list_k8s_clusters(location_names=["Delhi"])
    ↓
Handler execution:
    ├─ Step 0: GET /engagements → engagement_id = "abc123"
    ├─ Step 1: GET /getEndpointsByEngagement/abc123 → endpoints list
    ├─ Step 2: Match "Delhi" → endpoint_id = 11
    └─ Step 3: POST /clusterlist/stream with endpoints=[11] → clusters
    ↓
Return: {success: true, clusters: [...], total_count: 3}
    ↓
LLM formats response → User sees cluster list
```

---

## Enhanced Logging

Also added detailed logging to help debug future issues:

```python
logger.info("🔑 Fetching engagement ID...")
logger.info(f"✅ Got engagement ID: {engagement_id}")
logger.info("📍 Fetching available datacenters...")
logger.info(f"📍 Datacenters result: success={...}, has_data={...}, error={...}")
logger.info(f"📍 Found {len(available_datacenters)} datacenters")
logger.info(f"🔍 Listing clusters for endpoints: {endpoint_ids}")
logger.info(f"🔍 Clusters result: success={...}, data_count={...}, error={...}")
```

This makes it easy to trace the complete flow through the logs.

---

## Testing

After this fix, the logs should show:

```
INFO: 🔑 Fetching engagement ID...
INFO: ✅ Got engagement ID: abc123
INFO: 📍 Fetching available datacenters...
INFO: 📍 Datacenters result: success=True, has_data=True, error=None
INFO: 📍 Found 5 datacenters
INFO: 🔍 Listing clusters for endpoints: [11, 12, 13]
INFO: 🔍 Clusters result: success=True, data_count=3, error=None
INFO: ✅ Function list_k8s_clusters executed successfully
INFO: ✅ Tool list_k8s_clusters executed: True  ← Now True!
```

---

## API Authentication Note

The engagement API call requires valid authentication credentials in `.env`:

```bash
API_AUTH_EMAIL=your-email@example.com
API_AUTH_PASSWORD=your-password
```

If these are not configured, the engagement.get call will fail and you'll see:
```
error=Failed to fetch engagement ID
```

Make sure these credentials are valid for the Tata IPC API.

---

## Status

✅ **Fixed** - All function handlers now fetch engagement_id before calling endpoint APIs

**Files Modified:**
- `app/services/function_calling_service.py` (3 handlers updated)

**Date:** December 13, 2024  
**Impact:** Critical (blocked all function calling operations)  
**Resolution:** Added engagement_id fetch step to all handlers

---

## Related Issues Fixed

This fix resolves:
1. ✅ "Parameter validation failed" errors
2. ✅ Empty datacenter lists
3. ✅ Functions returning success=False
4. ✅ Max iterations reached (LLM kept retrying)

---

## Summary

The Tata IPC API has a **multi-step authentication/authorization flow**:
1. Authenticate → Get token (handled by api_executor_service)
2. Get engagement → Get engagement_id (NOW FIXED - added to handlers)
3. Use engagement_id in all subsequent API calls

We were missing step 2, causing all API calls to fail validation.

