# Enterprise RAG Bot Architecture

## Overview
This document explains how the API execution system works and the role of each component.

## Core Components

### 1. `resource_schema.json` - The Single Source of Truth
**Location**: `app/config/resource_schema.json`

**Purpose**: 
- Centralized configuration for **all API endpoints**
- Defines resource types, operations, parameters, validation rules, and workflows
- No hardcoded URLs or API logic in code

**Structure**:
```json
{
  "resources": {
    "k8s_cluster": {
      "operations": ["create", "list", "get_iks_images", "get_network_list", ...],
      "api_endpoints": {
        "get_iks_images": {
          "method": "GET",
          "url": "https://ipcloud.tatacommunications.com/paasservice/paas/{engagement_id}/iks/images/version",
          "description": "Get IKS images with datacenter options and k8s versions"
        }
      },
      "parameters": {
        "get_iks_images": {
          "required": ["engagement_id"],
          "optional": []
        }
      }
    }
  }
}
```

**Benefits**:
- ✅ Change API URLs without touching code
- ✅ Add new endpoints by editing JSON
- ✅ Validate parameters automatically
- ✅ Document APIs in one place
- ✅ Easy to audit and maintain

---

### 2. `APIExecutorService` - The Execution Engine
**Location**: `app/services/api_executor_service.py`

**Purpose**:
- Reads `resource_schema.json` and executes API calls
- Handles authentication, retries, error handling
- **Does NOT hardcode API URLs or mock data** - uses schema only
- **Returns error responses when APIs fail** - no fallback mocks

**Key Methods**:

#### Core Execution Method
```python
async def execute_operation(
    resource_type: str,      # e.g., "k8s_cluster"
    operation: str,          # e.g., "get_iks_images"
    params: Dict[str, Any],  # e.g., {"engagement_id": 123}
    user_roles: List[str]    # For permission checking
) -> Dict[str, Any]:
```

This method:
1. Looks up the operation in `resource_schema.json`
2. Validates parameters
3. Checks permissions
4. Makes the API call
5. Returns the result

#### Helper Methods (Use `execute_operation` internally)
```python
# These call execute_operation with proper resource_type and operation
async def get_iks_images_and_datacenters(engagement_id)
async def get_network_drivers(endpoint_id, k8s_version)
async def get_environments_and_business_units(engagement_id)
async def get_zones_list(engagement_id)
async def get_os_images(zone_id, circuit_id, k8s_version)
async def get_flavors(zone_id, circuit_id, os_model, node_type)
```

**Pattern** (All methods follow this):
```python
async def get_iks_images_and_datacenters(self, engagement_id: int):
    # 1. Call execute_operation (uses schema)
    result = await self.execute_operation(
        resource_type="k8s_cluster",
        operation="get_iks_images",
        params={"engagement_id": engagement_id},
        user_roles=None
    )
    
    # 2. Parse API response
    if result.get("success"):
        return process_api_data(result["data"])
    
    # 3. Return error (NO MOCK DATA)
    return {
        "success": False,
        "error": "Failed to fetch data from API",
        "datacenters": []
    }
```

**Key Principle**: Service layer is **thin and clean** - it only calls APIs and parses responses. No business logic, no mock data.

---

### 3. Agent Layer - Business Logic
**Location**: `app/agents/handlers/cluster_creation_handler.py`

**Purpose**:
- Orchestrates the 17-step cluster creation workflow
- Calls `APIExecutorService` helper methods
- Manages conversation state

**Example**:
```python
# Step 2: Ask for datacenter
engagement_id = await api_executor_service.get_engagement_id()
dc_result = await api_executor_service.get_iks_images_and_datacenters(engagement_id)

# Handler must check for errors
if not dc_result.get("success"):
    return {
        "agent_name": "ValidationAgent",
        "success": False,
        "output": "Sorry, I couldn't fetch datacenter options. Please try again later."
    }

state._datacenter_options = dc_result.get("datacenters", [])
```

**Responsibility**: Handlers contain the **business logic** - they decide what to do when APIs fail, how to present data to users, and manage the conversation flow.

---

## Data Flow

```
User Query
    ↓
Agent (cluster_creation_handler.py)
    ↓
APIExecutorService Helper Method
    ↓
execute_operation()
    ↓
resource_schema.json ← Reads endpoint config
    ↓
_make_api_call() ← Makes HTTP request
    ↓
Real API / Fallback Mock Data
    ↓
Parse & Return
    ↓
Agent processes response
    ↓
User receives answer
```

---

## Adding a New API Endpoint

### ❌ WRONG WAY (Don't do this):
```python
# DON'T hardcode URLs in api_executor_service.py
async def get_new_data(self):
    url = "https://ipcloud.../new/endpoint"  # ❌ BAD!
    response = await self.http_client.get(url)
```

### ✅ RIGHT WAY (Do this):

**Step 1**: Add to `resource_schema.json`
```json
{
  "resources": {
    "k8s_cluster": {
      "api_endpoints": {
        "get_new_data": {
          "method": "GET",
          "url": "https://ipcloud.../new/endpoint/{param}",
          "description": "Fetch new data"
        }
      },
      "parameters": {
        "get_new_data": {
          "required": ["param"],
          "optional": []
        }
      }
    }
  }
}
```

**Step 2**: Add helper method in `api_executor_service.py`
```python
async def get_new_data(self, param: str) -> Dict[str, Any]:
    """Fetch new data using resource_schema.json configuration."""
    
    # Use execute_operation (reads from schema)
    result = await self.execute_operation(
        resource_type="k8s_cluster",
        operation="get_new_data",
        params={"param": param},
        user_roles=None
    )
    
    if result.get("success"):
        return {"success": True, "data": result["data"]}
    
    # Fallback
    return {"success": False, "error": "API unavailable"}
```

**Step 3**: Use in agent
```python
result = await api_executor_service.get_new_data("value")
```

---

## Why This Architecture?

### ✅ Benefits:
1. **Separation of Concerns**: Schema = Config, Service = Execution, Agent = Logic
2. **Easy to Maintain**: Change API URL → Edit JSON only
3. **Testable**: Mock APIs by creating test schema or using test doubles
4. **Auditable**: All APIs documented in one place
5. **Scalable**: Add new resources without touching existing code
6. **Type-Safe**: Schema validates parameters automatically
7. **Clean Service Layer**: No mock data, no business logic - just API calls
8. **Error Handling**: Handlers decide how to handle API failures gracefully

### ❌ Anti-Patterns to Avoid:
1. **Hardcoding URLs** in Python files
2. **Duplicating API calls** across multiple files
3. **Mixing business logic with API calls** in service layer
4. **Bypassing execute_operation()** method
5. **Adding mock data fallbacks** in service layer (handlers should decide what to do on failure)

---

## Current API Endpoints (Configured in Schema)

| Operation | URL | Purpose |
|-----------|-----|---------|
| `get_iks_images` | `/paas/{engagement_id}/iks/images/version` | Get datacenters & K8s versions |
| `get_network_list` | `/paas/getNetworkList/{endpointId}/{k8sVersion}/APP` | Get CNI drivers |
| `get_environments` | `/environment/getEnvironmentListPerEngagement/{engagement_id}` | Get business units |
| `get_zones` | `/zone/getZoneList/{engagement_id}` | Get network zones |
| `get_os_images` | `/configservice/ppuEnabledImages/{zoneId}` | Get OS options |
| `get_flavors` | `/configservice/ppuEnabledFlavors/{zoneId}` | Get compute flavors |

All configured in `resource_schema.json` ✅

---

## Summary

**Golden Rule**: 
> **Never hardcode API URLs in Python. Always use `resource_schema.json` + `execute_operation()`**

This keeps the codebase clean, maintainable, and follows the Single Source of Truth principle.

