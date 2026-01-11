# Extended Managed Services Integration

## Date: December 12, 2025

---

## 🎯 Overview

Extended the managed services integration to include 4 additional service types, following the same pattern as Kafka and GitLab.

---

## ✅ New Services Added

### 1. Container Registry (IKSContainerRegistry)
- **Resource Type:** `container_registry`
- **Service Type:** `IKSContainerRegistry`
- **API Endpoint:** `https://ipcloud.tatacommunications.com/paasservice/api/v1/paas/listManagedServices/IKSContainerRegistry`
- **Aliases:** container registry, registry, registries, docker registry, image registry

### 2. Jenkins (IKSJenkins)
- **Resource Type:** `jenkins`
- **Service Type:** `IKSJenkins`
- **API Endpoint:** `https://ipcloud.tatacommunications.com/paasservice/api/v1/paas/listManagedServices/IKSJenkins`
- **Aliases:** jenkins, jenkins service, jenkins services, ci cd, continuous integration

### 3. PostgreSQL (IKSPostgres)
- **Resource Type:** `postgres`
- **Service Type:** `IKSPostgres`
- **API Endpoint:** `https://ipcloud.tatacommunications.com/paasservice/api/v1/paas/listManagedServices/IKSPostgres`
- **Aliases:** postgres, postgresql, postgres service, postgresql database, pg

### 4. DocumentDB (IKSDocumentDB)
- **Resource Type:** `documentdb`
- **Service Type:** `IKSDocumentDB`
- **API Endpoint:** `https://ipcloud.tatacommunications.com/paasservice/api/v1/paas/listManagedServices/IKSDocumentDB`
- **Aliases:** documentdb, document db, mongodb, mongo, nosql database

---

## 📝 Files Modified

### 1. `app/config/resource_schema.json`

**Changes:**
- Added 4 new service types to `managed_service.service_types`
- Created individual resource definitions for each service
- Configured API endpoints, parameters, and workflows

**Service Types Added:**
```json
{
  "container_registry": {
    "api_value": "IKSContainerRegistry",
    "display_name": "Container Registry",
    "description": "Docker container registry service"
  },
  "jenkins": {
    "api_value": "IKSJenkins",
    "display_name": "Jenkins",
    "description": "Jenkins CI/CD automation server"
  },
  "postgres": {
    "api_value": "IKSPostgres",
    "display_name": "PostgreSQL",
    "description": "PostgreSQL relational database service"
  },
  "documentdb": {
    "api_value": "IKSDocumentDB",
    "display_name": "DocumentDB",
    "description": "MongoDB-compatible document database"
  }
}
```

**Resource Definitions:**
Each service has:
- Operations: `["list"]`
- Aliases: Multiple natural language variations
- Parent resource: `managed_service`
- API endpoints with POST method
- Parameters: `endpoints` (required), `engagementId` and `serviceType` (internal)
- Permissions: `["admin", "developer", "viewer"]`
- Workflow: 4-step process (get engagement → convert to IPC → get endpoints → list services)

---

### 2. `app/services/api_executor_service.py`

**Changes:**
Added 4 new wrapper methods (after `list_gitlab`):

```python
async def list_container_registry(
    self,
    endpoint_ids: List[int] = None,
    ipc_engagement_id: int = None
) -> Dict[str, Any]:
    """List Container Registry managed services."""
    return await self.list_managed_services(
        service_type="IKSContainerRegistry",
        endpoint_ids=endpoint_ids,
        ipc_engagement_id=ipc_engagement_id
    )

async def list_jenkins(...) -> Dict[str, Any]:
    """List Jenkins managed services."""
    # Uses IKSJenkins

async def list_postgres(...) -> Dict[str, Any]:
    """List PostgreSQL managed services."""
    # Uses IKSPostgres

async def list_documentdb(...) -> Dict[str, Any]:
    """List DocumentDB managed services."""
    # Uses IKSDocumentDB
```

**Pattern:**
All methods follow the same pattern as `list_kafka` and `list_gitlab`, calling the generic `list_managed_services` method with the appropriate `service_type`.

---

### 3. `app/agents/execution_agent.py`

**Changes:**

#### A. Execution Handling (after GitLab section)
Added 4 new execution blocks:

```python
# Special handling for Container Registry listing
elif state.resource_type == "container_registry" and state.operation == "list":
    logger.info("📋 Using list_container_registry workflow method")
    # ... endpoint conversion logic ...
    execution_result = await api_executor_service.list_container_registry(
        endpoint_ids=endpoint_ids,
        ipc_engagement_id=None
    )

# Similar blocks for jenkins, postgres, documentdb
```

Each block includes:
- Endpoint name-to-ID conversion (same as Kafka/GitLab)
- Error handling for conversion failures
- Call to the appropriate wrapper method

#### B. Success Message Formatting (in `_format_success_message`)
Added 4 new formatting sections:

**Container Registry:**
```python
if state.resource_type == "container_registry" and state.operation == "list":
    # Format with Registry URL
    message += f"> **Registry URL:** `{ingress_url}`\n"
```

**Jenkins:**
```python
if state.resource_type == "jenkins" and state.operation == "list":
    # Format with Jenkins URL
    message += f"> **Jenkins URL:** `{ingress_url}`\n"
```

**PostgreSQL:**
```python
if state.resource_type == "postgres" and state.operation == "list":
    # Format with Storage size
    message += f"> **Storage:** {db_size}GB\n"
```

**DocumentDB:**
```python
if state.resource_type == "documentdb" and state.operation == "list":
    # Format with Storage size
    message += f"> **Storage:** {db_size}GB\n"
```

**Common Fields Displayed:**
- Service name
- Status (with emoji: ✅ Active/Running, ⚠️ Pending, ❌ Failed)
- Version
- Location (endpoint name)
- Cluster name
- Replicas
- Namespace
- Service-specific fields (URL for web services, Storage for databases)

---

### 4. `app/agents/intent_agent.py`

**Changes:**

#### A. System Prompt Updates
Added examples for each new service:

**Container Registry Examples:**
```
User: "List container registries" or "Show me container registry"
→ intent_detected: true, resource_type: container_registry, operation: list

User: "Show docker registry in Mumbai"
→ intent_detected: true, resource_type: container_registry, operation: list

User: "How many container registries?"
→ intent_detected: true, resource_type: container_registry, operation: list
```

**Similar examples added for:**
- Jenkins (with CI/CD variations)
- PostgreSQL (with Postgres/pg variations)
- DocumentDB (with MongoDB/NoSQL variations)

#### B. Updated Notes Section
```
- For "list" operation on k8s_cluster, kafka, gitlab, container_registry, 
  jenkins, postgres, documentdb: "endpoints" parameter is required
```

#### C. Updated Aliases
```
- Container Registry aliases: container registry, registry, registries, 
  docker registry, image registry
- Jenkins aliases: jenkins, jenkins service, jenkins services, ci cd, 
  continuous integration
- PostgreSQL aliases: postgres, postgresql, postgres service, 
  postgresql database, pg
- DocumentDB aliases: documentdb, document db, mongodb, mongo, 
  nosql database
```

---

## 🔄 Integration Pattern

All 4 new services follow the **exact same pattern** as Kafka and GitLab:

### 1. Resource Schema
```json
{
  "resource_name": {
    "operations": ["list"],
    "aliases": [...],
    "parent_resource": "managed_service",
    "service_type": "IKS{ServiceName}",
    "api_endpoints": {
      "list": {
        "method": "POST",
        "url": "https://.../listManagedServices/IKS{ServiceName}"
      }
    },
    "parameters": {
      "list": {
        "required": ["endpoints"],
        "internal": {
          "engagementId": "ipc_engagement_id",
          "serviceType": "IKS{ServiceName}"
        }
      }
    }
  }
}
```

### 2. API Executor Wrapper
```python
async def list_{service}(...) -> Dict[str, Any]:
    return await self.list_managed_services(
        service_type="IKS{ServiceName}",
        endpoint_ids=endpoint_ids,
        ipc_engagement_id=ipc_engagement_id
    )
```

### 3. Execution Agent Handling
```python
elif state.resource_type == "{service}" and state.operation == "list":
    # Convert endpoint names to IDs
    # Call wrapper method
    execution_result = await api_executor_service.list_{service}(...)
```

### 4. Success Message Formatting
```python
if state.resource_type == "{service}" and state.operation == "list":
    # Format service details
    # Display status, version, location, etc.
```

### 5. Intent Agent Examples
```
User: "List {service}" or "Show me {service}"
→ intent_detected: true, resource_type: {service}, operation: list
```

---

## 🧪 Testing

### Test Queries

**Container Registry:**
```
"List container registries"
"Show me docker registry"
"What registries do we have?"
"Show container registry in Mumbai"
"How many registries?"
```

**Jenkins:**
```
"List Jenkins services"
"Show me Jenkins"
"What CI/CD services do we have?"
"Show Jenkins in Delhi"
"How many Jenkins servers?"
```

**PostgreSQL:**
```
"List PostgreSQL services"
"Show me Postgres databases"
"What Postgres instances do we have?"
"Show Postgres in Chennai"
"How many PostgreSQL databases?"
```

**DocumentDB:**
```
"List DocumentDB services"
"Show me MongoDB services"
"What NoSQL databases do we have?"
"Show DocumentDB in Bengaluru"
"How many DocumentDB instances?"
```

### Expected Flow

1. **Intent Agent** detects resource type and operation
2. **Validation Agent** (if needed) validates/collects endpoints
3. **Execution Agent**:
   - Converts endpoint names to IDs
   - Fetches IPC engagement ID
   - Calls API with payload: `{"engagementId": <ipc_id>, "endpoints": [ids], "serviceType": "IKS..."}`
   - Formats response with service details

### Expected Response Format

```markdown
## ✅ Found X {Service} Service(s)
*Queried Y endpoint(s)*

---

**✅ service-name-01**
> **Status:** Active | **Version:** 1.2.3
> **Location:** Mumbai | **Cluster:** cluster-name
> **Replicas:** 3 | **Namespace:** namespace-name
> **{Service-specific field}:** value

**✅ service-name-02**
...

---

⏱️ *Completed in 2.34 seconds*
```

---

## 📊 API Payload Structure

All services use the **same payload structure**:

```json
{
  "engagementId": 1602,
  "endpoints": [10, 11, 12, 13, 14, 29, 30, 162, 204],
  "serviceType": "IKS{ServiceName}"
}
```

**Where:**
- `engagementId`: IPC Engagement ID (converted from PAAS engagement ID)
- `endpoints`: Array of endpoint IDs (converted from names if needed)
- `serviceType`: One of:
  - `IKSKafka`
  - `IKSGitlab`
  - `IKSContainerRegistry`
  - `IKSJenkins`
  - `IKSPostgres`
  - `IKSDocumentDB`

---

## 🔧 API Response Structure

Expected response format (based on GitLab example):

```json
{
  "status": "success",
  "data": {
    "data": [
      {
        "serviceType": "IKS{ServiceName}",
        "name": "service-name",
        "status": "Active",
        "locationName": "EP_V2_CHN_AMB",
        "version": "1.2.3",
        "clusterName": "cluster-name",
        "replicas": "3",
        "instanceNamespace": "namespace",
        "ingressUrl": "https://...",
        "volumeSize": "50",
        ...
      }
    ],
    "message": "success",
    "responseCode": 0
  }
}
```

**Key Fields:**
- `name` or `serviceName`: Service name
- `status`: Active, Running, Pending, Failed
- `locationName` or `endpointName`: Endpoint name
- `version`: Service version
- `clusterName`: Kubernetes cluster
- `replicas`: Number of replicas
- `instanceNamespace` or `namespace`: K8s namespace
- `ingressUrl` or `url`: Access URL (for web services)
- `volumeSize`: Storage size in GB (for databases)

---

## 🎯 Total Managed Services

The system now supports **6 managed services**:

1. ✅ Kafka (IKSKafka)
2. ✅ GitLab (IKSGitlab)
3. ✅ Container Registry (IKSContainerRegistry)
4. ✅ Jenkins (IKSJenkins)
5. ✅ PostgreSQL (IKSPostgres)
6. ✅ DocumentDB (IKSDocumentDB)

---

## 🚀 Adding Future Services

To add a new managed service (e.g., `IKSRedis`):

### 1. Update `resource_schema.json`
```json
// In managed_service.service_types
"redis": {
  "api_value": "IKSRedis",
  "display_name": "Redis",
  "description": "Redis in-memory data store"
}

// Add new resource definition
"redis": {
  "operations": ["list"],
  "aliases": ["redis", "redis service", "cache"],
  "parent_resource": "managed_service",
  "service_type": "IKSRedis",
  "api_endpoints": {
    "list": {
      "method": "POST",
      "url": "https://.../listManagedServices/IKSRedis"
    }
  },
  "parameters": {
    "list": {
      "required": ["endpoints"],
      "internal": {
        "engagementId": "ipc_engagement_id",
        "serviceType": "IKSRedis"
      }
    }
  },
  "permissions": {
    "list": ["admin", "developer", "viewer"]
  },
  "workflow": { /* same 4-step workflow */ }
}
```

### 2. Add wrapper in `api_executor_service.py`
```python
async def list_redis(
    self,
    endpoint_ids: List[int] = None,
    ipc_engagement_id: int = None
) -> Dict[str, Any]:
    """List Redis managed services."""
    return await self.list_managed_services(
        service_type="IKSRedis",
        endpoint_ids=endpoint_ids,
        ipc_engagement_id=ipc_engagement_id
    )
```

### 3. Add execution handling in `execution_agent.py`
```python
# In execute() method
elif state.resource_type == "redis" and state.operation == "list":
    logger.info("📋 Using list_redis workflow method")
    # ... endpoint conversion ...
    execution_result = await api_executor_service.list_redis(...)

# In _format_success_message() method
if state.resource_type == "redis" and state.operation == "list":
    # Format Redis-specific details
    message += f"> **Cache Size:** {cache_size}MB\n"
```

### 4. Add examples in `intent_agent.py`
```
**Redis Service Examples:**

User: "List Redis services" or "Show me Redis cache"
→ intent_detected: true, resource_type: redis, operation: list

# Update notes and aliases
- Redis aliases: redis, redis service, cache, in-memory cache
```

---

## ✅ Summary

**All 4 new services integrated successfully:**
- ✅ Container Registry (IKSContainerRegistry)
- ✅ Jenkins (IKSJenkins)
- ✅ PostgreSQL (IKSPostgres)
- ✅ DocumentDB (IKSDocumentDB)

**Files Modified:**
- ✅ `resource_schema.json` - Added service types and resource definitions
- ✅ `api_executor_service.py` - Added 4 wrapper methods
- ✅ `execution_agent.py` - Added execution handling and formatting
- ✅ `intent_agent.py` - Added examples and aliases

**Pattern Established:**
- ✅ Consistent structure across all managed services
- ✅ Easy to extend for future services
- ✅ Same API payload structure
- ✅ Same workflow (engagement → IPC → endpoints → list)

**Ready for Testing:**
- ✅ All code changes complete
- ✅ Backend needs restart to load new schema
- ✅ Test queries documented

---

**Status:** 🎉 **INTEGRATION COMPLETE**

All 4 new managed services are ready to use!

