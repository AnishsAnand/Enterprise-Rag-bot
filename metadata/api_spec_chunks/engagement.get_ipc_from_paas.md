# API Specification: engagement - get_ipc_from_paas

**Resource:** engagement
**Operation:** get_ipc_from_paas
**Aliases:** convert engagement, get ipc engagement, paas to ipc, engagement conversion, ipc id

## Endpoint
- **Method:** GET
- **URL:** h{BASE_URL_PAAS_SERVICE}/paas/getIpcEngFromPaasEng/{engagement_id}
- **Auth:** Bearer token (from Keycloak)
- **Description:** Convert a PaaS engagement ID to its corresponding IPC (IPCloud) engagement ID. This is essential for cross-service API calls that require IPC engagement IDs

## Required Parameters
- `engagement_id` - PaaS engagement identifier (path parameter)

## Optional Parameters
None

## Response Mapping
- `status`: status
- `message`: message
- `paas_engagement_id`: data.paasEngagementId
- `ipc_engagement_id`: data.ipcEngagementId
- `engagement_name`: data.engagementName
- `conversion_valid`: data.valid
- `created_at`: data.createdAt
- `mapping_type`: data.mappingType

## Response Example
```json
{
  "status": "success",
  "data": {
    "paasEngagementId": "paas-eng-12345",
    "ipcEngagementId": "ipc-67890",
    "engagementName": "Production Environment",
    "valid": true,
    "mappingType": "direct",
    "metadata": {
      "region": "india",
      "platform": "kubernetes",
      "tier": "enterprise"
    },
    "createdAt": "2024-01-15T10:30:00Z",
    "lastSync": "2025-02-13T10:00:00Z"
  },
  "message": "Engagement mapping retrieved successfully",
  "responseCode": 0
}
```

## Response Fields Details

### Engagement Fields
- **paasEngagementId** - Original PaaS engagement identifier
- **ipcEngagementId** - Corresponding IPC engagement identifier
- **engagementName** - Human-readable engagement name
- **valid** - Whether the mapping is currently valid
- **mappingType** - Type of mapping (direct, derived, legacy)
- **metadata** - Additional engagement metadata
- **createdAt** - When mapping was created
- **lastSync** - Last synchronization timestamp

### Mapping Types
- `direct` - 1:1 mapping between PaaS and IPC
- `derived` - IPC ID derived from PaaS hierarchy
- `legacy` - Migration from old system
- `shared` - Multiple PaaS engagements share one IPC

## Permissions
Roles: admin, developer, viewer

## Workflow Steps
### Workflow: get_ipc_engagement
Convert PaaS to IPC engagement ID
- Step 1: authenticate (auth.validate_token)
- Step 2: validate_paas_engagement (engagement.validate) (depends on: engagement_id)
- Step 3: get_ipc_mapping (engagement.get_ipc_from_paas) (depends on: engagement_id)

## Usage Notes
- This conversion is required for many cross-service operations
- IPC engagement ID is used by portal services (security, tags, etc.)
- PaaS engagement ID is used by PaaS services (clusters, volumes, etc.)
- Mapping is cached for 15 minutes
- Invalid engagements return 404
- Some legacy engagements may not have mappings

## Common Use Cases
1. **Cross-service calls**: "Get IPC engagement for PaaS operations"
2. **Department lookup**: "Convert engagement to list departments"
3. **Tag management**: "Get tags for PaaS engagement"
4. **Security operations**: "Check permissions for PaaS resource"
5. **Billing integration**: "Map PaaS usage to IPC billing"
6. **Migration**: "Verify engagement mapping after migration"

## Data Processing Examples

### Cache IPC Mapping
```python
# Store mapping for reuse
paas_id = "paas-eng-12345"
response = get_ipc_engagement(paas_id)

ipc_id = response['data']['ipcEngagementId']
engagement_name = response['data']['engagementName']

# Use in subsequent calls
departments = list_departments(ipc_id)
tags = list_tags(ipc_id)
```

### Validate Mapping
```python
response = get_ipc_engagement(paas_engagement_id)

if not response['data']['valid']:
    print(f"WARNING: Invalid mapping for {paas_engagement_id}")
    # Handle invalid mapping
elif response['data']['mappingType'] == 'legacy':
    print("NOTE: Using legacy mapping, consider migration")
```

### Batch Conversion
```python
paas_engagements = ["paas-eng-001", "paas-eng-002", "paas-eng-003"]
mappings = {}

for paas_id in paas_engagements:
    try:
        response = get_ipc_engagement(paas_id)
        mappings[paas_id] = response['data']['ipcEngagementId']
    except NotFoundError:
        print(f"No mapping for {paas_id}")
        mappings[paas_id] = None
```

## Integration with Other APIs

This endpoint is commonly used before calling:

### Portal Services (require IPC ID)
- `/portalservice/tag/list?engagementId={ipc_engagement_id}`
- `/portalservice/securityservice/departments/{ipc_engagement_id}`
- `/portalservice/billing/usage/{ipc_engagement_id}`

### Typical Flow
```
1. User has PaaS engagement: "paas-eng-12345"
2. Call: /getIpcEngFromPaasEng/paas-eng-12345
3. Get IPC ID: "ipc-67890"
4. Use IPC ID for: /portalservice/tag/list?engagementId=ipc-67890
```

### Code Example
```python
async def list_tags_for_paas_engagement(paas_engagement_id):
    # Convert to IPC ID
    ipc_response = await get_ipc_from_paas(paas_engagement_id)
    ipc_id = ipc_response['data']['ipcEngagementId']
    
    # Use IPC ID for tag service
    tags = await list_tags(engagement_id=ipc_id)
    return tags
```

## Kafka Use Cases

### Kafka Resource Tagging
```python
# Get IPC engagement for Kafka cluster
paas_eng = "paas-kafka-prod"
ipc_response = await get_ipc_from_paas(paas_eng)
ipc_id = ipc_response['data']['ipcEngagementId']

# List Kafka resources with tags
kafka_resources = await fetch_associated_tags(
    tags="kafka,production",
    engagement_id=ipc_id
)
```

### Kafka Department Access
```python
# Check which departments can access Kafka cluster
ipc_id = get_ipc_id_from_paas(kafka_cluster_engagement)
departments = list_departments(ipc_id)

for dept in departments:
    print(f"{dept['name']}: {dept['permissions']}")
```

## Related Operations
- `engagement.get` - Get PaaS engagement details
- `engagement.list` - List all engagements
- `engagement.sync` - Force sync PaaS-IPC mappings
- `department.list` - List departments (requires IPC ID)
- `tag.list` - List tags (requires IPC ID)

## Error Handling
- **400 Bad Request:** Invalid engagement ID format
- **401 Unauthorized:** Invalid or expired authentication token
- **403 Forbidden:** User does not have access to this engagement
- **404 Not Found:** Engagement ID not found or no mapping exists
- **500 Internal Server Error:** Mapping service unavailable

## Response Codes
- `0` - Success
- `1` - Engagement not found
- `2` - Mapping not found
- `3` - Invalid engagement
- `4` - Mapping expired

## Migration Scenarios

### Legacy Engagement Migration
Old systems may have different ID formats:
```json
{
  "paasEngagementId": "legacy-12345",
  "ipcEngagementId": "ipc-67890",
  "mappingType": "legacy",
  "migrationStatus": "completed",
  "originalId": "old-system-id-999"
}
```

### Shared Engagements
Multiple PaaS engagements sharing one IPC:
```json
{
  "paasEngagementId": "paas-dev-12345",
  "ipcEngagementId": "ipc-shared-001",
  "mappingType": "shared",
  "sharedWith": ["paas-staging-12345", "paas-test-12345"]
}
```

## Caching Strategy

### Recommended Caching
```python
from functools import lru_cache
from datetime import datetime, timedelta

cache_expiry = {}

@lru_cache(maxsize=100)
def get_ipc_with_cache(paas_id):
    # Check if cache expired (15 min)
    if paas_id in cache_expiry:
        if datetime.now() > cache_expiry[paas_id]:
            # Clear expired cache
            get_ipc_with_cache.cache_clear()
            del cache_expiry[paas_id]
    
    response = get_ipc_engagement(paas_id)
    cache_expiry[paas_id] = datetime.now() + timedelta(minutes=15)
    
    return response['data']['ipcEngagementId']
```

## Performance Notes
- Response time typically < 100ms
- Mapping cached for 15 minutes
- Bulk conversions should be batched
- Consider caching in application layer
- Mapping service is highly available (99.9% uptime)

## Best Practices

### Always Convert Before Portal Calls
```python
# WRONG - Using PaaS ID with portal service
tags = list_tags(paas_engagement_id)  # ❌ Will fail

# RIGHT - Convert first
ipc_id = get_ipc_from_paas(paas_engagement_id)
tags = list_tags(ipc_id)  # ✅ Works
```

### Handle Missing Mappings
```python
try:
    ipc_id = get_ipc_from_paas(paas_id)
except NotFoundError:
    # Log and handle gracefully
    logger.warning(f"No IPC mapping for {paas_id}")
    # Use fallback or skip operation
    return None
```

### Cache Aggressively
```python
# Cache at application level
engagement_cache = {}

def get_ipc_cached(paas_id):
    if paas_id not in engagement_cache:
        engagement_cache[paas_id] = get_ipc_from_paas(paas_id)
    return engagement_cache[paas_id]
```

## Troubleshooting

### Issue: 404 Not Found
**Cause:** Engagement doesn't exist or mapping not created
**Solution:**
1. Verify PaaS engagement exists
2. Check if mapping sync is pending
3. Contact support to create mapping

### Issue: Invalid Mapping
**Cause:** Stale or corrupted mapping
**Solution:**
1. Force sync: `/engagement/sync/{engagement_id}`
2. Clear cache
3. Retry after sync completes

### Issue: Slow Response
**Cause:** Cache miss or service degradation
**Solution:**
1. Implement client-side caching
2. Batch requests if possible
3. Check service health status

## Metadata
- **Generated:** 2025-02-13T12:30:00Z
- **Source:** Dynamic API Spec Generator
- **API Version:** v1
- **Base Path:** /paasservice/paas
- **Critical:** Required for cross-service integration
- **Cache TTL:** 15 minutes recommended