# API Specification: engagement - get_user_engagements

**Resource:** engagement
**Operation:** get_user_engagements
**Aliases:** user engagements, my engagements, list engagements, get engagements, user access

## Endpoint
- **Method:** GET
- **URL:** https://ipcloud.tatacommunications.com/catalyst-service-v2/data/configservice/getuserengagements
- **Auth:** Bearer token (from Keycloak)
- **Description:** Get all engagements accessible to the authenticated user, including their roles, permissions, and engagement metadata. Essential for determining user's access scope

## Required Parameters
None (uses authentication token to identify user)

## Optional Parameters
- `include_inactive` - Include inactive engagements (true/false, default: false)
- `role_filter` - Filter by role (admin, developer, viewer)
- `engagement_type` - Filter by type (paas, ipc, hybrid)

## Response Mapping
- `status`: status
- `message`: message
- `user_id`: data.userId
- `user_email`: data.userEmail
- `engagements`: data.engagements
- `engagement_ids`: data.engagements[*].id
- `engagement_names`: data.engagements[*].name
- `user_roles`: data.engagements[*].role
- `active_engagements`: data.engagements[*][?active==true]
- `total_count`: data.totalCount

## Response Example
```json
{
  "status": "success",
  "data": {
    "userId": "user-12345",
    "userEmail": "john.doe@company.com",
    "userName": "John Doe",
    "totalCount": 8,
    "activeCount": 6,
    "engagements": [
      {
        "id": "paas-eng-prod-001",
        "name": "Production Environment",
        "type": "paas",
        "role": "admin",
        "permissions": [
          "cluster.create",
          "cluster.delete",
          "cluster.update",
          "volume.create",
          "kafka.manage"
        ],
        "active": true,
        "region": "india",
        "platform": "kubernetes",
        "createdAt": "2024-01-15T10:30:00Z",
        "lastAccessed": "2025-02-13T09:45:00Z",
        "metadata": {
          "department": "Engineering",
          "costCenter": "CC-001",
          "environment": "production",
          "tier": "enterprise"
        },
        "resources": {
          "clusters": 12,
          "volumes": 45,
          "kafkaClusters": 3,
          "databases": 8
        },
        "quotas": {
          "maxClusters": 20,
          "maxVolumes": 100,
          "maxVCPUs": 200,
          "maxMemoryGB": 512
        }
      },
      {
        "id": "paas-eng-staging-001",
        "name": "Staging Environment",
        "type": "paas",
        "role": "developer",
        "permissions": [
          "cluster.create",
          "cluster.update",
          "volume.create",
          "kafka.view"
        ],
        "active": true,
        "region": "india",
        "platform": "kubernetes",
        "createdAt": "2024-02-01T14:20:00Z",
        "lastAccessed": "2025-02-12T16:30:00Z",
        "metadata": {
          "department": "Engineering",
          "costCenter": "CC-001",
          "environment": "staging",
          "tier": "standard"
        },
        "resources": {
          "clusters": 5,
          "volumes": 20,
          "kafkaClusters": 1,
          "databases": 3
        },
        "quotas": {
          "maxClusters": 10,
          "maxVolumes": 50,
          "maxVCPUs": 100,
          "maxMemoryGB": 256
        }
      },
      {
        "id": "paas-eng-dev-001",
        "name": "Development Environment",
        "type": "paas",
        "role": "developer",
        "permissions": [
          "cluster.create",
          "volume.create",
          "kafka.view"
        ],
        "active": true,
        "region": "india",
        "platform": "kubernetes",
        "createdAt": "2024-03-10T09:00:00Z",
        "lastAccessed": "2025-02-13T10:15:00Z",
        "metadata": {
          "department": "Engineering",
          "costCenter": "CC-001",
          "environment": "development",
          "tier": "basic"
        },
        "resources": {
          "clusters": 8,
          "volumes": 15,
          "kafkaClusters": 1,
          "databases": 2
        },
        "quotas": {
          "maxClusters": 15,
          "maxVolumes": 30,
          "maxVCPUs": 50,
          "maxMemoryGB": 128
        }
      },
      {
        "id": "ipc-eng-shared-001",
        "name": "Shared Services",
        "type": "ipc",
        "role": "viewer",
        "permissions": [
          "cluster.view",
          "volume.view",
          "kafka.view"
        ],
        "active": true,
        "region": "india",
        "platform": "multi",
        "createdAt": "2023-12-01T08:00:00Z",
        "lastAccessed": "2025-02-10T11:20:00Z",
        "metadata": {
          "department": "Shared Services",
          "costCenter": "CC-SHARED",
          "environment": "production",
          "tier": "enterprise"
        },
        "resources": {
          "clusters": 20,
          "volumes": 80,
          "kafkaClusters": 5,
          "databases": 15
        }
      }
    ],
    "summary": {
      "totalEngagements": 8,
      "activeEngagements": 6,
      "inactiveEngagements": 2,
      "byRole": {
        "admin": 2,
        "developer": 4,
        "viewer": 2
      },
      "byEnvironment": {
        "production": 2,
        "staging": 2,
        "development": 3,
        "shared": 1
      },
      "totalResources": {
        "clusters": 45,
        "volumes": 160,
        "kafkaClusters": 10,
        "databases": 28
      }
    }
  },
  "message": "User engagements retrieved successfully",
  "responseCode": 0
}
```

## Response Fields Details

### User Fields
- **userId** - Unique user identifier
- **userEmail** - User's email address
- **userName** - User's display name
- **totalCount** - Total engagements accessible
- **activeCount** - Number of active engagements

### Engagement Fields
- **id** - Unique engagement identifier
- **name** - Human-readable engagement name
- **type** - Engagement type (paas, ipc, hybrid)
- **role** - User's role in this engagement
- **permissions** - Array of permission strings
- **active** - Whether engagement is currently active
- **region** - Geographic region
- **platform** - Platform type (kubernetes, openshift, etc.)
- **createdAt** - When engagement was created
- **lastAccessed** - Last user access timestamp
- **metadata** - Additional engagement metadata
- **resources** - Resource counts in engagement
- **quotas** - Resource quota limits

### User Roles
- `admin` - Full administrative access
- `developer` - Create and manage resources
- `viewer` - Read-only access
- `operator` - Operational tasks only
- `billing` - Billing and cost access

### Engagement Types
- `paas` - PaaS-only engagement
- `ipc` - IPCloud portal engagement
- `hybrid` - Both PaaS and IPC
- `legacy` - Migrated from old system

## Permissions
Roles: All authenticated users (shows their own engagements)

## Workflow Steps
### Workflow: get_user_engagements
Get engagements for current user
- Step 1: authenticate (auth.validate_token)
- Step 2: get_user_engagements (engagement.get_user_engagements)

## Usage Notes
- Returns only engagements user has access to
- Authentication token determines which user
- Results are not cached (always fresh)
- Inactive engagements excluded by default
- Permissions are role-based + custom
- Resource counts are real-time snapshots

## Common Use Cases
1. **Initial app load**: "Load user's accessible engagements"
2. **Engagement switcher**: "Show engagement dropdown list"
3. **Permission check**: "Can user create Kafka cluster in this engagement?"
4. **Resource overview**: "How many clusters across all engagements?"
5. **Quota monitoring**: "Check quota usage across engagements"
6. **Access audit**: "When did user last access each engagement?"

## Data Processing Examples

### Build Engagement Dropdown
```python
response = get_user_engagements()
engagements = response['data']['engagements']

# Active engagements only
active = [e for e in engagements if e['active']]

dropdown_options = [
    {
        'value': e['id'],
        'label': f"{e['name']} ({e['metadata']['environment']})",
        'role': e['role']
    }
    for e in active
]
```

### Check Permission
```python
def can_user_create_kafka(engagement_id):
    response = get_user_engagements()
    
    engagement = next(
        (e for e in response['data']['engagements'] if e['id'] == engagement_id),
        None
    )
    
    if not engagement:
        return False
    
    return 'kafka.manage' in engagement.get('permissions', [])
```

### Calculate Total Resources
```python
response = get_user_engagements()
total_clusters = sum(
    e['resources'].get('clusters', 0)
    for e in response['data']['engagements']
    if e['active']
)

total_kafka = sum(
    e['resources'].get('kafkaClusters', 0)
    for e in response['data']['engagements']
    if e['active']
)

print(f"Total clusters: {total_clusters}")
print(f"Total Kafka clusters: {total_kafka}")
```

### Find Admin Engagements
```python
response = get_user_engagements()
admin_engagements = [
    e for e in response['data']['engagements']
    if e['role'] == 'admin' and e['active']
]

print(f"You have admin access to {len(admin_engagements)} engagements:")
for eng in admin_engagements:
    print(f"  - {eng['name']}")
```

## Kafka Integration

### Find Kafka-Enabled Engagements
```python
response = get_user_engagements()

kafka_engagements = [
    e for e in response['data']['engagements']
    if e['resources'].get('kafkaClusters', 0) > 0
]

for eng in kafka_engagements:
    print(f"{eng['name']}: {eng['resources']['kafkaClusters']} Kafka clusters")
    print(f"  Your role: {eng['role']}")
    
    # Check Kafka permissions
    kafka_perms = [p for p in eng['permissions'] if 'kafka' in p.lower()]
    print(f"  Kafka permissions: {', '.join(kafka_perms)}")
```

### Kafka Permission Matrix
```python
response = get_user_engagements()

print("Kafka Access Matrix:")
print("-" * 60)

for eng in response['data']['engagements']:
    if not eng['active']:
        continue
    
    kafka_manage = 'kafka.manage' in eng['permissions']
    kafka_view = 'kafka.view' in eng['permissions']
    
    access = "Full" if kafka_manage else "View" if kafka_view else "None"
    
    print(f"{eng['name']:30} | {access:10} | {eng['role']}")
```

## Related Operations
- `engagement.get` - Get specific engagement details
- `engagement.switch` - Switch active engagement context
- `user.get_profile` - Get user profile
- `permission.check` - Check specific permission
- `quota.get` - Get detailed quota information

## Error Handling
- **401 Unauthorized:** Invalid or expired authentication token
- **403 Forbidden:** Token valid but user account disabled
- **500 Internal Server Error:** Service unavailable

## Response Codes
- `0` - Success
- `1` - User not found
- `2` - No engagements found
- `3` - Service unavailable

## UI Integration

### React Example
```javascript
import { useEffect, useState } from 'react';

function EngagementSelector() {
  const [engagements, setEngagements] = useState([]);
  const [selected, setSelected] = useState(null);
  
  useEffect(() => {
    fetch('/catalyst-service-v2/data/configservice/getuserengagements')
      .then(res => res.json())
      .then(data => {
        const active = data.data.engagements.filter(e => e.active);
        setEngagements(active);
        setSelected(active[0]?.id);
      });
  }, []);
  
  return (
    <select value={selected} onChange={e => setSelected(e.target.value)}>
      {engagements.map(eng => (
        <option key={eng.id} value={eng.id}>
          {eng.name} ({eng.metadata.environment}) - {eng.role}
        </option>
      ))}
    </select>
  );
}
```

### Display Resource Summary
```javascript
function ResourceSummary({ engagements }) {
  const totals = engagements.reduce((acc, eng) => ({
    clusters: acc.clusters + eng.resources.clusters,
    volumes: acc.volumes + eng.resources.volumes,
    kafkaClusters: acc.kafkaClusters + eng.resources.kafkaClusters,
  }), { clusters: 0, volumes: 0, kafkaClusters: 0 });
  
  return (
    <div>
      <h3>Your Resources</h3>
      <p>Clusters: {totals.clusters}</p>
      <p>Volumes: {totals.volumes}</p>
      <p>Kafka Clusters: {totals.kafkaClusters}</p>
    </div>
  );
}
```

## Security Considerations

### Token-Based Access
- Engagements filtered by token permissions
- Cannot see other users' engagements
- Permissions are enforced server-side
- Token must not be expired

### Role Verification
```python
# Always verify role before operations
def verify_admin_access(engagement_id):
    engagements = get_user_engagements()
    
    for eng in engagements['data']['engagements']:
        if eng['id'] == engagement_id:
            if eng['role'] != 'admin':
                raise PermissionError(f"Admin role required for {engagement_id}")
            return True
    
    raise NotFoundError(f"Engagement {engagement_id} not accessible")
```

## Performance Notes
- Response time typically < 500ms
- Not cached (always fresh data)
- Resource counts may be cached (updated every 5 min)
- Large number of engagements (50+) may be slower
- Consider client-side caching with TTL

## Best Practices

### Cache Client-Side
```python
from datetime import datetime, timedelta

engagement_cache = {
    'data': None,
    'expires': None
}

def get_engagements_cached(ttl_minutes=5):
    now = datetime.now()
    
    if (engagement_cache['data'] is None or 
        engagement_cache['expires'] is None or
        now > engagement_cache['expires']):
        
        engagement_cache['data'] = get_user_engagements()
        engagement_cache['expires'] = now + timedelta(minutes=ttl_minutes)
    
    return engagement_cache['data']
```

### Filter by Context
```python
# Get only production engagements
def get_production_engagements():
    response = get_user_engagements()
    return [
        e for e in response['data']['engagements']
        if e['metadata'].get('environment') == 'production'
        and e['active']
    ]
```

### Check Quotas
```python
def check_quota_available(engagement_id, resource_type):
    response = get_user_engagements()
    
    eng = next((e for e in response['data']['engagements'] 
                if e['id'] == engagement_id), None)
    
    if not eng:
        return False
    
    current = eng['resources'].get(resource_type, 0)
    max_quota = eng['quotas'].get(f'max{resource_type.capitalize()}', 0)
    
    return current < max_quota
```

## Metadata
- **Generated:** 2025-02-13T12:30:00Z
- **Source:** Dynamic API Spec Generator
- **API Version:** v2
- **Base Path:** /catalyst-service-v2/data/configservice
- **Critical:** Essential for user context and permissions
- **Cache:** Not recommended (always fetch fresh)