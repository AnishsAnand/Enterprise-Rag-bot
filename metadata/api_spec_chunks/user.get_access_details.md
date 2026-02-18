# API Specification: user - get_access_details

**Resource:** user
**Operation:** get_access_details
**Aliases:** access details, user access, permissions, user permissions, get access

## Endpoint
- **Method:** GET
- **URL:** https://ipcloud.tatacommunications.com/catalyst-user-service/user/accessdetails
- **Auth:** Bearer token (from Keycloak)
- **Description:** Get comprehensive access details for the authenticated user including permissions, roles, accessible resources, and feature flags across all services and engagements

## Required Parameters
None (uses authentication token to identify user)

## Optional Parameters
- `service_name` - Filter by specific service (e.g., "paas", "kafka", "postgres")
- `engagement_id` - Filter by specific engagement
- `include_resources` - Include detailed resource access (true/false)

## Response Mapping
- `status`: status
- `message`: message
- `user_id`: data.userId
- `user_email`: data.userEmail
- `permissions`: data.permissions
- `roles`: data.roles
- `engagements`: data.engagements
- `services`: data.services
- `feature_flags`: data.featureFlags

## Response Example
```json
{
  "status": "success",
  "data": {
    "userId": "user-12345",
    "userEmail": "john.doe@company.com",
    "userName": "John Doe",
    "userType": "customer",
    "accountStatus": "active",
    "lastLogin": "2025-02-13T09:30:00Z",
    "roles": [
      {
        "id": "role-admin-001",
        "name": "admin",
        "displayName": "Administrator",
        "scope": "global",
        "description": "Full administrative access",
        "assignedAt": "2024-01-15T10:00:00Z"
      },
      {
        "id": "role-kafka-manager-001",
        "name": "kafka_manager",
        "displayName": "Kafka Manager",
        "scope": "service",
        "service": "kafka",
        "description": "Manage Kafka clusters",
        "assignedAt": "2024-02-20T14:30:00Z"
      }
    ],
    "permissions": {
      "global": [
        "user.view",
        "user.list",
        "engagement.view",
        "engagement.list"
      ],
      "paas": [
        "cluster.create",
        "cluster.delete",
        "cluster.update",
        "cluster.view",
        "volume.create",
        "volume.delete",
        "volume.view"
      ],
      "kafka": [
        "kafka.cluster.create",
        "kafka.cluster.delete",
        "kafka.cluster.manage",
        "kafka.topic.create",
        "kafka.topic.delete",
        "kafka.consumer.manage"
      ],
      "postgres": [
        "postgres.database.create",
        "postgres.database.view",
        "postgres.user.create",
        "postgres.backup.manage"
      ],
      "billing": [
        "billing.view",
        "billing.reports"
      ]
    },
    "engagements": [
      {
        "id": "paas-eng-prod-001",
        "name": "Production Environment",
        "role": "admin",
        "permissions": [
          "cluster.*",
          "volume.*",
          "kafka.*",
          "postgres.*"
        ],
        "resources": {
          "clusters": 12,
          "volumes": 45,
          "kafkaClusters": 3,
          "databases": 8
        },
        "accessLevel": "full"
      },
      {
        "id": "paas-eng-staging-001",
        "name": "Staging Environment",
        "role": "developer",
        "permissions": [
          "cluster.view",
          "cluster.create",
          "kafka.view",
          "kafka.topic.create"
        ],
        "resources": {
          "clusters": 5,
          "kafkaClusters": 1
        },
        "accessLevel": "limited"
      }
    ],
    "services": {
      "paas": {
        "enabled": true,
        "accessLevel": "full",
        "permissions": [
          "cluster.create",
          "cluster.delete",
          "volume.create"
        ]
      },
      "kafka": {
        "enabled": true,
        "accessLevel": "full",
        "permissions": [
          "kafka.cluster.create",
          "kafka.topic.manage"
        ]
      },
      "postgres": {
        "enabled": true,
        "accessLevel": "full",
        "permissions": [
          "postgres.database.create",
          "postgres.user.manage"
        ]
      },
      "mongodb": {
        "enabled": true,
        "accessLevel": "read",
        "permissions": [
          "mongodb.database.view"
        ]
      },
      "gitlab": {
        "enabled": false,
        "accessLevel": "none",
        "permissions": []
      }
    },
    "featureFlags": {
      "betaFeatures": true,
      "advancedMonitoring": true,
      "autoScaling": true,
      "multiRegion": true,
      "customNetworking": false,
      "aiInsights": true
    },
    "quotas": {
      "maxClusters": 50,
      "maxVolumes": 200,
      "maxKafkaClusters": 20,
      "maxDatabases": 100,
      "maxVCPUs": 500,
      "maxMemoryGB": 2048,
      "maxStorageGB": 10000
    },
    "restrictions": {
      "regionsAllowed": [
        "mumbai-bkc",
        "chennai-amb",
        "delhi",
        "bengaluru"
      ],
      "regionsRestricted": [],
      "ipWhitelist": [],
      "mfaRequired": true,
      "apiRateLimits": {
        "perMinute": 100,
        "perHour": 5000,
        "perDay": 100000
      }
    },
    "preferences": {
      "timezone": "Asia/Kolkata",
      "language": "en",
      "dateFormat": "DD/MM/YYYY",
      "notifications": {
        "email": true,
        "slack": false,
        "webhook": false
      }
    }
  },
  "message": "Access details retrieved successfully",
  "responseCode": 0
}
```

## Response Fields Details

### User Fields
- **userId** - Unique user identifier
- **userEmail** - User email address
- **userName** - Display name
- **userType** - Account type (customer, partner, internal)
- **accountStatus** - Account status (active, suspended, pending)
- **lastLogin** - Last login timestamp

### Role Fields
- **name** - Role identifier
- **displayName** - Human-readable role name
- **scope** - Role scope (global, engagement, service, resource)
- **service** - Specific service if service-scoped
- **assignedAt** - When role was assigned

### Permission Format
Permissions follow pattern: `<service>.<resource>.<action>`
- `cluster.create` - Create clusters
- `kafka.topic.delete` - Delete Kafka topics
- `*` wildcard - All permissions

### Access Levels
- `full` - Complete access with all permissions
- `limited` - Restricted access with subset of permissions
- `read` - Read-only access
- `none` - No access

## Permissions
Roles: All authenticated users (returns their own access details)

## Workflow Steps
### Workflow: get_user_access
Get user access details
- Step 1: authenticate (auth.validate_token)
- Step 2: get_access_details (user.get_access_details)

## Usage Notes
- Returns only current user's access details
- Authentication token determines which user
- Results not cached (always fresh)
- Includes all engagements user has access to
- Feature flags control UI/UX elements
- Quotas enforced at API level
- MFA status affects available operations

## Common Use Cases
1. **Permission check**: "Can user create Kafka cluster?"
2. **Feature availability**: "Is multi-region enabled for user?"
3. **UI customization**: "Show/hide features based on flags"
4. **Quota enforcement**: "How many clusters can user create?"
5. **Access audit**: "What permissions does user have?"
6. **Region filtering**: "Which regions can user deploy to?"

## Data Processing Examples

### Check Specific Permission
```python
def has_permission(access_details, permission):
    # Check global permissions
    global_perms = access_details['data']['permissions'].get('global', [])
    if permission in global_perms:
        return True
    
    # Check service-specific permissions
    for service, perms in access_details['data']['permissions'].items():
        if service != 'global' and permission in perms:
            return True
    
    return False

# Usage
access = get_user_access_details()
can_create_kafka = has_permission(access, 'kafka.cluster.create')
```

### Get Services User Can Access
```python
def get_accessible_services(access_details):
    services = access_details['data']['services']
    
    accessible = [
        {
            'name': name,
            'accessLevel': info['accessLevel'],
            'enabled': info['enabled']
        }
        for name, info in services.items()
        if info['enabled']
    ]
    
    return accessible
```

### Check Kafka Permissions
```python
def get_kafka_permissions(access_details):
    kafka_perms = access_details['data']['permissions'].get('kafka', [])
    
    return {
        'canCreateCluster': 'kafka.cluster.create' in kafka_perms,
        'canDeleteCluster': 'kafka.cluster.delete' in kafka_perms,
        'canCreateTopic': 'kafka.topic.create' in kafka_perms,
        'canManageConsumers': 'kafka.consumer.manage' in kafka_perms,
        'fullAccess': 'kafka.*' in kafka_perms
    }
```

### Get Available Regions
```python
def get_available_regions(access_details):
    restrictions = access_details['data']['restrictions']
    allowed = restrictions['regionsAllowed']
    restricted = restrictions.get('regionsRestricted', [])
    
    return [r for r in allowed if r not in restricted]
```

## Feature Flag Integration

### Check Feature Availability
```python
def is_feature_enabled(access_details, feature_name):
    flags = access_details['data']['featureFlags']
    return flags.get(feature_name, False)

# Usage
access = get_user_access_details()
if is_feature_enabled(access, 'autoScaling'):
    # Show auto-scaling UI
    pass
```

### Build Feature Matrix
```python
def build_feature_matrix(access_details):
    flags = access_details['data']['featureFlags']
    
    features = {
        'Beta Features': flags.get('betaFeatures', False),
        'Advanced Monitoring': flags.get('advancedMonitoring', False),
        'Auto Scaling': flags.get('autoScaling', False),
        'Multi-Region': flags.get('multiRegion', False),
        'AI Insights': flags.get('aiInsights', False)
    }
    
    enabled_features = [k for k, v in features.items() if v]
    return enabled_features
```

## Quota Management

### Check Quota Availability
```python
def can_create_resource(access_details, resource_type, engagement):
    quotas = access_details['data']['quotas']
    max_allowed = quotas.get(f'max{resource_type.capitalize()}', 0)
    
    current_usage = engagement['resources'].get(resource_type, 0)
    
    return current_usage < max_allowed, (max_allowed - current_usage)

# Usage
access = get_user_access_details()
engagement = access['data']['engagements'][0]

can_create, remaining = can_create_resource(access, 'kafkaClusters', engagement)
print(f"Can create Kafka cluster: {can_create}")
print(f"Remaining quota: {remaining}")
```

### Calculate Quota Utilization
```python
def calculate_quota_utilization(access_details):
    quotas = access_details['data']['quotas']
    
    total_usage = {
        'clusters': 0,
        'kafkaClusters': 0,
        'databases': 0
    }
    
    for engagement in access_details['data']['engagements']:
        resources = engagement.get('resources', {})
        total_usage['clusters'] += resources.get('clusters', 0)
        total_usage['kafkaClusters'] += resources.get('kafkaClusters', 0)
        total_usage['databases'] += resources.get('databases', 0)
    
    utilization = {
        'clusters': (total_usage['clusters'] / quotas['maxClusters']) * 100,
        'kafkaClusters': (total_usage['kafkaClusters'] / quotas['maxKafkaClusters']) * 100,
        'databases': (total_usage['databases'] / quotas['maxDatabases']) * 100
    }
    
    return utilization
```

## UI/UX Integration

### Build Navigation Menu
```python
def build_navigation_menu(access_details):
    services = access_details['data']['services']
    
    menu_items = []
    
    if services.get('paas', {}).get('enabled'):
        menu_items.append({
            'label': 'Clusters',
            'route': '/clusters',
            'icon': 'cluster'
        })
    
    if services.get('kafka', {}).get('enabled'):
        menu_items.append({
            'label': 'Kafka',
            'route': '/kafka',
            'icon': 'kafka'
        })
    
    if services.get('postgres', {}).get('enabled'):
        menu_items.append({
            'label': 'Databases',
            'route': '/databases',
            'icon': 'database'
        })
    
    return menu_items
```

### Show/Hide UI Elements
```javascript
// React component
function ConditionalFeature({ featureName, children }) {
  const { featureFlags } = useAccessDetails();
  
  if (!featureFlags[featureName]) {
    return null;
  }
  
  return <>{children}</>;
}

// Usage
<ConditionalFeature featureName="autoScaling">
  <AutoScalingControls />
</ConditionalFeature>
```

## Security Considerations

### MFA Enforcement
```python
def requires_mfa(access_details):
    return access_details['data']['restrictions']['mfaRequired']

def enforce_mfa_check(access_details, operation):
    if requires_mfa(access_details) and operation in ['delete', 'critical']:
        return True  # Prompt for MFA
    return False
```

### IP Whitelist Check
```python
def is_ip_allowed(access_details, client_ip):
    whitelist = access_details['data']['restrictions'].get('ipWhitelist', [])
    
    if not whitelist:  # Empty whitelist = all IPs allowed
        return True
    
    return client_ip in whitelist
```

## Rate Limiting

### Check Rate Limit
```python
def check_rate_limit(access_details, request_count, period):
    limits = access_details['data']['restrictions']['apiRateLimits']
    
    if period == 'minute':
        return request_count < limits['perMinute']
    elif period == 'hour':
        return request_count < limits['perHour']
    elif period == 'day':
        return request_count < limits['perDay']
    
    return False
```

## Related Operations
- `user.get_profile` - Get user profile information
- `user.update_preferences` - Update user preferences
- `user.list_roles` - List available roles
- `permission.check` - Check specific permission
- `engagement.get_access` - Get engagement-specific access

## Error Handling
- **401 Unauthorized:** Invalid or expired authentication token
- **403 Forbidden:** Account suspended or access revoked
- **500 Internal Server Error:** Service unavailable

## Response Codes
- `0` - Success
- `1` - User not found
- `2` - Access details unavailable

## Performance Notes
- Response time typically < 300ms
- Not cached (always fresh data)
- Include detailed resource counts
- Permissions calculated in real-time
- Feature flags from centralized config

## Best Practices

### Cache Access Details
```python
from datetime import datetime, timedelta

access_cache = {'data': None, 'expires': None}

def get_access_cached(ttl_minutes=5):
    now = datetime.now()
    
    if (access_cache['data'] is None or 
        access_cache['expires'] is None or
        now > access_cache['expires']):
        
        access_cache['data'] = get_user_access_details()
        access_cache['expires'] = now + timedelta(minutes=ttl_minutes)
    
    return access_cache['data']
```

### Permission Helper Class
```python
class UserAccess:
    def __init__(self, access_details):
        self.data = access_details['data']
    
    def can(self, permission):
        # Check all permission sources
        for service, perms in self.data['permissions'].items():
            if permission in perms or f"{service}.*" in perms:
                return True
        return False
    
    def has_service(self, service_name):
        return self.data['services'].get(service_name, {}).get('enabled', False)
    
    def has_feature(self, feature_name):
        return self.data['featureFlags'].get(feature_name, False)
    
    def within_quota(self, resource_type):
        quotas = self.data['quotas']
        total = sum(
            eng['resources'].get(resource_type, 0)
            for eng in self.data['engagements']
        )
        max_quota = quotas.get(f'max{resource_type.capitalize()}', 0)
        return total < max_quota
```

## Metadata
- **Generated:** 2025-02-13T13:30:00Z
- **Source:** Dynamic API Spec Generator
- **API Version:** v1
- **Base Path:** /catalyst-user-service
- **Critical:** Essential for RBAC and feature control