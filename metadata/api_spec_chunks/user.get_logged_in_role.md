# API Specification: user - get_logged_in_role

**Resource:** user
**Operation:** get_logged_in_role
**Aliases:** user role, current role, get role, logged in role, my role

## Endpoint
- **Method:** GET
- **URL:** https://ipcloud.tatacommunications.com/user-management/user/getloggedinuserrole
- **Auth:** Bearer token (from Keycloak)
- **Description:** Get the primary role and role hierarchy for the currently authenticated user. Returns the highest privilege role and all associated sub-roles

## Required Parameters
None (uses authentication token to identify user)

## Optional Parameters
- `include_hierarchy` - Include full role hierarchy (true/false, default: true)
- `include_permissions` - Include permission mappings (true/false)

## Response Mapping
- `status`: status
- `message`: message
- `user_id`: data.userId
- `primary_role`: data.primaryRole
- `role_name`: data.primaryRole.name
- `role_level`: data.primaryRole.level
- `all_roles`: data.allRoles
- `role_hierarchy`: data.roleHierarchy

## Response Example
```json
{
  "status": "success",
  "data": {
    "userId": "user-12345",
    "userEmail": "john.doe@company.com",
    "userName": "John Doe",
    "primaryRole": {
      "id": "role-admin-001",
      "name": "admin",
      "displayName": "Administrator",
      "level": 1,
      "priority": 100,
      "description": "Full system administrator with all permissions",
      "category": "system",
      "assignedAt": "2024-01-15T10:00:00Z",
      "assignedBy": "system",
      "expiresAt": null,
      "scope": "global"
    },
    "allRoles": [
      {
        "id": "role-admin-001",
        "name": "admin",
        "displayName": "Administrator",
        "level": 1,
        "priority": 100,
        "category": "system"
      },
      {
        "id": "role-kafka-manager-001",
        "name": "kafka_manager",
        "displayName": "Kafka Manager",
        "level": 2,
        "priority": 80,
        "category": "service",
        "service": "kafka"
      },
      {
        "id": "role-developer-001",
        "name": "developer",
        "displayName": "Developer",
        "level": 3,
        "priority": 60,
        "category": "standard"
      }
    ],
    "roleHierarchy": {
      "admin": {
        "inherits": [],
        "grants": ["developer", "viewer", "kafka_manager"],
        "level": 1
      },
      "kafka_manager": {
        "inherits": ["developer"],
        "grants": ["kafka_viewer"],
        "level": 2
      },
      "developer": {
        "inherits": ["viewer"],
        "grants": [],
        "level": 3
      }
    },
    "effectivePermissions": [
      "system.*",
      "user.*",
      "cluster.*",
      "volume.*",
      "kafka.*",
      "postgres.*",
      "mongodb.*",
      "billing.view"
    ],
    "permissionSources": {
      "admin": [
        "system.*",
        "user.*",
        "cluster.*",
        "volume.*"
      ],
      "kafka_manager": [
        "kafka.*"
      ],
      "developer": [
        "cluster.create",
        "volume.create"
      ]
    },
    "restrictions": {
      "canElevate": false,
      "canDelegate": true,
      "canCreateUsers": true,
      "canModifyRoles": true,
      "requiresMFA": false
    },
    "metadata": {
      "rolePath": "admin > kafka_manager > developer",
      "highestLevel": 1,
      "roleCount": 3,
      "systemRole": true
    }
  },
  "message": "User role retrieved successfully",
  "responseCode": 0
}
```

## Response Fields Details

### Primary Role Fields
- **name** - Role identifier (e.g., "admin", "developer")
- **displayName** - Human-readable role name
- **level** - Hierarchy level (1 = highest)
- **priority** - Priority value (higher = more privileged)
- **description** - Role description
- **category** - Role category (system, service, standard, custom)
- **assignedAt** - When role was assigned
- **assignedBy** - Who assigned the role
- **expiresAt** - Expiration date (null = permanent)
- **scope** - Role scope (global, engagement, resource)

### Role Levels
- `Level 1` - System administrators (admin, super_admin)
- `Level 2` - Service managers (kafka_manager, db_admin)
- `Level 3` - Standard users (developer, operator)
- `Level 4` - Read-only users (viewer, auditor)
- `Level 5` - Limited access (guest, readonly)

### Role Categories
- `system` - System-level roles (admin, super_admin)
- `service` - Service-specific roles (kafka_manager, postgres_admin)
- `standard` - Standard user roles (developer, operator, viewer)
- `custom` - Custom organization roles

### Role Priority
Higher priority = more permissions
- `100` - Super Admin
- `90` - Admin
- `80` - Service Manager
- `60` - Developer
- `40` - Operator
- `20` - Viewer
- `10` - Guest

## Permissions
Roles: All authenticated users (returns their own role)

## Workflow Steps
### Workflow: get_user_role
Get current user's role
- Step 1: authenticate (auth.validate_token)
- Step 2: get_logged_in_role (user.get_logged_in_role)

## Usage Notes
- Returns only current user's roles
- Primary role is highest privilege role
- Role hierarchy shows inheritance
- Effective permissions are aggregated from all roles
- System roles cannot be modified
- Role expiration enforced automatically

## Common Use Cases
1. **Permission check**: "Is user an admin?"
2. **UI customization**: "Show admin menu?"
3. **Access control**: "Can user access admin panel?"
4. **Role display**: "Show user's role badge"
5. **Delegation check**: "Can user assign roles?"
6. **Audit logging**: "Log role with actions"

## Data Processing Examples

### Check If User Is Admin
```python
def is_admin(role_data):
    primary_role = role_data['data']['primaryRole']
    return primary_role['name'] in ['admin', 'super_admin']

# Usage
role = get_logged_in_user_role()
if is_admin(role):
    # Show admin features
    pass
```

### Get Role Level
```python
def get_user_level(role_data):
    return role_data['data']['primaryRole']['level']

def has_min_level(role_data, required_level):
    user_level = get_user_level(role_data)
    return user_level <= required_level  # Lower number = higher privilege

# Usage
role = get_logged_in_user_role()
if has_min_level(role, 2):
    # User is level 1 or 2 (admin or manager)
    pass
```

### Check Service-Specific Role
```python
def has_service_role(role_data, service):
    all_roles = role_data['data']['allRoles']
    
    for role in all_roles:
        if role.get('service') == service:
            return True
    
    return False

# Usage
role = get_logged_in_user_role()
can_manage_kafka = has_service_role(role, 'kafka')
```

### Get Effective Permissions
```python
def get_permissions_for_service(role_data, service):
    permissions = role_data['data']['effectivePermissions']
    
    # Filter permissions for specific service
    service_perms = [
        p for p in permissions
        if p.startswith(f"{service}.") or p == f"{service}.*"
    ]
    
    return service_perms

# Usage
role = get_logged_in_user_role()
kafka_permissions = get_permissions_for_service(role, 'kafka')
```

## Role Hierarchy Analysis

### Get Role Path
```python
def get_role_path(role_data):
    metadata = role_data['data']['metadata']
    return metadata['rolePath']

# Usage
role = get_logged_in_user_role()
path = get_role_path(role)
print(f"Role path: {path}")  # "admin > kafka_manager > developer"
```

### Check Role Inheritance
```python
def inherits_from(role_data, role_name):
    hierarchy = role_data['data']['roleHierarchy']
    primary = role_data['data']['primaryRole']['name']
    
    def check_inheritance(current_role):
        if current_role == role_name:
            return True
        
        if current_role in hierarchy:
            for inherited in hierarchy[current_role].get('inherits', []):
                if check_inheritance(inherited):
                    return True
        
        return False
    
    return check_inheritance(primary)

# Usage
role = get_logged_in_user_role()
has_viewer_perms = inherits_from(role, 'viewer')
```

## UI Integration

### Display Role Badge
```javascript
function RoleBadge({ roleData }) {
  const role = roleData.data.primaryRole;
  
  const colorMap = {
    'admin': 'red',
    'kafka_manager': 'blue',
    'developer': 'green',
    'viewer': 'gray'
  };
  
  return (
    <span className={`badge badge-${colorMap[role.name] || 'gray'}`}>
      {role.displayName}
    </span>
  );
}
```

### Conditional Rendering
```javascript
function AdminPanel() {
  const { primaryRole } = useUserRole();
  
  if (primaryRole.level > 1) {
    return <AccessDenied />;
  }
  
  return <AdminDashboard />;
}
```

### Role-Based Menu
```python
def build_menu_for_role(role_data):
    role_level = role_data['data']['primaryRole']['level']
    
    menu_items = [
        {'label': 'Dashboard', 'route': '/', 'min_level': 5}
    ]
    
    if role_level <= 3:  # Developer or higher
        menu_items.append({'label': 'Resources', 'route': '/resources'})
    
    if role_level <= 2:  # Manager or higher
        menu_items.append({'label': 'Services', 'route': '/services'})
    
    if role_level <= 1:  # Admin only
        menu_items.append({'label': 'Admin', 'route': '/admin'})
    
    return menu_items
```

## Permission Aggregation

### Aggregate From All Roles
```python
def aggregate_permissions(role_data):
    perm_sources = role_data['data']['permissionSources']
    
    all_permissions = set()
    for role, perms in perm_sources.items():
        all_permissions.update(perms)
    
    return list(all_permissions)
```

### Check Wildcard Permissions
```python
def has_wildcard_permission(role_data, resource):
    effective_perms = role_data['data']['effectivePermissions']
    
    # Check for exact match or wildcard
    return (f"{resource}.*" in effective_perms or 
            "system.*" in effective_perms)

# Usage
role = get_logged_in_user_role()
full_kafka_access = has_wildcard_permission(role, 'kafka')
```

## Role Restrictions

### Check Delegation Rights
```python
def can_delegate_role(role_data, target_role_name):
    restrictions = role_data['data']['restrictions']
    
    if not restrictions['canDelegate']:
        return False
    
    # Can only delegate roles at same or lower level
    primary_level = role_data['data']['primaryRole']['level']
    
    # Get target role level from hierarchy
    all_roles = role_data['data']['allRoles']
    target_role = next((r for r in all_roles if r['name'] == target_role_name), None)
    
    if not target_role:
        return False
    
    return target_role['level'] >= primary_level
```

### Check Elevation Rights
```python
def can_elevate_privileges(role_data):
    return role_data['data']['restrictions']['canElevate']
```

## Audit Logging

### Log with Role Context
```python
def log_action_with_role(action, role_data):
    primary_role = role_data['data']['primaryRole']
    user_id = role_data['data']['userId']
    
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'userId': user_id,
        'userEmail': role_data['data']['userEmail'],
        'role': primary_role['name'],
        'roleLevel': primary_role['level'],
        'action': action,
        'allRoles': [r['name'] for r in role_data['data']['allRoles']]
    }
    
    # Write to audit log
    audit_logger.info(log_entry)
```

## Role Comparison

### Compare Two Users' Roles
```python
def compare_role_levels(role_data_1, role_data_2):
    level_1 = role_data_1['data']['primaryRole']['level']
    level_2 = role_data_2['data']['primaryRole']['level']
    
    if level_1 < level_2:
        return "user_1_higher"
    elif level_1 > level_2:
        return "user_2_higher"
    else:
        return "equal"
```

## Related Operations
- `user.get_access_details` - Get comprehensive access information
- `user.list_roles` - List all available roles
- `role.get` - Get role details
- `role.assign` - Assign role to user
- `permission.check` - Check specific permission

## Error Handling
- **401 Unauthorized:** Invalid or expired authentication token
- **403 Forbidden:** Account suspended or role revoked
- **404 Not Found:** User has no assigned roles
- **500 Internal Server Error:** Service unavailable

## Response Codes
- `0` - Success
- `1` - User not found
- `2` - No role assigned
- `3` - Role data unavailable

## Performance Notes
- Response time typically < 200ms
- Role data cached for 10 minutes
- Hierarchy calculated on-demand
- Permission aggregation optimized

## Best Practices

### Cache Role Data
```python
from datetime import datetime, timedelta

role_cache = {'data': None, 'expires': None}

def get_role_cached(ttl_minutes=10):
    now = datetime.now()
    
    if (role_cache['data'] is None or 
        role_cache['expires'] is None or
        now > role_cache['expires']):
        
        role_cache['data'] = get_logged_in_user_role()
        role_cache['expires'] = now + timedelta(minutes=ttl_minutes)
    
    return role_cache['data']
```

### Role Helper Class
```python
class UserRole:
    def __init__(self, role_data):
        self.data = role_data['data']
        self.primary = self.data['primaryRole']
    
    def is_admin(self):
        return self.primary['name'] in ['admin', 'super_admin']
    
    def has_level(self, min_level):
        return self.primary['level'] <= min_level
    
    def can_manage(self, service):
        return any(
            r['name'] == f'{service}_manager' or r['name'] == 'admin'
            for r in self.data['allRoles']
        )
    
    def get_display_name(self):
        return self.primary['displayName']
```

## Metadata
- **Generated:** 2025-02-13T13:30:00Z
- **Source:** Dynamic API Spec Generator
- **API Version:** v1
- **Base Path:** /user-management
- **Critical:** Essential for RBAC and access control