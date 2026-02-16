# API Specification: user - get_actions_by_service

**Resource:** user
**Operation:** get_actions_by_service
**Aliases:** service actions, user actions, get actions, permitted actions, service permissions

## Endpoint
- **Method:** GET
- **URL:** https://ipcloud.tatacommunications.com/catalyst-user-service/user/getactionsbyservicename
- **Auth:** Bearer token (from Keycloak)
- **Description:** Get all permitted actions/operations for the authenticated user within a specific service. Returns granular action-level permissions for UI/UX control

## Required Parameters
- `servicename` - Service identifier (query parameter)
  - Examples: `paas`, `kafka`, `postgres`, `mongodb`, `gitlab`

## Optional Parameters
- `resource_type` - Filter by resource type (cluster, volume, topic, database)
- `include_descriptions` - Include action descriptions (true/false)

## Response Mapping
- `status`: status
- `message`: message
- `service_name`: data.serviceName
- `user_id`: data.userId
- `actions`: data.actions
- `action_names`: data.actions[*].name
- `action_categories`: data.actions[*].category
- `total_actions`: data.totalActions

## Response Example (Service: kafka)
```json
{
  "status": "success",
  "data": {
    "serviceName": "kafka",
    "userId": "user-12345",
    "userEmail": "john.doe@company.com",
    "totalActions": 25,
    "actions": [
      {
        "name": "cluster.create",
        "displayName": "Create Kafka Cluster",
        "description": "Create new Kafka cluster",
        "category": "cluster",
        "resource": "kafka_cluster",
        "operation": "create",
        "requiresApproval": false,
        "requiresMFA": false,
        "riskLevel": "medium",
        "uiElements": [
          "create_cluster_button",
          "cluster_creation_form"
        ]
      },
      {
        "name": "cluster.delete",
        "displayName": "Delete Kafka Cluster",
        "description": "Delete existing Kafka cluster",
        "category": "cluster",
        "resource": "kafka_cluster",
        "operation": "delete",
        "requiresApproval": true,
        "requiresMFA": true,
        "riskLevel": "high",
        "uiElements": [
          "delete_cluster_button",
          "cluster_delete_confirmation"
        ]
      },
      {
        "name": "cluster.view",
        "displayName": "View Kafka Cluster",
        "description": "View Kafka cluster details",
        "category": "cluster",
        "resource": "kafka_cluster",
        "operation": "view",
        "requiresApproval": false,
        "requiresMFA": false,
        "riskLevel": "low",
        "uiElements": [
          "cluster_list",
          "cluster_details_page"
        ]
      },
      {
        "name": "cluster.update",
        "displayName": "Update Kafka Cluster",
        "description": "Modify Kafka cluster configuration",
        "category": "cluster",
        "resource": "kafka_cluster",
        "operation": "update",
        "requiresApproval": false,
        "requiresMFA": false,
        "riskLevel": "medium",
        "uiElements": [
          "edit_cluster_button",
          "cluster_settings_form"
        ]
      },
      {
        "name": "topic.create",
        "displayName": "Create Kafka Topic",
        "description": "Create new topic in Kafka cluster",
        "category": "topic",
        "resource": "kafka_topic",
        "operation": "create",
        "requiresApproval": false,
        "requiresMFA": false,
        "riskLevel": "low",
        "uiElements": [
          "create_topic_button",
          "topic_creation_form"
        ]
      },
      {
        "name": "topic.delete",
        "displayName": "Delete Kafka Topic",
        "description": "Delete existing Kafka topic",
        "category": "topic",
        "resource": "kafka_topic",
        "operation": "delete",
        "requiresApproval": true,
        "requiresMFA": false,
        "riskLevel": "high",
        "uiElements": [
          "delete_topic_button"
        ]
      },
      {
        "name": "topic.view",
        "displayName": "View Kafka Topics",
        "description": "View topic list and details",
        "category": "topic",
        "resource": "kafka_topic",
        "operation": "view",
        "requiresApproval": false,
        "requiresMFA": false,
        "riskLevel": "low",
        "uiElements": [
          "topic_list",
          "topic_details_page"
        ]
      },
      {
        "name": "topic.update",
        "displayName": "Update Kafka Topic",
        "description": "Modify topic configuration",
        "category": "topic",
        "resource": "kafka_topic",
        "operation": "update",
        "requiresApproval": false,
        "requiresMFA": false,
        "riskLevel": "medium",
        "uiElements": [
          "edit_topic_button",
          "topic_settings_form"
        ]
      },
      {
        "name": "consumer.create",
        "displayName": "Create Consumer Group",
        "description": "Create new consumer group",
        "category": "consumer",
        "resource": "kafka_consumer_group",
        "operation": "create",
        "requiresApproval": false,
        "requiresMFA": false,
        "riskLevel": "low",
        "uiElements": [
          "create_consumer_button"
        ]
      },
      {
        "name": "consumer.view",
        "displayName": "View Consumer Groups",
        "description": "View consumer group details and lag",
        "category": "consumer",
        "resource": "kafka_consumer_group",
        "operation": "view",
        "requiresApproval": false,
        "requiresMFA": false,
        "riskLevel": "low",
        "uiElements": [
          "consumer_group_list",
          "consumer_lag_dashboard"
        ]
      },
      {
        "name": "metrics.view",
        "displayName": "View Kafka Metrics",
        "description": "View performance metrics and monitoring",
        "category": "monitoring",
        "resource": "kafka_metrics",
        "operation": "view",
        "requiresApproval": false,
        "requiresMFA": false,
        "riskLevel": "low",
        "uiElements": [
          "metrics_dashboard",
          "performance_charts"
        ]
      },
      {
        "name": "config.update",
        "displayName": "Update Kafka Configuration",
        "description": "Modify cluster-level configuration",
        "category": "configuration",
        "resource": "kafka_config",
        "operation": "update",
        "requiresApproval": true,
        "requiresMFA": true,
        "riskLevel": "high",
        "uiElements": [
          "config_editor"
        ]
      }
    ],
    "actionsByCategory": {
      "cluster": 4,
      "topic": 4,
      "consumer": 2,
      "monitoring": 1,
      "configuration": 1
    },
    "actionsByRiskLevel": {
      "low": 7,
      "medium": 3,
      "high": 2
    },
    "approvalRequired": 3,
    "mfaRequired": 2
  },
  "message": "Actions retrieved successfully",
  "responseCode": 0
}
```

## Response Example (Service: postgres)
```json
{
  "status": "success",
  "data": {
    "serviceName": "postgres",
    "userId": "user-12345",
    "totalActions": 15,
    "actions": [
      {
        "name": "database.create",
        "displayName": "Create Database",
        "category": "database",
        "resource": "postgres_database",
        "operation": "create",
        "requiresApproval": false,
        "riskLevel": "medium"
      },
      {
        "name": "database.delete",
        "displayName": "Delete Database",
        "category": "database",
        "resource": "postgres_database",
        "operation": "delete",
        "requiresApproval": true,
        "requiresMFA": true,
        "riskLevel": "high"
      },
      {
        "name": "user.create",
        "displayName": "Create Database User",
        "category": "user",
        "resource": "postgres_user",
        "operation": "create",
        "requiresApproval": false,
        "riskLevel": "medium"
      },
      {
        "name": "backup.restore",
        "displayName": "Restore from Backup",
        "category": "backup",
        "resource": "postgres_backup",
        "operation": "restore",
        "requiresApproval": true,
        "requiresMFA": true,
        "riskLevel": "high"
      }
    ]
  }
}
```

## Response Fields Details

### Action Fields
- **name** - Action identifier (format: `resource.operation`)
- **displayName** - Human-readable action name
- **description** - Detailed action description
- **category** - Action category/grouping
- **resource** - Resource type this action applies to
- **operation** - Operation type (create, view, update, delete)
- **requiresApproval** - Whether action needs approval
- **requiresMFA** - Whether MFA is required
- **riskLevel** - Risk level (low, medium, high)
- **uiElements** - UI elements this action enables

### Action Categories
- `cluster` - Cluster management actions
- `topic` - Topic management (Kafka)
- `consumer` - Consumer group management (Kafka)
- `database` - Database operations (Postgres/MongoDB)
- `user` - User management
- `backup` - Backup/restore operations
- `monitoring` - Monitoring and metrics
- `configuration` - Configuration management

### Risk Levels
- `low` - Read operations, minimal risk
- `medium` - Create/update operations
- `high` - Delete operations, destructive changes

### Operation Types
- `create` - Create new resource
- `view` - Read/view resource
- `update` - Modify existing resource
- `delete` - Delete resource
- `manage` - Full management (CRUD)

## Permissions
Roles: All authenticated users (returns their permitted actions)

## Workflow Steps
### Workflow: get_service_actions
Get user actions for service
- Step 1: authenticate (auth.validate_token)
- Step 2: get_user_role (user.get_role)
- Step 3: get_actions_by_service (user.get_actions_by_service) (depends on: servicename)

## Usage Notes
- Returns only actions user has permission for
- Actions derived from user's roles
- UI elements mapped to actions for display control
- Approval workflows indicated per action
- MFA requirements specified per action
- Risk levels help with audit logging

## Common Use Cases
1. **UI control**: "Show/hide create button"
2. **Form validation**: "Can user delete resource?"
3. **Feature gating**: "Enable advanced features?"
4. **Approval workflow**: "Does action need approval?"
5. **MFA prompt**: "Require MFA for this action?"
6. **Audit logging**: "Log action risk level"

## Data Processing Examples

### Check Specific Action
```python
def can_perform_action(actions_data, action_name):
    actions = actions_data['data']['actions']
    return any(a['name'] == action_name for a in actions)

# Usage
kafka_actions = get_actions_by_service('kafka')
can_delete_cluster = can_perform_action(kafka_actions, 'cluster.delete')
```

### Get Actions by Category
```python
def get_actions_by_category(actions_data, category):
    actions = actions_data['data']['actions']
    return [a for a in actions if a['category'] == category]

# Usage
kafka_actions = get_actions_by_service('kafka')
topic_actions = get_actions_by_category(kafka_actions, 'topic')
```

### Filter High-Risk Actions
```python
def get_high_risk_actions(actions_data):
    actions = actions_data['data']['actions']
    return [
        a for a in actions
        if a['riskLevel'] == 'high'
    ]

# Usage
kafka_actions = get_actions_by_service('kafka')
risky_actions = get_high_risk_actions(kafka_actions)
for action in risky_actions:
    print(f"⚠️ {action['displayName']} - Requires approval: {action['requiresApproval']}")
```

### Get Actions Requiring Approval
```python
def get_approval_required_actions(actions_data):
    actions = actions_data['data']['actions']
    return [a for a in actions if a.get('requiresApproval', False)]
```

## UI Integration

### Button Visibility Control
```javascript
function CreateClusterButton() {
  const { actions } = useServiceActions('kafka');
  
  const canCreate = actions.some(a => a.name === 'cluster.create');
  
  if (!canCreate) {
    return null;
  }
  
  return <button onClick={createCluster}>Create Cluster</button>;
}
```

### Action Menu Generation
```python
def build_action_menu(actions_data, resource_type):
    actions = actions_data['data']['actions']
    
    menu_items = []
    for action in actions:
        if action['resource'] == resource_type:
            menu_items.append({
                'label': action['displayName'],
                'action': action['name'],
                'icon': get_icon_for_operation(action['operation']),
                'requiresConfirmation': action['riskLevel'] == 'high'
            })
    
    return menu_items
```

### Form Field Control
```javascript
function ClusterForm() {
  const { actions } = useServiceActions('kafka');
  
  const canUpdate = actions.some(a => a.name === 'cluster.update');
  const canDelete = actions.some(a => a.name === 'cluster.delete');
  
  return (
    <form>
      <input disabled={!canUpdate} />
      {canDelete && <button>Delete</button>}
    </form>
  );
}
```

## MFA and Approval Handling

### Check MFA Requirement
```python
def requires_mfa(actions_data, action_name):
    actions = actions_data['data']['actions']
    action = next((a for a in actions if a['name'] == action_name), None)
    
    return action and action.get('requiresMFA', False)

# Usage
kafka_actions = get_actions_by_service('kafka')
if requires_mfa(kafka_actions, 'cluster.delete'):
    # Prompt for MFA
    pass
```

### Check Approval Requirement
```python
def needs_approval(actions_data, action_name):
    actions = actions_data['data']['actions']
    action = next((a for a in actions if a['name'] == action_name), None)
    
    return action and action.get('requiresApproval', False)
```

## Risk Level Analysis

### Categorize Actions by Risk
```python
def categorize_by_risk(actions_data):
    actions = actions_data['data']['actions']
    
    by_risk = {
        'low': [],
        'medium': [],
        'high': []
    }
    
    for action in actions:
        risk = action['riskLevel']
        by_risk[risk].append(action['name'])
    
    return by_risk

# Usage
kafka_actions = get_actions_by_service('kafka')
risk_categories = categorize_by_risk(kafka_actions)

print(f"Low risk: {len(risk_categories['low'])} actions")
print(f"Medium risk: {len(risk_categories['medium'])} actions")
print(f"High risk: {len(risk_categories['high'])} actions")
```

## Service Comparison

### Compare Actions Across Services
```python
def compare_services(service1, service2):
    actions1 = get_actions_by_service(service1)
    actions2 = get_actions_by_service(service2)
    
    count1 = actions1['data']['totalActions']
    count2 = actions2['data']['totalActions']
    
    return {
        service1: count1,
        service2: count2,
        'difference': abs(count1 - count2)
    }

# Usage
comparison = compare_services('kafka', 'postgres')
print(f"Kafka: {comparison['kafka']} actions")
print(f"Postgres: {comparison['postgres']} actions")
```

## Kafka-Specific Actions

### Get Kafka CRUD Matrix
```python
def get_kafka_crud_matrix(kafka_actions):
    actions = kafka_actions['data']['actions']
    
    resources = ['cluster', 'topic', 'consumer']
    operations = ['create', 'view', 'update', 'delete']
    
    matrix = {}
    for resource in resources:
        matrix[resource] = {}
        for operation in operations:
            action_name = f"{resource}.{operation}"
            matrix[resource][operation] = any(
                a['name'] == action_name for a in actions
            )
    
    return matrix

# Usage
kafka_actions = get_actions_by_service('kafka')
crud = get_kafka_crud_matrix(kafka_actions)

print("Kafka CRUD Permissions:")
for resource, ops in crud.items():
    print(f"\n{resource.capitalize()}:")
    for op, allowed in ops.items():
        status = "✓" if allowed else "✗"
        print(f"  {status} {op}")
```

## Related Operations
- `user.get_access_details` - Get comprehensive access info
- `user.get_logged_in_role` - Get user's role
- `permission.check` - Check specific permission
- `service.list_actions` - List all available actions for service

## Error Handling
- **400 Bad Request:** Invalid service name
- **401 Unauthorized:** Invalid or expired authentication token
- **403 Forbidden:** User has no access to service
- **404 Not Found:** Service not found
- **500 Internal Server Error:** Service unavailable

## Response Codes
- `0` - Success
- `1` - Service not found
- `2` - User has no permissions for service
- `3` - Invalid service name

## Supported Services

### Available Services
- `paas` - PaaS cluster management
- `kafka` - Kafka messaging
- `postgres` - PostgreSQL database
- `mongodb` - MongoDB/DocumentDB
- `mysql` - MySQL database
- `redis` - Redis cache
- `gitlab` - GitLab CI/CD
- `elasticsearch` - Elasticsearch search

## Performance Notes
- Response time typically < 300ms
- Actions cached per user per service
- Cache TTL: 10 minutes
- Approval/MFA rules evaluated in real-time

## Best Practices

### Cache Actions Per Service
```python
service_action_cache = {}

def get_actions_cached(service_name, ttl_minutes=10):
    from datetime import datetime, timedelta
    
    now = datetime.now()
    cache_key = f"{service_name}_actions"
    
    if cache_key in service_action_cache:
        data, expires = service_action_cache[cache_key]
        if now < expires:
            return data
    
    data = get_actions_by_service(service_name)
    service_action_cache[cache_key] = (data, now + timedelta(minutes=ttl_minutes))
    
    return data
```

### Action Permission Helper
```python
class ServiceActions:
    def __init__(self, service_name):
        self.data = get_actions_by_service(service_name)['data']
        self.actions = {a['name']: a for a in self.data['actions']}
    
    def can(self, action_name):
        return action_name in self.actions
    
    def needs_approval(self, action_name):
        return self.actions.get(action_name, {}).get('requiresApproval', False)
    
    def needs_mfa(self, action_name):
        return self.actions.get(action_name, {}).get('requiresMFA', False)
    
    def get_risk(self, action_name):
        return self.actions.get(action_name, {}).get('riskLevel', 'unknown')

# Usage
kafka = ServiceActions('kafka')
if kafka.can('cluster.delete'):
    if kafka.needs_mfa('cluster.delete'):
        prompt_mfa()
```

## Metadata
- **Generated:** 2025-02-13T13:30:00Z
- **Source:** Dynamic API Spec Generator
- **API Version:** v1
- **Base Path:** /catalyst-user-service
- **Critical:** Essential for UI/UX permission control