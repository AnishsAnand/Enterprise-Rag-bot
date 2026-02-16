# API Specification: department - list

**Resource:** department
**Operation:** list
**Aliases:** departments, business units, list departments, get departments, org units, bu

## Endpoint
- **Method:** GET
- **URL:** {BASE_URL_PORTAL_SERVICE}/securityservice/departments/{id}
- **Auth:** Bearer token (from Keycloak)
- **Description:** List all departments (business units) within an engagement, including their zones, environments, VM counts, and access permissions. Essential for organizational hierarchy and resource allocation

## Required Parameters
- `id` - IPC engagement identifier (path parameter, NOT paas_engagement_id - convert using get_ipc_from_paas first)

## Optional Parameters
- `include_inactive` - Include inactive departments (true/false)
- `with_counts` - Include detailed resource counts (true/false, default: true)

## Response Mapping
- `status`: status
- `message`: message
- `engagement_id`: data.engagement
- `departments`: data.department
- `department_ids`: data.department[*].id
- `department_names`: data.department[*].name
- `department_codes`: data.department[*].code
- `zones`: data.department[*].zones
- `environments`: data.department[*].environments
- `vm_counts`: data.department[*].vmCount
- `total_vms`: data.totalVMs

## Response Example
```json
{
  "status": "success",
  "data": {
    "engagement": "ipc-67890",
    "engagementName": "Production Environment",
    "totalDepartments": 5,
    "totalVMs": 245,
    "department": [
      {
        "id": "dept-001",
        "name": "Engineering",
        "code": "ENG",
        "description": "Engineering Department",
        "active": true,
        "createdAt": "2024-01-15T10:30:00Z",
        "zones": [
          {
            "id": "zone-mumbai-bkc",
            "name": "Mumbai BKC",
            "location": "EP_V2_MUM_BKC",
            "vmCount": 45,
            "active": true
          },
          {
            "id": "zone-chennai-amb",
            "name": "Chennai AMB",
            "location": "EP_V2_CHN_AMB",
            "vmCount": 38,
            "active": true
          }
        ],
        "environments": [
          {
            "id": "env-prod",
            "name": "Production",
            "type": "production",
            "vmCount": 50,
            "clusterCount": 8,
            "kafkaCount": 2
          },
          {
            "id": "env-staging",
            "name": "Staging",
            "type": "staging",
            "vmCount": 25,
            "clusterCount": 4,
            "kafkaCount": 1
          },
          {
            "id": "env-dev",
            "name": "Development",
            "type": "development",
            "vmCount": 8,
            "clusterCount": 2,
            "kafkaCount": 0
          }
        ],
        "vmCount": 83,
        "clusterCount": 14,
        "volumeCount": 45,
        "kafkaCount": 3,
        "users": [
          {
            "userId": "user-123",
            "email": "john.doe@company.com",
            "role": "admin"
          },
          {
            "userId": "user-456",
            "email": "jane.smith@company.com",
            "role": "developer"
          }
        ],
        "permissions": [
          "vm.create",
          "cluster.create",
          "volume.create",
          "kafka.manage"
        ],
        "costCenter": "CC-ENG-001",
        "budget": {
          "monthly": 50000,
          "currency": "USD",
          "spent": 32500,
          "remaining": 17500
        }
      },
      {
        "id": "dept-002",
        "name": "Data Science",
        "code": "DS",
        "description": "Data Science & Analytics",
        "active": true,
        "createdAt": "2024-02-01T09:15:00Z",
        "zones": [
          {
            "id": "zone-chennai-amb",
            "name": "Chennai AMB",
            "location": "EP_V2_CHN_AMB",
            "vmCount": 62,
            "active": true
          }
        ],
        "environments": [
          {
            "id": "env-prod",
            "name": "Production",
            "type": "production",
            "vmCount": 45,
            "clusterCount": 5,
            "kafkaCount": 3
          },
          {
            "id": "env-dev",
            "name": "Development",
            "type": "development",
            "vmCount": 17,
            "clusterCount": 3,
            "kafkaCount": 1
          }
        ],
        "vmCount": 62,
        "clusterCount": 8,
        "volumeCount": 120,
        "kafkaCount": 4,
        "users": [
          {
            "userId": "user-789",
            "email": "data.scientist@company.com",
            "role": "admin"
          }
        ],
        "permissions": [
          "vm.create",
          "cluster.create",
          "volume.create",
          "kafka.manage",
          "gpu.access"
        ],
        "costCenter": "CC-DS-001",
        "budget": {
          "monthly": 75000,
          "currency": "USD",
          "spent": 68000,
          "remaining": 7000
        }
      },
      {
        "id": "dept-003",
        "name": "Platform Services",
        "code": "PLAT",
        "description": "Platform & Infrastructure",
        "active": true,
        "createdAt": "2023-12-10T14:45:00Z",
        "zones": [
          {
            "id": "zone-mumbai-bkc",
            "name": "Mumbai BKC",
            "location": "EP_V2_MUM_BKC",
            "vmCount": 55,
            "active": true
          },
          {
            "id": "zone-delhi",
            "name": "Delhi",
            "location": "EP_V2_DEL",
            "vmCount": 28,
            "active": true
          }
        ],
        "environments": [
          {
            "id": "env-prod",
            "name": "Production",
            "type": "production",
            "vmCount": 70,
            "clusterCount": 12,
            "kafkaCount": 5
          },
          {
            "id": "env-staging",
            "name": "Staging",
            "type": "staging",
            "vmCount": 13,
            "clusterCount": 3,
            "kafkaCount": 1
          }
        ],
        "vmCount": 83,
        "clusterCount": 15,
        "volumeCount": 90,
        "kafkaCount": 6,
        "users": [
          {
            "userId": "user-321",
            "email": "platform.admin@company.com",
            "role": "admin"
          }
        ],
        "permissions": [
          "vm.create",
          "vm.delete",
          "cluster.create",
          "cluster.delete",
          "volume.create",
          "kafka.manage",
          "network.manage"
        ],
        "costCenter": "CC-PLAT-001",
        "budget": {
          "monthly": 100000,
          "currency": "USD",
          "spent": 85000,
          "remaining": 15000
        }
      }
    ]
  },
  "message": "Departments retrieved successfully",
  "responseCode": 0
}
```

## Response Fields Details

### Department Fields
- **id** - Unique department identifier
- **name** - Department name
- **code** - Short department code
- **description** - Department description
- **active** - Whether department is active
- **createdAt** - When department was created
- **zones** - Array of zones department has access to
- **environments** - Array of environments (prod, staging, dev)
- **vmCount** - Total VMs in this department
- **clusterCount** - Total clusters
- **volumeCount** - Total volumes
- **kafkaCount** - Total Kafka clusters
- **users** - Users with access to this department
- **permissions** - Department-level permissions
- **costCenter** - Cost center code
- **budget** - Budget information

### Zone Fields
- **id** - Zone identifier
- **name** - Zone display name
- **location** - Technical location code
- **vmCount** - VMs in this zone for this department
- **active** - Whether zone is active

### Environment Fields
- **id** - Environment identifier
- **name** - Environment name
- **type** - Environment type (production, staging, development)
- **vmCount** - VMs in this environment
- **clusterCount** - Clusters in this environment
- **kafkaCount** - Kafka clusters in this environment

## Permissions
Roles: admin, developer, viewer (for departments they have access to)

## Workflow Steps
### Workflow: list_departments
List departments for engagement
- Step 1: authenticate (auth.validate_token)
- Step 2: get_paas_engagement (engagement.get)
- Step 3: convert_to_ipc_engagement (engagement.get_ipc_from_paas) (depends on: engagement_id)
- Step 4: list_departments (department.list) (depends on: ipc_engagement_id)

## Usage Notes
- **CRITICAL:** This endpoint requires IPC engagement ID, not PaaS ID
- Must convert PaaS engagement to IPC engagement first
- Returns only departments user has access to
- Resource counts are real-time
- Inactive departments excluded by default
- Budget information may be restricted based on role

## Common Use Cases
1. **Organization view**: "Show me all departments"
2. **Resource allocation**: "How many VMs per department?"
3. **Kafka distribution**: "Which departments have Kafka clusters?"
4. **Cost tracking**: "Show budget usage by department"
5. **Access control**: "List users by department"
6. **Capacity planning**: "Which departments need more resources?"

## Data Processing Examples

### Get Department by Name
```python
response = list_departments(ipc_engagement_id)
departments = response['data']['department']

eng_dept = next(
    (d for d in departments if d['name'] == 'Engineering'),
    None
)

if eng_dept:
    print(f"Engineering: {eng_dept['vmCount']} VMs, {eng_dept['kafkaCount']} Kafka clusters")
```

### Calculate Total Resources
```python
response = list_departments(ipc_engagement_id)
departments = response['data']['department']

total_vms = sum(d['vmCount'] for d in departments)
total_clusters = sum(d['clusterCount'] for d in departments)
total_kafka = sum(d['kafkaCount'] for d in departments)

print(f"Total: {total_vms} VMs, {total_clusters} clusters, {total_kafka} Kafka")
```

### Find Departments with Kafka
```python
response = list_departments(ipc_engagement_id)

kafka_depts = [
    d for d in response['data']['department']
    if d['kafkaCount'] > 0
]

for dept in kafka_depts:
    print(f"{dept['name']}: {dept['kafkaCount']} Kafka clusters")
    print(f"  Zones: {', '.join(z['name'] for z in dept['zones'])}")
```

### Group by Zone
```python
from collections import defaultdict

response = list_departments(ipc_engagement_id)
by_zone = defaultdict(lambda: {'departments': [], 'vmCount': 0})

for dept in response['data']['department']:
    for zone in dept['zones']:
        by_zone[zone['name']]['departments'].append(dept['name'])
        by_zone[zone['name']]['vmCount'] += zone['vmCount']

for zone_name, info in by_zone.items():
    print(f"{zone_name}: {info['vmCount']} VMs across {len(info['departments'])} departments")
```

## Kafka Integration

### Kafka Department Overview
```python
response = list_departments(ipc_engagement_id)

print("Kafka Distribution by Department:")
print("-" * 70)

for dept in response['data']['department']:
    if dept['kafkaCount'] > 0:
        print(f"\n{dept['name']} ({dept['code']}):")
        print(f"  Kafka Clusters: {dept['kafkaCount']}")
        
        # By environment
        for env in dept['environments']:
            if env['kafkaCount'] > 0:
                print(f"    {env['name']}: {env['kafkaCount']} clusters")
        
        # By zone
        print(f"  Zones: {', '.join(z['name'] for z in dept['zones'])}")
        
        # Budget
        if 'budget' in dept:
            print(f"  Budget: ${dept['budget']['spent']:,} / ${dept['budget']['monthly']:,}")
```

### Find Kafka Access by Department
```python
def get_kafka_permissions_by_dept(ipc_engagement_id):
    response = list_departments(ipc_engagement_id)
    
    kafka_access = {}
    
    for dept in response['data']['department']:
        dept_name = dept['name']
        has_kafka = dept['kafkaCount'] > 0
        can_manage = 'kafka.manage' in dept.get('permissions', [])
        
        kafka_access[dept_name] = {
            'hasKafka': has_kafka,
            'canManage': can_manage,
            'clusterCount': dept['kafkaCount'],
            'users': len(dept.get('users', []))
        }
    
    return kafka_access
```

## Related Operations
- `engagement.get_ipc_from_paas` - Convert PaaS to IPC engagement (REQUIRED FIRST)
- `department.get` - Get single department details
- `department.create` - Create new department
- `department.update` - Update department
- `zone.list` - List all zones
- `environment.list` - List all environments

## Error Handling
- **400 Bad Request:** Invalid engagement ID format
- **401 Unauthorized:** Invalid or expired authentication token
- **403 Forbidden:** User does not have access to this engagement
- **404 Not Found:** Engagement ID not found (often means need IPC ID, not PaaS ID)
- **500 Internal Server Error:** Service unavailable

## Response Codes
- `0` - Success
- `1` - Engagement not found
- `2` - No departments found
- `3` - Access denied

## Critical Workflow

### MUST Convert PaaS to IPC First
```python
# WRONG - Using PaaS ID directly
departments = list_departments("paas-eng-12345")  # ❌ Will fail with 404

# RIGHT - Convert first
ipc_id = get_ipc_from_paas("paas-eng-12345")
departments = list_departments(ipc_id)  # ✅ Works
```

### Complete Flow Example
```python
async def list_departments_for_paas_engagement(paas_engagement_id):
    # Step 1: Convert to IPC ID
    ipc_response = await get_ipc_from_paas(paas_engagement_id)
    ipc_id = ipc_response['data']['ipcEngagementId']
    
    # Step 2: List departments
    dept_response = await list_departments(ipc_id)
    
    return dept_response['data']['department']
```

## Budget Tracking

### Calculate Budget Utilization
```python
response = list_departments(ipc_engagement_id)

for dept in response['data']['department']:
    if 'budget' in dept:
        budget = dept['budget']
        utilization = (budget['spent'] / budget['monthly']) * 100
        
        status = "🟢" if utilization < 70 else "🟡" if utilization < 90 else "🔴"
        
        print(f"{status} {dept['name']}: {utilization:.1f}% budget used")
        print(f"   ${budget['spent']:,} / ${budget['monthly']:,}")
```

### Find Over-Budget Departments
```python
response = list_departments(ipc_engagement_id)

over_budget = [
    d for d in response['data']['department']
    if 'budget' in d and d['budget']['spent'] > d['budget']['monthly']
]

if over_budget:
    print("⚠️ Over-budget departments:")
    for dept in over_budget:
        overage = dept['budget']['spent'] - dept['budget']['monthly']
        print(f"  {dept['name']}: ${overage:,} over budget")
```

## Performance Notes
- Response time typically < 800ms
- Large engagements (20+ departments) may take longer
- Resource counts are real-time (not cached)
- Consider caching at application level
- Budget data updated hourly

## Best Practices

### Cache Department Data
```python
from datetime import datetime, timedelta

dept_cache = {}

def get_departments_cached(ipc_id, ttl_minutes=10):
    now = datetime.now()
    
    if ipc_id in dept_cache:
        cached_data, expires = dept_cache[ipc_id]
        if now < expires:
            return cached_data
    
    data = list_departments(ipc_id)
    dept_cache[ipc_id] = (data, now + timedelta(minutes=ttl_minutes))
    
    return data
```

### Access Control Validation
```python
def validate_department_access(ipc_id, department_id):
    response = list_departments(ipc_id)
    
    dept = next(
        (d for d in response['data']['department'] if d['id'] == department_id),
        None
    )
    
    if not dept:
        raise AccessDeniedError(f"No access to department {department_id}")
    
    return dept
```

## Metadata
- **Generated:** 2025-02-13T12:30:00Z
- **Source:** Dynamic API Spec Generator
- **API Version:** v1
- **Base Path:** /portalservice/securityservice
- **CRITICAL:** Requires IPC engagement ID (convert from PaaS first)
- **Kafka Support:** Full department-level Kafka resource tracking