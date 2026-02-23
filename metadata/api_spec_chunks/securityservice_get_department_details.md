# API Specification: securityservice - get_department_details

**Resource:** securityservice
**Operation:** get_department_details
**Aliases:** get departments, department details, list departments, show departments, departments for engagement

## Endpoint
- **Method:** GET
- **URL:** `{BASE_URL_PORTAL_SERVICE}/securityservice/deptDetailsForEngagement/{engagement_id}`
- **Auth:** Bearer token (assumed - needs confirmation)
- **Description:** Retrieves department details and organizational structure for a specific engagement. Returns department hierarchy, team information, and access control details.

## Required Parameters
- `engagement_id` - IPC engagement ID (NOT PaaS ID - must convert first) (type: path parameter, format: string)

## Optional Parameters
None

## Response Mapping
**⚠️ Note: Actual API response needed to complete this section accurately**

Assumed mappings (to be verified):
- `departments`: data.departments
- `department_ids`: data.departments[*].id
- `department_names`: data.departments[*].name
- `department_codes`: data.departments[*].code
- `team_count`: data.departments[*].team_count
- `user_count`: data.departments[*].user_count
- `engagement_id`: data.engagement_id
- `status`: status
- `message`: message

## Response Example
**⚠️ Placeholder - Replace with actual API response**

```json
{
  "status": "success",
  "data": {
    "engagement_id": "ipc-67890",
    "departments": [
      {
        "id": "dept-001",
        "name": "Engineering",
        "code": "ENG",
        "description": "Engineering and Development",
        "parent_department_id": null,
        "level": 1,
        "team_count": 5,
        "user_count": 45,
        "status": "active",
        "created_at": "2024-01-15T10:00:00Z"
      },
      {
        "id": "dept-002",
        "name": "Platform Engineering",
        "code": "PLAT-ENG",
        "description": "Platform and Infrastructure",
        "parent_department_id": "dept-001",
        "level": 2,
        "team_count": 2,
        "user_count": 15,
        "status": "active",
        "created_at": "2024-01-15T10:00:00Z"
      },
      {
        "id": "dept-003",
        "name": "Operations",
        "code": "OPS",
        "description": "Operations and Support",
        "parent_department_id": null,
        "level": 1,
        "team_count": 3,
        "user_count": 20,
        "status": "active",
        "created_at": "2024-01-20T14:30:00Z"
      }
    ],
    "total_departments": 3,
    "total_users": 80
  }
}
```

## Response Fields Details

### Core Fields
- **status** - Response status (success, error)
- **engagement_id** - IPC engagement ID
- **departments** - Array of department objects
- **total_departments** - Total number of departments
- **total_users** - Total users across all departments

### Department Object Fields
- **id** - Unique department identifier
- **name** - Department name
- **code** - Department code (abbreviation)
- **description** - Department description
- **parent_department_id** - Parent department ID (null for top-level departments)
- **level** - Hierarchy level (1 = top level, 2+ = sub-departments)
- **team_count** - Number of teams in department
- **user_count** - Number of users in department
- **status** - Department status (active, inactive)
- **created_at** - Department creation timestamp

## Permissions
**Roles:** To be documented (requires actual permission information)
Likely requires: Admin, Department Manager, Security Officer roles

## Workflow Steps

### Workflow: Access Department Information
**Prerequisites:**
- Step 1: **Convert Engagement ID** (engagement.convert_paas_to_ipc) - Convert PaaS ID to IPC ID first!

**Main Workflow:**
- Step 2: **Get Department Details** (securityservice.get_department_details) - Retrieve department structure

### Example Workflow
```
1. convert_paas_to_ipc("paas-12345") → "ipc-67890"
2. get_department_details("ipc-67890") → Department hierarchy
3. Use department info for access control, reporting, etc.
```

## Usage Notes
- **CRITICAL:** This endpoint requires IPC engagement ID, NOT PaaS ID
- Always call `getIpcEngFromPaasEng` first if you only have a PaaS ID
- Departments may have hierarchical structure (parent-child relationships)
- Use for access control, user management, reporting, and organizational queries
- Cache department data as it doesn't change frequently

## Common Use Cases
1. **List all departments**: "show departments", "get all departments", "list departments for engagement"
2. **Department hierarchy**: "show department structure", "get org chart", "department tree"
3. **User count by department**: "how many users in engineering", "department sizes"
4. **Access control**: "which departments exist", "check department access", "list active departments"
5. **Reporting**: "department breakdown", "organizational structure"

## Query Interpretations
- "get departments for {id}" → GET /deptDetailsForEngagement/{id}
- "list departments {id}" → GET /deptDetailsForEngagement/{id}
- "show org structure {id}" → GET /deptDetailsForEngagement/{id}
- "department details {id}" → GET /deptDetailsForEngagement/{id}

## Data Processing Examples

### Python Example
```python
import requests
from typing import List, Dict, Optional

def get_department_details(engagement_id: str, auth_token: str) -> Optional[Dict]:
    """
    Get department details for an engagement.
    
    Args:
        engagement_id: IPC engagement ID (not PaaS ID!)
        auth_token: Bearer authentication token
        
    Returns:
        dict: Department data or None if failed
    """
    url = f"https://ipcloud.tatacommunications.com/portalservice/securityservice/deptDetailsForEngagement/{engagement_id}"
    
    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        if data.get('status') == 'success':
            dept_data = data['data']
            print(f"Found {dept_data['total_departments']} departments with {dept_data['total_users']} total users")
            return dept_data
        else:
            print(f"Failed: {data.get('message')}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"API error: {e}")
        return None

def build_department_hierarchy(departments: List[Dict]) -> Dict:
    """Build hierarchical department tree"""
    dept_map = {dept['id']: dept for dept in departments}
    tree = {}
    
    for dept in departments:
        dept['children'] = []
        parent_id = dept.get('parent_department_id')
        
        if parent_id is None:
            # Top-level department
            tree[dept['id']] = dept
        else:
            # Add as child to parent
            if parent_id in dept_map:
                dept_map[parent_id]['children'].append(dept)
    
    return tree

def get_department_by_name(departments: List[Dict], name: str) -> Optional[Dict]:
    """Find department by name (case-insensitive)"""
    name_lower = name.lower()
    for dept in departments:
        if dept['name'].lower() == name_lower:
            return dept
    return None

# Usage
ipc_id = "ipc-67890"  # Must be IPC ID, not PaaS ID!
token = "your-token-here"

# Get department details
dept_data = get_department_details(ipc_id, token)

if dept_data:
    # Build hierarchy
    hierarchy = build_department_hierarchy(dept_data['departments'])
    
    # Find specific department
    eng_dept = get_department_by_name(dept_data['departments'], "Engineering")
    if eng_dept:
        print(f"Engineering has {eng_dept['user_count']} users in {eng_dept['team_count']} teams")
    
    # List top-level departments
    top_level = [d for d in dept_data['departments'] if d['parent_department_id'] is None]
    print(f"Top-level departments: {[d['name'] for d in top_level]}")
```

### JavaScript Example
```javascript
async function getDepartmentDetails(engagementId, authToken) {
    const url = `https://ipcloud.tatacommunications.com/portalservice/securityservice/deptDetailsForEngagement/${engagementId}`;
    
    try {
        const response = await fetch(url, {
            method: 'GET',
            headers: {
                'Authorization': `Bearer ${authToken}`,
                'Content-Type': 'application/json'
            }
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        if (data.status === 'success') {
            const deptData = data.data;
            console.log(`Found ${deptData.total_departments} departments with ${deptData.total_users} total users`);
            return deptData;
        } else {
            console.error(`Failed: ${data.message}`);
            return null;
        }
    } catch (error) {
        console.error(`API error: ${error.message}`);
        return null;
    }
}

function buildDepartmentHierarchy(departments) {
    const deptMap = new Map(departments.map(d => [d.id, {...d, children: []}]));
    const tree = {};
    
    departments.forEach(dept => {
        const deptNode = deptMap.get(dept.id);
        const parentId = dept.parent_department_id;
        
        if (parentId === null) {
            tree[dept.id] = deptNode;
        } else if (deptMap.has(parentId)) {
            deptMap.get(parentId).children.push(deptNode);
        }
    });
    
    return tree;
}

function getDepartmentByCode(departments, code) {
    return departments.find(d => d.code.toUpperCase() === code.toUpperCase());
}

// Usage
const ipcId = 'ipc-67890';  // Must be IPC ID!
const token = 'your-token-here';

const deptData = await getDepartmentDetails(ipcId, token);

if (deptData) {
    // Build hierarchy
    const hierarchy = buildDepartmentHierarchy(deptData.departments);
    
    // Find by code
    const engDept = getDepartmentByCode(deptData.departments, 'ENG');
    console.log(`Engineering: ${engDept.user_count} users`);
    
    // List all active departments
    const active = deptData.departments.filter(d => d.status === 'active');
    console.log(`Active departments: ${active.map(d => d.name).join(', ')}`);
}
```

## Integration Examples

### Complete Department Analysis
```python
class DepartmentAnalyzer:
    def __init__(self, dept_data):
        self.departments = dept_data['departments']
        self.dept_map = {d['id']: d for d in self.departments}
    
    def get_top_level_departments(self):
        """Get all top-level departments"""
        return [d for d in self.departments if d['parent_department_id'] is None]
    
    def get_sub_departments(self, parent_id):
        """Get all sub-departments of a parent"""
        return [d for d in self.departments if d['parent_department_id'] == parent_id]
    
    def get_department_path(self, dept_id):
        """Get full path from root to department"""
        path = []
        current = self.dept_map.get(dept_id)
        
        while current:
            path.insert(0, current['name'])
            parent_id = current.get('parent_department_id')
            current = self.dept_map.get(parent_id) if parent_id else None
        
        return ' > '.join(path)
    
    def get_largest_departments(self, n=5):
        """Get n largest departments by user count"""
        sorted_depts = sorted(self.departments, key=lambda d: d['user_count'], reverse=True)
        return sorted_depts[:n]
    
    def calculate_total_users_recursive(self, dept_id):
        """Calculate total users including all sub-departments"""
        dept = self.dept_map.get(dept_id)
        if not dept:
            return 0
        
        total = dept['user_count']
        sub_depts = self.get_sub_departments(dept_id)
        
        for sub in sub_depts:
            total += self.calculate_total_users_recursive(sub['id'])
        
        return total

# Usage
dept_data = get_department_details(ipc_id, token)
analyzer = DepartmentAnalyzer(dept_data)

# Analysis
top_depts = analyzer.get_top_level_departments()
print(f"Top-level departments: {[d['name'] for d in top_depts]}")

largest = analyzer.get_largest_departments(3)
for dept in largest:
    path = analyzer.get_department_path(dept['id'])
    total = analyzer.calculate_total_users_recursive(dept['id'])
    print(f"{path}: {dept['user_count']} direct users, {total} total users")
```

### Department-Based Access Control
```python
class DepartmentAccessControl:
    def __init__(self, client):
        self.client = client
        self.dept_cache = {}
    
    def get_departments(self, engagement_id):
        """Get departments with caching"""
        if engagement_id in self.dept_cache:
            return self.dept_cache[engagement_id]
        
        # Convert if needed
        ipc_id = self.client.convert_to_ipc(engagement_id)
        
        # Fetch departments
        dept_data = get_department_details(ipc_id, self.client.auth_token)
        self.dept_cache[engagement_id] = dept_data
        return dept_data
    
    def check_department_access(self, engagement_id, department_code):
        """Check if department exists and is active"""
        dept_data = self.get_departments(engagement_id)
        
        for dept in dept_data['departments']:
            if dept['code'] == department_code:
                return dept['status'] == 'active'
        
        return False
    
    def list_accessible_departments(self, engagement_id, user_role='user'):
        """List departments accessible by role"""
        dept_data = self.get_departments(engagement_id)
        
        # Filter based on role (customize logic)
        if user_role == 'admin':
            return dept_data['departments']
        else:
            # Regular users see active departments only
            return [d for d in dept_data['departments'] if d['status'] == 'active']

# Usage
acl = DepartmentAccessControl(client)

# Check access
has_access = acl.check_department_access("paas-12345", "ENG")
print(f"Has Engineering access: {has_access}")

# List accessible
accessible = acl.list_accessible_departments("paas-12345", "user")
print(f"Accessible departments: {[d['name'] for d in accessible]}")
```

## Related Operations
- `engagement.convert_paas_to_ipc` - PREREQUISITE: Convert PaaS ID to IPC ID
- `configservice.get_endpoints_by_engagement` - Get service endpoints for engagement
- User management APIs - Manage users within departments
- Access control APIs - Control department-level permissions

## Error Handling
- **400 Bad Request:** Invalid engagement ID format - verify ID structure
- **401 Unauthorized:** Invalid or expired authentication token - refresh token
- **403 Forbidden:** Insufficient permissions to view department details
- **404 Not Found:** Engagement not found OR you're using PaaS ID instead of IPC ID
  - **Solution:** Always convert PaaS ID to IPC ID first using `getIpcEngFromPaasEng`
- **500 Internal Server Error:** Service error - retry with exponential backoff
- **503 Service Unavailable:** Service temporarily unavailable - implement retry logic

## Response Codes
**⚠️ To be documented - requires actual API testing**

Likely patterns:
- `0` or `success` - Request successful
- Non-zero or `error` - Request failed (check message field)

## Performance Notes
- Response time typically < 500ms
- Department structure changes infrequently - safe to cache for extended periods
- No pagination (returns all departments for engagement)
- Rate limiting: To be documented

## Best Practices

### Always Convert PaaS ID First
```python
# ❌ WRONG - Using PaaS ID directly
try:
    depts = get_department_details("paas-12345", token)
    # This will FAIL with 404!
except Exception as e:
    print(f"Failed: {e}")

# ✅ RIGHT - Convert PaaS to IPC first
paas_id = "paas-12345"
ipc_id = convert_paas_to_ipc(paas_id, token)
depts = get_department_details(ipc_id, token)  # Works!
```

### Cache Department Data
```python
import time

class DepartmentCache:
    def __init__(self, ttl=7200):  # 2 hours
        self.cache = {}
        self.ttl = ttl
    
    def get(self, engagement_id):
        if engagement_id in self.cache:
            entry = self.cache[engagement_id]
            if time.time() - entry['timestamp'] < self.ttl:
                return entry['data']
        return None
    
    def set(self, engagement_id, data):
        self.cache[engagement_id] = {
            'data': data,
            'timestamp': time.time()
        }

# Usage - departments don't change often
cache = DepartmentCache(ttl=7200)
depts = cache.get(ipc_id)
if not depts:
    depts = get_department_details(ipc_id, token)
    cache.set(ipc_id, depts)
```

### Build Efficient Lookup Maps
```python
def create_department_lookups(departments):
    """Create efficient lookup structures"""
    return {
        'by_id': {d['id']: d for d in departments},
        'by_name': {d['name'].lower(): d for d in departments},
        'by_code': {d['code'].upper(): d for d in departments},
        'by_level': {}  # Group by hierarchy level
    }

lookups = create_department_lookups(dept_data['departments'])
eng_dept = lookups['by_code'].get('ENG')
```

## Metadata
- **Generated:** 2024-02-23T10:00:00Z
- **Source:** API endpoint analysis
- **API Version:** To be documented
- **Base Path:** {BASE_URL_PORTAL_SERVICE}/securityservice
- **Additional notes:** ⚠️ This documentation is preliminary. Please provide actual API response for complete accuracy. CRITICAL: This endpoint requires IPC engagement ID - always convert PaaS ID first!
