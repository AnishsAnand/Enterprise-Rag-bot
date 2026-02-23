# API Specification: engagement - list_user_engagements

**Resource:** engagement
**Operation:** list_user_engagements
**Aliases:** engagements, user engagements, my engagements, list engagements, show engagements, get engagements, engagement list, user accounts, customer accounts

## Endpoint
- **Method:** GET
- **URL:** {BASE_URL_PORTAL_SERVICE}/configservice/getuserengagements
- **Auth:** Bearer token (from Keycloak)
- **Description:** Get list of all engagements (customer accounts) accessible to the authenticated user, including engagement details, status, type, and associated permissions

## Required Parameters
None (returns engagements based on authenticated user's access)

## Optional Parameters
- `include_details` - Include detailed engagement information (query parameter, default: true)
- `status` - Filter by status: active, inactive, suspended (query parameter)
- `type` - Filter by engagement type: enterprise, smb, government (query parameter)

## Response Mapping
- `status`: status
- `message`: message
- `responseCode`: responseCode
- `engagements`: data.engagements
- `engagement_id`: data.engagements[*].engagementId
- `engagement_name`: data.engagements[*].engagementName
- `engagement_code`: data.engagements[*].engagementCode
- `engagement_type`: data.engagements[*].type
- `engagement_status`: data.engagements[*].status
- `created_time`: data.engagements[*].createdTime
- `user_role`: data.engagements[*].userRole
- `is_primary`: data.engagements[*].isPrimary
- `department_count`: data.engagements[*].departmentCount
- `vm_count`: data.engagements[*].vmCount
- `resource_quota`: data.engagements[*].resourceQuota
- `billing_account`: data.engagements[*].billingAccount

## Response Example
```json
{
  "status": "success",
  "data": {
    "userId": "user@domain.com",
    "totalEngagements": 3,
    "engagements": [
      {
        "engagementId": "ENG001",
        "engagementName": "Enterprise Cloud Platform",
        "engagementCode": "ECP",
        "type": "enterprise",
        "status": "active",
        "createdTime": 1650925643000,
        "createdBy": "admin@domain.com",
        "isPrimary": true,
        "userRole": "admin",
        "description": "Main enterprise engagement for cloud infrastructure",
        "departmentCount": 5,
        "vmCount": 324,
        "userCount": 45,
        "billingAccount": "BA-ENT-001",
        "costCenter": "CC-CLOUD-001",
        "resourceQuota": {
          "cpuCores": 1000,
          "memoryGB": 4096,
          "storageGB": 102400,
          "networkGbps": 500
        },
        "resourceUsage": {
          "cpuCores": 680,
          "memoryGB": 2800,
          "storageGB": 65000,
          "networkGbps": 320
        },
        "endpoints": [
          {
            "endpointId": "EP001",
            "endpointName": "EP_V2_BL",
            "displayName": "Bengaluru",
            "region": "India",
            "status": "active"
          },
          {
            "endpointId": "EP002",
            "endpointName": "EP_V2_SG_TCX",
            "displayName": "Singapore East",
            "region": "Asia Pacific",
            "status": "active"
          }
        ],
        "features": {
          "kubernetes": true,
          "storage": true,
          "networking": true,
          "database": true,
          "backup": true,
          "monitoring": true
        },
        "complianceStandards": [
          "ISO27001",
          "SOC2",
          "GDPR"
        ]
      },
      {
        "engagementId": "ENG002",
        "engagementName": "Development Environment",
        "engagementCode": "DEV",
        "type": "smb",
        "status": "active",
        "createdTime": 1652925643000,
        "createdBy": "admin@domain.com",
        "isPrimary": false,
        "userRole": "developer",
        "description": "Development and testing environment",
        "departmentCount": 2,
        "vmCount": 45,
        "userCount": 12,
        "billingAccount": "BA-DEV-002",
        "costCenter": "CC-DEV-002",
        "resourceQuota": {
          "cpuCores": 200,
          "memoryGB": 512,
          "storageGB": 10240,
          "networkGbps": 50
        },
        "resourceUsage": {
          "cpuCores": 120,
          "memoryGB": 340,
          "storageGB": 5500,
          "networkGbps": 25
        },
        "endpoints": [
          {
            "endpointId": "EP003",
            "endpointName": "EP_V2_DEL",
            "displayName": "Delhi",
            "region": "India",
            "status": "active"
          }
        ],
        "features": {
          "kubernetes": true,
          "storage": true,
          "networking": true,
          "database": false,
          "backup": false,
          "monitoring": true
        },
        "complianceStandards": []
      },
      {
        "engagementId": "ENG003",
        "engagementName": "Training Lab",
        "engagementCode": "LAB",
        "type": "smb",
        "status": "inactive",
        "createdTime": 1648925643000,
        "createdBy": "training@domain.com",
        "isPrimary": false,
        "userRole": "viewer",
        "description": "Training and demo environment",
        "departmentCount": 1,
        "vmCount": 8,
        "userCount": 5,
        "billingAccount": "BA-LAB-003",
        "costCenter": "CC-TRAIN-003",
        "resourceQuota": {
          "cpuCores": 50,
          "memoryGB": 128,
          "storageGB": 2048,
          "networkGbps": 10
        },
        "resourceUsage": {
          "cpuCores": 0,
          "memoryGB": 0,
          "storageGB": 0,
          "networkGbps": 0
        },
        "endpoints": [
          {
            "endpointId": "EP001",
            "endpointName": "EP_V2_BL",
            "displayName": "Bengaluru",
            "region": "India",
            "status": "active"
          }
        ],
        "features": {
          "kubernetes": false,
          "storage": true,
          "networking": true,
          "database": false,
          "backup": false,
          "monitoring": false
        },
        "complianceStandards": []
      }
    ]
  },
  "message": "success",
  "responseCode": 0
}
```

## Response Fields Details

### Engagement Object Fields
- **engagementId** - Unique identifier for the engagement (string)
- **engagementName** - Full name of the engagement (string)
- **engagementCode** - Short code for the engagement (string)
- **type** - Engagement type: enterprise, smb, government (string)
- **status** - Current status: active, inactive, suspended (string)
- **createdTime** - Unix timestamp when engagement was created (number)
- **createdBy** - Email of user who created the engagement (string)
- **isPrimary** - Whether this is the user's primary engagement (boolean)
- **userRole** - User's role in this engagement: admin, developer, viewer (string)
- **description** - Description of engagement purpose (string)
- **departmentCount** - Number of departments in engagement (number)
- **vmCount** - Total VMs across all departments (number)
- **userCount** - Number of users with access (number)
- **billingAccount** - Associated billing account ID (string)
- **costCenter** - Cost center code for chargeback (string)

### Resource Quota Fields
- **cpuCores** - Maximum CPU cores allocated (number)
- **memoryGB** - Maximum memory in GB (number)
- **storageGB** - Maximum storage in GB (number)
- **networkGbps** - Maximum network bandwidth in Gbps (number)

### Resource Usage Fields
- **cpuCores** - Current CPU cores in use (number)
- **memoryGB** - Current memory in use (number)
- **storageGB** - Current storage in use (number)
- **networkGbps** - Current network bandwidth in use (number)

### Endpoint Fields
- **endpointId** - Endpoint identifier (string)
- **endpointName** - Internal endpoint name (string)
- **displayName** - Human-readable location name (string)
- **region** - Geographic region (string)
- **status** - Endpoint status (string)

### Features
- **kubernetes** - CaaS/K8s service enabled (boolean)
- **storage** - Storage service enabled (boolean)
- **networking** - Network service enabled (boolean)
- **database** - Database service enabled (boolean)
- **backup** - Backup service enabled (boolean)
- **monitoring** - Monitoring service enabled (boolean)

## Permissions
Roles: All authenticated users (returns engagements based on user's access)

## Workflow Steps
### Workflow: list_user_engagements
Get all engagements accessible to user
- Step 1: authenticate_user (user.authenticate)
- Step 2: get_user_roles (user.get_roles)
- Step 3: list_engagements (engagement.list_by_user)
- Step 4: get_engagement_details (engagement.get_details)
- Step 5: get_resource_stats (engagement.get_resource_stats)
- Step 6: format_response (response.format)

## Common Use Cases

1. **List my engagements**: "Show me all my engagements"
2. **Find primary engagement**: "What is my primary engagement?"
3. **Check engagement status**: "Which engagements are active?"
4. **View engagement resources**: "How many VMs in each engagement?"
5. **Check user role**: "What role do I have in each engagement?"
6. **Filter by type**: "Show enterprise engagements only"
7. **Find engagement by name**: "Show engagement 'Enterprise Cloud Platform'"
8. **Check resource usage**: "Show resource usage across engagements"
9. **List available services**: "Which engagements have Kubernetes enabled?"
10. **Get billing info**: "Show billing accounts for my engagements"

## Query Interpretations
- "my engagements" → GET /getuserengagements
- "list accounts" → GET /getuserengagements
- "show customer accounts" → GET /getuserengagements
- "active engagements" → GET /getuserengagements?status=active
- "enterprise accounts" → GET /getuserengagements?type=enterprise
- "primary engagement" → Filter response where isPrimary=true

## Data Processing Examples

### Get Primary Engagement
```python
def get_primary_engagement(engagement_data):
    """Get the user's primary engagement."""
    for engagement in engagement_data['data']['engagements']:
        if engagement.get('isPrimary', False):
            return {
                'id': engagement['engagementId'],
                'name': engagement['engagementName'],
                'code': engagement['engagementCode'],
                'role': engagement['userRole']
            }
    return None
```

### Calculate Total Resources
```python
def calculate_total_resources(engagement_data):
    """Calculate total resources across all engagements."""
    totals = {
        'total_vms': 0,
        'total_users': 0,
        'total_departments': 0,
        'cpu_quota': 0,
        'cpu_used': 0,
        'memory_quota': 0,
        'memory_used': 0,
        'storage_quota': 0,
        'storage_used': 0
    }
    
    for engagement in engagement_data['data']['engagements']:
        totals['total_vms'] += engagement.get('vmCount', 0)
        totals['total_users'] += engagement.get('userCount', 0)
        totals['total_departments'] += engagement.get('departmentCount', 0)
        
        quota = engagement.get('resourceQuota', {})
        usage = engagement.get('resourceUsage', {})
        
        totals['cpu_quota'] += quota.get('cpuCores', 0)
        totals['cpu_used'] += usage.get('cpuCores', 0)
        totals['memory_quota'] += quota.get('memoryGB', 0)
        totals['memory_used'] += usage.get('memoryGB', 0)
        totals['storage_quota'] += quota.get('storageGB', 0)
        totals['storage_used'] += usage.get('storageGB', 0)
    
    # Calculate utilization percentages
    totals['cpu_utilization'] = (totals['cpu_used'] / totals['cpu_quota'] * 100) if totals['cpu_quota'] > 0 else 0
    totals['memory_utilization'] = (totals['memory_used'] / totals['memory_quota'] * 100) if totals['memory_quota'] > 0 else 0
    totals['storage_utilization'] = (totals['storage_used'] / totals['storage_quota'] * 100) if totals['storage_quota'] > 0 else 0
    
    return totals
```

### Filter Engagements by Feature
```python
def filter_by_feature(engagement_data, feature_name):
    """Filter engagements that have a specific feature enabled."""
    matching_engagements = []
    
    for engagement in engagement_data['data']['engagements']:
        features = engagement.get('features', {})
        if features.get(feature_name, False):
            matching_engagements.append({
                'id': engagement['engagementId'],
                'name': engagement['engagementName'],
                'type': engagement['type'],
                'status': engagement['status']
            })
    
    return matching_engagements

# Usage
k8s_engagements = filter_by_feature(data, 'kubernetes')
print(f"Engagements with Kubernetes: {len(k8s_engagements)}")
```

### Generate Engagement Summary
```python
def generate_engagement_summary(engagement_data):
    """Generate a comprehensive summary of user's engagements."""
    engagements = engagement_data['data']['engagements']
    
    summary = {
        'total_engagements': len(engagements),
        'active_engagements': sum(1 for e in engagements if e['status'] == 'active'),
        'inactive_engagements': sum(1 for e in engagements if e['status'] == 'inactive'),
        'by_type': {},
        'by_role': {},
        'total_resources': {
            'vms': sum(e.get('vmCount', 0) for e in engagements),
            'departments': sum(e.get('departmentCount', 0) for e in engagements),
            'users': sum(e.get('userCount', 0) for e in engagements)
        },
        'primary_engagement': None,
        'compliance_summary': {}
    }
    
    # Count by type
    for engagement in engagements:
        eng_type = engagement['type']
        summary['by_type'][eng_type] = summary['by_type'].get(eng_type, 0) + 1
        
        # Count by role
        role = engagement['userRole']
        summary['by_role'][role] = summary['by_role'].get(role, 0) + 1
        
        # Get primary engagement
        if engagement.get('isPrimary'):
            summary['primary_engagement'] = {
                'id': engagement['engagementId'],
                'name': engagement['engagementName']
            }
        
        # Aggregate compliance standards
        for standard in engagement.get('complianceStandards', []):
            summary['compliance_summary'][standard] = \
                summary['compliance_summary'].get(standard, 0) + 1
    
    return summary
```

## Integration Examples

### Python Client
```python
import requests
from typing import Dict, List, Optional

class EngagementClient:
    """Client for IPC Cloud Engagement API."""
    
    def __init__(self, base_url: str, bearer_token: str):
        self.base_url = base_url
        self.headers = {
            'Authorization': f'Bearer {bearer_token}',
            'Content-Type': 'application/json'
        }
    
    def get_user_engagements(
        self, 
        status: Optional[str] = None,
        eng_type: Optional[str] = None,
        include_details: bool = True
    ) -> Dict:
        """Get all engagements for authenticated user."""
        url = f"{self.base_url}/portalservice/configservice/getuserengagements"
        
        params = {'include_details': str(include_details).lower()}
        if status:
            params['status'] = status
        if eng_type:
            params['type'] = eng_type
        
        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        return response.json()
    
    def get_primary_engagement(self) -> Optional[Dict]:
        """Get the user's primary engagement."""
        data = self.get_user_engagements()
        
        for engagement in data['data']['engagements']:
            if engagement.get('isPrimary', False):
                return engagement
        return None
    
    def get_engagement_by_id(self, engagement_id: str) -> Optional[Dict]:
        """Get a specific engagement by ID."""
        data = self.get_user_engagements()
        
        for engagement in data['data']['engagements']:
            if engagement['engagementId'] == engagement_id:
                return engagement
        return None
    
    def get_engagements_by_role(self, role: str) -> List[Dict]:
        """Get all engagements where user has specific role."""
        data = self.get_user_engagements()
        
        return [
            e for e in data['data']['engagements']
            if e['userRole'] == role
        ]
    
    def check_feature_availability(
        self, 
        engagement_id: str, 
        feature: str
    ) -> bool:
        """Check if a feature is enabled for an engagement."""
        engagement = self.get_engagement_by_id(engagement_id)
        if not engagement:
            return False
        
        return engagement.get('features', {}).get(feature, False)
    
    def get_resource_utilization(
        self, 
        engagement_id: Optional[str] = None
    ) -> Dict:
        """Get resource utilization for engagement(s)."""
        data = self.get_user_engagements()
        engagements = data['data']['engagements']
        
        if engagement_id:
            engagements = [e for e in engagements if e['engagementId'] == engagement_id]
        
        utilization = []
        for eng in engagements:
            quota = eng.get('resourceQuota', {})
            usage = eng.get('resourceUsage', {})
            
            utilization.append({
                'engagement_id': eng['engagementId'],
                'engagement_name': eng['engagementName'],
                'cpu_percent': (usage.get('cpuCores', 0) / quota.get('cpuCores', 1)) * 100,
                'memory_percent': (usage.get('memoryGB', 0) / quota.get('memoryGB', 1)) * 100,
                'storage_percent': (usage.get('storageGB', 0) / quota.get('storageGB', 1)) * 100
            })
        
        return utilization

# Usage example
if __name__ == '__main__':
    client = EngagementClient(
        base_url='https://ipcloud.tatacommunications.com',
        bearer_token='your_token_here'
    )
    
    # Get all engagements
    engagements = client.get_user_engagements()
    print(f"Total engagements: {len(engagements['data']['engagements'])}")
    
    # Get primary engagement
    primary = client.get_primary_engagement()
    if primary:
        print(f"Primary: {primary['engagementName']} ({primary['engagementId']})")
    
    # Get admin engagements
    admin_engs = client.get_engagements_by_role('admin')
    print(f"Admin access to {len(admin_engs)} engagements")
    
    # Check Kubernetes availability
    if primary:
        has_k8s = client.check_feature_availability(primary['engagementId'], 'kubernetes')
        print(f"Kubernetes enabled: {has_k8s}")
```

### JavaScript Client
```javascript
const axios = require('axios');

class EngagementClient {
  constructor(baseUrl, bearerToken) {
    this.baseUrl = baseUrl;
    this.axiosInstance = axios.create({
      baseURL: baseUrl,
      headers: {
        'Authorization': `Bearer ${bearerToken}`,
        'Content-Type': 'application/json'
      }
    });
  }

  async getUserEngagements(options = {}) {
    const { status, type, includeDetails = true } = options;
    
    const params = { include_details: includeDetails };
    if (status) params.status = status;
    if (type) params.type = type;

    const response = await this.axiosInstance.get(
      '/portalservice/configservice/getuserengagements',
      { params }
    );
    return response.data;
  }

  async getPrimaryEngagement() {
    const data = await this.getUserEngagements();
    return data.data.engagements.find(e => e.isPrimary);
  }

  async getEngagementSummary() {
    const data = await this.getUserEngagements();
    const engagements = data.data.engagements;

    const summary = {
      total: engagements.length,
      active: engagements.filter(e => e.status === 'active').length,
      inactive: engagements.filter(e => e.status === 'inactive').length,
      byType: {},
      byRole: {},
      totalVMs: engagements.reduce((sum, e) => sum + (e.vmCount || 0), 0),
      totalDepartments: engagements.reduce((sum, e) => sum + (e.departmentCount || 0), 0)
    };

    // Count by type and role
    engagements.forEach(e => {
      summary.byType[e.type] = (summary.byType[e.type] || 0) + 1;
      summary.byRole[e.userRole] = (summary.byRole[e.userRole] || 0) + 1;
    });

    return summary;
  }

  async getResourceUtilization(engagementId = null) {
    const data = await this.getUserEngagements();
    let engagements = data.data.engagements;

    if (engagementId) {
      engagements = engagements.filter(e => e.engagementId === engagementId);
    }

    return engagements.map(eng => ({
      engagementId: eng.engagementId,
      engagementName: eng.engagementName,
      cpuPercent: (eng.resourceUsage.cpuCores / eng.resourceQuota.cpuCores) * 100,
      memoryPercent: (eng.resourceUsage.memoryGB / eng.resourceQuota.memoryGB) * 100,
      storagePercent: (eng.resourceUsage.storageGB / eng.resourceQuota.storageGB) * 100
    }));
  }
}

// Usage
(async () => {
  const client = new EngagementClient(
    'https://ipcloud.tatacommunications.com',
    'your_token_here'
  );

  try {
    // Get all engagements
    const engagements = await client.getUserEngagements();
    console.log(`Total engagements: ${engagements.data.engagements.length}`);

    // Get summary
    const summary = await client.getEngagementSummary();
    console.log('Summary:', summary);

    // Get utilization
    const utilization = await client.getResourceUtilization();
    console.log('Resource Utilization:', utilization);
  } catch (error) {
    console.error('Error:', error.message);
  }
})();
```

## Related Operations
- `engagement.get_details` - Get detailed information about a specific engagement
- `engagement.get_endpoints` - Get endpoints for an engagement
- `engagement.get_departments` - Get departments within an engagement
- `engagement.check_metering` - Check metering status for engagement
- `user.get_profile` - Get user profile information
- `billing.get_account` - Get billing account details
- `resource.get_quota` - Get resource quotas
- `compliance.get_standards` - Get compliance certifications

## Error Handling

### Common Errors
- **401 Unauthorized**: Invalid or expired bearer token
  - **Resolution**: Refresh authentication token from Keycloak
  
- **403 Forbidden**: User account suspended or no engagements accessible
  - **Resolution**: Contact administrator to verify account status
  
- **500 Internal Server Error**: Server error retrieving engagements
  - **Resolution**: Check service status, retry with exponential backoff

### Error Response Example
```json
{
  "status": "error",
  "message": "Failed to retrieve user engagements",
  "responseCode": 500,
  "errorDetails": {
    "code": "ENGAGEMENT_FETCH_ERROR",
    "requestId": "req-12345-67890"
  }
}
```

## Response Codes
- `0` - Success
- `1` - Authentication failed
- `2` - Authorization failed
- `404` - User has no accessible engagements
- `500` - Internal server error

## Performance Notes
- **Response time**: Typically < 1 second for up to 50 engagements
- **Caching**: User engagement data cached for 10 minutes
- **Rate limiting**: 200 requests per minute per user
- **Data size**: Approximately 1-2 KB per engagement

## Usage Notes

### Engagement Types
- **enterprise**: Large organizations with multiple departments and high resource quotas
- **smb**: Small-medium business accounts with moderate resources
- **government**: Government/public sector accounts with compliance requirements

### User Roles
- **admin**: Full administrative access, can create/modify resources
- **developer**: Can create and manage resources, limited admin functions
- **viewer**: Read-only access to engagement resources

### Primary Engagement
- Each user has one primary engagement (isPrimary: true)
- Primary engagement is the default for resource creation
- Can be changed through user preferences

### Resource Quotas
- Hard limits enforced at engagement level
- Exceeding quota prevents new resource creation
- Contact support to increase quotas

## Best Practices

1. **Cache engagement list**: Don't fetch on every request, cache for 10+ minutes
2. **Use primary engagement**: Default to primary engagement for user operations
3. **Check feature availability**: Verify feature enabled before offering to user
4. **Monitor resource usage**: Alert when utilization exceeds 80%
5. **Handle multiple engagements**: Support users switching between engagements
6. **Validate engagement access**: Always verify user has access before operations

## Metadata
- **Generated:** 2025-02-16T10:15:00.000000Z
- **Source:** IPC Cloud Portal Configuration Service
- **API Version:** v2
- **Base Path:** /portalservice/configservice/
- **Documentation:** User engagement and account management
- **Additional Notes:** Returns engagements based on authenticated user's access permissions