# API Specification: firewall - get_vdom_limit

**Resource:** firewall
**Operation:** get_vdom_limit
**Aliases:** vdom limit, firewall vdom, virtual domain limit, vdom quota, firewall virtual domains, vdom capacity, fortinet vdom

## Endpoint
- **Method:** GET
- **URL:** {BASE_URL_NETWORK_SERVICE}/firewallconfig/engagement/{engagement_id}/vdomLimit
- **Auth:** Bearer token (from Keycloak)
- **Description:** Get Virtual Domain (VDOM) limits and usage for firewall configuration in an engagement, shows maximum allowed VDOMs, current usage, and available capacity for FortiGate firewall virtual domains

## Required Parameters
- `engagement_id` - The unique identifier of the engagement (string, path parameter)

## Optional Parameters
None

## Response Mapping
- `status`: status
- `message`: message
- `responseCode`: responseCode
- `engagement_id`: data.engagementId
- `vdom_limit`: data.vdomLimit
- `vdom_used`: data.vdomUsed
- `vdom_available`: data.vdomAvailable
- `vdom_utilization_percent`: data.vdomUtilizationPercent
- `firewall_model`: data.firewallModel
- `license_type`: data.licenseType
- `vdom_list`: data.vdoms[*]
- `vdom_name`: data.vdoms[*].vdomName
- `vdom_status`: data.vdoms[*].status
- `can_create_vdom`: data.canCreateVdom

## Response Example
```json
{
  "status": "success",
  "data": {
    "engagementId": "ENG001",
    "engagementName": "Enterprise Cloud Platform",
    "firewallModel": "FortiGate-VM",
    "firewallVersion": "7.2.4",
    "licenseType": "enterprise",
    "vdomLimit": 10,
    "vdomUsed": 6,
    "vdomAvailable": 4,
    "vdomUtilizationPercent": 60.0,
    "canCreateVdom": true,
    "hardLimit": 10,
    "softLimit": 8,
    "alertThreshold": 80,
    "vdoms": [
      {
        "vdomId": "VDOM001",
        "vdomName": "root",
        "vdomType": "system",
        "status": "active",
        "createdTime": 1650925643000,
        "description": "Root administrative VDOM",
        "isPrimary": true,
        "interfaceCount": 4,
        "policyCount": 25,
        "vpnTunnelCount": 3,
        "resourceUsage": {
          "sessions": 1250,
          "policies": 25,
          "users": 45,
          "vpns": 3
        }
      },
      {
        "vdomId": "VDOM002",
        "vdomName": "production",
        "vdomType": "customer",
        "status": "active",
        "createdTime": 1652925643000,
        "description": "Production environment VDOM",
        "isPrimary": false,
        "interfaceCount": 6,
        "policyCount": 45,
        "vpnTunnelCount": 5,
        "resourceUsage": {
          "sessions": 3500,
          "policies": 45,
          "users": 120,
          "vpns": 5
        },
        "departments": ["ENG", "OPS"],
        "zones": ["PROD-WEB", "PROD-DB", "PROD-APP"]
      },
      {
        "vdomId": "VDOM003",
        "vdomName": "staging",
        "vdomType": "customer",
        "status": "active",
        "createdTime": 1654925643000,
        "description": "Staging environment VDOM",
        "isPrimary": false,
        "interfaceCount": 4,
        "policyCount": 30,
        "vpnTunnelCount": 2,
        "resourceUsage": {
          "sessions": 800,
          "policies": 30,
          "users": 25,
          "vpns": 2
        },
        "departments": ["ENG"],
        "zones": ["STAGE-WEB", "STAGE-DB"]
      },
      {
        "vdomId": "VDOM004",
        "vdomName": "development",
        "vdomType": "customer",
        "status": "active",
        "createdTime": 1656925643000,
        "description": "Development environment VDOM",
        "isPrimary": false,
        "interfaceCount": 3,
        "policyCount": 20,
        "vpnTunnelCount": 1,
        "resourceUsage": {
          "sessions": 450,
          "policies": 20,
          "users": 15,
          "vpns": 1
        },
        "departments": ["ENG"],
        "zones": ["DEV"]
      },
      {
        "vdomId": "VDOM005",
        "vdomName": "dmz",
        "vdomType": "customer",
        "status": "active",
        "createdTime": 1658925643000,
        "description": "DMZ VDOM for public-facing services",
        "isPrimary": false,
        "interfaceCount": 3,
        "policyCount": 35,
        "vpnTunnelCount": 0,
        "resourceUsage": {
          "sessions": 2100,
          "policies": 35,
          "users": 5,
          "vpns": 0
        },
        "departments": ["OPS"],
        "zones": ["DMZ"]
      },
      {
        "vdomId": "VDOM006",
        "vdomName": "management",
        "vdomType": "system",
        "status": "active",
        "createdTime": 1660925643000,
        "description": "Management VDOM",
        "isPrimary": false,
        "interfaceCount": 2,
        "policyCount": 15,
        "vpnTunnelCount": 2,
        "resourceUsage": {
          "sessions": 120,
          "policies": 15,
          "users": 10,
          "vpns": 2
        },
        "departments": ["OPS"],
        "zones": ["MGMT"]
      }
    ],
    "vdomByType": {
      "system": 2,
      "customer": 4
    },
    "vdomByStatus": {
      "active": 6,
      "inactive": 0,
      "pending": 0
    },
    "recommendations": [
      "VDOM usage at 60% - within normal range",
      "Consider creating separate VDOMs for test and QA environments",
      "4 VDOMs still available for new environments"
    ],
    "quotaInfo": {
      "canIncrease": true,
      "maxPossible": 50,
      "increaseRequiresLicense": true,
      "contactSupport": "support@tatacommunications.com"
    }
  },
  "message": "success",
  "responseCode": 0
}
```

## Response Fields Details

### VDOM Limit Fields
- **engagementId** - Engagement identifier (string)
- **engagementName** - Engagement name (string)
- **firewallModel** - FortiGate model (string)
- **firewallVersion** - Firewall software version (string)
- **licenseType** - License type: enterprise, standard, trial (string)
- **vdomLimit** - Maximum VDOMs allowed (number)
- **vdomUsed** - Currently used VDOMs (number)
- **vdomAvailable** - Available VDOM slots (number)
- **vdomUtilizationPercent** - Usage percentage (number)
- **canCreateVdom** - Whether new VDOMs can be created (boolean)
- **hardLimit** - Absolute maximum VDOMs (number)
- **softLimit** - Recommended maximum (number)
- **alertThreshold** - Alert when usage exceeds this percentage (number)

### VDOM Object Fields
- **vdomId** - Unique VDOM identifier (string)
- **vdomName** - Name of the VDOM (string)
- **vdomType** - Type: system, customer (string)
- **status** - Status: active, inactive, pending (string)
- **createdTime** - Unix timestamp when created (number)
- **description** - Purpose description (string)
- **isPrimary** - Whether this is primary VDOM (boolean)
- **interfaceCount** - Number of interfaces (number)
- **policyCount** - Number of firewall policies (number)
- **vpnTunnelCount** - Number of VPN tunnels (number)
- **departments** - Associated departments (array)
- **zones** - Network zones in this VDOM (array)

### Resource Usage
- **sessions** - Active sessions count (number)
- **policies** - Number of firewall policies (number)
- **users** - Number of users (number)
- **vpns** - Number of VPN connections (number)

### Quota Information
- **canIncrease** - Whether limit can be increased (boolean)
- **maxPossible** - Maximum possible VDOMs (number)
- **increaseRequiresLicense** - License upgrade needed (boolean)
- **contactSupport** - Support contact for limit increase (string)

## Permissions
Roles: admin, developer (viewer has read-only access)

## Workflow Steps
### Workflow: get_vdom_limit
Get VDOM limits for engagement
- Step 1: validate_engagement_id (engagement.validate) (depends on: engagement_id)
- Step 2: get_firewall_config (firewall.get_config) (depends on: engagement_id)
- Step 3: get_vdom_limit (firewall.get_vdom_limit) (depends on: engagement_id)
- Step 4: list_vdoms (firewall.list_vdoms) (depends on: engagement_id)
- Step 5: calculate_usage (vdom.calculate_usage)
- Step 6: format_response (response.format)

## Common Use Cases

1. **Check VDOM capacity**: "How many VDOMs can I create?"
2. **View current usage**: "How many VDOMs are being used?"
3. **List all VDOMs**: "Show me all virtual domains"
4. **Check if can create**: "Can I create a new VDOM?"
5. **View VDOM details**: "Show details of production VDOM"
6. **Check utilization**: "What's the VDOM utilization percentage?"
7. **Find available slots**: "How many VDOM slots are available?"
8. **Request limit increase**: "How do I increase VDOM limit?"
9. **View VDOM by type**: "How many system vs customer VDOMs?"
10. **Plan new VDOMs**: "Do I have capacity for 3 more VDOMs?"

## Query Interpretations
- "vdom limit" → GET /engagement/{id}/vdomLimit
- "vdom capacity" → GET /engagement/{id}/vdomLimit
- "virtual domains" → GET /engagement/{id}/vdomLimit
- "can create vdom" → Extract data.canCreateVdom
- "vdom available" → Extract data.vdomAvailable

## Data Processing Examples

### Check If Can Create VDOM
```python
def can_create_vdom(vdom_data, count=1):
    """Check if specified number of VDOMs can be created."""
    available = vdom_data['data']['vdomAvailable']
    can_create = vdom_data['data']['canCreateVdom']
    
    return {
        'can_create': can_create and available >= count,
        'available': available,
        'requested': count,
        'message': f"{'Can' if can_create and available >= count else 'Cannot'} create {count} VDOM(s)"
    }
```

### Get VDOM by Name
```python
def get_vdom_by_name(vdom_data, vdom_name):
    """Get details of a specific VDOM by name."""
    for vdom in vdom_data['data']['vdoms']:
        if vdom['vdomName'].lower() == vdom_name.lower():
            return vdom
    return None
```

### Calculate VDOM Resource Summary
```python
def calculate_vdom_summary(vdom_data):
    """Calculate summary statistics across all VDOMs."""
    vdoms = vdom_data['data']['vdoms']
    
    summary = {
        'total_vdoms': len(vdoms),
        'active_vdoms': sum(1 for v in vdoms if v['status'] == 'active'),
        'total_interfaces': sum(v.get('interfaceCount', 0) for v in vdoms),
        'total_policies': sum(v.get('policyCount', 0) for v in vdoms),
        'total_vpn_tunnels': sum(v.get('vpnTunnelCount', 0) for v in vdoms),
        'total_sessions': sum(
            v.get('resourceUsage', {}).get('sessions', 0) for v in vdoms
        ),
        'vdoms_by_type': {},
        'vdoms_by_department': {}
    }
    
    # Count by type
    for vdom in vdoms:
        vdom_type = vdom['vdomType']
        summary['vdoms_by_type'][vdom_type] = \
            summary['vdoms_by_type'].get(vdom_type, 0) + 1
        
        # Count by department
        for dept in vdom.get('departments', []):
            summary['vdoms_by_department'][dept] = \
                summary['vdoms_by_department'].get(dept, 0) + 1
    
    return summary
```

### Check Alert Thresholds
```python
def check_vdom_alerts(vdom_data):
    """Check if VDOM usage exceeds alert thresholds."""
    utilization = vdom_data['data']['vdomUtilizationPercent']
    threshold = vdom_data['data']['alertThreshold']
    limit = vdom_data['data']['vdomLimit']
    used = vdom_data['data']['vdomUsed']
    available = vdom_data['data']['vdomAvailable']
    
    alerts = []
    
    if utilization >= 100:
        alerts.append({
            'level': 'critical',
            'message': f'VDOM limit reached ({used}/{limit})',
            'action': 'Contact support to increase limit'
        })
    elif utilization >= threshold:
        alerts.append({
            'level': 'warning',
            'message': f'VDOM usage at {utilization:.0f}% ({used}/{limit})',
            'action': f'Only {available} VDOM(s) remaining'
        })
    elif utilization >= 50:
        alerts.append({
            'level': 'info',
            'message': f'VDOM usage at {utilization:.0f}%',
            'action': 'Monitor usage'
        })
    else:
        alerts.append({
            'level': 'ok',
            'message': f'VDOM usage normal ({utilization:.0f}%)',
            'action': 'No action needed'
        })
    
    return alerts
```

## Integration Examples

### Python Client
```python
import requests
from typing import Dict, List, Optional

class VDOMClient:
    """Client for IPC Cloud VDOM API."""
    
    def __init__(self, base_url: str, bearer_token: str):
        self.base_url = base_url
        self.headers = {
            'Authorization': f'Bearer {bearer_token}',
            'Content-Type': 'application/json'
        }
    
    def get_vdom_limit(self, engagement_id: str) -> Dict:
        """Get VDOM limits and usage for engagement."""
        url = f"{self.base_url}/networkservice/firewallconfig/engagement/{engagement_id}/vdomLimit"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def can_create_vdom(self, engagement_id: str, count: int = 1) -> bool:
        """Check if can create specified number of VDOMs."""
        data = self.get_vdom_limit(engagement_id)
        available = data['data']['vdomAvailable']
        can_create = data['data']['canCreateVdom']
        return can_create and available >= count
    
    def get_vdom_by_name(self, engagement_id: str, vdom_name: str) -> Optional[Dict]:
        """Get specific VDOM by name."""
        data = self.get_vdom_limit(engagement_id)
        for vdom in data['data']['vdoms']:
            if vdom['vdomName'].lower() == vdom_name.lower():
                return vdom
        return None
    
    def list_vdoms_by_type(self, engagement_id: str, vdom_type: str) -> List[Dict]:
        """List all VDOMs of specific type."""
        data = self.get_vdom_limit(engagement_id)
        return [
            vdom for vdom in data['data']['vdoms']
            if vdom['vdomType'] == vdom_type
        ]
    
    def get_vdom_utilization(self, engagement_id: str) -> Dict:
        """Get VDOM utilization metrics."""
        data = self.get_vdom_limit(engagement_id)
        return {
            'limit': data['data']['vdomLimit'],
            'used': data['data']['vdomUsed'],
            'available': data['data']['vdomAvailable'],
            'utilization_percent': data['data']['vdomUtilizationPercent'],
            'can_create': data['data']['canCreateVdom']
        }
    
    def check_vdom_capacity_for_plan(
        self, 
        engagement_id: str, 
        planned_vdoms: int
    ) -> Dict:
        """Check if there's capacity for planned VDOMs."""
        data = self.get_vdom_limit(engagement_id)
        available = data['data']['vdomAvailable']
        
        return {
            'planned': planned_vdoms,
            'available': available,
            'can_accommodate': available >= planned_vdoms,
            'shortage': max(0, planned_vdoms - available),
            'message': (
                f"{'Sufficient' if available >= planned_vdoms else 'Insufficient'} "
                f"capacity for {planned_vdoms} VDOMs"
            )
        }

# Usage example
if __name__ == '__main__':
    client = VDOMClient(
        base_url='https://ipcloud.tatacommunications.com',
        bearer_token='your_token_here'
    )
    
    engagement_id = 'ENG001'
    
    # Check if can create VDOM
    if client.can_create_vdom(engagement_id):
        print("Can create new VDOM")
    else:
        print("Cannot create new VDOM - limit reached")
    
    # Get utilization
    util = client.get_vdom_utilization(engagement_id)
    print(f"VDOM Utilization: {util['utilization_percent']:.1f}%")
    print(f"Used: {util['used']}/{util['limit']}")
    print(f"Available: {util['available']}")
    
    # Check capacity for expansion
    capacity = client.check_vdom_capacity_for_plan(engagement_id, 3)
    print(f"\n{capacity['message']}")
    if capacity['shortage'] > 0:
        print(f"Need {capacity['shortage']} more VDOM slot(s)")
    
    # List production VDOMs
    prod_vdom = client.get_vdom_by_name(engagement_id, 'production')
    if prod_vdom:
        print(f"\nProduction VDOM:")
        print(f"  Interfaces: {prod_vdom['interfaceCount']}")
        print(f"  Policies: {prod_vdom['policyCount']}")
        print(f"  VPN Tunnels: {prod_vdom['vpnTunnelCount']}")
```

### JavaScript Client
```javascript
const axios = require('axios');

class VDOMClient {
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

  async getVdomLimit(engagementId) {
    const response = await this.axiosInstance.get(
      `/networkservice/firewallconfig/engagement/${engagementId}/vdomLimit`
    );
    return response.data;
  }

  async canCreateVdom(engagementId, count = 1) {
    const data = await this.getVdomLimit(engagementId);
    return data.data.canCreateVdom && data.data.vdomAvailable >= count;
  }

  async getVdomSummary(engagementId) {
    const data = await this.getVdomLimit(engagementId);
    const vdoms = data.data.vdoms;

    return {
      total: vdoms.length,
      active: vdoms.filter(v => v.status === 'active').length,
      byType: data.data.vdomByType,
      totalPolicies: vdoms.reduce((sum, v) => sum + (v.policyCount || 0), 0),
      totalSessions: vdoms.reduce(
        (sum, v) => sum + (v.resourceUsage?.sessions || 0), 0
      ),
      utilizationPercent: data.data.vdomUtilizationPercent
    };
  }

  async checkCapacity(engagementId, plannedVdoms) {
    const data = await this.getVdomLimit(engagementId);
    const available = data.data.vdomAvailable;

    return {
      planned: plannedVdoms,
      available: available,
      canAccommodate: available >= plannedVdoms,
      shortage: Math.max(0, plannedVdoms - available)
    };
  }
}

// Usage
(async () => {
  const client = new VDOMClient(
    'https://ipcloud.tatacommunications.com',
    'your_token_here'
  );

  try {
    const engagementId = 'ENG001';

    // Check capacity
    const canCreate = await client.canCreateVdom(engagementId, 2);
    console.log(`Can create 2 VDOMs: ${canCreate}`);

    // Get summary
    const summary = await client.getVdomSummary(engagementId);
    console.log('VDOM Summary:', summary);

    // Check expansion capacity
    const capacity = await client.checkCapacity(engagementId, 5);
    console.log('Capacity Check:', capacity);
  } catch (error) {
    console.error('Error:', error.message);
  }
})();
```

## Related Operations
- `firewall.create_vdom` - Create new virtual domain
- `firewall.delete_vdom` - Delete virtual domain
- `firewall.update_vdom` - Modify VDOM configuration
- `firewall.list_policies` - List firewall policies in VDOM
- `engagement.get_details` - Get engagement information
- `license.upgrade` - Upgrade license to increase limits

## Error Handling

### Common Errors
- **401 Unauthorized**: Invalid or expired bearer token
  - **Resolution**: Refresh authentication token
  
- **403 Forbidden**: User doesn't have firewall access
  - **Resolution**: Requires admin or developer role
  
- **404 Not Found**: Engagement not found or no firewall configured
  - **Resolution**: Verify engagement ID and firewall setup
  
- **500 Internal Server Error**: Firewall service error
  - **Resolution**: Retry after a few seconds

### Error Response Example
```json
{
  "status": "error",
  "message": "Firewall not configured for engagement",
  "responseCode": 404,
  "errorDetails": {
    "code": "FIREWALL_NOT_CONFIGURED",
    "engagementId": "ENG001"
  }
}
```

## Response Codes
- `0` - Success
- `1` - Authentication failed
- `2` - Authorization failed
- `404` - Engagement or firewall not found
- `500` - Internal server error

## Performance Notes
- **Response time**: < 1 second
- **Caching**: VDOM data cached for 5 minutes
- **Rate limiting**: 100 requests per minute

## Usage Notes

### VDOM Types
- **system**: System VDOMs (root, management) - cannot be deleted
- **customer**: Customer VDOMs - can be created/deleted

### VDOM Limits
- Limits based on FortiGate model and license
- Enterprise licenses typically allow 10-50 VDOMs
- Contact support to increase limits

### VDOM Best Practices
1. **Plan capacity**: Reserve VDOMs for future environments
2. **Monitor usage**: Set alerts at 80% utilization
3. **Separate environments**: Use VDOMs to isolate prod/stage/dev
4. **Security isolation**: Separate security domains with VDOMs
5. **Resource allocation**: Distribute resources evenly across VDOMs

### When to Create New VDOM
- New environment (production, staging, development)
- Different security requirements
- Customer separation in multi-tenant setup
- Compliance requirements (PCI, HIPAA zones)
- DMZ or public-facing services

## Metadata
- **Generated:** 2025-02-16T10:45:00.000000Z
- **Source:** IPC Cloud Network Service
- **API Version:** v2
- **Base Path:** /networkservice/firewallconfig/
- **Documentation:** FortiGate Virtual Domain (VDOM) management
- **Additional Notes:** VDOM limits vary by FortiGate model and license type