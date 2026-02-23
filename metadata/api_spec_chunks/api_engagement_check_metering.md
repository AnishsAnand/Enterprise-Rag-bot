# API Specification: engagement - check_metering

**Resource:** engagement
**Operation:** check_metering
**Aliases:** metering, check metering, metering status, billing enabled, usage tracking, metering check, is metering enabled, billing status

## Endpoint
- **Method:** GET
- **URL:** {BASE_URL_PORTAL_SERVICE}/config/checkForMetering/{engagement_id}
- **Auth:** Bearer token (from Keycloak)
- **Description:** Check if metering (usage tracking and billing) is enabled for a specific engagement, returns metering configuration, billing status, and cost tracking settings

## Required Parameters
- `engagement_id` - The unique identifier of the engagement (string, path parameter)

## Optional Parameters
None

## Response Mapping
- `status`: status
- `message`: message
- `responseCode`: responseCode
- `metering_enabled`: data.meteringEnabled
- `engagement_id`: data.engagementId
- `billing_enabled`: data.billingEnabled
- `metering_type`: data.meteringType
- `cost_tracking_enabled`: data.costTrackingEnabled
- `billing_cycle`: data.billingCycle
- `currency`: data.currency
- `billing_start_date`: data.billingStartDate
- `last_metering_date`: data.lastMeteringDate
- `metering_frequency`: data.meteringFrequency
- `metering_services`: data.meteringServices[*]

## Response Example
```json
{
  "status": "success",
  "data": {
    "engagementId": "ENG001",
    "engagementName": "Enterprise Cloud Platform",
    "meteringEnabled": true,
    "billingEnabled": true,
    "meteringType": "detailed",
    "costTrackingEnabled": true,
    "billingCycle": "monthly",
    "currency": "USD",
    "billingAccount": "BA-ENT-001",
    "costCenter": "CC-CLOUD-001",
    "billingStartDate": 1650925643000,
    "lastMeteringDate": 1765190613000,
    "nextMeteringDate": 1765276800000,
    "meteringFrequency": "hourly",
    "meteringGranularity": "resource",
    "meteringServices": [
      {
        "serviceType": "compute",
        "meteringEnabled": true,
        "billingModel": "hourly",
        "meteredResources": ["vm", "cpu", "memory"],
        "costPerUnit": {
          "vm": 0.05,
          "cpu": 0.02,
          "memory": 0.01
        },
        "unitType": {
          "vm": "per_hour",
          "cpu": "per_core_hour",
          "memory": "per_GB_hour"
        }
      },
      {
        "serviceType": "storage",
        "meteringEnabled": true,
        "billingModel": "monthly",
        "meteredResources": ["block_storage", "object_storage"],
        "costPerUnit": {
          "block_storage": 0.10,
          "object_storage": 0.05
        },
        "unitType": {
          "block_storage": "per_GB_month",
          "object_storage": "per_GB_month"
        }
      },
      {
        "serviceType": "network",
        "meteringEnabled": true,
        "billingModel": "usage",
        "meteredResources": ["bandwidth", "load_balancer", "vpn"],
        "costPerUnit": {
          "bandwidth": 0.08,
          "load_balancer": 0.025,
          "vpn": 0.05
        },
        "unitType": {
          "bandwidth": "per_GB",
          "load_balancer": "per_hour",
          "vpn": "per_hour"
        }
      },
      {
        "serviceType": "kubernetes",
        "meteringEnabled": true,
        "billingModel": "cluster_hour",
        "meteredResources": ["cluster", "node", "pod"],
        "costPerUnit": {
          "cluster": 0.10,
          "node": 0.05,
          "pod": 0.01
        },
        "unitType": {
          "cluster": "per_hour",
          "node": "per_node_hour",
          "pod": "per_pod_hour"
        }
      }
    ],
    "costAllocation": {
      "enabled": true,
      "method": "department",
      "tagsRequired": false,
      "showbackEnabled": true,
      "chargebackEnabled": true
    },
    "reportingSettings": {
      "dailyReports": true,
      "monthlyReports": true,
      "budgetAlerts": true,
      "costAnomalyDetection": true,
      "reportRecipients": [
        "finance@domain.com",
        "admin@domain.com"
      ]
    },
    "currentMonth": {
      "totalCost": 45678.90,
      "projectedCost": 52000.00,
      "budget": 50000.00,
      "budgetUtilization": 91.36,
      "topCostServices": [
        {
          "service": "compute",
          "cost": 25000.00,
          "percentage": 54.74
        },
        {
          "service": "storage",
          "cost": 12000.00,
          "percentage": 26.27
        },
        {
          "service": "network",
          "cost": 6000.00,
          "percentage": 13.14
        }
      ]
    },
    "meteringConfig": {
      "dataRetentionDays": 365,
      "aggregationLevel": "hourly",
      "exportEnabled": true,
      "exportFormat": "csv",
      "apiAccessEnabled": true
    }
  },
  "message": "success",
  "responseCode": 0
}
```

## Response Fields Details

### Top-Level Fields
- **engagementId** - Engagement identifier (string)
- **engagementName** - Engagement name (string)
- **meteringEnabled** - Whether metering is enabled (boolean)
- **billingEnabled** - Whether billing is active (boolean)
- **meteringType** - Type of metering: basic, detailed, advanced (string)
- **costTrackingEnabled** - Whether costs are tracked (boolean)
- **billingCycle** - Billing frequency: monthly, quarterly, annual (string)
- **currency** - Billing currency code (string)
- **billingAccount** - Associated billing account (string)
- **costCenter** - Cost center code (string)
- **billingStartDate** - Unix timestamp when billing started (number)
- **lastMeteringDate** - Last metering timestamp (number)
- **nextMeteringDate** - Next scheduled metering (number)
- **meteringFrequency** - How often metering runs: hourly, daily (string)
- **meteringGranularity** - Level of detail: resource, service, department (string)

### Metering Services
- **serviceType** - Type of service being metered (string)
- **meteringEnabled** - Service-specific metering status (boolean)
- **billingModel** - How service is billed: hourly, monthly, usage (string)
- **meteredResources** - Array of resources being tracked (array)
- **costPerUnit** - Cost per unit for each resource (object)
- **unitType** - Unit description for pricing (object)

### Cost Allocation
- **enabled** - Cost allocation enabled (boolean)
- **method** - Allocation method: department, tag, project (string)
- **tagsRequired** - Whether tagging is mandatory (boolean)
- **showbackEnabled** - Show costs to departments (boolean)
- **chargebackEnabled** - Actually charge departments (boolean)

### Reporting Settings
- **dailyReports** - Send daily cost reports (boolean)
- **monthlyReports** - Send monthly summaries (boolean)
- **budgetAlerts** - Alert on budget thresholds (boolean)
- **costAnomalyDetection** - Detect unusual spending (boolean)
- **reportRecipients** - Email addresses for reports (array)

### Current Month Statistics
- **totalCost** - Month-to-date spending (number)
- **projectedCost** - Projected month-end cost (number)
- **budget** - Monthly budget (number)
- **budgetUtilization** - Percentage of budget used (number)
- **topCostServices** - Highest cost services (array)

## Permissions
Roles: admin, developer, viewer (billing details may be restricted)

## Workflow Steps
### Workflow: check_metering_status
Check if metering is enabled for engagement
- Step 1: validate_engagement_id (engagement.validate) (depends on: engagement_id)
- Step 2: check_metering (engagement.check_metering) (depends on: engagement_id)
- Step 3: get_billing_config (billing.get_config) (depends on: engagement_id)
- Step 4: get_current_usage (metering.get_current_month) (depends on: engagement_id)
- Step 5: format_response (response.format)

## Common Use Cases

1. **Check if metering enabled**: "Is metering enabled for this engagement?"
2. **Verify billing status**: "Is billing active for engagement ENG001?"
3. **Check metering frequency**: "How often is usage metered?"
4. **View billing cycle**: "What is the billing cycle?"
5. **Check cost tracking**: "Is cost tracking enabled?"
6. **View metered services**: "Which services are being metered?"
7. **Check current costs**: "What is the current month spending?"
8. **Budget verification**: "Are we within budget?"
9. **Service pricing**: "How much does compute cost per hour?"
10. **Cost allocation method**: "How are costs allocated?"

## Query Interpretations
- "is metering enabled" → GET /checkForMetering/{engagement_id}
- "billing status" → GET /checkForMetering/{engagement_id}
- "usage tracking" → GET /checkForMetering/{engagement_id}
- "current costs" → Extract data.currentMonth.totalCost
- "budget status" → Extract data.currentMonth.budgetUtilization

## Data Processing Examples

### Check If Metering Is Enabled
```python
def is_metering_enabled(metering_data):
    """Check if metering is enabled for engagement."""
    return {
        'metering_enabled': metering_data['data'].get('meteringEnabled', False),
        'billing_enabled': metering_data['data'].get('billingEnabled', False),
        'cost_tracking': metering_data['data'].get('costTrackingEnabled', False)
    }
```

### Calculate Service Costs
```python
def calculate_service_costs(metering_data, resource_usage):
    """
    Calculate estimated costs based on usage.
    
    Args:
        metering_data: Response from checkForMetering API
        resource_usage: Dict with resource usage data
            Example: {'vm': 10, 'cpu': 40, 'memory': 160}
    
    Returns:
        Dict with cost breakdown by service
    """
    costs = {}
    
    for service in metering_data['data']['meteringServices']:
        service_type = service['serviceType']
        costs[service_type] = 0
        
        for resource in service['meteredResources']:
            if resource in resource_usage:
                usage = resource_usage[resource]
                cost_per_unit = service['costPerUnit'].get(resource, 0)
                costs[service_type] += usage * cost_per_unit
    
    costs['total'] = sum(costs.values())
    return costs

# Usage example
usage = {
    'vm': 10,           # 10 VMs
    'cpu': 40,          # 40 CPU cores
    'memory': 160,      # 160 GB RAM
    'block_storage': 5000,  # 5000 GB storage
    'bandwidth': 1000   # 1000 GB transfer
}

costs = calculate_service_costs(metering_data, usage)
print(f"Estimated monthly cost: ${costs['total']:.2f}")
```

### Check Budget Status
```python
def check_budget_status(metering_data):
    """Check budget utilization and alert if necessary."""
    current_month = metering_data['data'].get('currentMonth', {})
    
    total_cost = current_month.get('totalCost', 0)
    budget = current_month.get('budget', 0)
    utilization = current_month.get('budgetUtilization', 0)
    projected = current_month.get('projectedCost', 0)
    
    status = {
        'current_cost': total_cost,
        'budget': budget,
        'remaining': budget - total_cost,
        'utilization_percent': utilization,
        'projected_cost': projected,
        'projected_overage': max(0, projected - budget),
        'alert_level': 'none'
    }
    
    # Determine alert level
    if utilization >= 100:
        status['alert_level'] = 'critical'
        status['message'] = 'Budget exceeded!'
    elif utilization >= 90:
        status['alert_level'] = 'warning'
        status['message'] = 'Approaching budget limit'
    elif utilization >= 75:
        status['alert_level'] = 'caution'
        status['message'] = '75% of budget used'
    else:
        status['alert_level'] = 'normal'
        status['message'] = 'Within budget'
    
    return status
```

### Get Service Pricing
```python
def get_service_pricing(metering_data, service_type):
    """Get pricing information for a specific service."""
    for service in metering_data['data']['meteringServices']:
        if service['serviceType'] == service_type:
            pricing = []
            for resource in service['meteredResources']:
                cost = service['costPerUnit'].get(resource, 0)
                unit = service['unitType'].get(resource, '')
                pricing.append({
                    'resource': resource,
                    'cost': cost,
                    'unit': unit,
                    'billing_model': service['billingModel']
                })
            return pricing
    return []

# Usage
compute_pricing = get_service_pricing(metering_data, 'compute')
for item in compute_pricing:
    print(f"{item['resource']}: ${item['cost']} {item['unit']}")
```

## Integration Examples

### Python Client
```python
import requests
from typing import Dict, Optional
from datetime import datetime

class MeteringClient:
    """Client for IPC Cloud Metering API."""
    
    def __init__(self, base_url: str, bearer_token: str):
        self.base_url = base_url
        self.headers = {
            'Authorization': f'Bearer {bearer_token}',
            'Content-Type': 'application/json'
        }
    
    def check_metering(self, engagement_id: str) -> Dict:
        """Check metering status for engagement."""
        url = f"{self.base_url}/portalservice/config/checkForMetering/{engagement_id}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def is_metering_enabled(self, engagement_id: str) -> bool:
        """Quick check if metering is enabled."""
        try:
            data = self.check_metering(engagement_id)
            return data['data'].get('meteringEnabled', False)
        except:
            return False
    
    def get_current_costs(self, engagement_id: str) -> Dict:
        """Get current month cost information."""
        data = self.check_metering(engagement_id)
        return data['data'].get('currentMonth', {})
    
    def estimate_monthly_cost(
        self, 
        engagement_id: str, 
        resource_usage: Dict
    ) -> float:
        """
        Estimate monthly cost based on resource usage.
        
        Args:
            engagement_id: Engagement ID
            resource_usage: Dict of resource: usage_amount
        
        Returns:
            Estimated monthly cost
        """
        data = self.check_metering(engagement_id)
        total_cost = 0
        
        for service in data['data']['meteringServices']:
            for resource in service['meteredResources']:
                if resource in resource_usage:
                    usage = resource_usage[resource]
                    cost_per_unit = service['costPerUnit'].get(resource, 0)
                    
                    # Adjust for billing model
                    if service['billingModel'] == 'hourly':
                        # Convert to monthly (730 hours average)
                        cost = usage * cost_per_unit * 730
                    elif service['billingModel'] == 'monthly':
                        cost = usage * cost_per_unit
                    else:  # usage-based
                        cost = usage * cost_per_unit
                    
                    total_cost += cost
        
        return total_cost
    
    def check_budget_alert(self, engagement_id: str) -> Optional[str]:
        """Check if budget alert should be raised."""
        data = self.check_metering(engagement_id)
        current = data['data'].get('currentMonth', {})
        
        utilization = current.get('budgetUtilization', 0)
        
        if utilization >= 100:
            return f"CRITICAL: Budget exceeded ({utilization:.1f}%)"
        elif utilization >= 90:
            return f"WARNING: Approaching budget limit ({utilization:.1f}%)"
        elif utilization >= 75:
            return f"CAUTION: 75% of budget used ({utilization:.1f}%)"
        return None

# Usage example
if __name__ == '__main__':
    client = MeteringClient(
        base_url='https://ipcloud.tatacommunications.com',
        bearer_token='your_token_here'
    )
    
    engagement_id = 'ENG001'
    
    # Check if metering enabled
    if client.is_metering_enabled(engagement_id):
        print("Metering is enabled")
        
        # Get current costs
        costs = client.get_current_costs(engagement_id)
        print(f"Current month: ${costs['totalCost']:.2f}")
        print(f"Budget: ${costs['budget']:.2f}")
        print(f"Utilization: {costs['budgetUtilization']:.1f}%")
        
        # Check for alerts
        alert = client.check_budget_alert(engagement_id)
        if alert:
            print(f"Alert: {alert}")
        
        # Estimate costs
        usage = {'vm': 10, 'cpu': 40, 'memory': 160}
        estimated = client.estimate_monthly_cost(engagement_id, usage)
        print(f"Estimated cost: ${estimated:.2f}")
    else:
        print("Metering is not enabled for this engagement")
```

### JavaScript Client
```javascript
const axios = require('axios');

class MeteringClient {
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

  async checkMetering(engagementId) {
    const response = await this.axiosInstance.get(
      `/portalservice/config/checkForMetering/${engagementId}`
    );
    return response.data;
  }

  async isMeteringEnabled(engagementId) {
    try {
      const data = await this.checkMetering(engagementId);
      return data.data.meteringEnabled || false;
    } catch (error) {
      return false;
    }
  }

  async getBudgetStatus(engagementId) {
    const data = await this.checkMetering(engagementId);
    const currentMonth = data.data.currentMonth || {};

    const utilization = currentMonth.budgetUtilization || 0;
    let status = 'normal';
    let message = 'Within budget';

    if (utilization >= 100) {
      status = 'critical';
      message = 'Budget exceeded!';
    } else if (utilization >= 90) {
      status = 'warning';
      message = 'Approaching budget limit';
    } else if (utilization >= 75) {
      status = 'caution';
      message = '75% of budget used';
    }

    return {
      totalCost: currentMonth.totalCost,
      budget: currentMonth.budget,
      utilization: utilization,
      status: status,
      message: message,
      projected: currentMonth.projectedCost
    };
  }

  async getServicePricing(engagementId, serviceType) {
    const data = await this.checkMetering(engagementId);
    const service = data.data.meteringServices.find(
      s => s.serviceType === serviceType
    );

    if (!service) return [];

    return service.meteredResources.map(resource => ({
      resource: resource,
      cost: service.costPerUnit[resource],
      unit: service.unitType[resource],
      billingModel: service.billingModel
    }));
  }
}

// Usage
(async () => {
  const client = new MeteringClient(
    'https://ipcloud.tatacommunications.com',
    'your_token_here'
  );

  try {
    const engagementId = 'ENG001';

    // Check metering status
    const enabled = await client.isMeteringEnabled(engagementId);
    console.log(`Metering enabled: ${enabled}`);

    if (enabled) {
      // Get budget status
      const budget = await client.getBudgetStatus(engagementId);
      console.log('Budget Status:', budget);

      // Get compute pricing
      const pricing = await client.getServicePricing(engagementId, 'compute');
      console.log('Compute Pricing:', pricing);
    }
  } catch (error) {
    console.error('Error:', error.message);
  }
})();
```

## Related Operations
- `engagement.get_details` - Get engagement information
- `billing.get_invoice` - Get billing invoices
- `billing.get_usage` - Get detailed usage data
- `cost.get_allocation` - Get cost allocation by department
- `budget.set` - Set or modify budget
- `report.get_cost_report` - Get cost analysis reports
- `alert.configure` - Configure budget alerts

## Error Handling

### Common Errors
- **401 Unauthorized**: Invalid or expired bearer token
  - **Resolution**: Refresh authentication token
  
- **403 Forbidden**: User doesn't have permission to view billing
  - **Resolution**: Requires admin or billing role
  
- **404 Not Found**: Engagement ID not found
  - **Resolution**: Verify engagement ID exists
  
- **500 Internal Server Error**: Metering service error
  - **Resolution**: Retry after a few seconds

### Error Response Example
```json
{
  "status": "error",
  "message": "Engagement not found or metering not configured",
  "responseCode": 404,
  "errorDetails": {
    "code": "ENGAGEMENT_NOT_FOUND",
    "engagementId": "ENG001"
  }
}
```

## Response Codes
- `0` - Success
- `1` - Authentication failed
- `2` - Authorization failed
- `404` - Engagement not found
- `500` - Internal server error

## Performance Notes
- **Response time**: < 500ms
- **Caching**: Metering config cached for 5 minutes
- **Rate limiting**: 100 requests per minute
- **Data freshness**: Cost data updated hourly

## Usage Notes

### Metering Types
- **basic**: Simple usage tracking, no detailed costs
- **detailed**: Resource-level metering with costs
- **advanced**: Full cost allocation and analytics

### Billing Models
- **hourly**: Charged per hour of usage
- **monthly**: Fixed monthly rate
- **usage**: Pay only for actual usage (e.g., bandwidth)

### Cost Allocation Methods
- **department**: Costs allocated by department
- **tag**: Resource tags determine allocation
- **project**: Project-based cost allocation

### Budget Alerts
Configure thresholds for automatic alerts:
- 75% - Caution notification
- 90% - Warning notification
- 100% - Critical alert
- 110% - Overage alert

## Best Practices

1. **Check before provisioning**: Verify metering enabled before creating resources
2. **Monitor regularly**: Check budget status weekly
3. **Set alerts**: Configure budget threshold alerts
4. **Review pricing**: Understand service costs before deployment
5. **Tag resources**: Use tags for cost allocation
6. **Optimize usage**: Review top cost services monthly
7. **Budget planning**: Use projected costs for planning

## Metadata
- **Generated:** 2025-02-16T10:30:00.000000Z
- **Source:** IPC Cloud Portal Configuration Service
- **API Version:** v2
- **Base Path:** /portalservice/config/
- **Documentation:** Engagement metering and billing configuration
- **Additional Notes:** Returns detailed metering configuration and current cost status