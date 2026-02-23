# API Specification: order - get_item_details

**Resource:** order
**Operation:** get_item_details
**Aliases:** order details, order item, get order, order information, order status, service order, order item details, purchase order

## Endpoint
- **Method:** GET
- **URL:** {BASE_URL_PORTAL_SERVICE}/configservice/getorderitemdetails/{engagement_id}/{order_item_id}?orderType=optimus
- **Auth:** Bearer token (from Keycloak)
- **Description:** Get detailed information about a specific order item including service configuration, deployment status, resource allocation, and provisioning details for Optimus-based orders

## Required Parameters
- `engagement_id` - The unique identifier of the engagement (string, path parameter)
- `order_item_id` - The unique identifier of the order item (number, path parameter)
- `orderType` - Type of order system: optimus, legacy (string, query parameter, default: optimus)

## Optional Parameters
- `include_history` - Include order history (query parameter, default: false)
- `include_details` - Include full configuration details (query parameter, default: true)

## Response Mapping
- `status`: status
- `message`: message
- `responseCode`: responseCode
- `order_item`: data.orderItem
- `order_item_id`: data.orderItem.orderItemId
- `order_id`: data.orderItem.orderId
- `engagement_id`: data.orderItem.engagementId
- `service_type`: data.orderItem.serviceType
- `order_status`: data.orderItem.status
- `created_time`: data.orderItem.createdTime
- `deployment_status`: data.orderItem.deploymentStatus
- `resource_details`: data.orderItem.resourceDetails
- `configuration`: data.orderItem.configuration
- `pricing`: data.orderItem.pricing

## Response Example
```json
{
  "status": "success",
  "data": {
    "orderItem": {
      "orderItemId": 12,
      "orderId": "ORD-2025-001234",
      "engagementId": "ENG001",
      "engagementName": "Enterprise Cloud Platform",
      "orderType": "optimus",
      "serviceType": "kubernetes",
      "serviceName": "Kubernetes Cluster",
      "status": "deployed",
      "deploymentStatus": "active",
      "createdTime": 1765190613000,
      "createdBy": "admin@domain.com",
      "approvedTime": 1765191613000,
      "approvedBy": "manager@domain.com",
      "deployedTime": 1765194613000,
      "completedTime": 1765196613000,
      "resourceDetails": {
        "resourceType": "cluster",
        "resourceId": "CLU-12345",
        "resourceName": "production-cluster",
        "endpoint": "EP_V2_BL",
        "endpointDisplayName": "Bengaluru",
        "zone": "PROD",
        "department": "ENG"
      },
      "configuration": {
        "clusterName": "production-cluster",
        "kubernetesVersion": "1.28.0",
        "nodeCount": 3,
        "nodeType": "standard",
        "nodeConfig": {
          "cpuCores": 4,
          "memoryGB": 16,
          "diskGB": 100,
          "instanceType": "m5.xlarge"
        },
        "networkConfig": {
          "vpcId": "vpc-12345",
          "subnetIds": ["subnet-001", "subnet-002"],
          "securityGroupIds": ["sg-001"],
          "loadBalancerEnabled": true
        },
        "storageConfig": {
          "storageClass": "premium",
          "volumeSize": 500,
          "backupEnabled": true,
          "snapshotSchedule": "daily"
        },
        "additionalFeatures": {
          "monitoring": true,
          "logging": true,
          "autoScaling": true,
          "highAvailability": true
        }
      },
      "pricing": {
        "currency": "USD",
        "setupFee": 0,
        "monthlyRecurring": 450.00,
        "hourlyRate": 0.62,
        "billingModel": "hourly",
        "priceBreakdown": {
          "compute": 300.00,
          "storage": 100.00,
          "network": 50.00
        },
        "discounts": {
          "volumeDiscount": 10,
          "commitmentDiscount": 0
        },
        "estimatedMonthlyCost": 405.00
      },
      "workflow": {
        "currentStep": "completed",
        "totalSteps": 5,
        "steps": [
          {
            "stepNumber": 1,
            "stepName": "Order Validation",
            "status": "completed",
            "startTime": 1765190613000,
            "endTime": 1765190713000,
            "duration": 100
          },
          {
            "stepNumber": 2,
            "stepName": "Resource Allocation",
            "status": "completed",
            "startTime": 1765190713000,
            "endTime": 1765192000000,
            "duration": 1287
          },
          {
            "stepNumber": 3,
            "stepName": "Network Configuration",
            "status": "completed",
            "startTime": 1765192000000,
            "endTime": 1765193200000,
            "duration": 1200
          },
          {
            "stepNumber": 4,
            "stepName": "Cluster Deployment",
            "status": "completed",
            "startTime": 1765193200000,
            "endTime": 1765196000000,
            "duration": 2800
          },
          {
            "stepNumber": 5,
            "stepName": "Post-Deployment Validation",
            "status": "completed",
            "startTime": 1765196000000,
            "endTime": 1765196613000,
            "duration": 613
          }
        ]
      },
      "dependencies": {
        "requiredOrders": [],
        "dependentOrders": ["ORD-2025-001235"],
        "sharedResources": ["vpc-12345", "sg-001"]
      },
      "contacts": {
        "orderedBy": "admin@domain.com",
        "technicalContact": "ops@domain.com",
        "billingContact": "finance@domain.com"
      },
      "notifications": {
        "emailSent": true,
        "smsNotification": false,
        "webhookTriggered": true,
        "recipients": ["admin@domain.com", "ops@domain.com"]
      },
      "metadata": {
        "orderSource": "portal",
        "orderChannel": "self-service",
        "priority": "normal",
        "tags": ["production", "kubernetes", "critical"],
        "notes": "Production cluster for main application"
      }
    },
    "orderHistory": [
      {
        "timestamp": 1765190613000,
        "action": "created",
        "user": "admin@domain.com",
        "details": "Order created"
      },
      {
        "timestamp": 1765191613000,
        "action": "approved",
        "user": "manager@domain.com",
        "details": "Order approved"
      },
      {
        "timestamp": 1765194613000,
        "action": "deployed",
        "user": "system",
        "details": "Resources deployed successfully"
      },
      {
        "timestamp": 1765196613000,
        "action": "completed",
        "user": "system",
        "details": "Order completed and activated"
      }
    ]
  },
  "message": "success",
  "responseCode": 0
}
```

## Response Fields Details

### Order Item Fields
- **orderItemId** - Order item identifier (number)
- **orderId** - Parent order identifier (string)
- **engagementId** - Engagement identifier (string)
- **engagementName** - Engagement name (string)
- **orderType** - Order system type: optimus, legacy (string)
- **serviceType** - Type of service: kubernetes, vm, storage, network (string)
- **serviceName** - Human-readable service name (string)
- **status** - Order status: pending, approved, deployed, completed, failed (string)
- **deploymentStatus** - Deployment status: pending, deploying, active, error (string)
- **createdTime** - Unix timestamp when created (number)
- **createdBy** - Email of creator (string)
- **approvedTime** - Unix timestamp when approved (number)
- **approvedBy** - Email of approver (string)
- **deployedTime** - Unix timestamp when deployed (number)
- **completedTime** - Unix timestamp when completed (number)

### Resource Details
- **resourceType** - Type of resource created (string)
- **resourceId** - Unique resource identifier (string)
- **resourceName** - Resource name (string)
- **endpoint** - Deployment endpoint (string)
- **endpointDisplayName** - Human-readable endpoint (string)
- **zone** - Zone identifier (string)
- **department** - Department code (string)

### Configuration
Service-specific configuration details (varies by service type)

### Pricing
- **currency** - Currency code (string)
- **setupFee** - One-time setup fee (number)
- **monthlyRecurring** - Monthly recurring cost (number)
- **hourlyRate** - Hourly rate (number)
- **billingModel** - How service is billed (string)
- **priceBreakdown** - Cost by component (object)
- **discounts** - Applied discounts (object)
- **estimatedMonthlyCost** - Total estimated cost (number)

### Workflow
- **currentStep** - Current workflow step (string)
- **totalSteps** - Total workflow steps (number)
- **steps** - Array of workflow step objects

### Dependencies
- **requiredOrders** - Orders that must complete first (array)
- **dependentOrders** - Orders waiting on this one (array)
- **sharedResources** - Resources shared with other orders (array)

## Permissions
Roles: admin, developer, viewer

## Workflow Steps
### Workflow: get_order_item_details
Get detailed order item information
- Step 1: validate_engagement_id (engagement.validate) (depends on: engagement_id)
- Step 2: validate_order_item (order.validate) (depends on: order_item_id)
- Step 3: get_order_details (order.get_item_details) (depends on: engagement_id, order_item_id, orderType)
- Step 4: get_resource_details (resource.get_details) (depends on: resourceId)
- Step 5: get_order_history (order.get_history) (depends on: order_item_id)
- Step 6: format_response (response.format)

## Common Use Cases

1. **Check order status**: "What is the status of order item 12?"
2. **View deployment progress**: "Is my cluster deployed yet?"
3. **Get resource details**: "What resources were created by this order?"
4. **Review configuration**: "Show me the cluster configuration from order"
5. **Check pricing**: "How much does this order cost?"
6. **View workflow progress**: "Which deployment step is running?"
7. **Track order history**: "When was this order approved?"
8. **Find resource ID**: "What is the cluster ID for order 12?"
9. **Check dependencies**: "Are there any dependent orders?"
10. **Review contacts**: "Who ordered this service?"

## Query Interpretations
- "order 12 status" → GET /getorderitemdetails/{eng_id}/12?orderType=optimus
- "order details" → GET /getorderitemdetails/{eng_id}/{item_id}
- "deployment status" → Extract data.orderItem.deploymentStatus
- "order cost" → Extract data.orderItem.pricing
- "resource created" → Extract data.orderItem.resourceDetails

## Data Processing Examples

### Check Order Status
```python
def get_order_status_summary(order_data):
    """Get a summary of order status."""
    order = order_data['data']['orderItem']
    
    return {
        'order_id': order['orderId'],
        'order_item_id': order['orderItemId'],
        'service': order['serviceName'],
        'status': order['status'],
        'deployment_status': order['deploymentStatus'],
        'is_completed': order['status'] == 'completed',
        'is_active': order['deploymentStatus'] == 'active',
        'created_date': order['createdTime'],
        'days_since_order': (
            datetime.now().timestamp() * 1000 - order['createdTime']
        ) / (1000 * 60 * 60 * 24)
    }
```

### Calculate Deployment Duration
```python
def calculate_deployment_duration(order_data):
    """Calculate how long deployment took."""
    workflow = order_data['data']['orderItem']['workflow']
    
    total_duration = sum(step['duration'] for step in workflow['steps'])
    
    return {
        'total_duration_seconds': total_duration,
        'total_duration_minutes': total_duration / 60,
        'steps': [
            {
                'step': step['stepName'],
                'duration_seconds': step['duration'],
                'status': step['status']
            }
            for step in workflow['steps']
        ],
        'slowest_step': max(
            workflow['steps'],
            key=lambda x: x['duration']
        )['stepName']
    }
```

### Get Resource Information
```python
def get_created_resources(order_data):
    """Get information about resources created by order."""
    order = order_data['data']['orderItem']
    resource = order['resourceDetails']
    config = order['configuration']
    
    return {
        'resource_id': resource['resourceId'],
        'resource_name': resource['resourceName'],
        'resource_type': resource['resourceType'],
        'location': resource['endpointDisplayName'],
        'zone': resource['zone'],
        'department': resource['department'],
        'configuration_summary': {
            k: v for k, v in config.items()
            if k not in ['networkConfig', 'storageConfig']
        }
    }
```

## Integration Examples

### Python Client
```python
import requests
from typing import Dict, Optional

class OrderClient:
    """Client for IPC Cloud Order API."""
    
    def __init__(self, base_url: str, bearer_token: str):
        self.base_url = base_url
        self.headers = {
            'Authorization': f'Bearer {bearer_token}',
            'Content-Type': 'application/json'
        }
    
    def get_order_item_details(
        self,
        engagement_id: str,
        order_item_id: int,
        order_type: str = 'optimus',
        include_history: bool = False
    ) -> Dict:
        """Get order item details."""
        url = f"{self.base_url}/portalservice/configservice/getorderitemdetails/{engagement_id}/{order_item_id}"
        
        params = {'orderType': order_type}
        if include_history:
            params['include_history'] = 'true'
        
        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        return response.json()
    
    def get_order_status(
        self,
        engagement_id: str,
        order_item_id: int
    ) -> str:
        """Get quick order status."""
        data = self.get_order_item_details(engagement_id, order_item_id)
        return data['data']['orderItem']['status']
    
    def is_order_completed(
        self,
        engagement_id: str,
        order_item_id: int
    ) -> bool:
        """Check if order is completed."""
        status = self.get_order_status(engagement_id, order_item_id)
        return status == 'completed'
    
    def get_resource_id(
        self,
        engagement_id: str,
        order_item_id: int
    ) -> Optional[str]:
        """Get resource ID created by order."""
        data = self.get_order_item_details(engagement_id, order_item_id)
        return data['data']['orderItem']['resourceDetails'].get('resourceId')

# Usage example
if __name__ == '__main__':
    client = OrderClient(
        base_url='https://ipcloud.tatacommunications.com',
        bearer_token='your_token_here'
    )
    
    details = client.get_order_item_details('ENG001', 12)
    order = details['data']['orderItem']
    
    print(f"Order: {order['orderId']}")
    print(f"Service: {order['serviceName']}")
    print(f"Status: {order['status']}")
    print(f"Resource: {order['resourceDetails']['resourceId']}")
```

## Related Operations
- `order.list` - List all orders for engagement
- `order.create` - Create new order
- `order.cancel` - Cancel pending order
- `order.approve` - Approve order
- `resource.get_details` - Get resource details
- `billing.get_invoice` - Get order invoice

## Error Handling

### Common Errors
- **401 Unauthorized**: Invalid bearer token
- **403 Forbidden**: No access to engagement/order
- **404 Not Found**: Order item not found
- **500 Internal Server Error**: Service error

### Error Response Example
```json
{
  "status": "error",
  "message": "Order item not found",
  "responseCode": 404
}
```

## Response Codes
- `0` - Success
- `1` - Authentication failed
- `2` - Authorization failed
- `404` - Order not found
- `500` - Internal server error

## Performance Notes
- **Response time**: < 1 second
- **Caching**: Order data cached for 2 minutes
- **Rate limiting**: 200 requests per minute

## Metadata
- **Generated:** 2025-02-16T11:00:00.000000Z
- **Source:** IPC Cloud Portal Configuration Service  
- **API Version:** v2
- **Base Path:** /portalservice/configservice/