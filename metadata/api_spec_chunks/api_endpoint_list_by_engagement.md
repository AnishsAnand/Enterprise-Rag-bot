# API Specification: endpoint - list_by_engagement

**Resource:** endpoint
**Operation:** list_by_engagement
**Aliases:** endpoints, list endpoints, engagement endpoints, get endpoints, show endpoints, endpoint configuration

## Endpoint
- **Method:** GET
- **URL:** {BASE_URL_PORTAL_SERVICE}/configservice/getEndpointsByEngagement/{engagement_id}
- **Auth:** Bearer token (from Keycloak)
- **Description:** Get all configured endpoints (data centers, availability zones) associated with a specific engagement for resource provisioning and deployment

## Required Parameters
- `engagement_id`: The unique identifier of the engagement (alphanumeric)

## Optional Parameters
None

## Response Mapping
- `status`: data.status
- `message`: data.message
- `responseCode`: data.responseCode
- `endpoints`: data.data.endpoints[*]
- `engagement_id`: data.data.engagementId
- `engagement_name`: data.data.engagementName
- `endpoint_id`: data.data.endpoints[*].endpointId
- `endpoint_name`: data.data.endpoints[*].endpointName
- `endpoint_code`: data.data.endpoints[*].endpointCode
- `display_name`: data.data.endpoints[*].displayName
- `location`: data.data.endpoints[*].location
- `region`: data.data.endpoints[*].region
- `endpoint_type`: data.data.endpoints[*].type
- `status`: data.data.endpoints[*].status
- `capabilities`: data.data.endpoints[*].capabilities[*]

## Response Structure
```json
{
  "status": "success",
  "data": {
    "engagementId": "ENG001",
    "engagementName": "Enterprise Cloud Platform",
    "endpoints": [
      {
        "endpointId": "EP001",
        "endpointName": "EP_V2_BL",
        "endpointCode": "EP_V2_BL",
        "displayName": "Bengaluru",
        "location": "Bengaluru",
        "region": "India",
        "country": "IN",
        "type": "V2",
        "status": "Active",
        "capabilities": [
          "compute",
          "storage",
          "kubernetes",
          "networking"
        ],
        "zones": [
          {
            "zoneId": "Z001",
            "zoneName": "zone-a",
            "status": "Active"
          },
          {
            "zoneId": "Z002",
            "zoneName": "zone-b",
            "status": "Active"
          }
        ]
      },
      {
        "endpointId": "EP002",
        "endpointName": "EP_V2_SG_TCX",
        "endpointCode": "EP_V2_SG_TCX",
        "displayName": "Singapore East",
        "location": "Singapore",
        "region": "Asia Pacific",
        "country": "SG",
        "type": "V2",
        "status": "Active",
        "capabilities": [
          "compute",
          "storage",
          "kubernetes",
          "networking",
          "gpu"
        ],
        "zones": [
          {
            "zoneId": "Z003",
            "zoneName": "zone-a",
            "status": "Active"
          }
        ]
      },
      {
        "endpointId": "EP003",
        "endpointName": "EP_V2_UKHB",
        "endpointCode": "EP_V2_UKHB",
        "displayName": "Highbridge",
        "location": "London",
        "region": "Europe",
        "country": "UK",
        "type": "V2",
        "status": "Active",
        "capabilities": [
          "compute",
          "storage",
          "kubernetes"
        ],
        "zones": [
          {
            "zoneId": "Z004",
            "zoneName": "zone-a",
            "status": "Active"
          }
        ]
      }
    ]
  },
  "message": "success",
  "responseCode": 0
}
```

## Permissions
Roles: admin, developer, viewer

## Workflow Steps
### Workflow: list_endpoints_by_engagement
Get all endpoints for an engagement
- Step 1: validate_engagement (engagement.validate) (depends on: engagement_id)
- Step 2: get_endpoints (endpoint.list_by_engagement) (depends on: engagement_id)
- Step 3: get_capabilities (endpoint.get_capabilities)
- Step 4: format_response (response.format)

## Field Descriptions
- **engagementId**: Unique identifier for the engagement
- **engagementName**: Name of the engagement
- **endpoints**: Array of endpoint configurations
- **endpointId**: Unique identifier for the endpoint
- **endpointName**: Internal name/code for the endpoint
- **endpointCode**: Short code used in resource naming
- **displayName**: Human-readable location name
- **location**: City or datacenter location
- **region**: Geographical region
- **country**: Country code (ISO 3166-1 alpha-2)
- **type**: Endpoint type (V2, GCC, etc.)
- **status**: Endpoint availability status (Active, Inactive, Maintenance)
- **capabilities**: Array of services available at this endpoint
- **zones**: Availability zones within the endpoint

## Usage Notes
This endpoint is used for:
- Displaying available deployment locations to users
- Validating resource placement requests
- Capacity planning across multiple locations
- Multi-region deployment configuration
- Disaster recovery planning

The engagement_id parameter determines which endpoints are accessible. Users can only see endpoints associated with their engagement.

## Endpoint Types
- **V2**: Standard IPC v2 endpoints with full capabilities
- **GCC**: Government Cloud Compute endpoints (specialized security/compliance)
- **Legacy**: Older generation endpoints (limited capabilities)

## Capabilities
Common endpoint capabilities include:
- **compute**: Virtual machine provisioning
- **storage**: Block and object storage
- **kubernetes**: Container orchestration (CaaS)
- **networking**: Virtual networks, load balancers, VPN
- **gpu**: GPU-accelerated instances
- **backup**: Backup and disaster recovery
- **database**: Managed database services

## Common Response Codes
- **0**: Success
- **1**: Authentication failed
- **2**: Authorization failed
- **404**: Engagement not found
- **403**: User not associated with engagement
- **500**: Internal server error