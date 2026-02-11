# API Specification: endpoint - list

**Resource:** endpoint
**Operation:** list
**Aliases:** datacenter, dc, data center, location, datacenters, data centers, locations, endpoints, dcs

## Endpoint
- **Method:** GET
- **URL:** https://ipcloud.tatacommunications.com/portalservice/configservice/getEndpointsByEngagement/{engagement_id}
- **Auth:** Bearer token (from Keycloak)
- **Description:** Fetch available endpoints (data centers) for a given engagement. User can select which endpoints to query for resources.

## Required Parameters
- `engagement_id`

## Optional Parameters
None

## Response Mapping
- `endpoints`: data
- `endpoint_ids`: data[*].endpointId
- `endpoint_names`: data[*].endpointDisplayName

## Permissions
Roles: admin, developer, viewer

## Workflow Steps
### Workflow: list_endpoints
Multi-step workflow to list available endpoints/datacenters
- Step 1: get_engagement (engagement.get)
- Step 2: list_endpoints (endpoint.list) (depends on: engagement_id)
