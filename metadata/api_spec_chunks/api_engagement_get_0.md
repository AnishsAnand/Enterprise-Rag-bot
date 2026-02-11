# API Specification: engagement - get

**Resource:** engagement
**Operation:** get

## Endpoint
- **Method:** GET
- **URL:** https://ipcloud.tatacommunications.com/paasservice/paas/engagements
- **Auth:** Bearer token (from Keycloak)
- **Description:** Fetch engagement details for authenticated user. Response contains engagementId which is used in subsequent API calls.

## Required Parameters
None

## Optional Parameters
None

## Response Mapping
- `engagement_id`: data[0].id
- `engagement_name`: data[0].engagementName
- `customer_name`: data[0].customerName

## Permissions
Roles: admin, developer, viewer

## Workflow Steps
No workflow defined