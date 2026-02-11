# API Specification: engagement - list

**Resource:** engagement
**Operation:** list

## Endpoint
- **Method:** GET
- **URL:** https://ipcloud.tatacommunications.com/paasservice/paas/engagements
- **Auth:** Bearer token (from Keycloak)
- **Description:** List all engagements for authenticated user. Used for engagement selection when user has multiple accounts.

## Required Parameters
None

## Optional Parameters
None

## Response Mapping
- `engagements`: data (array of engagement objects)

## Permissions
Roles: admin, developer, viewer

## Workflow Steps
No workflow defined
