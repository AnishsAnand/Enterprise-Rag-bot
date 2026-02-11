# API Specification: managed_service - list

**Resource:** managed_service
**Operation:** list
**Aliases:** managed service, managed services

## Endpoint
- **Method:** POST
- **URL:** https://ipcloud.tatacommunications.com/paasservice/api/v1/paas/listManagedServices/{serviceType}
- **Auth:** Bearer token (from Keycloak)
- **Description:** List managed services by type (Kafka, GitLab, etc.). Supports streaming response.

## Required Parameters
- `serviceType`
- `engagement_id`
- `endpoints`

## Optional Parameters
None

## Response Mapping
- `services`: data
- `total`: data.length

## Permissions
Roles: admin, developer, viewer

## Workflow Steps
### Workflow: list_managed_services
Multi-step workflow to list managed services by type
- Step 1: get_engagement (engagement.get)
- Step 2: get_ipc_engagement (k8s_cluster.get_ipc_engagement) (depends on: engagement_id)
- Step 3: get_endpoints (endpoint.list) (depends on: engagement_id)
- Step 4: list_managed_services (managed_service.list) (depends on: ipc_engagement_id, selected_endpoints, serviceType)
