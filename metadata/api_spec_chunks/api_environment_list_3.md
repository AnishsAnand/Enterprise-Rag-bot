# API Specification: environment - list

**Resource:** environment
**Operation:** list
**Aliases:** env, environment, environments

## Endpoint
- **Method:** GET
- **URL:** https://ipcloud.tatacommunications.com/portalservice/securityservice/environmentsperengagement/{ipc_engagement_id}
- **Auth:** Bearer token (from Keycloak)
- **Description:** List environments per engagement

## Required Parameters
- `ipc_engagement_id`

## Optional Parameters
None

## Response Mapping
- `environments`: data
- `environment_ids`: data[*].id
- `environment_names`: data[*].name

## Permissions
Roles: admin, developer, viewer

## Workflow Steps
### Workflow: list_environments
List environments for the engagement
- Step 1: get_paas_engagement (engagement.get)
- Step 2: convert_to_ipc_engagement (k8s_cluster.get_ipc_engagement) (depends on: engagement_id)
- Step 3: list_environments (environment.list) (depends on: ipc_engagement_id)
