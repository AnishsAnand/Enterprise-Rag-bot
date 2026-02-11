# API Specification: business_unit - list

**Resource:** business_unit
**Operation:** list
**Aliases:** bu, business unit, business units, department, departments

## Endpoint
- **Method:** GET
- **URL:** https://ipcloud.tatacommunications.com/portalservice/securityservice/departments/{ipc_engagement_id}
- **Auth:** Bearer token (from Keycloak)
- **Description:** List business units (departments) for engagement with zone, environment, and VM counts

## Required Parameters
- `ipc_engagement_id`

## Optional Parameters
None

## Response Mapping
- `engagement`: data.engagement
- `departments`: data.department
- `department_ids`: data.department[*].id
- `department_names`: data.department[*].name

## Permissions
Roles: admin, developer, viewer

## Workflow Steps
### Workflow: list_business_units
List business units for the engagement
- Step 1: get_paas_engagement (engagement.get)
- Step 2: convert_to_ipc_engagement (k8s_cluster.get_ipc_engagement) (depends on: engagement_id)
- Step 3: list_business_units (business_unit.list) (depends on: ipc_engagement_id)
