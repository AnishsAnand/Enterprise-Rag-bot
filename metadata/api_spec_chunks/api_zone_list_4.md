# API Specification: zone - list

**Resource:** zone
**Operation:** list
**Aliases:** zone, zones, network zone, network zones, vlan, vlans, subnet, subnets

## Endpoint
- **Method:** GET
- **URL:** https://ipcloud.tatacommunications.com/portalservice/api/v1/{ipc_engagement_id}/zonelist
- **Auth:** Bearer token (from Keycloak)
- **Description:** List all zones (network segments) for the engagement with CIDR, hypervisors, and status

## Required Parameters
- `ipc_engagement_id`

## Optional Parameters
None

## Response Mapping
- `zones`: data
- `zone_ids`: data[*].zoneId
- `zone_names`: data[*].zoneName
- `departments`: data[*].departmentName
- `environments`: data[*].environmentName

## Permissions
Roles: admin, developer, viewer

## Workflow Steps
### Workflow: list_zones
List zones for the engagement with CIDR and status info
- Step 1: get_paas_engagement (engagement.get)
- Step 2: convert_to_ipc_engagement (k8s_cluster.get_ipc_engagement) (depends on: engagement_id)
- Step 3: list_zones (zone.list) (depends on: ipc_engagement_id)
