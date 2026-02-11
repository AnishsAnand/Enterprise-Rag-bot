# API Specification: vm - list

**Resource:** vm
**Operation:** list
**Aliases:** vm, vms, virtual machine, virtual machines, instance, instances, server, servers

## Endpoint
- **Method:** GET
- **URL:** https://ipcloud.tatacommunications.com/portalservice/instances/vmlist/{ipc_engagement_id}
- **Auth:** Bearer token (from Keycloak)
- **Description:** List all virtual machines for the engagement

## Required Parameters
None

## Optional Parameters
- `endpoint`
- `zone`
- `department`

## Response Mapping
- `vms`: data.vmList
- `total`: data.vmList.length
- `last_synced`: data.lastSyncedAt

## Permissions
Roles: admin, developer, viewer

## Workflow Steps
### Workflow: list_vms
List all virtual machines for the engagement
- Step 1: get_paas_engagement (engagement.get)
- Step 2: convert_to_ipc_engagement (k8s_cluster.get_ipc_engagement) (depends on: engagement_id)
- Step 3: list_vms (.)
