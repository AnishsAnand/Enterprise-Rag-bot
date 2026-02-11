# API Specification: firewall - list

**Resource:** firewall
**Operation:** list
**Aliases:** firewall, firewalls, fw, vayu firewall, network firewall

## Endpoint
- **Method:** POST
- **URL:** https://ipcloud.tatacommunications.com/networkservice/firewallconfig/details
- **Auth:** Bearer token (from Keycloak)
- **Description:** List firewalls for a specific endpoint

## Required Parameters
- `endpoints`

## Optional Parameters
- `variant`

## Response Mapping
- `firewalls`: data
- `total`: data.length

## Permissions
Roles: admin, developer, viewer

## Workflow Steps
### Workflow: list_firewalls
List firewalls for specified endpoints
- Step 1: get_paas_engagement (engagement.get)
- Step 2: convert_to_ipc_engagement (k8s_cluster.get_ipc_engagement) (depends on: engagement_id)
- Step 3: get_endpoints (endpoint.list) (depends on: engagement_id)
- Step 4: list_firewalls_per_endpoint (.)
