# API Specification: load_balancer - get_virtual_services

**Resource:** load_balancer
**Operation:** get_virtual_services
**Aliases:** load balancer, load balancers, lb, lbs, loadbalancer, loadbalancers, vayu load balancer, network load balancer, application load balancer, alb, nlb

## Endpoint
- **Method:** GET
- **URL:** https://ipcloud.tatacommunications.com/networkservice/loadbalancer/list/virtualservices/{lbci}
- **Auth:** Bearer token (from Keycloak)
- **Description:** Get virtual services (VIPs, listeners) for a specific load balancer by LBCI

## Required Parameters
- `lbci`

## Optional Parameters
None

## Response Mapping
- `load_balancers`: data
- `total`: data.length
- `lb_names`: data[*].name
- `lb_status`: data[*].status
- `virtual_ips`: data[*].virtual_ip
- `lbci_values`: data[*].lbci

## Permissions
Roles: admin, developer, viewer

## Workflow Steps
### Workflow: list_load_balancers
List load balancers for the engagement using IPC engagement ID
- Step 1: get_paas_engagement (engagement.get)
- Step 2: convert_to_ipc_engagement (k8s_cluster.get_ipc_engagement) (depends on: engagement_id)
- Step 3: list_load_balancers (.)
- Step 4: format_response (.)

### Workflow: get_load_balancer_details
Get detailed configuration for a specific load balancer
- Step 1: extract_lbci (.)
- Step 2: get_lb_details (.)
- Step 3: format_details (.)

### Workflow: get_virtual_services
Get virtual services for a load balancer
- Step 1: extract_lbci (.)
- Step 2: get_virtual_services (.)
- Step 3: format_virtual_services (.)
