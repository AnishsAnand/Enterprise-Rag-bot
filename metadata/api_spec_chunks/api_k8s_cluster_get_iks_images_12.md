# API Specification: k8s_cluster - get_iks_images

**Resource:** k8s_cluster
**Operation:** get_iks_images

## Endpoint
- **Method:** GET
- **URL:** https://ipcloud.tatacommunications.com/paasservice/paas/getTemplatesByEngagement/{ipc_engagement_id}
- **Auth:** Bearer token (from Keycloak)
- **Description:** Get IKS images with datacenter options and k8s versions. Requires ipc_engagement_id from get_ipc_engagement.

## Required Parameters
- `ipc_engagement_id`

## Optional Parameters
None

## Permissions
Roles: 

## Workflow Steps
### Workflow: list_clusters
Multi-step workflow to list Kubernetes clusters
- Step 1: get_engagement (engagement.get)
- Step 2: get_endpoints (endpoint.list) (depends on: engagement_id)
- Step 3: list_clusters (k8s_cluster.list) (depends on: engagement_id, selected_endpoints)

### Workflow: create_cluster_customer
Customer workflow to create a new Kubernetes cluster (simplified)
- Step 1: collect_cluster_name (.)
- Step 2: select_datacenter (.)
- Step 3: select_k8s_version (.) (depends on: datacenter)
- Step 4: select_cni_driver (.) (depends on: datacenter, k8sVersion)
- Step 5: select_business_unit (.)
- Step 6: select_environment (.) (depends on: businessUnit)
- Step 7: select_zone (.) (depends on: businessUnit, environment)
- Step 8: select_operating_system (.) (depends on: zone, k8sVersion)
- Step 9: collect_worker_pool_name (.)
- Step 10: select_node_type (.) (depends on: operatingSystem)
- Step 11: select_flavor (.) (depends on: operatingSystem, workerNodePool.nodeType)
- Step 12: collect_replica_count (.)
- Step 13: ask_autoscaling (.)
- Step 14: collect_max_replicas (.)
- Step 15: collect_tags (.)
- Step 16: build_payload (.)
- Step 17: confirm_and_create (.)
