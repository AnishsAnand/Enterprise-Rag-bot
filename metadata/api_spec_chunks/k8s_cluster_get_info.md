# API Specification: k8s_cluster - get_info

**Resource:** k8s_cluster
**Operation:** get_info
**Aliases:** cluster info, get cluster info, cluster details, cluster information, show cluster

## Endpoint
- **Method:** GET
- **URL:** https://ipcloud.tatacommunications.com/paasservice/paas/cluster/{cluster_id}/getclusterinfo
- **Auth:** Bearer token (from Keycloak)
- **Description:** Retrieve detailed information about a specific Kubernetes cluster including configuration, status, nodes, and resources

## Required Parameters
- `cluster_id` - Unique identifier for the Kubernetes cluster

## Optional Parameters
None

## Response Mapping
- `cluster`: data
- `cluster_id`: data.id
- `cluster_name`: data.name
- `cluster_status`: data.status
- `cluster_type`: data.type
- `nodes`: data.nodes
- `node_count`: data.nodeCount
- `master_nodes`: data.masterNodes
- `worker_nodes`: data.workerNodes
- `cluster_version`: data.version
- `kubernetes_version`: data.kubernetesVersion
- `created_at`: data.createdAt
- `updated_at`: data.updatedAt
- `region`: data.region
- `zone`: data.zone
- `network_config`: data.networkConfig
- `storage_config`: data.storageConfig
- `resource_limits`: data.resourceLimits
- `endpoints`: data.endpoints
- `api_endpoint`: data.apiEndpoint
- `dashboard_url`: data.dashboardUrl

## Response Example
```json
{
  "data": {
    "id": "cluster-abc123",
    "name": "production-cluster-01",
    "status": "running",
    "type": "managed",
    "version": "1.28.0",
    "kubernetesVersion": "v1.28.0",
    "nodeCount": 5,
    "masterNodes": 3,
    "workerNodes": 2,
    "region": "asia-south",
    "zone": "mumbai-1",
    "createdAt": "2024-01-15T10:30:00Z",
    "updatedAt": "2024-02-13T08:45:00Z",
    "nodes": [
      {
        "id": "node-001",
        "name": "master-1",
        "type": "master",
        "status": "ready",
        "ip": "10.0.1.10"
      }
    ],
    "apiEndpoint": "https://cluster-abc123.api.ipcloud.com",
    "dashboardUrl": "https://dashboard.ipcloud.com/clusters/cluster-abc123",
    "networkConfig": {
      "podCIDR": "10.244.0.0/16",
      "serviceCIDR": "10.96.0.0/12"
    },
    "storageConfig": {
      "storageClass": "standard",
      "persistentVolumes": 10
    },
    "resourceLimits": {
      "maxPods": 110,
      "maxServices": 5000
    }
  },
  "success": true
}
```

## Permissions
Roles: admin, developer, viewer

## Workflow Steps
### Workflow: get_cluster_info
Retrieve complete cluster information
- Step 1: authenticate (auth.validate_token)
- Step 2: get_cluster_info (k8s_cluster.get_info) (depends on: cluster_id)

## Usage Notes
- This endpoint retrieves comprehensive information about a single cluster
- The `cluster_id` parameter is required and must be a valid cluster identifier
- Response includes node details, configuration, and resource information
- Use this endpoint to get full cluster state before performing operations
- Dashboard URL can be used for web-based cluster management

## Common Use Cases
1. **View cluster details**: "Show me information about cluster XYZ"
2. **Check cluster status**: "What is the status of my cluster?"
3. **Get node information**: "List nodes in the cluster"
4. **Verify configuration**: "Show cluster configuration"
5. **Monitor resources**: "What are the resource limits for this cluster?"

## Related Operations
- `k8s_cluster.list` - List all clusters
- `k8s_cluster.get_metrics` - Get cluster performance metrics
- `k8s_cluster.update` - Update cluster configuration
- `k8s_cluster.delete` - Delete a cluster

## Error Handling
- **401 Unauthorized:** Invalid or expired authentication token
- **403 Forbidden:** User does not have permission to view this cluster
- **404 Not Found:** Cluster with specified ID does not exist
- **500 Internal Server Error:** Server-side error retrieving cluster information

## Response Fields Details

### Status Values
- `running` - Cluster is operational
- `pending` - Cluster is being provisioned
- `updating` - Cluster configuration is being updated
- `error` - Cluster has encountered an error
- `stopped` - Cluster has been stopped

### Node Types
- `master` - Master/control plane node
- `worker` - Worker node for running workloads

### Node Status
- `ready` - Node is healthy and ready
- `not_ready` - Node is not ready
- `unknown` - Node status cannot be determined

## Metadata
- **Generated:** 2025-02-13T10:30:00Z
- **Source:** Dynamic API Spec Generator
- **API Version:** v1
- **Base Path:** /paasservice/paas