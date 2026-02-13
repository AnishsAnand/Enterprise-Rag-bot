# API Specification: k8s_cluster - list

**Resource:** k8s_cluster
**Operation:** list
**Aliases:** clusters, list clusters, show clusters, all clusters, cluster list, get clusters

## Endpoint
- **Method:** GET
- **URL:** https://ipcloud.tatacommunications.com/paasservice/paas/cluster/list
- **Auth:** Bearer token (from Keycloak)
- **Description:** List all Kubernetes clusters with their status, location, node count, and configuration details

## Required Parameters
None

## Optional Parameters
- `status` - Filter by cluster status (e.g., "Running", "Creating")
- `location` - Filter by location/endpoint (e.g., "EP_V2_BL", "EP_V2_CHN_AMB")
- `type` - Filter by cluster type (e.g., "APP")

## Response Mapping
- `status`: status
- `message`: message
- `response_code`: responseCode
- `clusters`: data
- `cluster_ids`: data[*].clusterId
- `cluster_names`: data[*].clusterName
- `cluster_statuses`: data[*].status
- `node_counts`: data[*].nodescount
- `locations`: data[*].location
- `display_names`: data[*].displayNameEndpoint
- `created_times`: data[*].createdTime
- `cluster_types`: data[*].type
- `ci_master_ids`: data[*].ciMasterId

## Response Example
```json
{
  "status": "success",
  "data": [
    {
      "nodescount": "3",
      "clusterName": "test",
      "displayNameEndpoint": "Bengaluru",
      "createdTime": 1752925643000,
      "location": "EP_V2_BL",
      "clusterId": 1267,
      "type": "APP",
      "ciMasterId": 337834,
      "status": "Running"
    },
    {
      "nodescount": "6",
      "clusterName": "aistdkubgpu01",
      "displayNameEndpoint": "Chennai-AMB",
      "createdTime": 1761278806000,
      "location": "EP_V2_CHN_AMB",
      "clusterId": 1501,
      "type": "APP",
      "ciMasterId": 349956,
      "status": "Running"
    },
    {
      "nodescount": "4",
      "clusterName": "aicloud-h100sxm",
      "displayNameEndpoint": "Chennai-AMB",
      "createdTime": 1766048324000,
      "location": "EP_V2_CHN_AMB",
      "clusterId": 1768,
      "type": "APP",
      "ciMasterId": 362339,
      "status": "Creating"
    }
  ],
  "message": "success",
  "responseCode": 0
}
```

## Response Fields Details

### Cluster Fields
- **clusterId** - Unique numeric identifier for the cluster
- **clusterName** - Human-readable cluster name
- **displayNameEndpoint** - Location display name (e.g., "Bengaluru", "Chennai-AMB", "Singapore East")
- **location** - Technical location code (e.g., "EP_V2_BL", "EP_V2_CHN_AMB", "EP_V2_SG_TCX")
- **nodescount** - Number of nodes in the cluster (as string)
- **type** - Cluster type (typically "APP")
- **status** - Current cluster status
- **ciMasterId** - Configuration item master ID
- **createdTime** - Unix timestamp in milliseconds

### Status Values
- `Running` - Cluster is operational and ready
- `Creating` - Cluster is being provisioned
- `Stopped` - Cluster has been stopped
- `Error` - Cluster has encountered an error
- `Updating` - Cluster configuration is being updated

### Location Codes
- `EP_V2_BL` - Bengaluru
- `EP_V2_CHN_AMB` - Chennai-AMB
- `EP_V2_MUM_BKC` - Mumbai-BKC
- `EP_V2_DEL` - Delhi
- `EP_V2_SG_TCX` - Singapore East
- `EP_V2_UKHB` - Highbridge (UK)
- `EP_GCC_DEL` - GCC Delhi

## Permissions
Roles: admin, developer, viewer

## Workflow Steps
### Workflow: list_clusters
List all Kubernetes clusters
- Step 1: authenticate (auth.validate_token)
- Step 2: list_clusters (k8s_cluster.list)

## Usage Notes
- Returns all clusters visible to the authenticated user
- Clusters are returned as an array in the `data` field
- `createdTime` is a Unix timestamp in milliseconds (divide by 1000 for seconds)
- `nodescount` is returned as a string, convert to integer if needed
- Response includes clusters in all statuses (Running, Creating, etc.)
- Filter by status or location for specific subsets

## Common Use Cases
1. **List all clusters**: "Show me all clusters"
2. **Check cluster count**: "How many clusters do I have?"
3. **Find by location**: "List clusters in Chennai"
4. **Check running clusters**: "Show only running clusters"
5. **Get cluster IDs**: "What are the cluster IDs?"
6. **Find by name**: "Find cluster named test"

## Data Processing Examples

### Get Total Cluster Count
```python
cluster_count = len(response['data'])
```

### Filter Running Clusters
```python
running = [c for c in response['data'] if c['status'] == 'Running']
```

### Group by Location
```python
from collections import defaultdict
by_location = defaultdict(list)
for cluster in response['data']:
    by_location[cluster['location']].append(cluster)
```

### Convert Created Time
```python
from datetime import datetime
for cluster in response['data']:
    timestamp = cluster['createdTime'] / 1000
    cluster['createdDate'] = datetime.fromtimestamp(timestamp)
```

### Get Total Node Count
```python
total_nodes = sum(int(c['nodescount']) for c in response['data'])
```

## Related Operations
- `k8s_cluster.get_info` - Get detailed information for specific cluster
- `k8s_cluster.get_metrics` - Get performance metrics for specific cluster
- `k8s_cluster.create` - Create new cluster
- `k8s_cluster.delete` - Delete cluster

## Error Handling
- **401 Unauthorized:** Invalid or expired authentication token
- **403 Forbidden:** User does not have permission to list clusters
- **500 Internal Server Error:** Server-side error retrieving clusters

## Response Codes
- `0` - Success
- Non-zero values indicate errors (check `message` field for details)

## Example Query Interpretations

### User Query → API Call
- "List all clusters" → GET /paasservice/paas/cluster/list
- "Show running clusters" → Filter response where status='Running'
- "Clusters in Mumbai" → Filter response where location contains 'MUM' or displayNameEndpoint='Mumbai-BKC'
- "How many nodes total?" → Sum all nodescount values

## Performance Notes
- This endpoint returns all clusters at once
- For large installations (100+ clusters), consider pagination if available
- Response is typically < 1MB for reasonable cluster counts
- Cache results if querying frequently within short time periods

## Metadata
- **Generated:** 2025-02-13T11:00:00Z
- **Source:** Dynamic API Spec Generator
- **Based on:** Live API response structure
- **API Version:** v1
- **Base Path:** /paasservice/paas