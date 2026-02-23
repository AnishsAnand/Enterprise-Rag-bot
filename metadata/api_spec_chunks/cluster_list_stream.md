# API Specification: cluster - list_stream

**Resource:** cluster
**Operation:** list_stream
**Aliases:** list clusters, show clusters, get clusters, cluster list, stream clusters, all clusters

## Endpoint
- **Method:** GET
- **URL:** `{BASE_URL_PAAS_SERVICE}/paas/{cluster_id}/clusterlist/stream`
- **Auth:** Bearer token (assumed - needs confirmation)
- **Description:** Retrieves a streaming list of all Kubernetes clusters with real-time status updates. Returns comprehensive cluster information including health status, resource utilization, node counts, and configuration details.

## Required Parameters
- `cluster_id` - Cluster identifier for context (type: path parameter, format: string)
  - **Note:** Despite appearing in URL, this may be a context parameter. Verify if it filters results or is required for auth context.

## Optional Parameters
None (to be documented based on actual API behavior)

## Response Mapping
**⚠️ Note: Actual API response needed to complete this section accurately**

Assumed mappings (to be verified):
- `cluster_ids`: data.clusters[*].id
- `cluster_names`: data.clusters[*].name
- `cluster_statuses`: data.clusters[*].status
- `cluster_versions`: data.clusters[*].version
- `node_counts`: data.clusters[*].node_count
- `pod_counts`: data.clusters[*].pod_count
- `namespaces`: data.clusters[*].namespace_count
- `cpu_usage`: data.clusters[*].metrics.cpu.usage_percent
- `memory_usage`: data.clusters[*].metrics.memory.usage_percent
- `health_status`: data.clusters[*].health.status
- `created_at`: data.clusters[*].created_at
- `region`: data.clusters[*].region
- `cluster_type`: data.clusters[*].type

## Response Example
**⚠️ Placeholder - Replace with actual API response**

```json
{
  "status": "success",
  "data": {
    "clusters": [
      {
        "id": "k8s-prod-001",
        "name": "production-cluster-east",
        "status": "running",
        "health": {
          "status": "healthy",
          "last_check": "2024-02-23T10:30:00Z"
        },
        "version": "1.28.5",
        "type": "kubernetes",
        "region": "us-east-1",
        "node_count": 15,
        "pod_count": 450,
        "namespace_count": 25,
        "metrics": {
          "cpu": {
            "total_cores": 60,
            "used_cores": 42,
            "usage_percent": 70
          },
          "memory": {
            "total_gb": 240,
            "used_gb": 168,
            "usage_percent": 70
          },
          "storage": {
            "total_gb": 2000,
            "used_gb": 1200,
            "usage_percent": 60
          }
        },
        "network": {
          "ingress_mbps": 1250,
          "egress_mbps": 980
        },
        "created_at": "2024-01-15T10:00:00Z",
        "updated_at": "2024-02-23T10:30:00Z"
      },
      {
        "id": "k8s-staging-001",
        "name": "staging-cluster",
        "status": "running",
        "health": {
          "status": "healthy",
          "last_check": "2024-02-23T10:30:00Z"
        },
        "version": "1.28.5",
        "type": "kubernetes",
        "region": "us-west-2",
        "node_count": 5,
        "pod_count": 120,
        "namespace_count": 10,
        "metrics": {
          "cpu": {
            "total_cores": 20,
            "used_cores": 8,
            "usage_percent": 40
          },
          "memory": {
            "total_gb": 80,
            "used_gb": 32,
            "usage_percent": 40
          },
          "storage": {
            "total_gb": 500,
            "used_gb": 200,
            "usage_percent": 40
          }
        },
        "network": {
          "ingress_mbps": 450,
          "egress_mbps": 320
        },
        "created_at": "2024-01-20T14:00:00Z",
        "updated_at": "2024-02-23T10:30:00Z"
      }
    ],
    "total_clusters": 2,
    "timestamp": "2024-02-23T10:30:00Z"
  }
}
```

## Response Fields Details

### Core Cluster Fields
- **id** - Unique cluster identifier
- **name** - Human-readable cluster name
- **status** - Cluster operational status (running, stopped, starting, error, maintenance)
- **type** - Cluster type (kubernetes, k8s, openshift)
- **version** - Kubernetes version
- **region** - Geographic region/availability zone
- **node_count** - Total number of nodes in cluster
- **pod_count** - Total number of running pods
- **namespace_count** - Number of namespaces
- **created_at** - Cluster creation timestamp
- **updated_at** - Last update timestamp

### Health Fields
- **health.status** - Overall health (healthy, degraded, unhealthy, unknown)
- **health.last_check** - Last health check timestamp

### Resource Metrics
- **metrics.cpu.total_cores** - Total CPU cores available
- **metrics.cpu.used_cores** - CPU cores in use
- **metrics.cpu.usage_percent** - CPU utilization percentage
- **metrics.memory.total_gb** - Total memory in GB
- **metrics.memory.used_gb** - Memory in use (GB)
- **metrics.memory.usage_percent** - Memory utilization percentage
- **metrics.storage.total_gb** - Total storage in GB
- **metrics.storage.used_gb** - Storage in use (GB)
- **metrics.storage.usage_percent** - Storage utilization percentage

### Network Metrics
- **network.ingress_mbps** - Incoming network traffic (Mbps)
- **network.egress_mbps** - Outgoing network traffic (Mbps)

## Permissions
**Roles:** To be documented (requires actual permission information)
Likely requires: Admin, Cluster Viewer, DevOps, SRE roles

## Workflow Steps

### Workflow: Cluster Discovery and Monitoring
**Main Workflow:**
- Step 1: **List Clusters** (cluster.list_stream) - Get all clusters with current status
- Step 2: **Select Cluster** - Choose cluster for detailed operations
- Step 3: **Get Cluster Details** (cluster.get_details) - Get comprehensive cluster information
- Step 4: **Get Node Metrics** (cluster.get_node_metrics) - Get detailed node-level metrics

### Monitoring Workflow
```
1. list_stream() → Get all clusters
2. Filter by status/health
3. For each cluster of interest:
   - get_cluster_details() → Detailed config
   - get_node_metrics() → Node-level data
   - get_all_namespaces() → Namespace info
```

## Usage Notes
- This endpoint may provide streaming updates (Server-Sent Events or WebSocket)
- Real-time data is ideal for monitoring dashboards
- Filter clusters by status, health, or resource usage for alerts
- Use for capacity planning and resource optimization
- Cluster list updates automatically as clusters are added/removed

## Kubernetes-Specific Features

### Cluster Health Monitoring
The API provides comprehensive health metrics essential for Kubernetes cluster management:
- **Node availability** - Track node status and readiness
- **Resource saturation** - Monitor CPU, memory, storage utilization
- **Pod health** - Overall pod count and health status
- **Network throughput** - Traffic patterns and bandwidth usage

### Resource Planning
Use cluster metrics for:
- **Capacity planning** - Identify when to scale clusters
- **Cost optimization** - Find underutilized clusters
- **Performance tuning** - Detect resource bottlenecks
- **HA configuration** - Ensure proper resource distribution

### Multi-Cluster Management
Essential for managing multiple Kubernetes clusters:
- Compare performance across environments (prod, staging, dev)
- Monitor regional deployments
- Track version consistency across clusters
- Identify clusters needing upgrades or maintenance

## Common Use Cases
1. **List all clusters**: "show me all clusters", "get cluster list", "list kubernetes clusters"
2. **Find running clusters**: "show running clusters", "active clusters", "which clusters are up"
3. **Check cluster health**: "cluster health status", "show healthy clusters", "any unhealthy clusters"
4. **Resource utilization**: "which clusters are using most CPU", "show cluster resource usage"
5. **Find by name**: "show production clusters", "get staging cluster", "clusters in us-east"
6. **Version tracking**: "which clusters need upgrade", "show kubernetes versions"
7. **Capacity planning**: "clusters with high utilization", "clusters under 50% usage"

## Query Interpretations
- "list clusters" → GET /clusterlist/stream
- "show all clusters" → GET /clusterlist/stream
- "get cluster status" → GET /clusterlist/stream (then filter by status)
- "running clusters" → GET /clusterlist/stream (filter status=running)
- "cluster health" → GET /clusterlist/stream (check health.status)

## Data Processing Examples

### Python Example
```python
import requests
from typing import List, Dict, Optional

def list_clusters(cluster_id: str, auth_token: str) -> Optional[Dict]:
    """
    Get list of all clusters with streaming updates.
    
    Args:
        cluster_id: Cluster context ID (verify if needed)
        auth_token: Bearer authentication token
        
    Returns:
        dict: Cluster data or None if failed
    """
    url = f"https://ipcloud.tatacommunications.com/paasservice/paas/{cluster_id}/clusterlist/stream"
    
    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        if data.get('status') == 'success':
            clusters = data['data']['clusters']
            print(f"Found {len(clusters)} clusters")
            return data['data']
        else:
            print(f"Failed: {data.get('message')}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"API error: {e}")
        return None

def filter_clusters_by_status(clusters: List[Dict], status: str) -> List[Dict]:
    """Filter clusters by operational status"""
    return [c for c in clusters if c['status'] == status]

def filter_clusters_by_health(clusters: List[Dict], health: str) -> List[Dict]:
    """Filter clusters by health status"""
    return [c for c in clusters if c.get('health', {}).get('status') == health]

def get_high_utilization_clusters(clusters: List[Dict], threshold: int = 80) -> List[Dict]:
    """Find clusters with high resource utilization"""
    high_util = []
    
    for cluster in clusters:
        metrics = cluster.get('metrics', {})
        cpu = metrics.get('cpu', {}).get('usage_percent', 0)
        memory = metrics.get('memory', {}).get('usage_percent', 0)
        
        if cpu >= threshold or memory >= threshold:
            high_util.append({
                'cluster': cluster,
                'cpu_usage': cpu,
                'memory_usage': memory
            })
    
    return high_util

# Usage
cluster_id = "k8s-context-001"  # Verify if this is needed
token = "your-token-here"

# Get all clusters
cluster_data = list_clusters(cluster_id, token)

if cluster_data:
    clusters = cluster_data['clusters']
    
    # Filter running clusters
    running = filter_clusters_by_status(clusters, 'running')
    print(f"Running clusters: {len(running)}")
    
    # Find healthy clusters
    healthy = filter_clusters_by_health(clusters, 'healthy')
    print(f"Healthy clusters: {len(healthy)}")
    
    # High utilization clusters
    high_util = get_high_utilization_clusters(clusters, threshold=80)
    for item in high_util:
        cluster = item['cluster']
        print(f"{cluster['name']}: CPU {item['cpu_usage']}%, Memory {item['memory_usage']}%")
```

### JavaScript Example
```javascript
async function listClusters(clusterId, authToken) {
    const url = `https://ipcloud.tatacommunications.com/paasservice/paas/${clusterId}/clusterlist/stream`;
    
    try {
        const response = await fetch(url, {
            method: 'GET',
            headers: {
                'Authorization': `Bearer ${authToken}`,
                'Content-Type': 'application/json'
            }
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        if (data.status === 'success') {
            console.log(`Found ${data.data.clusters.length} clusters`);
            return data.data;
        } else {
            console.error(`Failed: ${data.message}`);
            return null;
        }
    } catch (error) {
        console.error(`API error: ${error.message}`);
        return null;
    }
}

function analyzeClusterMetrics(clusters) {
    const analysis = {
        total: clusters.length,
        by_status: {},
        by_health: {},
        avg_cpu: 0,
        avg_memory: 0,
        total_nodes: 0,
        total_pods: 0
    };
    
    clusters.forEach(cluster => {
        // Count by status
        analysis.by_status[cluster.status] = (analysis.by_status[cluster.status] || 0) + 1;
        
        // Count by health
        const health = cluster.health?.status || 'unknown';
        analysis.by_health[health] = (analysis.by_health[health] || 0) + 1;
        
        // Sum metrics
        analysis.avg_cpu += cluster.metrics?.cpu?.usage_percent || 0;
        analysis.avg_memory += cluster.metrics?.memory?.usage_percent || 0;
        analysis.total_nodes += cluster.node_count || 0;
        analysis.total_pods += cluster.pod_count || 0;
    });
    
    // Calculate averages
    analysis.avg_cpu = Math.round(analysis.avg_cpu / clusters.length);
    analysis.avg_memory = Math.round(analysis.avg_memory / clusters.length);
    
    return analysis;
}

// Usage
const clusterId = 'k8s-context-001';
const token = 'your-token-here';

const clusterData = await listClusters(clusterId, token);

if (clusterData) {
    const analysis = analyzeClusterMetrics(clusterData.clusters);
    console.log('Cluster Analysis:', analysis);
    
    // Find production clusters
    const prodClusters = clusterData.clusters.filter(c => 
        c.name.toLowerCase().includes('prod')
    );
    console.log(`Production clusters: ${prodClusters.length}`);
}
```

## Integration Examples

### Cluster Monitoring Dashboard
```python
class ClusterMonitor:
    def __init__(self, cluster_id, auth_token):
        self.cluster_id = cluster_id
        self.auth_token = auth_token
        self.alert_thresholds = {
            'cpu': 85,
            'memory': 85,
            'storage': 90
        }
    
    def get_cluster_status(self):
        """Get current cluster status with analysis"""
        data = list_clusters(self.cluster_id, self.auth_token)
        
        if not data:
            return None
        
        clusters = data['clusters']
        
        return {
            'total_clusters': len(clusters),
            'running': len([c for c in clusters if c['status'] == 'running']),
            'healthy': len([c for c in clusters if c.get('health', {}).get('status') == 'healthy']),
            'alerts': self._check_alerts(clusters),
            'timestamp': data.get('timestamp')
        }
    
    def _check_alerts(self, clusters):
        """Check for resource utilization alerts"""
        alerts = []
        
        for cluster in clusters:
            metrics = cluster.get('metrics', {})
            cluster_name = cluster['name']
            
            # CPU alerts
            cpu_usage = metrics.get('cpu', {}).get('usage_percent', 0)
            if cpu_usage >= self.alert_thresholds['cpu']:
                alerts.append({
                    'cluster': cluster_name,
                    'type': 'cpu',
                    'value': cpu_usage,
                    'threshold': self.alert_thresholds['cpu'],
                    'severity': 'high' if cpu_usage >= 95 else 'medium'
                })
            
            # Memory alerts
            mem_usage = metrics.get('memory', {}).get('usage_percent', 0)
            if mem_usage >= self.alert_thresholds['memory']:
                alerts.append({
                    'cluster': cluster_name,
                    'type': 'memory',
                    'value': mem_usage,
                    'threshold': self.alert_thresholds['memory'],
                    'severity': 'high' if mem_usage >= 95 else 'medium'
                })
            
            # Storage alerts
            storage_usage = metrics.get('storage', {}).get('usage_percent', 0)
            if storage_usage >= self.alert_thresholds['storage']:
                alerts.append({
                    'cluster': cluster_name,
                    'type': 'storage',
                    'value': storage_usage,
                    'threshold': self.alert_thresholds['storage'],
                    'severity': 'critical' if storage_usage >= 95 else 'high'
                })
        
        return alerts
    
    def get_capacity_report(self):
        """Generate capacity planning report"""
        data = list_clusters(self.cluster_id, self.auth_token)
        
        if not data:
            return None
        
        clusters = data['clusters']
        
        report = {
            'total_capacity': {
                'cpu_cores': 0,
                'memory_gb': 0,
                'storage_gb': 0
            },
            'total_used': {
                'cpu_cores': 0,
                'memory_gb': 0,
                'storage_gb': 0
            },
            'utilization': {},
            'recommendations': []
        }
        
        for cluster in clusters:
            metrics = cluster.get('metrics', {})
            
            # Sum totals
            report['total_capacity']['cpu_cores'] += metrics.get('cpu', {}).get('total_cores', 0)
            report['total_capacity']['memory_gb'] += metrics.get('memory', {}).get('total_gb', 0)
            report['total_capacity']['storage_gb'] += metrics.get('storage', {}).get('total_gb', 0)
            
            report['total_used']['cpu_cores'] += metrics.get('cpu', {}).get('used_cores', 0)
            report['total_used']['memory_gb'] += metrics.get('memory', {}).get('used_gb', 0)
            report['total_used']['storage_gb'] += metrics.get('storage', {}).get('used_gb', 0)
            
            # Generate recommendations
            cpu_usage = metrics.get('cpu', {}).get('usage_percent', 0)
            if cpu_usage < 30:
                report['recommendations'].append(f"{cluster['name']}: Consider downsizing (CPU: {cpu_usage}%)")
            elif cpu_usage > 80:
                report['recommendations'].append(f"{cluster['name']}: Consider scaling up (CPU: {cpu_usage}%)")
        
        # Calculate overall utilization
        if report['total_capacity']['cpu_cores'] > 0:
            report['utilization']['cpu'] = round(
                (report['total_used']['cpu_cores'] / report['total_capacity']['cpu_cores']) * 100, 2
            )
        
        return report

# Usage
monitor = ClusterMonitor("k8s-context-001", "token")

# Get current status
status = monitor.get_cluster_status()
print(f"Status: {status['running']}/{status['total_clusters']} running")
print(f"Alerts: {len(status['alerts'])}")

for alert in status['alerts']:
    print(f"⚠️ {alert['cluster']}: {alert['type']} at {alert['value']}% (threshold: {alert['threshold']}%)")

# Capacity report
report = monitor.get_capacity_report()
print(f"
Overall CPU utilization: {report['utilization']['cpu']}%")
print("
Recommendations:")
for rec in report['recommendations']:
    print(f"  - {rec}")
```

## Related Operations
- `cluster.get_details` - Get detailed information for a specific cluster
- `cluster.get_node_metrics` - Get node-level metrics for a cluster
- `cluster.get_all_namespaces` - Get namespace list for a cluster
- Monitoring APIs - Integration with monitoring and alerting systems

## Error Handling
- **400 Bad Request:** Invalid cluster ID format - verify ID structure
- **401 Unauthorized:** Invalid or expired authentication token - refresh token
- **403 Forbidden:** Insufficient permissions to view clusters
- **404 Not Found:** Cluster context not found - verify cluster_id parameter
- **500 Internal Server Error:** Service error - retry with exponential backoff
- **503 Service Unavailable:** Service temporarily unavailable - implement retry logic

## Response Codes
**⚠️ To be documented - requires actual API testing**

Likely patterns:
- `0` or `success` - Request successful
- Non-zero or `error` - Request failed (check message field)

## Performance Notes
- Streaming endpoint may provide continuous updates
- Response time varies based on number of clusters
- Consider implementing client-side caching with periodic refresh
- For monitoring dashboards, poll every 30-60 seconds
- Rate limiting: To be documented

## Best Practices

### Efficient Cluster Filtering
```python
def get_clusters_by_criteria(cluster_id, token, **filters):
    """Get clusters matching multiple criteria"""
    data = list_clusters(cluster_id, token)
    
    if not data:
        return []
    
    clusters = data['clusters']
    
    # Apply filters
    if 'status' in filters:
        clusters = [c for c in clusters if c['status'] == filters['status']]
    
    if 'min_nodes' in filters:
        clusters = [c for c in clusters if c['node_count'] >= filters['min_nodes']]
    
    if 'region' in filters:
        clusters = [c for c in clusters if c['region'] == filters['region']]
    
    if 'max_cpu_usage' in filters:
        clusters = [c for c in clusters 
                   if c.get('metrics', {}).get('cpu', {}).get('usage_percent', 100) <= filters['max_cpu_usage']]
    
    return clusters

# Usage
running_low_util = get_clusters_by_criteria(
    cluster_id, 
    token,
    status='running',
    max_cpu_usage=50
)
```

### Real-Time Monitoring with Caching
```python
import time
from datetime import datetime, timedelta

class ClusterCache:
    def __init__(self, cluster_id, auth_token, refresh_interval=60):
        self.cluster_id = cluster_id
        self.auth_token = auth_token
        self.refresh_interval = refresh_interval
        self.cache = None
        self.last_update = None
    
    def get_clusters(self, force_refresh=False):
        """Get clusters with automatic cache refresh"""
        now = datetime.now()
        
        if (not self.cache or 
            not self.last_update or 
            (now - self.last_update) > timedelta(seconds=self.refresh_interval) or
            force_refresh):
            
            # Fetch fresh data
            self.cache = list_clusters(self.cluster_id, self.auth_token)
            self.last_update = now
        
        return self.cache
    
    def subscribe_to_changes(self, callback, interval=30):
        """Monitor for cluster changes"""
        previous_state = None
        
        while True:
            current_state = self.get_clusters(force_refresh=True)
            
            if previous_state and current_state:
                # Detect changes
                changes = self._detect_changes(previous_state, current_state)
                if changes:
                    callback(changes)
            
            previous_state = current_state
            time.sleep(interval)
    
    def _detect_changes(self, old, new):
        """Detect changes between states"""
        # Implementation depends on what changes to track
        pass

# Usage
cache = ClusterCache("k8s-context-001", "token", refresh_interval=60)

def on_cluster_change(changes):
    print(f"Cluster changes detected: {changes}")

# Start monitoring
# cache.subscribe_to_changes(on_cluster_change, interval=30)
```

## Metadata
- **Generated:** 2024-02-23T10:00:00Z
- **Source:** API endpoint analysis
- **API Version:** To be documented
- **Base Path:** {BASE_URL_PAAS_SERVICE}/paas
- **Additional notes:** ⚠️ This documentation is preliminary. Please provide actual API response for complete accuracy. Streaming behavior needs verification.
