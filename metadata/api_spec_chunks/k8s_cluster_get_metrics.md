# API Specification: k8s_cluster - get_metrics

**Resource:** k8s_cluster
**Operation:** get_metrics
**Aliases:** cluster metrics, get cluster metrics, cluster performance, show metrics, cluster stats, cluster statistics

## Endpoint
- **Method:** GET
- **URL:** https://ipcloud.tatacommunications.com/paasservice/paas/cluster/{cluster_id}/clustermetrics
- **Auth:** Bearer token (from Keycloak)
- **Description:** Retrieve real-time performance metrics and statistics for a Kubernetes cluster including CPU, memory, network, and storage utilization

## Required Parameters
- `cluster_id` - Unique identifier for the Kubernetes cluster

## Optional Parameters
- `time_range` - Time range for metrics (e.g., "1h", "24h", "7d") (default: "1h")
- `interval` - Data point interval (e.g., "1m", "5m", "1h") (default: "5m")

## Response Mapping
- `metrics`: data
- `cluster_id`: data.clusterId
- `timestamp`: data.timestamp
- `cpu_usage`: data.cpu.usage
- `cpu_limit`: data.cpu.limit
- `cpu_percentage`: data.cpu.percentage
- `memory_usage`: data.memory.usage
- `memory_limit`: data.memory.limit
- `memory_percentage`: data.memory.percentage
- `network_rx`: data.network.receivedBytes
- `network_tx`: data.network.transmittedBytes
- `storage_used`: data.storage.used
- `storage_total`: data.storage.total
- `storage_percentage`: data.storage.percentage
- `pod_count`: data.pods.count
- `pod_running`: data.pods.running
- `pod_pending`: data.pods.pending
- `pod_failed`: data.pods.failed
- `node_count`: data.nodes.total
- `node_ready`: data.nodes.ready
- `node_not_ready`: data.nodes.notReady

## Response Example
```json
{
  "data": {
    "clusterId": "cluster-abc123",
    "timestamp": "2025-02-13T10:30:00Z",
    "cpu": {
      "usage": 4.5,
      "limit": 16.0,
      "percentage": 28.13,
      "cores": {
        "used": 4.5,
        "total": 16
      }
    },
    "memory": {
      "usage": 12.5,
      "limit": 32.0,
      "percentage": 39.06,
      "unit": "GB",
      "bytes": {
        "used": 13421772800,
        "total": 34359738368
      }
    },
    "network": {
      "receivedBytes": 1073741824,
      "transmittedBytes": 2147483648,
      "receivedPackets": 1500000,
      "transmittedPackets": 2000000,
      "receivedMB": 1024,
      "transmittedMB": 2048
    },
    "storage": {
      "used": 100.5,
      "total": 500.0,
      "percentage": 20.1,
      "unit": "GB",
      "persistentVolumes": {
        "count": 10,
        "used": 85.3,
        "total": 250.0
      }
    },
    "pods": {
      "count": 45,
      "running": 42,
      "pending": 2,
      "failed": 1,
      "succeeded": 120
    },
    "nodes": {
      "total": 5,
      "ready": 5,
      "notReady": 0,
      "master": 3,
      "worker": 2
    },
    "services": {
      "count": 28,
      "loadBalancers": 5,
      "nodePort": 8,
      "clusterIP": 15
    },
    "health": {
      "status": "healthy",
      "score": 95,
      "issues": []
    },
    "alerts": {
      "critical": 0,
      "warning": 2,
      "info": 5
    }
  },
  "success": true,
  "timeRange": "1h",
  "interval": "5m"
}
```

## Permissions
Roles: admin, developer, viewer

## Workflow Steps
### Workflow: get_cluster_metrics
Retrieve cluster performance metrics
- Step 1: authenticate (auth.validate_token)
- Step 2: get_cluster_metrics (k8s_cluster.get_metrics) (depends on: cluster_id)

## Usage Notes
- Metrics are collected in real-time from the cluster monitoring system
- Default time range is 1 hour with 5-minute intervals
- Use `time_range` parameter for historical data (up to 30 days)
- CPU and memory values are aggregated across all nodes
- Network metrics show total cluster throughput
- Storage includes both node storage and persistent volumes
- Pod metrics include all namespaces unless filtered

## Common Use Cases
1. **Monitor cluster health**: "Show me cluster metrics"
2. **Check resource usage**: "What is the CPU usage of cluster XYZ?"
3. **Analyze performance**: "Show memory utilization trends"
4. **Capacity planning**: "How much storage is available?"
5. **Troubleshoot issues**: "Why is my cluster slow?"
6. **Alert investigation**: "Show recent metrics for cluster ABC"

## Metrics Interpretation

### CPU Metrics
- **usage**: Current CPU cores being used
- **limit**: Total available CPU cores
- **percentage**: Usage as percentage of limit
- Healthy range: < 80% for sustained workloads

### Memory Metrics
- **usage**: Current memory consumption in GB
- **limit**: Total available memory in GB
- **percentage**: Usage as percentage of limit
- Healthy range: < 85% to avoid OOM issues

### Network Metrics
- **receivedBytes/transmittedBytes**: Total network traffic
- Values in bytes, also provided as MB for convenience
- High values may indicate network-intensive workloads

### Storage Metrics
- **used/total**: Disk space utilization
- Includes both node storage and persistent volumes
- Monitor to prevent disk full conditions

### Pod Metrics
- **running**: Healthy pods serving traffic
- **pending**: Pods waiting for resources
- **failed**: Pods that terminated with errors
- High pending/failed counts indicate issues

### Node Metrics
- **ready**: Nodes accepting workloads
- **notReady**: Nodes with problems
- All nodes should be ready for healthy cluster

## Time Range Options
- `1h` - Last 1 hour (default)
- `6h` - Last 6 hours
- `24h` - Last 24 hours
- `7d` - Last 7 days
- `30d` - Last 30 days

## Interval Options
- `1m` - 1-minute intervals (high resolution)
- `5m` - 5-minute intervals (default)
- `15m` - 15-minute intervals
- `1h` - 1-hour intervals
- `1d` - 1-day intervals

## Related Operations
- `k8s_cluster.get_info` - Get cluster configuration details
- `k8s_cluster.list` - List all clusters
- `node.get_metrics` - Get metrics for specific node
- `pod.get_metrics` - Get metrics for specific pod

## Error Handling
- **401 Unauthorized:** Invalid or expired authentication token
- **403 Forbidden:** User does not have permission to view cluster metrics
- **404 Not Found:** Cluster with specified ID does not exist
- **422 Unprocessable Entity:** Invalid time_range or interval parameter
- **500 Internal Server Error:** Metrics collection service unavailable

## Alert Severity Levels
- **critical**: Immediate attention required (e.g., node down, out of memory)
- **warning**: Should be addressed soon (e.g., high CPU, disk space low)
- **info**: Informational notifications (e.g., scheduled maintenance)

## Health Status Values
- `healthy` - All systems operational (score > 90)
- `degraded` - Some issues present (score 70-90)
- `unhealthy` - Critical issues (score < 70)

## Performance Baselines

### Good Health Indicators
- CPU usage < 70%
- Memory usage < 80%
- All nodes ready
- < 5% failed pods
- No critical alerts
- Health score > 90

### Warning Signs
- CPU usage > 80%
- Memory usage > 85%
- Nodes not ready
- > 10% pending pods
- Multiple warnings
- Health score < 80

### Critical Issues
- CPU usage > 95%
- Memory usage > 95%
- Multiple nodes down
- High pod failure rate
- Critical alerts present
- Health score < 70

## Integration Examples

### Monitoring Dashboard
Use these metrics to build monitoring dashboards that show:
- Real-time resource utilization
- Historical trends
- Capacity forecasting
- Alert summaries

### Alerting Rules
Set up alerts based on thresholds:
- CPU > 85% for 10 minutes
- Memory > 90% for 5 minutes
- Nodes not ready > 0
- Failed pods > 5%

### Capacity Planning
Analyze trends to plan:
- When to scale up nodes
- Storage expansion needs
- Network bandwidth requirements

## Metadata
- **Generated:** 2025-02-13T10:30:00Z
- **Source:** Dynamic API Spec Generator
- **API Version:** v1
- **Base Path:** /paasservice/paas
- **Metrics Provider:** Prometheus/Grafana
- **Update Frequency:** Real-time (5-minute default interval)