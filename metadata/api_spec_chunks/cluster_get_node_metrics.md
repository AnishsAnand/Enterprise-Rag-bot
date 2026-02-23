# API Specification: cluster - get_node_metrics

**Resource:** cluster
**Operation:** get_node_metrics
**Aliases:** node metrics, cluster metrics, get node stats, node performance, node monitoring

## Endpoint
- **Method:** GET
- **URL:** `{BASE_URL_PAAS_SERVICE}/paas/cluster/{cluster_id}/clusternodemetrics`
- **Auth:** Bearer token (assumed - needs confirmation)
- **Description:** Retrieves detailed performance metrics for all nodes in a Kubernetes cluster. Returns real-time CPU, memory, disk, and network metrics for each node, essential for monitoring, capacity planning, and troubleshooting.

## Required Parameters
- `cluster_id` - Unique cluster identifier (type: path parameter, format: string)

## Optional Parameters
None (to be documented based on actual API behavior)

## Response Mapping
**⚠️ Note: Actual API response needed to complete this section accurately**

Assumed mappings (to be verified):
- `cluster_id`: data.cluster_id
- `nodes`: data.nodes
- `node_names`: data.nodes[*].name
- `node_statuses`: data.nodes[*].status
- `node_roles`: data.nodes[*].role
- `cpu_usage`: data.nodes[*].metrics.cpu.usage_percent
- `cpu_cores`: data.nodes[*].metrics.cpu.cores
- `memory_usage`: data.nodes[*].metrics.memory.usage_percent
- `memory_total`: data.nodes[*].metrics.memory.total_gb
- `disk_usage`: data.nodes[*].metrics.disk.usage_percent
- `disk_total`: data.nodes[*].metrics.disk.total_gb
- `network_rx`: data.nodes[*].metrics.network.rx_mbps
- `network_tx`: data.nodes[*].metrics.network.tx_mbps
- `pod_count`: data.nodes[*].pods.count
- `pod_capacity`: data.nodes[*].pods.capacity
- `node_ip`: data.nodes[*].ip_address
- `uptime`: data.nodes[*].uptime_seconds
- `timestamp`: data.timestamp

## Response Example
**⚠️ Placeholder - Replace with actual API response**

```json
{
  "status": "success",
  "data": {
    "cluster_id": "k8s-prod-001",
    "timestamp": "2024-02-23T10:30:00Z",
    "nodes": [
      {
        "name": "k8s-master-1",
        "role": "master",
        "status": "Ready",
        "ip_address": "10.0.1.10",
        "instance_type": "m5.xlarge",
        "uptime_seconds": 2592000,
        "kubernetes_version": "v1.28.5",
        "container_runtime": "containerd://1.7.2",
        "metrics": {
          "cpu": {
            "cores": 4,
            "usage_cores": 1.2,
            "usage_percent": 30,
            "requests_cores": 0.8,
            "limits_cores": 2.0
          },
          "memory": {
            "total_gb": 16,
            "used_gb": 6.4,
            "usage_percent": 40,
            "available_gb": 9.6,
            "requests_gb": 4.0,
            "limits_gb": 8.0
          },
          "disk": {
            "total_gb": 100,
            "used_gb": 35,
            "usage_percent": 35,
            "available_gb": 65,
            "iops": 1500
          },
          "network": {
            "rx_mbps": 125.5,
            "tx_mbps": 98.2,
            "rx_errors": 0,
            "tx_errors": 0
          }
        },
        "pods": {
          "count": 28,
          "capacity": 110,
          "usage_percent": 25
        },
        "conditions": {
          "ready": true,
          "memory_pressure": false,
          "disk_pressure": false,
          "pid_pressure": false,
          "network_unavailable": false
        },
        "labels": {
          "node-role.kubernetes.io/master": "",
          "topology.kubernetes.io/zone": "us-east-1a"
        },
        "taints": [
          {
            "key": "node-role.kubernetes.io/master",
            "effect": "NoSchedule"
          }
        ]
      },
      {
        "name": "k8s-worker-1",
        "role": "worker",
        "status": "Ready",
        "ip_address": "10.0.2.10",
        "instance_type": "m5.2xlarge",
        "uptime_seconds": 2592000,
        "kubernetes_version": "v1.28.5",
        "container_runtime": "containerd://1.7.2",
        "metrics": {
          "cpu": {
            "cores": 8,
            "usage_cores": 5.6,
            "usage_percent": 70,
            "requests_cores": 4.5,
            "limits_cores": 7.0
          },
          "memory": {
            "total_gb": 32,
            "used_gb": 22.4,
            "usage_percent": 70,
            "available_gb": 9.6,
            "requests_gb": 18.0,
            "limits_gb": 28.0
          },
          "disk": {
            "total_gb": 200,
            "used_gb": 120,
            "usage_percent": 60,
            "available_gb": 80,
            "iops": 3000
          },
          "network": {
            "rx_mbps": 450.8,
            "tx_mbps": 380.4,
            "rx_errors": 0,
            "tx_errors": 0
          }
        },
        "pods": {
          "count": 85,
          "capacity": 110,
          "usage_percent": 77
        },
        "conditions": {
          "ready": true,
          "memory_pressure": false,
          "disk_pressure": false,
          "pid_pressure": false,
          "network_unavailable": false
        },
        "labels": {
          "node-role.kubernetes.io/worker": "",
          "topology.kubernetes.io/zone": "us-east-1a"
        },
        "taints": []
      }
    ],
    "summary": {
      "total_nodes": 2,
      "ready_nodes": 2,
      "total_cpu_cores": 12,
      "total_memory_gb": 48,
      "avg_cpu_usage_percent": 50,
      "avg_memory_usage_percent": 55,
      "total_pods": 113,
      "total_pod_capacity": 220
    }
  }
}
```

## Response Fields Details

### Node Core Fields
- **name** - Node name/hostname
- **role** - Node role (master, worker, control-plane)
- **status** - Node status (Ready, NotReady, Unknown, SchedulingDisabled)
- **ip_address** - Node IP address
- **instance_type** - Cloud instance type
- **uptime_seconds** - Node uptime in seconds
- **kubernetes_version** - Kubelet version
- **container_runtime** - Container runtime version

### CPU Metrics
- **metrics.cpu.cores** - Total CPU cores
- **metrics.cpu.usage_cores** - CPU cores in use
- **metrics.cpu.usage_percent** - CPU utilization percentage
- **metrics.cpu.requests_cores** - Total requested CPU
- **metrics.cpu.limits_cores** - Total CPU limits

### Memory Metrics
- **metrics.memory.total_gb** - Total memory in GB
- **metrics.memory.used_gb** - Used memory in GB
- **metrics.memory.usage_percent** - Memory utilization percentage
- **metrics.memory.available_gb** - Available memory in GB
- **metrics.memory.requests_gb** - Total requested memory
- **metrics.memory.limits_gb** - Total memory limits

### Disk Metrics
- **metrics.disk.total_gb** - Total disk space in GB
- **metrics.disk.used_gb** - Used disk space in GB
- **metrics.disk.usage_percent** - Disk utilization percentage
- **metrics.disk.available_gb** - Available disk space
- **metrics.disk.iops** - Disk I/O operations per second

### Network Metrics
- **metrics.network.rx_mbps** - Receive throughput (Mbps)
- **metrics.network.tx_mbps** - Transmit throughput (Mbps)
- **metrics.network.rx_errors** - Receive error count
- **metrics.network.tx_errors** - Transmit error count

### Pod Information
- **pods.count** - Current pod count on node
- **pods.capacity** - Maximum pod capacity
- **pods.usage_percent** - Pod capacity utilization

### Node Conditions
- **conditions.ready** - Node is ready to accept pods
- **conditions.memory_pressure** - Memory pressure detected
- **conditions.disk_pressure** - Disk pressure detected
- **conditions.pid_pressure** - PID pressure detected
- **conditions.network_unavailable** - Network issues detected

### Labels and Taints
- **labels** - Node labels (key-value pairs)
- **taints** - Node taints for pod scheduling

## Permissions
**Roles:** To be documented (requires actual permission information)
Likely requires: Admin, Cluster Viewer, DevOps, SRE, Monitoring roles

## Workflow Steps

### Workflow: Node Performance Monitoring
**Prerequisites:**
- Step 1: **List Clusters** (cluster.list_stream) - Get cluster ID

**Main Workflow:**
- Step 2: **Get Node Metrics** (cluster.get_node_metrics) - Get detailed node metrics
- Step 3: **Analyze Metrics** - Process metrics for monitoring/alerting
- Step 4: **Take Action** - Scale, rebalance, or troubleshoot based on metrics

### Monitoring Workflow
```
1. list_stream() → Get cluster IDs
2. For each cluster:
   - get_node_metrics(cluster_id) → Real-time node metrics
   - Analyze CPU, memory, disk usage
   - Check for pressure conditions
   - Alert if thresholds exceeded
3. Scale or rebalance as needed
```

## Usage Notes
- Real-time metrics ideal for monitoring dashboards
- Use for capacity planning and resource optimization
- Check conditions for node health issues
- Monitor trends over time for predictive scaling
- Identify nodes under high load for workload rebalancing

## Kubernetes-Specific Features

### Node Health Monitoring
Essential for Kubernetes cluster health:
- **Ready Status** - Node ready to accept workloads
- **Pressure Conditions** - Early warning for resource exhaustion
  - Memory pressure → Risk of OOM kills
  - Disk pressure → Risk of disk full
  - PID pressure → Process limit approaching
- **Network Status** - Connectivity issues

### Capacity Planning
Use node metrics for:
- **Scale-up Decisions** - When to add nodes
- **Scale-down Opportunities** - Underutilized nodes to remove
- **Resource Requests/Limits** - Optimize pod resource settings
- **Node Pool Sizing** - Right-size node instance types

### Workload Distribution
Analyze metrics to:
- **Identify Hot Nodes** - Nodes under high load
- **Rebalance Workloads** - Move pods from overloaded nodes
- **Pod Placement** - Use metrics for scheduling decisions
- **Drain Operations** - Safely drain nodes for maintenance

### Performance Troubleshooting
Metrics help identify:
- **CPU Bottlenecks** - High CPU utilization
- **Memory Issues** - Memory pressure, OOM conditions
- **Disk Problems** - Disk I/O saturation, space issues
- **Network Congestion** - High traffic, errors

## Common Use Cases
1. **Node health check**: "node metrics", "show node status", "node health"
2. **Resource monitoring**: "cpu usage per node", "memory usage", "disk space"
3. **Capacity planning**: "node capacity", "available resources", "utilization"
4. **Performance issues**: "high cpu nodes", "nodes under pressure", "slow nodes"
5. **Load balancing**: "pod distribution", "which nodes are busy", "underutilized nodes"
6. **Troubleshooting**: "node problems", "why is node slow", "node conditions"

## Query Interpretations
- "get node metrics for {cluster_id}" → GET /cluster/{cluster_id}/clusternodemetrics
- "show node performance {cluster_id}" → GET /cluster/{cluster_id}/clusternodemetrics
- "{cluster_id} node stats" → GET /cluster/{cluster_id}/clusternodemetrics

## Data Processing Examples

### Python Example
```python
import requests
from typing import List, Dict, Optional

def get_node_metrics(cluster_id: str, auth_token: str) -> Optional[Dict]:
    """
    Get detailed node metrics for a cluster.
    
    Args:
        cluster_id: Cluster identifier
        auth_token: Bearer authentication token
        
    Returns:
        dict: Node metrics data or None if failed
    """
    url = f"https://ipcloud.tatacommunications.com/paasservice/paas/cluster/{cluster_id}/clusternodemetrics"
    
    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        if data.get('status') == 'success':
            metrics = data['data']
            print(f"Cluster {cluster_id}: {metrics['summary']['total_nodes']} nodes")
            print(f"Avg CPU: {metrics['summary']['avg_cpu_usage_percent']}%, Avg Memory: {metrics['summary']['avg_memory_usage_percent']}%")
            return metrics
        else:
            print(f"Failed: {data.get('message')}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"API error: {e}")
        return None

def find_high_cpu_nodes(nodes: List[Dict], threshold: int = 80) -> List[Dict]:
    """Find nodes with high CPU usage"""
    high_cpu = []
    
    for node in nodes:
        cpu_usage = node['metrics']['cpu']['usage_percent']
        if cpu_usage >= threshold:
            high_cpu.append({
                'name': node['name'],
                'role': node['role'],
                'cpu_usage': cpu_usage,
                'memory_usage': node['metrics']['memory']['usage_percent'],
                'pod_count': node['pods']['count']
            })
    
    return sorted(high_cpu, key=lambda x: x['cpu_usage'], reverse=True)

def find_nodes_with_pressure(nodes: List[Dict]) -> List[Dict]:
    """Find nodes experiencing pressure conditions"""
    pressure_nodes = []
    
    for node in nodes:
        conditions = node.get('conditions', {})
        pressures = []
        
        if conditions.get('memory_pressure'):
            pressures.append('memory')
        if conditions.get('disk_pressure'):
            pressures.append('disk')
        if conditions.get('pid_pressure'):
            pressures.append('pid')
        
        if pressures:
            pressure_nodes.append({
                'name': node['name'],
                'role': node['role'],
                'pressures': pressures,
                'status': node['status']
            })
    
    return pressure_nodes

def calculate_available_capacity(nodes: List[Dict]) -> Dict:
    """Calculate available cluster capacity"""
    total_cpu = 0
    used_cpu = 0
    total_memory = 0
    used_memory = 0
    total_pod_capacity = 0
    current_pods = 0
    
    for node in nodes:
        if node['status'] == 'Ready':
            metrics = node['metrics']
            total_cpu += metrics['cpu']['cores']
            used_cpu += metrics['cpu']['usage_cores']
            total_memory += metrics['memory']['total_gb']
            used_memory += metrics['memory']['used_gb']
            total_pod_capacity += node['pods']['capacity']
            current_pods += node['pods']['count']
    
    return {
        'cpu': {
            'total_cores': total_cpu,
            'used_cores': round(used_cpu, 2),
            'available_cores': round(total_cpu - used_cpu, 2),
            'usage_percent': round((used_cpu / total_cpu * 100), 2) if total_cpu > 0 else 0
        },
        'memory': {
            'total_gb': total_memory,
            'used_gb': round(used_memory, 2),
            'available_gb': round(total_memory - used_memory, 2),
            'usage_percent': round((used_memory / total_memory * 100), 2) if total_memory > 0 else 0
        },
        'pods': {
            'capacity': total_pod_capacity,
            'current': current_pods,
            'available': total_pod_capacity - current_pods,
            'usage_percent': round((current_pods / total_pod_capacity * 100), 2) if total_pod_capacity > 0 else 0
        }
    }

# Usage
cluster_id = "k8s-prod-001"
token = "your-token-here"

# Get node metrics
metrics_data = get_node_metrics(cluster_id, token)

if metrics_data:
    nodes = metrics_data['nodes']
    
    # Find high CPU nodes
    high_cpu = find_high_cpu_nodes(nodes, threshold=80)
    if high_cpu:
        print("
⚠️ High CPU Nodes:")
        for node in high_cpu:
            print(f"  - {node['name']} ({node['role']}): CPU {node['cpu_usage']}%, {node['pod_count']} pods")
    
    # Check for pressure
    pressure = find_nodes_with_pressure(nodes)
    if pressure:
        print("
⚠️ Nodes Under Pressure:")
        for node in pressure:
            print(f"  - {node['name']}: {', '.join(node['pressures'])} pressure")
    
    # Available capacity
    capacity = calculate_available_capacity(nodes)
    print(f"
Cluster Capacity:")
    print(f"  CPU: {capacity['cpu']['available_cores']:.1f} / {capacity['cpu']['total_cores']} cores available ({capacity['cpu']['usage_percent']}% used)")
    print(f"  Memory: {capacity['memory']['available_gb']:.1f} / {capacity['memory']['total_gb']} GB available ({capacity['memory']['usage_percent']}% used)")
    print(f"  Pods: {capacity['pods']['available']} / {capacity['pods']['capacity']} capacity available ({capacity['pods']['usage_percent']}% used)")
```

### JavaScript Example
```javascript
async function getNodeMetrics(clusterId, authToken) {
    const url = `https://ipcloud.tatacommunications.com/paasservice/paas/cluster/${clusterId}/clusternodemetrics`;
    
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
            const metrics = data.data;
            console.log(`Cluster ${clusterId}: ${metrics.summary.total_nodes} nodes`);
            console.log(`Avg CPU: ${metrics.summary.avg_cpu_usage_percent}%, Avg Memory: ${metrics.summary.avg_memory_usage_percent}%`);
            return metrics;
        } else {
            console.error(`Failed: ${data.message}`);
            return null;
        }
    } catch (error) {
        console.error(`API error: ${error.message}`);
        return null;
    }
}

function analyzeNodeDistribution(nodes) {
    const distribution = {
        by_role: {},
        by_status: {},
        high_utilization: [],
        low_utilization: []
    };
    
    nodes.forEach(node => {
        // Count by role
        distribution.by_role[node.role] = (distribution.by_role[node.role] || 0) + 1;
        
        // Count by status
        distribution.by_status[node.status] = (distribution.by_status[node.status] || 0) + 1;
        
        // Categorize by utilization
        const cpu = node.metrics.cpu.usage_percent;
        const memory = node.metrics.memory.usage_percent;
        const avgUtil = (cpu + memory) / 2;
        
        if (avgUtil > 80) {
            distribution.high_utilization.push({
                name: node.name,
                cpu,
                memory,
                avg: Math.round(avgUtil)
            });
        } else if (avgUtil < 30) {
            distribution.low_utilization.push({
                name: node.name,
                cpu,
                memory,
                avg: Math.round(avgUtil)
            });
        }
    });
    
    return distribution;
}

function generateScalingRecommendations(nodes) {
    const recommendations = [];
    
    // Analyze overall utilization
    const readyNodes = nodes.filter(n => n.status === 'Ready');
    
    let totalCpu = 0;
    let usedCpu = 0;
    let totalMemory = 0;
    let usedMemory = 0;
    
    readyNodes.forEach(node => {
        totalCpu += node.metrics.cpu.cores;
        usedCpu += node.metrics.cpu.usage_cores;
        totalMemory += node.metrics.memory.total_gb;
        usedMemory += node.metrics.memory.used_gb;
    });
    
    const cpuUtil = (usedCpu / totalCpu) * 100;
    const memUtil = (usedMemory / totalMemory) * 100;
    
    // Generate recommendations
    if (cpuUtil > 85 || memUtil > 85) {
        recommendations.push({
            type: 'scale-up',
            reason: `High utilization (CPU: ${cpuUtil.toFixed(1)}%, Memory: ${memUtil.toFixed(1)}%)`,
            action: 'Add more worker nodes'
        });
    }
    
    if (cpuUtil < 30 && memUtil < 30 && readyNodes.length > 3) {
        recommendations.push({
            type: 'scale-down',
            reason: `Low utilization (CPU: ${cpuUtil.toFixed(1)}%, Memory: ${memUtil.toFixed(1)}%)`,
            action: 'Consider reducing worker nodes'
        });
    }
    
    // Check for unbalanced nodes
    const highUtilNodes = nodes.filter(n => 
        n.metrics.cpu.usage_percent > 90 || n.metrics.memory.usage_percent > 90
    );
    
    if (highUtilNodes.length > 0 && cpuUtil < 70) {
        recommendations.push({
            type: 'rebalance',
            reason: `${highUtilNodes.length} nodes over 90% while cluster average is ${cpuUtil.toFixed(1)}%`,
            action: 'Rebalance workload distribution'
        });
    }
    
    return recommendations;
}

// Usage
const clusterId = 'k8s-prod-001';
const token = 'your-token-here';

const metricsData = await getNodeMetrics(clusterId, token);

if (metricsData) {
    // Analyze distribution
    const distribution = analyzeNodeDistribution(metricsData.nodes);
    console.log('Node Distribution:', distribution);
    
    // Get recommendations
    const recommendations = generateScalingRecommendations(metricsData.nodes);
    console.log('
Scaling Recommendations:');
    recommendations.forEach(rec => {
        console.log(`  [${rec.type.toUpperCase()}] ${rec.reason}`);
        console.log(`    → ${rec.action}`);
    });
}
```

## Integration Examples

### Node Monitoring and Alerting System
```python
class NodeMonitor:
    def __init__(self, cluster_id, auth_token):
        self.cluster_id = cluster_id
        self.auth_token = auth_token
        self.alert_thresholds = {
            'cpu': 85,
            'memory': 85,
            'disk': 90,
            'pod_capacity': 90
        }
    
    def check_node_health(self):
        """Comprehensive node health check"""
        metrics_data = get_node_metrics(self.cluster_id, self.auth_token)
        
        if not metrics_data:
            return None
        
        nodes = metrics_data['nodes']
        
        alerts = []
        warnings = []
        info = []
        
        for node in nodes:
            node_alerts = self._check_node(node)
            alerts.extend(node_alerts['critical'])
            warnings.extend(node_alerts['warning'])
            info.extend(node_alerts['info'])
        
        return {
            'cluster_id': self.cluster_id,
            'timestamp': metrics_data['timestamp'],
            'total_nodes': len(nodes),
            'ready_nodes': len([n for n in nodes if n['status'] == 'Ready']),
            'alerts': {
                'critical': alerts,
                'warning': warnings,
                'info': info
            },
            'summary': metrics_data['summary']
        }
    
    def _check_node(self, node):
        """Check individual node for issues"""
        alerts = {'critical': [], 'warning': [], 'info': []}
        
        node_name = node['name']
        metrics = node['metrics']
        conditions = node.get('conditions', {})
        
        # Critical: Node not ready
        if node['status'] != 'Ready':
            alerts['critical'].append({
                'node': node_name,
                'type': 'node_status',
                'message': f"Node not ready: {node['status']}"
            })
        
        # Critical: Pressure conditions
        if conditions.get('memory_pressure'):
            alerts['critical'].append({
                'node': node_name,
                'type': 'memory_pressure',
                'message': 'Memory pressure detected'
            })
        
        if conditions.get('disk_pressure'):
            alerts['critical'].append({
                'node': node_name,
                'type': 'disk_pressure',
                'message': 'Disk pressure detected'
            })
        
        # Warning: High resource utilization
        cpu_usage = metrics['cpu']['usage_percent']
        if cpu_usage >= self.alert_thresholds['cpu']:
            severity = 'critical' if cpu_usage >= 95 else 'warning'
            alerts[severity].append({
                'node': node_name,
                'type': 'high_cpu',
                'message': f"High CPU usage: {cpu_usage}%",
                'value': cpu_usage
            })
        
        memory_usage = metrics['memory']['usage_percent']
        if memory_usage >= self.alert_thresholds['memory']:
            severity = 'critical' if memory_usage >= 95 else 'warning'
            alerts[severity].append({
                'node': node_name,
                'type': 'high_memory',
                'message': f"High memory usage: {memory_usage}%",
                'value': memory_usage
            })
        
        disk_usage = metrics['disk']['usage_percent']
        if disk_usage >= self.alert_thresholds['disk']:
            severity = 'critical' if disk_usage >= 95 else 'warning'
            alerts[severity].append({
                'node': node_name,
                'type': 'high_disk',
                'message': f"High disk usage: {disk_usage}%",
                'value': disk_usage
            })
        
        # Info: Pod capacity approaching limit
        pod_usage = node['pods']['usage_percent']
        if pod_usage >= self.alert_thresholds['pod_capacity']:
            alerts['info'].append({
                'node': node_name,
                'type': 'pod_capacity',
                'message': f"Pod capacity at {pod_usage}%",
                'value': pod_usage
            })
        
        return alerts

# Usage
monitor = NodeMonitor("k8s-prod-001", "token")
health = monitor.check_node_health()

if health:
    print(f"Cluster Health Report - {health['timestamp']}")
    print(f"Nodes: {health['ready_nodes']}/{health['total_nodes']} ready")
    
    if health['alerts']['critical']:
        print(f"
🔴 CRITICAL ALERTS ({len(health['alerts']['critical'])}):")
        for alert in health['alerts']['critical']:
            print(f"  - {alert['node']}: {alert['message']}")
    
    if health['alerts']['warning']:
        print(f"
⚠️ WARNINGS ({len(health['alerts']['warning'])}):")
        for alert in health['alerts']['warning']:
            print(f"  - {alert['node']}: {alert['message']}")
```

## Related Operations
- `cluster.list_stream` - PREREQUISITE: Get cluster ID
- `cluster.get_details` - Get cluster configuration details
- `cluster.get_all_namespaces` - Get namespace information
- Metrics APIs - Historical metrics and trends
- Alerting APIs - Configure alerts based on metrics

## Error Handling
- **400 Bad Request:** Invalid cluster ID format - verify ID structure
- **401 Unauthorized:** Invalid or expired authentication token - refresh token
- **403 Forbidden:** Insufficient permissions to view node metrics
- **404 Not Found:** Cluster not found - verify cluster_id exists
- **500 Internal Server Error:** Service error - retry with exponential backoff
- **503 Service Unavailable:** Metrics service temporarily unavailable - implement retry logic

## Response Codes
**⚠️ To be documented - requires actual API testing**

## Performance Notes
- Response time varies with number of nodes (typically < 2000ms)
- Real-time metrics - no caching recommended
- For monitoring dashboards, poll every 30-60 seconds
- Consider aggregating metrics client-side for historical trends
- Rate limiting: To be documented

## Best Practices

### Regular Health Checks
```python
import time

def continuous_monitoring(cluster_id, token, interval=60):
    """Continuously monitor node metrics"""
    monitor = NodeMonitor(cluster_id, token)
    
    while True:
        health = monitor.check_node_health()
        
        # Log critical alerts
        if health['alerts']['critical']:
            for alert in health['alerts']['critical']:
                print(f"🔴 CRITICAL: {alert['node']} - {alert['message']}")
                # Send to alerting system
        
        time.sleep(interval)

# Run monitoring
# continuous_monitoring("k8s-prod-001", "token", interval=60)
```

### Capacity Planning Reports
```python
def generate_capacity_report(cluster_id, token):
    """Generate capacity planning report"""
    metrics_data = get_node_metrics(cluster_id, token)
    
    if not metrics_data:
        return None
    
    nodes = metrics_data['nodes']
    capacity = calculate_available_capacity(nodes)
    
    report = f"""
# Capacity Report: {cluster_id}
Generated: {metrics_data['timestamp']}

## Summary
- Total Nodes: {len(nodes)}
- Ready Nodes: {len([n for n in nodes if n['status'] == 'Ready'])}

## Resource Utilization
- CPU: {capacity['cpu']['usage_percent']}% ({capacity['cpu']['used_cores']:.1f} / {capacity['cpu']['total_cores']} cores)
- Memory: {capacity['memory']['usage_percent']}% ({capacity['memory']['used_gb']:.1f} / {capacity['memory']['total_gb']} GB)
- Pods: {capacity['pods']['usage_percent']}% ({capacity['pods']['current']} / {capacity['pods']['capacity']})

## Available Capacity
- CPU: {capacity['cpu']['available_cores']:.1f} cores
- Memory: {capacity['memory']['available_gb']:.1f} GB
- Pods: {capacity['pods']['available']} slots

## Recommendations
"""
    
    # Add recommendations
    if capacity['cpu']['usage_percent'] > 80:
        report += "- ⚠️ High CPU usage - consider adding nodes\n"
    if capacity['memory']['usage_percent'] > 80:
        report += "- ⚠️ High memory usage - consider adding nodes\n"
    if capacity['pods']['usage_percent'] > 85:
        report += "- ⚠️ Pod capacity approaching limit\n"
    
    return report

# Usage
report = generate_capacity_report("k8s-prod-001", "token")
print(report)
```

## Metadata
- **Generated:** 2024-02-23T10:00:00Z
- **Source:** API endpoint analysis
- **API Version:** To be documented
- **Base Path:** {BASE_URL_PAAS_SERVICE}/paas
- **Additional notes:** ⚠️ This documentation is preliminary. Please provide actual API response for complete accuracy. Real-time metrics - poll regularly for monitoring.
