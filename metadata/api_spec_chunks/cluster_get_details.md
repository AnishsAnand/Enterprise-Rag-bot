# API Specification: cluster - get_details

**Resource:** cluster
**Operation:** get_details
**Aliases:** get cluster, cluster details, show cluster, cluster info, cluster information

## Endpoint
- **Method:** GET
- **URL:** `{BASE_URL_PAAS_SERVICE}/paas/cluster/getClusterDetails/{cluster_id}?status=true`
- **Auth:** Bearer token (assumed - needs confirmation)
- **Description:** Retrieves comprehensive details for a specific Kubernetes cluster including configuration, resource allocation, node information, and current operational status. The `status=true` parameter includes real-time status information.

## Required Parameters
- `cluster_id` - Unique cluster identifier (type: path parameter, format: string)

## Optional Parameters
- `status` - Include real-time status information (type: query parameter, default: false, recommended: true)

## Response Mapping
**⚠️ Note: Actual API response needed to complete this section accurately**

Assumed mappings (to be verified):
- `cluster_id`: data.id
- `cluster_name`: data.name
- `cluster_status`: data.status
- `cluster_version`: data.version
- `cluster_type`: data.type
- `region`: data.region
- `availability_zone`: data.availability_zone
- `created_at`: data.created_at
- `updated_at`: data.updated_at
- `node_count`: data.nodes.count
- `node_details`: data.nodes.details
- `master_nodes`: data.nodes.master
- `worker_nodes`: data.nodes.worker
- `pod_capacity`: data.capacity.max_pods
- `current_pods`: data.capacity.current_pods
- `namespaces`: data.namespaces
- `api_endpoint`: data.api_server.endpoint
- `dashboard_url`: data.dashboard_url
- `health_status`: data.health.status
- `health_checks`: data.health.checks
- `resource_limits`: data.resources.limits
- `resource_requests`: data.resources.requests
- `network_config`: data.network
- `storage_classes`: data.storage.classes
- `addons`: data.addons
- `tags`: data.tags

## Response Example
**⚠️ Placeholder - Replace with actual API response**

```json
{
  "status": "success",
  "data": {
    "id": "k8s-prod-001",
    "name": "production-cluster-east",
    "status": "running",
    "health": {
      "status": "healthy",
      "checks": {
        "api_server": "passing",
        "etcd": "passing",
        "scheduler": "passing",
        "controller_manager": "passing"
      },
      "last_check": "2024-02-23T10:30:00Z"
    },
    "version": "1.28.5",
    "type": "kubernetes",
    "region": "us-east-1",
    "availability_zone": "us-east-1a",
    "nodes": {
      "count": 15,
      "master": {
        "count": 3,
        "instance_type": "m5.xlarge",
        "status": "ready"
      },
      "worker": {
        "count": 12,
        "instance_types": ["m5.2xlarge", "m5.xlarge"],
        "status": "ready"
      },
      "details": [
        {
          "name": "master-1",
          "type": "master",
          "status": "Ready",
          "cpu": "4 cores",
          "memory": "16 GB",
          "ip": "10.0.1.10"
        }
      ]
    },
    "capacity": {
      "max_pods": 1100,
      "current_pods": 450,
      "max_nodes": 20,
      "current_nodes": 15
    },
    "namespaces": {
      "count": 25,
      "list": ["default", "kube-system", "production", "staging"]
    },
    "api_server": {
      "endpoint": "https://k8s-api.example.com:6443",
      "version": "v1.28.5",
      "status": "active"
    },
    "dashboard_url": "https://k8s-dashboard.example.com",
    "resources": {
      "limits": {
        "cpu": "60 cores",
        "memory": "240 GB",
        "storage": "2 TB"
      },
      "requests": {
        "cpu": "42 cores",
        "memory": "168 GB",
        "storage": "1.2 TB"
      },
      "utilization": {
        "cpu_percent": 70,
        "memory_percent": 70,
        "storage_percent": 60
      }
    },
    "network": {
      "vpc_id": "vpc-12345",
      "subnet_ids": ["subnet-abc", "subnet-def"],
      "security_groups": ["sg-k8s-master", "sg-k8s-worker"],
      "load_balancer": "lb-k8s-prod-001",
      "ingress_controller": "nginx"
    },
    "storage": {
      "classes": [
        {
          "name": "gp3",
          "provisioner": "ebs.csi.aws.com",
          "default": true
        },
        {
          "name": "io2",
          "provisioner": "ebs.csi.aws.com",
          "default": false
        }
      ],
      "persistent_volumes": 45
    },
    "addons": [
      {
        "name": "metrics-server",
        "version": "0.6.3",
        "status": "running"
      },
      {
        "name": "cluster-autoscaler",
        "version": "1.28.0",
        "status": "running"
      }
    ],
    "tags": {
      "environment": "production",
      "team": "platform",
      "cost-center": "engineering"
    },
    "created_at": "2024-01-15T10:00:00Z",
    "updated_at": "2024-02-23T10:30:00Z"
  }
}
```

## Response Fields Details

### Core Cluster Fields
- **id** - Unique cluster identifier
- **name** - Human-readable cluster name
- **status** - Operational status (running, stopped, starting, error, maintenance)
- **version** - Kubernetes version
- **type** - Cluster type (kubernetes, k8s, openshift)
- **region** - Geographic region
- **availability_zone** - Specific availability zone
- **created_at** - Cluster creation timestamp
- **updated_at** - Last update timestamp

### Health Fields
- **health.status** - Overall health (healthy, degraded, unhealthy)
- **health.checks** - Individual component health checks
- **health.last_check** - Last health check timestamp

### Node Information
- **nodes.count** - Total number of nodes
- **nodes.master** - Master node configuration
- **nodes.worker** - Worker node configuration
- **nodes.details** - Detailed node information

### Capacity and Resources
- **capacity.max_pods** - Maximum pod capacity
- **capacity.current_pods** - Current pod count
- **resources.limits** - Resource limits
- **resources.requests** - Resource requests
- **resources.utilization** - Current utilization percentages

### Network Configuration
- **network.vpc_id** - VPC identifier
- **network.subnet_ids** - Subnet identifiers
- **network.security_groups** - Security group names
- **network.load_balancer** - Load balancer configuration
- **network.ingress_controller** - Ingress controller type

### Storage Configuration
- **storage.classes** - Available storage classes
- **storage.persistent_volumes** - PV count

### API Server
- **api_server.endpoint** - Kubernetes API server URL
- **api_server.version** - API server version
- **api_server.status** - API server status

## Permissions
**Roles:** To be documented (requires actual permission information)
Likely requires: Admin, Cluster Viewer, DevOps, SRE roles

## Workflow Steps

### Workflow: Detailed Cluster Inspection
**Prerequisites:**
- Step 1: **List Clusters** (cluster.list_stream) - Get cluster ID from list

**Main Workflow:**
- Step 2: **Get Cluster Details** (cluster.get_details) - Get comprehensive information
- Step 3: **Get Node Metrics** (cluster.get_node_metrics) - Get node-level metrics (optional)
- Step 4: **Get Namespaces** (cluster.get_all_namespaces) - Get namespace details (optional)

### Example Workflow
```
1. list_stream() → Get all cluster IDs
2. get_cluster_details("k8s-prod-001", status=true) → Full cluster info
3. Use cluster details for:
   - Configuration management
   - Capacity planning
   - Troubleshooting
   - Documentation
```

## Usage Notes
- **status=true** parameter is recommended for real-time status information
- Response includes comprehensive cluster configuration
- Use for troubleshooting, capacity planning, and documentation
- Cache cluster details as configuration changes infrequently
- API endpoint URL can be used for direct kubectl access

## Kubernetes-Specific Features

### Cluster Configuration Management
Essential cluster configuration details:
- **API Server Access** - Endpoint URL for kubectl and API access
- **Node Architecture** - Master/worker node distribution
- **Resource Allocation** - CPU, memory, storage limits and requests
- **Network Topology** - VPC, subnets, security groups, load balancers

### High Availability Setup
The response includes HA configuration details:
- **Master Node Count** - Should be 3+ for HA
- **Availability Zones** - Multi-AZ deployment for resilience
- **Health Checks** - Component-level health monitoring
- **Backup Configuration** - Disaster recovery setup

### Add-on Management
Track installed cluster add-ons:
- **Metrics Server** - Resource metrics collection
- **Cluster Autoscaler** - Automatic node scaling
- **Ingress Controllers** - Traffic routing
- **CNI Plugins** - Network connectivity

### Storage Configuration
Storage class information for:
- **PersistentVolume provisioning** - Dynamic volume creation
- **Performance tiers** - gp3, io2, etc.
- **Default storage class** - Auto-provisioning behavior

## Common Use Cases
1. **Get cluster details**: "show cluster k8s-prod-001", "get details for cluster", "cluster info"
2. **Check configuration**: "cluster configuration", "show cluster settings", "cluster setup"
3. **Capacity planning**: "cluster capacity", "how many pods can run", "resource limits"
4. **Health check**: "cluster health", "is cluster healthy", "component status"
5. **API access**: "cluster api endpoint", "how to connect to cluster", "kubectl endpoint"
6. **Node information**: "cluster nodes", "master nodes", "worker node count"
7. **Network setup**: "cluster network", "vpc configuration", "load balancer"

## Query Interpretations
- "get cluster {id}" → GET /cluster/getClusterDetails/{id}?status=true
- "cluster {id} details" → GET /cluster/getClusterDetails/{id}?status=true
- "show cluster {id}" → GET /cluster/getClusterDetails/{id}?status=true
- "{id} cluster info" → GET /cluster/getClusterDetails/{id}?status=true

## Data Processing Examples

### Python Example
```python
import requests
from typing import Optional, Dict

def get_cluster_details(cluster_id: str, auth_token: str, include_status: bool = True) -> Optional[Dict]:
    """
    Get comprehensive cluster details.
    
    Args:
        cluster_id: Cluster identifier
        auth_token: Bearer authentication token
        include_status: Include real-time status (recommended: True)
        
    Returns:
        dict: Cluster details or None if failed
    """
    url = f"https://ipcloud.tatacommunications.com/paasservice/paas/cluster/getClusterDetails/{cluster_id}"
    
    params = {}
    if include_status:
        params['status'] = 'true'
    
    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        
        data = response.json()
        if data.get('status') == 'success':
            cluster = data['data']
            print(f"Cluster: {cluster['name']} ({cluster['status']})")
            print(f"Nodes: {cluster['nodes']['count']}, Pods: {cluster['capacity']['current_pods']}")
            return cluster
        else:
            print(f"Failed: {data.get('message')}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"API error: {e}")
        return None

def extract_api_endpoint(cluster: Dict) -> str:
    """Extract Kubernetes API endpoint"""
    return cluster.get('api_server', {}).get('endpoint', '')

def check_cluster_health(cluster: Dict) -> Dict:
    """Analyze cluster health"""
    health = cluster.get('health', {})
    checks = health.get('checks', {})
    
    failing = [name for name, status in checks.items() if status != 'passing']
    
    return {
        'overall': health.get('status'),
        'healthy': len(failing) == 0,
        'failing_components': failing,
        'last_check': health.get('last_check')
    }

def calculate_capacity_metrics(cluster: Dict) -> Dict:
    """Calculate capacity utilization metrics"""
    capacity = cluster.get('capacity', {})
    resources = cluster.get('resources', {})
    
    pod_utilization = 0
    if capacity.get('max_pods', 0) > 0:
        pod_utilization = (capacity.get('current_pods', 0) / capacity['max_pods']) * 100
    
    return {
        'pod_utilization_percent': round(pod_utilization, 2),
        'pods_available': capacity.get('max_pods', 0) - capacity.get('current_pods', 0),
        'cpu_utilization_percent': resources.get('utilization', {}).get('cpu_percent', 0),
        'memory_utilization_percent': resources.get('utilization', {}).get('memory_percent', 0),
        'nodes_available': capacity.get('max_nodes', 0) - capacity.get('current_nodes', 0)
    }

# Usage
cluster_id = "k8s-prod-001"
token = "your-token-here"

# Get full cluster details with status
cluster = get_cluster_details(cluster_id, token, include_status=True)

if cluster:
    # Extract API endpoint for kubectl
    api_endpoint = extract_api_endpoint(cluster)
    print(f"API Endpoint: {api_endpoint}")
    
    # Check health
    health = check_cluster_health(cluster)
    print(f"Health: {health['overall']} (Healthy: {health['healthy']})")
    if health['failing_components']:
        print(f"⚠️ Failing components: {', '.join(health['failing_components'])}")
    
    # Capacity metrics
    metrics = calculate_capacity_metrics(cluster)
    print(f"Pod utilization: {metrics['pod_utilization_percent']}%")
    print(f"Pods available: {metrics['pods_available']}")
    print(f"CPU utilization: {metrics['cpu_utilization_percent']}%")
```

### JavaScript Example
```javascript
async function getClusterDetails(clusterId, authToken, includeStatus = true) {
    let url = `https://ipcloud.tatacommunications.com/paasservice/paas/cluster/getClusterDetails/${clusterId}`;
    
    if (includeStatus) {
        url += '?status=true';
    }
    
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
            const cluster = data.data;
            console.log(`Cluster: ${cluster.name} (${cluster.status})`);
            console.log(`Nodes: ${cluster.nodes.count}, Pods: ${cluster.capacity.current_pods}`);
            return cluster;
        } else {
            console.error(`Failed: ${data.message}`);
            return null;
        }
    } catch (error) {
        console.error(`API error: ${error.message}`);
        return null;
    }
}

function analyzeClusterResources(cluster) {
    const resources = cluster.resources || {};
    const limits = resources.limits || {};
    const requests = resources.requests || {};
    const utilization = resources.utilization || {};
    
    return {
        cpu: {
            limit: limits.cpu,
            requested: requests.cpu,
            utilization: utilization.cpu_percent
        },
        memory: {
            limit: limits.memory,
            requested: requests.memory,
            utilization: utilization.memory_percent
        },
        storage: {
            limit: limits.storage,
            requested: requests.storage,
            utilization: utilization.storage_percent
        }
    };
}

function generateKubectlConfig(cluster) {
    const apiEndpoint = cluster.api_server?.endpoint || '';
    const clusterName = cluster.name;
    
    return `
apiVersion: v1
kind: Config
clusters:
- cluster:
    server: ${apiEndpoint}
  name: ${clusterName}
contexts:
- context:
    cluster: ${clusterName}
    user: admin
  name: ${clusterName}
current-context: ${clusterName}
`.trim();
}

// Usage
const clusterId = 'k8s-prod-001';
const token = 'your-token-here';

const cluster = await getClusterDetails(clusterId, token, true);

if (cluster) {
    // Analyze resources
    const resources = analyzeClusterResources(cluster);
    console.log('Resource Analysis:', resources);
    
    // Generate kubectl config
    const kubectlConfig = generateKubectlConfig(cluster);
    console.log('Kubectl Config:', kubectlConfig);
    
    // List addons
    const addons = cluster.addons || [];
    console.log(`Installed addons: ${addons.map(a => a.name).join(', ')}`);
}
```

## Integration Examples

### Cluster Configuration Validator
```python
class ClusterValidator:
    def __init__(self, cluster):
        self.cluster = cluster
        self.issues = []
        self.warnings = []
    
    def validate(self):
        """Run all validation checks"""
        self.check_ha_configuration()
        self.check_resource_utilization()
        self.check_component_health()
        self.check_version_support()
        
        return {
            'valid': len(self.issues) == 0,
            'issues': self.issues,
            'warnings': self.warnings
        }
    
    def check_ha_configuration(self):
        """Validate high availability setup"""
        master_count = self.cluster['nodes']['master']['count']
        
        if master_count < 3:
            self.issues.append(f"Insufficient master nodes for HA: {master_count} (recommend 3+)")
        
        # Check if multi-AZ
        if 'availability_zones' in self.cluster:
            zones = self.cluster['availability_zones']
            if len(zones) < 2:
                self.warnings.append("Single AZ deployment - consider multi-AZ for better resilience")
    
    def check_resource_utilization(self):
        """Check resource utilization levels"""
        utilization = self.cluster.get('resources', {}).get('utilization', {})
        
        cpu_util = utilization.get('cpu_percent', 0)
        mem_util = utilization.get('memory_percent', 0)
        
        if cpu_util > 85:
            self.warnings.append(f"High CPU utilization: {cpu_util}% (consider scaling)")
        
        if mem_util > 85:
            self.warnings.append(f"High memory utilization: {mem_util}% (consider scaling)")
    
    def check_component_health(self):
        """Validate component health"""
        checks = self.cluster.get('health', {}).get('checks', {})
        
        for component, status in checks.items():
            if status != 'passing':
                self.issues.append(f"Component unhealthy: {component} ({status})")
    
    def check_version_support(self):
        """Check Kubernetes version support"""
        version = self.cluster.get('version', '')
        # Add version support logic
        pass

# Usage
cluster = get_cluster_details("k8s-prod-001", token)
validator = ClusterValidator(cluster)
result = validator.validate()

print(f"Validation: {'✓ PASS' if result['valid'] else '✗ FAIL'}")
for issue in result['issues']:
    print(f"  ✗ Issue: {issue}")
for warning in result['warnings']:
    print(f"  ⚠️ Warning: {warning}")
```

### Cluster Documentation Generator
```python
class ClusterDocGenerator:
    def __init__(self, cluster):
        self.cluster = cluster
    
    def generate_markdown(self):
        """Generate Markdown documentation"""
        doc = f"""# Cluster: {self.cluster['name']}

## Overview
- **ID**: {self.cluster['id']}
- **Status**: {self.cluster['status']}
- **Version**: {self.cluster['version']}
- **Region**: {self.cluster['region']}
- **Created**: {self.cluster['created_at']}

## Infrastructure
### Nodes
- **Total**: {self.cluster['nodes']['count']}
- **Master**: {self.cluster['nodes']['master']['count']} x {self.cluster['nodes']['master']['instance_type']}
- **Worker**: {self.cluster['nodes']['worker']['count']}

### Capacity
- **Max Pods**: {self.cluster['capacity']['max_pods']}
- **Current Pods**: {self.cluster['capacity']['current_pods']}
- **Utilization**: {self.cluster['capacity']['current_pods'] / self.cluster['capacity']['max_pods'] * 100:.1f}%

## Network
- **VPC**: {self.cluster['network']['vpc_id']}
- **Load Balancer**: {self.cluster['network']['load_balancer']}
- **Ingress**: {self.cluster['network']['ingress_controller']}

## API Access
```bash
kubectl --server={self.cluster['api_server']['endpoint']} get nodes
```

## Add-ons
{self._format_addons()}

## Tags
{self._format_tags()}
"""
        return doc
    
    def _format_addons(self):
        addons = self.cluster.get('addons', [])
        if not addons:
            return "None installed"
        
        lines = []
        for addon in addons:
            lines.append(f"- **{addon['name']}** v{addon['version']} ({addon['status']})")
        
        return '\n'.join(lines)
    
    def _format_tags(self):
        tags = self.cluster.get('tags', {})
        if not tags:
            return "None"
        
        lines = [f"- **{k}**: {v}" for k, v in tags.items()]
        return '\n'.join(lines)

# Usage
cluster = get_cluster_details("k8s-prod-001", token)
doc_gen = ClusterDocGenerator(cluster)
markdown = doc_gen.generate_markdown()
print(markdown)

# Save to file
with open(f"cluster-{cluster['id']}.md", 'w') as f:
    f.write(markdown)
```

## Related Operations
- `cluster.list_stream` - PREREQUISITE: Get cluster ID from list
- `cluster.get_node_metrics` - Get detailed node metrics
- `cluster.get_all_namespaces` - Get namespace information
- Monitoring APIs - Integration with monitoring systems
- kubectl API - Direct Kubernetes API access

## Error Handling
- **400 Bad Request:** Invalid cluster ID format - verify ID structure
- **401 Unauthorized:** Invalid or expired authentication token - refresh token
- **403 Forbidden:** Insufficient permissions to view cluster details
- **404 Not Found:** Cluster not found - verify cluster_id exists
- **500 Internal Server Error:** Service error - retry with exponential backoff
- **503 Service Unavailable:** Service temporarily unavailable - implement retry logic

## Response Codes
**⚠️ To be documented - requires actual API testing**

Likely patterns:
- `0` or `success` - Request successful
- Non-zero or `error` - Request failed (check message field)

## Performance Notes
- Response time typically < 1000ms
- Cluster details change infrequently - safe to cache for extended periods
- Use `status=true` for real-time status at cost of slightly slower response
- No pagination (single cluster query)
- Rate limiting: To be documented

## Best Practices

### Always Include Status Parameter
```python
# ❌ AVOID - Missing real-time status
cluster = get_cluster_details(cluster_id, token, include_status=False)

# ✅ RECOMMENDED - Include real-time status
cluster = get_cluster_details(cluster_id, token, include_status=True)
```

### Cache Cluster Configuration
```python
import time
from datetime import datetime, timedelta

class ClusterDetailsCache:
    def __init__(self, ttl=3600):  # 1 hour
        self.cache = {}
        self.ttl = ttl
    
    def get(self, cluster_id):
        if cluster_id in self.cache:
            entry = self.cache[cluster_id]
            if time.time() - entry['timestamp'] < self.ttl:
                return entry['data']
        return None
    
    def set(self, cluster_id, data):
        self.cache[cluster_id] = {
            'data': data,
            'timestamp': time.time()
        }

# Usage - cluster config doesn't change often
cache = ClusterDetailsCache(ttl=3600)
cluster = cache.get(cluster_id)
if not cluster:
    cluster = get_cluster_details(cluster_id, token, include_status=True)
    cache.set(cluster_id, cluster)
```

### Extract Kubectl Configuration
```python
def setup_kubectl_access(cluster):
    """Generate kubectl configuration from cluster details"""
    api_endpoint = cluster['api_server']['endpoint']
    cluster_name = cluster['name']
    
    # Generate kubeconfig
    config = {
        'apiVersion': 'v1',
        'kind': 'Config',
        'clusters': [{
            'cluster': {
                'server': api_endpoint
            },
            'name': cluster_name
        }],
        'contexts': [{
            'context': {
                'cluster': cluster_name,
                'user': 'admin'
            },
            'name': cluster_name
        }],
        'current-context': cluster_name
    }
    
    return config

# Usage
cluster = get_cluster_details(cluster_id, token, include_status=True)
kubeconfig = setup_kubectl_access(cluster)
```

## Metadata
- **Generated:** 2024-02-23T10:00:00Z
- **Source:** API endpoint analysis
- **API Version:** To be documented
- **Base Path:** {BASE_URL_PAAS_SERVICE}/paas
- **Additional notes:** ⚠️ This documentation is preliminary. Please provide actual API response for complete accuracy. Recommended to use status=true parameter.
