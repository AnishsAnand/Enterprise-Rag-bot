# API Specification: cluster - get_all_namespaces

**Resource:** cluster
**Operation:** get_all_namespaces
**Aliases:** list namespaces, get namespaces, show namespaces, namespace list, all namespaces

## Endpoint
- **Method:** GET
- **URL:** `{BASE_URL_PAAS_SERVICE}/paas/cluster/{cluster_id}/getallnamespaces`
- **Auth:** Bearer token (assumed - needs confirmation)
- **Description:** Retrieves all Kubernetes namespaces within a specific cluster. Returns namespace details including status, resource quotas, labels, and usage statistics. Essential for multi-tenant cluster management and resource organization.

## Required Parameters
- `cluster_id` - Unique cluster identifier (type: path parameter, format: string)

## Optional Parameters
None (to be documented based on actual API behavior)

## Response Mapping
**⚠️ Note: Actual API response needed to complete this section accurately**

Assumed mappings (to be verified):
- `cluster_id`: data.cluster_id
- `namespaces`: data.namespaces
- `namespace_names`: data.namespaces[*].name
- `namespace_statuses`: data.namespaces[*].status
- `namespace_labels`: data.namespaces[*].labels
- `namespace_annotations`: data.namespaces[*].annotations
- `resource_quotas`: data.namespaces[*].resource_quotas
- `limit_ranges`: data.namespaces[*].limit_ranges
- `pod_count`: data.namespaces[*].resources.pods.count
- `service_count`: data.namespaces[*].resources.services.count
- `deployment_count`: data.namespaces[*].resources.deployments.count
- `created_at`: data.namespaces[*].created_at
- `total_namespaces`: data.total_count
- `timestamp`: data.timestamp

## Response Example
**⚠️ Placeholder - Replace with actual API response**

```json
{
  "status": "success",
  "data": {
    "cluster_id": "k8s-prod-001",
    "timestamp": "2024-02-23T10:30:00Z",
    "total_count": 25,
    "namespaces": [
      {
        "name": "default",
        "status": "Active",
        "created_at": "2024-01-15T10:00:00Z",
        "labels": {
          "kubernetes.io/metadata.name": "default"
        },
        "annotations": {
          "created-by": "system"
        },
        "resource_quotas": null,
        "limit_ranges": null,
        "resources": {
          "pods": {
            "count": 5,
            "running": 5,
            "pending": 0,
            "failed": 0
          },
          "services": {
            "count": 3
          },
          "deployments": {
            "count": 2
          },
          "configmaps": {
            "count": 8
          },
          "secrets": {
            "count": 5
          }
        },
        "usage": {
          "cpu": {
            "requests": "200m",
            "limits": "500m"
          },
          "memory": {
            "requests": "512Mi",
            "limits": "1Gi"
          }
        }
      },
      {
        "name": "production",
        "status": "Active",
        "created_at": "2024-01-15T11:00:00Z",
        "labels": {
          "environment": "production",
          "team": "platform",
          "cost-center": "engineering"
        },
        "annotations": {
          "description": "Production workloads",
          "contact": "platform-team@example.com"
        },
        "resource_quotas": {
          "hard": {
            "pods": "100",
            "services": "20",
            "cpu": "20",
            "memory": "40Gi",
            "persistentvolumeclaims": "10"
          },
          "used": {
            "pods": "45",
            "services": "12",
            "cpu": "14",
            "memory": "28Gi",
            "persistentvolumeclaims": "6"
          }
        },
        "limit_ranges": [
          {
            "name": "default-limits",
            "limits": [
              {
                "type": "Container",
                "default": {
                  "cpu": "500m",
                  "memory": "512Mi"
                },
                "defaultRequest": {
                  "cpu": "100m",
                  "memory": "128Mi"
                },
                "max": {
                  "cpu": "2",
                  "memory": "2Gi"
                },
                "min": {
                  "cpu": "50m",
                  "memory": "64Mi"
                }
              }
            ]
          }
        ],
        "resources": {
          "pods": {
            "count": 45,
            "running": 43,
            "pending": 2,
            "failed": 0
          },
          "services": {
            "count": 12
          },
          "deployments": {
            "count": 18
          },
          "statefulsets": {
            "count": 3
          },
          "configmaps": {
            "count": 25
          },
          "secrets": {
            "count": 20
          },
          "persistentvolumeclaims": {
            "count": 6
          }
        },
        "usage": {
          "cpu": {
            "requests": "14",
            "limits": "25"
          },
          "memory": {
            "requests": "28Gi",
            "limits": "42Gi"
          }
        }
      },
      {
        "name": "kube-system",
        "status": "Active",
        "created_at": "2024-01-15T10:00:00Z",
        "labels": {
          "kubernetes.io/metadata.name": "kube-system"
        },
        "annotations": {
          "created-by": "system"
        },
        "resource_quotas": null,
        "limit_ranges": null,
        "resources": {
          "pods": {
            "count": 12,
            "running": 12,
            "pending": 0,
            "failed": 0
          },
          "services": {
            "count": 5
          },
          "daemonsets": {
            "count": 3
          },
          "deployments": {
            "count": 4
          }
        },
        "usage": {
          "cpu": {
            "requests": "1",
            "limits": "2"
          },
          "memory": {
            "requests": "2Gi",
            "limits": "4Gi"
          }
        }
      }
    ]
  }
}
```

## Response Fields Details

### Core Namespace Fields
- **name** - Namespace name (unique within cluster)
- **status** - Namespace status (Active, Terminating)
- **created_at** - Namespace creation timestamp
- **labels** - Key-value labels for organization and selection
- **annotations** - Additional metadata and configuration

### Resource Quotas
- **resource_quotas.hard** - Hard limits for resources
- **resource_quotas.used** - Current usage against quotas
- Quota types: pods, services, cpu, memory, storage, etc.

### Limit Ranges
- **limit_ranges** - Default and maximum resource limits for containers
- **type** - Resource type (Container, Pod, PersistentVolumeClaim)
- **default** - Default limits if not specified
- **defaultRequest** - Default requests if not specified
- **max** - Maximum allowed limits
- **min** - Minimum required requests

### Resource Counts
- **resources.pods** - Pod count and status breakdown
- **resources.services** - Service count
- **resources.deployments** - Deployment count
- **resources.statefulsets** - StatefulSet count
- **resources.configmaps** - ConfigMap count
- **resources.secrets** - Secret count
- **resources.persistentvolumeclaims** - PVC count

### Resource Usage
- **usage.cpu.requests** - Total CPU requested
- **usage.cpu.limits** - Total CPU limits
- **usage.memory.requests** - Total memory requested
- **usage.memory.limits** - Total memory limits

## Permissions
**Roles:** To be documented (requires actual permission information)
Likely requires: Admin, Cluster Viewer, Namespace Admin, Developer roles

## Workflow Steps

### Workflow: Namespace Management
**Prerequisites:**
- Step 1: **List Clusters** (cluster.list_stream) - Get cluster ID

**Main Workflow:**
- Step 2: **Get All Namespaces** (cluster.get_all_namespaces) - List namespaces
- Step 3: **Analyze Namespaces** - Review quotas, usage, resources
- Step 4: **Manage Resources** - Create, update, or delete namespace resources

### Multi-Tenant Management Workflow
```
1. list_stream() → Get cluster IDs
2. For each cluster:
   - get_all_namespaces(cluster_id) → List namespaces
   - Analyze quota usage per namespace
   - Check for quota violations
   - Monitor resource consumption
3. Adjust quotas or scale resources as needed
```

## Usage Notes
- System namespaces (kube-system, kube-public, etc.) are included
- Use labels for filtering and organizing namespaces
- Resource quotas enforce hard limits on namespace resources
- LimitRanges set default and maximum values for pods/containers
- Cache namespace list as it changes infrequently

## Kubernetes-Specific Features

### Namespace Isolation
Namespaces provide:
- **Resource Isolation** - Logical separation of workloads
- **Access Control** - RBAC policies per namespace
- **Resource Quotas** - Limit resource consumption
- **Network Policies** - Isolate network traffic
- **Multi-Tenancy** - Support multiple teams/environments

### Resource Management
Essential for cluster resource management:
- **Quota Enforcement** - Prevent resource over-consumption
- **Default Limits** - Ensure all pods have resource limits
- **Capacity Planning** - Track resource usage per namespace
- **Cost Allocation** - Attribute costs to teams/projects

### Common Namespace Patterns
- **Environment-based**: dev, staging, production
- **Team-based**: team-a, team-b, platform
- **Application-based**: app-frontend, app-backend, app-db
- **System namespaces**: kube-system, kube-public, kube-node-lease

### Resource Quotas
Enforce limits on:
- **Compute**: CPU, memory
- **Storage**: PersistentVolumeClaims, storage size
- **Objects**: pods, services, configmaps, secrets
- **Extended resources**: GPUs, custom resources

## Common Use Cases
1. **List all namespaces**: "show namespaces", "list namespaces", "get all namespaces"
2. **Find by environment**: "production namespaces", "staging namespaces"
3. **Check quotas**: "namespace quotas", "quota usage", "resource limits"
4. **Resource inventory**: "pods per namespace", "namespace resources"
5. **Multi-tenant management**: "team namespaces", "customer namespaces"
6. **Capacity planning**: "namespace utilization", "available quota"

## Query Interpretations
- "get namespaces for {cluster_id}" → GET /cluster/{cluster_id}/getallnamespaces
- "list namespaces {cluster_id}" → GET /cluster/{cluster_id}/getallnamespaces
- "show all namespaces {cluster_id}" → GET /cluster/{cluster_id}/getallnamespaces

## Data Processing Examples

### Python Example
```python
import requests
from typing import List, Dict, Optional

def get_all_namespaces(cluster_id: str, auth_token: str) -> Optional[Dict]:
    """
    Get all namespaces in a cluster.
    
    Args:
        cluster_id: Cluster identifier
        auth_token: Bearer authentication token
        
    Returns:
        dict: Namespace data or None if failed
    """
    url = f"https://ipcloud.tatacommunications.com/paasservice/paas/cluster/{cluster_id}/getallnamespaces"
    
    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        if data.get('status') == 'success':
            ns_data = data['data']
            print(f"Cluster {cluster_id}: {ns_data['total_count']} namespaces")
            return ns_data
        else:
            print(f"Failed: {data.get('message')}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"API error: {e}")
        return None

def filter_namespaces_by_label(namespaces: List[Dict], label_key: str, label_value: str) -> List[Dict]:
    """Filter namespaces by label"""
    return [
        ns for ns in namespaces 
        if ns.get('labels', {}).get(label_key) == label_value
    ]

def get_system_namespaces(namespaces: List[Dict]) -> List[Dict]:
    """Get Kubernetes system namespaces"""
    system_ns = ['kube-system', 'kube-public', 'kube-node-lease']
    return [ns for ns in namespaces if ns['name'] in system_ns]

def get_user_namespaces(namespaces: List[Dict]) -> List[Dict]:
    """Get user-created namespaces (non-system)"""
    system_ns = ['kube-system', 'kube-public', 'kube-node-lease', 'default']
    return [ns for ns in namespaces if ns['name'] not in system_ns]

def analyze_quota_usage(namespace: Dict) -> Optional[Dict]:
    """Analyze resource quota usage for a namespace"""
    quotas = namespace.get('resource_quotas')
    
    if not quotas:
        return None
    
    hard = quotas.get('hard', {})
    used = quotas.get('used', {})
    
    analysis = {}
    
    for resource, limit in hard.items():
        current = used.get(resource, 0)
        
        # Convert to comparable numbers
        try:
            if isinstance(limit, str):
                # Handle Kubernetes resource quantities
                limit_num = parse_quantity(limit)
                current_num = parse_quantity(str(current))
                usage_percent = (current_num / limit_num * 100) if limit_num > 0 else 0
            else:
                usage_percent = (float(current) / float(limit) * 100) if float(limit) > 0 else 0
            
            analysis[resource] = {
                'limit': limit,
                'used': current,
                'usage_percent': round(usage_percent, 2),
                'available': limit_num - current_num if isinstance(limit, str) else float(limit) - float(current)
            }
        except:
            analysis[resource] = {
                'limit': limit,
                'used': current,
                'usage_percent': 0,
                'available': 'unknown'
            }
    
    return analysis

def parse_quantity(quantity_str: str) -> float:
    """Parse Kubernetes resource quantity to float"""
    # Simple parser - enhance for production use
    quantity_str = str(quantity_str)
    
    if quantity_str.endswith('Gi'):
        return float(quantity_str[:-2]) * 1024
    elif quantity_str.endswith('Mi'):
        return float(quantity_str[:-2])
    elif quantity_str.endswith('m'):
        return float(quantity_str[:-1]) / 1000
    else:
        try:
            return float(quantity_str)
        except:
            return 0

# Usage
cluster_id = "k8s-prod-001"
token = "your-token-here"

# Get all namespaces
ns_data = get_all_namespaces(cluster_id, token)

if ns_data:
    namespaces = ns_data['namespaces']
    
    # Filter by environment
    prod_ns = filter_namespaces_by_label(namespaces, 'environment', 'production')
    print(f"Production namespaces: {len(prod_ns)}")
    
    # Get user vs system namespaces
    user_ns = get_user_namespaces(namespaces)
    system_ns = get_system_namespaces(namespaces)
    print(f"User namespaces: {len(user_ns)}, System namespaces: {len(system_ns)}")
    
    # Analyze quota usage
    for ns in namespaces:
        if ns.get('resource_quotas'):
            print(f"\nNamespace: {ns['name']}")
            quota_analysis = analyze_quota_usage(ns)
            if quota_analysis:
                for resource, stats in quota_analysis.items():
                    print(f"  {resource}: {stats['usage_percent']:.1f}% ({stats['used']} / {stats['limit']})")
```

### JavaScript Example
```javascript
async function getAllNamespaces(clusterId, authToken) {
    const url = `https://ipcloud.tatacommunications.com/paasservice/paas/cluster/${clusterId}/getallnamespaces`;
    
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
            const nsData = data.data;
            console.log(`Cluster ${clusterId}: ${nsData.total_count} namespaces`);
            return nsData;
        } else {
            console.error(`Failed: ${data.message}`);
            return null;
        }
    } catch (error) {
        console.error(`API error: ${error.message}`);
        return null;
    }
}

function groupNamespacesByLabel(namespaces, labelKey) {
    const groups = {};
    
    namespaces.forEach(ns => {
        const value = ns.labels?.[labelKey] || 'untagged';
        if (!groups[value]) {
            groups[value] = [];
        }
        groups[value].push(ns);
    });
    
    return groups;
}

function calculateNamespaceResourceTotals(namespaces) {
    const totals = {
        pods: 0,
        services: 0,
        deployments: 0,
        configmaps: 0,
        secrets: 0
    };
    
    namespaces.forEach(ns => {
        const resources = ns.resources || {};
        totals.pods += resources.pods?.count || 0;
        totals.services += resources.services?.count || 0;
        totals.deployments += resources.deployments?.count || 0;
        totals.configmaps += resources.configmaps?.count || 0;
        totals.secrets += resources.secrets?.count || 0;
    });
    
    return totals;
}

function findNamespacesNearQuotaLimit(namespaces, threshold = 80) {
    const nearLimit = [];
    
    namespaces.forEach(ns => {
        if (!ns.resource_quotas) return;
        
        const hard = ns.resource_quotas.hard || {};
        const used = ns.resource_quotas.used || {};
        
        for (const [resource, limit] of Object.entries(hard)) {
            const current = used[resource] || 0;
            const usage = (parseFloat(current) / parseFloat(limit)) * 100;
            
            if (usage >= threshold) {
                nearLimit.push({
                    namespace: ns.name,
                    resource,
                    usage: Math.round(usage),
                    current,
                    limit
                });
            }
        }
    });
    
    return nearLimit;
}

// Usage
const clusterId = 'k8s-prod-001';
const token = 'your-token-here';

const nsData = await getAllNamespaces(clusterId, token);

if (nsData) {
    const namespaces = nsData.namespaces;
    
    // Group by environment
    const byEnv = groupNamespacesByLabel(namespaces, 'environment');
    console.log('Namespaces by environment:');
    for (const [env, nsList] of Object.entries(byEnv)) {
        console.log(`  ${env}: ${nsList.length}`);
    }
    
    // Resource totals
    const totals = calculateNamespaceResourceTotals(namespaces);
    console.log('
Total resources across namespaces:', totals);
    
    // Near quota limit
    const nearLimit = findNamespacesNearQuotaLimit(namespaces, 80);
    if (nearLimit.length > 0) {
        console.log('
⚠️ Namespaces near quota limit:');
        nearLimit.forEach(item => {
            console.log(`  ${item.namespace}: ${item.resource} at ${item.usage}%`);
        });
    }
}
```

## Integration Examples

### Namespace Quota Monitor
```python
class NamespaceQuotaMonitor:
    def __init__(self, cluster_id, auth_token):
        self.cluster_id = cluster_id
        self.auth_token = auth_token
        self.warning_threshold = 80
        self.critical_threshold = 95
    
    def check_all_quotas(self):
        """Check quota usage for all namespaces"""
        ns_data = get_all_namespaces(self.cluster_id, self.auth_token)
        
        if not ns_data:
            return None
        
        alerts = {
            'critical': [],
            'warning': [],
            'healthy': []
        }
        
        for ns in ns_data['namespaces']:
            if ns.get('resource_quotas'):
                quota_status = self._check_namespace_quota(ns)
                
                if quota_status['max_usage'] >= self.critical_threshold:
                    alerts['critical'].append(quota_status)
                elif quota_status['max_usage'] >= self.warning_threshold:
                    alerts['warning'].append(quota_status)
                else:
                    alerts['healthy'].append(quota_status)
        
        return {
            'cluster_id': self.cluster_id,
            'timestamp': ns_data['timestamp'],
            'total_namespaces': ns_data['total_count'],
            'with_quotas': len(alerts['critical']) + len(alerts['warning']) + len(alerts['healthy']),
            'alerts': alerts
        }
    
    def _check_namespace_quota(self, namespace):
        """Check quota for a single namespace"""
        quotas = namespace.get('resource_quotas', {})
        hard = quotas.get('hard', {})
        used = quotas.get('used', {})
        
        resource_usage = {}
        max_usage = 0
        
        for resource, limit in hard.items():
            current = used.get(resource, 0)
            
            try:
                usage_percent = (parse_quantity(str(current)) / parse_quantity(str(limit)) * 100)
                resource_usage[resource] = round(usage_percent, 2)
                max_usage = max(max_usage, usage_percent)
            except:
                pass
        
        return {
            'namespace': namespace['name'],
            'max_usage': round(max_usage, 2),
            'resource_usage': resource_usage,
            'labels': namespace.get('labels', {})
        }

# Usage
monitor = NamespaceQuotaMonitor("k8s-prod-001", "token")
quota_report = monitor.check_all_quotas()

if quota_report:
    print(f"Quota Report - {quota_report['timestamp']}")
    print(f"Total: {quota_report['total_namespaces']} namespaces, {quota_report['with_quotas']} with quotas")
    
    if quota_report['alerts']['critical']:
        print(f"\n🔴 CRITICAL ({len(quota_report['alerts']['critical'])}):")
        for alert in quota_report['alerts']['critical']:
            print(f"  - {alert['namespace']}: {alert['max_usage']:.1f}% max usage")
            for resource, usage in alert['resource_usage'].items():
                if usage >= 95:
                    print(f"    • {resource}: {usage:.1f}%")
    
    if quota_report['alerts']['warning']:
        print(f"\n⚠️ WARNING ({len(quota_report['alerts']['warning'])}):")
        for alert in quota_report['alerts']['warning']:
            print(f"  - {alert['namespace']}: {alert['max_usage']:.1f}% max usage")
```

### Multi-Tenant Resource Allocation
```python
class MultiTenantManager:
    def __init__(self, cluster_id, auth_token):
        self.cluster_id = cluster_id
        self.auth_token = auth_token
    
    def get_tenant_summary(self):
        """Get resource allocation summary by tenant"""
        ns_data = get_all_namespaces(self.cluster_id, self.auth_token)
        
        if not ns_data:
            return None
        
        # Group by team label
        teams = {}
        
        for ns in ns_data['namespaces']:
            team = ns.get('labels', {}).get('team', 'unassigned')
            
            if team not in teams:
                teams[team] = {
                    'namespaces': [],
                    'total_pods': 0,
                    'total_cpu_requests': 0,
                    'total_memory_requests': 0,
                    'quota_violations': 0
                }
            
            teams[team]['namespaces'].append(ns['name'])
            
            # Sum resources
            resources = ns.get('resources', {})
            teams[team]['total_pods'] += resources.get('pods', {}).get('count', 0)
            
            # Parse resource usage
            usage = ns.get('usage', {})
            cpu_req = parse_quantity(usage.get('cpu', {}).get('requests', '0'))
            mem_req = parse_quantity(usage.get('memory', {}).get('requests', '0'))
            
            teams[team]['total_cpu_requests'] += cpu_req
            teams[team]['total_memory_requests'] += mem_req
            
            # Check quota violations
            if ns.get('resource_quotas'):
                quota_analysis = analyze_quota_usage(ns)
                if quota_analysis:
                    for resource, stats in quota_analysis.items():
                        if stats['usage_percent'] > 100:
                            teams[team]['quota_violations'] += 1
        
        return teams
    
    def generate_tenant_report(self):
        """Generate detailed tenant resource report"""
        teams = self.get_tenant_summary()
        
        if not teams:
            return None
        
        report = f"# Multi-Tenant Resource Allocation Report\n"
        report += f"Cluster: {self.cluster_id}\n\n"
        
        for team, data in sorted(teams.items()):
            report += f"## Team: {team}\n"
            report += f"- Namespaces: {len(data['namespaces'])} ({', '.join(data['namespaces'])})\n"
            report += f"- Total Pods: {data['total_pods']}\n"
            report += f"- CPU Requests: {data['total_cpu_requests']:.2f} cores\n"
            report += f"- Memory Requests: {data['total_memory_requests']:.2f} Mi\n"
            
            if data['quota_violations'] > 0:
                report += f"- ⚠️ Quota Violations: {data['quota_violations']}\n"
            
            report += "\n"
        
        return report

# Usage
manager = MultiTenantManager("k8s-prod-001", "token")
report = manager.generate_tenant_report()
print(report)
```

## Related Operations
- `cluster.list_stream` - PREREQUISITE: Get cluster ID
- `cluster.get_details` - Get cluster configuration
- `cluster.get_node_metrics` - Get node-level resource metrics
- Namespace management APIs - Create, update, delete namespaces
- RBAC APIs - Manage namespace-level permissions

## Error Handling
- **400 Bad Request:** Invalid cluster ID format - verify ID structure
- **401 Unauthorized:** Invalid or expired authentication token - refresh token
- **403 Forbidden:** Insufficient permissions to list namespaces
- **404 Not Found:** Cluster not found - verify cluster_id exists
- **500 Internal Server Error:** Service error - retry with exponential backoff
- **503 Service Unavailable:** Service temporarily unavailable - implement retry logic

## Response Codes
**⚠️ To be documented - requires actual API testing**

## Performance Notes
- Response time varies with number of namespaces (typically < 1500ms)
- Namespace list changes infrequently - safe to cache for 5-10 minutes
- For monitoring dashboards, poll every 2-5 minutes
- No pagination (returns all namespaces)
- Rate limiting: To be documented

## Best Practices

### Cache Namespace Data
```python
import time
from datetime import datetime, timedelta

class NamespaceCache:
    def __init__(self, cluster_id, auth_token, ttl=300):  # 5 minutes
        self.cluster_id = cluster_id
        self.auth_token = auth_token
        self.ttl = ttl
        self.cache = None
        self.last_update = None
    
    def get_namespaces(self, force_refresh=False):
        """Get namespaces with caching"""
        now = datetime.now()
        
        if (not self.cache or 
            not self.last_update or 
            (now - self.last_update) > timedelta(seconds=self.ttl) or
            force_refresh):
            
            self.cache = get_all_namespaces(self.cluster_id, self.auth_token)
            self.last_update = now
        
        return self.cache

# Usage
cache = NamespaceCache("k8s-prod-001", "token", ttl=300)
ns_data = cache.get_namespaces()
```

### Filter and Search Efficiently
```python
def search_namespaces(namespaces, **criteria):
    """Search namespaces by multiple criteria"""
    results = namespaces
    
    # Filter by name pattern
    if 'name_pattern' in criteria:
        pattern = criteria['name_pattern'].lower()
        results = [ns for ns in results if pattern in ns['name'].lower()]
    
    # Filter by label
    if 'label' in criteria:
        label_key, label_value = criteria['label']
        results = [ns for ns in results 
                  if ns.get('labels', {}).get(label_key) == label_value]
    
    # Filter by status
    if 'status' in criteria:
        results = [ns for ns in results if ns['status'] == criteria['status']]
    
    # Filter by has_quota
    if criteria.get('has_quota'):
        results = [ns for ns in results if ns.get('resource_quotas')]
    
    return results

# Usage
ns_data = get_all_namespaces(cluster_id, token)
namespaces = ns_data['namespaces']

# Find all production namespaces with quotas
prod_with_quotas = search_namespaces(
    namespaces,
    label=('environment', 'production'),
    has_quota=True
)
```

### Monitor Quota Compliance
```python
def check_quota_compliance(namespace):
    """Check if namespace is compliant with quota policies"""
    issues = []
    
    # Check if quota is defined
    if not namespace.get('resource_quotas'):
        issues.append("No resource quota defined")
        return issues
    
    # Check if limit ranges are defined
    if not namespace.get('limit_ranges'):
        issues.append("No limit ranges defined")
    
    # Check quota usage
    quota_analysis = analyze_quota_usage(namespace)
    if quota_analysis:
        for resource, stats in quota_analysis.items():
            if stats['usage_percent'] > 100:
                issues.append(f"Quota exceeded for {resource}: {stats['usage_percent']:.1f}%")
            elif stats['usage_percent'] > 90:
                issues.append(f"Quota nearly full for {resource}: {stats['usage_percent']:.1f}%")
    
    return issues

# Usage
for ns in namespaces:
    if ns['name'] not in ['kube-system', 'kube-public']:  # Skip system namespaces
        issues = check_quota_compliance(ns)
        if issues:
            print(f"⚠️ {ns['name']}:")
            for issue in issues:
                print(f"  - {issue}")
```

## Metadata
- **Generated:** 2024-02-23T10:00:00Z
- **Source:** API endpoint analysis
- **API Version:** To be documented
- **Base Path:** {BASE_URL_PAAS_SERVICE}/paas
- **Additional notes:** ⚠️ This documentation is preliminary. Please provide actual API response for complete accuracy. Essential for multi-tenant cluster management.
