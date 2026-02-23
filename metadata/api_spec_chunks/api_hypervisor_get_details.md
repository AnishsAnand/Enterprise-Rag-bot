# API Specification: hypervisor - get_details

**Resource:** hypervisor
**Operation:** get_details
**Aliases:** hypervisor, hypervisor details, compute hosts, host details, virtualization hosts, ESXi hosts, KVM hosts, hypervisor info

## Endpoint
- **Method:** GET
- **URL:** {BASE_URL_PORTAL_SERVICE}/configservice/hypervisordetails
- **Auth:** Bearer token (from Keycloak)
- **Description:** Get comprehensive details about all hypervisors (compute hosts) in the infrastructure including resource capacity, utilization, VM placement, hypervisor type, version, and health status

## Required Parameters
None (returns all accessible hypervisors)

## Optional Parameters
- `endpoint` - Filter by endpoint/location code (query parameter)
- `zone_id` - Filter by zone (query parameter)
- `status` - Filter by status: online, offline, maintenance (query parameter)
- `hypervisor_type` - Filter by type: vmware, kvm, openstack (query parameter)

## Response Mapping
- `status`: status
- `message`: message
- `responseCode`: responseCode
- `hypervisors`: data.hypervisors
- `hypervisor_id`: data.hypervisors[*].hypervisorId
- `hypervisor_name`: data.hypervisors[*].hypervisorName
- `hypervisor_type`: data.hypervisors[*].hypervisorType
- `hypervisor_version`: data.hypervisors[*].version
- `endpoint_code`: data.hypervisors[*].endpointCode
- `zone_id`: data.hypervisors[*].zoneId
- `status`: data.hypervisors[*].status
- `total_cpu_cores`: data.hypervisors[*].resources.cpu.total
- `used_cpu_cores`: data.hypervisors[*].resources.cpu.used
- `total_memory_gb`: data.hypervisors[*].resources.memory.totalGB
- `used_memory_gb`: data.hypervisors[*].resources.memory.usedGB
- `total_storage_gb`: data.hypervisors[*].resources.storage.totalGB
- `vm_count`: data.hypervisors[*].vmCount
- `cpu_utilization_percent`: data.hypervisors[*].utilization.cpuPercent
- `memory_utilization_percent`: data.hypervisors[*].utilization.memoryPercent

## Response Example
```json
{
  "status": "success",
  "data": {
    "totalHypervisors": 15,
    "onlineHypervisors": 14,
    "offlineHypervisors": 0,
    "maintenanceHypervisors": 1,
    "hypervisors": [
      {
        "hypervisorId": "HV001",
        "hypervisorName": "esxi-blr-01.ipc.local",
        "hypervisorType": "vmware",
        "version": "ESXi 7.0 U3",
        "buildNumber": "20328353",
        "endpointCode": "EP_V2_BL",
        "endpointDisplayName": "Bengaluru",
        "zoneId": "PROD",
        "zoneName": "Production",
        "status": "online",
        "uptime": 2592000,
        "lastBootTime": 1762598613000,
        "managementIP": "192.168.1.10",
        "hostname": "esxi-blr-01.ipc.local",
        "resources": {
          "cpu": {
            "model": "Intel Xeon Gold 6248R",
            "total": 96,
            "used": 64,
            "available": 32,
            "reserved": 4,
            "sockets": 2,
            "coresPerSocket": 24,
            "threadsPerCore": 2,
            "speedMHz": 3000
          },
          "memory": {
            "totalGB": 512,
            "usedGB": 340,
            "availableGB": 172,
            "reservedGB": 20,
            "swapGB": 0,
            "balloonedGB": 0
          },
          "storage": {
            "totalGB": 10240,
            "usedGB": 6500,
            "availableGB": 3740,
            "datastores": [
              {
                "name": "datastore1",
                "type": "vmfs",
                "capacityGB": 5120,
                "freeGB": 2000,
                "provisionedGB": 3500
              },
              {
                "name": "datastore2",
                "type": "nfs",
                "capacityGB": 5120,
                "freeGB": 1740,
                "provisionedGB": 3000
              }
            ]
          },
          "network": {
            "totalNICs": 4,
            "activePorts": 48,
            "vSwitches": 2,
            "portGroups": 8,
            "bandwidth": "40Gbps"
          }
        },
        "utilization": {
          "cpuPercent": 66.67,
          "memoryPercent": 66.41,
          "storagePercent": 63.48,
          "networkPercent": 35.0
        },
        "vmCount": 24,
        "vms": [
          {
            "vmId": "VM001",
            "vmName": "web-server-1",
            "powerState": "on",
            "cpuCount": 4,
            "memoryGB": 16,
            "storageGB": 200
          },
          {
            "vmId": "VM002",
            "vmName": "db-server-1",
            "powerState": "on",
            "cpuCount": 8,
            "memoryGB": 32,
            "storageGB": 500
          }
        ],
        "cluster": {
          "clusterId": "CL001",
          "clusterName": "prod-cluster",
          "haEnabled": true,
          "drsEnabled": true,
          "vMotionEnabled": true
        },
        "healthStatus": {
          "overall": "green",
          "cpu": "green",
          "memory": "yellow",
          "storage": "green",
          "network": "green",
          "sensors": {
            "temperature": "normal",
            "power": "normal",
            "fan": "normal"
          }
        },
        "features": {
          "vMotion": true,
          "ha": true,
          "drs": true,
          "ftEnabled": false,
          "vsan": false,
          "nsx": true
        },
        "licenses": {
          "edition": "Enterprise Plus",
          "expiryDate": 1798764000000,
          "isValid": true
        }
      },
      {
        "hypervisorId": "HV002",
        "hypervisorName": "kvm-blr-01.ipc.local",
        "hypervisorType": "kvm",
        "version": "QEMU 6.2.0",
        "kernelVersion": "5.15.0-56-generic",
        "endpointCode": "EP_V2_BL",
        "endpointDisplayName": "Bengaluru",
        "zoneId": "DEV",
        "zoneName": "Development",
        "status": "online",
        "uptime": 1728000,
        "lastBootTime": 1763462613000,
        "managementIP": "192.168.2.10",
        "hostname": "kvm-blr-01.ipc.local",
        "resources": {
          "cpu": {
            "model": "AMD EPYC 7543",
            "total": 64,
            "used": 32,
            "available": 32,
            "reserved": 2,
            "sockets": 1,
            "coresPerSocket": 32,
            "threadsPerCore": 2,
            "speedMHz": 2800
          },
          "memory": {
            "totalGB": 256,
            "usedGB": 128,
            "availableGB": 128,
            "reservedGB": 10,
            "swapGB": 4,
            "balloonedGB": 0
          },
          "storage": {
            "totalGB": 5120,
            "usedGB": 2000,
            "availableGB": 3120,
            "datastores": [
              {
                "name": "/dev/sda",
                "type": "lvm",
                "capacityGB": 5120,
                "freeGB": 3120,
                "provisionedGB": 2500
              }
            ]
          },
          "network": {
            "totalNICs": 2,
            "activePorts": 16,
            "bridges": 2,
            "bandwidth": "20Gbps"
          }
        },
        "utilization": {
          "cpuPercent": 50.0,
          "memoryPercent": 50.0,
          "storagePercent": 39.06,
          "networkPercent": 20.0
        },
        "vmCount": 12,
        "vms": [
          {
            "vmId": "VM015",
            "vmName": "dev-app-1",
            "powerState": "on",
            "cpuCount": 2,
            "memoryGB": 8,
            "storageGB": 100
          }
        ],
        "healthStatus": {
          "overall": "green",
          "cpu": "green",
          "memory": "green",
          "storage": "green",
          "network": "green",
          "sensors": {
            "temperature": "normal",
            "power": "normal",
            "fan": "normal"
          }
        },
        "features": {
          "liveMigration": true,
          "snapshots": true,
          "hotPlug": true
        }
      },
      {
        "hypervisorId": "HV003",
        "hypervisorName": "esxi-sg-01.ipc.local",
        "hypervisorType": "vmware",
        "version": "ESXi 7.0 U3",
        "buildNumber": "20328353",
        "endpointCode": "EP_V2_SG_TCX",
        "endpointDisplayName": "Singapore East",
        "zoneId": "PROD",
        "zoneName": "Production",
        "status": "maintenance",
        "uptime": 432000,
        "lastBootTime": 1764758613000,
        "managementIP": "192.168.10.10",
        "hostname": "esxi-sg-01.ipc.local",
        "resources": {
          "cpu": {
            "model": "Intel Xeon Gold 6248R",
            "total": 96,
            "used": 48,
            "available": 48,
            "reserved": 4,
            "sockets": 2,
            "coresPerSocket": 24,
            "threadsPerCore": 2,
            "speedMHz": 3000
          },
          "memory": {
            "totalGB": 512,
            "usedGB": 200,
            "availableGB": 312,
            "reservedGB": 20,
            "swapGB": 0,
            "balloonedGB": 0
          },
          "storage": {
            "totalGB": 10240,
            "usedGB": 3000,
            "availableGB": 7240,
            "datastores": [
              {
                "name": "datastore-sg1",
                "type": "vmfs",
                "capacityGB": 10240,
                "freeGB": 7240,
                "provisionedGB": 3500
              }
            ]
          },
          "network": {
            "totalNICs": 4,
            "activePorts": 32,
            "vSwitches": 2,
            "portGroups": 6,
            "bandwidth": "40Gbps"
          }
        },
        "utilization": {
          "cpuPercent": 50.0,
          "memoryPercent": 39.06,
          "storagePercent": 29.30,
          "networkPercent": 15.0
        },
        "vmCount": 16,
        "vms": [],
        "cluster": {
          "clusterId": "CL002",
          "clusterName": "sg-cluster",
          "haEnabled": true,
          "drsEnabled": true,
          "vMotionEnabled": true
        },
        "healthStatus": {
          "overall": "yellow",
          "cpu": "green",
          "memory": "green",
          "storage": "green",
          "network": "green",
          "sensors": {
            "temperature": "normal",
            "power": "normal",
            "fan": "normal"
          },
          "maintenanceReason": "Scheduled firmware upgrade"
        },
        "features": {
          "vMotion": true,
          "ha": true,
          "drs": true,
          "ftEnabled": false,
          "vsan": false,
          "nsx": true
        },
        "licenses": {
          "edition": "Enterprise Plus",
          "expiryDate": 1798764000000,
          "isValid": true
        }
      }
    ],
    "aggregateStats": {
      "totalCPU": 256,
      "usedCPU": 144,
      "totalMemoryGB": 1280,
      "usedMemoryGB": 668,
      "totalStorageGB": 25600,
      "usedStorageGB": 11500,
      "totalVMs": 52,
      "averageCPUUtilization": 56.25,
      "averageMemoryUtilization": 52.19
    }
  },
  "message": "success",
  "responseCode": 0
}
```

## Response Fields Details

### Hypervisor Fields
- **hypervisorId** - Unique identifier (string)
- **hypervisorName** - Hostname/FQDN (string)
- **hypervisorType** - Type: vmware, kvm, openstack, hyperv (string)
- **version** - Hypervisor software version (string)
- **buildNumber** - Build number (VMware) (string)
- **kernelVersion** - Kernel version (KVM/Linux) (string)
- **endpointCode** - Location code (string)
- **endpointDisplayName** - Location name (string)
- **zoneId** - Zone identifier (string)
- **zoneName** - Zone name (string)
- **status** - Status: online, offline, maintenance (string)
- **uptime** - Uptime in seconds (number)
- **lastBootTime** - Unix timestamp of last boot (number)
- **managementIP** - Management IP address (string)
- **hostname** - Hostname (string)

### Resource Capacity
- **cpu.total** - Total CPU cores (number)
- **cpu.used** - Used cores (number)
- **cpu.available** - Available cores (number)
- **cpu.model** - CPU model name (string)
- **memory.totalGB** - Total RAM in GB (number)
- **memory.usedGB** - Used RAM in GB (number)
- **storage.totalGB** - Total storage in GB (number)
- **storage.usedGB** - Used storage in GB (number)

### Utilization
- **cpuPercent** - CPU utilization percentage (number)
- **memoryPercent** - Memory utilization percentage (number)
- **storagePercent** - Storage utilization percentage (number)
- **networkPercent** - Network utilization percentage (number)

### VM Information
- **vmCount** - Number of VMs on host (number)
- **vms** - Array of VM objects

### Health Status
- **overall** - Overall health: green, yellow, red (string)
- **cpu** - CPU health (string)
- **memory** - Memory health (string)
- **storage** - Storage health (string)
- **network** - Network health (string)
- **sensors** - Hardware sensor status (object)

## Permissions
Roles: admin, developer (viewer may have limited access)

## Workflow Steps
### Workflow: get_hypervisor_details
Get comprehensive hypervisor information
- Step 1: authenticate_user (user.authenticate)
- Step 2: get_accessible_endpoints (endpoint.list_by_user)
- Step 3: get_hypervisor_details (hypervisor.get_details)
- Step 4: get_vm_placement (vm.list_by_hypervisor)
- Step 5: calculate_utilization (hypervisor.calculate_stats)
- Step 6: format_response (response.format)

## Common Use Cases

1. **List all hypervisors**: "Show me all compute hosts"
2. **Check capacity**: "Which hypervisors have available capacity?"
3. **Find high utilization**: "Which hosts are over 80% CPU?"
4. **Check host status**: "Are all hypervisors online?"
5. **View VM placement**: "Where is VM xyz hosted?"
6. **Capacity planning**: "How much total CPU is available?"
7. **Filter by location**: "Show hypervisors in Bengaluru"
8. **Check maintenance**: "Which hosts are in maintenance?"
9. **View resources**: "Show total memory across all hosts"
10. **Health check**: "Are there any unhealthy hypervisors?"

## Query Interpretations
- "hypervisors" → GET /hypervisordetails
- "compute hosts" → GET /hypervisordetails
- "ESXi hosts" → GET /hypervisordetails?hypervisor_type=vmware
- "hosts in bengaluru" → GET /hypervisordetails?endpoint=EP_V2_BL
- "hosts in production" → GET /hypervisordetails?zone_id=PROD

## Data Processing Examples

### Find Available Capacity
```python
def find_available_capacity(hypervisor_data, min_cpu=4, min_memory=16):
    """Find hypervisors with available capacity."""
    suitable_hosts = []
    
    for hypervisor in hypervisor_data['data']['hypervisors']:
        if hypervisor['status'] != 'online':
            continue
        
        cpu_available = hypervisor['resources']['cpu']['available']
        mem_available = hypervisor['resources']['memory']['availableGB']
        
        if cpu_available >= min_cpu and mem_available >= min_memory:
            suitable_hosts.append({
                'name': hypervisor['hypervisorName'],
                'location': hypervisor['endpointDisplayName'],
                'cpu_available': cpu_available,
                'memory_available': mem_available,
                'vm_count': hypervisor['vmCount']
            })
    
    return suitable_hosts
```

### Check High Utilization Hosts
```python
def check_high_utilization(hypervisor_data, threshold=80):
    """Find hypervisors with high resource utilization."""
    high_util_hosts = []
    
    for hypervisor in hypervisor_data['data']['hypervisors']:
        util = hypervisor['utilization']
        
        if (util['cpuPercent'] >= threshold or 
            util['memoryPercent'] >= threshold):
            high_util_hosts.append({
                'name': hypervisor['hypervisorName'],
                'cpu_util': util['cpuPercent'],
                'memory_util': util['memoryPercent'],
                'vm_count': hypervisor['vmCount'],
                'status': hypervisor['status']
            })
    
    return high_util_hosts
```

### Generate Capacity Report
```python
def generate_capacity_report(hypervisor_data):
    """Generate overall capacity report."""
    hypervisors = hypervisor_data['data']['hypervisors']
    online = [h for h in hypervisors if h['status'] == 'online']
    
    report = {
        'total_hosts': len(hypervisors),
        'online_hosts': len(online),
        'total_capacity': {
            'cpu_cores': sum(h['resources']['cpu']['total'] for h in online),
            'memory_gb': sum(h['resources']['memory']['totalGB'] for h in online),
            'storage_gb': sum(h['resources']['storage']['totalGB'] for h in online)
        },
        'used_capacity': {
            'cpu_cores': sum(h['resources']['cpu']['used'] for h in online),
            'memory_gb': sum(h['resources']['memory']['usedGB'] for h in online),
            'storage_gb': sum(h['resources']['storage']['usedGB'] for h in online)
        },
        'total_vms': sum(h['vmCount'] for h in online),
        'by_location': {},
        'by_type': {}
    }
    
    # Calculate utilization
    report['utilization'] = {
        'cpu_percent': (report['used_capacity']['cpu_cores'] / 
                       report['total_capacity']['cpu_cores'] * 100),
        'memory_percent': (report['used_capacity']['memory_gb'] / 
                          report['total_capacity']['memory_gb'] * 100),
        'storage_percent': (report['used_capacity']['storage_gb'] / 
                           report['total_capacity']['storage_gb'] * 100)
    }
    
    # Group by location
    for h in online:
        location = h['endpointDisplayName']
        if location not in report['by_location']:
            report['by_location'][location] = {
                'hosts': 0,
                'vms': 0,
                'cpu_cores': 0,
                'memory_gb': 0
            }
        report['by_location'][location]['hosts'] += 1
        report['by_location'][location]['vms'] += h['vmCount']
        report['by_location'][location]['cpu_cores'] += h['resources']['cpu']['total']
        report['by_location'][location]['memory_gb'] += h['resources']['memory']['totalGB']
    
    # Group by type
    for h in online:
        hv_type = h['hypervisorType']
        report['by_type'][hv_type] = report['by_type'].get(hv_type, 0) + 1
    
    return report
```

## Integration Examples

### Python Client
```python
import requests
from typing import Dict, List, Optional

class HypervisorClient:
    """Client for IPC Cloud Hypervisor API."""
    
    def __init__(self, base_url: str, bearer_token: str):
        self.base_url = base_url
        self.headers = {
            'Authorization': f'Bearer {bearer_token}',
            'Content-Type': 'application/json'
        }
    
    def get_hypervisor_details(
        self,
        endpoint: Optional[str] = None,
        zone_id: Optional[str] = None,
        status: Optional[str] = None,
        hypervisor_type: Optional[str] = None
    ) -> Dict:
        """Get hypervisor details with optional filters."""
        url = f"{self.base_url}/portalservice/configservice/hypervisordetails"
        
        params = {}
        if endpoint:
            params['endpoint'] = endpoint
        if zone_id:
            params['zone_id'] = zone_id
        if status:
            params['status'] = status
        if hypervisor_type:
            params['hypervisor_type'] = hypervisor_type
        
        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        return response.json()
    
    def find_best_host(self, min_cpu: int, min_memory: int) -> Optional[Dict]:
        """Find hypervisor with best available capacity."""
        data = self.get_hypervisor_details(status='online')
        
        best_host = None
        best_score = 0
        
        for hypervisor in data['data']['hypervisors']:
            cpu_avail = hypervisor['resources']['cpu']['available']
            mem_avail = hypervisor['resources']['memory']['availableGB']
            
            if cpu_avail >= min_cpu and mem_avail >= min_memory:
                # Score based on available resources and current utilization
                cpu_util = hypervisor['utilization']['cpuPercent']
                mem_util = hypervisor['utilization']['memoryPercent']
                score = cpu_avail + mem_avail - (cpu_util + mem_util) / 2
                
                if score > best_score:
                    best_score = score
                    best_host = hypervisor
        
        return best_host
    
    def get_utilization_summary(self) -> Dict:
        """Get overall utilization summary."""
        data = self.get_hypervisor_details(status='online')
        hypervisors = data['data']['hypervisors']
        
        if not hypervisors:
            return {}
        
        return {
            'total_hosts': len(hypervisors),
            'average_cpu_util': sum(
                h['utilization']['cpuPercent'] for h in hypervisors
            ) / len(hypervisors),
            'average_memory_util': sum(
                h['utilization']['memoryPercent'] for h in hypervisors
            ) / len(hypervisors),
            'total_vms': sum(h['vmCount'] for h in hypervisors),
            'hosts_over_80_percent': sum(
                1 for h in hypervisors 
                if h['utilization']['cpuPercent'] >= 80 or 
                   h['utilization']['memoryPercent'] >= 80
            )
        }

# Usage example
if __name__ == '__main__':
    client = HypervisorClient(
        base_url='https://ipcloud.tatacommunications.com',
        bearer_token='your_token_here'
    )
    
    # Get all hypervisors
    details = client.get_hypervisor_details()
    print(f"Total hypervisors: {len(details['data']['hypervisors'])}")
    
    # Find best host for new VM
    best = client.find_best_host(min_cpu=4, min_memory=16)
    if best:
        print(f"Best host: {best['hypervisorName']}")
        print(f"Available: {best['resources']['cpu']['available']} CPU, "
              f"{best['resources']['memory']['availableGB']} GB RAM")
    
    # Get utilization summary
    summary = client.get_utilization_summary()
    print(f"Average CPU utilization: {summary['average_cpu_util']:.1f}%")
```

## Related Operations
- `vm.list` - List VMs
- `vm.migrate` - Migrate VM to different host
- `cluster.get_details` - Get cluster information
- `endpoint.list` - List endpoints/locations
- `resource.get_capacity` - Get capacity reports

## Error Handling

### Common Errors
- **401 Unauthorized**: Invalid bearer token
- **403 Forbidden**: No access to hypervisor data
- **500 Internal Server Error**: Service error

## Response Codes
- `0` - Success
- `1` - Authentication failed
- `2` - Authorization failed
- `500` - Internal server error

## Performance Notes
- **Response time**: 1-3 seconds depending on hypervisor count
- **Caching**: Data cached for 5 minutes
- **Rate limiting**: 100 requests per minute

## Metadata
- **Generated:** 2025-02-16T11:15:00.000000Z
- **Source:** IPC Cloud Portal Configuration Service
- **API Version:** v2
- **Base Path:** /portalservice/configservice/