# API Specification: managed_service - list

**Resource:** managed_service
**Operation:** list
**Aliases:** managed services, list services, get services, platform services, IKS, GitLab services

## Endpoint
- **Method:** GET
- **URL:** https://ipcloud.tatacommunications.com/paasservice/api/v1/paas/listManagedServices/{service_type}
- **Auth:** Bearer token (from Keycloak)
- **Description:** List all managed services of a specific type (e.g., IKSGitlab, Kafka, PostgreSQL). Returns service instances with their configuration, status, and resource utilization

## Required Parameters
- `service_type` - Type of managed service (path parameter)
  - Examples: `IKSGitlab`, `Kafka`, `PostgreSQL`, `MySQL`, `Redis`, `MongoDB`, `Elasticsearch`

## Optional Parameters
- `status` - Filter by status (running, stopped, creating, error)
- `region` - Filter by region/zone
- `include_metrics` - Include resource metrics (true/false)

## Response Mapping
- `status`: status
- `message`: message
- `service_type`: data.serviceType
- `total_count`: data.totalCount
- `services`: data.services
- `service_ids`: data.services[*].id
- `service_names`: data.services[*].name
- `service_statuses`: data.services[*].status
- `service_urls`: data.services[*].url
- `service_versions`: data.services[*].version

## Response Example (IKSGitlab)
```json
{
  "status": "success",
  "data": {
    "serviceType": "IKSGitlab",
    "serviceDescription": "Integrated Kubernetes Service with GitLab CI/CD",
    "totalCount": 12,
    "services": [
      {
        "id": "iks-gitlab-prod-001",
        "name": "production-gitlab",
        "displayName": "Production GitLab",
        "type": "IKSGitlab",
        "status": "running",
        "health": "healthy",
        "version": "16.5.0-ee",
        "createdAt": "2024-01-15T10:30:00Z",
        "lastModified": "2025-02-13T08:45:00Z",
        "url": "https://gitlab.prod.company.com",
        "region": "mumbai-bkc",
        "zone": "EP_V2_MUM_BKC",
        "cluster": {
          "id": "cluster-1267",
          "name": "prod-k8s-01",
          "nodesCount": 5,
          "masterNodes": 3,
          "workerNodes": 2
        },
        "resources": {
          "vcpus": 16,
          "memoryGB": 64,
          "storageGB": 1000,
          "podCount": 25,
          "serviceCount": 15
        },
        "configuration": {
          "gitlab": {
            "runners": 5,
            "activeUsers": 150,
            "projects": 85,
            "groups": 12,
            "ciPipelines": 450,
            "containerRegistry": true,
            "pagesEnabled": true,
            "lfsEnabled": true
          },
          "kubernetes": {
            "namespace": "gitlab-production",
            "ingressEnabled": true,
            "tlsEnabled": true,
            "loadBalancerIP": "10.0.1.100",
            "storageClass": "ssd-premium"
          },
          "integrations": [
            "slack",
            "jira",
            "prometheus",
            "grafana"
          ]
        },
        "metrics": {
          "cpu": {
            "usage": 45.2,
            "limit": 100,
            "percentage": 45.2
          },
          "memory": {
            "usageGB": 38.5,
            "limitGB": 64,
            "percentage": 60.2
          },
          "storage": {
            "usedGB": 650,
            "totalGB": 1000,
            "percentage": 65.0
          },
          "network": {
            "inboundMbps": 125,
            "outboundMbps": 180
          }
        },
        "backup": {
          "enabled": true,
          "frequency": "daily",
          "retention": 30,
          "lastBackup": "2025-02-13T02:00:00Z",
          "status": "success"
        },
        "cost": {
          "monthly": 5000,
          "currency": "USD",
          "billingPeriod": "2025-02"
        },
        "tags": [
          "production",
          "gitlab",
          "cicd",
          "enterprise"
        ]
      },
      {
        "id": "iks-gitlab-staging-001",
        "name": "staging-gitlab",
        "displayName": "Staging GitLab",
        "type": "IKSGitlab",
        "status": "running",
        "health": "healthy",
        "version": "16.5.0-ee",
        "createdAt": "2024-02-01T14:20:00Z",
        "lastModified": "2025-02-12T16:30:00Z",
        "url": "https://gitlab.staging.company.com",
        "region": "delhi",
        "zone": "EP_V2_DEL",
        "cluster": {
          "id": "cluster-1422",
          "name": "staging-k8s-01",
          "nodesCount": 3,
          "masterNodes": 1,
          "workerNodes": 2
        },
        "resources": {
          "vcpus": 8,
          "memoryGB": 32,
          "storageGB": 500,
          "podCount": 15,
          "serviceCount": 10
        },
        "configuration": {
          "gitlab": {
            "runners": 3,
            "activeUsers": 50,
            "projects": 35,
            "groups": 5,
            "ciPipelines": 200,
            "containerRegistry": true,
            "pagesEnabled": false,
            "lfsEnabled": true
          },
          "kubernetes": {
            "namespace": "gitlab-staging",
            "ingressEnabled": true,
            "tlsEnabled": true,
            "loadBalancerIP": "10.0.2.100",
            "storageClass": "ssd"
          }
        },
        "metrics": {
          "cpu": {
            "usage": 25.5,
            "limit": 100,
            "percentage": 25.5
          },
          "memory": {
            "usageGB": 18.2,
            "limitGB": 32,
            "percentage": 56.9
          },
          "storage": {
            "usedGB": 280,
            "totalGB": 500,
            "percentage": 56.0
          }
        },
        "backup": {
          "enabled": true,
          "frequency": "daily",
          "retention": 7,
          "lastBackup": "2025-02-13T02:30:00Z",
          "status": "success"
        },
        "cost": {
          "monthly": 2500,
          "currency": "USD"
        },
        "tags": [
          "staging",
          "gitlab",
          "cicd"
        ]
      }
    ],
    "summary": {
      "totalServices": 12,
      "runningServices": 10,
      "stoppedServices": 1,
      "errorServices": 1,
      "totalCostMonthly": 45000,
      "totalResources": {
        "vcpus": 128,
        "memoryGB": 512,
        "storageGB": 8000
      }
    }
  },
  "message": "Managed services retrieved successfully",
  "responseCode": 0
}
```

## Response Example (Kafka)
```json
{
  "status": "success",
  "data": {
    "serviceType": "Kafka",
    "serviceDescription": "Apache Kafka Managed Service",
    "totalCount": 8,
    "services": [
      {
        "id": "kafka-prod-001",
        "name": "production-kafka",
        "displayName": "Production Kafka Cluster",
        "type": "Kafka",
        "status": "running",
        "health": "healthy",
        "version": "3.4.0",
        "createdAt": "2024-01-20T14:45:00Z",
        "url": "kafka-prod-001.company.internal:9092",
        "region": "mumbai-bkc",
        "zone": "EP_V2_MUM_BKC",
        "cluster": {
          "id": "cluster-1267",
          "name": "prod-k8s-01",
          "nodesCount": 3
        },
        "resources": {
          "vcpus": 24,
          "memoryGB": 96,
          "storageGB": 3000,
          "podCount": 6
        },
        "configuration": {
          "kafka": {
            "brokers": 3,
            "topics": 50,
            "partitions": 150,
            "consumerGroups": 12,
            "replicationFactor": 3,
            "minInsyncReplicas": 2,
            "zookeeperNodes": 3,
            "tlsEnabled": true,
            "saslEnabled": true,
            "compressionType": "snappy"
          },
          "kubernetes": {
            "namespace": "kafka-production",
            "storageClass": "ssd-premium",
            "persistentVolumes": 3
          }
        },
        "metrics": {
          "messagesPerSec": 125000,
          "bytesInPerSec": 125000000,
          "bytesOutPerSec": 180000000,
          "activeConnections": 250,
          "consumerLag": 1500,
          "storage": {
            "usedGB": 1850,
            "totalGB": 3000,
            "percentage": 61.7
          }
        },
        "cost": {
          "monthly": 7200,
          "currency": "USD"
        },
        "tags": [
          "production",
          "kafka",
          "messaging",
          "critical"
        ]
      }
    ]
  },
  "message": "Kafka services retrieved successfully",
  "responseCode": 0
}
```

## Response Fields Details

### Service Fields
- **id** - Unique service instance identifier
- **name** - Service instance name
- **displayName** - Human-readable display name
- **type** - Service type (IKSGitlab, Kafka, etc.)
- **status** - Current status (running, stopped, creating, error)
- **health** - Health status (healthy, degraded, unhealthy)
- **version** - Service version
- **createdAt** - Creation timestamp
- **lastModified** - Last modification timestamp
- **url** - Service access URL
- **region** - Geographic region
- **zone** - Technical zone code
- **cluster** - Associated Kubernetes cluster
- **resources** - Resource allocation
- **configuration** - Service-specific configuration
- **metrics** - Real-time metrics
- **backup** - Backup configuration
- **cost** - Cost information
- **tags** - Service tags

### Service Status Values
- `running` - Service is operational
- `stopped` - Service is stopped
- `creating` - Service is being provisioned
- `updating` - Service is being updated
- `error` - Service has encountered an error
- `deleting` - Service is being deleted

### Health Status Values
- `healthy` - All checks passing
- `degraded` - Some issues but operational
- `unhealthy` - Critical issues
- `unknown` - Health check unavailable

## Permissions
Roles: admin, developer, viewer

## Workflow Steps
### Workflow: list_managed_services
List managed services by type
- Step 1: authenticate (auth.validate_token)
- Step 2: list_managed_services (managed_service.list) (depends on: service_type)

## Usage Notes
- Service type is case-sensitive
- Returns only services user has access to
- Metrics updated every 5 minutes
- Cost data updated hourly
- Different service types have different configuration structures
- Use tags for filtering within service types

## Common Use Cases
1. **Service inventory**: "List all GitLab instances"
2. **Kafka clusters**: "Show all Kafka services"
3. **Cost analysis**: "What's the total cost of all GitLab services?"
4. **Health monitoring**: "Which services are unhealthy?"
5. **Resource utilization**: "Show GitLab storage usage"
6. **Version tracking**: "Which services need upgrades?"

## Data Processing Examples

### Get Service by Name
```python
response = list_managed_services("IKSGitlab")
services = response['data']['services']

prod_gitlab = next(
    (s for s in services if s['name'] == 'production-gitlab'),
    None
)

if prod_gitlab:
    print(f"GitLab URL: {prod_gitlab['url']}")
    print(f"Status: {prod_gitlab['status']}")
    print(f"Users: {prod_gitlab['configuration']['gitlab']['activeUsers']}")
```

### Calculate Total Costs
```python
response = list_managed_services("Kafka")
services = response['data']['services']

total_cost = sum(s['cost']['monthly'] for s in services)
print(f"Total Kafka monthly cost: ${total_cost:,}")

# By environment
prod_cost = sum(
    s['cost']['monthly'] for s in services
    if 'production' in s.get('tags', [])
)
print(f"Production Kafka cost: ${prod_cost:,}")
```

### Find Services Needing Attention
```python
response = list_managed_services("IKSGitlab")
services = response['data']['services']

# High storage usage
high_storage = [
    s for s in services
    if s['metrics']['storage']['percentage'] > 80
]

# Unhealthy services
unhealthy = [
    s for s in services
    if s['health'] != 'healthy'
]

# Outdated versions
current_version = "16.5.0-ee"
outdated = [
    s for s in services
    if s['version'] != current_version
]

print(f"High storage: {len(high_storage)}")
print(f"Unhealthy: {len(unhealthy)}")
print(f"Outdated: {len(outdated)}")
```

### Group by Region
```python
from collections import defaultdict

response = list_managed_services("Kafka")
by_region = defaultdict(list)

for service in response['data']['services']:
    by_region[service['region']].append(service)

for region, services in by_region.items():
    total_brokers = sum(
        s['configuration']['kafka']['brokers']
        for s in services
    )
    print(f"{region}: {len(services)} clusters, {total_brokers} brokers")
```

## Kafka-Specific Analysis

### Kafka Cluster Overview
```python
response = list_managed_services("Kafka")

print("Kafka Cluster Overview:")
print("-" * 70)

for service in response['data']['services']:
    kafka_config = service['configuration']['kafka']
    metrics = service['metrics']
    
    print(f"\n{service['displayName']}")
    print(f"  Brokers: {kafka_config['brokers']}")
    print(f"  Topics: {kafka_config['topics']}")
    print(f"  Partitions: {kafka_config['partitions']}")
    print(f"  Throughput: {metrics['messagesPerSec']:,} msg/s")
    print(f"  Consumer Lag: {metrics['consumerLag']}")
    print(f"  Storage: {metrics['storage']['percentage']:.1f}%")
    print(f"  Cost: ${service['cost']['monthly']:,}/month")
```

### Kafka Capacity Planning
```python
def analyze_kafka_capacity(service):
    kafka = service['configuration']['kafka']
    metrics = service['metrics']
    storage = metrics['storage']
    
    # Topic capacity
    avg_partitions_per_topic = kafka['partitions'] / kafka['topics']
    
    # Storage growth
    used_gb = storage['usedGB']
    total_gb = storage['totalGB']
    usage_pct = storage['percentage']
    
    days_until_full = (total_gb - used_gb) / 5.2  # Assume 5.2 GB/day growth
    
    return {
        'name': service['name'],
        'topicCapacity': f"{kafka['topics']}/{kafka['topics']*2}",  # Assume 2x headroom
        'storageStatus': 'good' if usage_pct < 70 else 'warning' if usage_pct < 85 else 'critical',
        'daysUntilFull': max(0, days_until_full),
        'throughputUtilization': metrics['messagesPerSec'] / 400000  # Assume 400k max
    }
```

## GitLab-Specific Analysis

### GitLab CI/CD Metrics
```python
response = list_managed_services("IKSGitlab")

for service in response['data']['services']:
    gitlab = service['configuration']['gitlab']
    
    pipelines_per_project = gitlab['ciPipelines'] / gitlab['projects']
    users_per_project = gitlab['activeUsers'] / gitlab['projects']
    
    print(f"{service['name']}:")
    print(f"  {gitlab['ciPipelines']} pipelines across {gitlab['projects']} projects")
    print(f"  {pipelines_per_project:.1f} pipelines/project avg")
    print(f"  {gitlab['runners']} runners for {gitlab['activeUsers']} users")
```

## Related Operations
- `managed_service.get` - Get specific service details
- `managed_service.create` - Create new managed service
- `managed_service.update` - Update service configuration
- `managed_service.delete` - Delete service
- `managed_service.restart` - Restart service
- `managed_service.backup` - Trigger backup

## Error Handling
- **400 Bad Request:** Invalid service type
- **401 Unauthorized:** Invalid or expired authentication token
- **403 Forbidden:** User does not have access to these services
- **404 Not Found:** Service type not supported
- **500 Internal Server Error:** Service unavailable

## Response Codes
- `0` - Success
- `1` - Service type not found
- `2` - No services found
- `3` - Access denied

## Supported Service Types

### Currently Available
- `IKSGitlab` - Integrated Kubernetes Service with GitLab
- `Kafka` - Apache Kafka messaging
- `PostgreSQL` - PostgreSQL database
- `MySQL` - MySQL database
- `Redis` - Redis cache/database
- `MongoDB` - MongoDB document database
- `Elasticsearch` - Elasticsearch search engine
- `RabbitMQ` - RabbitMQ message broker

### Service Type Format
- Case-sensitive
- Use exact names as shown
- Check API documentation for new service types

## Performance Notes
- Response time varies by service count (typically < 1s for < 50 services)
- Metrics are cached for 5 minutes
- Cost data cached for 1 hour
- Large deployments (100+ services) may be paginated

## Best Practices

### Cache Service Lists
```python
service_cache = {}

def get_services_cached(service_type, ttl_minutes=10):
    from datetime import datetime, timedelta
    
    now = datetime.now()
    
    if service_type in service_cache:
        data, expires = service_cache[service_type]
        if now < expires:
            return data
    
    data = list_managed_services(service_type)
    service_cache[service_type] = (data, now + timedelta(minutes=ttl_minutes))
    
    return data
```

### Monitor Critical Services
```python
def check_critical_services():
    critical_types = ['Kafka', 'PostgreSQL', 'IKSGitlab']
    alerts = []
    
    for service_type in critical_types:
        response = list_managed_services(service_type)
        
        for service in response['data']['services']:
            if service['health'] != 'healthy':
                alerts.append({
                    'service': service['name'],
                    'type': service_type,
                    'health': service['health'],
                    'status': service['status']
                })
    
    return alerts
```

## Metadata
- **Generated:** 2025-02-13T12:30:00Z
- **Source:** Dynamic API Spec Generator
- **API Version:** v1
- **Base Path:** /paasservice/api/v1/paas
- **Service Types:** Extensible, check docs for additions
- **Kafka Support:** Full Kafka cluster management and monitoring