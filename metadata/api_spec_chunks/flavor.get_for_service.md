# API Specification: flavor - get_for_service

**Resource:** flavor
**Operation:** get_for_service
**Aliases:** flavors, get flavors, service flavors, instance types, microservice flavors, kafka flavors

## Endpoint
- **Method:** GET
- **URL:** {BASE_URL_PAAS_SERVICE}/paas/getFlavorsForMsService/{service_name}
- **Auth:** Bearer token (from Keycloak)
- **Description:** Get available compute flavors (instance types/sizes) for a specific microservice, including optimized configurations for Kafka, databases, and other services

## Required Parameters
- `service_name` - Name of the microservice (path parameter, e.g., "kafka", "postgres", "redis", "mongodb")

## Optional Parameters
- `location` - Filter by location/zone
- `performance_tier` - Filter by tier (standard, high-performance, memory-optimized, storage-optimized)
- `include_pricing` - Include pricing information (true/false)

## Response Mapping
- `status`: status
- `message`: message
- `service_name`: data.serviceName
- `flavors`: data.flavors
- `flavor_ids`: data.flavors[*].id
- `flavor_names`: data.flavors[*].name
- `vcpus`: data.flavors[*].vcpus
- `memory_gb`: data.flavors[*].memoryGB
- `storage_gb`: data.flavors[*].storageGB
- `network_gbps`: data.flavors[*].networkGbps
- `recommended`: data.flavors[*].recommended
- `kafka_optimized`: data.flavors[*].kafkaOptimized

## Response Example
```json
{
  "status": "success",
  "data": {
    "serviceName": "kafka",
    "serviceType": "messaging",
    "totalFlavors": 8,
    "flavors": [
      {
        "id": "kafka-small",
        "name": "Kafka Small",
        "description": "Entry-level Kafka broker for dev/test",
        "vcpus": 4,
        "memoryGB": 16,
        "storageGB": 500,
        "storageType": "ssd",
        "networkGbps": 1,
        "iops": 3000,
        "tier": "standard",
        "recommended": false,
        "kafkaOptimized": true,
        "kafkaSpecs": {
          "maxPartitions": 1000,
          "maxTopics": 100,
          "maxThroughputMBps": 50,
          "maxMessagesPerSec": 50000,
          "recommendedBrokers": 3,
          "maxConsumers": 100
        },
        "pricing": {
          "hourly": 2.50,
          "monthly": 1800,
          "currency": "USD"
        },
        "available": true,
        "locations": ["mumbai-bkc", "delhi", "bengaluru"]
      },
      {
        "id": "kafka-medium",
        "name": "Kafka Medium",
        "description": "Production Kafka broker for moderate workloads",
        "vcpus": 8,
        "memoryGB": 32,
        "storageGB": 1000,
        "storageType": "ssd-premium",
        "networkGbps": 2,
        "iops": 6000,
        "tier": "high-performance",
        "recommended": true,
        "kafkaOptimized": true,
        "kafkaSpecs": {
          "maxPartitions": 2500,
          "maxTopics": 250,
          "maxThroughputMBps": 150,
          "maxMessagesPerSec": 150000,
          "recommendedBrokers": 3,
          "maxConsumers": 500,
          "replicationFactor": 3,
          "minInsyncReplicas": 2
        },
        "pricing": {
          "hourly": 5.00,
          "monthly": 3600,
          "currency": "USD"
        },
        "available": true,
        "locations": ["mumbai-bkc", "delhi", "bengaluru", "chennai-amb"]
      },
      {
        "id": "kafka-large",
        "name": "Kafka Large",
        "description": "High-throughput Kafka broker for large-scale production",
        "vcpus": 16,
        "memoryGB": 64,
        "storageGB": 2000,
        "storageType": "nvme-ssd",
        "networkGbps": 5,
        "iops": 15000,
        "tier": "high-performance",
        "recommended": false,
        "kafkaOptimized": true,
        "kafkaSpecs": {
          "maxPartitions": 5000,
          "maxTopics": 500,
          "maxThroughputMBps": 400,
          "maxMessagesPerSec": 400000,
          "recommendedBrokers": 3,
          "maxConsumers": 2000,
          "replicationFactor": 3,
          "minInsyncReplicas": 2
        },
        "pricing": {
          "hourly": 10.00,
          "monthly": 7200,
          "currency": "USD"
        },
        "available": true,
        "locations": ["mumbai-bkc", "chennai-amb"]
      },
      {
        "id": "kafka-xlarge",
        "name": "Kafka X-Large",
        "description": "Maximum-performance Kafka broker for extreme workloads",
        "vcpus": 32,
        "memoryGB": 128,
        "storageGB": 4000,
        "storageType": "nvme-ssd",
        "networkGbps": 10,
        "iops": 30000,
        "tier": "extreme-performance",
        "recommended": false,
        "kafkaOptimized": true,
        "kafkaSpecs": {
          "maxPartitions": 10000,
          "maxTopics": 1000,
          "maxThroughputMBps": 1000,
          "maxMessagesPerSec": 1000000,
          "recommendedBrokers": 3,
          "maxConsumers": 5000,
          "replicationFactor": 3,
          "minInsyncReplicas": 2
        },
        "pricing": {
          "hourly": 20.00,
          "monthly": 14400,
          "currency": "USD"
        },
        "available": true,
        "locations": ["mumbai-bkc", "chennai-amb"]
      }
    ],
    "recommendations": {
      "development": "kafka-small",
      "staging": "kafka-medium",
      "production": "kafka-medium",
      "highThroughput": "kafka-large",
      "enterprise": "kafka-xlarge"
    },
    "sizingGuide": {
      "smallWorkload": {
        "flavor": "kafka-small",
        "messagesPerSec": "< 50k",
        "throughputMBps": "< 50",
        "topics": "< 100"
      },
      "mediumWorkload": {
        "flavor": "kafka-medium",
        "messagesPerSec": "50k - 150k",
        "throughputMBps": "50 - 150",
        "topics": "100 - 250"
      },
      "largeWorkload": {
        "flavor": "kafka-large",
        "messagesPerSec": "150k - 400k",
        "throughputMBps": "150 - 400",
        "topics": "250 - 500"
      },
      "xlargeWorkload": {
        "flavor": "kafka-xlarge",
        "messagesPerSec": "> 400k",
        "throughputMBps": "> 400",
        "topics": "> 500"
      }
    }
  },
  "message": "Flavors retrieved successfully",
  "responseCode": 0
}
```

## Response Fields Details

### Flavor Fields
- **id** - Unique flavor identifier
- **name** - Human-readable flavor name
- **description** - Detailed flavor description
- **vcpus** - Number of virtual CPUs
- **memoryGB** - RAM in gigabytes
- **storageGB** - Included storage in gigabytes
- **storageType** - Storage technology (ssd, nvme-ssd, hdd)
- **networkGbps** - Network bandwidth in Gbps
- **iops** - Input/output operations per second
- **tier** - Performance tier classification
- **recommended** - Whether this is the recommended flavor
- **kafkaOptimized** - Whether optimized for Kafka workloads
- **kafkaSpecs** - Kafka-specific capacity specifications
- **pricing** - Cost information
- **available** - Whether currently available for provisioning
- **locations** - Available geographic locations

### Performance Tiers
- `standard` - Basic performance for dev/test
- `high-performance` - Production-grade performance
- `memory-optimized` - Enhanced memory for caching
- `storage-optimized` - High storage capacity
- `network-optimized` - High network throughput
- `extreme-performance` - Maximum performance tier

### Storage Types
- `hdd` - Traditional hard disk drives
- `ssd` - Standard solid-state drives
- `ssd-premium` - High-performance SSD
- `nvme-ssd` - NVMe-attached SSD (fastest)

## Kafka-Specific Specifications

### Kafka Capacity Metrics
```json
{
  "maxPartitions": 2500,
  "maxTopics": 250,
  "maxThroughputMBps": 150,
  "maxMessagesPerSec": 150000,
  "recommendedBrokers": 3,
  "maxConsumers": 500,
  "replicationFactor": 3,
  "minInsyncReplicas": 2
}
```

### Kafka Sizing Guidelines

**Small (kafka-small):**
- Dev/test environments
- < 50k messages/sec
- < 100 topics
- Single data center

**Medium (kafka-medium):**
- Production workloads
- 50k-150k messages/sec
- 100-250 topics
- Multi-AZ with replication

**Large (kafka-large):**
- High-volume production
- 150k-400k messages/sec
- 250-500 topics
- Multi-region replication

**X-Large (kafka-xlarge):**
- Enterprise/extreme scale
- > 400k messages/sec
- > 500 topics
- Global distribution

## Permissions
Roles: admin, developer, viewer

## Workflow Steps
### Workflow: get_flavors_for_service
Get available flavors for service
- Step 1: authenticate (auth.validate_token)
- Step 2: get_service_flavors (flavor.get_for_service) (depends on: service_name)

## Usage Notes
- Flavors are service-specific and optimized for workload type
- Kafka flavors include broker capacity specifications
- Recommended flavor based on typical production workloads
- Pricing varies by location and commitment level
- Some flavors may not be available in all locations
- IOPS and network specs are guaranteed minimums

## Common Use Cases
1. **Size Kafka cluster**: "What flavor should I use for Kafka?"
2. **Compare options**: "Show me Kafka flavor options"
3. **Cost estimation**: "How much does a Kafka medium flavor cost?"
4. **Capacity planning**: "Can this flavor handle 100k messages/sec?"
5. **Location availability**: "Is kafka-large available in Mumbai?"
6. **Upgrade planning**: "What's the next size up from kafka-medium?"

## Data Processing Examples

### Find Recommended Kafka Flavor
```python
flavors = response['data']['flavors']
recommended = [f for f in flavors if f.get('recommended')]

if recommended:
    flavor = recommended[0]
    print(f"Recommended: {flavor['name']}")
    print(f"  vCPUs: {flavor['vcpus']}")
    print(f"  Memory: {flavor['memoryGB']} GB")
    print(f"  Max throughput: {flavor['kafkaSpecs']['maxThroughputMBps']} MB/s")
    print(f"  Monthly cost: ${flavor['pricing']['monthly']}")
```

### Calculate Cost for 3-Broker Kafka Cluster
```python
flavor_name = "kafka-medium"
brokers = 3

flavor = next(f for f in response['data']['flavors'] if f['id'] == flavor_name)
monthly_cost = flavor['pricing']['monthly'] * brokers

print(f"3-broker {flavor['name']} cluster:")
print(f"  Total vCPUs: {flavor['vcpus'] * brokers}")
print(f"  Total Memory: {flavor['memoryGB'] * brokers} GB")
print(f"  Total Storage: {flavor['storageGB'] * brokers} GB")
print(f"  Monthly cost: ${monthly_cost}")
```

### Find Flavor by Throughput Requirements
```python
required_mbps = 200  # Required throughput

suitable_flavors = [
    f for f in response['data']['flavors']
    if f.get('kafkaSpecs', {}).get('maxThroughputMBps', 0) >= required_mbps
]

# Sort by price
suitable_flavors.sort(key=lambda x: x['pricing']['monthly'])

if suitable_flavors:
    cheapest = suitable_flavors[0]
    print(f"Most cost-effective: {cheapest['name']}")
    print(f"  Throughput: {cheapest['kafkaSpecs']['maxThroughputMBps']} MB/s")
    print(f"  Cost: ${cheapest['pricing']['monthly']}/month")
```

### Compare Flavors by Capability
```python
import pandas as pd

df = pd.DataFrame([
    {
        'Flavor': f['name'],
        'vCPUs': f['vcpus'],
        'RAM (GB)': f['memoryGB'],
        'Storage (GB)': f['storageGB'],
        'Max Throughput (MB/s)': f['kafkaSpecs']['maxThroughputMBps'],
        'Max Messages/sec': f['kafkaSpecs']['maxMessagesPerSec'],
        'Monthly $': f['pricing']['monthly']
    }
    for f in response['data']['flavors']
])

print(df.to_string(index=False))
```

## Kafka Cluster Sizing

### General Guidelines

**Broker Count:**
- Minimum: 3 brokers (for replication)
- Recommended: 3-5 brokers for production
- Large scale: 7+ brokers

**Flavor Selection:**
- Start with medium for most production workloads
- Scale up if CPU/memory/disk becomes bottleneck
- Scale out (add brokers) for throughput needs

**Replication:**
- Replication factor: 3 (recommended)
- Min in-sync replicas: 2 (recommended)
- Unclean leader election: disabled (recommended)

### Capacity Calculation Examples

**Example 1: E-commerce Events**
- 100k messages/sec peak
- Average message size: 1 KB
- Throughput: 100 MB/s
- Recommended: kafka-medium (3 brokers)

**Example 2: Financial Transactions**
- 500k messages/sec peak
- Average message size: 0.5 KB
- Throughput: 250 MB/s
- Recommended: kafka-large (3 brokers)

**Example 3: IoT Telemetry**
- 2M messages/sec peak
- Average message size: 0.2 KB
- Throughput: 400 MB/s
- Recommended: kafka-large (5 brokers) or kafka-xlarge (3 brokers)

## Related Operations
- `flavor.list_all` - List all available flavors
- `cluster.create` - Create cluster with specified flavor
- `cluster.resize` - Change cluster flavor
- `pricing.estimate` - Get detailed cost estimate

## Error Handling
- **400 Bad Request:** Invalid service name
- **401 Unauthorized:** Invalid or expired authentication token
- **403 Forbidden:** User does not have permission to view flavors
- **404 Not Found:** Service name not found
- **500 Internal Server Error:** Server-side error retrieving flavors

## Response Codes
- `0` - Success
- Non-zero values indicate errors (check `message` field for details)

## Supported Services

### Messaging Services
- `kafka` - Apache Kafka (with Kafka-optimized specs)
- `rabbitmq` - RabbitMQ message broker
- `redis` - Redis (pub/sub mode)

### Database Services
- `postgres` - PostgreSQL
- `mysql` - MySQL
- `mongodb` - MongoDB
- `cassandra` - Apache Cassandra

### Other Services
- `elasticsearch` - Elasticsearch
- `spark` - Apache Spark
- `flink` - Apache Flink

## Performance Notes
- Response time typically < 300ms
- Flavor availability updated every 15 minutes
- Pricing may vary by region and commitment
- Kafka specs based on standard 3-broker configuration

## Metadata
- **Generated:** 2025-02-13T12:00:00Z
- **Source:** Dynamic API Spec Generator
- **API Version:** v1
- **Base Path:** /paasservice/paas
- **Kafka Support:** Comprehensive Kafka broker sizing and capacity planning