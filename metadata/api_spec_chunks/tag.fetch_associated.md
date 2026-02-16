# API Specification: tag - fetch_associated

**Resource:** tag
**Operation:** fetch_associated
**Aliases:** associated tags, fetch tags, get associated tags, tags by name, find tags

## Endpoint
- **Method:** GET
- **URL:** {BASE_URL_PAAS_SERVICE}/paas/fetchAllAssociatedTags/{tags}
- **Auth:** Bearer token (from Keycloak)
- **Description:** Fetch all resources associated with specific tags, useful for discovering Kafka clusters, topics, and other resources tagged with particular labels

## Required Parameters
- `tags` - Comma-separated list of tag names or single tag name (path parameter)

## Optional Parameters
- `resource_type` - Filter by resource type (e.g., "cluster", "kafka", "volume")
- `include_metadata` - Include full resource metadata (true/false)
- `active_only` - Return only active resources (true/false, default: true)

## Response Mapping
- `status`: status
- `message`: message
- `tags_queried`: data.tagsQueried
- `resources`: data.resources
- `resource_ids`: data.resources[*].id
- `resource_names`: data.resources[*].name
- `resource_types`: data.resources[*].type
- `kafka_resources`: data.resources[*][?type=='kafka']
- `clusters`: data.resources[*][?type=='cluster']
- `total_count`: data.totalCount
- `tag_matches`: data.resources[*].matchedTags

## Response Example
```json
{
  "status": "success",
  "data": {
    "tagsQueried": ["kafka", "production", "customer-analytics"],
    "totalCount": 8,
    "resources": [
      {
        "id": "kafka-cluster-001",
        "name": "prod-kafka-01",
        "type": "kafka",
        "resourceType": "kafka_cluster",
        "status": "running",
        "matchedTags": ["kafka", "production"],
        "allTags": ["kafka", "production", "messaging", "critical"],
        "metadata": {
          "kafka_version": "3.4.0",
          "brokers": [
            {
              "id": "broker-1",
              "host": "kafka-broker-1.prod.local",
              "port": 9092,
              "status": "healthy"
            },
            {
              "id": "broker-2",
              "host": "kafka-broker-2.prod.local",
              "port": 9092,
              "status": "healthy"
            },
            {
              "id": "broker-3",
              "host": "kafka-broker-3.prod.local",
              "port": 9092,
              "status": "healthy"
            }
          ],
          "topics_count": 50,
          "consumer_groups": 12,
          "total_partitions": 150,
          "replication_factor": 3,
          "min_insync_replicas": 2
        },
        "location": "mumbai-bkc",
        "createdAt": "2024-01-15T10:30:00Z"
      },
      {
        "id": "kafka-topic-001",
        "name": "customer-events",
        "type": "kafka",
        "resourceType": "kafka_topic",
        "status": "active",
        "matchedTags": ["customer-analytics", "production"],
        "allTags": ["customer-analytics", "production", "high-throughput"],
        "metadata": {
          "cluster_id": "kafka-cluster-001",
          "partitions": 12,
          "replication_factor": 3,
          "retention_ms": 604800000,
          "retention_hours": 168,
          "compression_type": "snappy",
          "messages_per_sec": 5000,
          "bytes_per_sec": 5242880,
          "consumer_groups": ["analytics-cg", "reporting-cg"]
        },
        "createdAt": "2024-01-20T14:45:00Z"
      },
      {
        "id": "k8s-cluster-1267",
        "name": "test",
        "type": "cluster",
        "resourceType": "kubernetes_cluster",
        "status": "running",
        "matchedTags": ["production"],
        "allTags": ["production", "kubernetes", "bengaluru"],
        "metadata": {
          "nodescount": "3",
          "location": "EP_V2_BL",
          "displayNameEndpoint": "Bengaluru"
        },
        "createdAt": "2024-02-01T09:15:00Z"
      },
      {
        "id": "volume-storage-001",
        "name": "kafka-data-vol-01",
        "type": "volume",
        "resourceType": "persistent_volume",
        "status": "attached",
        "matchedTags": ["kafka", "production"],
        "allTags": ["kafka", "production", "storage", "ssd"],
        "metadata": {
          "size_gb": 500,
          "used_gb": 320,
          "usage_percentage": 64,
          "attached_to": "kafka-cluster-001",
          "storage_class": "ssd",
          "iops": 3000
        },
        "createdAt": "2024-01-15T10:35:00Z"
      }
    ]
  },
  "message": "Resources fetched successfully",
  "responseCode": 0
}
```

## Response Fields Details

### Resource Fields
- **id** - Unique resource identifier
- **name** - Human-readable resource name
- **type** - Primary resource type (kafka, cluster, volume, service)
- **resourceType** - Detailed resource type classification
- **status** - Current resource status
- **matchedTags** - Tags that matched the query
- **allTags** - All tags associated with the resource
- **metadata** - Resource-specific metadata (varies by type)
- **location** - Physical or logical location
- **createdAt** - ISO 8601 timestamp

### Resource Types
- `kafka` - Kafka-related resources (clusters, topics, consumer groups)
- `cluster` - Kubernetes or compute clusters
- `volume` - Storage volumes
- `service` - Microservices and applications
- `network` - Network resources (load balancers, subnets)
- `database` - Database instances

### Kafka Resource Types
- `kafka_cluster` - Kafka cluster/broker setup
- `kafka_topic` - Kafka topics
- `kafka_consumer_group` - Consumer groups
- `kafka_connector` - Kafka Connect connectors
- `kafka_schema` - Schema registry entries

## Kafka-Specific Metadata

### Kafka Cluster Metadata
```json
{
  "kafka_version": "3.4.0",
  "brokers": [...],
  "topics_count": 50,
  "consumer_groups": 12,
  "total_partitions": 150,
  "replication_factor": 3,
  "min_insync_replicas": 2,
  "zookeeper_nodes": 3
}
```

### Kafka Topic Metadata
```json
{
  "cluster_id": "kafka-cluster-001",
  "partitions": 12,
  "replication_factor": 3,
  "retention_ms": 604800000,
  "compression_type": "snappy",
  "messages_per_sec": 5000,
  "bytes_per_sec": 5242880,
  "consumer_groups": ["analytics-cg"]
}
```

### Kafka Consumer Group Metadata
```json
{
  "cluster_id": "kafka-cluster-001",
  "topics": ["customer-events", "order-events"],
  "members": 5,
  "state": "Stable",
  "coordinator": "broker-2",
  "lag": 1500
}
```

## Permissions
Roles: admin, developer, viewer

## Workflow Steps
### Workflow: fetch_associated_tags
Fetch resources by tags
- Step 1: authenticate (auth.validate_token)
- Step 2: parse_tags (tag.parse) (depends on: tags)
- Step 3: fetch_resources (tag.fetch_associated) (depends on: tags)

## Usage Notes
- Multiple tags can be queried: `/fetchAllAssociatedTags/kafka,production`
- Resources must match at least one tag (OR logic by default)
- Use commas to separate multiple tags (no spaces)
- Tag matching is case-insensitive
- Returns resources across all resource types unless filtered
- Kafka resources include comprehensive metadata for monitoring

## Common Use Cases
1. **Find Kafka clusters**: "Show me all Kafka resources"
2. **Production resources**: "List all production tagged resources"
3. **Kafka topics**: "Find customer-analytics Kafka topics"
4. **Multi-tag search**: "Resources tagged kafka AND production"
5. **Storage discovery**: "Find volumes attached to Kafka"
6. **Service mapping**: "Map all tagged microservices"

## Data Processing Examples

### Extract Kafka Clusters
```python
kafka_clusters = [
    r for r in response['data']['resources']
    if r['resourceType'] == 'kafka_cluster'
]

for cluster in kafka_clusters:
    print(f"Cluster: {cluster['name']}")
    print(f"  Brokers: {len(cluster['metadata']['brokers'])}")
    print(f"  Topics: {cluster['metadata']['topics_count']}")
    print(f"  Status: {cluster['status']}")
```

### Calculate Kafka Topic Throughput
```python
kafka_topics = [
    r for r in response['data']['resources']
    if r['resourceType'] == 'kafka_topic'
]

total_msgs_per_sec = sum(
    t['metadata'].get('messages_per_sec', 0)
    for t in kafka_topics
)

total_mb_per_sec = sum(
    t['metadata'].get('bytes_per_sec', 0) / 1024 / 1024
    for t in kafka_topics
)
```

### Group by Resource Type
```python
from collections import defaultdict

by_type = defaultdict(list)
for resource in response['data']['resources']:
    by_type[resource['resourceType']].append(resource)

print(f"Kafka Clusters: {len(by_type['kafka_cluster'])}")
print(f"Kafka Topics: {len(by_type['kafka_topic'])}")
print(f"K8s Clusters: {len(by_type['kubernetes_cluster'])}")
```

### Check Kafka Broker Health
```python
for resource in response['data']['resources']:
    if resource['resourceType'] == 'kafka_cluster':
        healthy = sum(
            1 for b in resource['metadata']['brokers']
            if b['status'] == 'healthy'
        )
        total = len(resource['metadata']['brokers'])
        health_pct = (healthy / total) * 100
        print(f"{resource['name']}: {healthy}/{total} ({health_pct}%) healthy")
```

## Kafka Overview Integration

This endpoint is essential for Kafka infrastructure management:

### Kafka Cluster Discovery
Query: `/fetchAllAssociatedTags/kafka`
Returns: All Kafka clusters, topics, consumer groups

### Kafka Topic Management
Query: `/fetchAllAssociatedTags/kafka-topic,customer-analytics`
Returns: Specific topics with throughput and retention data

### Kafka Monitoring
- Broker health status
- Topic partition distribution
- Consumer group lag
- Message throughput rates
- Storage utilization

### Kafka Resource Relationships
- Clusters → Topics → Consumer Groups
- Volumes attached to brokers
- Network configuration
- Service dependencies

## Related Operations
- `tag.list` - List all tags for engagement
- `kafka.list_clusters` - List Kafka clusters directly
- `kafka.list_topics` - List Kafka topics
- `resource.get_by_tag` - Get specific resource by tag

## Error Handling
- **400 Bad Request:** Invalid tag format
- **401 Unauthorized:** Invalid or expired authentication token
- **403 Forbidden:** User does not have permission to view resources
- **404 Not Found:** No resources found with specified tags
- **422 Unprocessable Entity:** Invalid tags parameter
- **500 Internal Server Error:** Server-side error fetching resources

## Response Codes
- `0` - Success
- Non-zero values indicate errors (check `message` field for details)

## URL Format Examples

### Single Tag
```
GET /paasservice/paas/fetchAllAssociatedTags/kafka
```

### Multiple Tags
```
GET /paasservice/paas/fetchAllAssociatedTags/kafka,production,mumbai
```

### With Query Parameters
```
GET /paasservice/paas/fetchAllAssociatedTags/kafka?resource_type=kafka_cluster&active_only=true
```

## Performance Notes
- Response time depends on number of tagged resources
- Typically < 1 second for < 100 resources
- Results are cached for 2 minutes
- Large result sets may be paginated
- Metadata inclusion increases response size

## Best Practices

### Querying Multiple Tags
- Start with most specific tags
- Combine related tags: `kafka,production,mumbai`
- Avoid very common tags alone: use `production` + more specific tags

### Kafka Resource Discovery
- Tag all Kafka resources consistently
- Include environment tags: `kafka,prod` or `kafka,staging`
- Tag by data classification: `kafka,customer-data,pii`
- Include location tags: `kafka,mumbai-bkc`

### Performance Optimization
- Request only needed metadata
- Use resource_type filters when possible
- Cache results when querying frequently
- Implement pagination for large result sets

## Metadata
- **Generated:** 2025-02-13T12:00:00Z
- **Source:** Dynamic API Spec Generator
- **API Version:** v1
- **Base Path:** /paasservice/paas
- **Kafka Support:** Full Kafka cluster, topic, and consumer group discovery