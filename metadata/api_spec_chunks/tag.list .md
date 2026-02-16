# API Specification: tag - list

**Resource:** tag
**Operation:** list
**Aliases:** tags, list tags, show tags, all tags, tag list, get tags, engagement tags

## Endpoint
- **Method:** GET
- **URL:** https://ipcloud.tatacommunications.com/portalservice/tag/list?engagementId={engagement_id}
- **Auth:** Bearer token (from Keycloak)
- **Description:** List all tags associated with a specific engagement, including resource tags, labels, and metadata tags for organization and filtering

## Required Parameters
- `engagement_id` - Unique identifier for the engagement (query parameter)

## Optional Parameters
- `type` - Filter by tag type (e.g., "resource", "label", "kafka", "cluster")
- `category` - Filter by category (e.g., "environment", "project", "service")
- `active` - Filter by active status (true/false)

## Response Mapping
- `status`: status
- `message`: message
- `tags`: data.tags
- `tag_ids`: data.tags[*].id
- `tag_names`: data.tags[*].name
- `tag_keys`: data.tags[*].key
- `tag_values`: data.tags[*].value
- `tag_types`: data.tags[*].type
- `tag_categories`: data.tags[*].category
- `engagement_id`: data.engagementId
- `total_count`: data.totalCount

## Response Example
```json
{
  "status": "success",
  "data": {
    "engagementId": "eng-12345",
    "totalCount": 15,
    "tags": [
      {
        "id": "tag-001",
        "name": "production",
        "key": "environment",
        "value": "prod",
        "type": "label",
        "category": "environment",
        "createdAt": "2024-01-15T10:30:00Z",
        "createdBy": "admin@example.com",
        "active": true
      },
      {
        "id": "tag-002",
        "name": "kafka-cluster",
        "key": "service",
        "value": "kafka",
        "type": "resource",
        "category": "messaging",
        "metadata": {
          "kafka_version": "3.4.0",
          "brokers": 3,
          "topics": 50
        },
        "createdAt": "2024-01-20T14:45:00Z",
        "active": true
      },
      {
        "id": "tag-003",
        "name": "customer-data",
        "key": "project",
        "value": "customer-analytics",
        "type": "label",
        "category": "project",
        "createdAt": "2024-02-01T09:15:00Z",
        "active": true
      }
    ]
  },
  "message": "Tags retrieved successfully",
  "responseCode": 0
}
```

## Response Fields Details

### Tag Fields
- **id** - Unique tag identifier
- **name** - Human-readable tag name
- **key** - Tag key for key-value pairing
- **value** - Tag value
- **type** - Tag type (resource, label, metadata, kafka, cluster)
- **category** - Organizational category
- **createdAt** - ISO 8601 timestamp when tag was created
- **createdBy** - User who created the tag
- **active** - Whether tag is currently active
- **metadata** - Optional additional metadata (object)

### Tag Types
- `resource` - Tags attached to specific resources (clusters, volumes, etc.)
- `label` - General organizational labels
- `metadata` - System metadata tags
- `kafka` - Kafka-specific tags for messaging resources
- `cluster` - Cluster-related tags
- `service` - Service-level tags

### Tag Categories
- `environment` - Environment tags (prod, staging, dev)
- `project` - Project-based organization
- `service` - Service type tags (kafka, redis, postgres)
- `messaging` - Messaging service tags
- `cost-center` - Cost allocation tags
- `compliance` - Compliance and regulatory tags

## Kafka-Specific Tags

When working with Kafka resources, tags may include:

### Kafka Metadata
```json
{
  "type": "kafka",
  "category": "messaging",
  "metadata": {
    "kafka_version": "3.4.0",
    "brokers": 3,
    "topics": 50,
    "partitions": 150,
    "replication_factor": 3,
    "cluster_type": "managed"
  }
}
```

### Kafka Tag Keys
- `kafka.cluster` - Kafka cluster identifier
- `kafka.topic` - Kafka topic name
- `kafka.consumer-group` - Consumer group name
- `kafka.environment` - Kafka environment type
- `kafka.retention` - Retention policy

## Permissions
Roles: admin, developer, viewer

## Workflow Steps
### Workflow: list_tags
List tags for engagement
- Step 1: authenticate (auth.validate_token)
- Step 2: get_engagement (engagement.get) (depends on: engagement_id)
- Step 3: list_tags (tag.list) (depends on: engagement_id)

## Usage Notes
- Tags are used for resource organization, cost allocation, and filtering
- Tags can be attached to clusters, volumes, services, and other resources
- Kafka-related tags help identify and manage messaging infrastructure
- Tags support hierarchical organization through categories
- Multiple tags can have the same key with different values
- Tags are case-sensitive

## Common Use Cases
1. **List all tags**: "Show me all tags for engagement"
2. **Find Kafka resources**: "List Kafka-related tags"
3. **Filter by environment**: "Show production tags"
4. **Cost tracking**: "List tags for cost-center finance"
5. **Service discovery**: "Find all tagged services"
6. **Compliance audit**: "Show compliance tags"

## Data Processing Examples

### Filter Kafka Tags
```python
kafka_tags = [t for t in response['data']['tags'] 
              if t.get('type') == 'kafka' or 
                 t.get('category') == 'messaging']
```

### Group by Category
```python
from collections import defaultdict
by_category = defaultdict(list)
for tag in response['data']['tags']:
    by_category[tag['category']].append(tag)
```

### Find Active Tags
```python
active_tags = [t for t in response['data']['tags'] 
               if t.get('active', True)]
```

### Extract Kafka Clusters
```python
kafka_clusters = []
for tag in response['data']['tags']:
    if tag.get('metadata', {}).get('kafka_version'):
        kafka_clusters.append({
            'name': tag['name'],
            'version': tag['metadata']['kafka_version'],
            'brokers': tag['metadata'].get('brokers', 0)
        })
```

## Kafka Integration

### Tagging Kafka Resources

Tags help manage Kafka infrastructure:

**Kafka Cluster Tags:**
- `kafka.cluster=prod-cluster-01`
- `kafka.version=3.4.0`
- `kafka.brokers=3`

**Kafka Topic Tags:**
- `kafka.topic=customer-events`
- `kafka.retention=7d`
- `kafka.partitions=12`

**Consumer Group Tags:**
- `kafka.consumer-group=analytics-consumers`
- `kafka.lag-threshold=1000`

### Kafka Tag Queries

- "Show Kafka cluster tags" → Filter type='kafka' and metadata contains kafka_version
- "List topics by retention" → Filter kafka.retention key
- "Find consumer groups" → Filter kafka.consumer-group key

## Related Operations
- `tag.create` - Create new tag
- `tag.update` - Update existing tag
- `tag.delete` - Delete tag
- `tag.attach` - Attach tag to resource
- `tag.detach` - Detach tag from resource
- `resource.list_by_tags` - List resources filtered by tags

## Error Handling
- **401 Unauthorized:** Invalid or expired authentication token
- **403 Forbidden:** User does not have permission to view engagement tags
- **404 Not Found:** Engagement not found
- **422 Unprocessable Entity:** Invalid engagement_id format
- **500 Internal Server Error:** Server-side error retrieving tags

## Response Codes
- `0` - Success
- Non-zero values indicate errors (check `message` field for details)

## Query String Format

The engagement ID is passed as a query parameter:
```
GET /portalservice/tag/list?engagementId=eng-12345
```

Additional filters can be added:
```
GET /portalservice/tag/list?engagementId=eng-12345&type=kafka&active=true
```

## Performance Notes
- Response time typically < 500ms for most engagements
- Tags are cached for 5 minutes by default
- Large tag sets (1000+) may take longer
- Consider pagination for very large tag collections

## Best Practices

### Tag Naming
- Use lowercase with hyphens: `kafka-prod-cluster`
- Be descriptive: `customer-analytics-kafka` not `ca-k`
- Include version info: `kafka-3.4.0`

### Tag Organization
- Use categories consistently
- Document tag schemas
- Regular cleanup of inactive tags
- Use metadata for structured data

### Kafka Tags
- Tag all Kafka clusters with version
- Include broker count in metadata
- Tag topics by data classification
- Track consumer group ownership

## Metadata
- **Generated:** 2025-02-13T12:00:00Z
- **Source:** Dynamic API Spec Generator
- **API Version:** v1
- **Base Path:** /portalservice
- **Supports:** Kafka resource tagging and discovery