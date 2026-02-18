# API Specification: volume - get_usage

**Resource:** volume
**Operation:** get_usage
**Aliases:** volume usage, storage usage, disk usage, volume stats, storage stats, volume capacity

## Endpoint
- **Method:** GET
- **URL:** {BASE_URL_PAAS_SERVICE}/paas/getVolumeUsage/{volume_id}
- **Auth:** Bearer token (from Keycloak)
- **Description:** Get detailed usage statistics for a persistent volume, including space utilization, I/O metrics, and attachment information. Particularly useful for monitoring Kafka broker storage

## Required Parameters
- `volume_id` - Unique identifier for the volume (path parameter)

## Optional Parameters
- `include_history` - Include historical usage data (true/false)
- `time_range` - Time range for metrics (e.g., "1h", "24h", "7d")
- `detailed` - Include detailed I/O statistics (true/false)

## Response Mapping
- `status`: status
- `message`: message
- `volume_id`: data.volumeId
- `volume_name`: data.volumeName
- `size_gb`: data.sizeGB
- `used_gb`: data.usedGB
- `available_gb`: data.availableGB
- `usage_percentage`: data.usagePercentage
- `attached_to`: data.attachedTo
- `attachment_type`: data.attachmentType
- `storage_class`: data.storageClass
- `iops`: data.iops
- `throughput_mbps`: data.throughputMBps
- `mount_point`: data.mountPoint
- `filesystem_type`: data.filesystemType

## Response Example
```json
{
  "status": "success",
  "data": {
    "volumeId": "vol-kafka-001",
    "volumeName": "kafka-data-vol-01",
    "sizeGB": 500,
    "usedGB": 320,
    "availableGB": 180,
    "usagePercentage": 64.0,
    "attachedTo": {
      "resourceId": "kafka-cluster-001",
      "resourceName": "prod-kafka-01",
      "resourceType": "kafka_cluster",
      "brokerId": "broker-1"
    },
    "attachmentType": "persistent",
    "status": "attached",
    "storageClass": "ssd-premium",
    "iops": 3000,
    "throughputMBps": 250,
    "mountPoint": "/var/lib/kafka/data",
    "filesystemType": "ext4",
    "created": "2024-01-15T10:30:00Z",
    "lastModified": "2025-02-13T10:00:00Z",
    "tags": ["kafka", "production", "storage", "ssd"],
    "location": "mumbai-bkc",
    "usageMetrics": {
      "inodes": {
        "total": 65536000,
        "used": 2500000,
        "available": 63036000,
        "usagePercentage": 3.8
      },
      "io": {
        "readOps": 150000,
        "writeOps": 500000,
        "readMBps": 45.2,
        "writeMBps": 123.8,
        "avgLatencyMs": 2.5
      },
      "kafka": {
        "logSegments": 450,
        "activeSegments": 50,
        "oldestSegment": "2024-11-15T00:00:00Z",
        "retentionPolicy": "7d",
        "compressionRatio": 2.3
      }
    },
    "alerts": [
      {
        "level": "warning",
        "message": "Volume usage above 60% threshold",
        "timestamp": "2025-02-13T09:30:00Z"
      }
    ],
    "history": {
      "dataPoints": [
        {
          "timestamp": "2025-02-13T08:00:00Z",
          "usedGB": 315,
          "usagePercentage": 63.0
        },
        {
          "timestamp": "2025-02-13T10:00:00Z",
          "usedGB": 320,
          "usagePercentage": 64.0
        }
      ],
      "trend": "increasing",
      "growthRateGBPerDay": 5.2
    }
  },
  "message": "Volume usage retrieved successfully",
  "responseCode": 0
}
```

## Response Fields Details

### Volume Fields
- **volumeId** - Unique volume identifier
- **volumeName** - Human-readable volume name
- **sizeGB** - Total volume size in gigabytes
- **usedGB** - Used space in gigabytes
- **availableGB** - Available space in gigabytes
- **usagePercentage** - Percentage of space used (0-100)
- **attachedTo** - Resource this volume is attached to
- **attachmentType** - Type of attachment (persistent, temporary)
- **status** - Current volume status
- **storageClass** - Storage class/tier (ssd, hdd, premium, standard)
- **iops** - Input/output operations per second
- **throughputMBps** - Throughput in megabytes per second
- **mountPoint** - File system mount point
- **filesystemType** - File system type (ext4, xfs, etc.)

### Volume Status Values
- `attached` - Volume is attached and mounted
- `detached` - Volume is not attached to any resource
- `creating` - Volume is being provisioned
- `deleting` - Volume is being deleted
- `error` - Volume has encountered an error
- `resizing` - Volume size is being changed

### Storage Classes
- `ssd-premium` - High-performance SSD storage
- `ssd-standard` - Standard SSD storage
- `hdd-standard` - Standard HDD storage
- `hdd-archive` - Low-cost archive storage

## Kafka-Specific Usage

### Kafka Broker Storage
When attached to Kafka brokers, volumes include additional metrics:

```json
{
  "kafka": {
    "logSegments": 450,
    "activeSegments": 50,
    "oldestSegment": "2024-11-15T00:00:00Z",
    "retentionPolicy": "7d",
    "compressionRatio": 2.3,
    "topicsCount": 50,
    "partitionsCount": 150
  }
}
```

### Kafka Log Segments
- **logSegments** - Total number of log segments
- **activeSegments** - Currently active (being written to) segments
- **oldestSegment** - Timestamp of oldest retained segment
- **retentionPolicy** - Configured retention period
- **compressionRatio** - Compression effectiveness (higher is better)

### Kafka Storage Monitoring
Monitor these for Kafka broker health:
- **Usage < 80%** - Optimal, plenty of headroom
- **Usage 80-90%** - Warning, plan for expansion
- **Usage > 90%** - Critical, immediate action needed
- **IOPS** - Should match broker write load
- **Write throughput** - Should handle peak message rates

## Permissions
Roles: admin, developer, viewer

## Workflow Steps
### Workflow: get_volume_usage
Get volume usage statistics
- Step 1: authenticate (auth.validate_token)
- Step 2: get_volume_usage (volume.get_usage) (depends on: volume_id)

## Usage Notes
- Usage statistics updated every 5 minutes
- Historical data retained for 30 days
- IOPS and throughput are averages over last 5 minutes
- Kafka-specific metrics only present for Kafka broker volumes
- Alerts are generated based on configurable thresholds
- Growth trend calculated from last 7 days

## Common Use Cases
1. **Check volume capacity**: "Show me volume usage"
2. **Kafka broker monitoring**: "How much storage is Kafka using?"
3. **Capacity planning**: "When will this volume be full?"
4. **Performance analysis**: "Is the volume IOPS sufficient?"
5. **Alert investigation**: "Why is volume usage high?"
6. **Storage optimization**: "Which Kafka topics use most space?"

## Data Processing Examples

### Check if Volume Needs Expansion
```python
usage = response['data']['usagePercentage']
growth_rate = response['data']['history']['growthRateGBPerDay']
available = response['data']['availableGB']

days_until_full = available / growth_rate if growth_rate > 0 else float('inf')

if usage > 80:
    print(f"WARNING: Volume at {usage}% capacity")
    print(f"Estimated days until full: {days_until_full:.1f}")
elif days_until_full < 30:
    print(f"INFO: Volume will be full in ~{days_until_full:.0f} days")
```

### Calculate Kafka Segment Cleanup
```python
kafka_metrics = response['data']['usageMetrics']['kafka']
segments = kafka_metrics['logSegments']
active = kafka_metrics['activeSegments']
inactive = segments - active

potential_savings_gb = (
    response['data']['usedGB'] * 
    (inactive / segments) * 
    0.7  # Assuming 70% of inactive can be deleted
)

print(f"Inactive segments: {inactive}")
print(f"Potential space savings: {potential_savings_gb:.1f} GB")
```

### Monitor I/O Performance
```python
io_metrics = response['data']['usageMetrics']['io']
read_mbps = io_metrics['readMBps']
write_mbps = io_metrics['writeMBps']
latency = io_metrics['avgLatencyMs']

# Kafka typically write-heavy
if write_mbps > read_mbps * 3:
    print("Kafka write-heavy pattern detected (normal)")

if latency > 10:
    print(f"WARNING: High I/O latency: {latency}ms")
```

### Calculate Kafka Compression Effectiveness
```python
kafka_metrics = response['data']['usageMetrics'].get('kafka', {})
compression_ratio = kafka_metrics.get('compressionRatio', 1.0)
used_gb = response['data']['usedGB']

uncompressed_size = used_gb * compression_ratio
saved_gb = uncompressed_size - used_gb

print(f"Compression ratio: {compression_ratio:.2f}x")
print(f"Space saved: {saved_gb:.1f} GB ({saved_gb/uncompressed_size*100:.1f}%)")
```

## Kafka Storage Best Practices

### Retention Policy
- Set appropriate retention based on business needs
- Monitor old segment accumulation
- Implement topic-level retention policies
- Regular compaction for key-based topics

### Capacity Planning
- Kafka volumes should have 20% free space minimum
- Plan for 2-3x peak message rates
- Consider retention period in sizing
- Account for replication factor

### Performance Tuning
- Use SSD storage for Kafka brokers
- Ensure IOPS match write throughput needs
- Monitor segment write patterns
- Optimize log segment size

### Monitoring Alerts
- Alert at 70% usage (plan expansion)
- Alert at 85% usage (urgent action)
- Alert on IOPS saturation
- Alert on write latency spikes

## Related Operations
- `volume.list` - List all volumes
- `volume.create` - Create new volume
- `volume.resize` - Resize existing volume
- `volume.attach` - Attach volume to resource
- `volume.detach` - Detach volume from resource
- `kafka.get_broker_storage` - Get Kafka broker storage details

## Error Handling
- **401 Unauthorized:** Invalid or expired authentication token
- **403 Forbidden:** User does not have permission to view volume
- **404 Not Found:** Volume with specified ID does not exist
- **500 Internal Server Error:** Server-side error retrieving volume data

## Response Codes
- `0` - Success
- Non-zero values indicate errors (check `message` field for details)

## Alert Thresholds

### Default Thresholds
- **Warning:** 70% usage
- **Critical:** 85% usage
- **Emergency:** 95% usage

### Kafka-Specific Thresholds
- **Segment accumulation:** > 1000 inactive segments
- **Write latency:** > 10ms average
- **IOPS saturation:** > 90% of provisioned IOPS

## Performance Notes
- Response time typically < 200ms
- Usage metrics cached for 5 minutes
- Historical data queries may take longer
- I/O metrics updated in real-time

## Metadata
- **Generated:** 2025-02-13T12:00:00Z
- **Source:** Dynamic API Spec Generator
- **API Version:** v1
- **Base Path:** /paasservice/paas
- **Kafka Support:** Kafka broker storage monitoring and analytics