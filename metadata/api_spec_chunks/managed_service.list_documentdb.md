# API Specification: managed_service - list_documentdb

**Resource:** managed_service
**Operation:** list_documentdb
**Aliases:** documentdb, mongodb, list mongodb, mongo services, document database, IKS documentdb

## Endpoint
- **Method:** GET
- **URL:** {BASE_URL_PAAS_SERVICE}/api/v1/paas/listManagedServices/IKSDocumentDB
- **Auth:** Bearer token (from Keycloak)
- **Description:** List all IKS-managed DocumentDB (MongoDB) instances with their configuration, replica sets, sharding details, and performance metrics

## Required Parameters
None (service type is in URL: `IKSDocumentDB`)

## Optional Parameters
- `status` - Filter by status (running, stopped, creating)
- `region` - Filter by region/zone
- `version` - Filter by MongoDB version
- `include_metrics` - Include performance metrics (true/false)

## Response Mapping
- `status`: status
- `message`: message
- `service_type`: data.serviceType
- `total_count`: data.totalCount
- `services`: data.services
- `service_ids`: data.services[*].id
- `service_names`: data.services[*].name
- `mongodb_versions`: data.services[*].version
- `replica_sets`: data.services[*].configuration.replicaSet

## Response Example
```json
{
  "status": "success",
  "data": {
    "serviceType": "IKSDocumentDB",
    "serviceDescription": "Integrated Kubernetes Service - DocumentDB (MongoDB)",
    "totalCount": 8,
    "services": [
      {
        "id": "iks-docdb-prod-001",
        "name": "production-mongodb",
        "displayName": "Production MongoDB Cluster",
        "type": "IKSDocumentDB",
        "status": "running",
        "health": "healthy",
        "version": "6.0.8",
        "mongodbVersion": "MongoDB 6.0.8",
        "createdAt": "2024-01-20T10:30:00Z",
        "region": "mumbai-bkc",
        "zone": "EP_V2_MUM_BKC",
        "cluster": {
          "id": "cluster-1267",
          "name": "prod-k8s-01",
          "nodesCount": 5
        },
        "connectionDetails": {
          "connectionString": "mongodb://mongodb-prod-001-0.mongodb-prod-001.default.svc.cluster.local:27017,mongodb-prod-001-1.mongodb-prod-001.default.svc.cluster.local:27017,mongodb-prod-001-2.mongodb-prod-001.default.svc.cluster.local:27017/admin?replicaSet=rs0",
          "hosts": [
            "mongodb-prod-001-0.mongodb-prod-001.default.svc.cluster.local:27017",
            "mongodb-prod-001-1.mongodb-prod-001.default.svc.cluster.local:27017",
            "mongodb-prod-001-2.mongodb-prod-001.default.svc.cluster.local:27017"
          ],
          "port": 27017,
          "authDatabase": "admin",
          "ssl": true,
          "replicaSet": "rs0"
        },
        "resources": {
          "vcpus": 12,
          "memoryGB": 48,
          "storageGB": 1000,
          "storageType": "ssd-premium",
          "iops": 9000
        },
        "configuration": {
          "replicaSet": {
            "name": "rs0",
            "members": 3,
            "priority": {
              "member-0": 2,
              "member-1": 1,
              "member-2": 1
            },
            "arbiters": 0,
            "readPreference": "primaryPreferred"
          },
          "sharding": {
            "enabled": false,
            "shards": 0,
            "configServers": 0
          },
          "storageEngine": "wiredTiger",
          "wiredTiger": {
            "cacheSize": "24GB",
            "journaling": true,
            "compression": "snappy"
          },
          "oplog": {
            "size": "50GB",
            "retentionHours": 48
          },
          "security": {
            "authentication": true,
            "authorization": true,
            "tlsMode": "requireTLS",
            "clusterAuthMode": "x509"
          },
          "backup": {
            "enabled": true,
            "method": "mongodump",
            "schedule": "0 2 * * *",
            "retention": 30,
            "lastBackup": "2025-02-13T02:00:00Z",
            "backupSize": "180GB"
          }
        },
        "databases": [
          {
            "name": "application",
            "sizeOnDisk": "145GB",
            "collections": 45,
            "indexes": 128,
            "views": 5
          },
          {
            "name": "analytics",
            "sizeOnDisk": "220GB",
            "collections": 12,
            "indexes": 35,
            "views": 8
          }
        ],
        "users": [
          {
            "username": "app_user",
            "database": "application",
            "roles": ["readWrite"],
            "customData": {}
          },
          {
            "username": "analytics_user",
            "database": "analytics",
            "roles": ["read"],
            "customData": {}
          }
        ],
        "metrics": {
          "cpu": {
            "usage": 42.5,
            "percentage": 42.5
          },
          "memory": {
            "usageGB": 35.2,
            "percentage": 73.3,
            "wiredTigerCache": "22GB"
          },
          "storage": {
            "usedGB": 365,
            "totalGB": 1000,
            "percentage": 36.5
          },
          "operations": {
            "insertsPerSec": 450,
            "queriesPerSec": 2100,
            "updatesPerSec": 320,
            "deletesPerSec": 45,
            "commandsPerSec": 180
          },
          "connections": {
            "current": 85,
            "available": 51200,
            "totalCreated": 12450
          },
          "replication": {
            "replicationLag": 0,
            "replicationLagUnit": "seconds",
            "oplogWindow": 47.5,
            "oplogWindowUnit": "hours"
          },
          "network": {
            "bytesInPerSec": 12500000,
            "bytesOutPerSec": 25000000,
            "requestsPerSec": 3115
          }
        },
        "monitoring": {
          "prometheusExporter": true,
          "grafanaDashboard": "https://grafana.company.com/d/mongodb-prod-001",
          "alertsEnabled": true
        },
        "cost": {
          "monthly": 6500,
          "currency": "USD"
        },
        "tags": [
          "production",
          "mongodb",
          "documentdb",
          "critical"
        ]
      },
      {
        "id": "iks-docdb-sharded-001",
        "name": "analytics-mongodb-sharded",
        "displayName": "Analytics MongoDB Sharded Cluster",
        "type": "IKSDocumentDB",
        "status": "running",
        "health": "healthy",
        "version": "6.0.8",
        "mongodbVersion": "MongoDB 6.0.8",
        "createdAt": "2024-03-10T14:20:00Z",
        "region": "chennai-amb",
        "zone": "EP_V2_CHN_AMB",
        "resources": {
          "vcpus": 24,
          "memoryGB": 96,
          "storageGB": 4000,
          "storageType": "nvme-ssd"
        },
        "configuration": {
          "replicaSet": {
            "name": null,
            "members": 0
          },
          "sharding": {
            "enabled": true,
            "shards": 3,
            "configServers": 3,
            "mongosRouters": 2,
            "shardKey": "userId",
            "chunksBalanced": true
          },
          "storageEngine": "wiredTiger",
          "backup": {
            "enabled": true,
            "method": "mongodump",
            "schedule": "0 3 * * *",
            "retention": 14
          }
        },
        "databases": [
          {
            "name": "events",
            "sizeOnDisk": "1850GB",
            "collections": 120,
            "sharded": true
          }
        ],
        "metrics": {
          "cpu": {
            "usage": 68.5,
            "percentage": 68.5
          },
          "memory": {
            "usageGB": 75.5,
            "percentage": 78.6
          },
          "storage": {
            "usedGB": 1950,
            "percentage": 48.8
          },
          "operations": {
            "insertsPerSec": 8500,
            "queriesPerSec": 12000,
            "updatesPerSec": 1200
          }
        },
        "cost": {
          "monthly": 15000,
          "currency": "USD"
        },
        "tags": [
          "analytics",
          "mongodb",
          "sharded",
          "timeseries"
        ]
      }
    ],
    "summary": {
      "totalServices": 8,
      "runningServices": 7,
      "stoppedServices": 1,
      "totalCostMonthly": 48000,
      "totalDatabases": 35,
      "totalStorageGB": 12000,
      "byVersion": {
        "6.0.8": 6,
        "5.0.20": 2
      },
      "byDeployment": {
        "replicaSet": 5,
        "sharded": 3
      }
    }
  },
  "message": "DocumentDB services retrieved successfully",
  "responseCode": 0
}
```

## Response Fields Details

### DocumentDB Service Fields
- **version** - MongoDB version
- **mongodbVersion** - Full MongoDB version string
- **connectionDetails** - Connection strings and hosts
- **configuration** - Replica set and sharding configuration
- **databases** - List of databases with size information
- **users** - Database users and their roles
- **metrics** - Real-time performance metrics
- **monitoring** - Monitoring integration

### Deployment Types
- `replicaSet` - Standard replica set (3+ members)
- `sharded` - Sharded cluster for horizontal scaling
- `standalone` - Single instance (dev/test only)

### Read Preferences
- `primary` - Read from primary only
- `primaryPreferred` - Prefer primary, fallback to secondary
- `secondary` - Read from secondary only
- `secondaryPreferred` - Prefer secondary, fallback to primary
- `nearest` - Read from nearest member

## Permissions
Roles: admin, developer, viewer

## Workflow Steps
### Workflow: list_documentdb_services
List all DocumentDB managed services
- Step 1: authenticate (auth.validate_token)
- Step 2: list_documentdb_services (managed_service.list_documentdb)

## Usage Notes
- Returns only MongoDB instances user has access to
- Metrics updated every 5 minutes
- Connection strings include replica set name
- Passwords never included in response
- SSL/TLS strongly recommended for all connections
- Sharded clusters require mongos router connections

## Common Use Cases
1. **Database inventory**: "List all MongoDB instances"
2. **Connection info**: "Get MongoDB connection string"
3. **Performance monitoring**: "Check MongoDB operations/sec"
4. **Replica set health**: "Show replica set status"
5. **Sharding status**: "Is cluster properly sharded?"
6. **Storage planning**: "MongoDB storage utilization"
7. **Version management**: "Which MongoDB versions deployed?"

## Data Processing Examples

### Get Connection String
```python
response = list_managed_services("IKSDocumentDB")
services = response['data']['services']

prod_mongo = next(
    (s for s in services if s['name'] == 'production-mongodb'),
    None
)

if prod_mongo:
    conn_str = prod_mongo['connectionDetails']['connectionString']
    print(f"MongoDB URI: {conn_str}")
```

### Check Replica Set Health
```python
response = list_managed_services("IKSDocumentDB")

for service in response['data']['services']:
    rs = service['configuration']['replicaSet']
    if rs['members'] > 0:
        metrics = service['metrics']['replication']
        lag = metrics['replicationLag']
        
        status = "🟢" if lag == 0 else "🟡" if lag < 5 else "🔴"
        print(f"{status} {service['name']}: {rs['members']} members, lag: {lag}s")
```

### Identify Sharded Clusters
```python
response = list_managed_services("IKSDocumentDB")

sharded_clusters = [
    s for s in response['data']['services']
    if s['configuration']['sharding']['enabled']
]

for cluster in sharded_clusters:
    sharding = cluster['configuration']['sharding']
    print(f"{cluster['name']}:")
    print(f"  Shards: {sharding['shards']}")
    print(f"  Config servers: {sharding['configServers']}")
    print(f"  Mongos routers: {sharding['mongosRouters']}")
```

### Calculate Operations Per Second
```python
response = list_managed_services("IKSDocumentDB")

print("MongoDB Operations Summary:")
for service in response['data']['services']:
    ops = service['metrics']['operations']
    total_ops = (ops['insertsPerSec'] + ops['queriesPerSec'] + 
                 ops['updatesPerSec'] + ops['deletesPerSec'])
    
    print(f"{service['name']}: {total_ops:,} ops/sec")
    print(f"  Queries: {ops['queriesPerSec']:,}/s")
    print(f"  Inserts: {ops['insertsPerSec']:,}/s")
```

## MongoDB-Specific Analysis

### WiredTiger Cache Analysis
```python
def analyze_cache(service):
    memory = service['metrics']['memory']
    wt_cache = float(memory.get('wiredTigerCache', '0GB').replace('GB', ''))
    total_mem = memory['usageGB']
    
    cache_ratio = (wt_cache / total_mem) * 100 if total_mem > 0 else 0
    
    return {
        'name': service['name'],
        'wiredTigerCache': wt_cache,
        'totalMemory': total_mem,
        'cacheRatio': cache_ratio,
        'recommendation': 'good' if 40 < cache_ratio < 70 else 'adjust'
    }
```

### Oplog Analysis
```python
response = list_managed_services("IKSDocumentDB")

for service in response['data']['services']:
    if 'replication' in service['metrics']:
        repl = service['metrics']['replication']
        window = repl['oplogWindow']
        
        print(f"{service['name']}:")
        print(f"  Oplog window: {window} hours")
        
        if window < 24:
            print(f"  ⚠️  Oplog window low! Consider increasing oplog size")
```

### Sharding Balance Check
```python
def check_shard_balance(service):
    if not service['configuration']['sharding']['enabled']:
        return None
    
    sharding = service['configuration']['sharding']
    
    return {
        'name': service['name'],
        'balanced': sharding.get('chunksBalanced', False),
        'shards': sharding['shards'],
        'needsBalancing': not sharding.get('chunksBalanced', True)
    }
```

## Integration Examples

### Python Connection
```python
from pymongo import MongoClient

response = list_managed_services("IKSDocumentDB")
service = response['data']['services'][0]

conn_details = service['connectionDetails']

client = MongoClient(
    conn_details['connectionString'],
    tls=True,
    tlsAllowInvalidCertificates=False
)

db = client['application']
collection = db['users']
```

### Node.js Connection
```javascript
const { MongoClient } = require('mongodb');

const service = services[0];
const uri = service.connectionDetails.connectionString;

const client = new MongoClient(uri, {
  tls: true,
  replicaSet: service.connectionDetails.replicaSet
});
```

## Backup and Recovery

### Verify Backup Status
```python
response = list_managed_services("IKSDocumentDB")

for service in response['data']['services']:
    backup = service['configuration']['backup']
    
    if backup['enabled']:
        from datetime import datetime
        last_backup = datetime.fromisoformat(backup['lastBackup'].replace('Z', '+00:00'))
        hours_ago = (datetime.now() - last_backup).total_seconds() / 3600
        
        status = "✅" if hours_ago < 26 else "⚠️"
        print(f"{status} {service['name']}: Last backup {hours_ago:.1f}h ago")
        print(f"  Backup size: {backup['backupSize']}")
```

## Performance Tuning

### WiredTiger Cache Sizing
- Default: 50% of RAM - 1GB
- Recommended: 50-60% of available RAM
- Monitor page faults and adjust if needed

### Connection Pool Sizing
- Light load: 50-100 connections
- Medium load: 100-500 connections
- Heavy load: 500-1000 connections

### Index Recommendations
- Create indexes for all query patterns
- Use compound indexes efficiently
- Monitor index usage with explain()
- Remove unused indexes

## Related Operations
- `documentdb.get_connection` - Get connection credentials
- `documentdb.create_database` - Create new database
- `documentdb.add_shard` - Add shard to cluster
- `documentdb.backup_now` - Trigger immediate backup
- `documentdb.restore` - Restore from backup

## Error Handling
- **401 Unauthorized:** Invalid token
- **403 Forbidden:** No access to DocumentDB services
- **404 Not Found:** No DocumentDB services found
- **500 Internal Server Error:** Service unavailable

## Response Codes
- `0` - Success
- `1` - Service type not found
- `2` - No services found

## Metadata
- **Generated:** 2025-02-13T13:00:00Z
- **Source:** Dynamic API Spec Generator
- **API Version:** v1
- **Base Path:** /paasservice/api/v1/paas
- **MongoDB Support:** Full managed MongoDB with sharding