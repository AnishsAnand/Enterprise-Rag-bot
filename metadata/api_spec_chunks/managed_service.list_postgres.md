# API Specification: managed_service - list_postgres

**Resource:** managed_service
**Operation:** list_postgres
**Aliases:** postgres, postgresql, list postgres, postgres services, postgres databases, IKS postgres

## Endpoint
- **Method:** GET
- **URL:** {BASE_URL_PAAS_SERVICE}/api/v1/paas/listManagedServices/IKSPostgres
- **Auth:** Bearer token (from Keycloak)
- **Description:** List all IKS-managed PostgreSQL database instances with their configuration, status, performance metrics, and connection details. Part of the Integrated Kubernetes Service (IKS) platform

## Required Parameters
None (service type is in URL: `IKSPostgres`)

## Optional Parameters
- `status` - Filter by status (running, stopped, creating, error)
- `region` - Filter by region/zone
- `version` - Filter by PostgreSQL version
- `include_metrics` - Include performance metrics (true/false)

## Response Mapping
- `status`: status
- `message`: message
- `service_type`: data.serviceType
- `total_count`: data.totalCount
- `services`: data.services
- `service_ids`: data.services[*].id
- `service_names`: data.services[*].name
- `postgres_versions`: data.services[*].version
- `connection_strings`: data.services[*].connectionString

## Response Example
```json
{
  "status": "success",
  "data": {
    "serviceType": "IKSPostgres",
    "serviceDescription": "Integrated Kubernetes Service - PostgreSQL Database",
    "totalCount": 15,
    "services": [
      {
        "id": "iks-postgres-prod-001",
        "name": "production-postgres",
        "displayName": "Production PostgreSQL Primary",
        "type": "IKSPostgres",
        "status": "running",
        "health": "healthy",
        "version": "15.4",
        "postgresVersion": "PostgreSQL 15.4",
        "createdAt": "2024-01-15T10:30:00Z",
        "lastModified": "2025-02-13T08:45:00Z",
        "region": "mumbai-bkc",
        "zone": "EP_V2_MUM_BKC",
        "cluster": {
          "id": "cluster-1267",
          "name": "prod-k8s-01",
          "nodesCount": 5
        },
        "connectionDetails": {
          "host": "postgres-prod-001.company.internal",
          "port": 5432,
          "database": "production_db",
          "sslMode": "require",
          "maxConnections": 200,
          "connectionString": "postgresql://postgres-prod-001.company.internal:5432/production_db?sslmode=require"
        },
        "resources": {
          "vcpus": 8,
          "memoryGB": 32,
          "storageGB": 500,
          "storageType": "ssd-premium",
          "iops": 6000
        },
        "configuration": {
          "postgres": {
            "maxConnections": 200,
            "sharedBuffers": "8GB",
            "effectiveCacheSize": "24GB",
            "maintenanceWorkMem": "2GB",
            "workMem": "41MB",
            "walLevel": "replica",
            "maxWalSize": "4GB",
            "maxReplicationSlots": 10,
            "logStatement": "all",
            "logDuration": true
          },
          "highAvailability": {
            "enabled": true,
            "replicaCount": 2,
            "replicationMode": "streaming",
            "automaticFailover": true,
            "failoverTimeout": 30
          },
          "backup": {
            "enabled": true,
            "method": "pg_basebackup",
            "schedule": "0 2 * * *",
            "retention": 30,
            "pointInTimeRecovery": true,
            "walArchiving": true,
            "lastBackup": "2025-02-13T02:00:00Z",
            "backupSize": "120GB"
          },
          "kubernetes": {
            "namespace": "postgres-production",
            "storageClass": "ssd-premium",
            "persistentVolumeClaim": "postgres-data-prod",
            "serviceType": "ClusterIP"
          }
        },
        "databases": [
          {
            "name": "production_db",
            "size": "85GB",
            "owner": "app_user",
            "encoding": "UTF8",
            "collation": "en_US.UTF-8",
            "connectionLimit": -1
          },
          {
            "name": "analytics_db",
            "size": "120GB",
            "owner": "analytics_user",
            "encoding": "UTF8",
            "collation": "en_US.UTF-8",
            "connectionLimit": 50
          }
        ],
        "users": [
          {
            "username": "app_user",
            "roles": ["CREATEDB", "REPLICATION"],
            "connectionLimit": 100,
            "validUntil": null
          },
          {
            "username": "analytics_user",
            "roles": ["READONLY"],
            "connectionLimit": 50,
            "validUntil": null
          }
        ],
        "metrics": {
          "cpu": {
            "usage": 35.5,
            "limit": 100,
            "percentage": 35.5
          },
          "memory": {
            "usageGB": 22.5,
            "limitGB": 32,
            "percentage": 70.3
          },
          "storage": {
            "usedGB": 205,
            "totalGB": 500,
            "percentage": 41.0
          },
          "connections": {
            "active": 45,
            "idle": 15,
            "max": 200,
            "percentage": 30.0
          },
          "transactions": {
            "commitsPerSec": 850,
            "rollbacksPerSec": 12,
            "commitRatio": 98.6
          },
          "queries": {
            "queriesPerSec": 1200,
            "avgQueryTime": 15.5,
            "slowQueries": 3
          },
          "replication": {
            "replicationLag": 250,
            "replicationLagUnit": "bytes",
            "replicasHealthy": 2,
            "replicasTotal": 2
          }
        },
        "extensions": [
          "pg_stat_statements",
          "postgis",
          "uuid-ossp",
          "hstore",
          "pg_trgm"
        ],
        "monitoring": {
          "prometheusExporter": true,
          "grafanaDashboard": "https://grafana.company.com/d/postgres-prod-001",
          "alertsEnabled": true
        },
        "cost": {
          "monthly": 4500,
          "currency": "USD",
          "billingPeriod": "2025-02"
        },
        "tags": [
          "production",
          "postgres",
          "database",
          "critical"
        ]
      },
      {
        "id": "iks-postgres-analytics-001",
        "name": "analytics-postgres",
        "displayName": "Analytics PostgreSQL",
        "type": "IKSPostgres",
        "status": "running",
        "health": "healthy",
        "version": "15.4",
        "postgresVersion": "PostgreSQL 15.4 with PostGIS",
        "createdAt": "2024-02-20T14:15:00Z",
        "region": "chennai-amb",
        "zone": "EP_V2_CHN_AMB",
        "cluster": {
          "id": "cluster-1501",
          "name": "analytics-k8s-01",
          "nodesCount": 6
        },
        "connectionDetails": {
          "host": "postgres-analytics-001.company.internal",
          "port": 5432,
          "database": "analytics",
          "sslMode": "require",
          "maxConnections": 300
        },
        "resources": {
          "vcpus": 16,
          "memoryGB": 64,
          "storageGB": 2000,
          "storageType": "nvme-ssd",
          "iops": 15000
        },
        "configuration": {
          "postgres": {
            "maxConnections": 300,
            "sharedBuffers": "16GB",
            "effectiveCacheSize": "48GB",
            "maintenanceWorkMem": "4GB",
            "workMem": "52MB"
          },
          "highAvailability": {
            "enabled": true,
            "replicaCount": 1,
            "replicationMode": "streaming",
            "automaticFailover": true
          },
          "backup": {
            "enabled": true,
            "method": "pg_basebackup",
            "schedule": "0 3 * * *",
            "retention": 14
          }
        },
        "databases": [
          {
            "name": "analytics",
            "size": "850GB",
            "owner": "analytics_admin",
            "encoding": "UTF8"
          }
        ],
        "metrics": {
          "cpu": {
            "usage": 65.2,
            "percentage": 65.2
          },
          "memory": {
            "usageGB": 52.5,
            "percentage": 82.0
          },
          "storage": {
            "usedGB": 950,
            "percentage": 47.5
          },
          "connections": {
            "active": 125,
            "max": 300,
            "percentage": 41.7
          }
        },
        "extensions": [
          "pg_stat_statements",
          "postgis",
          "timescaledb",
          "pg_partman"
        ],
        "cost": {
          "monthly": 8500,
          "currency": "USD"
        },
        "tags": [
          "analytics",
          "postgres",
          "database",
          "timeseries"
        ]
      }
    ],
    "summary": {
      "totalServices": 15,
      "runningServices": 13,
      "stoppedServices": 1,
      "errorServices": 1,
      "totalCostMonthly": 52000,
      "totalDatabases": 45,
      "totalStorageGB": 8500,
      "totalConnections": 850,
      "byVersion": {
        "15.4": 12,
        "14.9": 2,
        "13.12": 1
      },
      "byRegion": {
        "mumbai-bkc": 6,
        "chennai-amb": 5,
        "delhi": 4
      }
    }
  },
  "message": "PostgreSQL services retrieved successfully",
  "responseCode": 0
}
```

## Response Fields Details

### PostgreSQL Service Fields
- **id** - Unique service identifier
- **name** - Service instance name
- **version** - PostgreSQL version
- **postgresVersion** - Full PostgreSQL version string
- **connectionDetails** - Connection information
- **resources** - CPU, memory, storage allocation
- **configuration** - PostgreSQL and HA configuration
- **databases** - List of databases in this instance
- **users** - Database users and roles
- **metrics** - Real-time performance metrics
- **extensions** - Installed PostgreSQL extensions
- **monitoring** - Monitoring integration details
- **cost** - Cost information

### PostgreSQL Versions Supported
- `15.x` - PostgreSQL 15 (current)
- `14.x` - PostgreSQL 14
- `13.x` - PostgreSQL 13 (legacy)

### High Availability Modes
- `streaming` - Streaming replication (most common)
- `logical` - Logical replication
- `synchronous` - Synchronous replication
- `asynchronous` - Asynchronous replication

## Permissions
Roles: admin, developer, viewer

## Workflow Steps
### Workflow: list_postgres_services
List all PostgreSQL managed services
- Step 1: authenticate (auth.validate_token)
- Step 2: list_postgres_services (managed_service.list_postgres)

## Usage Notes
- Returns only PostgreSQL instances user has access to
- Metrics updated every 5 minutes
- Connection details may require additional permissions
- Passwords never included in response
- SSL/TLS enabled by default for all connections
- Point-in-time recovery available with WAL archiving

## Common Use Cases
1. **Database inventory**: "List all PostgreSQL instances"
2. **Connection info**: "Get PostgreSQL connection details"
3. **Performance monitoring**: "Check PostgreSQL CPU usage"
4. **HA status**: "Show replica status"
5. **Backup verification**: "When was last backup?"
6. **Cost analysis**: "Total PostgreSQL monthly cost"
7. **Version management**: "Which instances need upgrade?"

## Data Processing Examples

### Get Connection String
```python
response = list_managed_services("IKSPostgres")
services = response['data']['services']

prod_db = next(
    (s for s in services if s['name'] == 'production-postgres'),
    None
)

if prod_db:
    conn_str = prod_db['connectionDetails']['connectionString']
    host = prod_db['connectionDetails']['host']
    port = prod_db['connectionDetails']['port']
    print(f"Connection: {conn_str}")
```

### Check Replication Health
```python
response = list_managed_services("IKSPostgres")

for service in response['data']['services']:
    if 'highAvailability' in service['configuration']:
        ha = service['configuration']['highAvailability']
        if ha['enabled']:
            metrics = service['metrics']['replication']
            healthy = metrics['replicasHealthy']
            total = metrics['replicasTotal']
            lag = metrics['replicationLag']
            
            status = "🟢" if healthy == total else "🟡" if healthy > 0 else "🔴"
            print(f"{status} {service['name']}: {healthy}/{total} replicas, lag: {lag}B")
```

### Find Databases by Size
```python
response = list_managed_services("IKSPostgres")

all_databases = []
for service in response['data']['services']:
    for db in service['databases']:
        all_databases.append({
            'service': service['name'],
            'database': db['name'],
            'size': db['size'],
            'owner': db['owner']
        })

# Sort by size
all_databases.sort(key=lambda x: x['size'], reverse=True)

print("Largest databases:")
for db in all_databases[:10]:
    print(f"  {db['database']} ({db['service']}): {db['size']}")
```

### Calculate Total Costs
```python
response = list_managed_services("IKSPostgres")
services = response['data']['services']

total_cost = sum(s['cost']['monthly'] for s in services)
prod_cost = sum(
    s['cost']['monthly'] for s in services
    if 'production' in s.get('tags', [])
)

print(f"Total PostgreSQL: ${total_cost:,}/month")
print(f"Production only: ${prod_cost:,}/month")
```

## PostgreSQL-Specific Analysis

### Connection Pool Analysis
```python
def analyze_connections(service):
    metrics = service['metrics']['connections']
    active = metrics['active']
    idle = metrics['idle']
    max_conn = metrics['max']
    total = active + idle
    
    utilization = (total / max_conn) * 100
    
    return {
        'name': service['name'],
        'active': active,
        'idle': idle,
        'total': total,
        'max': max_conn,
        'utilization': utilization,
        'status': 'good' if utilization < 70 else 'warning' if utilization < 85 else 'critical'
    }
```

### Query Performance Tracking
```python
response = list_managed_services("IKSPostgres")

print("PostgreSQL Query Performance:")
print("-" * 70)

for service in response['data']['services']:
    queries = service['metrics']['queries']
    
    print(f"\n{service['name']}:")
    print(f"  Queries/sec: {queries['queriesPerSec']:,}")
    print(f"  Avg query time: {queries['avgQueryTime']}ms")
    print(f"  Slow queries: {queries['slowQueries']}")
    
    if queries['avgQueryTime'] > 50:
        print(f"  ⚠️  High average query time!")
    if queries['slowQueries'] > 10:
        print(f"  ⚠️  Many slow queries detected!")
```

### Storage Growth Prediction
```python
def predict_storage_full(service):
    storage = service['metrics']['storage']
    used = storage['usedGB']
    total = storage['totalGB']
    
    # Assume 2GB/day growth (adjust based on actual)
    growth_rate = 2.0
    available = total - used
    days_until_full = available / growth_rate
    
    return {
        'name': service['name'],
        'used': used,
        'total': total,
        'available': available,
        'daysUntilFull': max(0, days_until_full),
        'needsExpansion': days_until_full < 30
    }
```

## Integration with Applications

### Connection Example (Python)
```python
import psycopg2

response = list_managed_services("IKSPostgres")
service = response['data']['services'][0]

conn_details = service['connectionDetails']

conn = psycopg2.connect(
    host=conn_details['host'],
    port=conn_details['port'],
    database=conn_details['database'],
    user='app_user',
    password='<from-secrets>',
    sslmode=conn_details['sslMode']
)
```

### Health Check
```python
def check_postgres_health():
    response = list_managed_services("IKSPostgres")
    unhealthy = []
    
    for service in response['data']['services']:
        if service['health'] != 'healthy':
            unhealthy.append({
                'name': service['name'],
                'health': service['health'],
                'status': service['status']
            })
    
    return unhealthy
```

## Backup and Recovery

### Verify Backup Status
```python
response = list_managed_services("IKSPostgres")

for service in response['data']['services']:
    backup = service['configuration']['backup']
    
    if backup['enabled']:
        from datetime import datetime, timedelta
        last_backup = datetime.fromisoformat(backup['lastBackup'].replace('Z', '+00:00'))
        hours_ago = (datetime.now() - last_backup).total_seconds() / 3600
        
        status = "✅" if hours_ago < 26 else "⚠️"
        print(f"{status} {service['name']}: Last backup {hours_ago:.1f}h ago")
    else:
        print(f"❌ {service['name']}: Backups disabled!")
```

## Related Operations
- `managed_service.list` - List all managed service types
- `postgres.get_connection` - Get connection credentials
- `postgres.create_database` - Create new database
- `postgres.create_user` - Create database user
- `postgres.backup_now` - Trigger immediate backup
- `postgres.restore` - Restore from backup

## Error Handling
- **401 Unauthorized:** Invalid or expired authentication token
- **403 Forbidden:** User does not have access to PostgreSQL services
- **404 Not Found:** No PostgreSQL services found
- **500 Internal Server Error:** Service unavailable

## Response Codes
- `0` - Success
- `1` - Service type not found
- `2` - No services found
- `3` - Access denied

## PostgreSQL Extensions

### Common Extensions
- `pg_stat_statements` - Query statistics
- `postgis` - Geographic data support
- `timescaledb` - Time-series data
- `pg_partman` - Partition management
- `uuid-ossp` - UUID generation
- `hstore` - Key-value store
- `pg_trgm` - Trigram matching
- `pgcrypto` - Cryptographic functions

## Performance Tuning Guidelines

### Memory Configuration
- `shared_buffers`: 25% of total RAM
- `effective_cache_size`: 50-75% of total RAM
- `maintenance_work_mem`: 5-10% of RAM
- `work_mem`: (Total RAM / max_connections) / 10

### Connection Limits
- Small instances: 100-200 connections
- Medium instances: 200-400 connections
- Large instances: 400-600 connections

### Storage Recommendations
- Use SSD or NVMe for best performance
- Separate WAL on different disk if possible
- Monitor IOPS utilization
- Plan for 2-3x data growth

## Metadata
- **Generated:** 2025-02-13T13:00:00Z
- **Source:** Dynamic API Spec Generator
- **API Version:** v1
- **Base Path:** /paasservice/api/v1/paas
- **PostgreSQL Support:** Full managed PostgreSQL with HA