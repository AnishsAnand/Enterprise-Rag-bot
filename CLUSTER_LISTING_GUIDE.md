# Cluster Listing Workflow Guide

## 📋 Overview

The cluster listing feature provides a multi-step workflow to fetch and display Kubernetes clusters across different data centers (endpoints).

## 🔄 Workflow Steps

### 1. **Authentication** (Automatic)
- System automatically fetches Bearer token using configured credentials
- Token is cached for 8 minutes (actual TTL: 10 minutes)
- Auto-refresh on expiry

### 2. **Engagement ID Fetch** (Automatic, Cached)
- Fetches engagement details for the authenticated user
- Response includes:
  - `id`: Engagement ID (e.g., 1923)
  - `engagementName`: Name of the engagement
  - `customerName`: Customer organization name
- **Cached for 1 hour** to avoid repeated API calls

### 3. **Endpoints Fetch** (Automatic)
- Fetches available data centers (endpoints) for the engagement
- Each endpoint includes:
  - `endpointId`: Numeric ID (e.g., 11, 12, 204)
  - `endpointDisplayName`: Human-readable name (e.g., "Delhi", "Mumbai-BKC")
  - `endpoint`: Technical identifier (e.g., "EP_V2_DEL")
  - `aiCloudEnabled`: Whether AI cloud features are available

### 4. **Cluster List** (Dynamic)
- Fetches clusters for selected endpoints
- Supports:
  - **All endpoints**: Pass `endpoint_ids=None`
  - **Specific endpoints**: Pass list like `[11, 12, 204]`
- Returns cluster details:
  - Cluster name, status, node count
  - Kubernetes version
  - Location, creation time
  - Type (MGMT/APP), backup status

## 📡 API Endpoints

### Engagement API
```
GET https://ipcloud.tatacommunications.com/paasservice/paas/engagements
Authorization: Bearer {token}
```

**Response:**
```json
{
  "status": "success",
  "data": [{
    "id": 1923,
    "engagementName": "Tata Communications (Innovations) Ltd-PaaS",
    "engagementType": "PaaS",
    "customerName": "Tata Communications Hong Kong Ltd"
  }]
}
```

### Endpoints API
```
GET https://ipcloud.tatacommunications.com/portalservice/configservice/getEndpointsByEngagement/{engagement_id}
Authorization: Bearer {token}
```

**Response:**
```json
{
  "status": "success",
  "data": [
    {
      "endpointId": 11,
      "endpointDisplayName": "Delhi",
      "endpoint": "EP_V2_DEL",
      "aiCloudEnabled": "no"
    }
  ]
}
```

### Cluster List API
```
POST https://ipcloud.tatacommunications.com/paasservice/paas/{engagement_id}/clusterlist
Authorization: Bearer {token}
Content-Type: application/json

{
  "endpoints": [11, 12, 14, 162, 204]
}
```

**Response:**
```json
{
  "status": "success",
  "data": [
    {
      "clusterId": 1115,
      "clusterName": "mum-uat-testing",
      "nodescount": "10",
      "kubernetesVersion": "v1.26.15",
      "status": "Healthy",
      "displayNameEndpoint": "Mumbai-BKC",
      "location": "EP_V2_MUM_BKC",
      "type": "MGMT",
      "isIksBackupEnabled": "true",
      "createdTime": "2025-04-08T04:38:38.000+00:00"
    }
  ]
}
```

## 🔧 Usage

### Programmatic Usage

```python
from app.services.api_executor_service import api_executor_service

# List all clusters (all endpoints)
result = await api_executor_service.list_clusters()

# List clusters for specific endpoints
result = await api_executor_service.list_clusters(
    endpoint_ids=[11, 12],  # Delhi and Bengaluru
    engagement_id=1923       # Optional, will be fetched if not provided
)

# Get engagement ID (with caching)
engagement_id = await api_executor_service.get_engagement_id()

# Get endpoints
endpoints = await api_executor_service.get_endpoints(engagement_id)
```

### Agent Chat Usage

Users can ask natural language queries:

```
User: "Show me all clusters"
User: "List clusters in Mumbai and Delhi"
User: "What clusters are available?"
User: "How many k8s clusters do we have?"
```

## 🎯 Key Features

### ✅ Automatic Token Management
- Fetches token on first use
- Caches for 8 minutes
- Auto-refreshes on expiry
- Thread-safe with async locks

### ✅ Engagement Caching
- Fetches engagement ID once
- Caches for 1 hour
- Reduces API calls
- Force refresh available

### ✅ Flexible Endpoint Selection
- Query all endpoints at once
- Select specific endpoints
- Future: Interactive endpoint selection in chat

### ✅ Rich Cluster Information
- Status (Healthy, Draft, etc.)
- Node count and Kubernetes version
- Location and creation time
- Backup status and cluster type

## 📊 Test Results

```
✅ Engagement fetch: PASSED
   - Engagement ID: 1923
   - Customer: Tata Communications Hong Kong Ltd

✅ Endpoints fetch: PASSED
   - 5 endpoints available
   - Delhi, Bengaluru, Cressex, Mumbai-BKC, Chennai-AMB

✅ Cluster listing: PASSED
   - 63 total clusters across all endpoints
   - Breakdown:
     • Bengaluru: 17 clusters
     • Chennai-AMB: 21 clusters (AI-enabled)
     • Cressex: 4 clusters
     • Delhi: 13 clusters
     • Mumbai-BKC: 8 clusters

✅ Caching: PASSED
   - Engagement ID cached for 1 hour
   - Token cached for 8 minutes
```

## 🔐 Environment Variables

```bash
# Required
API_AUTH_EMAIL=your-email@example.com
API_AUTH_PASSWORD=your-password
API_AUTH_URL=https://ipcloud.tatacommunications.com/portalservice/api/v1/getAuthToken

# Optional (defaults shown)
API_EXECUTOR_TIMEOUT=30
API_EXECUTOR_MAX_RETRIES=3
```

## 🧪 Testing

Run the test script:

```bash
python test_cluster_list.py
```

This will test:
1. Engagement ID fetch and caching
2. Endpoints fetch
3. Cluster listing (all endpoints)
4. Cluster listing (specific endpoints)
5. Cache validation

## 📝 Resource Schema

The workflow is defined in `app/config/resource_schema.json`:

```json
{
  "resources": {
    "engagement": {
      "operations": ["get"],
      "api_endpoints": { ... }
    },
    "endpoint": {
      "operations": ["list"],
      "api_endpoints": { ... }
    },
    "k8s_cluster": {
      "operations": ["list", "create", "update", "delete"],
      "workflow": {
        "list_clusters": {
          "steps": [
            "get_engagement",
            "get_endpoints",
            "list_clusters"
          ]
        }
      }
    }
  }
}
```

## 🚀 Next Steps

1. **Interactive Endpoint Selection**: Allow users to select specific endpoints in chat
2. **Cluster Filtering**: Add filters for status, location, K8s version
3. **Cluster Details**: Fetch detailed info for specific clusters
4. **Cluster Operations**: Implement create, update, delete workflows
5. **Health Monitoring**: Real-time cluster health checks
6. **Cost Analysis**: Show resource utilization and costs

## 🐛 Troubleshooting

### Token Authentication Failed
```
❌ API_AUTH_EMAIL or API_AUTH_PASSWORD not configured
```
**Solution**: Ensure credentials are set in `.env` file

### Engagement ID Not Found
```
❌ Failed to fetch engagement ID
```
**Solution**: Verify user has access to at least one engagement

### No Clusters Found
```
⚠️ No clusters found for engagement
```
**Solution**: Check if endpoints have any clusters deployed

### API Timeout
```
❌ httpx.ReadTimeout
```
**Solution**: Increase `API_EXECUTOR_TIMEOUT` in environment variables

## 📚 Related Files

- `app/services/api_executor_service.py`: Main service implementation
- `app/config/resource_schema.json`: Resource and operation definitions
- `test_cluster_list.py`: Comprehensive test script
- `TOKEN_AUTH_SETUP.md`: Token authentication documentation
- `.env`: Environment configuration

---

**Last Updated**: 2025-11-21  
**Status**: ✅ Production Ready

