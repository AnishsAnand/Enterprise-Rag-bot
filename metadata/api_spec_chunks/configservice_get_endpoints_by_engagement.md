# API Specification: configservice - get_endpoints_by_engagement

**Resource:** configservice
**Operation:** get_endpoints_by_engagement
**Aliases:** get endpoints, list endpoints, show endpoints, endpoints for engagement, engagement endpoints

## Endpoint
- **Method:** GET
- **URL:** `{BASE_URL_PORTAL_SERVICE}/configservice/getEndpointsByEngagement/{engagement_id}`
- **Auth:** Bearer token (assumed - needs confirmation)
- **Description:** Retrieves all configured endpoints associated with a specific engagement. Returns endpoint URLs, types, and configuration details for accessing various services within the engagement.

## Required Parameters
- `engagement_id` - IPC engagement ID (NOT PaaS ID - must convert first) (type: path parameter, format: string)

## Optional Parameters
None

## Response Mapping
**⚠️ Note: Actual API response needed to complete this section accurately**

Assumed mappings (to be verified):
- `endpoints`: data.endpoints
- `endpoint_urls`: data.endpoints[*].url
- `endpoint_types`: data.endpoints[*].type
- `endpoint_names`: data.endpoints[*].name
- `endpoint_statuses`: data.endpoints[*].status
- `engagement_id`: data.engagement_id
- `status`: status
- `message`: message

## Response Example
**⚠️ Placeholder - Replace with actual API response**

```json
{
  "status": "success",
  "data": {
    "engagement_id": "ipc-67890",
    "endpoints": [
      {
        "id": "ep-001",
        "name": "api-gateway",
        "type": "api",
        "url": "https://api.example.com/v1",
        "status": "active",
        "protocol": "https",
        "port": 443
      },
      {
        "id": "ep-002",
        "name": "kafka-broker",
        "type": "kafka",
        "url": "kafka.example.com:9092",
        "status": "active",
        "protocol": "kafka",
        "port": 9092
      },
      {
        "id": "ep-003",
        "name": "postgres-db",
        "type": "database",
        "url": "postgres.example.com:5432",
        "status": "active",
        "protocol": "postgresql",
        "port": 5432
      }
    ],
    "total_endpoints": 3
  }
}
```

## Response Fields Details

### Core Fields
- **status** - Response status (success, error)
- **engagement_id** - IPC engagement ID
- **endpoints** - Array of endpoint configurations
- **total_endpoints** - Count of endpoints

### Endpoint Object Fields
- **id** - Unique endpoint identifier
- **name** - Human-readable endpoint name
- **type** - Endpoint type (api, kafka, database, storage, etc.)
- **url** - Full endpoint URL or connection string
- **status** - Endpoint status (active, inactive, maintenance)
- **protocol** - Communication protocol (https, kafka, postgresql, etc.)
- **port** - Port number for connections

## Permissions
**Roles:** To be documented (requires actual permission information)

## Workflow Steps

### Workflow: Access Engagement Services
**Prerequisites:**
- Step 1: **Convert Engagement ID** (engagement.convert_paas_to_ipc) - Convert PaaS ID to IPC ID first!

**Main Workflow:**
- Step 2: **Get Endpoints** (configservice.get_endpoints_by_engagement) - Retrieve all endpoints
- Step 3: **Use Endpoints** - Connect to services using endpoint URLs

### Example Workflow
```
1. convert_paas_to_ipc("paas-12345") → "ipc-67890"
2. get_endpoints_by_engagement("ipc-67890") → List of endpoints
3. Use endpoints to connect to Kafka, databases, APIs, etc.
```

## Usage Notes
- **CRITICAL:** This endpoint requires IPC engagement ID, NOT PaaS ID
- Always call `getIpcEngFromPaasEng` first if you only have a PaaS ID
- Endpoints may include various service types: APIs, databases, message brokers, storage
- Cache endpoint list during session to avoid repeated API calls
- Check endpoint status before attempting connections

## Common Use Cases
1. **List all endpoints**: "show endpoints for engagement", "get all endpoints", "list engagement endpoints"
2. **Find specific service**: "get kafka endpoint", "find database endpoint", "api gateway url"
3. **Service discovery**: "what services are available", "show all services for engagement"
4. **Connection setup**: "get connection details", "show endpoint urls"

## Query Interpretations
- "get endpoints for {id}" → GET /getEndpointsByEngagement/{id}
- "list endpoints {id}" → GET /getEndpointsByEngagement/{id}
- "show services for {id}" → GET /getEndpointsByEngagement/{id}

## Data Processing Examples

### Python Example
```python
import requests

def get_endpoints_by_engagement(engagement_id, auth_token):
    """
    Get all endpoints for an engagement.
    
    Args:
        engagement_id: IPC engagement ID (not PaaS ID!)
        auth_token: Bearer authentication token
        
    Returns:
        dict: Endpoint configuration data
    """
    url = f"https://ipcloud.tatacommunications.com/portalservice/configservice/getEndpointsByEngagement/{engagement_id}"
    
    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        if data.get('status') == 'success':
            endpoints = data['data']['endpoints']
            print(f"Found {len(endpoints)} endpoints")
            return data['data']
        else:
            print(f"Failed: {data.get('message')}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"API error: {e}")
        return None

def get_endpoint_by_type(engagement_id, auth_token, endpoint_type):
    """Get endpoints filtered by type"""
    data = get_endpoints_by_engagement(engagement_id, auth_token)
    
    if not data:
        return []
    
    return [ep for ep in data['endpoints'] if ep['type'] == endpoint_type]

# Usage
ipc_id = "ipc-67890"  # Must be IPC ID, not PaaS ID!
token = "your-token-here"

# Get all endpoints
endpoints = get_endpoints_by_engagement(ipc_id, token)

# Filter by type
kafka_endpoints = get_endpoint_by_type(ipc_id, token, 'kafka')
db_endpoints = get_endpoint_by_type(ipc_id, token, 'database')

print(f"Kafka endpoints: {kafka_endpoints}")
print(f"Database endpoints: {db_endpoints}")
```

### JavaScript Example
```javascript
async function getEndpointsByEngagement(engagementId, authToken) {
    const url = `https://ipcloud.tatacommunications.com/portalservice/configservice/getEndpointsByEngagement/${engagementId}`;
    
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
            console.log(`Found ${data.data.endpoints.length} endpoints`);
            return data.data;
        } else {
            console.error(`Failed: ${data.message}`);
            return null;
        }
    } catch (error) {
        console.error(`API error: ${error.message}`);
        return null;
    }
}

function getEndpointByType(endpoints, type) {
    return endpoints.filter(ep => ep.type === type);
}

// Usage
const ipcId = 'ipc-67890';  // Must be IPC ID!
const token = 'your-token-here';

const data = await getEndpointsByEngagement(ipcId, token);

if (data) {
    const kafkaEndpoints = getEndpointByType(data.endpoints, 'kafka');
    const dbEndpoints = getEndpointByType(data.endpoints, 'database');
    
    console.log('Kafka endpoints:', kafkaEndpoints);
    console.log('Database endpoints:', dbEndpoints);
}
```

## Integration Examples

### Complete Workflow with ID Conversion
```python
class EngagementServiceClient:
    def __init__(self, base_url, auth_token):
        self.base_url = base_url
        self.auth_token = auth_token
        self.ipc_cache = {}
        self.endpoint_cache = {}
    
    def _headers(self):
        return {
            "Authorization": f"Bearer {self.auth_token}",
            "Content-Type": "application/json"
        }
    
    def convert_to_ipc(self, paas_id):
        """Convert PaaS ID to IPC ID with caching"""
        if paas_id in self.ipc_cache:
            return self.ipc_cache[paas_id]
        
        url = f"{self.base_url}/paasservice/paas/getIpcEngFromPaasEng/{paas_id}"
        response = requests.get(url, headers=self._headers())
        response.raise_for_status()
        
        ipc_id = response.json()['data']['ipc_engagement_id']
        self.ipc_cache[paas_id] = ipc_id
        return ipc_id
    
    def get_endpoints(self, engagement_id, is_paas_id=True):
        """
        Get endpoints - automatically handles ID conversion
        
        Args:
            engagement_id: PaaS or IPC engagement ID
            is_paas_id: True if engagement_id is PaaS ID (default)
        """
        # Convert if needed
        ipc_id = self.convert_to_ipc(engagement_id) if is_paas_id else engagement_id
        
        # Check cache
        if ipc_id in self.endpoint_cache:
            return self.endpoint_cache[ipc_id]
        
        # Fetch endpoints
        url = f"{self.base_url}/portalservice/configservice/getEndpointsByEngagement/{ipc_id}"
        response = requests.get(url, headers=self._headers())
        response.raise_for_status()
        
        data = response.json()['data']
        self.endpoint_cache[ipc_id] = data
        return data
    
    def get_service_url(self, engagement_id, service_type, is_paas_id=True):
        """Get URL for specific service type"""
        endpoints = self.get_endpoints(engagement_id, is_paas_id)
        
        for ep in endpoints['endpoints']:
            if ep['type'] == service_type and ep['status'] == 'active':
                return ep['url']
        
        raise ValueError(f"No active {service_type} endpoint found")

# Usage
client = EngagementServiceClient("https://ipcloud.tatacommunications.com", "token")

# Can use PaaS ID - auto-converts
kafka_url = client.get_service_url("paas-12345", "kafka", is_paas_id=True)
db_url = client.get_service_url("paas-12345", "database", is_paas_id=True)

print(f"Kafka: {kafka_url}")
print(f"Database: {db_url}")
```

### Service Connection Helper
```python
from urllib.parse import urlparse

class EndpointConnectionHelper:
    @staticmethod
    def parse_endpoint(endpoint):
        """Parse endpoint to connection details"""
        parsed = urlparse(endpoint['url'])
        
        return {
            'host': parsed.hostname or endpoint['url'].split(':')[0],
            'port': endpoint.get('port', parsed.port),
            'protocol': endpoint['protocol'],
            'type': endpoint['type'],
            'name': endpoint['name']
        }
    
    @staticmethod
    def get_kafka_config(endpoints):
        """Get Kafka connection configuration"""
        kafka_eps = [ep for ep in endpoints if ep['type'] == 'kafka']
        
        if not kafka_eps:
            raise ValueError("No Kafka endpoints found")
        
        # Build broker list
        brokers = [f"{ep['url']}" for ep in kafka_eps if ep['status'] == 'active']
        
        return {
            'bootstrap_servers': brokers,
            'security_protocol': 'SASL_SSL',  # Adjust as needed
            'api_version': (2, 5, 0)
        }
    
    @staticmethod
    def get_db_connection_string(endpoint):
        """Get database connection string"""
        if endpoint['type'] != 'database':
            raise ValueError("Not a database endpoint")
        
        host = endpoint['url'].split(':')[0]
        port = endpoint.get('port', 5432)
        
        return f"postgresql://user:password@{host}:{port}/dbname"

# Usage
endpoints = client.get_endpoints("paas-12345")['endpoints']

# Kafka configuration
kafka_config = EndpointConnectionHelper.get_kafka_config(endpoints)
print(f"Kafka brokers: {kafka_config['bootstrap_servers']}")

# Database connection
db_ep = next(ep for ep in endpoints if ep['type'] == 'database')
conn_str = EndpointConnectionHelper.get_db_connection_string(db_ep)
print(f"Database: {conn_str}")
```

## Related Operations
- `engagement.convert_paas_to_ipc` - PREREQUISITE: Convert PaaS ID to IPC ID
- `securityservice.get_department_details` - Get department details for engagement
- `cluster.list` - Get cluster information using engagement context

## Error Handling
- **400 Bad Request:** Invalid engagement ID format - verify ID structure
- **401 Unauthorized:** Invalid or expired authentication token - refresh token
- **404 Not Found:** Engagement not found OR you're using PaaS ID instead of IPC ID
  - **Solution:** Always convert PaaS ID to IPC ID first using `getIpcEngFromPaasEng`
- **500 Internal Server Error:** Service error - retry with exponential backoff
- **503 Service Unavailable:** Service temporarily unavailable - implement retry logic

## Response Codes
**⚠️ To be documented - requires actual API testing**

Likely patterns:
- `0` or `success` - Request successful
- Non-zero or `error` - Request failed (check message field)

## Performance Notes
- Response time typically < 500ms depending on number of endpoints
- Results can be cached per session (endpoints don't change frequently)
- No pagination (single engagement query)
- Rate limiting: To be documented

## Best Practices

### Always Convert PaaS ID First
```python
# ❌ WRONG - Using PaaS ID directly
try:
    endpoints = get_endpoints_by_engagement("paas-12345", token)
    # This will FAIL with 404!
except Exception as e:
    print(f"Failed: {e}")

# ✅ RIGHT - Convert PaaS to IPC first
paas_id = "paas-12345"
ipc_id = convert_paas_to_ipc(paas_id, token)  # Convert first!
endpoints = get_endpoints_by_engagement(ipc_id, token)  # Now works!
```

### Cache Endpoints
```python
class EndpointCache:
    def __init__(self, ttl=3600):
        self.cache = {}
        self.ttl = ttl
    
    def get(self, engagement_id):
        if engagement_id in self.cache:
            entry = self.cache[engagement_id]
            if time.time() - entry['timestamp'] < self.ttl:
                return entry['data']
        return None
    
    def set(self, engagement_id, data):
        self.cache[engagement_id] = {
            'data': data,
            'timestamp': time.time()
        }

# Use cache to avoid repeated API calls
cache = EndpointCache(ttl=1800)  # 30 minutes
endpoints = cache.get(ipc_id)
if not endpoints:
    endpoints = get_endpoints_by_engagement(ipc_id, token)
    cache.set(ipc_id, endpoints)
```

### Filter Active Endpoints Only
```python
def get_active_endpoints(endpoints):
    """Get only active endpoints"""
    return [ep for ep in endpoints if ep.get('status') == 'active']

# Usage
all_endpoints = get_endpoints_by_engagement(ipc_id, token)
active = get_active_endpoints(all_endpoints['endpoints'])
print(f"Active endpoints: {len(active)}/{len(all_endpoints['endpoints'])}")
```

## Metadata
- **Generated:** 2024-02-23T10:00:00Z
- **Source:** API endpoint analysis
- **API Version:** To be documented
- **Base Path:** {BASE_URL_PORTAL_SERVICE}/configservice
- **Additional notes:** ⚠️ This documentation is preliminary. Please provide actual API response for complete accuracy. CRITICAL: This endpoint requires IPC engagement ID - always convert PaaS ID first!
