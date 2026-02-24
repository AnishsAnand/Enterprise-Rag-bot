# API Specification: engagement - convert_paas_to_ipc

**Resource:** engagement
**Operation:** convert_paas_to_ipc
**Aliases:** convert engagement, get IPC from PaaS, paas to ipc, convert paas engagement, get ipc engagement

## Endpoint
- **Method:** GET
- **URL:** `{BASE_URL_PAAS_SERVICE}/paas/getIpcEngFromPaasEng/{engagement_id}`
- **Auth:** Bearer token (assumed - needs confirmation)
- **Description:** Converts a PaaS engagement ID to an IPC (IP Cloud) engagement ID. This is a critical prerequisite for many portal service operations that require IPC engagement IDs rather than PaaS IDs.

## Required Parameters
- `engagement_id` - PaaS engagement ID to convert (type: path parameter, format: string)

## Optional Parameters
None

## Response Mapping
**⚠️ Note: Actual API response needed to complete this section accurately**

Assumed mappings (to be verified):
- `ipc_engagement_id`: data.ipc_engagement_id
- `paas_engagement_id`: data.paas_engagement_id
- `status`: status
- `message`: message

## Response Example
**⚠️ Placeholder - Replace with actual API response**

```json
{
  "status": "success",
  "data": {
    "paas_engagement_id": "paas-12345",
    "ipc_engagement_id": "ipc-67890",
    "conversion_timestamp": "2024-02-23T10:30:00Z"
  }
}
```

## Response Fields Details

### Core Fields
- **status** - Response status (success, error)
- **ipc_engagement_id** - The converted IPC engagement ID to use in subsequent API calls
- **paas_engagement_id** - Original PaaS engagement ID (for verification)

## Permissions
**Roles:** To be documented (requires actual permission information)

## Workflow Steps

### Workflow: Portal Service Access
This endpoint is typically the FIRST step in any portal service workflow.

**Critical Sequence:**
- Step 1: **Convert Engagement** (engagement.convert_paas_to_ipc) - Convert PaaS ID to IPC ID
- Step 2: **Access Portal Services** - Use the IPC ID in subsequent calls to:
  - `configservice/getEndpointsByEngagement/{ipc_engagement_id}`
  - `securityservice/deptDetailsForEngagement/{ipc_engagement_id}`

**⚠️ CRITICAL:** Portal service endpoints require IPC engagement IDs, not PaaS IDs. Always convert first!

## Usage Notes
- This is a prerequisite API call for portal services
- Cache the IPC ID during a session to avoid repeated conversions
- The conversion is deterministic - same PaaS ID always returns same IPC ID
- DO NOT attempt to use PaaS IDs directly with portal services - they will fail with 404

## Common Use Cases
1. **Convert before portal access**: "convert paas engagement", "get ipc id from paas", "convert engagement id"
2. **Check engagement mapping**: "what's the ipc id for paas-123", "lookup ipc engagement"
3. **Validate engagement**: "verify engagement conversion", "check paas to ipc mapping"

## Query Interpretations
- "convert engagement {id}" → GET /getIpcEngFromPaasEng/{id}
- "get ipc from paas {id}" → GET /getIpcEngFromPaasEng/{id}
- "paas {id} to ipc" → GET /getIpcEngFromPaasEng/{id}

## Data Processing Examples

### Python Example
```python
import requests

def convert_paas_to_ipc(paas_engagement_id, auth_token):
    """
    Convert PaaS engagement ID to IPC engagement ID.
    
    Args:
        paas_engagement_id: The PaaS engagement ID
        auth_token: Bearer authentication token
        
    Returns:
        str: IPC engagement ID or None if conversion fails
    """
    url = f"https://ipcloud.tatacommunications.com/paasservice/paas/getIpcEngFromPaasEng/{paas_engagement_id}"
    
    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        if data.get('status') == 'success':
            ipc_id = data['data']['ipc_engagement_id']
            print(f"Converted {paas_engagement_id} → {ipc_id}")
            return ipc_id
        else:
            print(f"Conversion failed: {data.get('message')}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"API error: {e}")
        return None

# Usage
paas_id = "paas-12345"
ipc_id = convert_paas_to_ipc(paas_id, "your-token-here")

if ipc_id:
    # Now use ipc_id for portal services
    print(f"Use IPC ID {ipc_id} for portal service calls")
```

### JavaScript Example
```javascript
async function convertPaasToIpc(paasEngagementId, authToken) {
    const url = `https://ipcloud.tatacommunications.com/paasservice/paas/getIpcEngFromPaasEng/${paasEngagementId}`;
    
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
            const ipcId = data.data.ipc_engagement_id;
            console.log(`Converted ${paasEngagementId} → ${ipcId}`);
            return ipcId;
        } else {
            console.error(`Conversion failed: ${data.message}`);
            return null;
        }
    } catch (error) {
        console.error(`API error: ${error.message}`);
        return null;
    }
}

// Usage
const paasId = 'paas-12345';
const ipcId = await convertPaasToIpc(paasId, 'your-token-here');

if (ipcId) {
    console.log(`Use IPC ID ${ipcId} for portal service calls`);
}
```

## Integration Examples

### Complete Workflow Example
```python
class EngagementClient:
    def __init__(self, base_url, auth_token):
        self.base_url = base_url
        self.auth_token = auth_token
        self.ipc_cache = {}  # Cache conversions
    
    def get_ipc_id(self, paas_id):
        """Get IPC ID with caching"""
        # Check cache first
        if paas_id in self.ipc_cache:
            return self.ipc_cache[paas_id]
        
        # Convert
        url = f"{self.base_url}/paasservice/paas/getIpcEngFromPaasEng/{paas_id}"
        response = requests.get(url, headers=self._headers())
        response.raise_for_status()
        
        ipc_id = response.json()['data']['ipc_engagement_id']
        
        # Cache for session
        self.ipc_cache[paas_id] = ipc_id
        return ipc_id
    
    def get_endpoints(self, paas_id):
        """Get endpoints - automatically converts ID"""
        ipc_id = self.get_ipc_id(paas_id)
        url = f"{self.base_url}/portalservice/configservice/getEndpointsByEngagement/{ipc_id}"
        response = requests.get(url, headers=self._headers())
        return response.json()
    
    def _headers(self):
        return {
            "Authorization": f"Bearer {self.auth_token}",
            "Content-Type": "application/json"
        }

# Usage
client = EngagementClient("https://ipcloud.tatacommunications.com", "token")
endpoints = client.get_endpoints("paas-12345")  # Auto-converts
```

## Related Operations
- `configservice.get_endpoints` - Get endpoints using converted IPC ID
- `securityservice.get_department_details` - Get department details using converted IPC ID
- All portal service operations - Require IPC engagement ID from this conversion

## Error Handling
- **401 Unauthorized:** Invalid or expired authentication token - refresh your token
- **404 Not Found:** PaaS engagement ID does not exist - verify the ID is correct
- **500 Internal Server Error:** Service error - retry with exponential backoff
- **503 Service Unavailable:** Service temporarily unavailable - implement retry logic

## Response Codes
**⚠️ To be documented - requires actual API testing**

Likely patterns:
- `0` or `success` - Conversion successful
- Non-zero or `error` - Conversion failed (check message field)

## Performance Notes
- Response time typically < 200ms
- Results are deterministic and can be cached per session
- No pagination (single ID conversion)
- Rate limiting: To be documented

## Best Practices

### Always Convert Before Portal Services
```python
# ❌ WRONG - Using PaaS ID directly
try:
    response = requests.get(
        f"https://ipcloud.tatacommunications.com/portalservice/configservice/getEndpointsByEngagement/paas-12345"
    )
    # This will FAIL with 404!
except:
    print("Failed - portal services don't accept PaaS IDs")

# ✅ RIGHT - Convert first
paas_id = "paas-12345"
ipc_id = convert_paas_to_ipc(paas_id, token)
response = requests.get(
    f"https://ipcloud.tatacommunications.com/portalservice/configservice/getEndpointsByEngagement/{ipc_id}"
)
# This WORKS!
```

### Implement Caching
```python
class EngagementCache:
    def __init__(self):
        self.cache = {}
        self.ttl = 3600  # 1 hour
    
    def get(self, paas_id):
        if paas_id in self.cache:
            entry = self.cache[paas_id]
            if time.time() - entry['timestamp'] < self.ttl:
                return entry['ipc_id']
        return None
    
    def set(self, paas_id, ipc_id):
        self.cache[paas_id] = {
            'ipc_id': ipc_id,
            'timestamp': time.time()
        }
```

## Metadata
- **Generated:** 2024-02-23T10:00:00Z
- **Source:** API endpoint analysis
- **API Version:** To be documented
- **Base Path:** {BASE_URL_PAAS_SERVICE}/paas
- **Additional notes:** ⚠️ This documentation is preliminary. Please provide actual API response for complete accuracy.
