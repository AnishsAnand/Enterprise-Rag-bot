# API Specification: customer - list

**Resource:** customer
**Operation:** list
**Aliases:** customers, list customers, get customers, masterdata customers, itsm customers

## Endpoint
- **Method:** GET
- **URL:** https://ipcloud.tatacommunications.com/itsmv1/api/ui/v1/masterdata/customers
- **Auth:** Bearer token (from Keycloak)
- **Description:** List all customers from the master data system. This is typically used for customer selection, filtering, and reporting across the IPCloud platform

## Required Parameters
None

## Optional Parameters
- `active` - Filter by active status (true/false)
- `search` - Search by customer name or code
- `page` - Page number for pagination
- `limit` - Number of results per page

## Response Mapping
- `status`: status
- `message`: message
- `customers`: data.customers
- `customer_ids`: data.customers[*].id
- `customer_names`: data.customers[*].name
- `customer_codes`: data.customers[*].code
- `customer_types`: data.customers[*].type
- `total_count`: data.totalCount

## Response Example
```json
{
  "status": "success",
  "data": {
    "totalCount": 150,
    "currentPage": 1,
    "totalPages": 6,
    "pageSize": 25,
    "customers": [
      {
        "id": "cust-001",
        "code": "ACME-001",
        "name": "ACME Corporation",
        "displayName": "ACME Corp",
        "type": "enterprise",
        "status": "active",
        "createdAt": "2023-01-15T10:30:00Z",
        "updatedAt": "2025-02-10T14:20:00Z",
        "tier": "platinum",
        "vertical": "technology",
        "region": "india",
        "contact": {
          "primaryContact": {
            "name": "John Doe",
            "email": "john.doe@acme.com",
            "phone": "+91-9876543210",
            "role": "CTO"
          },
          "billingContact": {
            "name": "Jane Smith",
            "email": "billing@acme.com",
            "phone": "+91-9876543211",
            "role": "Finance Manager"
          }
        },
        "address": {
          "line1": "Tech Park, 5th Floor",
          "line2": "Whitefield",
          "city": "Bengaluru",
          "state": "Karnataka",
          "country": "India",
          "postalCode": "560066"
        },
        "financials": {
          "currency": "USD",
          "paymentTerms": "NET30",
          "creditLimit": 100000,
          "currentBalance": 15000
        },
        "resources": {
          "engagements": 5,
          "clusters": 25,
          "volumes": 120,
          "kafkaClusters": 8,
          "databases": 35
        },
        "metadata": {
          "industry": "Software",
          "companySize": "1000-5000",
          "contractStartDate": "2023-01-15",
          "contractEndDate": "2026-01-14",
          "accountManager": "Sarah Johnson"
        }
      },
      {
        "id": "cust-002",
        "code": "FINTECH-001",
        "name": "FinTech Solutions Pvt Ltd",
        "displayName": "FinTech Solutions",
        "type": "enterprise",
        "status": "active",
        "createdAt": "2023-03-20T09:15:00Z",
        "tier": "gold",
        "vertical": "financial-services",
        "region": "india",
        "contact": {
          "primaryContact": {
            "name": "Rajesh Kumar",
            "email": "rajesh.kumar@fintech.com",
            "phone": "+91-9876543220"
          }
        },
        "address": {
          "city": "Mumbai",
          "state": "Maharashtra",
          "country": "India"
        },
        "resources": {
          "engagements": 3,
          "clusters": 12,
          "kafkaClusters": 4,
          "databases": 18
        }
      },
      {
        "id": "cust-003",
        "code": "RETAIL-001",
        "name": "Retail Giant India",
        "displayName": "Retail Giant",
        "type": "enterprise",
        "status": "active",
        "createdAt": "2023-06-10T14:45:00Z",
        "tier": "silver",
        "vertical": "retail",
        "region": "india",
        "resources": {
          "engagements": 2,
          "clusters": 8,
          "kafkaClusters": 2,
          "databases": 10
        }
      },
      {
        "id": "cust-004",
        "code": "STARTUP-001",
        "name": "TechStartup Innovations",
        "displayName": "TechStartup",
        "type": "smb",
        "status": "active",
        "createdAt": "2024-01-05T11:00:00Z",
        "tier": "bronze",
        "vertical": "technology",
        "region": "india",
        "resources": {
          "engagements": 1,
          "clusters": 3,
          "kafkaClusters": 1,
          "databases": 4
        }
      },
      {
        "id": "cust-005",
        "code": "HEALTHCARE-001",
        "name": "HealthCare Systems Ltd",
        "displayName": "HealthCare Systems",
        "type": "enterprise",
        "status": "active",
        "createdAt": "2023-09-15T10:20:00Z",
        "tier": "gold",
        "vertical": "healthcare",
        "region": "india",
        "resources": {
          "engagements": 4,
          "clusters": 18,
          "kafkaClusters": 5,
          "databases": 22
        },
        "metadata": {
          "complianceCertifications": ["HIPAA", "ISO27001"],
          "dataResidency": "india-only"
        }
      }
    ],
    "summary": {
      "totalCustomers": 150,
      "activeCustomers": 145,
      "inactiveCustomers": 5,
      "byTier": {
        "platinum": 15,
        "gold": 45,
        "silver": 60,
        "bronze": 30
      },
      "byType": {
        "enterprise": 120,
        "smb": 30
      },
      "byVertical": {
        "technology": 50,
        "financial-services": 30,
        "retail": 25,
        "healthcare": 20,
        "manufacturing": 15,
        "other": 10
      }
    }
  },
  "message": "Customers retrieved successfully",
  "responseCode": 0
}
```

## Response Fields Details

### Customer Fields
- **id** - Unique customer identifier
- **code** - Customer code/reference
- **name** - Full legal customer name
- **displayName** - Short display name
- **type** - Customer type (enterprise, smb, partner)
- **status** - Active status (active, inactive, suspended)
- **tier** - Service tier (platinum, gold, silver, bronze)
- **vertical** - Industry vertical
- **region** - Primary region
- **contact** - Contact information
- **address** - Physical address
- **financials** - Financial details
- **resources** - Resource usage summary
- **metadata** - Additional customer metadata

### Customer Types
- `enterprise` - Large enterprise customers
- `smb` - Small and medium businesses
- `partner` - Technology or reseller partners
- `internal` - Internal/test accounts

### Service Tiers
- `platinum` - Premium tier with highest SLA
- `gold` - High tier with enhanced support
- `silver` - Standard tier
- `bronze` - Basic tier

### Industry Verticals
- `technology` - Software/IT companies
- `financial-services` - Banks, FinTech
- `retail` - Retail and e-commerce
- `healthcare` - Healthcare providers
- `manufacturing` - Manufacturing companies
- `telecom` - Telecommunications
- `government` - Government agencies
- `education` - Educational institutions

## Permissions
Roles: admin, account_manager, billing, viewer

## Workflow Steps
### Workflow: list_customers
List all customers
- Step 1: authenticate (auth.validate_token)
- Step 2: check_permissions (user.check_permission) (permission: customer.list)
- Step 3: list_customers (customer.list)

## Usage Notes
- Returns only customers user has permission to view
- Results paginated with default 25 per page
- Search performs partial match on name and code
- Inactive customers excluded by default
- Resource counts updated daily
- Financial data restricted based on role

## Common Use Cases
1. **Customer selection**: "Show customer dropdown"
2. **Customer search**: "Find customer by name"
3. **Resource allocation**: "Which customers have most resources?"
4. **Billing reports**: "List customers by tier"
5. **Account management**: "Show my assigned customers"
6. **Compliance filtering**: "Healthcare customers only"

## Data Processing Examples

### Search Customers
```python
def search_customers(query):
    response = list_customers()
    customers = response['data']['customers']
    
    query_lower = query.lower()
    matches = [
        c for c in customers
        if query_lower in c['name'].lower() or
           query_lower in c.get('code', '').lower()
    ]
    
    return matches
```

### Group by Vertical
```python
from collections import defaultdict

response = list_customers()
by_vertical = defaultdict(list)

for customer in response['data']['customers']:
    vertical = customer.get('vertical', 'other')
    by_vertical[vertical].append(customer)

for vertical, customers in by_vertical.items():
    print(f"{vertical}: {len(customers)} customers")
```

### Calculate Total Resources
```python
response = list_customers()
total_resources = {
    'clusters': 0,
    'kafkaClusters': 0,
    'databases': 0
}

for customer in response['data']['customers']:
    resources = customer.get('resources', {})
    total_resources['clusters'] += resources.get('clusters', 0)
    total_resources['kafkaClusters'] += resources.get('kafkaClusters', 0)
    total_resources['databases'] += resources.get('databases', 0)

print(f"Total across all customers:")
print(f"  Clusters: {total_resources['clusters']}")
print(f"  Kafka: {total_resources['kafkaClusters']}")
print(f"  Databases: {total_resources['databases']}")
```

### Find High-Value Customers
```python
response = list_customers()

high_value = [
    c for c in response['data']['customers']
    if c['tier'] in ['platinum', 'gold'] and
       c['resources'].get('clusters', 0) > 10
]

print(f"High-value customers ({len(high_value)}):")
for customer in high_value:
    print(f"  {customer['name']} - {customer['tier']}")
    print(f"    Clusters: {customer['resources']['clusters']}")
```

## Customer Analytics

### Tier Distribution
```python
response = list_customers()
summary = response['data']['summary']

tier_dist = summary['byTier']
total = sum(tier_dist.values())

print("Customer Distribution by Tier:")
for tier, count in tier_dist.items():
    percentage = (count / total) * 100
    print(f"  {tier}: {count} ({percentage:.1f}%)")
```

### Kafka Adoption Analysis
```python
response = list_customers()
customers = response['data']['customers']

with_kafka = [c for c in customers if c['resources'].get('kafkaClusters', 0) > 0]
kafka_adoption = (len(with_kafka) / len(customers)) * 100

print(f"Kafka Adoption: {kafka_adoption:.1f}%")
print(f"Customers with Kafka: {len(with_kafka)}/{len(customers)}")

# By vertical
from collections import defaultdict
kafka_by_vertical = defaultdict(lambda: {'total': 0, 'with_kafka': 0})

for customer in customers:
    vertical = customer.get('vertical', 'other')
    kafka_by_vertical[vertical]['total'] += 1
    if customer['resources'].get('kafkaClusters', 0) > 0:
        kafka_by_vertical[vertical]['with_kafka'] += 1

print("\nKafka Adoption by Vertical:")
for vertical, stats in kafka_by_vertical.items():
    adoption = (stats['with_kafka'] / stats['total']) * 100
    print(f"  {vertical}: {adoption:.1f}%")
```

## Customer Contact Management

### Get Primary Contacts
```python
response = list_customers()

contacts = []
for customer in response['data']['customers']:
    if 'contact' in customer and 'primaryContact' in customer['contact']:
        primary = customer['contact']['primaryContact']
        contacts.append({
            'customer': customer['name'],
            'name': primary['name'],
            'email': primary['email'],
            'phone': primary.get('phone', 'N/A')
        })

for contact in contacts:
    print(f"{contact['customer']}: {contact['name']} ({contact['email']})")
```

## Compliance and Data Residency

### Filter by Compliance
```python
def get_compliant_customers(certification):
    response = list_customers()
    
    compliant = []
    for customer in response['data']['customers']:
        certs = customer.get('metadata', {}).get('complianceCertifications', [])
        if certification in certs:
            compliant.append(customer)
    
    return compliant

# Example: HIPAA-compliant customers
hipaa_customers = get_compliant_customers('HIPAA')
print(f"HIPAA-compliant customers: {len(hipaa_customers)}")
```

### Data Residency Check
```python
def check_data_residency(customer_id):
    response = list_customers()
    
    customer = next(
        (c for c in response['data']['customers'] if c['id'] == customer_id),
        None
    )
    
    if customer:
        residency = customer.get('metadata', {}).get('dataResidency', 'global')
        return residency
    
    return None
```

## Integration Examples

### Build Customer Dropdown
```javascript
// React component
function CustomerSelector() {
  const [customers, setCustomers] = useState([]);
  
  useEffect(() => {
    fetch('/itsmv1/api/ui/v1/masterdata/customers')
      .then(res => res.json())
      .then(data => setCustomers(data.data.customers));
  }, []);
  
  return (
    <select>
      {customers.map(customer => (
        <option key={customer.id} value={customer.id}>
          {customer.displayName} ({customer.code})
        </option>
      ))}
    </select>
  );
}
```

### Customer Dashboard
```python
def generate_customer_dashboard(customer_id):
    response = list_customers()
    
    customer = next(
        (c for c in response['data']['customers'] if c['id'] == customer_id),
        None
    )
    
    if not customer:
        return None
    
    return {
        'name': customer['name'],
        'tier': customer['tier'],
        'status': customer['status'],
        'resources': customer['resources'],
        'contact': customer['contact']['primaryContact'],
        'financials': customer.get('financials', {})
    }
```

## Related Operations
- `customer.get` - Get single customer details
- `customer.create` - Create new customer
- `customer.update` - Update customer information
- `customer.deactivate` - Deactivate customer
- `customer.resources` - Get detailed resource usage
- `customer.billing` - Get billing information

## Error Handling
- **401 Unauthorized:** Invalid or expired authentication token
- **403 Forbidden:** User does not have customer.list permission
- **500 Internal Server Error:** Master data service unavailable

## Response Codes
- `0` - Success
- `1` - No customers found
- `2` - Access denied
- `3` - Invalid filter parameters

## Pagination

### Request Parameters
```
GET /itsmv1/api/ui/v1/masterdata/customers?page=2&limit=50
```

### Response Pagination Info
```json
{
  "totalCount": 150,
  "currentPage": 2,
  "totalPages": 3,
  "pageSize": 50
}
```

## Search Examples

### Search by Name
```
GET /itsmv1/api/ui/v1/masterdata/customers?search=ACME
```

### Filter Active Only
```
GET /itsmv1/api/ui/v1/masterdata/customers?active=true
```

### Combined Filters
```
GET /itsmv1/api/ui/v1/masterdata/customers?active=true&search=Tech&page=1&limit=10
```

## Performance Notes
- Response time typically < 500ms
- Customer data cached for 15 minutes
- Resource counts updated daily at midnight
- Large result sets automatically paginated
- Use search parameter to reduce payload size

## Best Practices

### Cache Customer List
```python
from datetime import datetime, timedelta

customer_cache = {'data': None, 'expires': None}

def get_customers_cached(ttl_minutes=15):
    now = datetime.now()
    
    if (customer_cache['data'] is None or 
        customer_cache['expires'] is None or
        now > customer_cache['expires']):
        
        customer_cache['data'] = list_customers()
        customer_cache['expires'] = now + timedelta(minutes=ttl_minutes)
    
    return customer_cache['data']
```

### Efficient Searching
```python
# Client-side filtering for better UX
def filter_customers_client_side(customers, filters):
    results = customers
    
    if 'search' in filters:
        query = filters['search'].lower()
        results = [c for c in results 
                  if query in c['name'].lower() or 
                     query in c.get('code', '').lower()]
    
    if 'tier' in filters:
        results = [c for c in results if c['tier'] == filters['tier']]
    
    if 'vertical' in filters:
        results = [c for c in results if c['vertical'] == filters['vertical']]
    
    return results
```

## Metadata
- **Generated:** 2025-02-13T13:30:00Z
- **Source:** Dynamic API Spec Generator
- **API Version:** v1
- **Base Path:** /itsmv1/api/ui/v1/masterdata
- **Master Data:** Customer repository for IPCloud platform