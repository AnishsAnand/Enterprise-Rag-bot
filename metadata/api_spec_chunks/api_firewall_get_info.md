# API Specification: firewall - get_info

**Resource:** firewall
**Operation:** get_info
**Aliases:** firewall, firewall rules, security groups, firewall configuration, network security, firewall info, get firewall, show firewall, firewall details, security rules, access control lists, ACL, network firewall

## Endpoint
- **Method:** GET
- **URL:** https://ipcloud.tatacommunications.com/cloud/console/network/resources/firewallInfo.json
- **Auth:** Bearer token (from Keycloak)
- **Description:** Get comprehensive firewall configuration and rules for network resources including ingress/egress rules, allowed ports, protocols, source/destination IP ranges, and security policy status

## Required Parameters
None (returns firewall information for authenticated user's accessible resources)

## Optional Parameters
- `firewall_id` - Specific firewall ID to retrieve (query parameter)
- `zone_id` - Filter by zone (query parameter)
- `status` - Filter by status: active, inactive, pending (query parameter)
- `include_rules` - Include detailed rules (default: true)

## Response Mapping
- `status`: status
- `message`: message
- `responseCode`: responseCode
- `firewalls`: data.firewalls
- `firewall_id`: data.firewalls[*].firewallId
- `firewall_name`: data.firewalls[*].firewallName
- `firewall_type`: data.firewalls[*].type
- `zone_id`: data.firewalls[*].zoneId
- `zone_name`: data.firewalls[*].zoneName
- `status`: data.firewalls[*].status
- `created_time`: data.firewalls[*].createdTime
- `ingress_rules`: data.firewalls[*].ingressRules[*]
- `egress_rules`: data.firewalls[*].egressRules[*]
- `rule_id`: data.firewalls[*].ingressRules[*].ruleId
- `protocol`: data.firewalls[*].ingressRules[*].protocol
- `port_range`: data.firewalls[*].ingressRules[*].portRange
- `source_cidr`: data.firewalls[*].ingressRules[*].sourceCidr
- `destination_cidr`: data.firewalls[*].egressRules[*].destinationCidr
- `action`: data.firewalls[*].ingressRules[*].action
- `applied_to_vms`: data.firewalls[*].appliedToVMs[*]
- `applied_to_subnets`: data.firewalls[*].appliedToSubnets[*]
- `default_policy`: data.firewalls[*].defaultPolicy
- `logging_enabled`: data.firewalls[*].loggingEnabled

## Response Example
```json
{
  "status": "success",
  "data": {
    "totalCount": 5,
    "firewalls": [
      {
        "firewallId": "FW001",
        "firewallName": "production-web-firewall",
        "type": "stateful",
        "zoneId": "Z001",
        "zoneName": "Production",
        "status": "active",
        "createdTime": 1752925643000,
        "createdBy": "admin@domain.com",
        "updatedTime": 1765190613000,
        "defaultPolicy": "deny",
        "loggingEnabled": true,
        "description": "Firewall for production web servers",
        "ingressRules": [
          {
            "ruleId": "RULE001",
            "priority": 100,
            "protocol": "tcp",
            "portRange": "80,443",
            "sourceCidr": "0.0.0.0/0",
            "action": "allow",
            "description": "Allow HTTP and HTTPS from anywhere",
            "enabled": true,
            "logMatches": true
          },
          {
            "ruleId": "RULE002",
            "priority": 200,
            "protocol": "tcp",
            "portRange": "22",
            "sourceCidr": "10.0.0.0/8",
            "action": "allow",
            "description": "Allow SSH from internal network",
            "enabled": true,
            "logMatches": true
          },
          {
            "ruleId": "RULE003",
            "priority": 300,
            "protocol": "icmp",
            "portRange": "*",
            "sourceCidr": "10.0.0.0/8",
            "action": "allow",
            "description": "Allow ping from internal network",
            "enabled": true,
            "logMatches": false
          }
        ],
        "egressRules": [
          {
            "ruleId": "ERULE001",
            "priority": 100,
            "protocol": "tcp",
            "portRange": "443",
            "destinationCidr": "0.0.0.0/0",
            "action": "allow",
            "description": "Allow HTTPS to anywhere",
            "enabled": true,
            "logMatches": false
          },
          {
            "ruleId": "ERULE002",
            "priority": 200,
            "protocol": "tcp",
            "portRange": "3306",
            "destinationCidr": "10.1.0.0/24",
            "action": "allow",
            "description": "Allow MySQL to database subnet",
            "enabled": true,
            "logMatches": true
          }
        ],
        "appliedToVMs": [
          {
            "vmId": "VM001",
            "vmName": "web-server-1",
            "ipAddress": "10.0.1.10",
            "status": "active"
          },
          {
            "vmId": "VM002",
            "vmName": "web-server-2",
            "ipAddress": "10.0.1.11",
            "status": "active"
          }
        ],
        "appliedToSubnets": [
          {
            "subnetId": "SUBNET001",
            "subnetName": "web-subnet",
            "cidr": "10.0.1.0/24"
          }
        ],
        "statistics": {
          "totalIngressRules": 3,
          "totalEgressRules": 2,
          "activeRules": 5,
          "disabledRules": 0,
          "lastMatchTime": 1765190613000,
          "totalMatches24h": 125678
        }
      },
      {
        "firewallId": "FW002",
        "firewallName": "database-firewall",
        "type": "stateful",
        "zoneId": "Z001",
        "zoneName": "Production",
        "status": "active",
        "createdTime": 1753025643000,
        "createdBy": "admin@domain.com",
        "updatedTime": 1765190613000,
        "defaultPolicy": "deny",
        "loggingEnabled": true,
        "description": "Firewall for database servers",
        "ingressRules": [
          {
            "ruleId": "RULE004",
            "priority": 100,
            "protocol": "tcp",
            "portRange": "3306",
            "sourceCidr": "10.0.1.0/24",
            "action": "allow",
            "description": "Allow MySQL from web subnet",
            "enabled": true,
            "logMatches": true
          },
          {
            "ruleId": "RULE005",
            "priority": 200,
            "protocol": "tcp",
            "portRange": "22",
            "sourceCidr": "10.0.0.0/8",
            "action": "allow",
            "description": "Allow SSH from internal network",
            "enabled": true,
            "logMatches": true
          }
        ],
        "egressRules": [
          {
            "ruleId": "ERULE003",
            "priority": 100,
            "protocol": "tcp",
            "portRange": "443",
            "destinationCidr": "0.0.0.0/0",
            "action": "allow",
            "description": "Allow HTTPS for updates",
            "enabled": true,
            "logMatches": false
          }
        ],
        "appliedToVMs": [
          {
            "vmId": "VM003",
            "vmName": "db-primary",
            "ipAddress": "10.1.0.10",
            "status": "active"
          },
          {
            "vmId": "VM004",
            "vmName": "db-replica",
            "ipAddress": "10.1.0.11",
            "status": "active"
          }
        ],
        "appliedToSubnets": [
          {
            "subnetId": "SUBNET002",
            "subnetName": "database-subnet",
            "cidr": "10.1.0.0/24"
          }
        ],
        "statistics": {
          "totalIngressRules": 2,
          "totalEgressRules": 1,
          "activeRules": 3,
          "disabledRules": 0,
          "lastMatchTime": 1765190613000,
          "totalMatches24h": 45678
        }
      }
    ]
  },
  "message": "success",
  "responseCode": 0
}
```

## Response Fields Details

### Firewall Object Fields
- **firewallId** - Unique identifier for the firewall (string)
- **firewallName** - User-defined name for the firewall (string)
- **type** - Firewall type: stateful, stateless (string)
- **zoneId** - Zone identifier where firewall is deployed (string)
- **zoneName** - Human-readable zone name (string)
- **status** - Current status: active, inactive, pending, error (string)
- **createdTime** - Unix timestamp in milliseconds when created (number)
- **createdBy** - Email of user who created the firewall (string)
- **updatedTime** - Unix timestamp of last update (number)
- **defaultPolicy** - Default action for unmatched traffic: allow, deny (string)
- **loggingEnabled** - Whether rule matching is logged (boolean)
- **description** - Description of firewall purpose (string)

### Ingress Rule Fields
- **ruleId** - Unique identifier for the rule (string)
- **priority** - Rule priority (lower number = higher priority) (number)
- **protocol** - Network protocol: tcp, udp, icmp, all (string)
- **portRange** - Port or port range: "80", "443", "8000-9000", "*" (string)
- **sourceCidr** - Source IP range in CIDR notation (string)
- **action** - Action to take: allow, deny (string)
- **description** - Rule description (string)
- **enabled** - Whether rule is active (boolean)
- **logMatches** - Whether to log when rule matches (boolean)

### Egress Rule Fields
- **ruleId** - Unique identifier for the rule (string)
- **priority** - Rule priority (lower number = higher priority) (number)
- **protocol** - Network protocol: tcp, udp, icmp, all (string)
- **portRange** - Port or port range: "80", "443", "8000-9000", "*" (string)
- **destinationCidr** - Destination IP range in CIDR notation (string)
- **action** - Action to take: allow, deny (string)
- **description** - Rule description (string)
- **enabled** - Whether rule is active (boolean)
- **logMatches** - Whether to log when rule matches (boolean)

### Applied Resources
- **appliedToVMs** - Array of VMs where firewall is applied
  - **vmId** - Virtual machine identifier (string)
  - **vmName** - VM name (string)
  - **ipAddress** - VM IP address (string)
  - **status** - VM status (string)
- **appliedToSubnets** - Array of subnets where firewall is applied
  - **subnetId** - Subnet identifier (string)
  - **subnetName** - Subnet name (string)
  - **cidr** - Subnet CIDR range (string)

### Statistics
- **totalIngressRules** - Count of ingress rules (number)
- **totalEgressRules** - Count of egress rules (number)
- **activeRules** - Count of enabled rules (number)
- **disabledRules** - Count of disabled rules (number)
- **lastMatchTime** - Timestamp of last rule match (number)
- **totalMatches24h** - Rule matches in last 24 hours (number)

## Permissions
Roles: admin, developer, viewer

## Workflow Steps
### Workflow: get_firewall_info
Get firewall configuration and rules
- Step 1: authenticate_user (user.authenticate)
- Step 2: get_user_zones (zone.list_by_user)
- Step 3: get_firewall_info (firewall.get_info)
- Step 4: get_applied_resources (firewall.get_applied_resources)
- Step 5: get_statistics (firewall.get_statistics)
- Step 6: format_response (response.format)

## Common Use Cases

1. **List all firewalls**: "Show me all firewalls"
2. **Get firewall details**: "What are the firewall rules for production?"
3. **Check security configuration**: "Show firewall configuration for web servers"
4. **Audit access rules**: "What ports are open on the database firewall?"
5. **Review ingress rules**: "Show me all ingress rules allowing SSH"
6. **Check egress restrictions**: "What outbound traffic is allowed?"
7. **Find firewall by VM**: "Which firewall protects web-server-1?"
8. **Review security policies**: "Show default deny policies"
9. **Audit logging configuration**: "Which firewalls have logging enabled?"
10. **Monitor firewall usage**: "Show firewall statistics and match counts"

## Query Interpretations
- "list firewalls" → GET /firewallInfo.json
- "firewall rules" → GET /firewallInfo.json?include_rules=true
- "production firewall" → GET /firewallInfo.json?zone_id=production
- "active firewalls" → GET /firewallInfo.json?status=active
- "firewall for VM xyz" → Filter response by appliedToVMs
- "ingress rules" → Extract data.firewalls[*].ingressRules
- "egress rules" → Extract data.firewalls[*].egressRules
- "security groups" → Synonym for firewalls

## Data Processing Examples

### Extract All Open Ports
```python
def get_all_open_ports(firewall_data):
    """Extract all allowed ports from firewall rules."""
    open_ports = {
        'ingress': {},
        'egress': {}
    }
    
    for firewall in firewall_data['data']['firewalls']:
        fw_name = firewall['firewallName']
        
        # Process ingress rules
        for rule in firewall.get('ingressRules', []):
            if rule['action'] == 'allow' and rule['enabled']:
                port = rule['portRange']
                protocol = rule['protocol']
                source = rule['sourceCidr']
                
                if fw_name not in open_ports['ingress']:
                    open_ports['ingress'][fw_name] = []
                
                open_ports['ingress'][fw_name].append({
                    'port': port,
                    'protocol': protocol,
                    'from': source,
                    'description': rule.get('description', '')
                })
        
        # Process egress rules
        for rule in firewall.get('egressRules', []):
            if rule['action'] == 'allow' and rule['enabled']:
                port = rule['portRange']
                protocol = rule['protocol']
                dest = rule['destinationCidr']
                
                if fw_name not in open_ports['egress']:
                    open_ports['egress'][fw_name] = []
                
                open_ports['egress'][fw_name].append({
                    'port': port,
                    'protocol': protocol,
                    'to': dest,
                    'description': rule.get('description', '')
                })
    
    return open_ports
```

### Check Firewall Compliance
```python
def check_firewall_compliance(firewall_data, compliance_rules):
    """Check if firewalls meet security compliance requirements."""
    violations = []
    
    for firewall in firewall_data['data']['firewalls']:
        fw_id = firewall['firewallId']
        fw_name = firewall['firewallName']
        
        # Check 1: Logging should be enabled
        if not firewall.get('loggingEnabled', False):
            violations.append({
                'firewall': fw_name,
                'severity': 'high',
                'issue': 'Logging is not enabled',
                'recommendation': 'Enable logging for audit trail'
            })
        
        # Check 2: Default policy should be deny
        if firewall.get('defaultPolicy') != 'deny':
            violations.append({
                'firewall': fw_name,
                'severity': 'critical',
                'issue': 'Default policy is not deny',
                'recommendation': 'Change default policy to deny'
            })
        
        # Check 3: No overly permissive rules
        for rule in firewall.get('ingressRules', []):
            if (rule['sourceCidr'] == '0.0.0.0/0' and 
                rule['portRange'] == '*' and 
                rule['action'] == 'allow'):
                violations.append({
                    'firewall': fw_name,
                    'severity': 'critical',
                    'issue': f'Rule {rule["ruleId"]} allows all traffic from anywhere',
                    'recommendation': 'Restrict source CIDR and port range'
                })
            
            # Check for SSH open to internet
            if (rule['sourceCidr'] == '0.0.0.0/0' and 
                '22' in rule['portRange'] and 
                rule['action'] == 'allow'):
                violations.append({
                    'firewall': fw_name,
                    'severity': 'high',
                    'issue': f'Rule {rule["ruleId"]} allows SSH from internet',
                    'recommendation': 'Restrict SSH to internal networks only'
                })
    
    return violations
```

### Find VMs Protected by Firewall
```python
def find_protected_vms(firewall_data, firewall_name):
    """Find all VMs protected by a specific firewall."""
    for firewall in firewall_data['data']['firewalls']:
        if firewall['firewallName'] == firewall_name:
            return {
                'firewall_id': firewall['firewallId'],
                'firewall_name': firewall['firewallName'],
                'vms': [
                    {
                        'id': vm['vmId'],
                        'name': vm['vmName'],
                        'ip': vm['ipAddress'],
                        'status': vm['status']
                    }
                    for vm in firewall.get('appliedToVMs', [])
                ],
                'subnets': [
                    {
                        'id': subnet['subnetId'],
                        'name': subnet['subnetName'],
                        'cidr': subnet['cidr']
                    }
                    for subnet in firewall.get('appliedToSubnets', [])
                ]
            }
    return None
```

### Generate Firewall Summary Report
```python
def generate_firewall_summary(firewall_data):
    """Generate a summary report of all firewalls."""
    summary = {
        'total_firewalls': len(firewall_data['data']['firewalls']),
        'active_firewalls': 0,
        'total_ingress_rules': 0,
        'total_egress_rules': 0,
        'total_protected_vms': 0,
        'compliance_issues': [],
        'firewalls': []
    }
    
    for firewall in firewall_data['data']['firewalls']:
        if firewall['status'] == 'active':
            summary['active_firewalls'] += 1
        
        stats = firewall.get('statistics', {})
        summary['total_ingress_rules'] += stats.get('totalIngressRules', 0)
        summary['total_egress_rules'] += stats.get('totalEgressRules', 0)
        summary['total_protected_vms'] += len(firewall.get('appliedToVMs', []))
        
        # Check for compliance issues
        if not firewall.get('loggingEnabled'):
            summary['compliance_issues'].append(
                f"{firewall['firewallName']}: Logging disabled"
            )
        
        if firewall.get('defaultPolicy') != 'deny':
            summary['compliance_issues'].append(
                f"{firewall['firewallName']}: Default policy not deny"
            )
        
        summary['firewalls'].append({
            'name': firewall['firewallName'],
            'status': firewall['status'],
            'zone': firewall['zoneName'],
            'ingress_rules': stats.get('totalIngressRules', 0),
            'egress_rules': stats.get('totalEgressRules', 0),
            'protected_vms': len(firewall.get('appliedToVMs', [])),
            'matches_24h': stats.get('totalMatches24h', 0)
        })
    
    return summary
```

## Integration Examples

### Python with Requests
```python
import requests
from typing import Dict, List, Optional

class FirewallClient:
    """Client for IPC Cloud Firewall API."""
    
    def __init__(self, base_url: str, bearer_token: str):
        self.base_url = base_url
        self.headers = {
            'Authorization': f'Bearer {bearer_token}',
            'Content-Type': 'application/json'
        }
    
    def get_all_firewalls(
        self, 
        zone_id: Optional[str] = None,
        status: Optional[str] = None,
        include_rules: bool = True
    ) -> Dict:
        """Get all firewalls with optional filtering."""
        url = f"{self.base_url}/cloud/console/network/resources/firewallInfo.json"
        
        params = {}
        if zone_id:
            params['zone_id'] = zone_id
        if status:
            params['status'] = status
        if not include_rules:
            params['include_rules'] = 'false'
        
        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        return response.json()
    
    def get_firewall_by_id(self, firewall_id: str) -> Optional[Dict]:
        """Get a specific firewall by ID."""
        data = self.get_all_firewalls()
        
        for firewall in data['data']['firewalls']:
            if firewall['firewallId'] == firewall_id:
                return firewall
        return None
    
    def get_firewalls_by_zone(self, zone_name: str) -> List[Dict]:
        """Get all firewalls in a specific zone."""
        data = self.get_all_firewalls()
        
        return [
            fw for fw in data['data']['firewalls']
            if fw['zoneName'] == zone_name
        ]
    
    def check_port_accessibility(
        self, 
        firewall_id: str, 
        port: int, 
        protocol: str = 'tcp',
        direction: str = 'ingress'
    ) -> List[Dict]:
        """Check if a specific port is allowed in firewall rules."""
        firewall = self.get_firewall_by_id(firewall_id)
        if not firewall:
            return []
        
        rules_key = f'{direction}Rules'
        matching_rules = []
        
        for rule in firewall.get(rules_key, []):
            if not rule['enabled']:
                continue
            
            if rule['protocol'] not in [protocol, 'all']:
                continue
            
            # Check port range
            port_range = rule['portRange']
            if port_range == '*':
                matching_rules.append(rule)
            elif '-' in port_range:
                start, end = map(int, port_range.split('-'))
                if start <= port <= end:
                    matching_rules.append(rule)
            elif ',' in port_range:
                ports = [int(p.strip()) for p in port_range.split(',')]
                if port in ports:
                    matching_rules.append(rule)
            elif int(port_range) == port:
                matching_rules.append(rule)
        
        return matching_rules

# Usage example
if __name__ == '__main__':
    client = FirewallClient(
        base_url='https://ipcloud.tatacommunications.com',
        bearer_token='your_token_here'
    )
    
    # Get all firewalls
    firewalls = client.get_all_firewalls()
    print(f"Total firewalls: {len(firewalls['data']['firewalls'])}")
    
    # Get production zone firewalls
    prod_firewalls = client.get_firewalls_by_zone('Production')
    print(f"Production firewalls: {len(prod_firewalls)}")
    
    # Check if SSH is allowed
    if prod_firewalls:
        fw_id = prod_firewalls[0]['firewallId']
        ssh_rules = client.check_port_accessibility(fw_id, 22, 'tcp', 'ingress')
        print(f"SSH rules: {len(ssh_rules)}")
```

### JavaScript with Axios
```javascript
const axios = require('axios');

class FirewallClient {
  constructor(baseUrl, bearerToken) {
    this.baseUrl = baseUrl;
    this.axiosInstance = axios.create({
      baseURL: baseUrl,
      headers: {
        'Authorization': `Bearer ${bearerToken}`,
        'Content-Type': 'application/json'
      }
    });
  }

  async getAllFirewalls(options = {}) {
    const { zoneId, status, includeRules = true } = options;
    
    const params = {};
    if (zoneId) params.zone_id = zoneId;
    if (status) params.status = status;
    if (!includeRules) params.include_rules = 'false';

    const response = await this.axiosInstance.get(
      '/cloud/console/network/resources/firewallInfo.json',
      { params }
    );
    return response.data;
  }

  async getFirewallById(firewallId) {
    const data = await this.getAllFirewalls();
    return data.data.firewalls.find(fw => fw.firewallId === firewallId);
  }

  async findOpenPorts(firewallId, direction = 'ingress') {
    const firewall = await this.getFirewallById(firewallId);
    if (!firewall) return [];

    const rulesKey = `${direction}Rules`;
    return firewall[rulesKey]
      .filter(rule => rule.enabled && rule.action === 'allow')
      .map(rule => ({
        port: rule.portRange,
        protocol: rule.protocol,
        cidr: direction === 'ingress' ? rule.sourceCidr : rule.destinationCidr,
        description: rule.description
      }));
  }

  async generateSecurityReport() {
    const data = await this.getAllFirewalls();
    const firewalls = data.data.firewalls;

    return {
      totalFirewalls: firewalls.length,
      activeFirewalls: firewalls.filter(fw => fw.status === 'active').length,
      loggingEnabled: firewalls.filter(fw => fw.loggingEnabled).length,
      defaultDeny: firewalls.filter(fw => fw.defaultPolicy === 'deny').length,
      vulnerabilities: this._checkVulnerabilities(firewalls)
    };
  }

  _checkVulnerabilities(firewalls) {
    const vulnerabilities = [];

    for (const fw of firewalls) {
      // Check for SSH open to internet
      const sshRules = fw.ingressRules.filter(rule =>
        rule.enabled &&
        rule.action === 'allow' &&
        rule.sourceCidr === '0.0.0.0/0' &&
        rule.portRange.includes('22')
      );

      if (sshRules.length > 0) {
        vulnerabilities.push({
          firewall: fw.firewallName,
          severity: 'high',
          issue: 'SSH exposed to internet'
        });
      }

      // Check for overly permissive rules
      const permissiveRules = fw.ingressRules.filter(rule =>
        rule.enabled &&
        rule.action === 'allow' &&
        rule.sourceCidr === '0.0.0.0/0' &&
        rule.portRange === '*'
      );

      if (permissiveRules.length > 0) {
        vulnerabilities.push({
          firewall: fw.firewallName,
          severity: 'critical',
          issue: 'All ports open to internet'
        });
      }
    }

    return vulnerabilities;
  }
}

// Usage
(async () => {
  const client = new FirewallClient(
    'https://ipcloud.tatacommunications.com',
    'your_token_here'
  );

  try {
    // Get all firewalls
    const firewalls = await client.getAllFirewalls();
    console.log(`Total firewalls: ${firewalls.data.firewalls.length}`);

    // Generate security report
    const report = await client.generateSecurityReport();
    console.log('Security Report:', report);

    // Find open ports
    if (firewalls.data.firewalls.length > 0) {
      const fwId = firewalls.data.firewalls[0].firewallId;
      const openPorts = await client.findOpenPorts(fwId, 'ingress');
      console.log('Open ports:', openPorts);
    }
  } catch (error) {
    console.error('Error:', error.message);
  }
})();
```

## Network Security Best Practices

### Rule Design Principles
1. **Default Deny**: Set default policy to "deny" and explicitly allow required traffic
2. **Least Privilege**: Only open ports necessary for application functionality
3. **CIDR Restrictions**: Avoid 0.0.0.0/0 except for public-facing services
4. **Protocol Specificity**: Specify exact protocols (tcp/udp) instead of "all"
5. **Port Ranges**: Use specific ports instead of wildcards

### Common Security Patterns

#### Web Server Firewall
```python
web_firewall_rules = {
    'ingress': [
        {
            'protocol': 'tcp',
            'portRange': '80,443',
            'sourceCidr': '0.0.0.0/0',  # Public HTTP/HTTPS
            'action': 'allow'
        },
        {
            'protocol': 'tcp',
            'portRange': '22',
            'sourceCidr': '10.0.0.0/8',  # SSH from internal only
            'action': 'allow'
        }
    ],
    'egress': [
        {
            'protocol': 'tcp',
            'portRange': '443',
            'destinationCidr': '0.0.0.0/0',  # HTTPS to anywhere
            'action': 'allow'
        }
    ],
    'defaultPolicy': 'deny'
}
```

#### Database Firewall
```python
db_firewall_rules = {
    'ingress': [
        {
            'protocol': 'tcp',
            'portRange': '3306',
            'sourceCidr': '10.0.1.0/24',  # MySQL from app subnet only
            'action': 'allow'
        },
        {
            'protocol': 'tcp',
            'portRange': '22',
            'sourceCidr': '10.0.0.0/8',  # SSH from internal
            'action': 'allow'
        }
    ],
    'egress': [
        {
            'protocol': 'tcp',
            'portRange': '443',
            'destinationCidr': '0.0.0.0/0',  # HTTPS for updates
            'action': 'allow'
        }
    ],
    'defaultPolicy': 'deny'
}
```

### Audit and Compliance Checks

```python
def audit_firewall_security(firewall_data):
    """Perform comprehensive security audit on firewalls."""
    findings = {
        'critical': [],
        'high': [],
        'medium': [],
        'low': []
    }
    
    for firewall in firewall_data['data']['firewalls']:
        fw_name = firewall['firewallName']
        
        # Critical: Default allow policy
        if firewall.get('defaultPolicy') == 'allow':
            findings['critical'].append(
                f"{fw_name}: Default policy is ALLOW - should be DENY"
            )
        
        # High: No logging
        if not firewall.get('loggingEnabled'):
            findings['high'].append(
                f"{fw_name}: Logging is disabled"
            )
        
        # Check ingress rules
        for rule in firewall.get('ingressRules', []):
            if not rule['enabled']:
                continue
            
            # Critical: All traffic from anywhere
            if (rule['sourceCidr'] == '0.0.0.0/0' and 
                rule['portRange'] == '*'):
                findings['critical'].append(
                    f"{fw_name} Rule {rule['ruleId']}: Allows ALL traffic from internet"
                )
            
            # High: Management ports exposed
            mgmt_ports = ['22', '3389', '5985', '5986']
            if rule['sourceCidr'] == '0.0.0.0/0':
                for port in mgmt_ports:
                    if port in rule['portRange']:
                        findings['high'].append(
                            f"{fw_name} Rule {rule['ruleId']}: " +
                            f"Management port {port} exposed to internet"
                        )
            
            # Medium: Overly broad CIDR
            if rule['sourceCidr'].endswith('/8'):
                findings['medium'].append(
                    f"{fw_name} Rule {rule['ruleId']}: " +
                    f"Very broad CIDR range ({rule['sourceCidr']})"
                )
    
    return findings
```

## Related Operations
- `network.list_subnets` - List network subnets where firewalls are applied
- `vm.list` - List VMs to see which firewalls protect them
- `zone.list` - List zones to filter firewalls by zone
- `firewall.create` - Create new firewall configuration
- `firewall.update_rules` - Modify firewall rules
- `firewall.delete` - Delete firewall
- `security_group.list` - Alternative security group management
- `network_acl.list` - Network ACL configuration

## Error Handling

### Common Errors
- **401 Unauthorized**: Invalid or expired bearer token
  - **Resolution**: Refresh authentication token from Keycloak
  
- **403 Forbidden**: User doesn't have permission to view firewalls
  - **Resolution**: Verify user has viewer, developer, or admin role
  
- **404 Not Found**: Firewall ID not found
  - **Resolution**: Verify firewall exists and user has access to the zone
  
- **500 Internal Server Error**: Server error retrieving firewall data
  - **Resolution**: Check service status, retry with exponential backoff
  
- **503 Service Unavailable**: Firewall service temporarily unavailable
  - **Resolution**: Retry after a few seconds

### Error Response Example
```json
{
  "status": "error",
  "message": "Firewall not found or access denied",
  "responseCode": 404,
  "errorDetails": {
    "code": "FIREWALL_NOT_FOUND",
    "requestId": "req-12345-67890"
  }
}
```

### Handling Errors in Code
```python
def safe_get_firewalls(client, max_retries=3):
    """Get firewalls with retry logic and error handling."""
    import time
    
    for attempt in range(max_retries):
        try:
            return client.get_all_firewalls()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                raise Exception("Authentication failed - refresh token")
            elif e.response.status_code == 403:
                raise Exception("Access denied - insufficient permissions")
            elif e.response.status_code in [500, 503]:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    time.sleep(wait_time)
                    continue
                raise Exception("Service unavailable after retries")
            else:
                raise
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise Exception(f"Network error: {str(e)}")
    
    raise Exception("Failed after maximum retries")
```

## Response Codes
- `0` - Success
- `1` - Authentication failed
- `2` - Authorization failed
- `404` - Firewall not found
- `500` - Internal server error
- `503` - Service unavailable

## Performance Notes
- **Response time**: Typically < 2 seconds for up to 100 firewalls
- **Caching**: Firewall data cached for 5 minutes
- **Pagination**: Not currently implemented, returns all accessible firewalls
- **Rate limiting**: 100 requests per minute per user
- **Data size**: Approximately 2-5 KB per firewall with full rules

### Optimization Tips
```python
# Cache firewall data to reduce API calls
from functools import lru_cache
from datetime import datetime, timedelta

class CachedFirewallClient:
    def __init__(self, client):
        self.client = client
        self._cache = {}
        self._cache_duration = timedelta(minutes=5)
    
    def get_all_firewalls(self, force_refresh=False):
        cache_key = 'all_firewalls'
        
        if not force_refresh and cache_key in self._cache:
            cached_data, cached_time = self._cache[cache_key]
            if datetime.now() - cached_time < self._cache_duration:
                return cached_data
        
        # Fetch fresh data
        data = self.client.get_all_firewalls()
        self._cache[cache_key] = (data, datetime.now())
        return data
```

## Usage Notes

### Firewall Types
- **Stateful**: Tracks connection state, automatically allows return traffic
- **Stateless**: Requires explicit rules for both directions

### Rule Priority
- Rules are evaluated in priority order (lower number = higher priority)
- First matching rule determines action (allow/deny)
- If no rules match, default policy applies

### CIDR Notation
- `0.0.0.0/0` - All IPv4 addresses (use cautiously)
- `10.0.0.0/8` - Private network class A (10.0.0.0 - 10.255.255.255)
- `192.168.0.0/16` - Private network class C
- `/32` - Single IP address

### Port Ranges
- Single port: `"80"`
- Multiple ports: `"80,443,8080"`
- Range: `"8000-9000"`
- All ports: `"*"`

### Best Practices
1. Always enable logging for audit trails
2. Set default policy to "deny"
3. Use specific CIDR ranges instead of 0.0.0.0/0 when possible
4. Regularly audit and remove unused rules
5. Document rule purposes in descriptions
6. Test rule changes in non-production first
7. Monitor firewall statistics for anomalies

## Metadata
- **Generated:** 2025-02-16T09:45:00.000000Z
- **Source:** IPC Cloud Console Network Resources API
- **API Version:** v2
- **Base Path:** /cloud/console/network/resources/
- **Documentation:** Firewall security group configuration and rule management
- **Additional Notes:** Supports stateful and stateless firewalls with ingress/egress rules