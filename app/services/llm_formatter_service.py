"""
LLM Response Formatter Service - Single source of truth for LLM-based formatting.
All agents should use this service to format API responses.
"""

import json
import logging
import os
from typing import Any, Dict, Optional

from app.services.ai_service import ai_service

logger = logging.getLogger(__name__)


class LLMFormatterService:
    """
    Centralized LLM-based response formatting service.
    
    Provides consistent formatting across all resource types with:
    - Common base formatting logic
    - Resource-specific prompt customization
    - Graceful fallback on LLM failure
    """
    
    def __init__(self):
        self.temperature = 0.3
        self.max_tokens = 2000
        self.timeout = float(os.getenv("HTTP_TIMEOUT_SECONDS", "30"))  # Use env var, default 30s
        logger.info(f"✅ LLMFormatterService initialized (timeout={self.timeout}s)")
    
    async def format_response(
        self,
        resource_type: str,
        operation: str,
        raw_data: Any,
        user_query: str = "",
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Format API response using LLM.
        
        Args:
            resource_type: Type of resource (k8s_cluster, vm, business_unit, etc.)
            operation: Operation performed (list, create, update, delete)
            raw_data: Raw API response data
            user_query: Original user query for context
            context: Additional context (endpoint_names, service_type, etc.)
            
        Returns:
            User-friendly formatted response string
        """
        try:
            # Build the prompt
            prompt = self._build_prompt(resource_type, operation, raw_data, user_query, context)
            
            # Call LLM
            response = await ai_service._call_chat_with_retries(
                prompt=prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                timeout=self.timeout
            )
            
            if response:
                return response
            else:
                return self._fallback_format(resource_type, operation, raw_data)
                
        except Exception as e:
            logger.error(f"Error formatting response with LLM: {str(e)}")
            return self._fallback_format(resource_type, operation, raw_data)
    
    def _build_prompt(
        self,
        resource_type: str,
        operation: str,
        raw_data: Any,
        user_query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Build formatting prompt based on resource type.
        
        Args:
            resource_type: Resource type
            operation: Operation performed
            raw_data: Raw data to format
            user_query: User's query
            context: Additional context
            
        Returns:
            Prompt string for LLM
        """
        # Calculate ACTUAL count BEFORE truncation
        actual_count = 0
        if isinstance(raw_data, list):
            actual_count = len(raw_data)
        elif isinstance(raw_data, dict):
            # Try common keys for nested data
            for key in ["data", "clusters", "vms", "services", "department", "environments"]:
                if key in raw_data and isinstance(raw_data[key], list):
                    actual_count = len(raw_data[key])
                    break
        
        # Truncate data if too large
        data_str = json.dumps(raw_data, indent=2, default=str)
        is_truncated = False
        if len(data_str) > 8000:
            data_str = data_str[:8000] + "\n... (truncated)"
            is_truncated = True
        
        # Get resource-specific instructions
        resource_instructions = self._get_resource_instructions(resource_type, context)
        
        # Add count notice if truncated
        count_notice = ""
        if is_truncated and actual_count > 0:
            count_notice = f"\n\n**IMPORTANT: The data below is truncated for processing. The ACTUAL total count is {actual_count} items. Always report this exact count in your summary.**"
        
        return f"""You are a cloud infrastructure assistant. Format the following API response data for the user in a clear, helpful way.

**User's Query:** {user_query or f"{operation} {resource_type}"}

**Operation:** {operation}

**Resource Type:** {resource_type}
{count_notice}

**Raw Data:**
```json
{data_str}
```

**General Instructions:**
1. Present the information in a user-friendly format
2. Use markdown for better readability (tables, lists, bold text)
3. Highlight key information (status, names, counts, locations)
4. Add helpful emojis for visual clarity:
   - ✅ for Active/Running/Success
   - ⚠️ for Pending/Warning
   - ❌ for Failed/Error
   - 📁 for business units/departments
   - 🌍 for environments
   - 🖥️ for VMs
   - 🚢 for Kubernetes clusters
   - 🔥 for firewalls
   - 📦 for managed services
5. Include a summary at the top (e.g., "Found X items")
6. Keep it concise - use tables for multiple items
7. Be conversational and helpful

{resource_instructions}

Do NOT include raw JSON. Present only the formatted, user-friendly response."""
    
    def _get_resource_instructions(
        self,
        resource_type: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Get resource-specific formatting instructions.
        
        Args:
            resource_type: Resource type
            context: Additional context
            
        Returns:
            Resource-specific instruction string
        """
        instructions = {
            "business_unit": """**Business Unit Specific:**
- Show engagement name first
- List departments with their key metrics
- Key fields: name, ID, location, zones count, VMs count, environments count
- Group by location if multiple locations""",
            
            "environment": """**Environment Specific:**
- Show total environment count
- Key fields: name, ID, department, zone info
- Group by department or zone if available""",
            
            "zone": """**Zone/Network Segment Specific:**
- Show zone name, CIDR, and status (DRAFT/DEPLOYED)
- Key fields: zoneName, cidr, status, departmentName, environmentName, endpointName
- Highlight hypervisors (VCD_ESXI, ESXI, etc.)
- Show network type (Direct, VLAN, etc.)
- Group by department or endpoint for clarity
- Add 🟢 for DEPLOYED, 🟡 for DRAFT status
- Show usable IPs count if available
- Note if zone is AI-enabled (isAiZone) or NAS-enabled (isNasZone)""",
            
            "k8s_cluster": """**Kubernetes Cluster Specific:**
- Show cluster name, status, K8s version, location
- Highlight control plane type (managed vs self-managed)
- Show node count if available
- Group by datacenter/location""",
            
            "vm": """**Virtual Machine Specific:**
- Key fields: vmName, endpoint, storage, engagement
- Show status with emojis
- Mention PPU metering and budgeting if enabled
- Group by endpoint if multiple locations""",
            
            "kafka": """**Kafka Service Specific:**
- Show service name, status, version, URL
- Highlight broker configuration
- Note replication factor if available""",
            
            "gitlab": """**GitLab Service Specific:**
- Show service name, status, version, URL
- Highlight repository/project info if available""",
            
            "firewall": """**Firewall Specific:**
- Show firewall name, status, rules count
- Highlight security policies
- Group by zone/location""",
            
            "endpoint": """**Endpoint/Datacenter Specific:**
- Show datacenter name, ID, type
- Use location emojis (📍)
- Present as a clean reference list""",
  "load_balancer": """**Load Balancer Specific Instructions:**
- Show total count with ⚖️ emoji in summary
- **CRITICAL KEY FIELDS** to display for each load balancer:
  - Name/ID (primary identifier)
  - Status with emojis:
    * ✅ Active/Running/Healthy
    * ⚠️ Degraded/Warning/Some backends down
    * ❌ Inactive/Failed/All backends down
  - Endpoint/Datacenter location (📍 emoji)
  - Virtual IP (VIP) address - the public-facing IP clients connect to
  - Protocol (HTTP, HTTPS 🔒, TCP, UDP)
  - Port numbers
  - Backend pool information:
    * Pool name
    * Number of backend members
    * Health status of pool
  - Load balancing algorithm (Round Robin, Least Connections, IP Hash, etc.)
  - SSL/TLS status (🔒 emoji if SSL enabled)
  - Session persistence/affinity settings if present

**Grouping and Organization:**
- Group by endpoint/datacenter for multi-location queries
- For each endpoint, show load balancers in a clear list or table

**Health Status Indicators:**
- Backend pool health is CRITICAL - highlight clearly:
  * 🟢 **Healthy** - All backend members are up and responding
  * 🟡 **Degraded** - Some backend members are down
  * 🔴 **Critical** - All backend members are down or unreachable
- Show individual backend member status if available

**SSL/Certificate Information:**
- If SSL/TLS is enabled, show:
  * Certificate status
  * Certificate expiration date (if available)
  * SSL termination point

**Traffic and Performance:**
- If available, show:
  * Current connections
  * Traffic statistics
  * Backend pool utilization

**Formatting Guidelines:**
- Use **tables** for multiple load balancers (easier to compare)
- For single load balancer, show detailed configuration in sections
- Always include a summary at the top:
  * Total LBs found
  * Number of endpoints queried
  * Active vs Inactive count
  * SSL-enabled count
  * Total backend pools and members
  
**Error Handling:**
- If some endpoints failed to query, note this SEPARATELY at the bottom
- Don't mix failed endpoints with successful results
- Clearly indicate which endpoints were successfully queried

**Context Usage:**
- Always include endpoint names in results for clarity
- If user queried specific location, highlight that in summary
- If user filtered by status/protocol, mention applied filters

**Example Summary Format:**
```
⚖️ **Found 5 Load Balancers** across 3 datacenters

**Summary:**
- Active: 4 ✅
- Inactive: 1 ❌
- SSL-enabled: 3 🔒
- Total backend pools: 12
- Total backend members: 48
```

**Example Table Format:**
| Name | Status | Endpoint | VIP | Protocol | Port | Backend Pool | SSL |
|------|--------|----------|-----|----------|------|--------------|-----|
| web-lb-01 | ✅ Active | Delhi | 10.0.1.100 | HTTPS 🔒 | 443 | 4/4 healthy 🟢 | Yes |
| api-lb-02 | ⚠️ Degraded | Mumbai | 10.0.2.50 | HTTP | 80 | 2/4 healthy 🟡 | No |

**Important Notes:**
- Virtual IP (VIP) is what clients connect to - always show this
- Backend pool health is critical for troubleshooting
- SSL status affects security posture
- Algorithm affects how traffic is distributed
- Session persistence affects user experience

Remember: Load balancers are critical infrastructure - provide clear, actionable information!""",
        }
        
        return instructions.get(resource_type, f"""**{resource_type.title()} Specific:**
- Show key identifying fields (name, ID, status)
- Present in table format if multiple items
- Highlight important attributes""")
    
    def _fallback_format(
        self,
        resource_type: str,
        operation: str,
        raw_data: Any
    ) -> str:
        """
        Fallback formatting if LLM fails.
        
        Args:
            resource_type: Resource type
            operation: Operation performed
            raw_data: Raw data
            
        Returns:
            Simple formatted string
        """
        try:
            emoji_map = {
                "business_unit": "📁",
                "environment": "🌍",
                "k8s_cluster": "🚢",
                "vm": "🖥️",
                "kafka": "📦",
                "gitlab": "📦",
                "firewall": "🔥",
                "endpoint": "📍"
            }
            emoji = emoji_map.get(resource_type, "✅")
            
            if isinstance(raw_data, list):
                count = len(raw_data)
                return f"{emoji} Found **{count} {resource_type}(s)**"
            elif isinstance(raw_data, dict):
                # Try to extract count from common response structures
                if "department" in raw_data:
                    items = raw_data.get("department", [])
                    if isinstance(items, list):
                        count = len(items)
                        return f"{emoji} Found **{count} business unit(s)**"
                elif "data" in raw_data:
                    items = raw_data.get("data", [])
                    if isinstance(items, list):
                        count = len(items)
                        return f"{emoji} Found **{count} {resource_type}(s)**"
                        
                return f"{emoji} Successfully retrieved {resource_type} data"
            else:
                return f"{emoji} Operation '{operation}' completed successfully"
                
        except Exception as e:
            return f"✅ Operation completed. (Formatting note: {str(e)})"


# Global instance
llm_formatter = LLMFormatterService()

