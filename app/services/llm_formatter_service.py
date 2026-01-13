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
            count_notice = f"\n\n**IMPORTANT: The data below is truncated for processing. The ACTUAL total count is {actual_count} items. Always report this exact count in your summary. Also, at the END of your response, add a note: '📌 _Some results were truncated. Let me know if you'd like to see the complete list or filter by specific criteria._'**"
        
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
- Present as a clean reference list"""
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

