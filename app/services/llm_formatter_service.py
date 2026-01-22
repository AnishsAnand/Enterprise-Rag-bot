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
        Format API response using LLM with context awareness.
        
        Args:
            resource_type: Type of resource
            operation: Operation performed
            raw_data: Raw API response data
            user_query: Original user query
            context: CRITICAL - Must include query_type (specific/general/detailed)
            
        Returns:
            Context-aware formatted response
        """
        try:
            # Get query type from context
            query_type = context.get("query_type", "general") if context else "general"
            
            logger.info(f"📝 Formatting {resource_type} response (query_type: {query_type})")
            
            # Build context-aware prompt
            prompt = self._build_prompt(
                resource_type,
                operation,
                raw_data,
                user_query,
                context,
                query_type
            )
            
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
                return self._fallback_format(resource_type, operation, raw_data, query_type)
                
        except Exception as e:
            logger.error(f"Error formatting response: {str(e)}")
            return self._fallback_format(resource_type, operation, raw_data, query_type)
    
    def _build_prompt(
        self,
        resource_type: str,
        operation: str,
        raw_data: Any,
        user_query: str,
        context: Optional[Dict[str, Any]],
        query_type: str
    ) -> str:
        """Build context-aware formatting prompt."""
        
        # Truncate data if too large
        data_str = json.dumps(raw_data, indent=2, default=str)
        is_truncated = False
        actual_count = self._get_actual_count(raw_data)
        
        if len(data_str) > 8000:
            data_str = data_str[:8000] + "\n... (truncated)"
            is_truncated = True
        
        # Count notice
        count_notice = ""
        if is_truncated and actual_count > 0:
            count_notice = (
                "\n\n**IMPORTANT: The data below is truncated for processing. "
                f"The ACTUAL total count is {actual_count} items. "
                "Always report this exact count in your summary. Also, at the END of your response, add a note: "
                "'📌 _Some results were truncated. Let me know if you'd like to see the complete list or filter by specific criteria._'**"
            )
        
        # Get query-type-specific instructions
        query_instructions = self._get_query_type_instructions(query_type, context)
        
        # Get resource-specific instructions
        resource_instructions = self._get_resource_instructions(resource_type, context)
        
        return f"""You are a cloud infrastructure assistant. Format the following API response data for the user in a clear, helpful way.

**User's Query:** {user_query or f"{operation} {resource_type}"}

**Operation:** {operation}
**Resource Type:** {resource_type}
**Query Type:** {query_type}
{count_notice}

**Raw Data:**
```json
{data_str}
```

{query_instructions}

{resource_instructions}

**General Guidelines:**
- Use markdown for readability
- Add helpful emojis
- Be concise - don't overwhelm with details
- No raw JSON in response

**CRITICAL: Respect the query_type above. Format accordingly.**"""
    
    def _get_query_type_instructions(
        self,
        query_type: str,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Get instructions based on query type."""
        
        instructions = {
            "specific": """**QUERY TYPE: SPECIFIC (Single Resource)**

This user asked about a SPECIFIC load balancer, NOT all of them.

**Your response should:**
1. Focus ONLY on the requested load balancer
2. Start with "⚖️ **Load Balancer: [name]**"
3. Show key details in a clean format:
   - Status with emoji (✅/⚠️/❌)
   - VIP address
   - Protocol and port
   - Backend pool health (if available)
   - Location/datacenter
4. Keep it concise - 5-8 lines max
5. Add hint: "💡 Use 'details for [name]' for full configuration"

**DO NOT:**
- List multiple load balancers
- Show tables with many rows
- Include unnecessary information
- Mention "Found X load balancers" (user asked for ONE)

**Example:**
```
⚖️ **Load Balancer: web-prod-lb-01**

✅ **Status:** Active and Healthy
📍 **Location:** Delhi Datacenter
🌐 **VIP:** 10.0.1.100:443 (HTTPS 🔒)
🖥️ **Backend Pool:** 4/4 servers healthy 🟢
⚙️ **Algorithm:** Round Robin

💡 **Tip:** Use 'details for web-prod-lb-01' for full configuration
```
""",
            
            "specific_detailed": """**QUERY TYPE: SPECIFIC DETAILED (Full Configuration)**

User wants DETAILED information about a specific load balancer.

**Your response should:**
1. Comprehensive but organized sections
2. Use headers (###) to separate sections
3. Show all configuration details
4. Include virtual services if available
5. Explain technical terms briefly

**Sections to include:**
- Overview (status, VIP, location)
- Configuration (protocol, port, SSL, algorithm)
- Backend Pools (health status, members)
- Virtual Services (if requested)
- SSL/TLS Configuration (if enabled)
""",
            
            "general": """**QUERY TYPE: GENERAL (List Multiple Resources)**

User wants to see MULTIPLE load balancers (or all).

**Your response should:**
1. Start with summary: "⚖️ Found X load balancers across Y datacenters"
2. Use table format if 3+ items
3. Group by datacenter/location
4. Show key info only: name, status, VIP, location
5. Add filter summary if applied
6. Limit to 10 items per location (mention "+ X more")

**Table Format (if 3+ items):**
| Name | Status | Location | VIP | Protocol |
|------|--------|----------|-----|----------|
| ... | ... | ... | ... | ... |

**List Format (if 1-2 items):**
✅ **name1** (Location) - VIP: x.x.x.x
✅ **name2** (Location) - VIP: y.y.y.y
""",
            
            "virtual_services": """**QUERY TYPE: VIRTUAL SERVICES**

User wants to see virtual services (VIPs/listeners) for a load balancer.

**Your response should:**
1. List each virtual service clearly
2. Show VIP, port, protocol
3. Include backend pool for each
4. Note SSL status
5. Use table if 3+ services
""",
        }
        
        return instructions.get(query_type, instructions["general"])
    
    def _get_resource_instructions(
        self,
        resource_type: str,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Get resource-specific formatting instructions."""
        
        if resource_type == "load_balancer":
            return """**Load Balancer Specific Fields:**

**CRITICAL Fields to Show:**
- Name (primary identifier)
- Status with emoji (✅ Active | ⚠️ Degraded | ❌ Inactive)
- VIP (Virtual IP) - what clients connect to
- Protocol (HTTP/HTTPS 🔒/TCP/UDP)
- Port number
- Backend pool health: X/Y healthy 🟢🟡🔴
- Location/Datacenter (📍)
- SSL status (🔒 if enabled)

**Health Status Indicators:**
- 🟢 Healthy - All backends up
- 🟡 Degraded - Some backends down
- 🔴 Critical - All backends down

**Algorithm (if available):**
- Round Robin, Least Connections, IP Hash, etc.

**Remember:** Load balancers are critical infrastructure - be clear and actionable!"""
        
        elif resource_type == "k8s_cluster":
            return """**Kubernetes Cluster Fields:**
- Cluster name, status, K8s version
- Node count and health
- Control plane type
- Location/datacenter"""
        
        else:
            return f"""**{resource_type.title()} Fields:**
- Show key identifying fields
- Use tables for multiple items
- Highlight status and health"""
    
    def _get_actual_count(self, raw_data: Any) -> int:
        """Get actual count of items before truncation."""
        if isinstance(raw_data, list):
            return len(raw_data)
        elif isinstance(raw_data, dict):
            for key in ["data", "clusters", "vms", "services", "department"]:
                if key in raw_data and isinstance(raw_data[key], list):
                    return len(raw_data[key])
        return 0
    
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

