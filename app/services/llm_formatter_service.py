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
        
        # Get actual count BEFORE any processing
        actual_count = self._get_actual_count(raw_data)
        
        # NO TRUNCATION: Send all data to LLM for accurate formatting
        is_prompt_truncated = False
        display_data = raw_data
        
        # Log data size for monitoring
        logger.info(f"📊 Formatting {actual_count} items (no truncation)")
        
        data_str = json.dumps(display_data, indent=2, default=str)
        
        # Only log if data is very large, but DO NOT truncate
        if len(data_str) > 50000:
            logger.info(f"📋 Large data payload: {len(data_str)} chars for {actual_count} items")
        
        # Check if data was truncated upstream (by API or other services)
        data_truncated = False
        if context and isinstance(context, dict):
            data_truncated = bool(context.get("data_truncated"))
        if not data_truncated and isinstance(raw_data, dict):
            data_truncated = bool(raw_data.get("truncated"))
        
        # Count notice - CRITICAL: Tell LLM the actual count
        count_notice = ""
        
        # ALWAYS tell LLM the actual count to prevent it from filtering/omitting items
        if actual_count > 0:
            count_notice = f"\n\n**CRITICAL: The data contains {actual_count} items. You MUST display ALL {actual_count} items in your response. Do NOT filter or omit any items - the data is already filtered by the system.**"
        
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
- SHOW ALL ITEMS in the data - do NOT filter or omit any
- The data is already filtered by location/endpoint - display everything provided

**CRITICAL: Respect the query_type above. Format accordingly. Display ALL items in the data.**"""
    
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

        elif resource_type == "firewall":
            return """**Firewall Fields (CRITICAL - extract correctly):**

**Name Extraction (in priority order):**
1. `displayName` - primary firewall name
2. `department[0].name` - department/tenant name (often the only name available)
3. Format as: "FirewallName (DepartmentName)" if both exist, otherwise just use what's available
4. Fallback to ID only if nothing else exists

**Type Extraction:**
- Check `LOGO` field first: "IZO FW (F)" → "Vayu Firewall(F)", "IZO FW (N)" → "Vayu Firewall(N)", "Fortinet" → "Fortinet"
- Check `component` field: May contain "Vayu Firewall(F)", etc.
- Use 🔵 for Vayu Firewall(F), 🟢 for Vayu Firewall(N), 🟧 for Fortinet

**IP Extraction:**
- Check `ip` or `IP` field
- If value is 0, "0", "None", or empty, show "N/A"

**Table Format (required for lists):**
| Name | IP | Type |
|------|-----|------|
| **DisplayName (Dept)** | 100.108.0.100 | 🔵 Vayu Firewall(F) |

**Formatting Rules:**
- Start with: 🔥 Found **X firewall(s)** [in Location if filtered]
- Group by `_location` field if present
- Use ### 📍 LocationName for each group
- Do NOT add VIP, protocol, status columns (API doesn't provide these)
- Do NOT invent values; if a field is missing, omit it
- Add tip at end: 💡 **Tip:** Ask about a specific firewall by name for more details.

**CRITICAL - SHOW ALL ITEMS:**
- The data provided is ALREADY FILTERED by the system based on user's location/endpoint query
- You MUST display ALL firewalls in the data, not just ones matching a keyword
- Do NOT filter by name pattern - if user asked for "blr endpoint", ALL firewalls from that endpoint are in the data
- Count the items in the JSON and ensure your table has the same number of rows"""
        
        elif resource_type == "k8s_cluster":
            return """**Kubernetes Cluster Formatting (REQUIRED FORMAT):**

**Summary Line (required):**
Start with: 🚢 Found X Kubernetes Cluster(s) across Y datacenter(s)

**Table Format (REQUIRED for all cluster lists):**
| Cluster Name | Status | K8s Version | Nodes | Control Plane | Datacenter |
|--------------|--------|-------------|-------|---------------|------------|
| cluster-name | ✅ Healthy | v1.32.10 | 7 | ⚪ APP | EP_V2_BL |

**Status Emoji (required in Status column):**
- ✅ Healthy/Running/Active
- ⏳ Creating  
- ⚠️ Degraded/Warning
- ❌ Failed/Stopped/Error

**Control Plane Types:**
- ⚪ APP = Application workloads
- ⚪ MGMT = Management/system workloads

**Key Identifying Fields Section:**
Show for each cluster:
- clusterId
- clusterName
- status
- locationName (displayNameEndpoint)
- kubernetesVersion
- nodescount
- type (APP/MGMT)
- ciMasterId

**Additional Information Section:**
Show for each cluster:
- backupEnabled (true/false)
- createdTime
- Any other fields from API

**CRITICAL:**
- Extract ALL fields from API response - do NOT filter or omit any fields
- ALWAYS use table format for cluster lists
- Include ALL clusters in the table (do not omit any)
- Show datacenter name from endpoint info
- Add detailed sections with all available information

**Closing Line:**
End with: "💡 **Tip:** Ask about a specific cluster by name for more details."""
        
        elif resource_type == "vm":
            return """**Virtual Machine Fields:**
- VM name, status, storage
- Health and location/datacenter
- Include tags only if user asked

**Emoji Guidance (required):**
- Prefix status with emoji: ✅ Running, ⚠️ Deleting, ⏳ Creating, ❌ Stopped/Failed
- Use 📍 for location and 🖥️ for the summary line
- If using a table, include the emoji in the Status column"""
        
        elif resource_type == "managed_service":
            service_type = context.get("service_type", "") if context else ""
            service_display_name = context.get("service_display_name", "Managed Service") if context else "Managed Service"
            return f"""**Managed Service Fields (CRITICAL - Extract ALL fields):**

**Required Format:**
1. Summary: "✔ Found X {service_display_name} instance(s) across Y datacenter(s)"
2. Table with columns: Name | Status | Location | VIP | Protocol
3. Key Identifying Fields section with:
   - serviceType
   - instanceNamespace
   - version
   - status
   - locationName
   - clusterName
   - engagementName
   - departmentName
4. Additional Information section with:
   - logs (as clickable URL if present)
   - analyticsUrl (as clickable URL if present)
   - backup (true/false)
   - plugins (comma-separated list)
   - Any other fields from the API

**CRITICAL:**
- Extract ALL fields from the API response - do NOT filter or omit any fields
- Format URLs as clickable markdown links: [text](url)
- Show all details for each service instance
- Group by location/datacenter if multiple instances"""
        
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
    
    async def format_response_agentic(
        self,
        resource_type: str,
        operation: str,
        raw_data: Any,
        user_query: str = "",
        context: Optional[Dict[str, Any]] = None,
        chunk_size: int = 15
    ) -> str:
        """
        AGENTIC formatting with validation - adapts to API structure changes.
        
        Strategy:
        1. For large datasets: Use chunked formatting + validation
        2. For small datasets: Use structured output + validation
        3. Always validate output against source data
        
        Args:
            resource_type: Type of resource
            operation: Operation performed
            raw_data: Raw API response data
            user_query: Original user query
            context: Additional context
            chunk_size: Items per chunk for large datasets
            
        Returns:
            Formatted response with validation
        """
        try:
            items = self._extract_items_list(raw_data)
            if not items:
                query_type = context.get("query_type", "general") if context else "general"
                return self._fallback_format(resource_type, operation, raw_data, query_type)
            
            actual_count = len(items)
            query_type = context.get("query_type", "general") if context else "general"
            
            logger.info(f"🤖 Agentic formatting: {actual_count} {resource_type} items")
            
            # Strategy 1: Large datasets - use chunked formatting
            if actual_count > chunk_size:
                return await self._format_chunked_with_validation(
                    resource_type, operation, items, user_query, context, chunk_size
                )
            
            # Strategy 2: Small datasets - use structured output with validation
            else:
                return await self._format_structured_with_validation(
                    resource_type, operation, items, user_query, context
                )
                
        except Exception as e:
            logger.error(f"❌ Agentic formatting failed: {e}", exc_info=True)
            query_type = context.get("query_type", "general") if context else "general"
            return await self.format_response(resource_type, operation, raw_data, user_query, context)
    
    async def format_response_agentic_streaming(
        self,
        resource_type: str,
        operation: str,
        raw_data: Any,
        user_query: str = "",
        context: Optional[Dict[str, Any]] = None,
        chunk_size: int = 15
    ):
        """
        AGENTIC formatting with STREAMING - yields chunks as they're processed.
        
        This allows the UI to start displaying results immediately while chunks
        are still being processed, making the response feel much faster.
        
        Args:
            resource_type: Type of resource
            operation: Operation performed
            raw_data: Raw API response data
            user_query: Original user query
            context: Additional context
            chunk_size: Items per chunk for large datasets
            
        Yields:
            Formatted chunks as they're ready (str)
        """
        try:
            items = self._extract_items_list(raw_data)
            if not items:
                query_type = context.get("query_type", "general") if context else "general"
                yield self._fallback_format(resource_type, operation, raw_data, query_type)
                return
            
            actual_count = len(items)
            query_type = context.get("query_type", "general") if context else "general"
            
            logger.info(f"🤖 Agentic streaming formatting: {actual_count} {resource_type} items")
            
            # Strategy 1: Large datasets - stream chunks as they're processed
            if actual_count > chunk_size:
                async for chunk in self._format_chunked_with_validation_streaming(
                    resource_type, operation, items, user_query, context, chunk_size
                ):
                    yield chunk
            # Strategy 2: Small datasets - format and yield immediately
            else:
                formatted = await self._format_structured_with_validation(
                    resource_type, operation, items, user_query, context
                )
                yield formatted
                
        except Exception as e:
            logger.error(f"❌ Agentic streaming formatting failed: {e}", exc_info=True)
            query_type = context.get("query_type", "general") if context else "general"
            formatted = await self.format_response(resource_type, operation, raw_data, user_query, context)
            yield formatted
    
    def _extract_items_list(self, raw_data: Any) -> list:
        """Extract list of items from various data structures."""
        if isinstance(raw_data, list):
            return raw_data
        elif isinstance(raw_data, dict):
            # Try common keys
            for key in ["data", "firewalls", "clusters", "vms", "services", "items", "results"]:
                if key in raw_data and isinstance(raw_data[key], list):
                    return raw_data[key]
            # If no list found, return as single item
            return [raw_data]
        return []
    
    async def _format_chunked_with_validation(
        self,
        resource_type: str,
        operation: str,
        items: list,
        user_query: str,
        context: Optional[Dict[str, Any]],
        chunk_size: int
    ) -> str:
        """
        Format large datasets by chunking, then validate and combine.
        This prevents LLM hallucination by processing smaller batches.
        """
        total_count = len(items)
        logger.info(f"📦 Chunking {total_count} items into batches of {chunk_size}")
        
        # Process in chunks
        formatted_chunks = []
        for i in range(0, total_count, chunk_size):
            chunk = items[i:i + chunk_size]
            chunk_num = (i // chunk_size) + 1
            total_chunks = (total_count + chunk_size - 1) // chunk_size
            
            logger.info(f"🔄 Processing chunk {chunk_num}/{total_chunks} ({len(chunk)} items)")
            
            # Format this chunk with structured output
            chunk_formatted = await self._format_chunk_structured(
                resource_type, operation, chunk, user_query, context,
                chunk_info=f"Chunk {chunk_num}/{total_chunks}"
            )
            
            # Validate chunk output
            validation = self._validate_chunk_output(chunk_formatted, chunk, resource_type)
            if not validation["valid"]:
                logger.warning(f"⚠️ Chunk {chunk_num} validation failed: {validation['error']}")
                logger.info(f"🔄 Falling back to programmatic formatter for chunk {chunk_num}")
                # Fallback: format programmatically for this chunk
                chunk_formatted = self._format_chunk_programmatic(chunk, resource_type)
                # Validate the programmatic output
                prog_validation = self._validate_chunk_output(chunk_formatted, chunk, resource_type)
                if prog_validation["valid"]:
                    logger.info(f"✅ Programmatic formatter succeeded for chunk {chunk_num}")
                else:
                    logger.error(f"❌ Programmatic formatter also failed for chunk {chunk_num}: {prog_validation['error']}")
            
            formatted_chunks.append(chunk_formatted)
        
        # Combine chunks with summary
        return self._combine_chunks(formatted_chunks, total_count, resource_type, context)
    
    async def _format_chunked_with_validation_streaming(
        self,
        resource_type: str,
        operation: str,
        items: list,
        user_query: str,
        context: Optional[Dict[str, Any]],
        chunk_size: int
    ):
        """
        Format large datasets by chunking with STREAMING - yields chunks as they're ready.
        This allows the UI to start displaying results immediately.
        """
        total_count = len(items)
        logger.info(f"📦 Streaming: Chunking {total_count} items into batches of {chunk_size}")
        
        # Yield header immediately
        yield f"🔥 **Found {total_count} {resource_type}(s) across multiple datacenters**\n\n"
        yield "📊 Processing data...\n\n"
        
        # Process and yield chunks as they're ready
        all_parsed_items = []
        chunk_num = 0
        
        for i in range(0, total_count, chunk_size):
            chunk = items[i:i + chunk_size]
            chunk_num += 1
            total_chunks = (total_count + chunk_size - 1) // chunk_size
            
            logger.info(f"🔄 Streaming chunk {chunk_num}/{total_chunks} ({len(chunk)} items)")
            
            # Format this chunk with structured output
            chunk_formatted = await self._format_chunk_structured(
                resource_type, operation, chunk, user_query, context,
                chunk_info=f"Chunk {chunk_num}/{total_chunks}"
            )
            
            # Validate chunk output
            validation = self._validate_chunk_output(chunk_formatted, chunk, resource_type)
            if not validation["valid"]:
                logger.warning(f"⚠️ Chunk {chunk_num} validation failed: {validation['error']}")
                # Fallback: format programmatically for this chunk
                parsed_chunk = [self._parse_item_for_streaming(item, resource_type) for item in chunk]
            else:
                # Parse validated JSON
                parsed_chunk = self._extract_json_from_response(chunk_formatted)
                if not parsed_chunk or not isinstance(parsed_chunk, list):
                    parsed_chunk = [self._parse_item_for_streaming(item, resource_type) for item in chunk]
            
            all_parsed_items.extend(parsed_chunk)
            
            # Yield formatted chunk immediately (group by location if we have location data)
            chunk_markdown = self._format_chunk_for_streaming(parsed_chunk, resource_type, chunk_num, total_chunks)
            yield chunk_markdown
        
        # Yield final summary
        yield f"\n✅ **Complete: {total_count} {resource_type}(s) processed**\n"
        yield f"💡 **Tip:** Ask about a specific {resource_type} by name for more details."
    
    def _parse_item_for_streaming(self, item: Dict[str, Any], resource_type: str) -> Dict[str, Any]:
        """Parse a single item for streaming (extract key fields)."""
        return {
            "name": item.get("displayName") or item.get("technicalName") or item.get("name") or item.get("firewallName") or "Unknown",
            "ip": item.get("ip") or item.get("ipAddress") or item.get("vipIp") or item.get("publicIP") or "N/A",
            "type": item.get("component") or item.get("componentType") or item.get("type") or "Unknown",
            "location": item.get("endpointName") or item.get("location") or item.get("_location") or item.get("datacenter") or "Unknown"
        }
    
    def _format_chunk_for_streaming(self, parsed_chunk: list, resource_type: str, chunk_num: int, total_chunks: int) -> str:
        """Format a parsed chunk into markdown for streaming."""
        if not parsed_chunk:
            return ""
        
        # Group by location
        by_location: Dict[str, list] = {}
        for item in parsed_chunk:
            location = item.get("location", "Unknown")
            if location not in by_location:
                by_location[location] = []
            by_location[location].append(item)
        
        lines = []
        
        # If first chunk, add table headers
        if chunk_num == 1:
            for location, items_list in sorted(by_location.items()):
                lines.append(f"\n### 📍 {location} ({len(items_list)})")
                lines.append("| Name | IP | Type |")
                lines.append("|------|-----|------|")
                break  # Only add headers once
        
        # Add items for each location
        for location, items_list in sorted(by_location.items()):
            # Add location header if not first chunk
            if chunk_num > 1:
                lines.append(f"\n### 📍 {location} ({len(items_list)})")
                if chunk_num == 2:  # Add headers on second chunk if we didn't before
                    lines.append("| Name | IP | Type |")
                    lines.append("|------|-----|------|")
            
            for item in items_list:
                name = item.get("name", "Unknown")
                ip = item.get("ip", "N/A")
                fw_type = item.get("type", "Unknown")
                
                # Add type emoji
                if "Vayu" in fw_type or "IZO" in fw_type:
                    if "(N)" in fw_type or "EdgeGateway" in fw_type:
                        type_display = "🔵 Vayu Firewall(N)"
                    else:
                        type_display = "🟢 Vayu Firewall(F)"
                elif "Fortinet" in fw_type:
                    type_display = "🟠 Fortinet"
                else:
                    type_display = f"⚪ {fw_type}"
                
                lines.append(f"| {name} | {ip} | {type_display} |")
        
        return "\n".join(lines)
    
    async def _format_chunk_structured(
        self,
        resource_type: str,
        operation: str,
        chunk: list,
        user_query: str,
        context: Optional[Dict[str, Any]],
        chunk_info: str
    ) -> str:
        """Format a single chunk using structured JSON output from LLM."""
        
        # Special handling for managed services and clusters - extract ALL fields
        if resource_type == "managed_service":
            prompt = f"""Extract ALL fields from this managed service data and output ONLY a JSON array.

**{chunk_info}** - Format these {len(chunk)} items:

```json
{json.dumps(chunk, indent=2, default=str)}
```

**CRITICAL: Extract ALL fields from each item. Include:**
- **name**: name, serviceName, displayName, technicalName
- **status**: status, healthStatus, state
- **location**: locationName, endpointName, location, datacenter
- **vip**: ingressUrl, vip, vipIp, url, accessUrl (preserve as-is)
- **protocol**: protocol (if available, otherwise "-")
- **serviceType**: serviceType
- **instanceNamespace**: instanceNamespace, namespace
- **version**: version
- **clusterName**: clusterName
- **engagementName**: engagementName, engagement.name
- **departmentName**: departmentName, department.name
- **logs**: logs, logsUrl (preserve as-is)
- **analyticsUrl**: analyticsUrl, monitoringUrl (preserve as-is)
- **backup**: backup, backupEnabled, isBackupEnabled (boolean)
- **plugins**: plugins, pluginList (comma-separated string if array)
- **ALL other fields**: Include any other fields present in the data

**Output Format:** JSON array with EXACTLY {len(chunk)} items, preserving ALL fields:
```json
[
  {{
    "name": "...",
    "status": "...",
    "location": "...",
    "vip": "...",
    "protocol": "...",
    "serviceType": "...",
    "instanceNamespace": "...",
    "version": "...",
    "clusterName": "...",
    "engagementName": "...",
    "departmentName": "...",
    "logs": "...",
    "analyticsUrl": "...",
    "backup": false,
    "plugins": "...",
    ... (all other fields from source)
  }},
  ...
]
```

**CRITICAL RULES:**
- Output EXACTLY {len(chunk)} items (one per input item)
- Preserve ALL fields from source data - do NOT filter or omit any fields
- If a field is missing, use null or "N/A" appropriately
- URLs should be preserved as-is
- Output ONLY valid JSON array, no markdown, no explanation, no code blocks

**Output:**"""
        elif resource_type == "k8s_cluster":
            prompt = f"""Extract ALL fields from this Kubernetes cluster data and output ONLY a JSON array.

**{chunk_info}** - Format these {len(chunk)} items:

```json
{json.dumps(chunk, indent=2, default=str)}
```

**CRITICAL: Extract ALL fields from each cluster. Include:**
- **name**: clusterName, name
- **status**: status, healthStatus, state
- **location**: displayNameEndpoint, endpointName, location, datacenter
- **k8sVersion**: kubernetesVersion, k8sVersion, version
- **nodes**: nodescount, nodesCount, nodeCount, nodes
- **type**: type, clusterType, controlPlane (APP/MGMT)
- **clusterId**: clusterId, id
- **backupEnabled**: isIksBackupEnabled, backupEnabled, isBackupEnabled (boolean)
- **createdTime**: createdTime, createdAt, created
- **ciMasterId**: ciMasterId, masterId
- **ALL other fields**: Include any other fields present in the data

**Output Format:** JSON array with EXACTLY {len(chunk)} items, preserving ALL fields:
```json
[
  {{
    "name": "...",
    "status": "...",
    "location": "...",
    "k8sVersion": "...",
    "nodes": 0,
    "type": "...",
    "clusterId": 0,
    "backupEnabled": false,
    "createdTime": "...",
    "ciMasterId": 0,
    ... (all other fields from source)
  }},
  ...
]
```

**CRITICAL RULES:**
- Output EXACTLY {len(chunk)} items (one per input item)
- Preserve ALL fields from source data - do NOT filter or omit any fields
- If a field is missing, use null or "N/A" appropriately
- Output ONLY valid JSON array, no markdown, no explanation, no code blocks

**Output:**"""
        else:
            prompt = f"""You are formatting {resource_type} data. Extract key fields intelligently and output ONLY a JSON array.

**{chunk_info}** - Format these {len(chunk)} items:

```json
{json.dumps(chunk, indent=2, default=str)}
```

**CRITICAL INSTRUCTIONS:**
1. Extract fields intelligently - look for:
   - **name**: displayName, technicalName, name, firewallName, clusterName, vmName
   - **ip**: ip, ipAddress, vipIp, publicIP
   - **type**: component, componentType, type, resourceType
   - **location**: endpointName, location, datacenter, _location, endId
   - **status**: status, healthStatus, state (if available)

2. Output EXACTLY {len(chunk)} items as JSON array (one per input item)
3. If a field doesn't exist, use "N/A" or "Unknown"
4. Output ONLY valid JSON array, no markdown, no explanation, no code blocks

**Output Format (JSON array):**
```json
[
  {{
    "name": "[extracted name]",
    "ip": "[extracted IP or 'N/A']",
    "type": "[extracted type]",
    "location": "[extracted location or 'Unknown']",
    "status": "[extracted status or 'N/A']"
  }},
  ...
]
```

**CRITICAL RULES:**
- Output EXACTLY {len(chunk)} items (one per input item)
- Do NOT skip any items
- Do NOT duplicate any items
- Preserve ALL items from input

**Output:**"""
        
        try:
            response = await ai_service._call_chat_with_retries(
                prompt=prompt,
                max_tokens=5000,  # Increased for clusters with many fields
                temperature=0.1,  # Low temperature for accuracy
                timeout=self.timeout * 2
            )
            
            if response:
                # Try to extract JSON from response
                json_data = self._extract_json_from_response(response)
                if json_data and isinstance(json_data, list):
                    return json.dumps(json_data, indent=2)
                else:
                    logger.warning(f"⚠️ Failed to extract JSON from LLM response. Response length: {len(response)}")
                    logger.debug(f"Response preview: {response[:500]}")
            
            logger.warning(f"⚠️ Empty response from LLM for chunk formatting, falling back to programmatic formatter")
            return ""
        except Exception as e:
            logger.error(f"❌ Chunk formatting error: {e}", exc_info=True)
            return ""
    
    def _validate_chunk_output(self, formatted_output: str, source_chunk: list, resource_type: str) -> Dict[str, Any]:
        """Validate that formatted output contains all source items."""
        try:
            if not formatted_output:
                return {"valid": False, "error": "Empty output"}
            
            # Try to parse as JSON
            parsed = self._extract_json_from_response(formatted_output)
            
            if not parsed or not isinstance(parsed, list):
                return {"valid": False, "error": "Output is not a valid JSON array"}
            
            # Check count matches
            if len(parsed) != len(source_chunk):
                return {
                    "valid": False,
                    "error": f"Count mismatch: expected {len(source_chunk)}, got {len(parsed)}"
                }
            
            # Check for duplicates in output (by name)
            names = [item.get("name") for item in parsed if item.get("name") and item.get("name") != "N/A" and item.get("name") != "Unknown"]
            if len(names) != len(set(names)):
                duplicates = [name for name in names if names.count(name) > 1]
                return {"valid": False, "error": f"Duplicate items detected: {set(duplicates)}"}
            
            # Check that we have at least name field for all items
            missing_names = sum(1 for item in parsed if not item.get("name") or item.get("name") in ["N/A", "Unknown"])
            if missing_names == len(parsed):
                return {"valid": False, "error": "All items missing name field"}
            
            return {"valid": True}
            
        except Exception as e:
            return {"valid": False, "error": f"Validation error: {str(e)}"}
    
    def _format_chunk_programmatic(self, chunk: list, resource_type: str) -> str:
        """Fallback programmatic formatter for a chunk when LLM validation fails.
        Returns JSON array so it can be combined with other chunks properly."""
        import json
        
        structured_items = []
        for item in chunk:
            if resource_type == "k8s_cluster":
                # Extract all cluster fields
                structured_item = {
                    "name": item.get("clusterName") or item.get("name") or "Unknown",
                    "status": item.get("status") or "N/A",
                    "location": item.get("displayNameEndpoint") or item.get("endpointName") or item.get("location") or "Unknown",
                    "k8sVersion": item.get("kubernetesVersion") or item.get("k8sVersion") or "N/A",
                    "nodes": item.get("nodescount") or item.get("nodesCount") or item.get("nodes") or 0,
                    "type": item.get("type") or item.get("clusterType") or "APP",
                    "clusterId": item.get("clusterId") or item.get("id") or None,
                    "backupEnabled": item.get("isIksBackupEnabled") or item.get("backupEnabled") or False,
                    "createdTime": item.get("createdTime") or item.get("createdAt") or None,
                    "ciMasterId": item.get("ciMasterId") or item.get("masterId") or None
                }
                # Add all other fields
                for key, value in item.items():
                    if key not in structured_item and value is not None:
                        structured_item[key] = value
                structured_items.append(structured_item)
            elif resource_type == "managed_service":
                # Extract all managed service fields
                structured_item = {
                    "name": item.get("name") or item.get("serviceName") or "Unknown",
                    "status": item.get("status") or "N/A",
                    "location": item.get("locationName") or item.get("endpointName") or item.get("location") or "Unknown",
                    "vip": item.get("ingressUrl") or item.get("vip") or item.get("url") or "N/A",
                    "protocol": item.get("protocol") or "-",
                    "serviceType": item.get("serviceType") or None,
                    "instanceNamespace": item.get("instanceNamespace") or item.get("namespace") or None,
                    "version": item.get("version") or None,
                    "clusterName": item.get("clusterName") or None,
                    "engagementName": item.get("engagementName") or None,
                    "departmentName": item.get("departmentName") or None,
                    "logs": item.get("logs") or item.get("logsUrl") or None,
                    "analyticsUrl": item.get("analyticsUrl") or item.get("monitoringUrl") or None,
                    "backup": item.get("backup") or item.get("backupEnabled") or item.get("isBackupEnabled") or False,
                    "plugins": item.get("plugins") or item.get("pluginList") or None
                }
                # Add all other fields
                for key, value in item.items():
                    if key not in structured_item and value is not None:
                        structured_item[key] = value
                structured_items.append(structured_item)
            else:
                # Generic fallback for other resource types
                structured_item = {
                    "name": item.get("displayName") or item.get("technicalName") or item.get("name") or item.get("firewallName") or "Unknown",
                    "ip": item.get("ip") or item.get("ipAddress") or item.get("vipIp") or item.get("publicIP") or "N/A",
                    "type": item.get("component") or item.get("componentType") or item.get("type") or "Unknown",
                    "location": item.get("endpointName") or item.get("location") or item.get("_location") or item.get("datacenter") or "Unknown",
                    "status": item.get("status") or item.get("healthStatus") or "N/A"
                }
                # Add all other fields
                for key, value in item.items():
                    if key not in structured_item and value is not None:
                        structured_item[key] = value
                structured_items.append(structured_item)
        
        # Return as JSON string so it can be parsed and combined
        return json.dumps(structured_items, indent=2, default=str)
    
    def _combine_chunks(
        self,
        formatted_chunks: list,
        total_count: int,
        resource_type: str,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Combine formatted chunks into final response with grouping by location."""
        # Parse all chunks into structured data
        all_items = []
        failed_chunks = 0
        
        for idx, chunk_str in enumerate(formatted_chunks, 1):
            try:
                if not chunk_str or not chunk_str.strip():
                    logger.warning(f"⚠️ Chunk {idx} is empty")
                    failed_chunks += 1
                    continue
                    
                parsed = self._extract_json_from_response(chunk_str)
                if parsed and isinstance(parsed, list):
                    all_items.extend(parsed)
                    logger.debug(f"✅ Chunk {idx}: Parsed {len(parsed)} items")
                else:
                    logger.warning(f"⚠️ Chunk {idx}: Failed to parse as JSON array. Type: {type(parsed)}")
                    failed_chunks += 1
            except Exception as e:
                logger.warning(f"⚠️ Chunk {idx}: Exception parsing chunk: {e}")
                failed_chunks += 1
        
        logger.info(f"📊 Combined chunks: {len(all_items)} items from {len(formatted_chunks)} chunks ({failed_chunks} failed)")
        
        # If we have structured data, format it nicely
        if all_items:
            if len(all_items) != total_count:
                logger.warning(f"⚠️ Item count mismatch: expected {total_count}, got {len(all_items)}")
            return self._format_validated_structured(all_items, total_count, resource_type, context)
        
        # Fallback: combine raw chunks (shouldn't happen if programmatic formatter works)
        logger.warning(f"⚠️ No structured data extracted from chunks, using raw fallback")
        lines = [f"🔥 **Found {total_count} {resource_type}(s)**\n"]
        for chunk in formatted_chunks:
            if chunk and chunk.strip():
                lines.append(chunk)
        lines.append(f"\n💡 **Tip:** Ask about a specific {resource_type} by name for more details.")
        return "\n".join(lines)
    
    async def _format_structured_with_validation(
        self,
        resource_type: str,
        operation: str,
        items: list,
        user_query: str,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """
        Format small datasets using structured JSON output, then validate.
        LLM outputs structured format, we validate it matches source data.
        """
        # Ask LLM to output structured JSON format
        structured_prompt = self._build_structured_prompt(
            resource_type, operation, items, user_query, context
        )
        
        try:
            response = await ai_service._call_chat_with_retries(
                prompt=structured_prompt,
                max_tokens=self.max_tokens * 2,  # More tokens for structured output
                temperature=0.1,  # Lower temperature for accuracy
                timeout=self.timeout * 2
            )
            
            if response:
                # Extract structured data from response
                structured_data = self._extract_json_from_response(response)
                
                # Validate against source
                validation = self._validate_structured_output(structured_data, items, resource_type)
                
                if validation["valid"]:
                    # Format the validated structured data
                    return self._format_validated_structured(structured_data, len(items), resource_type, context)
                else:
                    logger.warning(f"⚠️ Structured output validation failed: {validation['error']}")
                    # Fallback to regular formatting
                    return await self.format_response(resource_type, operation, items, user_query, context)
            else:
                return await self.format_response(resource_type, operation, items, user_query, context)
                
        except Exception as e:
            logger.error(f"❌ Structured formatting failed: {e}")
            return await self.format_response(resource_type, operation, items, user_query, context)
    
    def _build_structured_prompt(
        self,
        resource_type: str,
        operation: str,
        items: list,
        user_query: str,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Build prompt for structured JSON output."""
        
        # Special handling for managed services - extract ALL fields
        if resource_type == "managed_service":
            service_type = context.get("service_type", "") if context else ""
            return f"""Extract ALL fields from this managed service data and output ONLY a JSON array.

**Data ({len(items)} items):**
```json
{json.dumps(items, indent=2, default=str)}
```

**CRITICAL: Extract ALL fields from each item. Include:**
- **name**: name, serviceName, displayName, technicalName
- **status**: status, healthStatus, state
- **location**: locationName, endpointName, location, datacenter
- **vip**: ingressUrl, vip, vipIp, url, accessUrl (format as clickable URL if present)
- **protocol**: protocol (if available, otherwise "-")
- **serviceType**: serviceType
- **instanceNamespace**: instanceNamespace, namespace
- **version**: version
- **clusterName**: clusterName
- **engagementName**: engagementName, engagement.name
- **departmentName**: departmentName, department.name
- **logs**: logs, logsUrl (format as clickable URL if present)
- **analyticsUrl**: analyticsUrl, monitoringUrl (format as clickable URL if present)
- **backup**: backup, backupEnabled, isBackupEnabled (boolean)
- **plugins**: plugins, pluginList (comma-separated string if array)
- **ALL other fields**: Include any other fields present in the data

**Output Format:** JSON array with EXACTLY {len(items)} items, preserving ALL fields:
```json
[
  {{
    "name": "...",
    "status": "...",
    "location": "...",
    "vip": "...",
    "protocol": "...",
    "serviceType": "...",
    "instanceNamespace": "...",
    "version": "...",
    "clusterName": "...",
    "engagementName": "...",
    "departmentName": "...",
    "logs": "...",
    "analyticsUrl": "...",
    "backup": false,
    "plugins": "...",
    ... (all other fields from source)
  }},
  ...
]
```

**CRITICAL RULES:**
- Output EXACTLY {len(items)} items (one per input item)
- Preserve ALL fields from source data - do NOT filter or omit any fields
- If a field is missing, use null or "N/A" appropriately
- URLs should be preserved as-is (they will be formatted as clickable links)
- Output ONLY valid JSON array, no markdown, no explanation

**Output:**"""
        
        elif resource_type == "k8s_cluster":
            return f"""Extract ALL fields from this Kubernetes cluster data and output ONLY a JSON array.

**Data ({len(items)} items):**
```json
{json.dumps(items, indent=2, default=str)}
```

**CRITICAL: Extract ALL fields from each cluster. Include:**
- **name**: clusterName, name
- **status**: status, healthStatus, state
- **location**: displayNameEndpoint, endpointName, location, datacenter
- **k8sVersion**: kubernetesVersion, k8sVersion, version
- **nodes**: nodescount, nodesCount, nodeCount, nodes
- **type**: type, clusterType, controlPlane (APP/MGMT)
- **clusterId**: clusterId, id
- **backupEnabled**: isIksBackupEnabled, backupEnabled, isBackupEnabled (boolean)
- **createdTime**: createdTime, createdAt, created
- **ciMasterId**: ciMasterId, masterId
- **ALL other fields**: Include any other fields present in the data

**Output Format:** JSON array with EXACTLY {len(items)} items, preserving ALL fields:
```json
[
  {{
    "name": "...",
    "status": "...",
    "location": "...",
    "k8sVersion": "...",
    "nodes": 0,
    "type": "...",
    "clusterId": 0,
    "backupEnabled": false,
    "createdTime": "...",
    "ciMasterId": 0,
    ... (all other fields from source)
  }},
  ...
]
```

**CRITICAL RULES:**
- Output EXACTLY {len(items)} items (one per input item)
- Preserve ALL fields from source data - do NOT filter or omit any fields
- If a field is missing, use null or "N/A" appropriately
- Output ONLY valid JSON array, no markdown, no explanation

**Output:**"""
        
        # Default prompt for other resource types
        return f"""Extract key fields from this {resource_type} data and output ONLY a JSON array.

**Data ({len(items)} items):**
```json
{json.dumps(items, indent=2, default=str)}
```

**Extract these fields intelligently:**
- **name**: displayName, technicalName, name, firewallName, clusterName, vmName
- **ip**: ip, ipAddress, vipIp, publicIP
- **type**: component, componentType, type, resourceType
- **location**: endpointName, location, datacenter, _location, endId
- **status**: status, healthStatus, state (if available)

**Output Format:** JSON array with EXACTLY {len(items)} items:
```json
[
  {{"name": "...", "ip": "...", "type": "...", "location": "...", "status": "..."}},
  ...
]
```

**CRITICAL RULES:**
- Output EXACTLY {len(items)} items (one per input item)
- No duplicates, no omissions
- If field missing, use "N/A" or "Unknown"
- Output ONLY valid JSON array, no markdown, no explanation

**Output:**"""
    
    def _validate_structured_output(self, structured_data: Optional[list], source_items: list, resource_type: str) -> Dict[str, Any]:
        """Validate structured output matches source."""
        if not structured_data:
            return {"valid": False, "error": "No structured data"}
        
        if not isinstance(structured_data, list):
            return {"valid": False, "error": "Output is not a list"}
        
        if len(structured_data) != len(source_items):
            return {"valid": False, "error": f"Count mismatch: {len(structured_data)} vs {len(source_items)}"}
        
        # Check for duplicates
        names = [item.get("name") for item in structured_data if item.get("name") and item.get("name") not in ["N/A", "Unknown"]]
        if len(names) != len(set(names)):
            return {"valid": False, "error": "Duplicate items detected"}
        
        return {"valid": True}
    
    def _format_validated_structured(
        self,
        structured_data: list,
        total_count: int,
        resource_type: str,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Format validated structured data into markdown with grouping by location."""
        
        # Special formatting for managed services and clusters - detailed format
        if resource_type == "managed_service":
            return self._format_managed_service_detailed(structured_data, total_count, context)
        elif resource_type == "k8s_cluster":
            return self._format_cluster_detailed(structured_data, total_count, context)
        
        # Group by location for other resource types
        by_location: Dict[str, list] = {}
        for item in structured_data:
            location = item.get("location", "Unknown")
            if location not in by_location:
                by_location[location] = []
            by_location[location].append(item)
        
        lines = [f"🔥 **Found {total_count} {resource_type}(s) across {len(by_location)} datacenter(s)**\n"]
        
        for location, items_list in sorted(by_location.items()):
            lines.append(f"\n### 📍 {location} ({len(items_list)})")
            lines.append("| Name | IP | Type |")
            lines.append("|------|-----|------|")
            
            for item in items_list:
                name = item.get("name", "Unknown")
                ip = item.get("ip", "N/A")
                fw_type = item.get("type", "Unknown")
                
                # Add type emoji
                if "Vayu" in fw_type or "IZO" in fw_type:
                    if "(N)" in fw_type or "EdgeGateway" in fw_type:
                        type_display = "🔵 Vayu Firewall(N)"
                    else:
                        type_display = "🟢 Vayu Firewall(F)"
                elif "Fortinet" in fw_type:
                    type_display = "🟠 Fortinet"
                else:
                    type_display = f"⚪ {fw_type}"
                
                lines.append(f"| {name} | {ip} | {type_display} |")
        
        lines.append(f"\n💡 **Tip:** Ask about a specific {resource_type} by name for more details.")
        
        return "\n".join(lines)
    
    def _format_managed_service_detailed(
        self,
        structured_data: list,
        total_count: int,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Format managed services with detailed information like the Jenkins example."""
        service_type = context.get("service_type", "") if context else ""
        service_display_name = context.get("service_display_name", "Managed Service") if context else "Managed Service"
        
        # Group by location for table display
        by_location: Dict[str, list] = {}
        for item in structured_data:
            location = item.get("location") or item.get("locationName") or item.get("endpointName") or "Unknown"
            if location not in by_location:
                by_location[location] = []
            by_location[location].append(item)
        
        lines = []
        
        # Summary line
        lines.append(f"✔ Found {total_count} {service_display_name} instance(s) across {len(by_location)} datacenter(s)\n")
        
        # Table with all instances grouped by location
        lines.append(f"\n**{service_display_name} Instances:**")
        lines.append("| Name | Status | Location | VIP | Protocol |")
        lines.append("|------|--------|----------|-----|----------|")
        
        for location, items_list in sorted(by_location.items()):
            for item in items_list:
                name = item.get("name") or item.get("serviceName") or "Unknown"
                status = item.get("status") or "N/A"
                location_name = item.get("location") or item.get("locationName") or location
                vip = item.get("vip") or item.get("ingressUrl") or item.get("url") or "N/A"
                protocol = item.get("protocol") or "-"
                
                # Format status with emoji
                status_display = f"✅ {status}" if status.lower() in ["active", "running"] else status
                
                # Format VIP as clickable link if it's a URL
                if vip and vip.startswith("http"):
                    vip_display = f"[{vip}]({vip})"
                else:
                    vip_display = vip or "N/A"
                
                lines.append(f"| {name} | {status_display} | {location_name} | {vip_display} | {protocol} |")
        
        # Key Identifying Fields section - show for each instance
        lines.append(f"\n**Key Identifying Fields:**")
        for item in structured_data:
            name = item.get("name") or item.get("serviceName") or "Unknown"
            
            # Extract key fields
            key_fields = []
            if item.get("serviceType"):
                key_fields.append(f"`serviceType: {item['serviceType']}`")
            if item.get("instanceNamespace"):
                key_fields.append(f"`instanceNamespace: {item['instanceNamespace']}`")
            if item.get("version"):
                key_fields.append(f"`version: {item['version']}`")
            if item.get("status"):
                key_fields.append(f"`status: {item['status']}`")
            if item.get("locationName") or item.get("location"):
                loc = item.get("locationName") or item.get("location")
                key_fields.append(f"`locationName: {loc}`")
            if item.get("clusterName"):
                key_fields.append(f"`clusterName: {item['clusterName']}`")
            if item.get("engagementName"):
                key_fields.append(f"`engagementName: {item['engagementName']}`")
            if item.get("departmentName"):
                key_fields.append(f"`departmentName: {item['departmentName']}`")
            
            if key_fields:
                lines.append(f"\n**{name}:**")
                lines.append("\n".join(f"- {field}" for field in key_fields))
        
        # Additional Information section - show for each instance
        lines.append(f"\n**Additional Information:**")
        for item in structured_data:
            name = item.get("name") or item.get("serviceName") or "Unknown"
            additional_info = []
            
            # Logs URL
            if item.get("logs") or item.get("logsUrl"):
                logs_url = item.get("logs") or item.get("logsUrl")
                if logs_url and isinstance(logs_url, str) and logs_url.startswith("http"):
                    additional_info.append(f"`logs: [{logs_url}]({logs_url})`")
                else:
                    additional_info.append(f"`logs: {logs_url}`")
            
            # Analytics URL
            if item.get("analyticsUrl") or item.get("monitoringUrl"):
                analytics_url = item.get("analyticsUrl") or item.get("monitoringUrl")
                if analytics_url and isinstance(analytics_url, str) and analytics_url.startswith("http"):
                    additional_info.append(f"`analyticsUrl: [{analytics_url}]({analytics_url})`")
                else:
                    additional_info.append(f"`analyticsUrl: {analytics_url}`")
            
            # Backup
            backup_val = item.get("backup") or item.get("backupEnabled") or item.get("isBackupEnabled")
            if backup_val is not None:
                backup_display = "true" if (isinstance(backup_val, bool) and backup_val) or (isinstance(backup_val, str) and backup_val.lower() in ["true", "yes", "enabled"]) else "false"
                additional_info.append(f"`backup: {backup_display}`")
            
            # Plugins
            plugins_val = item.get("plugins") or item.get("pluginList")
            if plugins_val:
                if isinstance(plugins_val, list):
                    plugins_display = ", ".join(str(p) for p in plugins_val)
                else:
                    plugins_display = str(plugins_val)
                additional_info.append(f"`plugins: {plugins_display}`")
            
            # Add any other fields that aren't in key fields
            excluded_keys = {"name", "serviceName", "status", "location", "locationName", "vip", "ingressUrl", "url", "protocol", 
                           "serviceType", "instanceNamespace", "version", "clusterName", "engagementName", "departmentName",
                           "logs", "logsUrl", "analyticsUrl", "monitoringUrl", "backup", "backupEnabled", "isBackupEnabled", 
                           "plugins", "pluginList"}
            
            for key, value in item.items():
                if key not in excluded_keys and value is not None and value != "N/A" and value != "":
                    if isinstance(value, (dict, list)):
                        continue  # Skip complex objects
                    additional_info.append(f"`{key}: {value}`")
            
            if additional_info:
                lines.append(f"\n**{name}:**")
                lines.append("\n".join(f"- {info}" for info in additional_info))
        
        lines.append(f"\n💡 **Tip:** Ask about a specific {service_display_name.lower()} instance by name for more details.")
        
        return "\n".join(lines)
    
    def _format_cluster_detailed(
        self,
        structured_data: list,
        total_count: int,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Format Kubernetes clusters with detailed information."""
        # Group by location
        by_location: Dict[str, list] = {}
        for item in structured_data:
            location = item.get("location") or item.get("displayNameEndpoint") or item.get("endpointName") or "Unknown"
            if location not in by_location:
                by_location[location] = []
            by_location[location].append(item)
        
        lines = []
        
        # Summary line
        lines.append(f"🚢 Found {total_count} Kubernetes Cluster(s) across {len(by_location)} datacenter(s)\n")
        
        # Table with all clusters grouped by location
        lines.append(f"\n**Kubernetes Clusters:**")
        lines.append("| Cluster Name | Status | K8s Version | Nodes | Control Plane | Datacenter |")
        lines.append("|--------------|--------|-------------|-------|---------------|------------|")
        
        for location, items_list in sorted(by_location.items()):
            for item in items_list:
                name = item.get("name") or item.get("clusterName") or "Unknown"
                status = item.get("status") or "N/A"
                k8s_version = item.get("k8sVersion") or item.get("kubernetesVersion") or "N/A"
                nodes = item.get("nodes") or item.get("nodescount") or item.get("nodesCount") or 0
                cluster_type = item.get("type") or item.get("clusterType") or "APP"
                location_name = item.get("location") or item.get("displayNameEndpoint") or location
                
                # Format status with emoji
                if isinstance(status, str):
                    status_lower = status.lower()
                    if status_lower in ["healthy", "running", "active"]:
                        status_display = f"✅ {status}"
                    elif status_lower in ["degraded", "warning"]:
                        status_display = f"⚠️ {status}"
                    elif status_lower in ["failed", "stopped", "error"]:
                        status_display = f"❌ {status}"
                    else:
                        status_display = status
                else:
                    status_display = str(status)
                
                # Format control plane type
                if isinstance(cluster_type, str):
                    type_upper = cluster_type.upper()
                    if type_upper == "MGMT":
                        type_display = "⚪ MGMT"
                    elif type_upper == "APP":
                        type_display = "⚪ APP"
                    else:
                        type_display = f"⚪ {cluster_type}"
                else:
                    type_display = f"⚪ {cluster_type}"
                
                lines.append(f"| {name} | {status_display} | {k8s_version} | {nodes} | {type_display} | {location_name} |")
        
        # Key Identifying Fields section - show for each cluster
        lines.append(f"\n**Key Identifying Fields:**")
        for item in structured_data:
            name = item.get("name") or item.get("clusterName") or "Unknown"
            
            # Extract key fields
            key_fields = []
            if item.get("clusterId") or item.get("id"):
                cluster_id = item.get("clusterId") or item.get("id")
                key_fields.append(f"`clusterId: {cluster_id}`")
            if item.get("name") or item.get("clusterName"):
                cluster_name = item.get("name") or item.get("clusterName")
                key_fields.append(f"`clusterName: {cluster_name}`")
            if item.get("status"):
                key_fields.append(f"`status: {item['status']}`")
            if item.get("location") or item.get("displayNameEndpoint"):
                loc = item.get("location") or item.get("displayNameEndpoint")
                key_fields.append(f"`locationName: {loc}`")
            if item.get("k8sVersion") or item.get("kubernetesVersion"):
                version = item.get("k8sVersion") or item.get("kubernetesVersion")
                key_fields.append(f"`kubernetesVersion: {version}`")
            if item.get("nodes") is not None or item.get("nodescount") is not None:
                nodes_count = item.get("nodes") or item.get("nodescount") or item.get("nodesCount") or 0
                key_fields.append(f"`nodescount: {nodes_count}`")
            if item.get("type") or item.get("clusterType"):
                cluster_type = item.get("type") or item.get("clusterType")
                key_fields.append(f"`type: {cluster_type}`")
            if item.get("ciMasterId") or item.get("masterId"):
                master_id = item.get("ciMasterId") or item.get("masterId")
                key_fields.append(f"`ciMasterId: {master_id}`")
            
            if key_fields:
                lines.append(f"\n**{name}:**")
                lines.append("\n".join(f"- {field}" for field in key_fields))
        
        # Additional Information section - show for each cluster
        lines.append(f"\n**Additional Information:**")
        for item in structured_data:
            name = item.get("name") or item.get("clusterName") or "Unknown"
            additional_info = []
            
            # Backup enabled
            backup_val = item.get("backupEnabled") or item.get("isIksBackupEnabled") or item.get("isBackupEnabled")
            if backup_val is not None:
                backup_display = "true" if (isinstance(backup_val, bool) and backup_val) or (isinstance(backup_val, str) and backup_val.lower() in ["true", "yes", "enabled"]) else "false"
                additional_info.append(f"`backupEnabled: {backup_display}`")
            
            # Created time
            if item.get("createdTime") or item.get("createdAt") or item.get("created"):
                created = item.get("createdTime") or item.get("createdAt") or item.get("created")
                additional_info.append(f"`createdTime: {created}`")
            
            # Add any other fields that aren't in key fields
            excluded_keys = {"name", "clusterName", "status", "location", "displayNameEndpoint", "endpointName", 
                           "k8sVersion", "kubernetesVersion", "nodes", "nodescount", "nodesCount", "type", 
                           "clusterType", "clusterId", "id", "ciMasterId", "masterId", "backupEnabled", 
                           "isIksBackupEnabled", "isBackupEnabled", "createdTime", "createdAt", "created"}
            
            for key, value in item.items():
                if key not in excluded_keys and value is not None and value != "N/A" and value != "":
                    if isinstance(value, (dict, list)):
                        continue  # Skip complex objects
                    additional_info.append(f"`{key}: {value}`")
            
            if additional_info:
                lines.append(f"\n**{name}:**")
                lines.append("\n".join(f"- {info}" for info in additional_info))
        
        lines.append(f"\n💡 **Tip:** Ask about a specific cluster by name for more details.")
        
        return "\n".join(lines)
    
    def _extract_json_from_response(self, response: str) -> Optional[Any]:
        """Extract JSON from LLM response (handles markdown code blocks and various formats)."""
        import re
        import json
        
        if not response or not response.strip():
            return None
            
        try:
            # Strategy 1: Try markdown code block with json
            json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            
            # Strategy 2: Try markdown code block without language
            json_match = re.search(r'```\s*(\[.*?\])\s*```', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            
            # Strategy 3: Try to find JSON array with balanced brackets (more robust)
            # Find the first [ and match until the last ]
            bracket_start = response.find('[')
            if bracket_start != -1:
                bracket_count = 0
                bracket_end = -1
                for i in range(bracket_start, len(response)):
                    if response[i] == '[':
                        bracket_count += 1
                    elif response[i] == ']':
                        bracket_count -= 1
                        if bracket_count == 0:
                            bracket_end = i
                            break
                
                if bracket_end != -1:
                    json_str = response[bracket_start:bracket_end + 1]
                    return json.loads(json_str)
            
            # Strategy 4: Try to find JSON array directly (simple regex)
            json_match = re.search(r'(\[.*?\])', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except:
                    pass
            
            # Strategy 5: Try direct JSON parse of entire response
            return json.loads(response.strip())
            
        except json.JSONDecodeError as e:
            logger.debug(f"JSON extraction failed: {e}. Response preview: {response[:200]}")
            return None
        except Exception as e:
            logger.debug(f"JSON extraction error: {e}")
            return None


# Global instance
llm_formatter = LLMFormatterService()

