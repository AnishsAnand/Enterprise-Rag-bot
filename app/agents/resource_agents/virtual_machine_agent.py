"""
Virtual Machine Agent - Handles VM/Instance operations.
"""

from typing import Any, Dict, List
import logging

from app.agents.resource_agents.base_resource_agent import BaseResourceAgent
from app.services.api_executor_service import api_executor_service

logger = logging.getLogger(__name__)


class VirtualMachineAgent(BaseResourceAgent):
    """Agent for virtual machine operations."""
    
    def __init__(self):
        super().__init__(
            agent_name="VirtualMachineAgent",
            agent_description="Specialized agent for VM/Instance operations with intelligent filtering",
            resource_type="vm",
            temperature=0.2
        )
    
    def get_supported_operations(self) -> List[str]:
        return ["list", "create", "stop", "start", "delete"]
    
    async def execute_operation(
        self,
        operation: str,
        params: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            logger.info(f"🖥️ VirtualMachineAgent executing: {operation}")
            
            if operation == "list":
                return await self._list_vms(params, context)
            else:
                return {
                    "success": False,
                    "response": f"Operation '{operation}' for VMs is not yet implemented."
                }
        except Exception as e:
            logger.error(f"❌ VirtualMachineAgent error: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e), "response": f"Error: {str(e)}"}
    
    async def _list_vms(self, params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """List VMs with intelligent filtering."""
        try:
            # Handle both parameter naming conventions
            # ValidationAgent might pass "endpoints" while schema expects "endpoint_filter"
            endpoint_ids = params.get("endpoints") or params.get("endpoint_ids") or []
            endpoint_names = params.get("endpoint_names") or []
            endpoint_filter = params.get("endpoint_filter") or params.get("endpoint")
            zone_filter = params.get("zone_filter") or params.get("zone")
            department_filter = params.get("department_filter") or params.get("department")
            
            # INTELLIGENT MAPPING: If we have endpoint_ids but no endpoint_filter, 
            # AND endpoint is a short abbreviation, use the full endpoint name
            if endpoint_ids and not endpoint_filter and endpoint_names:
                endpoint_filter = endpoint_names[0] if len(endpoint_names) == 1 else None
                logger.info(f"🔍 Using endpoint filter from endpoint_names: {endpoint_filter}")
            elif endpoint_filter and not endpoint_names:
                # endpoint_filter is a short abbreviation like "blr" from IntentAgent
                # Try to map it to full name using ValidationAgent's logic
                logger.info(f"🔍 Got abbreviation '{endpoint_filter}', will fetch full endpoint name")
                
                # Map common abbreviations
                abbrev_map = {
                    "blr": "Bengaluru",
                    "del": "Delhi", 
                    "mum": "Mumbai-BKC",
                    "chennai": "Chennai-AMB",
                    "singapore": "Singapore East",
                    "sg": "Singapore East"
                }
                
                mapped_name = abbrev_map.get(endpoint_filter.lower())
                if mapped_name:
                    endpoint_filter = mapped_name
                    logger.info(f"✅ Mapped abbreviation to full name: {endpoint_filter}")
            
            logger.info(f"🔍 Listing VMs with filters: endpoint={endpoint_filter}, zone={zone_filter}, dept={department_filter}")
            
            result = await api_executor_service.list_vms(
                endpoint_filter=endpoint_filter,
                zone_filter=zone_filter,
                department_filter=department_filter
            )
            
            if not result.get("success"):
                return {
                    "success": False,
                    "error": result.get("error"),
                    "response": f"Failed to list VMs: {result.get('error')}"
                }
            
            vms = result.get("data", [])
            logger.info(f"✅ Found {len(vms)} VMs")
            
            # Format with LLM - provide better context
            user_query = context.get("user_query", "")
            
            # Create a more VM-friendly formatted response
            formatted_response = await self._format_vm_response(vms, user_query, endpoint_names)
            
            return {
                "success": True,
                "data": vms,
                "response": formatted_response,
                "metadata": {"count": len(vms), "endpoints": endpoint_names}
            }
        except Exception as e:
            logger.error(f"❌ Error listing VMs: {str(e)}", exc_info=True)
            raise
    
    async def _format_vm_response(self, vms: List[Dict[str, Any]], user_query: str, endpoint_names: List[str]) -> str:
        """Format VM data using LLM with VM-specific context."""
        try:
            from app.services.ai_service import ai_service
            
            # Simplify VM data for LLM (the nested structure is too complex)
            simplified_vms = []
            for vm_item in vms:
                vm = vm_item.get("virtualMachine", {})
                simplified_vms.append({
                    "vmName": vm.get("vmName", "N/A"),
                    "vmuuid": vm.get("vmuuid", "N/A"),
                    "endpoint": vm.get("endpoint", {}).get("endpointName", "N/A"),
                    "engagement": vm.get("engagement", {}).get("engagementName", "N/A"),
                    "vmId": vm.get("vmId", "N/A"),
                    "storage": vm.get("storage", 0),
                    "isPpuMeteringEnabled": vm.get("isPpuMeteringEnabled", "no"),
                    "isBudgetingEnabled": vm.get("isBudgetingEnabled", "no"),
                    "tags": vm.get("tags", []),
                    "vmAttributes": vm.get("vmAttributes", {})
                })
            
            locations_str = ", ".join(endpoint_names) if endpoint_names else "all locations"
            
            prompt = f"""You are a cloud infrastructure assistant. Format this VM list for the user.

**User's Query:** {user_query}

**VMs Found:** {len(simplified_vms)} vm(s) in {locations_str}

**VM Data:**
```json
{simplified_vms[:50]}
```

**Instructions:**
1. Start with a summary: "Found {len(simplified_vms)} vm(s)" with location info
2. Present VMs in a clean format - you can use tables or lists
3. Key fields to show: vmName, endpoint, storage, engagement
4. Use emojis for visual clarity
5. Keep it concise and readable
6. If there are many VMs, show the most important ones and mention the total count

Format as markdown with tables or lists. Be helpful and conversational."""
            
            response = await ai_service._call_chat_with_retries(
                prompt=prompt,
                max_tokens=2000,
                temperature=0.3,
                timeout=15
            )
            
            return response if response else f"Found {len(vms)} VM(s) in {locations_str}"
            
        except Exception as e:
            logger.error(f"Error formatting VM response: {str(e)}")
            return f"Found {len(vms)} VM(s) across {', '.join(endpoint_names) if endpoint_names else 'all endpoints'}"

