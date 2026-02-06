"""
Base Resource Agent - Foundation for all specialized resource agents.
Provides LLM-powered intelligence for filtering, formatting, and analyzing API responses.
"""
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod
import logging
from app.services.llm_formatter_service import llm_formatter
logger = logging.getLogger(__name__)
from app.services.ai_service import ai_service
import os

class BaseResourceAgent(ABC):
    """
    Base class for resource-specific agents.
    Each resource agent handles operations for a specific domain (K8s, VMs, etc.)
    and uses LLM intelligence to:
    - Filter API responses based on user criteria
    - Format responses in natural language
    - Analyze data and provide insights
    - Handle complex queries with contextual understanding
    """
    def __init__(
        self,
        agent_name: str,
        agent_description: str,
        resource_type: str,
        temperature: float = 0.2
    ):
        """
        Initialize base resource agent.
        Args:
            agent_name: Name of the agent (e.g., "K8sClusterAgent")
            agent_description: Description of agent's purpose
            resource_type: Resource type this agent handles (e.g., "k8s_cluster")
            temperature: LLM temperature for response generation
        """
        self.agent_name = agent_name
        self.agent_description = agent_description
        self.temperature = temperature
        self.resource_type = resource_type
        logger.info(f"✅ Initialized {agent_name} for resource type: {resource_type}")
    
    @abstractmethod
    def get_supported_operations(self) -> List[str]:
        """
        Get list of operations this agent supports.
        Returns:
            List of operation names (e.g., ['list', 'create', 'delete'])
        """
        pass
    @abstractmethod
    async def execute_operation(self,operation: str,params: Dict[str, Any],context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a resource operation.
        Args:
            operation: Operation to perform (list, create, etc.)
            params: Parameters for the operation
            context: Execution context (user_id, session_id, etc.)
        Returns:
            Dict with operation result
        """
        pass
    
    async def get_engagement_id(self, auth_token: str = None, user_id: str = None, user_type: str = None, selected_engagement_id: int = None) -> Optional[int]:
        """
        Common utility: Get engagement ID.
        Args:
            auth_token: Bearer token from UI (Keycloak)
            user_id: User ID for session lookup
            user_type: User type (ENG/CUS) for selection logic
            selected_engagement_id: Pre-selected engagement ID from context
        Returns:
            Engagement ID or None if not found
        """
        from app.services.api_executor_service import api_executor_service
        
        # If we already have a selected engagement ID, use it
        if selected_engagement_id:
            logger.info(f"✅ Using pre-selected engagement ID: {selected_engagement_id}")
            return selected_engagement_id
        
        try:
            # Use api_executor_service.get_engagement_id which handles all the logic
            engagement_id = await api_executor_service.get_engagement_id(
                auth_token=auth_token,
                user_id=user_id,
                user_type=user_type
            )
            if engagement_id:
                return engagement_id
            
            # Fallback to execute_operation
            result = await api_executor_service.execute_operation(
                resource_type="engagement",
                operation="get",
                params={},
                user_type=user_type,
                auth_token=auth_token)
            if not result.get("success"):
                logger.error(f"Failed to fetch engagement ID: {result.get('error')}")
                return None
            engagement_data = result.get("data", {})
            # Handle nested response
            if isinstance(engagement_data, dict) and "data" in engagement_data:
                engagement_list = engagement_data.get("data", [])
                if isinstance(engagement_list, list) and len(engagement_list) > 0:
                    return engagement_list[0].get("id")
            return None
        except Exception as e:
            logger.error(f"Error fetching engagement ID: {str(e)}")
            return None

    async def maybe_get_engagement_id(self, operation: str, auth_token: str = None, user_id: str = None, user_type: str = None, selected_engagement_id: int = None) -> Optional[int]:
        """
        Engagement is required ONLY for mutating operations.
        LIST / READ operations must work without engagement.
        """
        if operation in ("list", "get", "describe"):
            return None
        return await self.get_engagement_id(auth_token=auth_token, user_id=user_id, user_type=user_type, selected_engagement_id=selected_engagement_id)
    
    async def get_datacenters(self, operation: str = "list", engagement_id: int = None, auth_token: str = None, user_id: str = None, user_type: str = None) -> List[Dict[str, Any]]:
        """
        Common utility: Get available datacenters.
        Args:
            engagement_id: Optional engagement ID (fetches if not provided)
            auth_token: Bearer token from UI (Keycloak)
            user_id: User ID for session lookup
            user_type: User type (ENG/CUS) for selection logic
        Returns:
            List of datacenter objects
        """
        from app.services.api_executor_service import api_executor_service
        try:
            if not engagement_id:
                engagement_id = await self.maybe_get_engagement_id(
                    operation=operation,
                    auth_token=auth_token,
                    user_id=user_id,
                    user_type=user_type)
            params = {}
            if engagement_id:
                params["engagement_id"] = engagement_id
            result = await api_executor_service.execute_operation(
                resource_type="endpoint",
                operation="list",
                params={"engagement_id": engagement_id},
                user_type=user_type,
                auth_token=auth_token)
            if not result.get("success"):
                logger.error(f"Failed to fetch datacenters: {result.get('error')}")
                return []
            datacenters = result.get("data", [])
            # Handle nested response
            if isinstance(datacenters, dict) and "data" in datacenters:
                datacenters = datacenters.get("data", [])
            return datacenters if isinstance(datacenters, list) else []
        except Exception as e:
            logger.error(f"Error fetching datacenters: {str(e)}")
            return []

    async def resolve_location_names(self,location_names: List[str],datacenters: List[Dict[str, Any]] = None) -> List[int]:
        """
        Common utility: Convert location names to endpoint IDs.
        Args:
            location_names: List of location names (e.g., ["Delhi", "Mumbai"])
            datacenters: Optional list of datacenters (fetches if not provided)
        Returns:
            List of endpoint IDs
        """
        if not location_names:
            # Return all endpoints if no locations specified
            if not datacenters:
                datacenters = await self.get_datacenters()
            return [dc.get("endpointId") for dc in datacenters if dc.get("endpointId")]
        if not datacenters:
            datacenters = await self.get_datacenters()
        endpoint_ids = []
        for loc_name in location_names:
            loc_lower = loc_name.lower()
            for dc in datacenters:
                dc_name = dc.get("endpointDisplayName", "").lower()
                if loc_lower in dc_name or dc_name in loc_lower:
                    endpoint_ids.append(dc.get("endpointId"))
                    logger.info(f"✅ Matched '{loc_name}' to endpoint {dc.get('endpointId')} ({dc.get('endpointDisplayName')})")
                    break
        return endpoint_ids
    
    async def format_response_with_llm(self,operation: str,raw_data: Any,user_query: str,context: Dict[str, Any] = None) -> str:
        """
        Use LLM to intelligently format the API response.
        Delegates to the centralized LLMFormatterService for consistent
        formatting across all resource types.
        Args:
            operation: Operation performed (list, create, etc.)
            raw_data: Raw API response data
            user_query: Original user query for context
            context: Additional context
        Returns:
            Natural language formatted response
        """
        return await llm_formatter.format_response(
            resource_type=self.resource_type,
            operation=operation,
            raw_data=raw_data,
            user_query=user_query,
            context=context)
    
    async def format_response_agentic(self, operation: str, raw_data: Any, user_query: str, context: Dict[str, Any] = None, chunk_size: int = 15) -> str:
        """
        Use agentic LLM formatting with validation to prevent hallucination.
        Automatically chunks large datasets and validates output.
        
        This method maintains agentic behavior (adapts to API structure changes)
        while preventing LLM hallucination through chunking and validation.
        
        Args:
            operation: Operation performed (list, create, etc.)
            raw_data: Raw API response data
            user_query: Original user query for context
            context: Additional context
            chunk_size: Items per chunk for large datasets (default: 15)
            
        Returns:
            Natural language formatted response with validated accuracy
        """
        return await llm_formatter.format_response_agentic(
            resource_type=self.resource_type,
            operation=operation,
            raw_data=raw_data,
            user_query=user_query,
            context=context,
            chunk_size=chunk_size
        )
    
    async def filter_with_llm(self,data: List[Dict[str, Any]],filter_criteria: str,user_query: str) -> List[Dict[str, Any]]:
        """
        Use LLM to intelligently filter data based on natural language criteria.
        
        Example:
            User: "Show me active clusters running version 1.28 in production"
            LLM filters: status==Active AND version==1.28 AND env==production
        
        Args:
            data: List of data items
            filter_criteria: Natural language filter criteria
            user_query: Original user query
        Returns:
            Filtered list of data items
        """
        if not filter_criteria or not data:
            return data
        try:
            prompt = f"""You are a data filtering assistant. Given the following data and filter criteria, return ONLY the indices of items that match the criteria.

**User's Query:** {user_query}

**Filter Criteria:** {filter_criteria}

**Data:**
```json
{data}
```

**Instructions:**
1. Analyze each item against the filter criteria
1.1. Ignore polite phrases like "for me/us" and any chat/meta wording.
1.2. Unless explicitly requested, ignore datacenter/endpoint/engagement scoping (those are handled elsewhere); focus on fields within each item (status/state, version, size, nodes, tags, etc.).
1.3. Status matching: treat "status" and "state" as equivalent; match values like running/active/healthy/failed/pending when present under either key (or close variants).
1.4. Numeric matching: if the criteria implies comparisons (e.g., "nodes > 3", "more than 2"), interpret common numeric fields (nodes, nodeCount, nodescount, cpu, memory, size) and apply the comparison when possible.
2. Return ONLY a JSON array of matching indices (0-based)
3. Example output: [0, 2, 5] (means items at index 0, 2, and 5 match)
4. If no items match, return: []
5. If all items match, return all indices

**Output format (JSON array only):**"""

            # Optional model override for filtering quality (e.g., openai/gpt-oss-120b)
            filter_model = os.getenv("FILTER_CHAT_MODEL")
            response = await ai_service._call_chat_with_retries(
                prompt=prompt,
                max_tokens=500,
                temperature=0.1,
                timeout=10,
                model=filter_model.strip() if isinstance(filter_model, str) and filter_model.strip() else None,
            )
            # Parse response
            import json
            import re
            # Extract JSON array from response
            match = re.search(r'\[[\d,\s]*\]', response)
            if match:
                indices = json.loads(match.group())
                return [data[i] for i in indices if 0 <= i < len(data)]
            # Fallback: return all data
            return data
        except Exception as e:
            logger.error(f"Error filtering with LLM: {str(e)}")
            return data  # Fallback: return all data

    # =========================================================================
    # Shared client-side (post-fetch) filtering for ALL list operations
    # =========================================================================
    def _should_apply_client_side_llm_filter(self, user_query: str, params: Optional[Dict[str, Any]] = None) -> bool:
        """
        Lightweight gate to avoid extra LLM calls on plain 'list/show' requests.
        We still use the LLM to do the actual filtering when this returns True.
        """
        if not user_query or not isinstance(user_query, str):
            return False

        q = user_query.lower()

        # Common signals that user is applying constraints on the returned data
        filter_signals = [
            "with", "where", "having", "only", "except", "excluding", "include", "including",
            "status", "state", "health", "healthy", "unhealthy",
            "running", "active", "inactive", "failed", "pending", "stopped",
            "version", "k8s", "kubernetes", "node", "nodes",
            "greater than", "less than", "at least", "at most",
        ]
        if any(s in q for s in filter_signals):
            return True

        if any(op in q for op in [">", "<", ">=", "<=", "==", "!="]):
            return True

        # If upstream intent extraction already provided extra params (beyond endpoints paging),
        # treat that as a filter intent.
        if params:
            ignore = {"endpoints", "endpoint_names", "endpoint_ids", "businessUnits", "environments", "zones", "page", "size", "limit", "offset"}
            for k, v in params.items():
                if k in ignore:
                    continue
                if v is None or v == "" or v == [] or v == {}:
                    continue
                return True

        return False    def _build_client_side_filter_criteria(self, user_query: str, params: Optional[Dict[str, Any]] = None) -> str:
        """
        Build a concise, model-friendly criteria string.
        Uses extracted params (state/status/version/etc.) when available, plus the raw user query.
        """
        q = (user_query or "").strip()
        # Strip common conversational references to prior messages (we handle "reuse previous data" elsewhere)
        for phrase in [
            "from above response",
            "from above",
            "above response",
            "previous response",
            "from the above",
            "from the previous",
            "from earlier",
        ]:
            q = q.replace(phrase, "")
        q = " ".join(q.split())

        parts: List[str] = []
        if params:
            ignore = {
                "endpoints", "endpoint_names", "endpoint_ids",
                "businessUnits", "environments", "zones",
                "page", "size", "limit", "offset",
                "force_refresh",
            }
            for k, v in params.items():
                if k in ignore:
                    continue
                if v is None or v == "" or v == [] or v == {}:
                    continue
                parts.append(f"{k}={v}")

        if parts and q:
            return f"{', '.join(parts)}; query={q}"
        if parts:
            return ", ".join(parts)
        return q

    async def apply_client_side_llm_filter(
        self,
        items: List[Dict[str, Any]],
        user_query: str,
        params: Optional[Dict[str, Any]] = None,
        chunk_size: int = 50,
    ) -> Dict[str, Any]:
        """
        Apply LLM filtering to already-fetched list data based on the user's query.

        Returns a dict:
          {
            "items": <filtered_items>,
            "filter_applied": bool,
            "original_count": int,
            "filtered_count": int
          }
        """
        original_count = len(items or [])
        if not items or not user_query:
            return {
                "items": items,
                "filter_applied": False,
                "original_count": original_count,
                "filtered_count": original_count,
            }

        if not self._should_apply_client_side_llm_filter(user_query=user_query, params=params):
            return {
                "items": items,
                "filter_applied": False,
                "original_count": original_count,
                "filtered_count": original_count,
            }

        # Build criteria from extracted params + user query.
        criteria = self._build_client_side_filter_criteria(user_query=user_query, params=params)

        # Chunk large datasets to avoid huge prompts/timeouts
        if len(items) <= chunk_size:
            filtered = await self.filter_with_llm(items, criteria, user_query)
        else:
            filtered = []
            for i in range(0, len(items), chunk_size):
                chunk = items[i : i + chunk_size]
                filtered_chunk = await self.filter_with_llm(chunk, criteria, user_query)
                filtered.extend(filtered_chunk)

        return {
            "items": filtered,
            "filter_applied": True,
            "original_count": original_count,
            "filtered_count": len(filtered),
        }
