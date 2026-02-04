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
    
    async def get_engagement_id(self, user_roles: List[str] = None, auth_token: str = None, user_id: str = None, user_type: str = None, selected_engagement_id: int = None) -> Optional[int]:
        """
        Common utility: Get engagement ID.
        Args:
            user_roles: User roles for permission checking
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
                user_roles=user_roles or [],
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

    async def maybe_get_engagement_id(self, operation: str, user_roles: List[str] = None, auth_token: str = None, user_id: str = None, user_type: str = None, selected_engagement_id: int = None) -> Optional[int]:
        """
        Engagement is required ONLY for mutating operations.
        LIST / READ operations must work without engagement.
        """
        if operation in ("list", "get", "describe"):
            return None
        return await self.get_engagement_id(user_roles=user_roles, auth_token=auth_token, user_id=user_id, user_type=user_type, selected_engagement_id=selected_engagement_id)
    
    async def get_datacenters(self, operation: str = "list", engagement_id: int = None, user_roles: List[str] = None, auth_token: str = None, user_id: str = None, user_type: str = None) -> List[Dict[str, Any]]:
        """
        Common utility: Get available datacenters.
        Args:
            engagement_id: Optional engagement ID (fetches if not provided)
            user_roles: User roles for permission checking
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
                    user_roles=user_roles,
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
                user_roles=user_roles or [],
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
2. Return ONLY a JSON array of matching indices (0-based)
3. Example output: [0, 2, 5] (means items at index 0, 2, and 5 match)
4. If no items match, return: []
5. If all items match, return all indices

**Output format (JSON array only):**"""

            response = await ai_service._call_chat_with_retries(
                prompt=prompt,
                max_tokens=500,
                temperature=0.1,
                timeout=10)
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

