"""
Production-grade tool definitions for LangChain agents.
Defines all available tools that agents can call.
"""

from typing import Any, Dict, List, Optional, Callable
from pydantic import BaseModel, Field
from enum import Enum
import logging


logger = logging.getLogger(__name__)


class ToolCategory(str, Enum):
    """Tool categories for organization and filtering"""
    SEARCH = "search"
    DATA_RETRIEVAL = "data_retrieval"
    API_CALL = "api_call"
    ANALYSIS = "analysis"
    KNOWLEDGE_BASE = "knowledge_base"
    EXTERNAL_SERVICE = "external_service"


class ToolParameter(BaseModel):
    """Tool parameter definition"""
    name: str
    type: str
    description: str
    required: bool = True
    default: Optional[Any] = None
    enum_values: Optional[List[str]] = None


class ToolDefinition(BaseModel):
    """Complete tool definition for agent use"""
    name: str
    description: str
    category: ToolCategory
    parameters: List[ToolParameter]
    return_type: str
    timeout_seconds: int = 30
    retry_count: int = 3
    requires_auth: bool = False
    rate_limit_per_minute: Optional[int] = None
    
    def to_langchain_format(self) -> Dict[str, Any]:
        """Convert to LangChain tool format"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    param.name: {
                        "type": param.type,
                        "description": param.description,
                        "enum": param.enum_values
                    } if param.enum_values else {
                        "type": param.type,
                        "description": param.description
                    }
                    for param in self.parameters
                },
                "required": [p.name for p in self.parameters if p.required]
            }
        }


# ==================== SEARCH TOOLS ====================

SEARCH_KNOWLEDGE_BASE = ToolDefinition(
    name="search_knowledge_base",
    description="Search the RAG knowledge base for relevant documents and information",
    category=ToolCategory.KNOWLEDGE_BASE,
    parameters=[
        ToolParameter(
            name="query",
            type="string",
            description="Search query to find relevant documents",
            required=True
        ),
        ToolParameter(
            name="max_results",
            type="integer",
            description="Maximum number of results to return (1-100)",
            required=False,
            default=10
        ),
        ToolParameter(
            name="search_depth",
            type="string",
            description="Search depth level",
            required=False,
            default="balanced",
            enum_values=["quick", "balanced", "deep"]
        ),
    ],
    return_type="List[Dict[str, Any]]",
    timeout_seconds=15,
    retry_count=2)

SEARCH_WEB = ToolDefinition(
    name="search_web",
    description="Search the web for current information and external data",
    category=ToolCategory.SEARCH,
    parameters=[
        ToolParameter(
            name="query",
            type="string",
            description="Web search query",
            required=True
        ),
        ToolParameter(
            name="num_results",
            type="integer",
            description="Number of results to return (1-20)",
            required=False,
            default=5
        ),
    ],
    return_type="List[Dict[str, str]]",
    timeout_seconds=20,
    retry_count=2,
    rate_limit_per_minute=30)

# ==================== DATA RETRIEVAL TOOLS ====================

GET_USER_DATA = ToolDefinition(
    name="get_user_data",
    description="Retrieve user profile and account information",
    category=ToolCategory.DATA_RETRIEVAL,
    parameters=[
        ToolParameter(
            name="user_id",
            type="string",
            description="User ID to retrieve data for",
            required=True
        ),
        ToolParameter(
            name="data_type",
            type="string",
            description="Type of user data to retrieve",
            required=False,
            default="profile",
            enum_values=["profile", "history", "preferences", "all"]
        ),
    ],
    return_type="Dict[str, Any]",
    timeout_seconds=10,
    retry_count=2,
    requires_auth=True)

GET_DOCUMENT = ToolDefinition(
    name="get_document",
    description="Retrieve a specific document from the knowledge base",
    category=ToolCategory.DATA_RETRIEVAL,
    parameters=[
        ToolParameter(
            name="document_id",
            type="string",
            description="ID of the document to retrieve",
            required=True
        ),
    ],
    return_type="Dict[str, Any]",
    timeout_seconds=10,
    retry_count=2)

# ==================== API CALL TOOLS ====================

CALL_EXTERNAL_API = ToolDefinition(
    name="call_external_api",
    description="Make HTTP requests to external APIs",
    category=ToolCategory.API_CALL,
    parameters=[
        ToolParameter(
            name="url",
            type="string",
            description="API endpoint URL",
            required=True
        ),
        ToolParameter(
            name="method",
            type="string",
            description="HTTP method",
            required=False,
            default="GET",
            enum_values=["GET", "POST", "PUT", "DELETE", "PATCH"]
        ),
        ToolParameter(
            name="payload",
            type="object",
            description="Request payload for POST/PUT/PATCH",
            required=False
        ),
        ToolParameter(
            name="headers",
            type="object",
            description="Custom HTTP headers",
            required=False
        ),
    ],
    return_type="Dict[str, Any]",
    timeout_seconds=30,
    retry_count=3,
    requires_auth=True)

# ==================== ANALYSIS TOOLS ====================

ANALYZE_SENTIMENT = ToolDefinition(
    name="analyze_sentiment",
    description="Analyze sentiment of text content",
    category=ToolCategory.ANALYSIS,
    parameters=[
        ToolParameter(
            name="text",
            type="string",
            description="Text to analyze",
            required=True
        ),
    ],
    return_type="Dict[str, Any]",
    timeout_seconds=10,
    retry_count=2)

EXTRACT_ENTITIES = ToolDefinition(
    name="extract_entities",
    description="Extract named entities from text",
    category=ToolCategory.ANALYSIS,
    parameters=[
        ToolParameter(
            name="text",
            type="string",
            description="Text to extract entities from",
            required=True
        ),
        ToolParameter(
            name="entity_types",
            type="array",
            description="Types of entities to extract",
            required=False,
            default=["PERSON", "ORG", "LOCATION", "DATE"]
        ),
    ],
    return_type="List[Dict[str, Any]]",
    timeout_seconds=10,
    retry_count=2)

SUMMARIZE_TEXT = ToolDefinition(
    name="summarize_text",
    description="Generate a summary of provided text",
    category=ToolCategory.ANALYSIS,
    parameters=[
        ToolParameter(
            name="text",
            type="string",
            description="Text to summarize",
            required=True
        ),
        ToolParameter(
            name="max_length",
            type="integer",
            description="Maximum length of summary in characters",
            required=False,
            default=500
        ),
    ],
    return_type="str",
    timeout_seconds=15,
    retry_count=2)

# ==================== TOOL REGISTRY ====================

class ToolRegistry:
    """Central registry for all available tools"""
    
    def __init__(self):
        self.tools: Dict[str, ToolDefinition] = {}
        self.tool_handlers: Dict[str, Callable] = {}
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register all default tools"""
        default_tools = [
            SEARCH_KNOWLEDGE_BASE,
            SEARCH_WEB,
            GET_USER_DATA,
            GET_DOCUMENT,
            CALL_EXTERNAL_API,
            ANALYZE_SENTIMENT,
            EXTRACT_ENTITIES,
            SUMMARIZE_TEXT,
        ]
        
        for tool in default_tools:
            self.register_tool(tool)
    
    def register_tool(self, tool: ToolDefinition, handler: Optional[Callable] = None):
        """Register a new tool"""
        self.tools[tool.name] = tool
        if handler:
            self.tool_handlers[tool.name] = handler
        logger.info(f"Registered tool: {tool.name} (category: {tool.category})")
    
    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Get tool definition by name"""
        return self.tools.get(name)
    
    def get_tools_by_category(self, category: ToolCategory) -> List[ToolDefinition]:
        """Get all tools in a category"""
        return [t for t in self.tools.values() if t.category == category]
    
    def get_all_tools(self) -> List[ToolDefinition]:
        """Get all registered tools"""
        return list(self.tools.values())
    
    def get_tool_handler(self, name: str) -> Optional[Callable]:
        """Get handler function for a tool"""
        return self.tool_handlers.get(name)
    
    def set_tool_handler(self, name: str, handler: Callable):
        """Set handler for a tool"""
        if name not in self.tools:
            logger.warning(f"Tool {name} not registered")
            return
        self.tool_handlers[name] = handler
        logger.info(f"Set handler for tool: {name}")
    
    def to_langchain_format(self) -> List[Dict[str, Any]]:
        """Convert all tools to LangChain format"""
        return [tool.to_langchain_format() for tool in self.tools.values()]
    
# Global registry instance
tool_registry = ToolRegistry()