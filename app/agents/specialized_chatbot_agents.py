"""
Specialized Chatbot Agents for Different Domains
Each agent handles specific types of queries and tasks
"""

from typing import List, Dict, Any, Optional
from app.agents.chatbot_base_agent import ChatbotBaseAgent, AgentRole, AgentMessage, MessageType
from app.agents.chatbot_tools import chatbot_tool_registry
import logging
import json

logger = logging.getLogger(__name__)


class SearchChatbotAgent(ChatbotBaseAgent):
    """Agent for searching and retrieving information"""
    
    def __init__(self):
        super().__init__(
            name="SearchAgent",
            role=AgentRole.SEARCH,
            description="Searches knowledge base and documentation"
        )
        self.add_tools(chatbot_tool_registry.get_tools_by_category("search"))
    
    async def process_message(self, message: AgentMessage) -> AgentMessage:
        """Process search queries"""
        await super().process_message(message)
        
        try:
            query = message.content.get("query", "")
            search_type = message.content.get("search_type", "knowledge_base")
            
            if search_type == "kubernetes":
                result = await self.execute_tool(
                    "search_kubernetes_docs",
                    {"query": query, "topic": message.content.get("topic", "services")}
                )
            else:
                result = await self.execute_tool(
                    "search_knowledge_base",
                    {"query": query, "max_results": message.content.get("max_results", 5)}
                )
            
            response = AgentMessage(
                sender=self.name,
                receiver=message.sender,
                message_type=MessageType.RESPONSE,
                content={
                    "status": "success",
                    "search_results": result,
                    "query": query
                },
                parent_id=message.id
            )
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            response = AgentMessage(
                sender=self.name,
                receiver=message.sender,
                message_type=MessageType.ERROR,
                content={"error": str(e)},
                parent_id=message.id
            )
        
        self.message_history.append(response)
        return response


class InfrastructureChatbotAgent(ChatbotBaseAgent):
    """Agent for infrastructure management and automation"""
    
    def __init__(self):
        super().__init__(
            name="InfrastructureAgent",
            role=AgentRole.INFRASTRUCTURE,
            description="Manages Kubernetes services and infrastructure"
        )
        self.add_tools(chatbot_tool_registry.get_tools_by_category("infrastructure"))
    
    async def process_message(self, message: AgentMessage) -> AgentMessage:
        """Process infrastructure commands"""
        await super().process_message(message)
        
        try:
            action = message.content.get("action", "")
            service_name = message.content.get("service_name", "")
            
            if action == "enable":
                result = await self.execute_tool(
                    "enable_k8s_service",
                    {"service_name": service_name, "config": message.content.get("config")}
                )
            elif action == "disable":
                result = await self.execute_tool(
                    "disable_k8s_service",
                    {"service_name": service_name}
                )
            elif action == "status":
                result = await self.execute_tool(
                    "get_service_status",
                    {"service_name": service_name}
                )
            elif action == "list":
                result = await self.execute_tool(
                    "get_k8s_services",
                    {}
                )
            else:
                raise ValueError(f"Unknown action: {action}")
            
            response = AgentMessage(
                sender=self.name,
                receiver=message.sender,
                message_type=MessageType.RESPONSE,
                content={
                    "status": "success",
                    "action": action,
                    "result": result,
                    "service": service_name
                },
                parent_id=message.id
            )
        except Exception as e:
            logger.error(f"Infrastructure operation failed: {str(e)}")
            response = AgentMessage(
                sender=self.name,
                receiver=message.sender,
                message_type=MessageType.ERROR,
                content={"error": str(e)},
                parent_id=message.id
            )
        
        self.message_history.append(response)
        return response


class APIChatbotAgent(ChatbotBaseAgent):
    """Agent for API calls and external integrations"""
    
    def __init__(self):
        super().__init__(
            name="APIAgent",
            role=AgentRole.API,
            description="Handles external API calls and integrations"
        )
        self.add_tools(chatbot_tool_registry.get_tools_by_category("api"))
    
    async def process_message(self, message: AgentMessage) -> AgentMessage:
        """Process API requests"""
        await super().process_message(message)
        
        try:
            url = message.content.get("url", "")
            method = message.content.get("method", "GET")
            api_key = message.content.get("api_key")
            
            # Validate API key if provided
            if api_key:
                validation = await self.execute_tool(
                    "validate_api_key",
                    {"api_key": api_key, "service": message.content.get("service", "unknown")}
                )
                if not validation.get("is_valid"):
                    raise ValueError("Invalid API key")
            
            result = await self.execute_tool(
                "call_external_api",
                {
                    "url": url,
                    "method": method,
                    "headers": message.content.get("headers"),
                    "payload": message.content.get("payload"),
                    "api_key": api_key
                }
            )
            
            response = AgentMessage(
                sender=self.name,
                receiver=message.sender,
                message_type=MessageType.RESPONSE,
                content={
                    "status": "success",
                    "api_response": result,
                    "url": url
                },
                parent_id=message.id
            )
        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            response = AgentMessage(
                sender=self.name,
                receiver=message.sender,
                message_type=MessageType.ERROR,
                content={"error": str(e)},
                parent_id=message.id
            )
        
        self.message_history.append(response)
        return response


class DataChatbotAgent(ChatbotBaseAgent):
    """Agent for data retrieval and monitoring"""
    
    def __init__(self):
        super().__init__(
            name="DataAgent",
            role=AgentRole.DATA,
            description="Retrieves logs, metrics, and monitoring data"
        )
        self.add_tools(chatbot_tool_registry.get_tools_by_category("data"))
    
    async def process_message(self, message: AgentMessage) -> AgentMessage:
        """Process data retrieval requests"""
        await super().process_message(message)
        
        try:
            data_type = message.content.get("data_type", "")
            service_name = message.content.get("service_name", "")
            
            if data_type == "logs":
                result = await self.execute_tool(
                    "get_service_logs",
                    {"service_name": service_name, "lines": message.content.get("lines", 50)}
                )
            elif data_type == "metrics":
                result = await self.execute_tool(
                    "get_service_metrics",
                    {"service_name": service_name}
                )
            else:
                raise ValueError(f"Unknown data type: {data_type}")
            
            response = AgentMessage(
                sender=self.name,
                receiver=message.sender,
                message_type=MessageType.RESPONSE,
                content={
                    "status": "success",
                    "data_type": data_type,
                    "data": result,
                    "service": service_name
                },
                parent_id=message.id
            )
        except Exception as e:
            logger.error(f"Data retrieval failed: {str(e)}")
            response = AgentMessage(
                sender=self.name,
                receiver=message.sender,
                message_type=MessageType.ERROR,
                content={"error": str(e)},
                parent_id=message.id
            )
        
        self.message_history.append(response)
        return response


class AnalysisChatbotAgent(ChatbotBaseAgent):
    """Agent for analysis and recommendations"""
    
    def __init__(self):
        super().__init__(
            name="AnalysisAgent",
            role=AgentRole.ANALYSIS,
            description="Analyzes service health and generates recommendations"
        )
        self.add_tools(chatbot_tool_registry.get_tools_by_category("analysis"))
    
    async def process_message(self, message: AgentMessage) -> AgentMessage:
        """Process analysis requests"""
        await super().process_message(message)
        
        try:
            analysis_type = message.content.get("analysis_type", "health")
            service_name = message.content.get("service_name", "")
            
            if analysis_type == "health":
                result = await self.execute_tool(
                    "analyze_service_health",
                    {"service_name": service_name}
                )
            elif analysis_type == "recommendations":
                result = await self.execute_tool(
                    "generate_recommendations",
                    {"service_name": service_name}
                )
            else:
                raise ValueError(f"Unknown analysis type: {analysis_type}")
            
            response = AgentMessage(
                sender=self.name,
                receiver=message.sender,
                message_type=MessageType.RESPONSE,
                content={
                    "status": "success",
                    "analysis_type": analysis_type,
                    "analysis": result,
                    "service": service_name
                },
                parent_id=message.id
            )
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            response = AgentMessage(
                sender=self.name,
                receiver=message.sender,
                message_type=MessageType.ERROR,
                content={"error": str(e)},
                parent_id=message.id
            )
        
        self.message_history.append(response)
        return response
