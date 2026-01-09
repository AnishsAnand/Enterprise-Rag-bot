from typing import List, Dict, Any, Optional
from app.agents.chatbot_base_agent import ChatbotBaseAgent, AgentRole, AgentMessage, MessageType
from app.agents.specialized_chatbot_agents import (
    SearchChatbotAgent,
    InfrastructureChatbotAgent,
    APIChatbotAgent,
    DataChatbotAgent,
    AnalysisChatbotAgent
)
import logging
import json
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)


class CaptainChatbotAgent(ChatbotBaseAgent):
    """
    Captain Agent - Main orchestrator for all specialized agents
    Routes queries to appropriate agents and coordinates execution
    """
    
    def __init__(self):
        super().__init__(
            name="CaptainAgent",
            role=AgentRole.CAPTAIN,
            description="Main orchestrator that routes queries to specialized agents"
        )
        
        # Initialize all specialized agents
        self.search_agent = SearchChatbotAgent()
        self.infrastructure_agent = InfrastructureChatbotAgent()
        self.api_agent = APIChatbotAgent()
        self.data_agent = DataChatbotAgent()
        self.analysis_agent = AnalysisChatbotAgent()
        
        self.agents: Dict[str, ChatbotBaseAgent] = {
            "search": self.search_agent,
            "infrastructure": self.infrastructure_agent,
            "api": self.api_agent,
            "data": self.data_agent,
            "analysis": self.analysis_agent
        }
        
        self.conversation_history: List[Dict[str, Any]] = []
        self.task_queue: List[Dict[str, Any]] = []
        
        logger.info("Captain Agent initialized with 5 specialized agents")
    
    async def process_user_query(self, user_query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Main entry point for processing user queries
        Analyzes intent and routes to appropriate agents
        """
        try:
            logger.info(f"Captain Agent processing query: {user_query}")
            
            # Analyze query intent
            intent = await self._analyze_intent(user_query)
            logger.info(f"Detected intent: {intent}")
            
            # Route to appropriate agent(s)
            agent_responses = await self._route_to_agents(user_query, intent, context)
            
            # Aggregate and format response
            final_response = await self._aggregate_responses(user_query, intent, agent_responses)
            
            # Store in conversation history
            self.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "user_query": user_query,
                "intent": intent,
                "response": final_response
            })
            
            return final_response
        
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "message": "Failed to process your query. Please try again."
            }
    
    async def _analyze_intent(self, query: str) -> Dict[str, Any]:
        """
        Analyze user query to determine intent and required agents
        """
        query_lower = query.lower()
        
        intent = {
            "primary": None,
            "secondary": [],
            "requires_search": False,
            "requires_infrastructure": False,
            "requires_api": False,
            "requires_data": False,
            "requires_analysis": False,
            "action": None,
            "target": None
        }
        
        # Detect infrastructure operations
        if any(word in query_lower for word in ["enable", "disable", "start", "stop", "restart"]):
            intent["primary"] = "infrastructure"
            intent["requires_infrastructure"] = True
            
            if "enable" in query_lower or "start" in query_lower:
                intent["action"] = "enable"
            elif "disable" in query_lower or "stop" in query_lower:
                intent["action"] = "disable"
            
            # Extract service name
            for word in query.split():
                if any(service in word.lower() for service in ["service", "k8s", "kubernetes", "deployment"]):
                    intent["target"] = word
        
        # Detect search queries
        if any(word in query_lower for word in ["search", "find", "look for", "what is", "how to", "tell me about"]):
            intent["requires_search"] = True
            if not intent["primary"]:
                intent["primary"] = "search"
        
        # Detect API operations
        if any(word in query_lower for word in ["api", "call", "request", "fetch", "post", "get"]):
            intent["requires_api"] = True
            if not intent["primary"]:
                intent["primary"] = "api"
        
        # Detect data/monitoring queries
        if any(word in query_lower for word in ["logs", "metrics", "status", "health", "monitor", "check"]):
            intent["requires_data"] = True
            if not intent["primary"]:
                intent["primary"] = "data"
        
        # Detect analysis requests
        if any(word in query_lower for word in ["analyze", "recommend", "suggest", "improve", "optimize"]):
            intent["requires_analysis"] = True
            if not intent["primary"]:
                intent["primary"] = "analysis"
        
        # Default to search if no specific intent detected
        if not intent["primary"]:
            intent["primary"] = "search"
            intent["requires_search"] = True
        
        return intent
    
    async def _route_to_agents(
        self,
        query: str,
        intent: Dict[str, Any],
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Route query to appropriate agents based on intent
        """
        agent_responses = {}
        tasks = []
        
        # Create messages for each required agent
        if intent["requires_search"]:
            msg = AgentMessage(
                sender=self.name,
                receiver="SearchAgent",
                message_type=MessageType.QUERY,
                content={
                    "query": query,
                    "search_type": "kubernetes" if "kubernetes" in query.lower() else "knowledge_base",
                    "max_results": 5
                }
            )
            tasks.append(("search", self.search_agent.process_message(msg)))
        
        if intent["requires_infrastructure"]:
            msg = AgentMessage(
                sender=self.name,
                receiver="InfrastructureAgent",
                message_type=MessageType.TASK,
                content={
                    "action": intent.get("action", "status"),
                    "service_name": intent.get("target", ""),
                    "query": query
                }
            )
            tasks.append(("infrastructure", self.infrastructure_agent.process_message(msg)))
        
        if intent["requires_api"]:
            msg = AgentMessage(
                sender=self.name,
                receiver="APIAgent",
                message_type=MessageType.QUERY,
                content={
                    "url": context.get("url") if context else "",
                    "method": context.get("method", "GET") if context else "GET",
                    "api_key": context.get("api_key") if context else None,
                    "query": query
                }
            )
            tasks.append(("api", self.api_agent.process_message(msg)))
        
        if intent["requires_data"]:
            msg = AgentMessage(
                sender=self.name,
                receiver="DataAgent",
                message_type=MessageType.QUERY,
                content={
                    "data_type": "metrics" if "metrics" in query.lower() else "logs",
                    "service_name": intent.get("target", ""),
                    "query": query
                }
            )
            tasks.append(("data", self.data_agent.process_message(msg)))
        
        if intent["requires_analysis"]:
            msg = AgentMessage(
                sender=self.name,
                receiver="AnalysisAgent",
                message_type=MessageType.QUERY,
                content={
                    "analysis_type": "recommendations" if "recommend" in query.lower() else "health",
                    "service_name": intent.get("target", ""),
                    "query": query
                }
            )
            tasks.append(("analysis", self.analysis_agent.process_message(msg)))
        
        # Execute all tasks concurrently
        if tasks:
            results = await asyncio.gather(*[task[1] for task in tasks], return_exceptions=True)
            for (agent_name, _), result in zip(tasks, results):
                if isinstance(result, Exception):
                    agent_responses[agent_name] = {"error": str(result)}
                else:
                    agent_responses[agent_name] = result.content if hasattr(result, 'content') else result
        
        return agent_responses
    
    async def _aggregate_responses(
        self,
        query: str,
        intent: Dict[str, Any],
        agent_responses: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Aggregate responses from multiple agents into a cohesive answer
        """
        aggregated = {
            "status": "success",
            "query": query,
            "intent": intent["primary"],
            "timestamp": datetime.now().isoformat(),
            "agent_responses": agent_responses,
            "summary": "",
            "actions_taken": [],
            "next_steps": []
        }
        
        # Build summary based on responses
        summaries = []
        
        if "infrastructure" in agent_responses:
            infra_resp = agent_responses["infrastructure"]
            if infra_resp.get("status") == "success":
                action = infra_resp.get("action", "")
                service = infra_resp.get("service", "")
                summaries.append(f"✓ {action.capitalize()} operation completed for {service}")
                aggregated["actions_taken"].append(f"{action}_{service}")
        
        if "search" in agent_responses:
            search_resp = agent_responses["search"]
            if search_resp.get("status") == "success":
                results_count = search_resp.get("search_results", {}).get("count", 0)
                summaries.append(f"✓ Found {results_count} relevant documents")
        
        if "data" in agent_responses:
            data_resp = agent_responses["data"]
            if data_resp.get("status") == "success":
                data_type = data_resp.get("data_type", "")
                summaries.append(f"✓ Retrieved {data_type} for analysis")
        
        if "analysis" in agent_responses:
            analysis_resp = agent_responses["analysis"]
            if analysis_resp.get("status") == "success":
                analysis = analysis_resp.get("analysis", {})
                if isinstance(analysis, dict):
                    if "health_score" in analysis:
                        summaries.append(f"✓ Health score: {analysis.get('health_score')}/100")
                    if "recommendations" in analysis:
                        aggregated["next_steps"] = analysis.get("recommendations", [])
        
        aggregated["summary"] = " | ".join(summaries) if summaries else "Query processed successfully"
        
        return aggregated
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history"""
        return self.conversation_history
    
    def get_all_agents_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        return {
            "captain": self.get_status(),
            "agents": {
                name: agent.get_status()
                for name, agent in self.agents.items()
            }
        }
    
    def get_agent_by_role(self, role: str) -> Optional[ChatbotBaseAgent]:
        """Get agent by role"""
        return self.agents.get(role)


# Global captain agent instance
captain_chatbot_agent = CaptainChatbotAgent()
