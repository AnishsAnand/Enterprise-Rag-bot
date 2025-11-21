"""
RAG Agent - Answers questions using the existing RAG system.
Integrates with the existing rag_widget.widget_query for documentation queries.
"""

from typing import Any, Dict, List, Optional
from langchain.tools import Tool
import logging
import json

from app.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class RAGAgent(BaseAgent):
    """
    Agent specialized in answering questions using the existing RAG system.
    Leverages the existing rag_widget.widget_query implementation with Milvus.
    """
    
    def __init__(self):
        """
        Initialize RAG agent.
        Uses the existing widget_query system - no need for separate services.
        """
        super().__init__(
            agent_name="RAGAgent",
            agent_description=(
                "Answers questions using documentation stored in Milvus vector database. "
                "Provides accurate, context-aware responses based on retrieved knowledge. "
                "Uses the existing trained RAG system with admin-managed knowledge base."
            ),
            temperature=0.4
        )
        
        # Setup agent
        self.setup_agent()
    
    def get_system_prompt(self) -> str:
        """Return system prompt for RAG agent."""
        return """You are the RAG Agent, specialized in answering questions using documentation.

**Your responsibilities:**
1. **Search the knowledge base** for relevant information
2. **Synthesize answers** from retrieved documents
3. **Provide accurate, helpful responses** based on documentation
4. **Cite sources** when possible
5. **Admit when you don't know** rather than making up information

**Response guidelines:**
- Base answers on retrieved documentation
- Be clear and concise
- Use examples when helpful
- Format responses for readability
- If information is not in the knowledge base, say so
- Suggest related topics when relevant

**Example responses:**

User: "How do I create a Kubernetes cluster?"
"Based on our documentation, here's how to create a Kubernetes cluster:

1. **Choose your cluster configuration:**
   - Cluster name (lowercase letters, numbers, hyphens)
   - Data center location
   - Kubernetes version
   - Node count and instance types

2. **Specify network settings:**
   - VPC and subnet configuration
   - Security groups
   - Load balancer settings

3. **Submit the creation request** through the API or web interface

The cluster typically takes 10-15 minutes to provision. You'll receive a notification when it's ready.

**Related topics:** Cluster management, Node scaling, Cluster upgrades"

User: "What are the pricing tiers?"
"I don't have specific pricing information in the documentation I have access to. 
For current pricing details, I recommend:
- Checking the pricing page on our website
- Contacting our sales team
- Reviewing your account's billing section

Is there anything else about cluster features or configuration I can help with?"

Always be helpful, accurate, and transparent about the source of your information."""
    
    def get_tools(self) -> List[Tool]:
        """Return tools for RAG agent."""
        return [
            Tool(
                name="query_knowledge_base",
                func=self._query_knowledge_base,
                description=(
                    "Query the knowledge base using the existing RAG system. "
                    "This uses the trained Milvus vector database with admin-managed documentation. "
                    "Input: user's question as a string"
                )
            )
        ]
    
    def _query_knowledge_base(self, query: str) -> str:
        """
        Query knowledge base using the existing widget_query system.
        
        Args:
            query: User's question
            
        Returns:
            JSON string with RAG response
        """
        try:
            # Import the existing widget_query function and models
            from app.api.routes.rag_widget import widget_query, WidgetQueryRequest
            from fastapi import BackgroundTasks
            
            # Create request for existing RAG system
            widget_req = WidgetQueryRequest(
                query=query,
                max_results=5,
                include_sources=True,
                enable_advanced_search=True,
                search_depth="balanced",
                auto_execute=False,  # Don't auto-execute tasks for Q&A
                store_interaction=False  # Don't store agent internal queries
            )
            
            # Call existing RAG system
            import asyncio
            loop = asyncio.get_event_loop()
            background_tasks = BackgroundTasks()
            
            result = loop.run_until_complete(
                widget_query(widget_req, background_tasks)
            )
            
            # Extract relevant information
            response_data = {
                "answer": result.get("answer", ""),
                "sources": result.get("sources", [])[:3],  # Top 3 sources
                "confidence": result.get("confidence", 0.0),
                "search_results_count": len(result.get("sources", []))
            }
            
            return json.dumps(response_data, indent=2)
            
        except Exception as e:
            logger.error(f"❌ Knowledge base query failed: {str(e)}")
            return json.dumps({
                "error": str(e),
                "answer": "I encountered an error while searching the knowledge base."
            })
    
    async def execute(
        self,
        input_text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute RAG query using the existing widget_query system.
        
        Args:
            input_text: User's question
            context: Additional context
            
        Returns:
            Dict with RAG response
        """
        try:
            logger.info(f"📚 RAGAgent answering: {input_text[:100]}...")
            
            # Import the existing widget_query function and models
            from app.api.routes.rag_widget import widget_query, WidgetQueryRequest
            from fastapi import BackgroundTasks
            
            # Create request for existing RAG system
            widget_req = WidgetQueryRequest(
                query=input_text,
                max_results=5,
                include_sources=True,
                enable_advanced_search=True,
                search_depth="balanced",
                auto_execute=False,  # Don't auto-execute tasks for Q&A
                store_interaction=False  # Don't store agent internal queries
            )
            
            # Call existing RAG system
            background_tasks = BackgroundTasks()
            result = await widget_query(widget_req, background_tasks)
            
            # Format response
            answer = result.get("answer", "")
            sources = result.get("sources", [])
            confidence = result.get("confidence", 0.0)
            
            # Add source citations if available
            if sources and answer:
                answer += "\n\n**Sources:**"
                for i, source in enumerate(sources[:3], 1):
                    source_title = source.get("title", source.get("url", "Unknown"))
                    answer += f"\n{i}. {source_title}"
            
            logger.info(f"✅ RAGAgent completed with {len(sources)} sources, confidence: {confidence}")
            
            return {
                "agent_name": self.agent_name,
                "success": True,
                "output": answer,
                "sources_count": len(sources),
                "confidence": confidence,
                "metadata": {
                    "search_depth": "balanced",
                    "sources": sources[:3],  # Top 3 sources
                    "intent_detected": result.get("intent_detected"),
                    "used_existing_rag": True
                }
            }
            
        except Exception as e:
            logger.error(f"❌ RAG agent failed: {str(e)}")
            return {
                "agent_name": self.agent_name,
                "success": False,
                "error": str(e),
                "output": f"I encountered an error while searching the knowledge base: {str(e)}"
            }

