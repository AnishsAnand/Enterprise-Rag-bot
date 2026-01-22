"""
Base Agent class for the multi-agent system using LangChain.
All specialized agents inherit from this base class.
"""
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod
import logging
from datetime import datetime
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
import os

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """
    Abstract base class for all agents in the system.
    Provides common functionality for agent communication, state management, and execution.
    """
    
    def __init__(self,agent_name: str,agent_description: str,model_name: str = None,temperature: float = 0.2,max_tokens: int = 2000,):
        """
        Initialize base agent with LangChain components.
        Args:
            agent_name: Unique identifier for the agent
            agent_description: Description of agent's purpose and capabilities
            model_name: LLM model to use (defaults to env CHAT_MODEL)
            temperature: Model temperature for response generation
            max_tokens: Maximum tokens in response
        """
        self.agent_name = agent_name
        self.agent_description = agent_description
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Get model configuration from environment
        grok_base_url = os.getenv("GROK_BASE_URL", "https://models.cloudservices.tatacommunications.com/v1")
        grok_api_key = os.getenv("GROK_API_KEY") or os.getenv("OPENAI_API_KEY", "dummy-key")
        self.model_name = model_name or os.getenv("CHAT_MODEL", "openai/gpt-oss-120b")
        # Initialize LangChain LLM with OpenAI-compatible endpoint
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            openai_api_base=grok_base_url,
            openai_api_key=grok_api_key,
            model_kwargs={"top_p": 0.9})
        # Agent state
        self.state: Dict[str, Any] = {
            "agent_name": agent_name,
            "created_at": datetime.utcnow().isoformat(),
            "execution_count": 0,
            "last_execution": None,
            "errors": []}
        # Memory for conversation context
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output")
        # Tools and executor (to be set by subclasses)
        self.tools: List[Tool] = []
        self.agent_executor: Optional[AgentExecutor] = None
        logger.info(f"✅ Initialized {agent_name} with model {self.model_name}")

    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Return the system prompt for this agent.
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def get_tools(self) -> List[Tool]:
        """
        Return the list of tools this agent can use.
        Must be implemented by subclasses.
        """
        pass
    
    def setup_agent(self) -> None:
        """
        Setup the LangChain agent with tools and prompts.
        Called after initialization to configure the agent executor.
        """
        try:
            # Get tools from subclass
            self.tools = self.get_tools()
            
            # Create prompt template
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.get_system_prompt()),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),])
            
            # Create agent
            agent = create_openai_functions_agent(llm=self.llm,tools=self.tools,prompt=prompt)
            # Create agent executor
            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                memory=self.memory,
                verbose=True,
                max_iterations=5,
                handle_parsing_errors=True,
                return_intermediate_steps=True
            )
            
            logger.info(f"✅ Agent {self.agent_name} setup complete with {len(self.tools)} tools")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup agent {self.agent_name}: {str(e)}")
            self.state["errors"].append({
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "context": "setup_agent"})
            raise

    async def execute(self,input_text: str,context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute the agent with given input and context.
        Args:
            input_text: User input or task description
            context: Additional context for the agent
        Returns:
            Dict containing agent output and metadata
        """
        try:
            self.state["execution_count"] += 1
            self.state["last_execution"] = datetime.utcnow().isoformat()
            # Prepare input with context
            full_input = input_text
            if context:
                context_str = "\n\n**Context:**\n" + "\n".join(
                    f"- {k}: {v}" for k, v in context.items())
                full_input = f"{input_text}{context_str}"
            logger.info(f"🤖 {self.agent_name} executing with input: {input_text[:100]}...")
            # Execute agent
            if self.agent_executor:
                result = await self.agent_executor.ainvoke({"input": full_input})
            else:
                # Fallback to direct LLM call if no executor
                result = {"output": await self._direct_llm_call(full_input)}
            # Format response
            response = {
                "agent_name": self.agent_name,
                "success": True,
                "output": result.get("output", ""),
                "intermediate_steps": result.get("intermediate_steps", []),
                "timestamp": datetime.utcnow().isoformat(),
                "execution_count": self.state["execution_count"]}
            logger.info(f"✅ {self.agent_name} completed execution")
            return response
        except Exception as e:
            logger.error(f"❌ {self.agent_name} execution failed: {str(e)}")
            self.state["errors"].append({
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "input": input_text[:200]})
            return {
                "agent_name": self.agent_name,
                "success": False,
                "error": str(e),
                "output": f"Agent {self.agent_name} encountered an error: {str(e)}",
                "timestamp": datetime.utcnow().isoformat()}
    
    async def _direct_llm_call(self, input_text: str) -> str:
        """
        Direct LLM call without agent executor (fallback).
        """
        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": input_text}]
        response = await self.llm.ainvoke(messages)
        return response.content

    def communicate_with_agent(self,target_agent: "BaseAgent",message: str,context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Send a message to another agent and get response.
        Args:
            target_agent: The agent to communicate with
            message: Message to send
            context: Additional context
            
        Returns:
            Response from target agent
        """
        logger.info(f"📨 {self.agent_name} -> {target_agent.agent_name}: {message[:50]}...")
        # Add sender information to context
        if context is None:
            context = {}
        context["sender_agent"] = self.agent_name
        context["message_type"] = "agent_communication"
        # Execute target agent
        import asyncio
        loop = asyncio.get_event_loop()
        response = loop.run_until_complete(target_agent.execute(message, context))
        return response
    
    def get_state(self) -> Dict[str, Any]:
        """Get current agent state."""
        return self.state.copy()
    
    def update_state(self, updates: Dict[str, Any]) -> None:
        """Update agent state."""
        self.state.update(updates)
        logger.debug(f"📊 {self.agent_name} state updated: {list(updates.keys())}")
    
    def reset(self) -> None:
        """Reset agent state and memory."""
        self.memory.clear()
        self.state["execution_count"] = 0
        self.state["last_execution"] = None
        self.state["errors"] = []
        logger.info(f"🔄 {self.agent_name} reset complete")
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.agent_name}, executions={self.state['execution_count']})>"
