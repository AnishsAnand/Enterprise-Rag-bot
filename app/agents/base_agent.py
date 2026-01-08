"""
Base Agent class for the multi-agent system using LangChain.
All specialized agents inherit from this base class.

Enhanced with Agentic Metrics for evaluation:
- Task Adherence: How well does the response satisfy the request?
- Tool Call Accuracy: Were tools used correctly?
- Intent Resolution: Was the user's goal understood?

Reference: https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/evaluating-agentic-ai-systems-a-deep-dive-into-agentic-metrics/4403923
"""

from typing import Any, Dict, List, Optional, Callable, Union
from abc import ABC, abstractmethod
import logging
import time
from datetime import datetime
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult
import os

logger = logging.getLogger(__name__)


# =============================================================================
# LangChain Callback Handler for Prometheus Metrics
# =============================================================================
class PrometheusCallbackHandler(BaseCallbackHandler):
    """
    LangChain callback handler that tracks LLM calls to Prometheus metrics.
    
    This ensures LangChain agent LLM calls are properly tracked alongside
    direct ai_service calls.
    """
    
    def __init__(self, agent_name: str = "unknown"):
        self.agent_name = agent_name
        self._call_start_times: Dict[str, float] = {}
        self._prom_metrics = None
    
    def _get_metrics(self):
        """Lazy load prometheus metrics to avoid circular imports."""
        if self._prom_metrics is None:
            try:
                from app.services.prometheus_metrics import metrics
                self._prom_metrics = metrics
            except ImportError:
                logger.warning("⚠️ Prometheus metrics not available for LangChain callback")
        return self._prom_metrics
    
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Called when LLM starts running."""
        run_id = str(kwargs.get("run_id", "unknown"))
        self._call_start_times[run_id] = time.time()
        
        # Get model name from serialized data
        model = serialized.get("kwargs", {}).get("model", "unknown")
        logger.info(f"🔄 LangChain LLM START: agent={self.agent_name}, model={model}, run_id={run_id[:8]}")
    
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Called when LLM finishes running."""
        run_id = str(kwargs.get("run_id", "unknown"))
        start_time = self._call_start_times.pop(run_id, time.time())
        duration = time.time() - start_time
        
        # Extract token usage from response
        input_tokens = 0
        output_tokens = 0
        model = "langchain-agent"
        
        if response.llm_output:
            usage = response.llm_output.get("token_usage", {})
            input_tokens = usage.get("prompt_tokens", 0) or 0
            output_tokens = usage.get("completion_tokens", 0) or 0
            model = response.llm_output.get("model_name", model)
        
        # Track metrics
        metrics = self._get_metrics()
        if metrics:
            metrics.track_llm_call(
                model=model,
                operation=f"langchain-{self.agent_name}",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                duration=duration,
                success=True
            )
        
        logger.info(f"✅ LangChain LLM END: agent={self.agent_name}, model={model}, "
                   f"tokens={input_tokens}/{output_tokens}, duration={duration:.2f}s")
    
    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Called when LLM encounters an error."""
        run_id = str(kwargs.get("run_id", "unknown"))
        start_time = self._call_start_times.pop(run_id, time.time())
        duration = time.time() - start_time
        
        # Track error metrics
        metrics = self._get_metrics()
        if metrics:
            metrics.track_llm_call(
                model="langchain-agent",
                operation=f"langchain-{self.agent_name}",
                input_tokens=0,
                output_tokens=0,
                duration=duration,
                success=False,
                error_type=type(error).__name__
            )
        
        logger.error(f"❌ LangChain LLM ERROR: agent={self.agent_name}, error={error}, "
                    f"duration={duration:.2f}s")

# Lazy import for agentic metrics to avoid circular imports
_metrics_evaluator = None

def get_metrics_evaluator():
    """Get the singleton agentic metrics evaluator."""
    global _metrics_evaluator
    if _metrics_evaluator is None:
        try:
            from app.services.agentic_metrics_service import agentic_metrics_evaluator
            _metrics_evaluator = agentic_metrics_evaluator
        except ImportError:
            logger.warning("⚠️ Agentic metrics service not available")
            _metrics_evaluator = None
    return _metrics_evaluator


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the system.
    Provides common functionality for agent communication, state management, and execution.
    """
    
    def __init__(
        self,
        agent_name: str,
        agent_description: str,
        model_name: str = None,
        temperature: float = 0.2,
        max_tokens: int = 2000,
    ):
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
        grok_base_url = os.getenv("GROK_BASE_URL")
        grok_api_key = os.getenv("GROK_API_KEY") or os.getenv("OPENAI_API_KEY", "dummy-key")
        self.model_name = model_name or os.getenv("CHAT_MODEL", "openai/gpt-oss-120b")
        
        # Create callback handler for Prometheus metrics tracking
        self.metrics_callback = PrometheusCallbackHandler(agent_name=agent_name)
        
        # Initialize LangChain LLM with OpenAI-compatible endpoint and metrics callback
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            openai_api_base=grok_base_url,
            openai_api_key=grok_api_key,
            model_kwargs={"top_p": 0.9},
            callbacks=[self.metrics_callback]
        )
        
        # Agent state
        self.state: Dict[str, Any] = {
            "agent_name": agent_name,
            "created_at": datetime.utcnow().isoformat(),
            "execution_count": 0,
            "last_execution": None,
            "errors": []
        }
        
        # Memory for conversation context
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )
        
        # Tools and executor (to be set by subclasses)
        self.tools: List[Tool] = []
        self.agent_executor: Optional[AgentExecutor] = None
        
        # Agentic metrics configuration
        self.metrics_enabled: bool = os.getenv("ENABLE_AGENTIC_METRICS", "true").lower() == "true"
        self._current_session_id: Optional[str] = None
        
        logger.info(f"✅ Initialized {agent_name} with model {self.model_name}")
        if self.metrics_enabled:
            logger.info(f"📊 Agentic metrics enabled for {agent_name}")
    
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
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])
            
            # Create agent
            agent = create_openai_functions_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=prompt
            )
            
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
                "context": "setup_agent"
            })
            raise
    
    async def execute(
        self,
        input_text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute the agent with given input and context.
        
        Args:
            input_text: User input or task description
            context: Additional context for the agent
            
        Returns:
            Dict containing agent output and metadata
        """
        # Extract session_id from context for metrics tracking
        session_id = context.get("session_id") if context else None
        self._current_session_id = session_id
        
        # Start metrics trace if enabled
        if self.metrics_enabled and session_id:
            self._start_metrics_trace(session_id, input_text)
        
        # Get Prometheus metrics for agent execution tracking
        prom_metrics = self._get_prometheus_metrics()
        start_time = time.time()
        execution_success = False
        
        try:
            # Track active sessions
            if prom_metrics:
                prom_metrics.agent_active_sessions.labels(agent_name=self.agent_name).inc()
            
            self.state["execution_count"] += 1
            self.state["last_execution"] = datetime.utcnow().isoformat()
            
            # Prepare input with context
            full_input = input_text
            if context:
                context_str = "\n\n**Context:**\n" + "\n".join(
                    f"- {k}: {v}" for k, v in context.items()
                )
                full_input = f"{input_text}{context_str}"
            
            logger.info(f"🤖 {self.agent_name} executing with input: {input_text[:100]}...")
            
            # Execute agent
            if self.agent_executor:
                result = await self.agent_executor.ainvoke({"input": full_input})
            else:
                # Fallback to direct LLM call if no executor
                result = {"output": await self._direct_llm_call(full_input)}
            
            # Record intermediate steps for metrics
            intermediate_steps = result.get("intermediate_steps", [])
            if self.metrics_enabled and session_id:
                self._record_intermediate_steps(session_id, intermediate_steps)
            
            # Track intermediate steps count
            if prom_metrics and intermediate_steps:
                prom_metrics.agent_steps_per_execution.labels(
                    agent_name=self.agent_name
                ).observe(len(intermediate_steps))
            
            # Format response
            response = {
                "agent_name": self.agent_name,
                "success": True,
                "output": result.get("output", ""),
                "intermediate_steps": intermediate_steps,
                "timestamp": datetime.utcnow().isoformat(),
                "execution_count": self.state["execution_count"]
            }
            
            # Complete metrics trace on success
            if self.metrics_enabled and session_id:
                self._complete_metrics_trace(session_id, result.get("output", ""), success=True)
            
            execution_success = True
            logger.info(f"✅ {self.agent_name} completed execution")
            return response
            
        except Exception as e:
            logger.error(f"❌ {self.agent_name} execution failed: {str(e)}")
            self.state["errors"].append({
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "input": input_text[:200]
            })
            
            # Complete metrics trace on failure
            if self.metrics_enabled and session_id:
                self._complete_metrics_trace(
                    session_id, 
                    f"Agent {self.agent_name} encountered an error: {str(e)}",
                    success=False,
                    error=str(e)
                )
            
            return {
                "agent_name": self.agent_name,
                "success": False,
                "error": str(e),
                "output": f"Agent {self.agent_name} encountered an error: {str(e)}",
                "timestamp": datetime.utcnow().isoformat()
            }
        finally:
            # Track agent session completion and duration
            if prom_metrics:
                duration = time.time() - start_time
                status = 'success' if execution_success else 'error'
                prom_metrics.agent_sessions_total.labels(
                    agent_name=self.agent_name, status=status
                ).inc()
                prom_metrics.agent_execution_duration.labels(
                    agent_name=self.agent_name, operation='execute'
                ).observe(duration)
                prom_metrics.agent_active_sessions.labels(agent_name=self.agent_name).dec()
    
    async def _direct_llm_call(self, input_text: str) -> str:
        """
        Direct LLM call without agent executor (fallback).
        """
        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": input_text}
        ]
        response = await self.llm.ainvoke(messages)
        return response.content
    
    def _get_prometheus_metrics(self):
        """Get Prometheus metrics instance for tracking."""
        try:
            from app.services.prometheus_metrics import metrics
            return metrics
        except ImportError:
            return None
    
    # =========================================================================
    # AGENTIC METRICS METHODS
    # Reference: Azure AI Evaluation Agentic Metrics
    # =========================================================================
    
    def _start_metrics_trace(self, session_id: str, user_query: str) -> None:
        """
        Start a metrics trace for this execution.
        
        Args:
            session_id: Unique session identifier
            user_query: The user's original query
        """
        evaluator = get_metrics_evaluator()
        if evaluator:
            evaluator.start_trace(
                session_id=session_id,
                user_query=user_query,
                agent_name=self.agent_name
            )
            logger.debug(f"📊 Started metrics trace for {session_id}")
    
    def _record_intermediate_steps(
        self, 
        session_id: str, 
        intermediate_steps: List[Any]
    ) -> None:
        """
        Record intermediate steps (including tool calls) from agent execution.
        
        Args:
            session_id: Session identifier
            intermediate_steps: List of (action, observation) tuples from LangChain
        """
        evaluator = get_metrics_evaluator()
        if not evaluator:
            return
        
        for step in intermediate_steps:
            try:
                # LangChain intermediate steps are (AgentAction, observation) tuples
                if isinstance(step, tuple) and len(step) >= 2:
                    action, observation = step[0], step[1]
                    
                    # Extract tool call information
                    tool_name = getattr(action, 'tool', 'unknown')
                    tool_input = getattr(action, 'tool_input', {})
                    
                    # Record the tool call
                    evaluator.record_tool_call(
                        session_id=session_id,
                        tool_name=tool_name,
                        tool_args=tool_input if isinstance(tool_input, dict) else {"input": tool_input},
                        tool_result=observation,
                        success=True
                    )
                    
                    # Also record as intermediate step
                    evaluator.record_intermediate_step(
                        session_id=session_id,
                        step_name=f"tool_call:{tool_name}",
                        step_data={
                            "tool_input": str(tool_input)[:500],
                            "tool_output": str(observation)[:500]
                        }
                    )
            except Exception as e:
                logger.debug(f"Could not record intermediate step: {e}")
    
    def _complete_metrics_trace(
        self,
        session_id: str,
        final_response: str,
        success: bool = True,
        error: Optional[str] = None
    ) -> None:
        """
        Complete the metrics trace for this execution.
        
        Args:
            session_id: Session identifier
            final_response: The agent's final response
            success: Whether execution succeeded
            error: Error message if failed
        """
        evaluator = get_metrics_evaluator()
        if evaluator:
            evaluator.complete_trace(
                session_id=session_id,
                final_response=final_response,
                success=success,
                error=error
            )
            logger.debug(f"📊 Completed metrics trace for {session_id}")
    
    def record_tool_call(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_result: Any,
        success: bool = True,
        error: Optional[str] = None
    ) -> None:
        """
        Manually record a tool call for metrics tracking.
        Use this for custom tool calls not handled by LangChain executor.
        
        Args:
            tool_name: Name of the tool
            tool_args: Arguments passed to the tool
            tool_result: Result from the tool
            success: Whether the call succeeded
            error: Error message if failed
        """
        if not self.metrics_enabled or not self._current_session_id:
            return
        
        evaluator = get_metrics_evaluator()
        if evaluator:
            evaluator.record_tool_call(
                session_id=self._current_session_id,
                tool_name=tool_name,
                tool_args=tool_args,
                tool_result=tool_result,
                success=success,
                error=error
            )
    
    def record_intent(
        self,
        intent: str,
        resource_type: Optional[str] = None,
        operation: Optional[str] = None
    ) -> None:
        """
        Record the detected intent for metrics tracking.
        
        Args:
            intent: The detected intent
            resource_type: Type of resource being operated on
            operation: Operation type (create, read, update, delete, list)
        """
        if not self.metrics_enabled or not self._current_session_id:
            return
        
        evaluator = get_metrics_evaluator()
        if evaluator:
            evaluator.record_intent(
                session_id=self._current_session_id,
                intent=intent,
                resource_type=resource_type,
                operation=operation
            )
    
    async def evaluate_execution(self, session_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Evaluate the current or specified execution using agentic metrics.
        
        Args:
            session_id: Optional session ID (uses current if not specified)
            
        Returns:
            EvaluationResult as dict or None if not available
        """
        if not self.metrics_enabled:
            return None
        
        evaluator = get_metrics_evaluator()
        if not evaluator:
            return None
        
        target_session = session_id or self._current_session_id
        if not target_session:
            logger.warning("No session ID available for evaluation")
            return None
        
        result = await evaluator.evaluate_session(target_session)
        if result:
            return result.to_dict()
        return None
    
    def communicate_with_agent(
        self,
        target_agent: "BaseAgent",
        message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
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

