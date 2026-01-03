"""
Agentic Metrics Service - Evaluation metrics for agentic AI systems.

Implements the three key metrics from Azure AI Evaluation library:
1. Task Adherence - Did the agent answer the right question?
2. Tool Call Accuracy - Did the agent use tools correctly?
3. Intent Resolution - Did the agent understand the user's goal?

Reference: https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/evaluating-agentic-ai-systems-a-deep-dive-into-agentic-metrics/4403923
"""

import logging
import json
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)


class MetricScore(Enum):
    """Score levels for agentic metrics."""
    EXCELLENT = 5
    GOOD = 4
    ACCEPTABLE = 3
    POOR = 2
    FAILED = 1


@dataclass
class ToolCall:
    """Represents a tool/function call made by the agent."""
    tool_name: str
    tool_args: Dict[str, Any]
    tool_result: Any
    timestamp: str
    success: bool
    error: Optional[str] = None


@dataclass
class AgentTrace:
    """Complete trace of an agent execution for evaluation."""
    session_id: str
    user_query: str
    agent_name: str
    intent_detected: Optional[str] = None
    resource_type: Optional[str] = None
    operation: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    intermediate_steps: List[Dict[str, Any]] = field(default_factory=list)
    final_response: str = ""
    start_time: str = ""
    end_time: str = ""
    success: bool = True
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trace to dictionary for storage/analysis."""
        return asdict(self)


@dataclass
class EvaluationResult:
    """Result of evaluating an agent trace."""
    session_id: str
    agent_name: str
    task_adherence: float
    task_adherence_reasoning: str
    tool_call_accuracy: float
    tool_call_accuracy_reasoning: str
    intent_resolution: float
    intent_resolution_reasoning: str
    overall_score: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return asdict(self)


class AgenticMetricsEvaluator:
    """
    Evaluator for agentic AI systems using LLM-based assessment.
    
    Implements three key metrics:
    - Task Adherence: How well the response satisfies the original request
    - Tool Call Accuracy: Whether tools were used correctly
    - Intent Resolution: Whether the agent understood the user's goal
    
    Features:
    - LLM-based evaluation using your existing AI service
    - PostgreSQL persistence for evaluation results
    - In-memory caching for active traces
    """
    
    def __init__(self, ai_service=None, enable_persistence: bool = True):
        """
        Initialize the evaluator.
        
        Args:
            ai_service: Optional AI service for LLM-based evaluation.
                       If not provided, will import from app.services.ai_service
            enable_persistence: If True, persist results to PostgreSQL
        """
        self.ai_service = ai_service
        self._traces: Dict[str, AgentTrace] = {}
        self._evaluation_results: List[EvaluationResult] = []
        
        # PostgreSQL persistence
        self.enable_persistence = enable_persistence
        self._persistence = None
        
        if enable_persistence:
            try:
                from app.services.agentic_metrics_persistence import get_metrics_persistence
                self._persistence = get_metrics_persistence()
                logger.info("✅ AgenticMetricsEvaluator initialized with PostgreSQL persistence")
            except Exception as e:
                logger.warning(f"⚠️ PostgreSQL persistence unavailable: {e}. Using in-memory only.")
                self._persistence = None
        else:
            logger.info("✅ AgenticMetricsEvaluator initialized (in-memory only)")
    
    async def _get_ai_service(self):
        """Lazy load AI service."""
        if self.ai_service is None:
            from app.services.ai_service import ai_service
            self.ai_service = ai_service
        return self.ai_service
    
    def start_trace(
        self,
        session_id: str,
        user_query: str,
        agent_name: str
    ) -> AgentTrace:
        """
        Start tracing an agent execution.
        
        Args:
            session_id: Unique session identifier
            user_query: The user's original query
            agent_name: Name of the agent being traced
            
        Returns:
            AgentTrace object to record execution details
        """
        trace = AgentTrace(
            session_id=session_id,
            user_query=user_query,
            agent_name=agent_name,
            start_time=datetime.utcnow().isoformat()
        )
        self._traces[session_id] = trace
        logger.debug(f"📊 Started trace for session {session_id}")
        return trace
    
    def record_tool_call(
        self,
        session_id: str,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_result: Any,
        success: bool = True,
        error: Optional[str] = None
    ) -> None:
        """
        Record a tool call made during agent execution.
        
        Args:
            session_id: Session ID for the trace
            tool_name: Name of the tool called
            tool_args: Arguments passed to the tool
            tool_result: Result from the tool
            success: Whether the tool call succeeded
            error: Error message if failed
        """
        if session_id not in self._traces:
            logger.warning(f"⚠️ No trace found for session {session_id}")
            return
        
        tool_call = ToolCall(
            tool_name=tool_name,
            tool_args=tool_args,
            tool_result=tool_result,
            timestamp=datetime.utcnow().isoformat(),
            success=success,
            error=error
        )
        self._traces[session_id].tool_calls.append(tool_call)
        logger.debug(f"🔧 Recorded tool call: {tool_name}")
    
    def record_intent(
        self,
        session_id: str,
        intent: str,
        resource_type: Optional[str] = None,
        operation: Optional[str] = None
    ) -> None:
        """
        Record detected intent for the session.
        
        Args:
            session_id: Session ID
            intent: Detected intent string
            resource_type: Type of resource being operated on
            operation: Operation type (create, read, update, delete, list)
        """
        if session_id not in self._traces:
            logger.warning(f"⚠️ No trace found for session {session_id}")
            return
        
        trace = self._traces[session_id]
        trace.intent_detected = intent
        trace.resource_type = resource_type
        trace.operation = operation
        logger.debug(f"🎯 Recorded intent: {intent} ({operation} on {resource_type})")
    
    def record_intermediate_step(
        self,
        session_id: str,
        step_name: str,
        step_data: Dict[str, Any]
    ) -> None:
        """
        Record an intermediate step in agent execution.
        
        Args:
            session_id: Session ID
            step_name: Name/type of the step
            step_data: Data associated with the step
        """
        if session_id not in self._traces:
            return
        
        self._traces[session_id].intermediate_steps.append({
            "step_name": step_name,
            "timestamp": datetime.utcnow().isoformat(),
            **step_data
        })
    
    def complete_trace(
        self,
        session_id: str,
        final_response: str,
        success: bool = True,
        error: Optional[str] = None
    ) -> Optional[AgentTrace]:
        """
        Complete the trace for a session.
        
        Args:
            session_id: Session ID
            final_response: The agent's final response
            success: Whether the overall execution succeeded
            error: Error message if failed
            
        Returns:
            Completed AgentTrace or None if not found
        """
        if session_id not in self._traces:
            logger.warning(f"⚠️ No trace found for session {session_id}")
            return None
        
        trace = self._traces[session_id]
        trace.final_response = final_response
        trace.end_time = datetime.utcnow().isoformat()
        trace.success = success
        trace.error = error
        
        # Persist trace to PostgreSQL
        if self._persistence:
            try:
                # Convert tool_calls to serializable format
                trace_dict = trace.to_dict()
                trace_dict["tool_calls"] = [
                    {
                        "tool_name": tc.tool_name,
                        "tool_args": tc.tool_args,
                        "tool_result": str(tc.tool_result)[:1000] if tc.tool_result else None,
                        "timestamp": tc.timestamp,
                        "success": tc.success,
                        "error": tc.error
                    }
                    for tc in trace.tool_calls
                ]
                self._persistence.save_trace(trace_dict)
            except Exception as e:
                logger.error(f"❌ Failed to persist trace: {e}")
        
        logger.info(f"✅ Completed trace for session {session_id}")
        return trace
    
    def get_trace(self, session_id: str) -> Optional[AgentTrace]:
        """Get a trace by session ID."""
        return self._traces.get(session_id)
    
    async def evaluate_task_adherence(
        self,
        user_query: str,
        agent_response: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[float, str]:
        """
        Evaluate how well the agent's response satisfies the user's request.
        
        Uses LLM-based evaluation to assess:
        - Relevance: Is the response on topic?
        - Completeness: Does it fully address the request?
        - Alignment: Does it match user expectations?
        
        Args:
            user_query: The original user query
            agent_response: The agent's final response
            context: Optional additional context
            
        Returns:
            Tuple of (score 0-1, reasoning string)
        """
        ai_service = await self._get_ai_service()
        
        evaluation_prompt = f"""You are an AI evaluation expert. Evaluate how well the agent's response satisfies the user's original request.

USER REQUEST: "{user_query}"

AGENT RESPONSE: "{agent_response}"

{f"ADDITIONAL CONTEXT: {json.dumps(context)}" if context else ""}

Evaluate based on these criteria:
1. **Relevance** (0-3): Is the response on topic and addresses the user's question?
2. **Completeness** (0-3): Does the response fully address all aspects of the request?
3. **Alignment** (0-2): Does the response match what the user would expect?
4. **Clarity** (0-2): Is the response clear and actionable?

Scoring:
- 9-10: Excellent - Fully satisfies the request with no issues
- 7-8: Good - Mostly satisfies with minor gaps
- 5-6: Acceptable - Partially satisfies but has notable gaps
- 3-4: Poor - Barely addresses the request
- 0-2: Failed - Doesn't address the request at all

Respond in this exact JSON format:
{{
    "relevance_score": <0-3>,
    "completeness_score": <0-3>,
    "alignment_score": <0-2>,
    "clarity_score": <0-2>,
    "total_score": <0-10>,
    "reasoning": "<brief explanation of the evaluation>"
}}"""

        try:
            response = await ai_service._call_chat_with_retries(
                prompt=evaluation_prompt,
                max_tokens=500,
                temperature=0.1
            )
            
            # Parse JSON response
            result = json.loads(response)
            score = result.get("total_score", 5) / 10.0  # Normalize to 0-1
            reasoning = result.get("reasoning", "Evaluation completed")
            
            logger.info(f"📊 Task Adherence Score: {score:.2f}")
            return (score, reasoning)
            
        except json.JSONDecodeError:
            logger.error("❌ Failed to parse task adherence evaluation response")
            return (0.5, "Evaluation parsing failed - default score applied")
        except Exception as e:
            logger.error(f"❌ Task adherence evaluation failed: {e}")
            return (0.5, f"Evaluation error: {str(e)}")
    
    async def evaluate_tool_call_accuracy(
        self,
        user_query: str,
        tool_calls: List[ToolCall],
        expected_tools: Optional[List[str]] = None
    ) -> Tuple[float, str]:
        """
        Evaluate the accuracy of tool/function calls made by the agent.
        
        Assesses:
        - Tool Selection: Were the right tools chosen?
        - Argument Accuracy: Were arguments correct and well-formatted?
        - Logical Consistency: Were tool calls in a sensible order?
        - Necessity: Were unnecessary tools avoided?
        
        Args:
            user_query: The original user query
            tool_calls: List of tool calls made
            expected_tools: Optional list of expected tool names
            
        Returns:
            Tuple of (score 0-1, reasoning string)
        """
        if not tool_calls:
            return (1.0, "No tool calls required or made - N/A for this metric")
        
        ai_service = await self._get_ai_service()
        
        # Format tool calls for evaluation
        tool_calls_formatted = []
        for tc in tool_calls:
            tool_calls_formatted.append({
                "tool": tc.tool_name,
                "args": tc.tool_args,
                "success": tc.success,
                "result_preview": str(tc.tool_result)[:200] if tc.tool_result else None
            })
        
        evaluation_prompt = f"""You are an AI evaluation expert. Evaluate the accuracy of tool calls made by an AI agent.

USER REQUEST: "{user_query}"

TOOL CALLS MADE:
{json.dumps(tool_calls_formatted, indent=2)}

{f"EXPECTED TOOLS: {expected_tools}" if expected_tools else ""}

Evaluate based on these criteria:
1. **Tool Selection** (0-3): Were the right tools chosen for the task?
2. **Argument Accuracy** (0-3): Were the arguments correct and well-formatted?
3. **Logical Order** (0-2): Were tools called in a sensible sequence?
4. **Efficiency** (0-2): Were unnecessary tool calls avoided?

Scoring:
- 9-10: Excellent - Perfect tool usage
- 7-8: Good - Mostly correct with minor issues
- 5-6: Acceptable - Works but suboptimal
- 3-4: Poor - Incorrect tools or arguments
- 0-2: Failed - Completely wrong tool usage

Respond in this exact JSON format:
{{
    "selection_score": <0-3>,
    "argument_score": <0-3>,
    "order_score": <0-2>,
    "efficiency_score": <0-2>,
    "total_score": <0-10>,
    "issues": ["<list any issues found>"],
    "reasoning": "<brief explanation>"
}}"""

        try:
            response = await ai_service._call_chat_with_retries(
                prompt=evaluation_prompt,
                max_tokens=600,
                temperature=0.1
            )
            
            result = json.loads(response)
            score = result.get("total_score", 5) / 10.0
            issues = result.get("issues", [])
            reasoning = result.get("reasoning", "Evaluation completed")
            
            if issues:
                reasoning += f" Issues: {', '.join(issues)}"
            
            logger.info(f"📊 Tool Call Accuracy Score: {score:.2f}")
            return (score, reasoning)
            
        except json.JSONDecodeError:
            logger.error("❌ Failed to parse tool call accuracy evaluation")
            return (0.5, "Evaluation parsing failed - default score applied")
        except Exception as e:
            logger.error(f"❌ Tool call accuracy evaluation failed: {e}")
            return (0.5, f"Evaluation error: {str(e)}")
    
    async def evaluate_intent_resolution(
        self,
        user_query: str,
        detected_intent: Optional[str],
        resource_type: Optional[str],
        operation: Optional[str],
        initial_actions: List[Dict[str, Any]] = None
    ) -> Tuple[float, str]:
        """
        Evaluate how well the agent understood the user's underlying goal.
        
        Assesses:
        - Intent Detection: Did the agent correctly identify the intent?
        - Goal Understanding: Did it understand the underlying need?
        - Plan Quality: Did initial actions reflect correct understanding?
        
        Args:
            user_query: The original user query
            detected_intent: What intent the agent detected
            resource_type: Resource type identified
            operation: Operation type identified
            initial_actions: First actions/decisions made
            
        Returns:
            Tuple of (score 0-1, reasoning string)
        """
        ai_service = await self._get_ai_service()
        
        intent_info = {
            "detected_intent": detected_intent,
            "resource_type": resource_type,
            "operation": operation,
            "initial_actions": initial_actions or []
        }
        
        evaluation_prompt = f"""You are an AI evaluation expert. Evaluate how well the agent understood the user's underlying goal.

USER REQUEST: "{user_query}"

AGENT'S UNDERSTANDING:
{json.dumps(intent_info, indent=2)}

Evaluate based on these criteria:
1. **Intent Detection** (0-4): Did the agent correctly identify what the user wants to do?
2. **Goal Understanding** (0-3): Did it understand the underlying need (not just surface request)?
3. **Context Awareness** (0-3): Did it pick up on implied requirements or constraints?

Examples of good intent resolution:
- User: "Show me clusters in Delhi" → Intent: list, Resource: clusters, Filter: Delhi ✓
- User: "I need a new cluster" → Intent: create, Resource: cluster, Workflow: parameter collection ✓

Examples of poor intent resolution:
- User: "What clusters do we have?" → Treated as documentation question ✗
- User: "Create a cluster called prod" → Missed the name parameter ✗

Scoring:
- 9-10: Excellent - Perfect understanding of intent and goals
- 7-8: Good - Correct intent with minor gaps
- 5-6: Acceptable - Basically understood but missed nuances
- 3-4: Poor - Significant misunderstanding
- 0-2: Failed - Completely wrong interpretation

Respond in this exact JSON format:
{{
    "intent_score": <0-4>,
    "goal_score": <0-3>,
    "context_score": <0-3>,
    "total_score": <0-10>,
    "correct_aspects": ["<what was understood correctly>"],
    "missed_aspects": ["<what was missed or misunderstood>"],
    "reasoning": "<brief explanation>"
}}"""

        try:
            response = await ai_service._call_chat_with_retries(
                prompt=evaluation_prompt,
                max_tokens=600,
                temperature=0.1
            )
            
            result = json.loads(response)
            score = result.get("total_score", 5) / 10.0
            correct = result.get("correct_aspects", [])
            missed = result.get("missed_aspects", [])
            reasoning = result.get("reasoning", "Evaluation completed")
            
            if missed:
                reasoning += f" Missed: {', '.join(missed)}"
            
            logger.info(f"📊 Intent Resolution Score: {score:.2f}")
            return (score, reasoning)
            
        except json.JSONDecodeError:
            logger.error("❌ Failed to parse intent resolution evaluation")
            return (0.5, "Evaluation parsing failed - default score applied")
        except Exception as e:
            logger.error(f"❌ Intent resolution evaluation failed: {e}")
            return (0.5, f"Evaluation error: {str(e)}")
    
    async def evaluate_trace(
        self,
        trace: AgentTrace
    ) -> EvaluationResult:
        """
        Evaluate a complete agent trace using all three metrics.
        
        Args:
            trace: Complete AgentTrace to evaluate
            
        Returns:
            EvaluationResult with all metric scores
        """
        logger.info(f"📊 Evaluating trace for session {trace.session_id}")
        
        # Run all evaluations in parallel
        task_adherence_task = self.evaluate_task_adherence(
            user_query=trace.user_query,
            agent_response=trace.final_response,
            context={
                "resource_type": trace.resource_type,
                "operation": trace.operation,
                "success": trace.success
            }
        )
        
        tool_call_accuracy_task = self.evaluate_tool_call_accuracy(
            user_query=trace.user_query,
            tool_calls=trace.tool_calls
        )
        
        intent_resolution_task = self.evaluate_intent_resolution(
            user_query=trace.user_query,
            detected_intent=trace.intent_detected,
            resource_type=trace.resource_type,
            operation=trace.operation,
            initial_actions=trace.intermediate_steps[:3] if trace.intermediate_steps else []
        )
        
        # Await all evaluations
        (task_score, task_reasoning), \
        (tool_score, tool_reasoning), \
        (intent_score, intent_reasoning) = await asyncio.gather(
            task_adherence_task,
            tool_call_accuracy_task,
            intent_resolution_task
        )
        
        # Calculate weighted overall score
        # Task Adherence: 40%, Tool Accuracy: 30%, Intent Resolution: 30%
        overall_score = (task_score * 0.4) + (tool_score * 0.3) + (intent_score * 0.3)
        
        result = EvaluationResult(
            session_id=trace.session_id,
            agent_name=trace.agent_name,
            task_adherence=task_score,
            task_adherence_reasoning=task_reasoning,
            tool_call_accuracy=tool_score,
            tool_call_accuracy_reasoning=tool_reasoning,
            intent_resolution=intent_score,
            intent_resolution_reasoning=intent_reasoning,
            overall_score=overall_score,
            metadata={
                "user_query": trace.user_query,
                "tool_calls_count": len(trace.tool_calls),
                "execution_success": trace.success,
                "resource_type": trace.resource_type,
                "operation": trace.operation
            }
        )
        
        # Store in memory
        self._evaluation_results.append(result)
        
        # Persist to PostgreSQL
        if self._persistence:
            try:
                self._persistence.save_evaluation(result.to_dict())
            except Exception as e:
                logger.error(f"❌ Failed to persist evaluation: {e}")
        
        logger.info(f"✅ Evaluation complete - Overall Score: {overall_score:.2f}")
        logger.info(f"   Task Adherence: {task_score:.2f}")
        logger.info(f"   Tool Call Accuracy: {tool_score:.2f}")
        logger.info(f"   Intent Resolution: {intent_score:.2f}")
        
        return result
    
    async def evaluate_session(self, session_id: str) -> Optional[EvaluationResult]:
        """
        Evaluate a session by its ID.
        
        Args:
            session_id: Session ID to evaluate
            
        Returns:
            EvaluationResult or None if trace not found
        """
        trace = self.get_trace(session_id)
        if not trace:
            logger.warning(f"⚠️ No trace found for session {session_id}")
            return None
        
        return await self.evaluate_trace(trace)
    
    async def batch_evaluate(
        self,
        traces: List[AgentTrace]
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple traces in batch.
        
        Args:
            traces: List of AgentTrace objects
            
        Returns:
            List of EvaluationResult objects
        """
        logger.info(f"📊 Batch evaluating {len(traces)} traces")
        
        results = await asyncio.gather(*[
            self.evaluate_trace(trace) for trace in traces
        ])
        
        return list(results)
    
    def get_evaluation_summary(self, from_database: bool = True) -> Dict[str, Any]:
        """
        Get summary statistics of all evaluations.
        
        Args:
            from_database: If True and persistence is available, get stats from PostgreSQL
        
        Returns:
            Dict with aggregate metrics and statistics
        """
        # Prefer database stats if persistence is available
        if from_database and self._persistence:
            try:
                db_summary = self._persistence.get_summary_stats()
                db_summary["source"] = "postgresql"
                return db_summary
            except Exception as e:
                logger.warning(f"⚠️ Database summary failed, using in-memory: {e}")
        
        # Fall back to in-memory results
        if not self._evaluation_results:
            return {"message": "No evaluations performed yet", "count": 0, "source": "memory"}
        
        results = self._evaluation_results
        count = len(results)
        
        avg_task = sum(r.task_adherence for r in results) / count
        avg_tool = sum(r.tool_call_accuracy for r in results) / count
        avg_intent = sum(r.intent_resolution for r in results) / count
        avg_overall = sum(r.overall_score for r in results) / count
        
        return {
            "total_evaluations": count,
            "average_scores": {
                "task_adherence": round(avg_task, 3),
                "tool_call_accuracy": round(avg_tool, 3),
                "intent_resolution": round(avg_intent, 3),
                "overall": round(avg_overall, 3)
            },
            "score_distribution": {
                "excellent": sum(1 for r in results if r.overall_score >= 0.9),
                "good": sum(1 for r in results if 0.7 <= r.overall_score < 0.9),
                "acceptable": sum(1 for r in results if 0.5 <= r.overall_score < 0.7),
                "poor": sum(1 for r in results if 0.3 <= r.overall_score < 0.5),
                "failed": sum(1 for r in results if r.overall_score < 0.3)
            },
            "by_agent": self._get_scores_by_agent(),
            "by_operation": self._get_scores_by_operation(),
            "source": "memory"
        }
    
    def _get_scores_by_agent(self) -> Dict[str, Dict[str, float]]:
        """Get average scores grouped by agent."""
        agent_scores: Dict[str, List[float]] = {}
        
        for result in self._evaluation_results:
            if result.agent_name not in agent_scores:
                agent_scores[result.agent_name] = []
            agent_scores[result.agent_name].append(result.overall_score)
        
        return {
            agent: {
                "average": round(sum(scores) / len(scores), 3),
                "count": len(scores)
            }
            for agent, scores in agent_scores.items()
        }
    
    def _get_scores_by_operation(self) -> Dict[str, Dict[str, float]]:
        """Get average scores grouped by operation type."""
        op_scores: Dict[str, List[float]] = {}
        
        for result in self._evaluation_results:
            operation = result.metadata.get("operation", "unknown")
            if operation not in op_scores:
                op_scores[operation] = []
            op_scores[operation].append(result.overall_score)
        
        return {
            op: {
                "average": round(sum(scores) / len(scores), 3),
                "count": len(scores)
            }
            for op, scores in op_scores.items()
        }
    
    def export_results(self, format: str = "json") -> str:
        """
        Export evaluation results.
        
        Args:
            format: Output format ('json' or 'jsonl')
            
        Returns:
            String representation of results
        """
        results_data = [r.to_dict() for r in self._evaluation_results]
        
        if format == "jsonl":
            return "\n".join(json.dumps(r) for r in results_data)
        else:
            return json.dumps(results_data, indent=2)
    
    def clear_results(self) -> None:
        """Clear all stored traces and evaluation results."""
        self._traces.clear()
        self._evaluation_results.clear()
        logger.info("🗑️ Cleared all traces and evaluation results")


# Singleton instance
agentic_metrics_evaluator = AgenticMetricsEvaluator()

