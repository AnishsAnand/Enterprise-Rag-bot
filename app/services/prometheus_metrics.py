"""
Prometheus Metrics Service for Vayu Maya RAG Bot.

Provides comprehensive metrics collection for:
- LLM API calls (tokens, latency, errors)
- RAG pipeline (retrieval, embedding, reranking)
- Agent execution (tool calls, success rates)
- Agentic evaluation scores
- System health (memory, connections)

Usage:
    from app.services.prometheus_metrics import metrics
    
    # Track LLM call
    with metrics.llm_call_duration.labels(model="gpt-4", operation="chat").time():
        response = await llm.chat(...)
    metrics.llm_tokens_total.labels(model="gpt-4", type="input").inc(input_tokens)
    metrics.llm_tokens_total.labels(model="gpt-4", type="output").inc(output_tokens)
"""

import os
import time
import logging
import traceback
from typing import Optional, Callable, List, Dict
from functools import wraps
from contextlib import contextmanager
from datetime import datetime
from collections import deque

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Summary,
    Info,
    REGISTRY,
    generate_latest,
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    multiprocess,
    make_asgi_app,
)

logger = logging.getLogger(__name__)


class PrometheusMetrics:
    """
    Centralized Prometheus metrics for the RAG application.
    
    Metrics Categories:
    1. LLM Metrics - Token usage, latency, costs
    2. RAG Metrics - Retrieval performance, embedding
    3. Agent Metrics - Tool calls, execution time
    4. Agentic Evaluation - Quality scores
    5. HTTP Metrics - Request/response tracking
    6. System Metrics - Health indicators
    """
    
    def __init__(self):
        """Initialize all Prometheus metrics."""
        self._initialized = False
        self._registry = REGISTRY
        
        # =====================================================================
        # LLM METRICS - Track all LLM API interactions
        # =====================================================================
        
        # Total tokens processed (input/output)
        self.llm_tokens_total = Counter(
            'vayu_llm_tokens_total',
            'Total tokens processed by LLM',
            ['model', 'type', 'operation']  # type: input/output, operation: chat/embedding/completion
        )
        
        # LLM API call duration
        self.llm_call_duration = Histogram(
            'vayu_llm_call_duration_seconds',
            'LLM API call duration in seconds',
            ['model', 'operation'],
            buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)
        )
        
        # LLM API calls total
        self.llm_calls_total = Counter(
            'vayu_llm_calls_total',
            'Total LLM API calls',
            ['model', 'operation', 'status']  # status: success/error
        )
        
        # Estimated cost tracking (based on token counts)
        self.llm_estimated_cost = Counter(
            'vayu_llm_estimated_cost_usd',
            'Estimated LLM API cost in USD',
            ['model', 'operation']
        )
        
        # Current active LLM requests
        self.llm_active_requests = Gauge(
            'vayu_llm_active_requests',
            'Currently active LLM API requests',
            ['model']
        )
        
        # LLM errors by type
        self.llm_errors_total = Counter(
            'vayu_llm_errors_total',
            'LLM API errors by type',
            ['model', 'error_type']  # error_type: rate_limit, timeout, api_error, etc.
        )
        
        # =====================================================================
        # RAG PIPELINE METRICS
        # =====================================================================
        
        # Document retrieval metrics
        self.rag_retrieval_duration = Histogram(
            'vayu_rag_retrieval_duration_seconds',
            'RAG document retrieval duration',
            ['source'],  # source: milvus, web, cache
            buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
        )
        
        self.rag_documents_retrieved = Histogram(
            'vayu_rag_documents_retrieved',
            'Number of documents retrieved per query',
            ['source'],
            buckets=(1, 3, 5, 10, 20, 50, 100)
        )
        
        self.rag_relevance_score = Histogram(
            'vayu_rag_relevance_score',
            'Relevance scores of retrieved documents',
            ['source'],
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
        )
        
        # Embedding metrics
        self.embedding_duration = Histogram(
            'vayu_embedding_duration_seconds',
            'Embedding generation duration',
            ['model', 'batch_size'],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)
        )
        
        self.embedding_dimensions = Gauge(
            'vayu_embedding_dimensions',
            'Embedding vector dimensions',
            ['model']
        )
        
        # Reranking metrics
        self.rerank_duration = Histogram(
            'vayu_rerank_duration_seconds',
            'Reranking operation duration',
            ['method'],
            buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
        )
        
        # =====================================================================
        # AGENT EXECUTION METRICS
        # =====================================================================
        
        # Agent execution duration
        self.agent_execution_duration = Histogram(
            'vayu_agent_execution_duration_seconds',
            'Agent execution duration',
            ['agent_name', 'operation'],
            buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0)
        )
        
        # Tool calls
        self.agent_tool_calls_total = Counter(
            'vayu_agent_tool_calls_total',
            'Total tool calls made by agents',
            ['agent_name', 'tool_name', 'status']
        )
        
        self.agent_tool_call_duration = Histogram(
            'vayu_agent_tool_call_duration_seconds',
            'Tool call execution duration',
            ['tool_name'],
            buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)
        )
        
        # Agent sessions
        self.agent_sessions_total = Counter(
            'vayu_agent_sessions_total',
            'Total agent sessions',
            ['agent_name', 'status']
        )
        
        self.agent_active_sessions = Gauge(
            'vayu_agent_active_sessions',
            'Currently active agent sessions',
            ['agent_name']
        )
        
        # Intermediate steps per execution
        self.agent_steps_per_execution = Histogram(
            'vayu_agent_steps_per_execution',
            'Number of intermediate steps per agent execution',
            ['agent_name'],
            buckets=(1, 2, 3, 5, 7, 10, 15, 20)
        )
        
        # =====================================================================
        # AGENTIC EVALUATION METRICS
        # =====================================================================
        
        # Task adherence scores
        self.agentic_task_adherence = Histogram(
            'vayu_agentic_task_adherence',
            'Task adherence score distribution',
            ['agent_name'],
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
        )
        
        # Tool call accuracy scores
        self.agentic_tool_accuracy = Histogram(
            'vayu_agentic_tool_accuracy',
            'Tool call accuracy score distribution',
            ['agent_name'],
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
        )
        
        # Intent resolution scores
        self.agentic_intent_resolution = Histogram(
            'vayu_agentic_intent_resolution',
            'Intent resolution score distribution',
            ['agent_name'],
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
        )
        
        # Overall evaluation scores
        self.agentic_overall_score = Histogram(
            'vayu_agentic_overall_score',
            'Overall agentic evaluation score',
            ['agent_name'],
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
        )
        
        # Evaluations performed
        self.agentic_evaluations_total = Counter(
            'vayu_agentic_evaluations_total',
            'Total agentic evaluations performed',
            ['agent_name', 'score_category']  # excellent, good, acceptable, poor, failed
        )
        
        # =====================================================================
        # HTTP REQUEST METRICS
        # =====================================================================
        
        self.http_requests_total = Counter(
            'vayu_http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status_code']
        )
        
        self.http_request_duration = Histogram(
            'vayu_http_request_duration_seconds',
            'HTTP request duration',
            ['method', 'endpoint'],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
        )
        
        self.http_request_size = Histogram(
            'vayu_http_request_size_bytes',
            'HTTP request body size',
            ['method', 'endpoint'],
            buckets=(100, 1000, 10000, 100000, 1000000)
        )
        
        self.http_response_size = Histogram(
            'vayu_http_response_size_bytes',
            'HTTP response body size',
            ['method', 'endpoint'],
            buckets=(100, 1000, 10000, 100000, 1000000)
        )
        
        # =====================================================================
        # CHAT/QUERY METRICS
        # =====================================================================
        
        self.chat_queries_total = Counter(
            'vayu_chat_queries_total',
            'Total chat queries processed',
            ['source', 'intent']  # source: webui, api, widget
        )
        
        self.chat_query_duration = Histogram(
            'vayu_chat_query_duration_seconds',
            'End-to-end chat query duration',
            ['source'],
            buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)
        )
        
        self.chat_response_length = Histogram(
            'vayu_chat_response_length_chars',
            'Chat response length in characters',
            ['source'],
            buckets=(100, 250, 500, 1000, 2500, 5000, 10000)
        )
        
        # =====================================================================
        # SYSTEM/HEALTH METRICS
        # =====================================================================
        
        self.database_connections = Gauge(
            'vayu_database_connections',
            'Active database connections',
            ['database']  # postgres, milvus, redis
        )
        
        self.cache_hits_total = Counter(
            'vayu_cache_hits_total',
            'Cache hits',
            ['cache_type']
        )
        
        self.cache_misses_total = Counter(
            'vayu_cache_misses_total',
            'Cache misses',
            ['cache_type']
        )
        
        # Application info
        self.app_info = Info(
            'vayu_app',
            'Application information'
        )
        
        # Set application info
        self.app_info.info({
            'name': 'Vayu Maya',
            'version': os.getenv('APP_VERSION', '2.0.0'),
            'environment': os.getenv('ENVIRONMENT', 'production'),
        })
        
        self._initialized = True
        
        # =====================================================================
        # DEBUG: LLM Call Source Tracking
        # =====================================================================
        self._llm_call_log: deque = deque(maxlen=100)  # Keep last 100 calls
        self._llm_call_sources: Dict[str, int] = {}  # Track call sources
        
        logger.info("✅ Prometheus metrics initialized")
    
    # =========================================================================
    # DEBUG: LLM Call Tracking Methods
    # =========================================================================
    
    def get_llm_call_log(self) -> List[Dict]:
        """Get the recent LLM call log for debugging."""
        return list(self._llm_call_log)
    
    def get_llm_call_sources(self) -> Dict[str, int]:
        """Get a summary of where LLM calls are coming from."""
        return dict(self._llm_call_sources)
    
    def clear_llm_call_log(self):
        """Clear the LLM call log."""
        self._llm_call_log.clear()
        self._llm_call_sources.clear()
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def track_llm_call(
        self,
        model: str,
        operation: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        duration: float = 0.0,
        success: bool = True,
        error_type: Optional[str] = None
    ):
        """
        Track a complete LLM API call with all metrics.
        
        Args:
            model: LLM model name (e.g., 'gpt-4', 'gpt-3.5-turbo')
            operation: Operation type ('chat', 'embedding', 'completion')
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            duration: Call duration in seconds
            success: Whether the call succeeded
            error_type: Error type if failed
        """
        # DEBUG: Capture call stack to identify source
        stack = traceback.extract_stack()
        # Get the relevant frames (skip the last 2 which are this function and its caller)
        relevant_frames = stack[:-2]
        # Get the most relevant caller (last 5 frames)
        caller_info = []
        for frame in relevant_frames[-5:]:
            caller_info.append(f"{frame.filename.split('/')[-1]}:{frame.lineno}:{frame.name}")
        
        caller_key = " -> ".join(caller_info[-3:]) if caller_info else "unknown"
        
        # Log this call for debugging
        call_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "model": model,
            "operation": operation,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "duration": round(duration, 3),
            "success": success,
            "error_type": error_type,
            "call_stack": caller_info,
            "caller_summary": caller_key
        }
        self._llm_call_log.append(call_entry)
        
        # Track source counts
        if caller_key in self._llm_call_sources:
            self._llm_call_sources[caller_key] += 1
        else:
            self._llm_call_sources[caller_key] = 1
        
        # Log every call for debugging
        logger.info(f"📊 LLM CALL #{len(self._llm_call_log)}: model={model}, op={operation}, "
                   f"tokens={input_tokens}/{output_tokens}, success={success}, "
                   f"source={caller_key}")
        
        # Track token counts
        if input_tokens > 0:
            self.llm_tokens_total.labels(
                model=model, type='input', operation=operation
            ).inc(input_tokens)
        
        if output_tokens > 0:
            self.llm_tokens_total.labels(
                model=model, type='output', operation=operation
            ).inc(output_tokens)
        
        # Track call duration
        self.llm_call_duration.labels(
            model=model, operation=operation
        ).observe(duration)
        
        # Track call count
        status = 'success' if success else 'error'
        self.llm_calls_total.labels(
            model=model, operation=operation, status=status
        ).inc()
        
        # Track errors
        if not success and error_type:
            self.llm_errors_total.labels(
                model=model, error_type=error_type
            ).inc()
        
        # Estimate cost (approximate pricing)
        if success:
            cost = self._estimate_cost(model, input_tokens, output_tokens)
            self.llm_estimated_cost.labels(
                model=model, operation=operation
            ).inc(cost)
    
    def _estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost based on model and token counts."""
        # Approximate pricing per 1K tokens (USD)
        pricing = {
            'gpt-4': {'input': 0.03, 'output': 0.06},
            'gpt-4-turbo': {'input': 0.01, 'output': 0.03},
            'gpt-4o': {'input': 0.005, 'output': 0.015},
            'gpt-4o-mini': {'input': 0.00015, 'output': 0.0006},
            'gpt-3.5-turbo': {'input': 0.0005, 'output': 0.0015},
            'text-embedding-3-small': {'input': 0.00002, 'output': 0},
            'text-embedding-3-large': {'input': 0.00013, 'output': 0},
            'text-embedding-ada-002': {'input': 0.0001, 'output': 0},
        }
        
        # Find matching model (partial match)
        model_lower = model.lower()
        rates = {'input': 0.01, 'output': 0.03}  # default
        
        for model_key, model_rates in pricing.items():
            if model_key in model_lower:
                rates = model_rates
                break
        
        input_cost = (input_tokens / 1000) * rates['input']
        output_cost = (output_tokens / 1000) * rates['output']
        
        return input_cost + output_cost
    
    def track_agentic_evaluation(
        self,
        agent_name: str,
        task_adherence: float,
        tool_accuracy: float,
        intent_resolution: float,
        overall_score: float
    ):
        """
        Track agentic evaluation scores.
        
        Args:
            agent_name: Name of the agent
            task_adherence: Task adherence score (0-1)
            tool_accuracy: Tool call accuracy score (0-1)
            intent_resolution: Intent resolution score (0-1)
            overall_score: Overall weighted score (0-1)
        """
        self.agentic_task_adherence.labels(agent_name=agent_name).observe(task_adherence)
        self.agentic_tool_accuracy.labels(agent_name=agent_name).observe(tool_accuracy)
        self.agentic_intent_resolution.labels(agent_name=agent_name).observe(intent_resolution)
        self.agentic_overall_score.labels(agent_name=agent_name).observe(overall_score)
        
        # Categorize the score
        if overall_score >= 0.9:
            category = 'excellent'
        elif overall_score >= 0.7:
            category = 'good'
        elif overall_score >= 0.5:
            category = 'acceptable'
        elif overall_score >= 0.3:
            category = 'poor'
        else:
            category = 'failed'
        
        self.agentic_evaluations_total.labels(
            agent_name=agent_name, score_category=category
        ).inc()
    
    @contextmanager
    def track_llm_request(self, model: str, operation: str = 'chat'):
        """
        Context manager to track LLM request timing.
        
        Usage:
            with metrics.track_llm_request('gpt-4', 'chat') as tracker:
                response = await llm.chat(...)
                tracker.set_tokens(response.usage.input, response.usage.output)
        """
        start_time = time.time()
        self.llm_active_requests.labels(model=model).inc()
        
        tracker = _LLMTracker(self, model, operation, start_time)
        
        try:
            yield tracker
        except Exception as e:
            tracker.set_error(type(e).__name__)
            raise
        finally:
            self.llm_active_requests.labels(model=model).dec()
            tracker.finish()
    
    @contextmanager
    def track_agent_execution(self, agent_name: str, operation: str = 'execute'):
        """Context manager to track agent execution timing."""
        start_time = time.time()
        self.agent_active_sessions.labels(agent_name=agent_name).inc()
        
        try:
            yield
            self.agent_sessions_total.labels(
                agent_name=agent_name, status='success'
            ).inc()
        except Exception:
            self.agent_sessions_total.labels(
                agent_name=agent_name, status='error'
            ).inc()
            raise
        finally:
            duration = time.time() - start_time
            self.agent_execution_duration.labels(
                agent_name=agent_name, operation=operation
            ).observe(duration)
            self.agent_active_sessions.labels(agent_name=agent_name).dec()
    
    def track_tool_call(
        self,
        agent_name: str,
        tool_name: str,
        duration: float,
        success: bool = True
    ):
        """Track a tool call made by an agent."""
        status = 'success' if success else 'error'
        self.agent_tool_calls_total.labels(
            agent_name=agent_name, tool_name=tool_name, status=status
        ).inc()
        self.agent_tool_call_duration.labels(tool_name=tool_name).observe(duration)
    
    def track_rag_retrieval(
        self,
        source: str,
        duration: float,
        doc_count: int,
        avg_relevance: float = 0.0
    ):
        """Track RAG document retrieval."""
        self.rag_retrieval_duration.labels(source=source).observe(duration)
        self.rag_documents_retrieved.labels(source=source).observe(doc_count)
        if avg_relevance > 0:
            self.rag_relevance_score.labels(source=source).observe(avg_relevance)
    
    def track_chat_query(
        self,
        source: str,
        intent: str,
        duration: float,
        response_length: int
    ):
        """Track a chat query."""
        self.chat_queries_total.labels(source=source, intent=intent).inc()
        self.chat_query_duration.labels(source=source).observe(duration)
        self.chat_response_length.labels(source=source).observe(response_length)


class _LLMTracker:
    """Helper class for tracking LLM requests."""
    
    def __init__(self, metrics: PrometheusMetrics, model: str, operation: str, start_time: float):
        self.metrics = metrics
        self.model = model
        self.operation = operation
        self.start_time = start_time
        self.input_tokens = 0
        self.output_tokens = 0
        self.error_type: Optional[str] = None
    
    def set_tokens(self, input_tokens: int, output_tokens: int):
        """Set token counts."""
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
    
    def set_error(self, error_type: str):
        """Set error type."""
        self.error_type = error_type
    
    def finish(self):
        """Finalize tracking."""
        duration = time.time() - self.start_time
        success = self.error_type is None
        
        self.metrics.track_llm_call(
            model=self.model,
            operation=self.operation,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            duration=duration,
            success=success,
            error_type=self.error_type
        )


# Singleton instance
_metrics_instance: Optional[PrometheusMetrics] = None


def get_metrics() -> PrometheusMetrics:
    """Get or create the singleton metrics instance."""
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = PrometheusMetrics()
    return _metrics_instance


# Convenience export
metrics = get_metrics()


def llm_metrics(model: str, operation: str = 'chat'):
    """
    Decorator to automatically track LLM calls.
    
    Usage:
        @llm_metrics('gpt-4', 'chat')
        async def call_gpt4(prompt: str):
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            with metrics.track_llm_request(model, operation) as tracker:
                result = await func(*args, **kwargs)
                # Try to extract token usage from result
                if hasattr(result, 'usage'):
                    usage = result.usage
                    if hasattr(usage, 'prompt_tokens'):
                        tracker.set_tokens(
                            usage.prompt_tokens or 0,
                            usage.completion_tokens or 0
                        )
                return result
        return wrapper
    return decorator


def agent_metrics(agent_name: str):
    """
    Decorator to automatically track agent execution.
    
    Usage:
        @agent_metrics('OrchestratorAgent')
        async def execute(self, query: str):
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            with metrics.track_agent_execution(agent_name):
                return await func(*args, **kwargs)
        return wrapper
    return decorator

