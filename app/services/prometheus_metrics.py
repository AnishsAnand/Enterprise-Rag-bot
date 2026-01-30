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
    with metrics.track_llm_request(model="gpt-4", operation="chat") as tracker:
        response = await llm.chat(...)
        tracker.set_tokens(response.usage.prompt_tokens, response.usage.completion_tokens)
    
    # Or use direct tracking
    metrics.track_llm_call(
        model="gpt-4",
        operation="chat",
        input_tokens=100,
        output_tokens=200,
        duration=1.5,
        success=True
    )
"""

import os
import time
import logging
import traceback
import threading
from typing import Optional, Callable, List, Dict, Any
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

# ============================================================================
# PRODUCTION: Thread-safe singleton with lock
# ============================================================================
_metrics_instance: Optional['PrometheusMetrics'] = None
_metrics_lock = threading.Lock()


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
    
    Thread-safe singleton implementation with enhanced error handling
    and comprehensive call tracking for production environments.
    """
    
    # =========================================================================
    # PRODUCTION: Standardized Error Types
    # =========================================================================
    ERROR_TYPES = {
        'rate_limit': ['429', 'rate_limit', 'too_many_requests', 'quota_exceeded'],
        'timeout': ['timeout', 'timed_out', 'deadline_exceeded', 'request_timeout'],
        'auth_error': ['401', '403', 'unauthorized', 'forbidden', 'invalid_api_key', 'authentication'],
        'server_error': ['500', '502', '503', '504', 'internal_error', 'service_unavailable', 'bad_gateway'],
        'invalid_request': ['400', 'invalid_request', 'bad_request', 'validation_error'],
        'not_found': ['404', 'not_found', 'model_not_found'],
        'connection_error': ['connection', 'network', 'dns', 'connection_reset'],
        'context_length': ['context_length', 'token_limit', 'max_tokens'],
        'content_filter': ['content_filter', 'content_policy', 'safety'],
        'unknown': []
    }
    
    # =========================================================================
    # PRODUCTION: Updated Pricing (Q1 2025)
    # =========================================================================
    PRICING = {
        # OpenAI GPT-4 family
        'gpt-4o': {'input': 0.0025, 'output': 0.010},
        'gpt-4o-mini': {'input': 0.00015, 'output': 0.0006},
        'gpt-4-turbo': {'input': 0.01, 'output': 0.03},
        'gpt-4': {'input': 0.03, 'output': 0.06},
        'gpt-3.5-turbo': {'input': 0.0005, 'output': 0.0015},
        
        # OpenAI Embeddings
        'text-embedding-3-small': {'input': 0.00002, 'output': 0},
        'text-embedding-3-large': {'input': 0.00013, 'output': 0},
        'text-embedding-ada-002': {'input': 0.0001, 'output': 0},
        
        # Anthropic Claude
        'claude-3-opus': {'input': 0.015, 'output': 0.075},
        'claude-3-sonnet': {'input': 0.003, 'output': 0.015},
        'claude-3-haiku': {'input': 0.00025, 'output': 0.00125},
        'claude-3.5-sonnet': {'input': 0.003, 'output': 0.015},
        
        # Open source / other
        'qwen': {'input': 0.0001, 'output': 0.0001},
        'llama': {'input': 0.0002, 'output': 0.0002},
        'mistral': {'input': 0.0002, 'output': 0.0002},
    }
    
    def __init__(self):
        """Initialize all Prometheus metrics with production-grade configuration."""
        self._initialized = False
        self._registry = REGISTRY
        
        # =====================================================================
        # LLM METRICS - Track all LLM API interactions
        # =====================================================================
        
        self.llm_tokens_total = Counter(
            'vayu_llm_tokens_total',
            'Total tokens processed by LLM',
            ['model', 'type', 'operation']  # type: input/output, operation: chat/embedding/completion
        )
        
        self.llm_call_duration = Histogram(
            'vayu_llm_call_duration_seconds',
            'LLM API call duration in seconds',
            ['model', 'operation'],
            buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0)
        )
        
        self.llm_calls_total = Counter(
            'vayu_llm_calls_total',
            'Total LLM API calls',
            ['model', 'operation', 'status']
        )
        
        self.llm_estimated_cost = Counter(
            'vayu_llm_estimated_cost_usd',
            'Estimated LLM API cost in USD',
            ['model', 'operation']
        )
        
        self.llm_active_requests = Gauge(
            'vayu_llm_active_requests',
            'Currently active LLM API requests',
            ['model']
        )
        
        self.llm_errors_total = Counter(
            'vayu_llm_errors_total',
            'LLM API errors by type',
            ['model', 'error_type']
        )
        
        # Additional LLM metrics
        self.llm_tokens_per_second = Histogram(
            'vayu_llm_tokens_per_second',
            'Token generation speed (tokens/second)',
            ['model'],
            buckets=(10, 25, 50, 100, 200, 500, 1000, 2000)
        )
        
        self.llm_retry_attempts = Counter(
            'vayu_llm_retry_attempts_total',
            'Total retry attempts for LLM calls',
            ['model', 'error_type']
        )
        
        # =====================================================================
        # RAG PIPELINE METRICS
        # =====================================================================
        
        self.rag_retrieval_duration = Histogram(
            'vayu_rag_retrieval_duration_seconds',
            'RAG document retrieval duration',
            ['source'],
            buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
        )
        
        self.rag_documents_retrieved = Histogram(
            'vayu_rag_documents_retrieved',
            'Number of documents retrieved per query',
            ['source'],
            buckets=(1, 3, 5, 10, 20, 50, 100, 200)
        )
        
        self.rag_relevance_score = Histogram(
            'vayu_rag_relevance_score',
            'Relevance scores of retrieved documents',
            ['source'],
            buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
        )
        
        self.embedding_duration = Histogram(
            'vayu_embedding_duration_seconds',
            'Embedding generation duration',
            ['model', 'batch_size'],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
        )
        
        self.embedding_dimensions = Gauge(
            'vayu_embedding_dimensions',
            'Embedding vector dimensions',
            ['model']
        )
        
        self.rerank_duration = Histogram(
            'vayu_rerank_duration_seconds',
            'Reranking operation duration',
            ['method'],
            buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
        )
        
        self.rag_cache_hit_rate = Gauge(
            'vayu_rag_cache_hit_rate',
            'RAG cache hit rate percentage',
            ['cache_type']
        )
        
        # =====================================================================
        # AGENT EXECUTION METRICS
        # =====================================================================
        
        self.agent_execution_duration = Histogram(
            'vayu_agent_execution_duration_seconds',
            'Agent execution duration',
            ['agent_name', 'operation'],
            buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0)
        )
        
        self.agent_tool_calls_total = Counter(
            'vayu_agent_tool_calls_total',
            'Total tool calls made by agents',
            ['agent_name', 'tool_name', 'status']
        )
        
        self.agent_tool_call_duration = Histogram(
            'vayu_agent_tool_call_duration_seconds',
            'Tool call execution duration',
            ['tool_name'],
            buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)
        )
        
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
        
        self.agent_steps_per_execution = Histogram(
            'vayu_agent_steps_per_execution',
            'Number of intermediate steps per agent execution',
            ['agent_name'],
            buckets=(1, 2, 3, 5, 7, 10, 15, 20, 30, 50)
        )
        
        # =====================================================================
        # AGENTIC EVALUATION METRICS
        # =====================================================================
        
        self.agentic_task_adherence = Histogram(
            'vayu_agentic_task_adherence',
            'Task adherence score distribution',
            ['agent_name'],
            buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
        )
        
        self.agentic_tool_accuracy = Histogram(
            'vayu_agentic_tool_accuracy',
            'Tool call accuracy score distribution',
            ['agent_name'],
            buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
        )
        
        self.agentic_intent_resolution = Histogram(
            'vayu_agentic_intent_resolution',
            'Intent resolution score distribution',
            ['agent_name'],
            buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
        )
        
        self.agentic_overall_score = Histogram(
            'vayu_agentic_overall_score',
            'Overall agentic evaluation score',
            ['agent_name'],
            buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
        )
        
        self.agentic_evaluations_total = Counter(
            'vayu_agentic_evaluations_total',
            'Total agentic evaluations performed',
            ['agent_name', 'score_category']
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
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)
        )
        
        self.http_request_size = Histogram(
            'vayu_http_request_size_bytes',
            'HTTP request body size',
            ['method', 'endpoint'],
            buckets=(100, 1000, 10000, 100000, 1000000, 10000000)
        )
        
        self.http_response_size = Histogram(
            'vayu_http_response_size_bytes',
            'HTTP response body size',
            ['method', 'endpoint'],
            buckets=(100, 1000, 10000, 100000, 1000000, 10000000)
        )
        
        # =====================================================================
        # CHAT/QUERY METRICS
        # =====================================================================
        
        self.chat_queries_total = Counter(
            'vayu_chat_queries_total',
            'Total chat queries processed',
            ['source', 'intent']
        )
        
        self.chat_query_duration = Histogram(
            'vayu_chat_query_duration_seconds',
            'End-to-end chat query duration',
            ['source'],
            buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0)
        )
        
        self.chat_response_length = Histogram(
            'vayu_chat_response_length_chars',
            'Chat response length in characters',
            ['source'],
            buckets=(100, 250, 500, 1000, 2500, 5000, 10000, 25000)
        )
        
        self.chat_user_satisfaction = Histogram(
            'vayu_chat_user_satisfaction',
            'User satisfaction scores (if collected)',
            ['source'],
            buckets=(1, 2, 3, 4, 5)
        )
        
        # =====================================================================
        # SYSTEM/HEALTH METRICS
        # =====================================================================
        
        self.database_connections = Gauge(
            'vayu_database_connections',
            'Active database connections',
            ['database']
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
        
        self.memory_usage_bytes = Gauge(
            'vayu_memory_usage_bytes',
            'Application memory usage in bytes',
            ['type']  # rss, vms, shared
        )
        
        self.active_connections = Gauge(
            'vayu_active_connections',
            'Active connections by type',
            ['connection_type']
        )
        
        # Application info
        self.app_info = Info('vayu_app', 'Application information')
        self.app_info.info({
            'name': 'Vayu Maya RAG Bot',
            'version': os.getenv('APP_VERSION', '2.0.0'),
            'environment': os.getenv('ENVIRONMENT', 'production'),
            'python_version': os.getenv('PYTHON_VERSION', '3.11'),
            'deployment_date': datetime.utcnow().isoformat(),
        })
        
        # =====================================================================
        # DEBUG: Enhanced Call Tracking (Thread-safe)
        # =====================================================================
        self._llm_call_log: deque = deque(maxlen=1000)  # Last 1000 calls
        self._llm_call_sources: Dict[str, int] = {}
        self._call_log_lock = threading.Lock()
        
        # Performance tracking
        self._performance_window: deque = deque(maxlen=100)  # Last 100 operations
        
        self._initialized = True
        logger.info("✅ Vayu Maya Prometheus metrics initialized successfully")
    
    # =========================================================================
    # PRODUCTION: Standardized Error Type Classification
    # =========================================================================
    
    @classmethod
    def standardize_error_type(cls, error: Any) -> str:
        """
        Standardize error types for consistent tracking across the application.
        
        Args:
            error: Error object, exception, or string
            
        Returns:
            Standardized error type string
        """
        if isinstance(error, str):
            error_str = error.lower()
        elif isinstance(error, Exception):
            error_str = f"{type(error).__name__} {str(error)}".lower()
        else:
            error_str = str(type(error).__name__).lower()
        
        for standard_type, patterns in cls.ERROR_TYPES.items():
            if any(pattern in error_str for pattern in patterns):
                return standard_type
        
        return 'unknown'
    
    # =========================================================================
    # PRODUCTION: Enhanced Call Stack Analysis
    # =========================================================================
    
    def _extract_caller_info(self) -> tuple[List[Dict[str, Any]], str]:
        """
        Extract meaningful caller information from stack trace.
        
        Returns:
            Tuple of (relevant_frames, caller_summary)
        """
        stack = traceback.extract_stack()
        
        # Patterns to skip (framework and internal files)
        skip_patterns = [
            'prometheus_metrics.py',
            'threading.py',
            'asyncio',
            'contextlib.py',
            'functools.py',
            'site-packages',
            '<frozen',
        ]
        
        relevant_frames = []
        for frame in stack[:-2]:  # Skip current and immediate caller
            filename = frame.filename.split('/')[-1]
            
            # Skip internal/framework files
            if any(pattern in filename for pattern in skip_patterns):
                continue
            
            relevant_frames.append({
                'file': filename,
                'line': frame.lineno,
                'function': frame.name,
                'code': frame.line
            })
        
        # Build caller summary (last 3 relevant frames)
        caller_parts = []
        for frame in relevant_frames[-3:]:
            caller_parts.append(f"{frame['file']}:{frame['function']}")
        
        caller_summary = " -> ".join(caller_parts) if caller_parts else "unknown"
        
        return relevant_frames, caller_summary
    
    # =========================================================================
    # CORE TRACKING METHODS
    # =========================================================================
    
    def track_llm_call(
        self,
        model: str,
        operation: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        duration: float = 0.0,
        success: bool = True,
        error_type: Optional[str] = None,
        retry_count: int = 0
    ):
        """
        Track a complete LLM API call with all metrics.
        
        PRODUCTION FEATURES:
        - Standardized error types
        - Thread-safe logging
        - Enhanced stack trace analysis
        - Cost estimation with latest pricing
        - Performance metrics (tokens/second)
        - Retry tracking
        
        Args:
            model: LLM model name (e.g., 'gpt-4o', 'claude-3-sonnet')
            operation: Operation type ('chat', 'embedding', 'completion')
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            duration: Call duration in seconds
            success: Whether the call succeeded
            error_type: Error type if failed (will be standardized)
            retry_count: Number of retry attempts made
        """
        # Extract caller information
        relevant_frames, caller_summary = self._extract_caller_info()
        
        # Standardize error type
        if error_type:
            error_type = self.standardize_error_type(error_type)
        
        # Normalize token counts
        input_tokens = max(0, input_tokens)
        output_tokens = max(0, output_tokens)
        duration = max(0.0, duration)
        
        # Calculate tokens per second
        tokens_per_second = 0.0
        if duration > 0 and output_tokens > 0:
            tokens_per_second = output_tokens / duration
        
        # Build call entry for debugging
        call_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "model": model,
            "operation": operation,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "duration": round(duration, 3),
            "tokens_per_second": round(tokens_per_second, 2),
            "success": success,
            "error_type": error_type,
            "retry_count": retry_count,
            "caller_summary": caller_summary,
            "call_stack": [
                f"{f['file']}:{f['line']}:{f['function']}"
                for f in relevant_frames[-5:]  # Last 5 frames
            ]
        }
        
        # Thread-safe logging
        with self._call_log_lock:
            self._llm_call_log.append(call_entry)
            self._llm_call_sources[caller_summary] = \
                self._llm_call_sources.get(caller_summary, 0) + 1
        
        # Debug logging (controlled by environment variable)
        if os.getenv('DEBUG_LLM_CALLS', 'false').lower() == 'true':
            logger.debug(
                f"📊 LLM CALL: model={model}, op={operation}, "
                f"tokens={input_tokens}/{output_tokens} ({tokens_per_second:.0f} tok/s), "
                f"duration={duration:.3f}s, success={success}, "
                f"retries={retry_count}, source={caller_summary}"
            )
        
        # Update Prometheus metrics
        try:
            # Token counts
            if input_tokens > 0:
                self.llm_tokens_total.labels(
                    model=model, type='input', operation=operation
                ).inc(input_tokens)
            
            if output_tokens > 0:
                self.llm_tokens_total.labels(
                    model=model, type='output', operation=operation
                ).inc(output_tokens)
            
            # Call duration
            if duration > 0:
                self.llm_call_duration.labels(
                    model=model, operation=operation
                ).observe(duration)
            
            # Tokens per second
            if tokens_per_second > 0:
                self.llm_tokens_per_second.labels(model=model).observe(tokens_per_second)
            
            # Call count
            status = 'success' if success else 'error'
            self.llm_calls_total.labels(
                model=model, operation=operation, status=status
            ).inc()
            
            # Retry tracking
            if retry_count > 0:
                self.llm_retry_attempts.labels(
                    model=model, error_type=error_type or 'unknown'
                ).inc(retry_count)
            
            # Error tracking
            if not success and error_type:
                self.llm_errors_total.labels(
                    model=model, error_type=error_type
                ).inc()
            
            # Cost estimation
            if success and (input_tokens > 0 or output_tokens > 0):
                cost = self._estimate_cost(model, input_tokens, output_tokens)
                if cost > 0:
                    self.llm_estimated_cost.labels(
                        model=model, operation=operation
                    ).inc(cost)
        
        except Exception as e:
            logger.error(f"❌ Error updating Prometheus metrics: {e}", exc_info=True)
    
    def _estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """
        Estimate cost based on model and token counts using latest pricing.
        
        Args:
            model: Model name
            input_tokens: Input token count
            output_tokens: Output token count
            
        Returns:
            Estimated cost in USD
        """
        model_lower = model.lower()
        rates = {'input': 0.001, 'output': 0.002}  # Conservative default
        
        # Find matching model with partial match
        for model_key, model_rates in self.PRICING.items():
            if model_key in model_lower:
                rates = model_rates
                break
        
        input_cost = (input_tokens / 1000) * rates['input']
        output_cost = (output_tokens / 1000) * rates['output']
        
        return input_cost + output_cost
    
    # =========================================================================
    # AGENTIC EVALUATION TRACKING
    # =========================================================================
    
    def track_agentic_evaluation(
        self,
        agent_name: str,
        task_adherence: float,
        tool_accuracy: float,
        intent_resolution: float,
        overall_score: float
    ):
        """
        Track agentic evaluation scores with proper normalization.
        
        Args:
            agent_name: Name of the agent being evaluated
            task_adherence: Task adherence score (0-1)
            tool_accuracy: Tool call accuracy score (0-1)
            intent_resolution: Intent resolution score (0-1)
            overall_score: Overall weighted score (0-1)
        """
        # Clamp all scores to [0, 1] range
        task_adherence = max(0.0, min(1.0, task_adherence))
        tool_accuracy = max(0.0, min(1.0, tool_accuracy))
        intent_resolution = max(0.0, min(1.0, intent_resolution))
        overall_score = max(0.0, min(1.0, overall_score))
        
        # Record individual metrics
        self.agentic_task_adherence.labels(agent_name=agent_name).observe(task_adherence)
        self.agentic_tool_accuracy.labels(agent_name=agent_name).observe(tool_accuracy)
        self.agentic_intent_resolution.labels(agent_name=agent_name).observe(intent_resolution)
        self.agentic_overall_score.labels(agent_name=agent_name).observe(overall_score)
        
        # Categorize overall score
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
        
        logger.info(
            f"📈 Agentic Evaluation - {agent_name}: "
            f"overall={overall_score:.3f} ({category}), "
            f"task={task_adherence:.3f}, tool={tool_accuracy:.3f}, "
            f"intent={intent_resolution:.3f}"
        )
    
    # =========================================================================
    # CONTEXT MANAGERS FOR AUTOMATIC TRACKING
    # =========================================================================
    
    @contextmanager
    def track_llm_request(self, model: str, operation: str = 'chat'):
        """
        Context manager to track LLM request timing and metrics.
        
        Usage:
            with metrics.track_llm_request('gpt-4o', 'chat') as tracker:
                response = await llm.chat(...)
                tracker.set_tokens(response.usage.prompt_tokens, response.usage.completion_tokens)
        
        Args:
            model: Model name
            operation: Operation type
            
        Yields:
            _LLMTracker: Tracker object for setting tokens and errors
        """
        start_time = time.time()
        self.llm_active_requests.labels(model=model).inc()
        
        tracker = _LLMTracker(self, model, operation, start_time)
        
        try:
            yield tracker
        except Exception as e:
            error_type = self.standardize_error_type(e)
            tracker.set_error(error_type)
            raise
        finally:
            self.llm_active_requests.labels(model=model).dec()
            tracker.finish()
    
    @contextmanager
    def track_agent_execution(self, agent_name: str, operation: str = 'execute'):
        """
        Context manager to track agent execution timing.
        
        Usage:
            with metrics.track_agent_execution('OrchestratorAgent'):
                result = await agent.execute(query)
        
        Args:
            agent_name: Name of the agent
            operation: Operation being performed
        """
        start_time = time.time()
        self.agent_active_sessions.labels(agent_name=agent_name).inc()
        
        try:
            yield
            self.agent_sessions_total.labels(
                agent_name=agent_name, status='success'
            ).inc()
        except Exception as e:
            self.agent_sessions_total.labels(
                agent_name=agent_name, status='error'
            ).inc()
            logger.error(f"❌ Agent {agent_name} execution failed: {e}")
            raise
        finally:
            duration = time.time() - start_time
            self.agent_execution_duration.labels(
                agent_name=agent_name, operation=operation
            ).observe(duration)
            self.agent_active_sessions.labels(agent_name=agent_name).dec()
    
    # =========================================================================
    # CONVENIENCE TRACKING METHODS
    # =========================================================================
    
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
        """Track RAG document retrieval metrics."""
        self.rag_retrieval_duration.labels(source=source).observe(duration)
        self.rag_documents_retrieved.labels(source=source).observe(doc_count)
        if avg_relevance > 0:
            avg_relevance = max(0.0, min(1.0, avg_relevance))
            self.rag_relevance_score.labels(source=source).observe(avg_relevance)
    
    def track_chat_query(
        self,
        source: str,
        intent: str,
        duration: float,
        response_length: int
    ):
        """Track a complete chat query."""
        self.chat_queries_total.labels(source=source, intent=intent).inc()
        self.chat_query_duration.labels(source=source).observe(duration)
        self.chat_response_length.labels(source=source).observe(response_length)
    
    # =========================================================================
    # DEBUG AND MONITORING METHODS
    # =========================================================================
    
    def get_llm_call_log(self, limit: int = 100) -> List[Dict]:
        """
        Get recent LLM call log for debugging.
        
        Args:
            limit: Maximum number of calls to return
            
        Returns:
            List of call entries
        """
        with self._call_log_lock:
            return list(self._llm_call_log)[-limit:]
    
    def get_llm_call_sources(self) -> Dict[str, int]:
        """
        Get summary of where LLM calls originate.
        
        Returns:
            Dictionary mapping source to call count
        """
        with self._call_log_lock:
            return dict(self._llm_call_sources)
    
    def get_llm_call_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive LLM call statistics.
        
        Returns:
            Dictionary with statistics
        """
        with self._call_log_lock:
            total_calls = len(self._llm_call_log)
            if total_calls == 0:
                return {
                    'total_calls': 0,
                    'successful_calls': 0,
                    'failed_calls': 0,
                    'success_rate': 0.0,
                    'avg_duration': 0.0,
                    'avg_tokens_per_second': 0.0,
                    'total_input_tokens': 0,
                    'total_output_tokens': 0,
                    'total_tokens': 0,
                    'estimated_total_cost': 0.0,
                    'top_sources': []
                }
            
            successful = sum(1 for call in self._llm_call_log if call['success'])
            failed = total_calls - successful
            
            durations = [call['duration'] for call in self._llm_call_log]
            avg_duration = sum(durations) / total_calls
            
            tokens_per_second_values = [
                call['tokens_per_second'] 
                for call in self._llm_call_log 
                if call['tokens_per_second'] > 0
            ]
            avg_tokens_per_second = (
                sum(tokens_per_second_values) / len(tokens_per_second_values)
                if tokens_per_second_values else 0.0
            )
            
            total_input_tokens = sum(call['input_tokens'] for call in self._llm_call_log)
            total_output_tokens = sum(call['output_tokens'] for call in self._llm_call_log)
            
            # Top sources
            top_sources = sorted(
                self._llm_call_sources.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            return {
                'total_calls': total_calls,
                'successful_calls': successful,
                'failed_calls': failed,
                'success_rate': round(successful / total_calls, 3),
                'avg_duration': round(avg_duration, 3),
                'avg_tokens_per_second': round(avg_tokens_per_second, 2),
                'total_input_tokens': total_input_tokens,
                'total_output_tokens': total_output_tokens,
                'total_tokens': total_input_tokens + total_output_tokens,
                'top_sources': top_sources
            }
    
    def clear_llm_call_log(self):
        """Clear the LLM call log and source tracking."""
        with self._call_log_lock:
            self._llm_call_log.clear()
            self._llm_call_sources.clear()
        logger.info("🧹 LLM call log cleared")
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get overall health status of the metrics system.
        
        Returns:
            Health status dictionary
        """
        return {
            'initialized': self._initialized,
            'total_llm_calls_tracked': len(self._llm_call_log),
            'unique_call_sources': len(self._llm_call_sources),
            'metrics_registry_collectors': len(self._registry._collector_to_names),
        }


class _LLMTracker:
    """
    Helper class for tracking LLM requests with context manager.
    
    This class is used internally by track_llm_request() context manager
    to accumulate metrics during a request and finalize them on completion.
    """
    
    def __init__(
        self, 
        metrics: PrometheusMetrics, 
        model: str, 
        operation: str, 
        start_time: float
    ):
        self.metrics = metrics
        self.model = model
        self.operation = operation
        self.start_time = start_time
        self.input_tokens = 0
        self.output_tokens = 0
        self.error_type: Optional[str] = None
        self.retry_count = 0
    
    def set_tokens(self, input_tokens: int, output_tokens: int):
        """Set token counts from response."""
        self.input_tokens = max(0, input_tokens)
        self.output_tokens = max(0, output_tokens)
    
    def set_error(self, error_type: str):
        """Set error type if request failed."""
        self.error_type = error_type
    
    def set_retry_count(self, count: int):
        """Set number of retry attempts made."""
        self.retry_count = max(0, count)
    
    def finish(self):
        """Finalize tracking and record metrics."""
        duration = time.time() - self.start_time
        success = self.error_type is None
        
        self.metrics.track_llm_call(
            model=self.model,
            operation=self.operation,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            duration=duration,
            success=success,
            error_type=self.error_type,
            retry_count=self.retry_count
        )


# ============================================================================
# PRODUCTION: Thread-safe Singleton Pattern
# ============================================================================

def get_metrics() -> PrometheusMetrics:
    """
    Get or create the singleton metrics instance (thread-safe).
    
    This uses double-checked locking pattern to ensure thread-safe
    singleton creation without unnecessary lock contention.
    
    Returns:
        PrometheusMetrics: The singleton metrics instance
    """
    global _metrics_instance
    
    if _metrics_instance is None:
        with _metrics_lock:
            # Double-check locking pattern
            if _metrics_instance is None:
                _metrics_instance = PrometheusMetrics()
    
    return _metrics_instance


# Convenience export - use this in your application
metrics = get_metrics()


# ============================================================================
# DECORATORS FOR AUTOMATIC TRACKING
# ============================================================================

def llm_metrics(model: str, operation: str = 'chat'):
    """
    Decorator to automatically track LLM calls.
    
    Supports both sync and async functions. Automatically extracts
    token usage from response if available.
    
    Usage:
        @llm_metrics('gpt-4o', 'chat')
        async def call_gpt4(prompt: str):
            response = await openai.chat(...)
            return response
    
    Args:
        model: Model name
        operation: Operation type
    """
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with metrics.track_llm_request(model, operation) as tracker:
                result = await func(*args, **kwargs)
                
                # Try to extract token usage from result
                if hasattr(result, 'usage'):
                    usage = result.usage
                    tracker.set_tokens(
                        getattr(usage, 'prompt_tokens', 0),
                        getattr(usage, 'completion_tokens', 0)
                    )
                
                return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with metrics.track_llm_request(model, operation) as tracker:
                result = func(*args, **kwargs)
                
                # Try to extract token usage from result
                if hasattr(result, 'usage'):
                    usage = result.usage
                    tracker.set_tokens(
                        getattr(usage, 'prompt_tokens', 0),
                        getattr(usage, 'completion_tokens', 0)
                    )
                
                return result
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def agent_metrics(agent_name: str):
    """
    Decorator to automatically track agent execution.
    
    Supports both sync and async functions.
    
    Usage:
        @agent_metrics('OrchestratorAgent')
        async def execute(self, query: str):
            # Agent logic here
            return result
    
    Args:
        agent_name: Name of the agent
    """
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with metrics.track_agent_execution(agent_name):
                return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with metrics.track_agent_execution(agent_name):
                return func(*args, **kwargs)
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator