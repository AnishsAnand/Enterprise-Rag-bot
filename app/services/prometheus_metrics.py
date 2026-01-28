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
)

logger = logging.getLogger(__name__)

# ============================================================================
# PRODUCTION FIX: Thread-safe singleton with lock
# ============================================================================
_metrics_instance: Optional['PrometheusMetrics'] = None
_metrics_lock = threading.Lock()


class PrometheusMetrics:
    """
    Centralized Prometheus metrics for the RAG application.
    
    PRODUCTION IMPROVEMENTS:
    - Thread-safe initialization
    - Enhanced error tracking
    - Better call stack analysis
    - Memory-efficient logging
    """
    
    def __init__(self):
        """Initialize all Prometheus metrics."""
        self._initialized = False
        self._registry = REGISTRY
        
        # =====================================================================
        # LLM METRICS
        # =====================================================================
        
        self.llm_tokens_total = Counter(
            'rag_llm_tokens_total',
            'Total tokens processed by LLM',
            ['model', 'type', 'operation']
        )
        
        self.llm_call_duration = Histogram(
            'rag_llm_call_duration_seconds',
            'LLM API call duration in seconds',
            ['model', 'operation'],
            buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)
        )
        
        self.llm_calls_total = Counter(
            'rag_llm_calls_total',
            'Total LLM API calls',
            ['model', 'operation', 'status']
        )
        
        self.llm_estimated_cost = Counter(
            'rag_llm_estimated_cost_usd',
            'Estimated LLM API cost in USD',
            ['model', 'operation']
        )
        
        self.llm_active_requests = Gauge(
            'rag_llm_active_requests',
            'Currently active LLM API requests',
            ['model']
        )
        
        # PRODUCTION FIX: Standardized error types
        self.llm_errors_total = Counter(
            'rag_llm_errors_total',
            'LLM API errors by type',
            ['model', 'error_type']
        )
        
        # =====================================================================
        # RAG PIPELINE METRICS
        # =====================================================================
        
        self.rag_retrieval_duration = Histogram(
            'rag_retrieval_duration_seconds',
            'RAG document retrieval duration',
            ['source'],
            buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
        )
        
        self.rag_documents_retrieved = Histogram(
            'rag_documents_retrieved',
            'Number of documents retrieved per query',
            ['source'],
            buckets=(1, 3, 5, 10, 20, 50, 100, 200)
        )
        
        self.rag_relevance_score = Histogram(
            'rag_relevance_score',
            'Relevance scores of retrieved documents',
            ['source'],
            buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
        )
        
        self.embedding_duration = Histogram(
            'rag_embedding_duration_seconds',
            'Embedding generation duration',
            ['model', 'batch_size'],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
        )
        
        self.embedding_dimensions = Gauge(
            'rag_embedding_dimensions',
            'Embedding vector dimensions',
            ['model']
        )
        
        self.rerank_duration = Histogram(
            'rag_rerank_duration_seconds',
            'Reranking operation duration',
            ['method'],
            buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
        )
        
        # =====================================================================
        # AGENT EXECUTION METRICS
        # =====================================================================
        
        self.agent_execution_duration = Histogram(
            'rag_agent_execution_duration_seconds',
            'Agent execution duration',
            ['agent_name', 'operation'],
            buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0)
        )
        
        self.agent_tool_calls_total = Counter(
            'rag_agent_tool_calls_total',
            'Total tool calls made by agents',
            ['agent_name', 'tool_name', 'status']
        )
        
        self.agent_tool_call_duration = Histogram(
            'rag_agent_tool_call_duration_seconds',
            'Tool call execution duration',
            ['tool_name'],
            buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)
        )
        
        self.agent_sessions_total = Counter(
            'rag_agent_sessions_total',
            'Total agent sessions',
            ['agent_name', 'status']
        )
        
        self.agent_active_sessions = Gauge(
            'rag_agent_active_sessions',
            'Currently active agent sessions',
            ['agent_name']
        )
        
        self.agent_steps_per_execution = Histogram(
            'rag_agent_steps_per_execution',
            'Number of intermediate steps per agent execution',
            ['agent_name'],
            buckets=(1, 2, 3, 5, 7, 10, 15, 20, 30)
        )
        
        # =====================================================================
        # AGENTIC EVALUATION METRICS
        # =====================================================================
        
        self.agentic_task_adherence = Histogram(
            'rag_agentic_task_adherence',
            'Task adherence score distribution',
            ['agent_name'],
            buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
        )
        
        self.agentic_tool_accuracy = Histogram(
            'rag_agentic_tool_accuracy',
            'Tool call accuracy score distribution',
            ['agent_name'],
            buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
        )
        
        self.agentic_intent_resolution = Histogram(
            'rag_agentic_intent_resolution',
            'Intent resolution score distribution',
            ['agent_name'],
            buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
        )
        
        self.agentic_overall_score = Histogram(
            'rag_agentic_overall_score',
            'Overall agentic evaluation score',
            ['agent_name'],
            buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
        )
        
        self.agentic_evaluations_total = Counter(
            'rag_agentic_evaluations_total',
            'Total agentic evaluations performed',
            ['agent_name', 'score_category']
        )
        
        # =====================================================================
        # HTTP REQUEST METRICS
        # =====================================================================
        
        self.http_requests_total = Counter(
            'rag_http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status_code']
        )
        
        self.http_request_duration = Histogram(
            'rag_http_request_duration_seconds',
            'HTTP request duration',
            ['method', 'endpoint'],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)
        )
        
        self.http_request_size = Histogram(
            'rag_http_request_size_bytes',
            'HTTP request body size',
            ['method', 'endpoint'],
            buckets=(100, 1000, 10000, 100000, 1000000, 10000000)
        )
        
        self.http_response_size = Histogram(
            'rag_http_response_size_bytes',
            'HTTP response body size',
            ['method', 'endpoint'],
            buckets=(100, 1000, 10000, 100000, 1000000, 10000000)
        )
        
        # =====================================================================
        # CHAT/QUERY METRICS
        # =====================================================================
        
        self.chat_queries_total = Counter(
            'rag_chat_queries_total',
            'Total chat queries processed',
            ['source', 'intent']
        )
        
        self.chat_query_duration = Histogram(
            'rag_chat_query_duration_seconds',
            'End-to-end chat query duration',
            ['source'],
            buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0)
        )
        
        self.chat_response_length = Histogram(
            'rag_chat_response_length_chars',
            'Chat response length in characters',
            ['source'],
            buckets=(100, 250, 500, 1000, 2500, 5000, 10000, 25000)
        )
        
        # =====================================================================
        # SYSTEM/HEALTH METRICS
        # =====================================================================
        
        self.database_connections = Gauge(
            'rag_database_connections',
            'Active database connections',
            ['database']
        )
        
        self.cache_hits_total = Counter(
            'rag_cache_hits_total',
            'Cache hits',
            ['cache_type']
        )
        
        self.cache_misses_total = Counter(
            'rag_cache_misses_total',
            'Cache misses',
            ['cache_type']
        )
        
        # PRODUCTION FIX: Better version tracking
        self.app_info = Info('rag_app', 'Application information')
        self.app_info.info({
            'name': 'Enterprise RAG Bot',
            'version': os.getenv('APP_VERSION', '2.0.0'),
            'environment': os.getenv('ENVIRONMENT', 'production'),
            'python_version': os.getenv('PYTHON_VERSION', '3.11'),
        })
        
        # =====================================================================
        # DEBUG: Enhanced Call Tracking
        # =====================================================================
        self._llm_call_log: deque = deque(maxlen=1000)  # Increased from 100
        self._llm_call_sources: Dict[str, int] = {}
        self._call_log_lock = threading.Lock()  # Thread safety
        
        self._initialized = True
        logger.info("✅ Prometheus metrics initialized successfully")
    
    # =========================================================================
    # PRODUCTION FIX: Standardized Error Types
    # =========================================================================
    
    ERROR_TYPES = {
        'rate_limit': ['429', 'rate_limit', 'too_many_requests'],
        'timeout': ['timeout', 'timed_out', 'deadline_exceeded'],
        'auth_error': ['401', '403', 'unauthorized', 'forbidden', 'invalid_api_key'],
        'server_error': ['500', '502', '503', '504', 'internal_error', 'service_unavailable'],
        'invalid_request': ['400', 'invalid_request', 'bad_request'],
        'not_found': ['404', 'not_found'],
        'connection_error': ['connection', 'network', 'dns'],
        'unknown': []
    }
    
    @classmethod
    def standardize_error_type(cls, error: Any) -> str:
        """Standardize error types for consistent tracking."""
        if isinstance(error, str):
            error_str = error.lower()
        else:
            error_str = str(type(error).__name__).lower()
        
        for standard_type, patterns in cls.ERROR_TYPES.items():
            if any(pattern in error_str for pattern in patterns):
                return standard_type
        
        return 'unknown'
    
    # =========================================================================
    # DEBUG: Enhanced Call Tracking
    # =========================================================================
    
    def get_llm_call_log(self, limit: int = 100) -> List[Dict]:
        """Get recent LLM call log for debugging."""
        with self._call_log_lock:
            return list(self._llm_call_log)[-limit:]
    
    def get_llm_call_sources(self) -> Dict[str, int]:
        """Get summary of where LLM calls originate."""
        with self._call_log_lock:
            return dict(self._llm_call_sources)
    
    def get_llm_call_stats(self) -> Dict[str, Any]:
        """Get comprehensive LLM call statistics."""
        with self._call_log_lock:
            total_calls = len(self._llm_call_log)
            if total_calls == 0:
                return {
                    'total_calls': 0,
                    'successful_calls': 0,
                    'failed_calls': 0,
                    'avg_duration': 0.0,
                    'total_tokens': 0,
                    'top_sources': []
                }
            
            successful = sum(1 for call in self._llm_call_log if call['success'])
            failed = total_calls - successful
            avg_duration = sum(call['duration'] for call in self._llm_call_log) / total_calls
            total_tokens = sum(
                call['input_tokens'] + call['output_tokens'] 
                for call in self._llm_call_log
            )
            
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
                'total_tokens': total_tokens,
                'top_sources': top_sources
            }
    
    def clear_llm_call_log(self):
        """Clear the LLM call log."""
        with self._call_log_lock:
            self._llm_call_log.clear()
            self._llm_call_sources.clear()
        logger.info("🧹 LLM call log cleared")
    
    # =========================================================================
    # PRODUCTION FIX: Improved Call Stack Analysis
    # =========================================================================
    
    def _extract_caller_info(self) -> tuple[List[str], str]:
        """Extract meaningful caller information from stack."""
        stack = traceback.extract_stack()
        
        # Filter out framework and internal calls
        relevant_frames = []
        skip_patterns = [
            'prometheus_metrics.py',
            'threading.py',
            'asyncio',
            'contextlib.py',
            'functools.py'
        ]
        
        for frame in stack[:-2]:  # Skip current and caller
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
        error_type: Optional[str] = None
    ):
        """
        Track a complete LLM API call with all metrics.
        
        PRODUCTION IMPROVEMENTS:
        - Standardized error types
        - Better stack trace analysis
        - Thread-safe logging
        - Memory-efficient tracking
        """
        # Extract caller information
        relevant_frames, caller_summary = self._extract_caller_info()
        
        # Standardize error type
        if error_type:
            error_type = self.standardize_error_type(error_type)
        
        # Build call entry
        call_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "model": model,
            "operation": operation,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "duration": round(duration, 3),
            "success": success,
            "error_type": error_type,
            "caller_summary": caller_summary,
            "call_stack": [
                f"{f['file']}:{f['line']}:{f['function']}"
                for f in relevant_frames[-5:]  # Last 5 frames
            ]
        }
        
        # Thread-safe logging
        with self._call_log_lock:
            self._llm_call_log.append(call_entry)
            
            # Update source counts
            self._llm_call_sources[caller_summary] = \
                self._llm_call_sources.get(caller_summary, 0) + 1
        
        # Log every call for debugging (optional based on env)
        if os.getenv('DEBUG_LLM_CALLS', 'false').lower() == 'true':
            logger.debug(
                f"📊 LLM CALL: model={model}, op={operation}, "
                f"tokens={input_tokens}/{output_tokens}, success={success}, "
                f"source={caller_summary}"
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
            
            # Call count
            status = 'success' if success else 'error'
            self.llm_calls_total.labels(
                model=model, operation=operation, status=status
            ).inc()
            
            # Errors
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
            logger.error(f"❌ Error updating Prometheus metrics: {e}")
    
    def _estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """
        Estimate cost based on model and token counts.
        
        PRODUCTION FIX: Updated pricing (as of 2024)
        """
        # Pricing per 1K tokens (USD) - Updated 2024
        pricing = {
            # OpenAI GPT-4 family
            'gpt-4o': {'input': 0.0025, 'output': 0.010},  # Updated
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
            
            # Open source / other
            'qwen': {'input': 0.0001, 'output': 0.0001},
            'llama': {'input': 0.0002, 'output': 0.0002},
        }
        
        # Find matching model
        model_lower = model.lower()
        rates = {'input': 0.001, 'output': 0.002}  # Conservative default
        
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
        """Track agentic evaluation scores."""
        # Clamp scores to [0, 1] range
        task_adherence = max(0.0, min(1.0, task_adherence))
        tool_accuracy = max(0.0, min(1.0, tool_accuracy))
        intent_resolution = max(0.0, min(1.0, intent_resolution))
        overall_score = max(0.0, min(1.0, overall_score))
        
        self.agentic_task_adherence.labels(agent_name=agent_name).observe(task_adherence)
        self.agentic_tool_accuracy.labels(agent_name=agent_name).observe(tool_accuracy)
        self.agentic_intent_resolution.labels(agent_name=agent_name).observe(intent_resolution)
        self.agentic_overall_score.labels(agent_name=agent_name).observe(overall_score)
        
        # Categorize score
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
    
    # =========================================================================
    # CONTEXT MANAGERS
    # =========================================================================
    
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
            error_type = self.standardize_error_type(e)
            tracker.set_error(error_type)
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
            self.rag_relevance_score.labels(source=source).observe(
                max(0.0, min(1.0, avg_relevance))
            )
    
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
    
    def set_tokens(self, input_tokens: int, output_tokens: int):
        """Set token counts."""
        self.input_tokens = max(0, input_tokens)
        self.output_tokens = max(0, output_tokens)
    
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


# ============================================================================
# PRODUCTION FIX: Thread-safe singleton getter
# ============================================================================

def get_metrics() -> PrometheusMetrics:
    """Get or create the singleton metrics instance (thread-safe)."""
    global _metrics_instance
    
    if _metrics_instance is None:
        with _metrics_lock:
            # Double-check locking pattern
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
            response = await openai.chat(...)
            return response
    """
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with metrics.track_llm_request(model, operation) as tracker:
                result = await func(*args, **kwargs)
                
                # Try to extract token usage from result
                if hasattr(result, 'usage'):
                    usage = result.usage
                    if hasattr(usage, 'prompt_tokens'):
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
Usage:
    @agent_metrics('OrchestratorAgent')
    async def execute(self, query: str):
        ...
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