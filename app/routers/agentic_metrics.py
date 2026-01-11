"""
Agentic Metrics API Router - Endpoints for evaluating agent performance.

Provides APIs for:
- Evaluating individual sessions
- Batch evaluation of multiple sessions
- Getting evaluation summaries and statistics
- Exporting evaluation results

Reference: https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/evaluating-agentic-ai-systems-a-deep-dive-into-agentic-metrics/4403923
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/metrics", tags=["Agentic Metrics"])


# =========================================================================
# REQUEST/RESPONSE MODELS
# =========================================================================

class EvaluateSessionRequest(BaseModel):
    """Request to evaluate a single session."""
    session_id: str = Field(..., description="Session ID to evaluate")


class EvaluateManualRequest(BaseModel):
    """Request to manually evaluate an agent interaction."""
    user_query: str = Field(..., description="The original user query")
    agent_response: str = Field(..., description="The agent's final response")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="List of tool calls made (optional)"
    )
    detected_intent: Optional[str] = Field(
        default=None,
        description="Detected intent (optional)"
    )
    resource_type: Optional[str] = Field(
        default=None,
        description="Resource type operated on (optional)"
    )
    operation: Optional[str] = Field(
        default=None,
        description="Operation type (optional)"
    )


class BatchEvaluateRequest(BaseModel):
    """Request to batch evaluate multiple sessions."""
    session_ids: List[str] = Field(..., description="List of session IDs to evaluate")


class MetricScore(BaseModel):
    """Individual metric score."""
    score: float = Field(..., ge=0, le=1, description="Score from 0 to 1")
    reasoning: str = Field(..., description="Explanation for the score")


class EvaluationResponse(BaseModel):
    """Response for a single evaluation."""
    session_id: str
    agent_name: str
    task_adherence: MetricScore
    tool_call_accuracy: MetricScore
    intent_resolution: MetricScore
    overall_score: float
    timestamp: str
    metadata: Dict[str, Any] = {}


class EvaluationSummaryResponse(BaseModel):
    """Summary statistics of all evaluations."""
    total_evaluations: int
    average_scores: Dict[str, float]
    score_distribution: Dict[str, int]
    by_agent: Dict[str, Dict[str, Any]]
    by_operation: Dict[str, Dict[str, Any]]


# =========================================================================
# API ENDPOINTS
# =========================================================================

@router.post(
    "/evaluate/session",
    response_model=EvaluationResponse,
    summary="Evaluate a session by ID",
    description="Evaluate an agent session using Task Adherence, Tool Call Accuracy, and Intent Resolution metrics."
)
async def evaluate_session(request: EvaluateSessionRequest):
    """
    Evaluate a session that has been traced.
    
    The session must have been previously traced using the metrics system.
    Returns scores for:
    - Task Adherence (40% weight)
    - Tool Call Accuracy (30% weight)  
    - Intent Resolution (30% weight)
    """
    try:
        from app.services.agentic_metrics_service import agentic_metrics_evaluator
        
        result = await agentic_metrics_evaluator.evaluate_session(request.session_id)
        
        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"No trace found for session {request.session_id}"
            )
        
        return EvaluationResponse(
            session_id=result.session_id,
            agent_name=result.agent_name,
            task_adherence=MetricScore(
                score=result.task_adherence,
                reasoning=result.task_adherence_reasoning
            ),
            tool_call_accuracy=MetricScore(
                score=result.tool_call_accuracy,
                reasoning=result.tool_call_accuracy_reasoning
            ),
            intent_resolution=MetricScore(
                score=result.intent_resolution,
                reasoning=result.intent_resolution_reasoning
            ),
            overall_score=result.overall_score,
            timestamp=result.timestamp,
            metadata=result.metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Session evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/evaluate/manual",
    summary="Evaluate an agent interaction manually",
    description="Evaluate an agent interaction by providing query, response, and optional metadata."
)
async def evaluate_manual(request: EvaluateManualRequest):
    """
    Manually evaluate an agent interaction without a pre-existing trace.
    
    Useful for:
    - Testing evaluation metrics
    - Evaluating historical interactions
    - A/B testing different agent responses
    """
    try:
        from app.services.agentic_metrics_service import (
            agentic_metrics_evaluator,
            AgentTrace,
            ToolCall
        )
        
        # Convert tool calls if provided
        tool_calls = []
        if request.tool_calls:
            for tc in request.tool_calls:
                tool_calls.append(ToolCall(
                    tool_name=tc.get("tool_name", "unknown"),
                    tool_args=tc.get("tool_args", {}),
                    tool_result=tc.get("tool_result"),
                    timestamp=tc.get("timestamp", ""),
                    success=tc.get("success", True),
                    error=tc.get("error")
                ))
        
        # Create a trace manually
        trace = AgentTrace(
            session_id="manual_evaluation",
            user_query=request.user_query,
            agent_name="ManualEvaluation",
            intent_detected=request.detected_intent,
            resource_type=request.resource_type,
            operation=request.operation,
            tool_calls=tool_calls,
            final_response=request.agent_response,
            success=True
        )
        
        # Evaluate the trace
        result = await agentic_metrics_evaluator.evaluate_trace(trace)
        
        return {
            "task_adherence": {
                "score": result.task_adherence,
                "reasoning": result.task_adherence_reasoning
            },
            "tool_call_accuracy": {
                "score": result.tool_call_accuracy,
                "reasoning": result.tool_call_accuracy_reasoning
            },
            "intent_resolution": {
                "score": result.intent_resolution,
                "reasoning": result.intent_resolution_reasoning
            },
            "overall_score": result.overall_score,
            "evaluation_timestamp": result.timestamp
        }
        
    except Exception as e:
        logger.error(f"❌ Manual evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/evaluate/batch",
    summary="Batch evaluate multiple sessions",
    description="Evaluate multiple sessions at once for efficiency."
)
async def batch_evaluate(request: BatchEvaluateRequest):
    """
    Evaluate multiple sessions in batch.
    
    Returns evaluation results for all found sessions.
    Sessions without traces will be skipped.
    """
    try:
        from app.services.agentic_metrics_service import agentic_metrics_evaluator
        
        results = []
        not_found = []
        
        for session_id in request.session_ids:
            trace = agentic_metrics_evaluator.get_trace(session_id)
            if trace:
                result = await agentic_metrics_evaluator.evaluate_trace(trace)
                results.append({
                    "session_id": result.session_id,
                    "agent_name": result.agent_name,
                    "task_adherence": result.task_adherence,
                    "tool_call_accuracy": result.tool_call_accuracy,
                    "intent_resolution": result.intent_resolution,
                    "overall_score": result.overall_score
                })
            else:
                not_found.append(session_id)
        
        return {
            "evaluated": len(results),
            "not_found": len(not_found),
            "results": results,
            "missing_sessions": not_found
        }
        
    except Exception as e:
        logger.error(f"❌ Batch evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/summary",
    response_model=EvaluationSummaryResponse,
    summary="Get evaluation summary statistics",
    description="Get aggregate statistics across all evaluations performed."
)
async def get_evaluation_summary():
    """
    Get summary statistics of all evaluations.
    
    Returns:
    - Average scores for each metric
    - Score distribution (excellent, good, acceptable, poor, failed)
    - Breakdown by agent and operation type
    """
    try:
        from app.services.agentic_metrics_service import agentic_metrics_evaluator
        
        summary = agentic_metrics_evaluator.get_evaluation_summary()
        
        if "message" in summary:
            # No evaluations yet
            return EvaluationSummaryResponse(
                total_evaluations=0,
                average_scores={},
                score_distribution={},
                by_agent={},
                by_operation={}
            )
        
        return EvaluationSummaryResponse(**summary)
        
    except Exception as e:
        logger.error(f"❌ Failed to get evaluation summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/export",
    summary="Export evaluation results",
    description="Export all evaluation results in JSON or JSONL format."
)
async def export_results(
    format: str = Query("json", enum=["json", "jsonl"], description="Export format")
):
    """
    Export all evaluation results.
    
    Formats:
    - json: Standard JSON array
    - jsonl: JSON Lines format (one JSON object per line)
    """
    try:
        from app.services.agentic_metrics_service import agentic_metrics_evaluator
        
        data = agentic_metrics_evaluator.export_results(format=format)
        
        return {
            "format": format,
            "data": data
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to export results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/trace/{session_id}",
    summary="Get a trace by session ID",
    description="Retrieve the recorded trace for a session."
)
async def get_trace(session_id: str):
    """
    Get the trace data for a specific session.
    
    Useful for debugging and understanding agent behavior.
    """
    try:
        from app.services.agentic_metrics_service import agentic_metrics_evaluator
        
        trace = agentic_metrics_evaluator.get_trace(session_id)
        
        if not trace:
            raise HTTPException(
                status_code=404,
                detail=f"No trace found for session {session_id}"
            )
        
        return trace.to_dict()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get trace: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "/clear",
    summary="Clear all traces and evaluation results",
    description="Clear all stored traces and evaluation results."
)
async def clear_all():
    """
    Clear all stored traces and evaluation results.
    
    Use with caution - this cannot be undone.
    """
    try:
        from app.services.agentic_metrics_service import agentic_metrics_evaluator
        
        agentic_metrics_evaluator.clear_results()
        
        return {"message": "All traces and evaluation results cleared"}
        
    except Exception as e:
        logger.error(f"❌ Failed to clear results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/history",
    summary="Get evaluation history from database",
    description="Retrieve historical evaluations from PostgreSQL database."
)
async def get_evaluation_history(
    limit: int = Query(50, ge=1, le=500, description="Maximum records to return"),
    offset: int = Query(0, ge=0, description="Number of records to skip"),
    agent_name: Optional[str] = Query(None, description="Filter by agent name"),
    operation: Optional[str] = Query(None, description="Filter by operation type"),
    min_score: Optional[float] = Query(None, ge=0, le=1, description="Minimum overall score")
):
    """
    Get historical evaluations from the PostgreSQL database.
    
    Supports pagination and filtering by:
    - Agent name
    - Operation type (list, create, delete, etc.)
    - Minimum overall score
    """
    try:
        from app.services.agentic_metrics_persistence import get_metrics_persistence
        
        persistence = get_metrics_persistence()
        evaluations = persistence.get_all_evaluations(
            limit=limit,
            offset=offset,
            agent_name=agent_name,
            operation=operation,
            min_score=min_score
        )
        
        return {
            "count": len(evaluations),
            "limit": limit,
            "offset": offset,
            "evaluations": evaluations
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get evaluation history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/history/{session_id}",
    summary="Get a specific evaluation from database",
    description="Retrieve a specific evaluation by session ID from PostgreSQL."
)
async def get_evaluation_by_session(session_id: str):
    """
    Get a specific evaluation from the database by session ID.
    """
    try:
        from app.services.agentic_metrics_persistence import get_metrics_persistence
        
        persistence = get_metrics_persistence()
        evaluation = persistence.get_evaluation(session_id)
        
        if not evaluation:
            raise HTTPException(
                status_code=404,
                detail=f"No evaluation found for session {session_id}"
            )
        
        return evaluation
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/health",
    summary="Check metrics service health",
    description="Check if the agentic metrics service is healthy."
)
async def health_check():
    """
    Health check for the agentic metrics service.
    """
    try:
        from app.services.agentic_metrics_service import agentic_metrics_evaluator
        
        summary = agentic_metrics_evaluator.get_evaluation_summary()
        
        # Check database connection
        db_status = "connected"
        try:
            from app.services.agentic_metrics_persistence import get_metrics_persistence
            persistence = get_metrics_persistence()
            db_stats = persistence.get_summary_stats()
            db_evaluations = db_stats.get("total_evaluations", 0)
        except Exception as e:
            db_status = f"error: {e}"
            db_evaluations = 0
        
        return {
            "status": "healthy",
            "service": "agentic_metrics",
            "evaluations_count": summary.get("total_evaluations", summary.get("count", 0)),
            "database": {
                "status": db_status,
                "evaluations_stored": db_evaluations
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Metrics service health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "agentic_metrics",
            "error": str(e)
        }

