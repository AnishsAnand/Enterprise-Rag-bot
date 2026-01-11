"""
PostgreSQL Persistence for Agentic Metrics.

Stores evaluation results and traces in the existing ragbot_sessions database.
Tables:
- agentic_evaluation_results: Stores evaluation scores and reasoning
- agentic_traces: Stores full execution traces for analysis

Reference: https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/evaluating-agentic-ai-systems-a-deep-dive-into-agentic-metrics/4403923
"""

import logging
import json
from typing import Any, Dict, List, Optional
from datetime import datetime

from sqlalchemy import create_engine, Column, String, Text, DateTime, Float, Boolean, Integer, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import settings

logger = logging.getLogger(__name__)

# Create base for metrics tables
MetricsBase = declarative_base()


class AgenticEvaluationRecord(MetricsBase):
    """
    PostgreSQL table for storing evaluation results.
    
    Stores the three key agentic metrics:
    - Task Adherence (0-1): Did the agent answer the right question?
    - Tool Call Accuracy (0-1): Did the agent use tools correctly?
    - Intent Resolution (0-1): Did the agent understand the user's goal?
    """
    __tablename__ = "agentic_evaluation_results"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(255), nullable=False, index=True)
    agent_name = Column(String(255), nullable=False, index=True)
    
    # User query and response
    user_query = Column(Text, nullable=True)
    agent_response = Column(Text, nullable=True)
    
    # The three key metrics (0.0 to 1.0)
    task_adherence = Column(Float, nullable=False)
    task_adherence_reasoning = Column(Text, nullable=True)
    
    tool_call_accuracy = Column(Float, nullable=False)
    tool_call_accuracy_reasoning = Column(Text, nullable=True)
    
    intent_resolution = Column(Float, nullable=False)
    intent_resolution_reasoning = Column(Text, nullable=True)
    
    # Overall weighted score
    overall_score = Column(Float, nullable=False)
    
    # Metadata
    resource_type = Column(String(100), nullable=True, index=True)
    operation = Column(String(50), nullable=True, index=True)
    tool_calls_count = Column(Integer, default=0)
    execution_success = Column(Boolean, default=True)
    
    # Extra metadata as JSON
    extra_metadata = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    evaluation_timestamp = Column(String(50), nullable=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "agent_name": self.agent_name,
            "user_query": self.user_query,
            "agent_response": self.agent_response[:200] + "..." if self.agent_response and len(self.agent_response) > 200 else self.agent_response,
            "task_adherence": self.task_adherence,
            "task_adherence_reasoning": self.task_adherence_reasoning,
            "tool_call_accuracy": self.tool_call_accuracy,
            "tool_call_accuracy_reasoning": self.tool_call_accuracy_reasoning,
            "intent_resolution": self.intent_resolution,
            "intent_resolution_reasoning": self.intent_resolution_reasoning,
            "overall_score": self.overall_score,
            "resource_type": self.resource_type,
            "operation": self.operation,
            "tool_calls_count": self.tool_calls_count,
            "execution_success": self.execution_success,
            "metadata": self.extra_metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class AgenticTraceRecord(MetricsBase):
    """
    PostgreSQL table for storing agent execution traces.
    
    Full trace data for detailed analysis and debugging.
    """
    __tablename__ = "agentic_traces"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(255), nullable=False, unique=True, index=True)
    user_query = Column(Text, nullable=False)
    agent_name = Column(String(255), nullable=False)
    
    # Intent information
    intent_detected = Column(String(255), nullable=True)
    resource_type = Column(String(100), nullable=True)
    operation = Column(String(50), nullable=True)
    
    # Tool calls as JSON array
    tool_calls = Column(JSON, nullable=True)
    
    # Intermediate steps as JSON array
    intermediate_steps = Column(JSON, nullable=True)
    
    # Final response
    final_response = Column(Text, nullable=True)
    
    # Execution metadata
    success = Column(Boolean, default=True)
    error = Column(Text, nullable=True)
    start_time = Column(String(50), nullable=True)
    end_time = Column(String(50), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "user_query": self.user_query,
            "agent_name": self.agent_name,
            "intent_detected": self.intent_detected,
            "resource_type": self.resource_type,
            "operation": self.operation,
            "tool_calls": self.tool_calls,
            "intermediate_steps": self.intermediate_steps,
            "final_response": self.final_response,
            "success": self.success,
            "error": self.error,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class AgenticMetricsPersistence:
    """
    PostgreSQL persistence manager for agentic metrics.
    
    Provides CRUD operations for evaluation results and traces.
    """
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize persistence with database connection.
        
        Args:
            database_url: PostgreSQL connection string. Defaults to app settings.
        """
        self.database_url = database_url or settings.DATABASE_URL
        
        # Create engine with appropriate settings
        if self.database_url.startswith("sqlite"):
            self.engine = create_engine(
                self.database_url,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool
            )
        else:
            self.engine = create_engine(
                self.database_url,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True
            )
        
        # Create session factory
        self.SessionFactory = sessionmaker(bind=self.engine)
        
        # Create tables if they don't exist
        MetricsBase.metadata.create_all(self.engine)
        
        logger.info(f"✅ AgenticMetricsPersistence initialized with database")
    
    def save_evaluation(self, evaluation_result: Dict[str, Any]) -> int:
        """
        Save an evaluation result to PostgreSQL.
        
        Args:
            evaluation_result: EvaluationResult as dictionary
            
        Returns:
            ID of the saved record
        """
        session = self.SessionFactory()
        try:
            record = AgenticEvaluationRecord(
                session_id=evaluation_result.get("session_id"),
                agent_name=evaluation_result.get("agent_name"),
                user_query=evaluation_result.get("metadata", {}).get("user_query"),
                task_adherence=evaluation_result.get("task_adherence", 0.0),
                task_adherence_reasoning=evaluation_result.get("task_adherence_reasoning"),
                tool_call_accuracy=evaluation_result.get("tool_call_accuracy", 0.0),
                tool_call_accuracy_reasoning=evaluation_result.get("tool_call_accuracy_reasoning"),
                intent_resolution=evaluation_result.get("intent_resolution", 0.0),
                intent_resolution_reasoning=evaluation_result.get("intent_resolution_reasoning"),
                overall_score=evaluation_result.get("overall_score", 0.0),
                resource_type=evaluation_result.get("metadata", {}).get("resource_type"),
                operation=evaluation_result.get("metadata", {}).get("operation"),
                tool_calls_count=evaluation_result.get("metadata", {}).get("tool_calls_count", 0),
                execution_success=evaluation_result.get("metadata", {}).get("execution_success", True),
                extra_metadata=evaluation_result.get("metadata"),
                evaluation_timestamp=evaluation_result.get("timestamp")
            )
            
            session.add(record)
            session.commit()
            
            logger.info(f"💾 Saved evaluation for session {record.session_id} (ID: {record.id})")
            return record.id
            
        except Exception as e:
            session.rollback()
            logger.error(f"❌ Failed to save evaluation: {e}")
            raise
        finally:
            session.close()
    
    def save_trace(self, trace_dict: Dict[str, Any]) -> int:
        """
        Save an agent trace to PostgreSQL.
        
        Args:
            trace_dict: AgentTrace as dictionary
            
        Returns:
            ID of the saved record
        """
        session = self.SessionFactory()
        try:
            # Check if trace already exists
            existing = session.query(AgenticTraceRecord).filter_by(
                session_id=trace_dict.get("session_id")
            ).first()
            
            if existing:
                # Update existing trace
                existing.user_query = trace_dict.get("user_query")
                existing.agent_name = trace_dict.get("agent_name")
                existing.intent_detected = trace_dict.get("intent_detected")
                existing.resource_type = trace_dict.get("resource_type")
                existing.operation = trace_dict.get("operation")
                existing.tool_calls = trace_dict.get("tool_calls")
                existing.intermediate_steps = trace_dict.get("intermediate_steps")
                existing.final_response = trace_dict.get("final_response")
                existing.success = trace_dict.get("success", True)
                existing.error = trace_dict.get("error")
                existing.start_time = trace_dict.get("start_time")
                existing.end_time = trace_dict.get("end_time")
                
                session.commit()
                logger.debug(f"📝 Updated trace for session {existing.session_id}")
                return existing.id
            else:
                # Create new trace
                record = AgenticTraceRecord(
                    session_id=trace_dict.get("session_id"),
                    user_query=trace_dict.get("user_query"),
                    agent_name=trace_dict.get("agent_name"),
                    intent_detected=trace_dict.get("intent_detected"),
                    resource_type=trace_dict.get("resource_type"),
                    operation=trace_dict.get("operation"),
                    tool_calls=trace_dict.get("tool_calls"),
                    intermediate_steps=trace_dict.get("intermediate_steps"),
                    final_response=trace_dict.get("final_response"),
                    success=trace_dict.get("success", True),
                    error=trace_dict.get("error"),
                    start_time=trace_dict.get("start_time"),
                    end_time=trace_dict.get("end_time")
                )
                
                session.add(record)
                session.commit()
                
                logger.debug(f"💾 Saved trace for session {record.session_id}")
                return record.id
                
        except Exception as e:
            session.rollback()
            logger.error(f"❌ Failed to save trace: {e}")
            raise
        finally:
            session.close()
    
    def get_evaluation(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get evaluation result by session ID."""
        session = self.SessionFactory()
        try:
            record = session.query(AgenticEvaluationRecord).filter_by(
                session_id=session_id
            ).order_by(AgenticEvaluationRecord.created_at.desc()).first()
            
            return record.to_dict() if record else None
        finally:
            session.close()
    
    def get_trace(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get trace by session ID."""
        session = self.SessionFactory()
        try:
            record = session.query(AgenticTraceRecord).filter_by(
                session_id=session_id
            ).first()
            
            return record.to_dict() if record else None
        finally:
            session.close()
    
    def get_all_evaluations(
        self, 
        limit: int = 100, 
        offset: int = 0,
        agent_name: Optional[str] = None,
        operation: Optional[str] = None,
        min_score: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all evaluations with optional filtering.
        
        Args:
            limit: Maximum records to return
            offset: Number of records to skip
            agent_name: Filter by agent name
            operation: Filter by operation type
            min_score: Filter by minimum overall score
            
        Returns:
            List of evaluation records
        """
        session = self.SessionFactory()
        try:
            query = session.query(AgenticEvaluationRecord)
            
            if agent_name:
                query = query.filter(AgenticEvaluationRecord.agent_name == agent_name)
            if operation:
                query = query.filter(AgenticEvaluationRecord.operation == operation)
            if min_score is not None:
                query = query.filter(AgenticEvaluationRecord.overall_score >= min_score)
            
            records = query.order_by(
                AgenticEvaluationRecord.created_at.desc()
            ).offset(offset).limit(limit).all()
            
            return [r.to_dict() for r in records]
        finally:
            session.close()
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get aggregate statistics from stored evaluations.
        
        Returns:
            Summary statistics
        """
        session = self.SessionFactory()
        try:
            from sqlalchemy import func
            
            # Get total count
            total = session.query(func.count(AgenticEvaluationRecord.id)).scalar() or 0
            
            if total == 0:
                return {"total_evaluations": 0, "message": "No evaluations stored yet"}
            
            # Get averages
            avg_task = session.query(func.avg(AgenticEvaluationRecord.task_adherence)).scalar() or 0
            avg_tool = session.query(func.avg(AgenticEvaluationRecord.tool_call_accuracy)).scalar() or 0
            avg_intent = session.query(func.avg(AgenticEvaluationRecord.intent_resolution)).scalar() or 0
            avg_overall = session.query(func.avg(AgenticEvaluationRecord.overall_score)).scalar() or 0
            
            # Get score distribution
            excellent = session.query(func.count(AgenticEvaluationRecord.id)).filter(
                AgenticEvaluationRecord.overall_score >= 0.9
            ).scalar() or 0
            good = session.query(func.count(AgenticEvaluationRecord.id)).filter(
                AgenticEvaluationRecord.overall_score >= 0.7,
                AgenticEvaluationRecord.overall_score < 0.9
            ).scalar() or 0
            acceptable = session.query(func.count(AgenticEvaluationRecord.id)).filter(
                AgenticEvaluationRecord.overall_score >= 0.5,
                AgenticEvaluationRecord.overall_score < 0.7
            ).scalar() or 0
            poor = session.query(func.count(AgenticEvaluationRecord.id)).filter(
                AgenticEvaluationRecord.overall_score >= 0.3,
                AgenticEvaluationRecord.overall_score < 0.5
            ).scalar() or 0
            failed = session.query(func.count(AgenticEvaluationRecord.id)).filter(
                AgenticEvaluationRecord.overall_score < 0.3
            ).scalar() or 0
            
            # Get by agent
            by_agent = {}
            agent_stats = session.query(
                AgenticEvaluationRecord.agent_name,
                func.count(AgenticEvaluationRecord.id),
                func.avg(AgenticEvaluationRecord.overall_score)
            ).group_by(AgenticEvaluationRecord.agent_name).all()
            
            for agent_name, count, avg in agent_stats:
                by_agent[agent_name] = {"count": count, "average": round(float(avg), 3)}
            
            # Get by operation
            by_operation = {}
            op_stats = session.query(
                AgenticEvaluationRecord.operation,
                func.count(AgenticEvaluationRecord.id),
                func.avg(AgenticEvaluationRecord.overall_score)
            ).group_by(AgenticEvaluationRecord.operation).all()
            
            for op, count, avg in op_stats:
                if op:
                    by_operation[op] = {"count": count, "average": round(float(avg), 3)}
            
            return {
                "total_evaluations": total,
                "average_scores": {
                    "task_adherence": round(float(avg_task), 3),
                    "tool_call_accuracy": round(float(avg_tool), 3),
                    "intent_resolution": round(float(avg_intent), 3),
                    "overall": round(float(avg_overall), 3)
                },
                "score_distribution": {
                    "excellent": excellent,
                    "good": good,
                    "acceptable": acceptable,
                    "poor": poor,
                    "failed": failed
                },
                "by_agent": by_agent,
                "by_operation": by_operation
            }
        finally:
            session.close()
    
    def delete_evaluation(self, session_id: str) -> bool:
        """Delete an evaluation by session ID."""
        session = self.SessionFactory()
        try:
            deleted = session.query(AgenticEvaluationRecord).filter_by(
                session_id=session_id
            ).delete()
            session.commit()
            return deleted > 0
        except Exception as e:
            session.rollback()
            logger.error(f"❌ Failed to delete evaluation: {e}")
            return False
        finally:
            session.close()
    
    def delete_all(self) -> int:
        """Delete all evaluations and traces."""
        session = self.SessionFactory()
        try:
            eval_count = session.query(AgenticEvaluationRecord).delete()
            trace_count = session.query(AgenticTraceRecord).delete()
            session.commit()
            logger.info(f"🗑️ Deleted {eval_count} evaluations and {trace_count} traces")
            return eval_count + trace_count
        except Exception as e:
            session.rollback()
            logger.error(f"❌ Failed to delete all: {e}")
            return 0
        finally:
            session.close()


# Singleton instance
_persistence_instance = None

def get_metrics_persistence() -> AgenticMetricsPersistence:
    """Get or create the singleton persistence instance."""
    global _persistence_instance
    if _persistence_instance is None:
        _persistence_instance = AgenticMetricsPersistence()
    return _persistence_instance

