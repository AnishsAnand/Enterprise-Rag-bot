"""
Health check endpoints for monitoring and orchestration.
Supports K8s, Docker, and other container orchestration systems.
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from app.monitoring.health_service import health_service

router = APIRouter(tags=["health"])

@router.get("/health")
async def health_check():
    """
    Comprehensive health check endpoint.
    Returns detailed status of all system components.
    Use for monitoring dashboards and alerts.
    """
    status_data = await health_service.get_full_health_status()
    # Determine HTTP status code
    overall_status = status_data.get("status")
    status_code = {"healthy": 200,"degraded": 503,"unhealthy": 503,}.get(overall_status, 503)
    return JSONResponse(content=status_data, status_code=status_code)

@router.get("/health/live")
async def liveness_check():
    """
    Kubernetes/Docker liveness probe endpoint.
    Returns 200 if the process is running.
    No dependencies checked - use for restart decisions.
    Response time: <10ms
    """
    status = await health_service.get_liveness_status()
    return JSONResponse(content=status, status_code=200)

@router.get("/health/ready")
async def readiness_check():
    """
    Kubernetes/Docker readiness probe endpoint.
    Returns 200 only if all critical dependencies are available.
    Checks: Database, Cache
    Use for traffic routing decisions.
    """
    status = await health_service.get_readiness_status()
    status_code = 200 if status.get("ready") else 503
    return JSONResponse(content=status, status_code=status_code)

@router.get("/health/metrics")
async def metrics_summary():
    """
    Quick metrics summary for monitoring systems.
    Returns key performance indicators.
    Note: Prometheus metrics are at /metrics (root level)
    """
    health_data = await health_service.get_full_health_status()
    return {
        "status": health_data["status"],
        "timestamp": health_data["timestamp"],
        "metrics": health_data.get("metrics", {})}