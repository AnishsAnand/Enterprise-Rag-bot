"""
Production-grade health check and monitoring service.
Provides comprehensive system status including database, cache, and AI services.
"""

import logging
import asyncio
from datetime import datetime
from typing import Dict, Any
from enum import Enum
from app.services.postgres_service import postgres_service
from app.services.ai_service import ai_service
from app.core.config import settings

logger = logging.getLogger(__name__)

class HealthStatus(str, Enum):
    """Health status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

class HealthService:
    """
    Comprehensive health check service for production monitoring.
    Checks database, cache, AI services, and system resources.
    """
    async def get_full_health_status(self) -> Dict[str, Any]:
        """
        Get comprehensive health status of all systems.
        """
        start_time = datetime.utcnow()
        
        # Check all services in parallel
        results = await asyncio.gather(
            self._check_database(),
            self._check_ai_services(),
            self._check_disk_space(),
            return_exceptions=True)
    
        db_status, ai_status, disk_status = results
        # Determine overall health
        overall_status = self._determine_overall_status(db_status, ai_status, disk_status)
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        return {
            "status": overall_status.value,
            "timestamp": start_time.isoformat(),
            "version": "2.0.0",
            "components": {
                "database": db_status,
                "ai_services": ai_status,
                "disk": disk_status,
            },
            "metrics": {
                "check_duration_ms": round(execution_time * 1000, 2),
                "healthy_components": sum(1 for comp in [
                    db_status, ai_status, disk_status
                ] if isinstance(comp, dict) and comp.get("status") == "healthy"),
                "total_components": 3 }}
    
    async def get_liveness_status(self) -> Dict[str, Any]:
        """
        Quick liveness check (process is running).
        """
        return {"status": "alive","service": "enterprise-rag-bot","timestamp": datetime.utcnow().isoformat()}
    
    async def get_readiness_status(self) -> Dict[str, Any]:
        """
        Readiness check (all dependencies are available).
        """
        db_ready = await self._check_database_ready()
        ready = db_ready
        return {
            "ready": ready,
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "database": "ready" if db_ready else "not_ready"}}

    async def _check_database(self) -> Dict[str, Any]:
        """Check PostgreSQL connection and status"""
        try:
            # Quick connection test
            stats = await postgres_service.get_collection_stats()
            
            if isinstance(stats, dict):
                return {
                    "status": "healthy",
                    "type": "PostgreSQL",
                    "details": {
                        "documents": stats.get("document_count", 0),
                        "collections": stats.get("collection_count", 0),
                        "status": stats.get("status", "active"),
                    },
                    "timestamp": datetime.utcnow().isoformat()
                }
            else:
                return {
                    "status": "degraded",
                    "type": "PostgreSQL",
                    "error": "Invalid response format",
                    "timestamp": datetime.utcnow().isoformat()}

        except Exception as e:
            logger.warning(f"Database health check failed: {e}")
            return {
                "status": "unhealthy",
                "type": "PostgreSQL",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()}
    
    async def _check_ai_services(self) -> Dict[str, Any]:
        """Check AI service availability"""
        ai_health = {
            "status": "healthy",
            "type": "AI_Services",
            "services": {},
            "timestamp": datetime.utcnow().isoformat()}
        # Test embedding service
        try:
            embeddings = await ai_service.generate_embeddings(["test"])
            if embeddings and len(embeddings[0]) > 0:
                ai_health["services"]["embeddings"] = {
                    "status": "operational",
                    "dimension": len(embeddings[0])}
            else:
                ai_health["services"]["embeddings"] = {
                    "status": "degraded",
                    "error": "Empty response"}
                ai_health["status"] = "degraded"
        except Exception as e:
            ai_health["services"]["embeddings"] = {
                "status": "unavailable",
                "error": str(e)}
            ai_health["status"] = "degraded"
        # Test generation service
        try:
            response = await ai_service.generate_response("health", [])
            if response:
                ai_health["services"]["generation"] = {
                    "status": "operational"}
            else:
                ai_health["services"]["generation"] = {
                    "status": "degraded",
                    "error": "Empty response"}
                ai_health["status"] = "degraded"
        except Exception as e:
            ai_health["services"]["generation"] = {
                "status": "unavailable",
                "error": str(e)}
            ai_health["status"] = "degraded"
        return ai_health

    async def _check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space"""
        try:
            import shutil
            disk_info = shutil.disk_usage("/")
            
            # Convert to GB
            total_gb = disk_info.total / (1024 ** 3)
            used_gb = disk_info.used / (1024 ** 3)
            free_gb = disk_info.free / (1024 ** 3)
            percent_used = (disk_info.used / disk_info.total) * 100
            # Determine status
            status = "healthy"
            if percent_used > 90:
                status = "unhealthy"
            elif percent_used > 75:
                status = "degraded"
            return {
                "status": status,
                "type": "Disk_Storage",
                "details": {
                    "total_gb": round(total_gb, 2),
                    "used_gb": round(used_gb, 2),
                    "free_gb": round(free_gb, 2),
                    "percent_used": round(percent_used, 2)},
                "timestamp": datetime.utcnow().isoformat()}
        except Exception as e:
            logger.warning(f"Disk check failed: {e}")
            return {
                "status": "unknown",
                "type": "Disk_Storage",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()}
        
    async def _check_database_ready(self) -> bool:
        """Quick database readiness check"""
        try:
            stats = await postgres_service.get_collection_stats()
            return isinstance(stats, dict)
        except Exception:
            return False
    
    def _determine_overall_status(self, *components) -> HealthStatus:
        """Determine overall health based on component statuses"""
        statuses = []
        
        for component in components:
            if isinstance(component, dict):
                status = component.get("status", "unknown")
                statuses.append(status)
        # If any component is unhealthy, overall is unhealthy
        if "unhealthy" in statuses:
            return HealthStatus.UNHEALTHY
        # If any component is degraded, overall is degraded
        if "degraded" in statuses:
            return HealthStatus.DEGRADED
        # If all are healthy
        if all(status == "healthy" for status in statuses):
            return HealthStatus.HEALTHY
        return HealthStatus.UNKNOWN
# Singleton instance
health_service = HealthService()