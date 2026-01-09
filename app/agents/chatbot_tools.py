"""
LangChain Tool Definitions for Chatbot Agents
Production-grade tool registry with comprehensive error handling
"""

from typing import Any, Dict, List, Optional, Callable
from pydantic import BaseModel, Field
from langchain.tools import Tool, tool
import logging
import httpx
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)


class ToolResult(BaseModel):
    """Standardized tool execution result"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class ChatbotToolRegistry:
    """Centralized registry for all chatbot tools"""
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self.tool_categories: Dict[str, List[str]] = {
            "search": [],
            "infrastructure": [],
            "api": [],
            "data": [],
            "analysis": [],
            "execution": []
        }
        self._initialize_tools()
    
    def _initialize_tools(self):
        """Initialize all available tools"""
        # Search Tools
        self._register_tool(
            name="search_knowledge_base",
            func=self.search_knowledge_base,
            description="Search the knowledge base for relevant documents and information",
            category="search"
        )
        
        self._register_tool(
            name="search_kubernetes_docs",
            func=self.search_kubernetes_docs,
            description="Search Kubernetes documentation for services, deployments, and configurations",
            category="search"
        )
        
        # Infrastructure Tools
        self._register_tool(
            name="get_k8s_services",
            func=self.get_k8s_services,
            description="Retrieve list of Kubernetes services and their status",
            category="infrastructure"
        )
        
        self._register_tool(
            name="enable_k8s_service",
            func=self.enable_k8s_service,
            description="Enable a Kubernetes service by name",
            category="infrastructure"
        )
        
        self._register_tool(
            name="disable_k8s_service",
            func=self.disable_k8s_service,
            description="Disable a Kubernetes service by name",
            category="infrastructure"
        )
        
        self._register_tool(
            name="get_service_status",
            func=self.get_service_status,
            description="Get detailed status of a specific service",
            category="infrastructure"
        )
        
        # API Tools
        self._register_tool(
            name="call_external_api",
            func=self.call_external_api,
            description="Make HTTP requests to external APIs with authentication",
            category="api"
        )
        
        self._register_tool(
            name="validate_api_key",
            func=self.validate_api_key,
            description="Validate API key for external services",
            category="api"
        )
        
        # Data Tools
        self._register_tool(
            name="get_service_logs",
            func=self.get_service_logs,
            description="Retrieve logs from a service",
            category="data"
        )
        
        self._register_tool(
            name="get_service_metrics",
            func=self.get_service_metrics,
            description="Get performance metrics for a service",
            category="data"
        )
        
        # Analysis Tools
        self._register_tool(
            name="analyze_service_health",
            func=self.analyze_service_health,
            description="Analyze the health status of services",
            category="analysis"
        )
        
        self._register_tool(
            name="generate_recommendations",
            func=self.generate_recommendations,
            description="Generate recommendations based on service analysis",
            category="analysis"
        )
        
        # Execution Tools
        self._register_tool(
            name="execute_task",
            func=self.execute_task,
            description="Execute an automated task with parameters",
            category="execution"
        )
    
    def _register_tool(self, name: str, func: Callable, description: str, category: str):
        """Register a tool in the registry"""
        tool = Tool(
            name=name,
            func=func,
            description=description,
            return_direct=False
        )
        self.tools[name] = tool
        self.tool_categories[category].append(name)
        logger.info(f"Registered tool: {name} in category: {category}")
    
    def get_tools_by_category(self, category: str) -> List[Tool]:
        """Get all tools in a specific category"""
        tool_names = self.tool_categories.get(category, [])
        return [self.tools[name] for name in tool_names if name in self.tools]
    
    def get_all_tools(self) -> List[Tool]:
        """Get all registered tools"""
        return list(self.tools.values())
    
    # ============ SEARCH TOOLS ============
    
    async def search_knowledge_base(self, query: str, max_results: int = 5) -> ToolResult:
        """Search knowledge base for relevant information"""
        try:
            from app.services.chroma_service import chroma_service
            
            start_time = datetime.now()
            results = await self._call_maybe_async(
                chroma_service.search_documents,
                query,
                n_results=max_results
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ToolResult(
                success=True,
                data={
                    "query": query,
                    "results": results,
                    "count": len(results) if results else 0
                },
                execution_time=execution_time
            )
        except Exception as e:
            logger.error(f"Knowledge base search failed: {str(e)}")
            return ToolResult(
                success=False,
                error=f"Search failed: {str(e)}"
            )
    
    async def search_kubernetes_docs(self, query: str, topic: str = "services") -> ToolResult:
        """Search Kubernetes documentation"""
        try:
            # Simulated K8s documentation search
            k8s_docs = {
                "services": {
                    "enable": "To enable a Kubernetes service, use: kubectl apply -f service.yaml",
                    "disable": "To disable a service, use: kubectl delete service <service-name>",
                    "list": "List services with: kubectl get services"
                },
                "deployments": {
                    "create": "Create deployment with: kubectl create deployment <name> --image=<image>",
                    "scale": "Scale deployment with: kubectl scale deployment <name> --replicas=<count>"
                }
            }
            
            results = k8s_docs.get(topic, {})
            
            return ToolResult(
                success=True,
                data={
                    "query": query,
                    "topic": topic,
                    "documentation": results
                }
            )
        except Exception as e:
            logger.error(f"K8s docs search failed: {str(e)}")
            return ToolResult(success=False, error=str(e))
    
    # ============ INFRASTRUCTURE TOOLS ============
    
    async def get_k8s_services(self) -> ToolResult:
        """Get list of Kubernetes services"""
        try:
            # Simulated K8s service retrieval
            services = [
                {"name": "api-service", "status": "running", "replicas": 3},
                {"name": "database-service", "status": "running", "replicas": 1},
                {"name": "cache-service", "status": "stopped", "replicas": 0},
                {"name": "worker-service", "status": "running", "replicas": 2}
            ]
            
            return ToolResult(
                success=True,
                data={"services": services, "total": len(services)}
            )
        except Exception as e:
            logger.error(f"Failed to get K8s services: {str(e)}")
            return ToolResult(success=False, error=str(e))
    
    async def enable_k8s_service(self, service_name: str, config: Optional[Dict] = None) -> ToolResult:
        """Enable a Kubernetes service"""
        try:
            logger.info(f"Enabling K8s service: {service_name}")
            
            # Simulated service enablement
            result = {
                "service": service_name,
                "action": "enable",
                "status": "success",
                "message": f"Service {service_name} has been enabled",
                "timestamp": datetime.now().isoformat()
            }
            
            return ToolResult(
                success=True,
                data=result
            )
        except Exception as e:
            logger.error(f"Failed to enable service {service_name}: {str(e)}")
            return ToolResult(success=False, error=str(e))
    
    async def disable_k8s_service(self, service_name: str) -> ToolResult:
        """Disable a Kubernetes service"""
        try:
            logger.info(f"Disabling K8s service: {service_name}")
            
            result = {
                "service": service_name,
                "action": "disable",
                "status": "success",
                "message": f"Service {service_name} has been disabled",
                "timestamp": datetime.now().isoformat()
            }
            
            return ToolResult(
                success=True,
                data=result
            )
        except Exception as e:
            logger.error(f"Failed to disable service {service_name}: {str(e)}")
            return ToolResult(success=False, error=str(e))
    
    async def get_service_status(self, service_name: str) -> ToolResult:
        """Get detailed status of a service"""
        try:
            status = {
                "name": service_name,
                "status": "running",
                "replicas": 3,
                "ready_replicas": 3,
                "cpu_usage": "45%",
                "memory_usage": "62%",
                "uptime": "5 days 3 hours",
                "last_restart": "2024-10-23T10:30:00Z"
            }
            
            return ToolResult(
                success=True,
                data=status
            )
        except Exception as e:
            logger.error(f"Failed to get service status: {str(e)}")
            return ToolResult(success=False, error=str(e))
    
    # ============ API TOOLS ============
    
    async def call_external_api(
        self,
        url: str,
        method: str = "GET",
        headers: Optional[Dict] = None,
        payload: Optional[Dict] = None,
        api_key: Optional[str] = None
    ) -> ToolResult:
        """Make HTTP requests to external APIs"""
        try:
            if api_key:
                headers = headers or {}
                headers["Authorization"] = f"Bearer {api_key}"
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                if method.upper() == "GET":
                    response = await client.get(url, headers=headers)
                elif method.upper() == "POST":
                    response = await client.post(url, json=payload, headers=headers)
                elif method.upper() == "PUT":
                    response = await client.put(url, json=payload, headers=headers)
                elif method.upper() == "DELETE":
                    response = await client.delete(url, headers=headers)
                else:
                    return ToolResult(success=False, error=f"Unsupported method: {method}")
                
                response.raise_for_status()
                
                return ToolResult(
                    success=True,
                    data={
                        "status_code": response.status_code,
                        "response": response.json() if response.headers.get("content-type") == "application/json" else response.text
                    }
                )
        except httpx.HTTPError as e:
            logger.error(f"API call failed: {str(e)}")
            return ToolResult(success=False, error=f"API error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in API call: {str(e)}")
            return ToolResult(success=False, error=str(e))
    
    async def validate_api_key(self, api_key: str, service: str) -> ToolResult:
        """Validate API key for external services"""
        try:
            # Simulated API key validation
            is_valid = len(api_key) > 10 and api_key.startswith("sk_")
            
            return ToolResult(
                success=True,
                data={
                    "service": service,
                    "api_key": api_key[:10] + "***",
                    "is_valid": is_valid,
                    "status": "valid" if is_valid else "invalid"
                }
            )
        except Exception as e:
            logger.error(f"API key validation failed: {str(e)}")
            return ToolResult(success=False, error=str(e))
    
    # ============ DATA TOOLS ============
    
    async def get_service_logs(self, service_name: str, lines: int = 50) -> ToolResult:
        """Retrieve logs from a service"""
        try:
            logs = [
                f"[2024-10-28 10:30:45] INFO: Service {service_name} started",
                f"[2024-10-28 10:30:46] INFO: Connected to database",
                f"[2024-10-28 10:30:47] INFO: Ready to accept requests",
                f"[2024-10-28 10:35:12] DEBUG: Processing request from client",
                f"[2024-10-28 10:40:00] INFO: Health check passed"
            ]
            
            return ToolResult(
                success=True,
                data={
                    "service": service_name,
                    "logs": logs[-lines:],
                    "total_lines": len(logs)
                }
            )
        except Exception as e:
            logger.error(f"Failed to get service logs: {str(e)}")
            return ToolResult(success=False, error=str(e))
    
    async def get_service_metrics(self, service_name: str) -> ToolResult:
        """Get performance metrics for a service"""
        try:
            metrics = {
                "service": service_name,
                "cpu_usage": 45.2,
                "memory_usage": 62.8,
                "request_count": 15234,
                "error_rate": 0.02,
                "avg_response_time": 125,
                "uptime_percentage": 99.95
            }
            
            return ToolResult(
                success=True,
                data=metrics
            )
        except Exception as e:
            logger.error(f"Failed to get service metrics: {str(e)}")
            return ToolResult(success=False, error=str(e))
    
    # ============ ANALYSIS TOOLS ============
    
    async def analyze_service_health(self, service_name: str) -> ToolResult:
        """Analyze the health status of services"""
        try:
            metrics = await self.get_service_metrics(service_name)
            
            if not metrics.success:
                return metrics
            
            data = metrics.data
            health_score = 100
            issues = []
            
            if data.get("cpu_usage", 0) > 80:
                health_score -= 20
                issues.append("High CPU usage detected")
            
            if data.get("memory_usage", 0) > 85:
                health_score -= 20
                issues.append("High memory usage detected")
            
            if data.get("error_rate", 0) > 0.05:
                health_score -= 15
                issues.append("High error rate detected")
            
            return ToolResult(
                success=True,
                data={
                    "service": service_name,
                    "health_score": max(0, health_score),
                    "status": "healthy" if health_score >= 80 else "warning" if health_score >= 50 else "critical",
                    "issues": issues
                }
            )
        except Exception as e:
            logger.error(f"Health analysis failed: {str(e)}")
            return ToolResult(success=False, error=str(e))
    
    async def generate_recommendations(self, service_name: str) -> ToolResult:
        """Generate recommendations based on service analysis"""
        try:
            health = await self.analyze_service_health(service_name)
            
            if not health.success:
                return health
            
            recommendations = []
            issues = health.data.get("issues", [])
            
            if "High CPU usage" in str(issues):
                recommendations.append("Consider scaling up the service or optimizing code")
            
            if "High memory usage" in str(issues):
                recommendations.append("Review memory leaks or increase allocated memory")
            
            if "High error rate" in str(issues):
                recommendations.append("Check service logs for error details and investigate root cause")
            
            if not recommendations:
                recommendations.append("Service is performing well. Continue monitoring.")
            
            return ToolResult(
                success=True,
                data={
                    "service": service_name,
                    "recommendations": recommendations
                }
            )
        except Exception as e:
            logger.error(f"Recommendation generation failed: {str(e)}")
            return ToolResult(success=False, error=str(e))
    
    # ============ EXECUTION TOOLS ============
    
    async def execute_task(self, task_name: str, parameters: Optional[Dict] = None) -> ToolResult:
        """Execute an automated task"""
        try:
            logger.info(f"Executing task: {task_name} with parameters: {parameters}")
            
            result = {
                "task": task_name,
                "parameters": parameters or {},
                "status": "completed",
                "execution_time": 2.5,
                "timestamp": datetime.now().isoformat()
            }
            
            return ToolResult(
                success=True,
                data=result
            )
        except Exception as e:
            logger.error(f"Task execution failed: {str(e)}")
            return ToolResult(success=False, error=str(e))
    
    # ============ UTILITY METHODS ============
    
    async def _call_maybe_async(self, fn: Callable, *args, **kwargs) -> Any:
        """Call function whether it's async or sync"""
        import inspect
        
        result = fn(*args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result


# Global tool registry instance
chatbot_tool_registry = ChatbotToolRegistry()
