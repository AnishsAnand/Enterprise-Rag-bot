import os
import logging
import sys
import signal
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from urllib.parse import urljoin
import json
import uuid
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
import uvicorn
import httpx

# Import core services
from app.api.routes import orchestrator
from app.api.routes import rag_widget
from app.api.routes.auth import router as auth_router
from app.core.database import init_db
from app.services.milvus_service import milvus_service
from app.services.ai_service import ai_service

# Import Intelligent Action Agent
try:
    from app.api.routes.action_bot import (
        IntelligentActionAgent,
        IntentType,
        ActionType,
        SessionState,
        Parameter,
        ParameterType,
        ActionIntent,
        ConversationSession
    )
    ACTION_AGENT_AVAILABLE = True
except ImportError:
    ACTION_AGENT_AVAILABLE = False
    logging.warning("⚠️ Intelligent Action Agent module not available")

load_dotenv()

# ======================== Configuration ========================
class Config:
    """Unified application configuration"""
    # Server settings
    HOST = os.getenv("APP_HOST", "0.0.0.0")
    PORT = int(os.getenv("APP_PORT", "8001"))
    RELOAD = os.getenv("DEV_RELOAD", "false").lower() in ("1", "true")
    WORKERS = int(os.getenv("APP_WORKERS", "2"))
    
    # JWT & Widget
    JWT_SECRET = os.getenv("WIDGET_JWT_SECRET", "dev-secret-change-me")
    JWT_ALG = "HS256"
    WIDGET_API_KEY = os.getenv("WIDGET_API_KEY", "dev-widget-key")
    WIDGET_STATIC_DIR = os.getenv("WIDGET_STATIC_DIR", "widget_static")
    WIDGET_URL = os.getenv("WIDGET_URL", "")
    
    # Service URLs
    ADMIN_SERVICE_URL = os.getenv("ADMIN_SERVICE_URL", "http://localhost:8001")
    RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://localhost:8001")
    SCRAPER_SERVICE_URL = os.getenv("SCRAPER_SERVICE_URL", "http://localhost:8003")
    AI_SERVICE_URL = os.getenv("AI_SERVICE_URL", "http://localhost:8004")
    
    # Tata IP Cloud Authentication
    TATA_AUTH_URL = os.getenv("TATA_AUTH_URL", "https://ipcloud.tatacommunications.com/portalservice/api/v1/getAuthToken")
    TATA_USERNAME = os.getenv("TATA_USERNAME","izo_cloud_admin@tatacommunications.onmicrosoft.com")
    TATA_PASSWORD = os.getenv("TATA_PASSWORD","Tata@1234")
    
    # Cluster API Configuration
    CLUSTER_API_BASE_URL = os.getenv("CLUSTER_API_BASE_URL", "https://ipcloud.tatacommunications.com/paasservice/paas")
    CLUSTER_API_PROJECT_ID = os.getenv("CLUSTER_API_PROJECT_ID", "1923")
    CLUSTER_API_TIMEOUT = int(os.getenv("CLUSTER_API_TIMEOUT", "30"))
    
    # Token Management
    TOKEN_REFRESH_BUFFER = 300  # Refresh token 5 minutes before expiry
    TOKEN_MAX_RETRIES = 3
    
    # CORS
    ALLOWED_ORIGINS_ENV = os.getenv("USER_ALLOWED_ORIGINS", "")
    ALLOW_ALL_ORIGINS = os.getenv("USER_ALLOW_ALL_ORIGINS", "false").lower() in ("1", "true", "yes")
    
    # Session settings
    SESSION_CLEANUP_INTERVAL = 300  # 5 minutes
    HEALTH_CHECK_INTERVAL = 60  # 1 minute

# ======================== Logging ========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('unified_app.log')
    ]
)
logger = logging.getLogger("unified_app")

# Suppress noisy logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# ======================== Authentication Manager ========================
class TataAuthManager:
    """Manages Tata IP Cloud authentication and token lifecycle"""
    
    def __init__(self, http_client: httpx.AsyncClient):
        self.http_client = http_client
        self.auth_url = Config.TATA_AUTH_URL
        self.username = Config.TATA_USERNAME
        self.password = Config.TATA_PASSWORD
        self.token: Optional[str] = None
        self.token_expiry: Optional[datetime] = None
        self.token_lock = asyncio.Lock()
        
    async def get_valid_token(self) -> str:
        """Get a valid authentication token, refreshing if necessary"""
        async with self.token_lock:
            # Check if current token is still valid
            if self.token and self.token_expiry:
                time_until_expiry = (self.token_expiry - datetime.now()).total_seconds()
                if time_until_expiry > Config.TOKEN_REFRESH_BUFFER:
                    logger.debug(f"✅ Using cached token (expires in {time_until_expiry:.0f}s)")
                    return self.token
            
            # Token missing or expiring soon - get new one
            logger.info("🔐 Fetching new authentication token...")
            return await self._fetch_new_token()
    
    async def _fetch_new_token(self) -> str:
        """Fetch a new authentication token from Tata API"""
        if not self.username or not self.password:
            raise ValueError("TATA_USERNAME and TATA_PASSWORD must be set in environment")
        
        for attempt in range(1, Config.TOKEN_MAX_RETRIES + 1):
            try:
                logger.info(f"🔑 Authentication attempt {attempt}/{Config.TOKEN_MAX_RETRIES}")
                
                response = await self.http_client.post(
                    self.auth_url,
                    auth=(self.username, self.password),
                    headers={
                        "Accept": "application/json",
                        "Content-Type": "application/json"
                    },
                    timeout=30.0
                )
                
                response.raise_for_status()
                data = response.json()
                
                # Extract token from response
                if isinstance(data, dict):
                    token = data.get('token') or data.get('access_token') or data.get('authToken')
                    if token:
                        self.token = token
                        # Set expiry (default 1 hour if not provided)
                        expires_in = data.get('expires_in', 3600)
                        self.token_expiry = datetime.now() + timedelta(seconds=expires_in)
                        
                        logger.info(f"✅ Authentication successful (expires at {self.token_expiry.isoformat()})")
                        return self.token
                
                # If we get here, token extraction failed
                logger.error(f"❌ Token not found in response: {data}")
                raise ValueError("Authentication response missing token")
                
            except httpx.HTTPStatusError as e:
                logger.error(f"❌ Authentication failed (attempt {attempt}): HTTP {e.response.status_code}")
                if e.response.status_code == 401:
                    raise ValueError("Invalid credentials - check TATA_USERNAME and TATA_PASSWORD")
                if attempt == Config.TOKEN_MAX_RETRIES:
                    raise
                await asyncio.sleep(2 ** attempt)
                
            except Exception as e:
                logger.exception(f"❌ Authentication error (attempt {attempt}): {e}")
                if attempt == Config.TOKEN_MAX_RETRIES:
                    raise
                await asyncio.sleep(2 ** attempt)
        
        raise RuntimeError("Failed to obtain authentication token after all retries")
    
    async def invalidate_token(self):
        """Invalidate current token to force refresh"""
        async with self.token_lock:
            logger.warning("⚠️ Invalidating authentication token")
            self.token = None
            self.token_expiry = None

# ======================== Request/Response Models ========================
class UserQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    max_results: int = Field(default=5, ge=1, le=50)
    include_images: bool = Field(default=True)
    session_id: Optional[str] = Field(None, description="Session ID for continuing conversation")

class ContinueConversationRequest(BaseModel):
    session_id: str = Field(..., description="Session ID from previous interaction")
    user_input: str = Field(..., min_length=1, description="User's response/input")

class ActionResponse(BaseModel):
    status: str
    message: str
    details: Optional[Dict[str, Any]] = None
    next_step: Optional[str] = None
    missing_fields: Optional[List[str]] = None
    suggestions: Optional[List[str]] = None
    options: Optional[List[str]] = None
    confidence: float = 0.0
    session_id: Optional[str] = None
    execution_time: Optional[float] = None
    parameter: Optional[Dict[str, Any]] = None
    collected_so_far: Optional[Dict[str, Any]] = None
    pending: Optional[List[str]] = None

# ======================== Cluster API Handler ========================
class ClusterAPIHandler:
    """Handler for Cluster API operations with authentication"""
    
    def __init__(self, http_client: httpx.AsyncClient, project_id: str, auth_manager: TataAuthManager):
        self.http_client = http_client
        self.project_id = project_id
        self.auth_manager = auth_manager
        self.base_url = f"{Config.CLUSTER_API_BASE_URL}/{project_id}"
        self.cluster_list_endpoint = "/clusterlist"
    
    async def get_clusters(
        self,
        endpoint_ids: Optional[List[int]] = None,
        filter_status: Optional[str] = None,
        filter_type: Optional[str] = None,
        retry_count: int = 0
    ) -> Dict[str, Any]:
        """Fetch clusters with automatic authentication"""
        
        try:
            if not endpoint_ids:
                endpoint_ids = [11, 12, 14, 162, 204]

            url = f"{self.base_url}{self.cluster_list_endpoint}"
            payload = {"endpoints": endpoint_ids}

            logger.info(f"🔍 Fetching clusters for endpoints: {endpoint_ids}")

            # Get valid authentication token
            try:
                token = await self.auth_manager.get_valid_token()
            except Exception as e:
                logger.error(f"❌ Failed to obtain authentication token: {e}")
                return {
                    "success": False,
                    "message": "Authentication failed",
                    "error": str(e)
                }

            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}"
            }

            response = await self.http_client.post(
                url,
                json=payload,
                headers=headers,
                timeout=Config.CLUSTER_API_TIMEOUT
            )

            # Handle 401 - token might be expired, retry once
            if response.status_code == 401 and retry_count == 0:
                logger.warning("⚠️ Received 401 - token may be expired, refreshing...")
                await self.auth_manager.invalidate_token()
                return await self.get_clusters(endpoint_ids, filter_status, filter_type, retry_count=1)

            response.raise_for_status()
            result = response.json()

            if result.get('status') != 'success':
                return {
                    'success': False,
                    'message': f"API returned status: {result.get('status')}",
                    'error': result.get('message')
                }

            clusters = self._parse_clusters(result.get('data', []))

            # Apply filters
            if filter_status and filter_status != 'all':
                clusters = [c for c in clusters if c.get('status', '').lower() == filter_status.lower()]

            if filter_type and filter_type != 'all':
                clusters = [c for c in clusters if c.get('type', '').upper() == filter_type.upper()]

            stats = self._generate_statistics(clusters)

            logger.info(f"✅ Retrieved {len(clusters)} clusters")

            return {
                'success': True,
                'clusters': clusters,
                'total_count': len(clusters),
                'statistics': stats,
                'endpoint_ids': endpoint_ids,
                'timestamp': datetime.now().isoformat()
            }

        except httpx.HTTPStatusError as e:
            logger.error(f"❌ HTTP error fetching clusters: {e.response.status_code}")
            return {
                'success': False,
                'message': f"API request failed with status {e.response.status_code}",
                'error': str(e)
            }
        except Exception as e:
            logger.exception(f"❌ Error fetching clusters: {e}")
            return {
                'success': False,
                'message': f"Failed to fetch clusters: {str(e)}",
                'error': str(e)
            }
    
    def _parse_clusters(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parse raw cluster data"""
        clusters = []
        
        for item in data:
            try:
                cluster = {
                    'cluster_id': int(item.get('clusterId', 0)),
                    'cluster_name': item.get('clusterName', 'Unknown'),
                    'location': item.get('location', 'Unknown'),
                    'endpoint': item.get('displayNameEndpoint', 'Unknown'),
                    'status': item.get('status', 'Unknown'),
                    'type': item.get('type', 'APP'),
                    'nodes': int(item.get('nodescount', 0)),
                    'k8s_version': item.get('kubernetesVersion', 'N/A'),
                    'backup_enabled': str(item.get('isIksBackupEnabled', 'false')).lower() == 'true',
                    'created': item.get('createdTime', ''),
                    'ci_master_id': int(item.get('ciMasterId', 0))
                }
                clusters.append(cluster)
            except Exception as e:
                logger.warning(f"⚠️ Failed to parse cluster: {e}")
                continue
        
        return clusters
    
    def _generate_statistics(self, clusters: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate cluster statistics"""
        if not clusters:
            return {}
        
        stats = {
            'total': len(clusters),
            'by_status': {},
            'by_type': {},
            'by_location': {},
            'total_nodes': 0,
            'backup_enabled_count': 0,
            'kubernetes_versions': {}
        }
        
        for cluster in clusters:
            status = cluster.get('status', 'Unknown')
            stats['by_status'][status] = stats['by_status'].get(status, 0) + 1
            
            cluster_type = cluster.get('type', 'APP')
            stats['by_type'][cluster_type] = stats['by_type'].get(cluster_type, 0) + 1
            
            location = cluster.get('endpoint', 'Unknown')
            stats['by_location'][location] = stats['by_location'].get(location, 0) + 1
            
            stats['total_nodes'] += cluster.get('nodes', 0)
            
            if cluster.get('backup_enabled'):
                stats['backup_enabled_count'] += 1
            
            k8s_version = cluster.get('k8s_version')
            if k8s_version and k8s_version != 'N/A':
                stats['kubernetes_versions'][k8s_version] = stats['kubernetes_versions'].get(k8s_version, 0) + 1
        
        return stats
    
    def format_clusters_output(self, clusters: List[Dict[str, Any]], show_details: bool = False) -> str:
        """Format clusters for display"""
        if not clusters:
            return "ℹ️ No clusters found matching your criteria."
        
        output = [f"📊 **Found {len(clusters)} Clusters**\n"]
        
        if show_details:
            for i, cluster in enumerate(clusters, 1):
                output.append(
                    f"\n**{i}. {cluster['cluster_name']}** ({cluster['type']})\n"
                    f"  • Location: {cluster['endpoint']}\n"
                    f"  • Status: {cluster['status']}\n"
                    f"  • Nodes: {cluster['nodes']}\n"
                    f"  • K8s Version: {cluster['k8s_version']}\n"
                    f"  • Backup: {'✅ Enabled' if cluster['backup_enabled'] else '❌ Disabled'}\n"
                    f"  • Cluster ID: {cluster['cluster_id']}"
                )
        else:
            by_location = {}
            for cluster in clusters:
                location = cluster['endpoint']
                if location not in by_location:
                    by_location[location] = []
                by_location[location].append(cluster)
            
            for location, loc_clusters in sorted(by_location.items()):
                output.append(f"\n### 📍 {location} ({len(loc_clusters)} clusters)")
                for cluster in loc_clusters:
                    status_icon = "✅" if cluster['status'] == "Healthy" else "⚠️"
                    output.append(
                        f"  {status_icon} **{cluster['cluster_name']}** "
                        f"({cluster['type']}) - "
                        f"{cluster['nodes']} nodes - "
                        f"{cluster['k8s_version']}"
                    )
        
        return "\n".join(output)

# ======================== Intelligent Query Router ========================
class IntelligentQueryRouter:
    """Routes queries intelligently between LLM, RAG, and Actions"""
    
    def __init__(self, ai_service):
        self.ai_service = ai_service
        
        # Action-oriented keywords
        self.action_keywords = {
            'cluster': ['list', 'show', 'get', 'fetch', 'display', 'retrieve', 'find'],
            'management': ['create', 'delete', 'update', 'modify', 'configure', 'setup', 'enable', 'disable'],
            'status': ['check', 'status', 'health', 'monitor', 'verify'],
            'filter': ['filter', 'where', 'in location', 'at', 'healthy', 'unhealthy', 'type']
        }
        
        # Question indicators
        self.question_indicators = [
            'what', 'how', 'why', 'when', 'where', 'who', 'which',
            'explain', 'describe', 'tell me about', 'can you explain'
        ]
        
        # Cluster-specific patterns
        self.cluster_patterns = [
            r'(?:list|show|get|fetch|display)\s+(?:all\s+)?(?:kubernetes\s+)?clusters?',
            r'clusters?\s+in\s+(\w+)',
            r'k8s\s+cluster',
            r'kubernetes\s+cluster',
            r'cluster\s+(?:status|info|details|health)',
            r'how many clusters'
        ]
    
    async def route_query(self, query: str) -> Dict[str, Any]:
        """
        Intelligently route query to appropriate handler
        Returns: {
            'route': 'action' | 'llm' | 'rag',
            'confidence': float,
            'reasoning': str,
            'detected_intent': str
        }
        """
        query_lower = query.lower().strip()
        
        # Step 1: Check for explicit cluster action patterns
        import re
        for pattern in self.cluster_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return {
                    'route': 'action',
                    'confidence': 0.95,
                    'reasoning': 'Explicit cluster operation detected',
                    'detected_intent': 'cluster_management'
                }
        
        # Step 2: Score based on keywords
        action_score = 0
        question_score = 0
        
        # Check action keywords
        for category, keywords in self.action_keywords.items():
            if any(kw in query_lower for kw in keywords):
                action_score += 1
                if category == 'cluster':
                    action_score += 2
        
        # Check question indicators
        question_score = sum(1 for indicator in self.question_indicators if indicator in query_lower)
        
        # Step 3: Use AI to analyze intent for ambiguous cases
        if action_score > 0 and question_score > 0:
            try:
                intent_analysis = await self._analyze_intent_with_ai(query)
                if intent_analysis:
                    return intent_analysis
            except Exception as e:
                logger.warning(f"⚠️ AI intent analysis failed: {e}")
        
        # Step 4: Make routing decision
        if action_score >= 2:
            return {
                'route': 'action',
                'confidence': min(0.7 + (action_score * 0.1), 0.95),
                'reasoning': f'High action score ({action_score})',
                'detected_intent': 'action_required'
            }
        elif question_score >= 2 or any(q in query_lower for q in ['what is', 'how does', 'explain']):
            return {
                'route': 'llm',
                'confidence': 0.8,
                'reasoning': 'Informational question detected',
                'detected_intent': 'information_seeking'
            }
        else:
            return {
                'route': 'rag',
                'confidence': 0.6,
                'reasoning': 'Default to RAG for general queries',
                'detected_intent': 'general_query'
            }
    
    async def _analyze_intent_with_ai(self, query: str) -> Optional[Dict[str, Any]]:
        """Use AI to analyze query intent for ambiguous cases"""
        try:
            prompt = f"""Analyze this user query and determine the intent:

Query: "{query}"

Respond with ONLY a JSON object in this exact format:
{{
    "intent": "action" or "question" or "general",
    "confidence": 0.0 to 1.0,
    "reasoning": "brief explanation"
}}

Guidelines:
- "action": User wants to perform an operation (list, create, modify, check status)
- "question": User wants to learn or understand something (what, how, why)
- "general": General conversation or unclear intent

Response:"""

            response = await self.ai_service.generate_text(
                prompt=prompt,
                max_tokens=150,
                temperature=0.1
            )
            
            # Parse JSON response
            import json
            import re
            
            # Extract JSON from response
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                result = json.loads(json_match.group())
                intent = result.get('intent', 'general')
                confidence = float(result.get('confidence', 0.5))
                reasoning = result.get('reasoning', 'AI analysis')
                
                route_map = {
                    'action': 'action',
                    'question': 'llm',
                    'general': 'rag'
                }
                
                return {
                    'route': route_map.get(intent, 'rag'),
                    'confidence': confidence,
                    'reasoning': f"AI analysis: {reasoning}",
                    'detected_intent': intent
                }
        
        except Exception as e:
            logger.debug(f"AI intent analysis error: {e}")
            return None

# ======================== Enhanced Action Agent ========================
class EnhancedActionAgent:
    """Enhanced Action Agent with Cluster API integration and authentication"""
    
    def __init__(self, base_agent, http_client: httpx.AsyncClient, auth_manager: TataAuthManager):
        self.base_agent = base_agent
        self.http_client = http_client
        self.auth_manager = auth_manager
        self.cluster_handler = ClusterAPIHandler(http_client, Config.CLUSTER_API_PROJECT_ID, auth_manager)
        
        # Extend action type detection patterns
        self.cluster_patterns = {
            'cluster_list': [
                r'(?:list|show|get|fetch|display)\s+(?:all\s+)?(?:kubernetes\s+)?clusters?',
                r'(?:show|get)\s+(?:me\s+)?(?:cluster|k8s)\s+(?:list|info)',
                r'clusters?\s+in\s+(\w+)',
                r'enable\s+k8s\s+cluster',
                r'kubernetes\s+cluster\s+(?:status|info|details)',
            ]
        }
    
    async def detect_cluster_intent(self, query: str) -> Optional[Dict[str, Any]]:
        """Detect cluster-related intents"""
        query_lower = query.lower()
        
        for action_type, patterns in self.cluster_patterns.items():
            for pattern in patterns:
                import re
                match = re.search(pattern, query_lower, re.IGNORECASE)
                if match:
                    return {
                        'type': 'cluster_list',
                        'confidence': 0.9,
                        'extracted_params': match.groups() if match.groups() else [],
                        'original_query': query
                    }
        
        return None
    
    async def handle_cluster_action(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle cluster-specific actions with authentication"""
        start_time = datetime.now()
        
        try:
            endpoints_param = params.get('endpoints', 'all')
            endpoint_ids = None
            
            # Parse endpoint selection
            if endpoints_param == 'all':
                endpoint_ids = [11, 12, 14, 162, 204]
            elif endpoints_param == 'mumbai':
                endpoint_ids = [11]
            elif endpoints_param == 'delhi':
                endpoint_ids = [12]
            elif endpoints_param == 'bengaluru':
                endpoint_ids = [14]
            elif endpoints_param == 'chennai':
                endpoint_ids = [162]
            elif endpoints_param == 'cressex':
                endpoint_ids = [204]
            elif endpoints_param == 'custom':
                custom_ids = params.get('custom_endpoint_ids', '')
                if custom_ids:
                    try:
                        endpoint_ids = [int(x.strip()) for x in custom_ids.split(',')]
                    except ValueError:
                        return {
                            'success': False,
                            'message': '❌ Invalid endpoint IDs format. Please provide comma-separated numbers.'
                        }
            
            # Fetch clusters with authentication
            result = await self.cluster_handler.get_clusters(
                endpoint_ids=endpoint_ids,
                filter_status=params.get('filter_status'),
                filter_type=params.get('filter_type')
            )
            
            if not result['success']:
                return result
            
            # Format output
            clusters = result['clusters']
            show_details = params.get('show_details', False)
            formatted_output = self.cluster_handler.format_clusters_output(clusters, show_details)
            
            # Generate statistics summary
            stats = result['statistics']
            stats_summary = [
                "\n\n📈 **Statistics:**",
                f"  • Total Clusters: {stats.get('total', 0)}",
                f"  • Total Nodes: {stats.get('total_nodes', 0)}",
                f"  • Backup Enabled: {stats.get('backup_enabled_count', 0)}",
                "\n**By Status:**"
            ]
            
            for status, count in stats.get('by_status', {}).items():
                stats_summary.append(f"  • {status}: {count}")
            
            stats_summary.append("\n**By Type:**")
            for cluster_type, count in stats.get('by_type', {}).items():
                stats_summary.append(f"  • {cluster_type}: {count}")
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'success': True,
                'message': f'✅ Successfully retrieved cluster information\n\n{formatted_output}\n{"".join(stats_summary)}',
                'details': {
                    'total_clusters': len(clusters),
                    'clusters': clusters,
                    'statistics': stats
                },
                'execution_time': execution_time
            }
            
        except Exception as e:
            logger.exception(f"❌ Cluster action execution failed: {e}")
            execution_time = (datetime.now() - start_time).total_seconds()
            return {
                'success': False,
                'message': f'❌ Failed to retrieve clusters: {str(e)}',
                'error': str(e),
                'execution_time': execution_time
            }
    
    def get_cluster_parameters(self) -> List:
        """Get parameters for cluster operations"""
        if not ACTION_AGENT_AVAILABLE:
            return []
        
        return [
            Parameter(
                name='endpoints',
                description='Location endpoints to query',
                required=False,
                param_type=ParameterType.STRING,
                default='all',
                choices=['all', 'mumbai', 'delhi', 'bengaluru', 'chennai', 'cressex', 'custom']
            ),
            Parameter(
                name='custom_endpoint_ids',
                description='Custom endpoint IDs (comma-separated)',
                required=False,
                param_type=ParameterType.STRING,
                validation_regex=r'^[\d,\s]+'
            ),
            Parameter(
                name='filter_status',
                description='Filter by cluster status',
                required=False,
                param_type=ParameterType.STRING,
                choices=['all', 'healthy', 'draft', 'unhealthy']
            ),
            Parameter(
                name='filter_type',
                description='Filter by cluster type',
                required=False,
                param_type=ParameterType.STRING,
                choices=['all', 'MGMT', 'APP']
            ),
            Parameter(
                name='show_details',
                description='Show detailed cluster information?',
                required=False,
                param_type=ParameterType.BOOLEAN,
                default=False
            )
        ]

# ======================== Application Manager ========================
class ApplicationManager:
    """Manages unified application lifecycle"""
    
    def __init__(self):
        self.action_agent: Optional[IntelligentActionAgent] = None
        self.enhanced_agent: Optional[EnhancedActionAgent] = None
        self.query_router: Optional[IntelligentQueryRouter] = None
        self.auth_manager: Optional[TataAuthManager] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        self.health_check_task: Optional[asyncio.Task] = None
        self.http_client: Optional[httpx.AsyncClient] = None
        self.is_shutting_down = False
        self.service_registry = {
            "admin": Config.ADMIN_SERVICE_URL,
            "rag": Config.RAG_SERVICE_URL,
            "scraper": Config.SCRAPER_SERVICE_URL,
            "cluster_api": Config.CLUSTER_API_BASE_URL
        }
    
    async def initialize(self):
        """Initialize all application components"""
        try:
            logger.info("🚀 Starting unified application initialization...")
            
            # Initialize HTTP client
            self.http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(60.0, connect=10.0),
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
                follow_redirects=True
            )
            logger.info("✅ HTTP client initialized")
            
            # Initialize Authentication Manager
            self.auth_manager = TataAuthManager(self.http_client)
            logger.info("✅ Authentication manager initialized")
            
            # Pre-fetch authentication token
            try:
                await self.auth_manager.get_valid_token()
                logger.info("✅ Initial authentication successful")
            except Exception as e:
                logger.warning(f"⚠️ Initial authentication failed (will retry on demand): {e}")
            
            # Initialize database
            try:
                await init_db()
                logger.info("✅ Database initialized")
            except Exception as e:
                logger.exception("Database initialization failed (continuing): %s", e)
            
            # Initialize Milvus
            try:
                await milvus_service.initialize()
                logger.info("✅ Milvus connected successfully")
            except Exception as e:
                logger.exception("Milvus initialization failed (continuing): %s", e)
            
            # Test AI services
            try:
                test_embeddings = await ai_service.generate_embeddings(["test"])
                if test_embeddings and len(test_embeddings[0]) > 0:
                    logger.info("✅ AI services operational")
                else:
                    logger.warning("⚠️ AI service test returned empty result")
            except Exception as e:
                logger.exception("AI service test failed (continuing): %s", e)
            
            # Initialize Query Router
            self.query_router = IntelligentQueryRouter(ai_service)
            logger.info("✅ Intelligent query router initialized")
            
            # Initialize Intelligent Action Agent
            if ACTION_AGENT_AVAILABLE:
                try:
                    logger.info("🤖 Initializing Intelligent Action Agent...")
                    self.action_agent = IntelligentActionAgent(
                        ai_service=ai_service,
                        service_registry=self.service_registry
                    )
                    await self.action_agent.initialize()
                    
                    # Create enhanced agent with cluster API support
                    self.enhanced_agent = EnhancedActionAgent(
                        self.action_agent,
                        self.http_client,
                        self.auth_manager
                    )
                    
                    logger.info("✅ Intelligent Action Agent initialized with Cluster API support")
                    
                    # Start background tasks
                    self.cleanup_task = asyncio.create_task(self._periodic_cleanup())
                    self.health_check_task = asyncio.create_task(self._periodic_health_check())
                    logger.info("✅ Background tasks started")
                except Exception as e:
                    logger.exception("Action Agent initialization failed (continuing): %s", e)
                    self.action_agent = None
                    self.enhanced_agent = None
            else:
                logger.info("ℹ️ Intelligent Action Agent not available, running in RAG-only mode")
            
            logger.info("✅ Unified application initialization complete")
            
        except Exception as e:
            logger.exception(f"❌ Failed to initialize application: {e}")
            raise
    
    async def shutdown(self):
        """Graceful shutdown of all components"""
        if self.is_shutting_down:
            return
        
        self.is_shutting_down = True
        logger.info("🛑 Starting graceful shutdown...")
        
        try:
            # Cancel background tasks
            if self.cleanup_task:
                self.cleanup_task.cancel()
                try:
                    await self.cleanup_task
                except asyncio.CancelledError:
                    pass
            
            if self.health_check_task:
                self.health_check_task.cancel()
                try:
                    await self.health_check_task
                except asyncio.CancelledError:
                    pass
            
            # Close HTTP client
            if self.http_client:
                await self.http_client.aclose()
                logger.info("✅ HTTP client closed")
            
            # Close Action Agent
            if self.action_agent:
                await self.action_agent.close()
            
            # Close Milvus
            try:
                await milvus_service.close()
                logger.info("✅ Milvus connection closed")
            except Exception as e:
                logger.warning("⚠️ Error closing Milvus: %s", e)
            
            logger.info("✅ Graceful shutdown complete")
            
        except Exception as e:
            logger.exception(f"❌ Error during shutdown: {e}")
    
    async def _periodic_cleanup(self):
        """Periodically clean up expired sessions"""
        while not self.is_shutting_down:
            try:
                await asyncio.sleep(Config.SESSION_CLEANUP_INTERVAL)
                
                if self.action_agent:
                    expired = await self.action_agent.cleanup_expired_sessions(
                        max_age_seconds=1800  # 30 minutes
                    )
                    if expired > 0:
                        logger.info(f"🧹 Cleaned up {expired} expired sessions")
                
                logger.debug("🧹 Periodic cleanup cycle completed")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"❌ Error in periodic cleanup: {e}")
    
    async def _periodic_health_check(self):
        """Periodically check service health"""
        while not self.is_shutting_down:
            try:
                await asyncio.sleep(Config.HEALTH_CHECK_INTERVAL)
                
                # Check authentication token validity
                if self.auth_manager and self.auth_manager.token_expiry:
                    time_until_expiry = (self.auth_manager.token_expiry - datetime.now()).total_seconds()
                    if time_until_expiry < 600:  # Less than 10 minutes
                        logger.warning(f"⚠️ Auth token expiring soon ({time_until_expiry:.0f}s)")
                
                # Check service health
                if self.action_agent:
                    health = await self.action_agent.check_service_health()
                    unhealthy = [
                        svc for svc, status in health.items() 
                        if not status.get("available")
                    ]
                    if unhealthy:
                        logger.warning(f"⚠️ Unhealthy services: {', '.join(unhealthy)}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"❌ Error in health check: {e}")

# ======================== Global Manager ========================
app_manager = ApplicationManager()

# ======================== FastAPI Lifespan ========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    await app_manager.initialize()
    
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}")
        asyncio.create_task(app_manager.shutdown())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    yield
    await app_manager.shutdown()

# ======================== FastAPI App ========================
app = FastAPI(
    title="Unified RAG + Intelligent Action Agent API with Cluster Integration",
    description="Production-grade intelligent system with RAG, automated task orchestration, and Kubernetes cluster management",
    version="2.3.0",
    lifespan=lifespan,
)

# ======================== CORS ========================
if Config.ALLOWED_ORIGINS_ENV:
    allowed_origins = [o.strip() for o in Config.ALLOWED_ORIGINS_ENV.split(",") if o.strip()]
else:
    allowed_origins = ["http://localhost:4201", "http://127.0.0.1:4201"]

if Config.ALLOW_ALL_ORIGINS:
    allowed_origins = ["*"]

logger.info("✅ Allowed origins: %s", allowed_origins)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS", "DELETE", "PUT", "PATCH"],
    allow_headers=["*"],
)

# ======================== Security Headers ========================
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        try:
            if allowed_origins and allowed_origins != ["*"]:
                frame_ancestors = " ".join(allowed_origins)
                csp_value = f"default-src 'self'; frame-ancestors {frame_ancestors};"
                response.headers["Content-Security-Policy"] = csp_value
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["Referrer-Policy"] = "no-referrer-when-downgrade"
        except Exception:
            pass
        return response

app.add_middleware(SecurityHeadersMiddleware)

# ======================== Include Routers ========================
app.include_router(auth_router)
app.include_router(orchestrator.router)

if hasattr(rag_widget, "router"):
    app.include_router(rag_widget.router)
    logger.info("✅ Included rag_widget.router")
else:
    logger.warning("⚠️ rag_widget.router not available")

# ======================== Unified Chat API ========================
@app.post("/api/chat/query")
async def unified_chat_query(request: UserQueryRequest, background_tasks: BackgroundTasks):
    """
    Unified endpoint with intelligent routing between RAG, LLM, and Action Agent
    Supports cluster API integration and dynamic task automation
    """
    try:
        query = request.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        logger.info(f"📨 Received query: '{query[:100]}...'")
        
        # If continuing an existing session, route directly to action agent
        if request.session_id and app_manager.action_agent:
            logger.info(f"🔄 Continuing session: {request.session_id}")
            try:
                result = await app_manager.action_agent.handle_query(
                    user_query=query,
                    session_id=request.session_id
                )
                return result
            except Exception as e:
                logger.exception(f"Session continuation error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # For new queries, use intelligent routing
        if app_manager.query_router and app_manager.enhanced_agent:
            try:
                # Get routing decision
                routing = await app_manager.query_router.route_query(query)
                logger.info(f"🎯 Routing decision: {routing['route']} (confidence: {routing['confidence']:.2f})")
                
                # Route to Action Agent for action-oriented queries
                if routing['route'] == 'action' and routing['confidence'] > 0.7:
                    # Check for cluster-specific intent
                    cluster_intent = await app_manager.enhanced_agent.detect_cluster_intent(query)
                    
                    if cluster_intent and cluster_intent['confidence'] > 0.7:
                        logger.info(f"🎯 Detected cluster intent: {cluster_intent['type']}")
                        
                        # Start interactive parameter collection
                        session_id = str(uuid.uuid4())
                        
                        action_intent_obj = ActionIntent(
                            action_type=ActionType.CLUSTER_LIST,
                            confidence=cluster_intent['confidence'],
                            service_target='cluster_api',
                            endpoint='/clusterlist',
                            method='POST',
                            parameters=app_manager.enhanced_agent.get_cluster_parameters(),
                            description='Retrieve Kubernetes cluster information',
                            requires_confirmation=True
                        )
                        
                        session = ConversationSession(
                            session_id=session_id,
                            user_query=query,
                            intent_type=IntentType.ACTION,
                            action_intent=action_intent_obj,
                            state=SessionState.PARAMETER_COLLECTION
                        )
                        
                        app_manager.action_agent.active_sessions[session_id] = session
                        
                        return {
                            'status': 'parameter_collection',
                            'intent_type': 'cluster_action',
                            'message': '🎯 I can help you retrieve Kubernetes cluster information!\n\n**Which locations would you like to query?**\n\nOptions:\n  • **all** - All available locations\n  • **mumbai** - Mumbai-BKC\n  • **delhi** - Delhi\n  • **bengaluru** - Bengaluru\n  • **chennai** - Chennai-AMB\n  • **cressex** - Cressex\n  • **custom** - Specify custom endpoint IDs',
                            'parameter': {
                                'name': 'endpoints',
                                'description': 'Location endpoints to query',
                                'type': 'string',
                                'choices': ['all', 'mumbai', 'delhi', 'bengaluru', 'chennai', 'cressex', 'custom']
                            },
                            'session_id': session_id,
                            'options': ['all', 'mumbai', 'delhi', 'bengaluru', 'chennai', 'cressex'],
                            'routing_info': routing
                        }
                    
                    # Let the intelligent agent decide for non-cluster actions
                    result = await app_manager.action_agent.handle_query(query)
                    
                    if isinstance(result, dict):
                        result['routing_info'] = routing
                    
                    return result
                
                # Route to LLM for informational questions
                elif routing['route'] == 'llm' and routing['confidence'] > 0.7:
                    logger.info(f"📚 Generating LLM response")
                    response_text = await ai_service.generate_text(
                        prompt=f"""You are a helpful AI assistant. Answer this question clearly and concisely:

Question: {query}

Provide a helpful, accurate answer:""",
                        max_tokens=500,
                        temperature=0.7
                    )
                    
                    return {
                        'query': query,
                        'answer': response_text,
                        'source': 'llm_direct',
                        'confidence': routing['confidence'],
                        'timestamp': datetime.now().isoformat(),
                        'steps': [],
                        'images': [],
                        'routing_info': routing
                    }
                
            except Exception as e:
                logger.exception(f"Intelligent routing error: {e}")
        
        # Default fallback to RAG
        logger.info(f"📚 Routing to RAG: '{query[:100]}...'")
        search_results = await milvus_service.search_documents(query, n_results=request.max_results)
        
        if not search_results:
            return {
                "query": query,
                "answer": "I don't have enough information to answer that question.",
                "images": [],
                "timestamp": datetime.now().isoformat(),
                "source": "no_results",
                "confidence": 0.0,
                "results_found": 0
            }
        
        context = [r.get("content", "") for r in search_results[:5]]
        
        try:
            enhanced = await ai_service.generate_enhanced_response(query, context, None)
            answer = enhanced.get("text") if isinstance(enhanced, dict) else enhanced
            confidence = enhanced.get("quality_score", 0.0) if isinstance(enhanced, dict) else 0.0
        except Exception as e:
            logger.exception(f"Enhanced response generation failed: {e}")
            answer = "\n\n".join(context[:2])
            confidence = 0.5
        
        return {
            "query": query,
            "answer": answer or "Unable to generate answer.",
            "steps": [],
            "images": [],
            "confidence": confidence,
            "results_found": len(search_results),
            "results_used": min(len(search_results), request.max_results),
            "timestamp": datetime.now().isoformat(),
            "source": "rag"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unified query error: {e}")
        raise HTTPException(status_code=500, detail="An error occurred processing your request")

# ======================== Conversation Continuation ========================
@app.post("/api/chat/continue", response_model=ActionResponse)
async def continue_conversation(request: ContinueConversationRequest):
    """
    Continue an ongoing multi-turn conversation with the action agent
    """
    try:
        if not app_manager.action_agent:
            raise HTTPException(status_code=503, detail="Action agent not available")
        
        logger.info(f"🔄 Continuing session {request.session_id}: '{request.user_input[:50]}...'")
        
        session = app_manager.action_agent.active_sessions.get(request.session_id)
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Handle cluster action session
        if session and session.action_intent and session.action_intent.service_target == 'cluster_api':
            if session.state == SessionState.PARAMETER_COLLECTION:
                # Collect parameter
                pending_params = [
                    p for p in session.action_intent.parameters 
                    if p.name in session.parameters_pending
                ]
                
                if pending_params:
                    current_param = pending_params[0]
                    session.parameters_collected[current_param.name] = request.user_input
                    current_param.collected = True
                    current_param.value = request.user_input
                    session.parameters_pending.remove(current_param.name)
                
                # Check for more parameters
                remaining = [
                    p for p in session.action_intent.parameters 
                    if p.required and not p.collected
                ]
                
                if remaining:
                    next_param = remaining[0]
                    
                    if next_param.name == 'custom_endpoint_ids':
                        message = '📝 **Please enter custom endpoint IDs** (comma-separated numbers)\n\nExample: 11,12,14'
                    elif next_param.name == 'filter_status':
                        message = '🔍 **Filter by cluster status?**\n\nOptions:\n  • all\n  • healthy\n  • draft\n  • unhealthy'
                    elif next_param.name == 'filter_type':
                        message = '📦 **Filter by cluster type?**\n\nOptions:\n  • all\n  • MGMT\n  • APP'
                    else:
                        message = f'Please provide: {next_param.description}'
                    
                    return ActionResponse(
                        status='parameter_collection',
                        message=message,
                        parameter=next_param.to_dict(),
                        collected_so_far=session.parameters_collected,
                        pending=session.parameters_pending,
                        session_id=request.session_id,
                        options=next_param.choices if hasattr(next_param, 'choices') else None
                    )
                
                # All parameters collected, move to confirmation
                session.state = SessionState.CONFIRMATION_PENDING
                session.awaiting_confirmation = True
                
                summary = "📋 **Cluster Query Summary**\n\n"
                summary += f"**Endpoints**: {session.parameters_collected.get('endpoints', 'all')}\n"
                if 'filter_status' in session.parameters_collected:
                    summary += f"**Status Filter**: {session.parameters_collected['filter_status']}\n"
                if 'filter_type' in session.parameters_collected:
                    summary += f"**Type Filter**: {session.parameters_collected['filter_type']}\n"
                
                summary += "\n✅ **Ready to fetch cluster information. Proceed?**"
                
                return ActionResponse(
                    status='awaiting_confirmation',
                    message=summary,
                    options=['Yes, proceed', 'No, cancel'],
                    session_id=request.session_id
                )
            
            elif session.state == SessionState.CONFIRMATION_PENDING:
                user_input_lower = request.user_input.lower().strip()
                
                if any(word in user_input_lower for word in ['yes', 'proceed', 'confirm', 'go', 'continue']):
                    # Execute cluster action
                    params = session.parameters_collected
                    result = await app_manager.enhanced_agent.handle_cluster_action(params)
                    
                    session.execution_result = result
                    session.state = SessionState.COMPLETED if result['success'] else SessionState.FAILED
                    
                    return ActionResponse(
                        status='completed' if result['success'] else 'failed',
                        message=result['message'],
                        details=result.get('details'),
                        execution_time=result.get('execution_time'),
                        session_id=request.session_id
                    )
                else:
                    session.state = SessionState.CANCELLED
                    return ActionResponse(
                        status='cancelled',
                        message='❌ Cluster query cancelled.',
                        session_id=request.session_id
                    )
        
        # Standard session continuation
        result = await app_manager.action_agent.handle_query(
            user_query=request.user_input,
            session_id=request.session_id
        )
        
        return ActionResponse(
            status=result.get('status', 'unknown'),
            message=result.get('message', ''),
            details=result.get('details'),
            next_step=result.get('next_step'),
            parameter=result.get('parameter'),
            options=result.get('options'),
            collected_so_far=result.get('collected_so_far', {}),
            pending=result.get('pending', []),
            session_id=result.get('session_id'),
            execution_time=result.get('execution_time')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Conversation continuation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ======================== Cluster API Direct Endpoints ========================
@app.post("/api/clusters/list")
async def list_clusters(
    endpoints: str = 'all',
    filter_status: Optional[str] = None,
    filter_type: Optional[str] = None,
    show_details: bool = False
):
    """Direct endpoint to list Kubernetes clusters with authentication"""
    try:
        if not app_manager.enhanced_agent:
            raise HTTPException(status_code=503, detail="Cluster API not available")
        
        params = {
            'endpoints': endpoints,
            'filter_status': filter_status,
            'filter_type': filter_type,
            'show_details': show_details
        }
        
        result = await app_manager.enhanced_agent.handle_cluster_action(params)
        
        if not result['success']:
            raise HTTPException(status_code=500, detail=result.get('message'))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Cluster list error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/clusters/endpoints")
async def list_available_endpoints():
    """List available cluster endpoints"""
    return {
        "endpoints": [
            {"id": 11, "name": "mumbai", "location": "Mumbai-BKC"},
            {"id": 12, "name": "delhi", "location": "Delhi"},
            {"id": 14, "name": "bengaluru", "location": "Bengaluru"},
            {"id": 162, "name": "chennai", "location": "Chennai-AMB"},
            {"id": 204, "name": "cressex", "location": "Cressex"}
        ],
        "default": [11, 12, 14, 162, 204]
    }

# ======================== Authentication Endpoints ========================
@app.post("/api/auth/token/refresh")
async def refresh_auth_token():
    """Manually refresh the authentication token"""
    try:
        if not app_manager.auth_manager:
            raise HTTPException(status_code=503, detail="Authentication manager not available")
        
        await app_manager.auth_manager.invalidate_token()
        token = await app_manager.auth_manager.get_valid_token()
        
        return {
            "status": "success",
            "message": "Token refreshed successfully",
            "expires_at": app_manager.auth_manager.token_expiry.isoformat() if app_manager.auth_manager.token_expiry else None,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.exception(f"Token refresh error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/auth/token/status")
async def check_token_status():
    """Check current authentication token status"""
    try:
        if not app_manager.auth_manager:
            raise HTTPException(status_code=503, detail="Authentication manager not available")
        
        if not app_manager.auth_manager.token or not app_manager.auth_manager.token_expiry:
            return {
                "status": "no_token",
                "message": "No authentication token available",
                "timestamp": datetime.now().isoformat()
            }
        
        time_until_expiry = (app_manager.auth_manager.token_expiry - datetime.now()).total_seconds()
        
        return {
            "status": "valid" if time_until_expiry > 0 else "expired",
            "expires_at": app_manager.auth_manager.token_expiry.isoformat(),
            "time_until_expiry_seconds": max(0, time_until_expiry),
            "needs_refresh": time_until_expiry < Config.TOKEN_REFRESH_BUFFER,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.exception(f"Token status check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ======================== Session Management ========================
@app.get("/api/sessions")
async def list_sessions():
    """List all active sessions"""
    try:
        if not app_manager.action_agent:
            return {
                "active_sessions": [], 
                "count": 0, 
                "timestamp": datetime.now().isoformat()
            }
        
        sessions = await app_manager.action_agent.get_active_sessions()
        return {
            "active_sessions": sessions,
            "count": len(sessions),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.exception(f"Failed to list sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/sessions/{session_id}")
async def clear_session(session_id: str):
    """Clear/delete a specific session"""
    try:
        if not app_manager.action_agent:
            raise HTTPException(status_code=503, detail="Action agent not available")
        
        success = await app_manager.action_agent.clear_session(session_id)
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {
            "message": "Session cleared successfully", 
            "session_id": session_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to clear session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ======================== Service Management ========================
@app.get("/api/services/health")
async def services_health():
    """Check health of all backend services"""
    try:
        health = {
            "services": {},
            "timestamp": datetime.now().isoformat(),
            "action_agent_available": False,
            "cluster_api_available": False,
            "authentication_status": "unknown"
        }
        
        # Check authentication
        if app_manager.auth_manager:
            try:
                token_status = await check_token_status()
                health["authentication_status"] = token_status.get("status", "unknown")
            except Exception as e:
                health["authentication_status"] = "error"
        
        # Check action agent services
        if app_manager.action_agent:
            service_health = await app_manager.action_agent.check_service_health()
            health["services"] = service_health
            health["action_agent_available"] = True
        
        # Check cluster API
        if app_manager.enhanced_agent:
            health["cluster_api_available"] = True
            health["services"]["cluster_api"] = {
                "available": True,
                "status": "healthy",
                "base_url": Config.CLUSTER_API_BASE_URL,
                "project_id": Config.CLUSTER_API_PROJECT_ID
            }
        
        return health
        
    except Exception as e:
        logger.exception(f"Failed to check services health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ======================== Health Check ========================
@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    try:
        health_status = {
            "status": "healthy",
            "service": "unified_rag_action_agent_cluster",
            "timestamp": datetime.now().isoformat(),
            "version": app.version,
            "components": {}
        }
        
        # Check Milvus
        try:
            stats = await milvus_service.get_collection_stats()
            health_status["components"]["milvus"] = {
                "status": "healthy",
                "document_count": stats.get("document_count", 0) if isinstance(stats, dict) else 0
            }
        except Exception as e:
            health_status["components"]["milvus"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["status"] = "degraded"
        
        # Check AI Service
        try:
            health = await ai_service.get_service_health()
            is_healthy = health.get("overall_status") == "healthy"
            health_status["components"]["ai_service"] = {
                "status": "healthy" if is_healthy else "degraded",
                "model": health.get("service", {}).get("current_model") if isinstance(health, dict) else None
            }
        except Exception as e:
            health_status["components"]["ai_service"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["status"] = "degraded"
        
        # Check Action Agent
        if ACTION_AGENT_AVAILABLE and app_manager.action_agent:
            try:
                service_health = await app_manager.action_agent.check_service_health()
                active_sessions = await app_manager.action_agent.get_active_sessions()
                health_status["components"]["action_agent"] = {
                    "status": "healthy",
                    "active_sessions": len(active_sessions),
                    "services": service_health
                }
            except Exception as e:
                health_status["components"]["action_agent"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
        else:
            health_status["components"]["action_agent"] = {
                "status": "not_available"
            }
        
        # Check Cluster API
        if app_manager.enhanced_agent:
            try:
                test_result = await app_manager.enhanced_agent.cluster_handler.get_clusters(
                    endpoint_ids=[11]
                )
                health_status["components"]["cluster_api"] = {
                    "status": "healthy" if test_result.get('success') else "unhealthy",
                    "base_url": Config.CLUSTER_API_BASE_URL,
                    "project_id": Config.CLUSTER_API_PROJECT_ID
                }
            except Exception as e:
                health_status["components"]["cluster_api"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
        
        # Check Query Router
        if app_manager.query_router:
            health_status["components"]["query_router"] = {
                "status": "healthy",
                "enabled": True
            }
        
        status_code = 200 if health_status["status"] == "healthy" else 503
        return JSONResponse(health_status, status_code=status_code)
        
    except Exception as e:
        logger.exception("Health check failed")
        return JSONResponse({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }, status_code=503)

# ======================== Root Endpoint ========================
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Unified RAG + Intelligent Action Agent API with Cluster Integration",
        "version": app.version,
        "timestamp": datetime.now().isoformat(),
        "features": {
            "rag_retrieval": True,
            "intelligent_action_agent": ACTION_AGENT_AVAILABLE and app_manager.action_agent is not None,
            "multi_turn_conversation": ACTION_AGENT_AVAILABLE and app_manager.action_agent is not None,
            "cluster_api_integration": app_manager.enhanced_agent is not None,
            "intelligent_query_routing": app_manager.query_router is not None,
            "authentication_management": app_manager.auth_manager is not None,
            "websocket": True,
            "service_discovery": ACTION_AGENT_AVAILABLE and app_manager.action_agent is not None
        },
        "endpoints": {
            "chat": "/api/chat/query",
            "continue_conversation": "/api/chat/continue",
            "cluster_list": "/api/clusters/list",
            "cluster_endpoints": "/api/clusters/endpoints",
            "auth_token_refresh": "/api/auth/token/refresh",
            "auth_token_status": "/api/auth/token/status",
            "sessions": "/api/sessions",
            "services_health": "/api/services/health",
            "health": "/health"
        },
        "authentication": {
            "provider": "Tata IP Cloud",
            "auth_url": Config.TATA_AUTH_URL,
            "token_refresh_buffer": f"{Config.TOKEN_REFRESH_BUFFER}s"
        }
    }

# ======================== Global Exception Handler ========================
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for production safety"""
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "An internal error occurred",
            "details": {
                "error": str(exc),
                "type": type(exc).__name__,
                "path": str(request.url.path)
            },
            "timestamp": datetime.now().isoformat()
        }
    )

# ======================== Main Entry Point ========================
if __name__ == "__main__":
    logger.info("🚀 Starting Unified RAG + Intelligent Action Agent API with Cluster Integration...")
    logger.info(f"📍 Server will run on {Config.HOST}:{Config.PORT}")
    logger.info(f"🔧 Reload mode: {Config.RELOAD}")
    logger.info(f"🤖 Intelligent Action Agent: {'Enabled' if ACTION_AGENT_AVAILABLE else 'Disabled'}")
    logger.info(f"🔗 Cluster API: {Config.CLUSTER_API_BASE_URL}")
    logger.info(f"🔐 Authentication URL: {Config.TATA_AUTH_URL}")
    logger.info(f"🎯 Intelligent Query Routing: Enabled")
    
    uvicorn.run(
        "unified_main:app",
        host=Config.HOST,
        port=Config.PORT,
        reload=Config.RELOAD,
        workers=Config.WORKERS if not Config.RELOAD else 1,
        log_level=os.getenv("LOG_LEVEL", "info"),
        access_log=True
    )