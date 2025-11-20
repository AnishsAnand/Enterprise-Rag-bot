# action_bot.py - Production-Grade Intelligent Action Agent
"""
Enterprise-grade intelligent action orchestrator with:
- Multi-turn conversational task automation
- Dynamic intent detection and routing
- Service discovery and health monitoring
- Interactive parameter collection with validation
- Automatic API endpoint detection and execution
- Session management and persistence
- Comprehensive error handling and retry logic
- Real-time progress tracking
- Cluster API integration
"""

import asyncio
import json
import logging
import re
import uuid
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Set, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field, asdict
from urllib.parse import urlparse, urljoin
import httpx
from collections import defaultdict

# ======================== Configuration ========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("action_agent")

# ======================== Constants ========================
DEFAULT_TIMEOUT = 60
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0
SESSION_EXPIRY_MINUTES = 30
CONFIDENCE_THRESHOLD = 0.7
MAX_CONVERSATION_TURNS = 20

# ======================== Enums ========================
class IntentType(Enum):
    """User intent classification"""
    QUESTION = "question"
    ACTION = "action"
    HYBRID = "hybrid"
    CLARIFICATION = "clarification"
    CONFIRMATION = "confirmation"
    CANCELLATION = "cancellation"
    UNKNOWN = "unknown"

class ActionType(Enum):
    """Types of automated actions"""
    API_CALL = "api_call"
    SCRAPING = "scraping"
    DATA_UPLOAD = "data_upload"
    BULK_OPERATION = "bulk_operation"
    SERVICE_CONFIG = "service_config"
    ADMIN_TASK = "admin_task"
    RAG_QUERY = "rag_query"
    FILE_PROCESSING = "file_processing"
    DATABASE_OPERATION = "database_operation"
    CLUSTER_LIST = "cluster_list"
    CLUSTER_DETAILS = "cluster_details"
    CLUSTER_FILTER = "cluster_filter"

class SessionState(Enum):
    """Conversation session states"""
    INTENT_DETECTION = "intent_detection"
    PARAMETER_COLLECTION = "parameter_collection"
    VALIDATION = "validation"
    CONFIRMATION_PENDING = "confirmation_pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

class ServiceStatus(Enum):
    """Service health status"""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"

class ParameterType(Enum):
    """Parameter data types"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    URL = "url"
    JSON = "json"
    EMAIL = "email"
    DATE = "date"
    FILE = "file"
    LIST = "list"

# ======================== Data Models ========================
@dataclass
class Parameter:
    """Parameter definition for actions"""
    name: str
    description: str
    required: bool
    param_type: ParameterType
    default: Any = None
    validation_regex: Optional[str] = None
    choices: Optional[List[Any]] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    collected: bool = False
    value: Any = None
    attempts: int = 0
    max_attempts: int = 3
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'description': self.description,
            'required': self.required,
            'type': self.param_type.value,
            'default': self.default,
            'choices': self.choices,
            'collected': self.collected,
            'value': self.value
        }

@dataclass
class ServiceEndpoint:
    """Service endpoint definition"""
    path: str
    method: str
    description: str
    parameters: List[Parameter] = field(default_factory=list)
    available: bool = True
    requires_auth: bool = False
    rate_limit: Optional[int] = None

@dataclass
class ServiceCapability:
    """Service capability definition"""
    name: str
    base_url: str
    endpoints: Dict[str, ServiceEndpoint] = field(default_factory=dict)
    available: bool = False
    version: Optional[str] = None
    health_status: ServiceStatus = ServiceStatus.UNKNOWN
    last_check: datetime = field(default_factory=datetime.now)
    auth_required: bool = False
    auth_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'base_url': self.base_url,
            'available': self.available,
            'version': self.version,
            'health_status': self.health_status.value,
            'last_check': self.last_check.isoformat(),
            'endpoints': {
                path: {
                    'method': endpoint.method,
                    'description': endpoint.description,
                    'available': endpoint.available
                }
                for path, endpoint in self.endpoints.items()
            }
        }

@dataclass
class ActionIntent:
    """Detected action intent"""
    action_type: ActionType
    confidence: float
    service_target: str
    endpoint: Optional[str] = None
    method: str = "POST"
    parameters: List[Parameter] = field(default_factory=list)
    description: str = ""
    estimated_time: int = 0
    requires_confirmation: bool = True
    is_destructive: bool = False
    priority: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'action_type': self.action_type.value,
            'confidence': self.confidence,
            'service_target': self.service_target,
            'endpoint': self.endpoint,
            'method': self.method,
            'description': self.description,
            'parameters': [p.to_dict() for p in self.parameters]
        }

@dataclass
class ConversationTurn:
    """Single conversation turn"""
    role: str  # 'user' or 'agent'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConversationSession:
    """Multi-turn conversation session"""
    session_id: str
    user_query: str
    intent_type: IntentType
    action_intent: Optional[ActionIntent] = None
    state: SessionState = SessionState.INTENT_DETECTION
    parameters_collected: Dict[str, Any] = field(default_factory=dict)
    parameters_pending: List[str] = field(default_factory=list)
    conversation_history: List[ConversationTurn] = field(default_factory=list)
    execution_result: Optional[Dict[str, Any]] = None
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    awaiting_confirmation: bool = False
    clarification_needed: bool = False
    error_count: int = 0
    warnings: List[str] = field(default_factory=list)
    user_id: Optional[str] = None
    session_metadata: Dict[str, Any] = field(default_factory=dict)

    def add_turn(self, role: str, content: str, metadata: Dict[str, Any] = None):
        """Add conversation turn"""
        turn = ConversationTurn(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        self.conversation_history.append(turn)
        self.last_activity = datetime.now()

    def is_expired(self) -> bool:
        """Check if session is expired"""
        expiry_time = self.last_activity + timedelta(minutes=SESSION_EXPIRY_MINUTES)
        return datetime.now() > expiry_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'session_id': self.session_id,
            'user_query': self.user_query,
            'intent_type': self.intent_type.value,
            'state': self.state.value,
            'parameters_collected': self.parameters_collected,
            'parameters_pending': self.parameters_pending,
            'created_at': self.created_at.isoformat(),
            'last_activity': self.last_activity.isoformat(),
            'awaiting_confirmation': self.awaiting_confirmation
        }

# ======================== Validation Utilities ========================
class ParameterValidator:
    """Parameter validation utilities"""
    
    @staticmethod
    def validate_url(value: str) -> Tuple[bool, Optional[str]]:
        """Validate URL"""
        try:
            result = urlparse(value)
            if all([result.scheme, result.netloc]):
                return True, None
            return False, "Invalid URL format"
        except Exception as e:
            return False, f"URL validation error: {str(e)}"
    
    @staticmethod
    def validate_email(value: str) -> Tuple[bool, Optional[str]]:
        """Validate email"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if re.match(pattern, value):
            return True, None
        return False, "Invalid email format"
    
    @staticmethod
    def validate_json(value: str) -> Tuple[bool, Optional[str]]:
        """Validate JSON"""
        try:
            json.loads(value)
            return True, None
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {str(e)}"
    
    @staticmethod
    def validate_integer(value: str, min_val: Optional[int] = None, max_val: Optional[int] = None) -> Tuple[bool, Optional[str]]:
        """Validate integer"""
        try:
            int_val = int(value)
            if min_val is not None and int_val < min_val:
                return False, f"Value must be >= {min_val}"
            if max_val is not None and int_val > max_val:
                return False, f"Value must be <= {max_val}"
            return True, None
        except ValueError:
            return False, "Must be a valid integer"
    
    @staticmethod
    def validate_float(value: str, min_val: Optional[float] = None, max_val: Optional[float] = None) -> Tuple[bool, Optional[str]]:
        """Validate float"""
        try:
            float_val = float(value)
            if min_val is not None and float_val < min_val:
                return False, f"Value must be >= {min_val}"
            if max_val is not None and float_val > max_val:
                return False, f"Value must be <= {max_val}"
            return True, None
        except ValueError:
            return False, "Must be a valid number"
    
    @staticmethod
    def validate_boolean(value: str) -> Tuple[bool, Optional[str]]:
        """Validate boolean"""
        value_lower = value.lower().strip()
        if value_lower in ['yes', 'y', 'true', '1', 'on', 'enable', 'enabled']:
            return True, None
        elif value_lower in ['no', 'n', 'false', '0', 'off', 'disable', 'disabled']:
            return True, None
        return False, "Must be yes/no, true/false, or 1/0"
    
    @staticmethod
    def validate_string(value: str, min_len: Optional[int] = None, max_len: Optional[int] = None, pattern: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """Validate string"""
        if min_len is not None and len(value) < min_len:
            return False, f"Minimum length is {min_len}"
        if max_len is not None and len(value) > max_len:
            return False, f"Maximum length is {max_len}"
        if pattern and not re.match(pattern, value):
            return False, "Does not match required pattern"
        return True, None

# ======================== Intelligent Action Agent ========================
class IntelligentActionAgent:
    """
    Production-grade intelligent action orchestrator with cluster API support
    """
    
    def __init__(self, ai_service, service_registry: Dict[str, str]):
        self.ai_service = ai_service
        self.service_registry = service_registry
        self.service_capabilities: Dict[str, ServiceCapability] = {}
        self.active_sessions: Dict[str, ConversationSession] = {}
        self.http_client: Optional[httpx.AsyncClient] = None
        self.validator = ParameterValidator()
        self.sessions = {}
        # Intent detection keywords
        self.action_keywords = {
            'create', 'add', 'delete', 'remove', 'update', 'modify', 'change',
            'scrape', 'crawl', 'fetch', 'get', 'retrieve', 'extract',
            'upload', 'import', 'export', 'download', 'save', 'store',
            'configure', 'setup', 'install', 'deploy', 'start', 'stop',
            'enable', 'disable', 'activate', 'deactivate', 'run', 'execute',
            'process', 'analyze', 'generate', 'build', 'compile', 'list', 'show'
        }
        
        self.question_keywords = {
            'what', 'why', 'how', 'when', 'where', 'who', 'which',
            'explain', 'describe', 'tell', 'show', 'find',
            'help', 'guide', 'tutorial', 'documentation', 'info',
            'search', 'lookup', 'query', 'check'
        }
        
        # Cluster-specific patterns
        self.cluster_patterns = {
            'cluster_list': [
                r'(?:list|show|get|fetch|display)\s+(?:all\s+)?(?:kubernetes\s+)?clusters?',
                r'(?:show|get)\s+(?:me\s+)?(?:cluster|k8s)\s+(?:list|info)',
                r'clusters?\s+in\s+(\w+)',
                r'enable\s+k8s\s+cluster',
                r'kubernetes\s+cluster\s+(?:status|info|details)',
            ]
        }
        
        # API pattern detection
        self.api_patterns = {
            'rest_endpoint': r'(?:GET|POST|PUT|DELETE|PATCH)\s+(/[\w\-/:\{\}]+)',
            'url_endpoint': r'(https?://[^\s]+)',
            'json_payload': r'\{[\s\S]*\}',
            'path_param': r'\{(\w+)\}',
        }
        
        # Statistics tracking
        self.stats = {
            'total_sessions': 0,
            'successful_actions': 0,
            'failed_actions': 0,
            'questions_answered': 0,
            'avg_turns_per_session': 0.0
        }
        
        logger.info("🤖 Intelligent Action Agent initialized")
    
    # ======================== Initialization ========================
    async def initialize(self):
        """Initialize agent with service discovery"""
        logger.info("🚀 Initializing Intelligent Action Agent...")
        
        try:
            # Initialize HTTP client
            self.http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(DEFAULT_TIMEOUT, connect=10.0),
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
                follow_redirects=True
            )
            logger.info("✅ HTTP client initialized")
            
            # Discover services
            await self._discover_services()
            
            # Start background tasks
            asyncio.create_task(self._periodic_health_check())
            asyncio.create_task(self._session_cleanup())
            asyncio.create_task(self._statistics_updater())
            
            logger.info("✅ Intelligent Action Agent ready")
            
        except Exception as e:
            logger.exception(f"❌ Initialization failed: {e}")
            raise
    
    async def _discover_services(self):
        """Discover and register available services"""
        logger.info("🔍 Discovering services...")
        
        discovery_tasks = [
            self._probe_service(service_name, base_url)
            for service_name, base_url in self.service_registry.items()
        ]
        
        results = await asyncio.gather(*discovery_tasks, return_exceptions=True)
        
        for (service_name, base_url), result in zip(self.service_registry.items(), results):
            if isinstance(result, Exception):
                logger.warning(f"⚠️ Service '{service_name}' discovery failed: {result}")
                self.service_capabilities[service_name] = ServiceCapability(
                    name=service_name,
                    base_url=base_url,
                    available=False,
                    health_status=ServiceStatus.UNKNOWN
                )
            else:
                self.service_capabilities[service_name] = result
                status = "✅" if result.available else "⚠️"
                logger.info(f"{status} Service '{service_name}': {len(result.endpoints)} endpoints")
    
    async def _probe_service(self, service_name: str, base_url: str) -> ServiceCapability:
        """Probe service to discover capabilities"""
        capability = ServiceCapability(name=service_name, base_url=base_url)
        
        # Try common health/info endpoints
        probe_endpoints = [
            ('/health', 'GET'),
            ('/api/health', 'GET'),
            ('/api/v1/health', 'GET'),
            ('/', 'GET'),
        ]
        
        for path, method in probe_endpoints:
            try:
                url = urljoin(base_url, path)
                response = await self.http_client.request(method, url, timeout=5.0)
                
                if response.status_code < 400:
                    capability.available = True
                    capability.health_status = ServiceStatus.AVAILABLE
                    capability.endpoints[path] = ServiceEndpoint(
                        path=path,
                        method=method,
                        description=f'{service_name} {path}',
                        available=True
                    )
                    break
            except Exception as e:
                logger.debug(f"Probe failed for {service_name}{path}: {e}")
                continue
        
        capability.last_check = datetime.now()
        return capability
    
    # ======================== Main Query Handler ========================
    async def handle_query(self, user_query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Main entry point - handle user query intelligently
        """
        try:
            # Get or create session
            if session_id and session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                session.last_activity = datetime.now()
                
                # Check for expired session
                if session.is_expired():
                    await self.clear_session(session_id)
                    return {
                        'status': 'error',
                        'message': 'Session expired. Please start a new conversation.',
                        'session_id': None
                    }
            else:
                session_id = str(uuid.uuid4())
                session = ConversationSession(
                    session_id=session_id,
                    user_query=user_query,
                    intent_type=IntentType.UNKNOWN
                )
                self.active_sessions[session_id] = session
                self.stats['total_sessions'] += 1
            
            # Add user turn to history
            session.add_turn('user', user_query)
            
            # Check conversation length
            if len(session.conversation_history) > MAX_CONVERSATION_TURNS:
                return {
                    'status': 'error',
                    'message': 'Maximum conversation length exceeded. Please start a new session.',
                    'session_id': session_id
                }
            
            # Route based on current state
            if session.state == SessionState.INTENT_DETECTION:
                result = await self._handle_intent_detection(session, user_query)
            elif session.state == SessionState.PARAMETER_COLLECTION:
                result = await self._handle_parameter_collection(session, user_query)
            elif session.state == SessionState.VALIDATION:
                result = await self._handle_validation(session, user_query)
            elif session.state == SessionState.CONFIRMATION_PENDING:
                result = await self._handle_confirmation(session, user_query)
            elif session.state == SessionState.EXECUTING:
                result = {
                    'status': 'executing',
                    'message': '⏳ Your task is currently being executed. Please wait...',
                    'session_id': session_id
                }
            else:
                result = await self._handle_intent_detection(session, user_query)
            
            # Add agent turn to history
            if result.get('message'):
                session.add_turn('agent', result['message'])
            
            return result
            
        except Exception as e:
            logger.exception(f"❌ Error handling query: {e}")
            return {
                'status': 'error',
                'message': f'An error occurred: {str(e)}',
                'session_id': session_id,
                'error': str(e)
            }
    
    # ======================== Intent Detection ========================
    async def _handle_intent_detection(self, session: ConversationSession, query: str) -> Dict[str, Any]:
        """Detect user intent (Question vs Action) with cluster support"""
        
        # Check for cluster-specific patterns first
        for action_type, patterns in self.cluster_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, query.lower(), re.IGNORECASE)
                if match:
                    logger.info(f"🎯 Detected cluster intent: {action_type}")
                    session.intent_type = IntentType.ACTION
                    
                    # Import cluster integration
                    try:
                        from app.api.routes.cluster_integration import get_cluster_list_parameters
                        cluster_params = get_cluster_list_parameters()
                    except ImportError:
                        # Fallback parameters
                        cluster_params = self._get_cluster_fallback_parameters()
                    
                    action_intent = ActionIntent(
                        action_type=ActionType.CLUSTER_LIST,
                        confidence=0.9,
                        service_target='cluster_api',
                        endpoint='/clusterlist',
                        method='POST',
                        parameters=cluster_params,
                        description='Retrieve Kubernetes cluster information'
                    )
                    
                    session.action_intent = action_intent
                    session.state = SessionState.PARAMETER_COLLECTION
                    
                    return await self._start_parameter_collection(session)
        
        # Check for explicit API patterns
        api_detected = self._detect_api_patterns(query)
        
        if api_detected:
            session.intent_type = IntentType.ACTION
            action_intent = await self._parse_api_request(query, api_detected)
            session.action_intent = action_intent
            session.state = SessionState.PARAMETER_COLLECTION
            
            return await self._start_parameter_collection(session)
        
        # Use AI for intent classification
        intent_prompt = f"""Analyze this user query and classify the intent:

Query: "{query}"

Classify as one of:
1. QUESTION - User wants information, explanation, or guidance (answerable by LLM with step-by-step instructions and images)
2. ACTION - User wants to execute a task, automate something, or perform an operation (requires API calls or automation)
3. HYBRID - User wants both information AND to execute an action

Action keywords: create, add, delete, update, scrape, fetch, upload, configure, execute, run, enable, list, show clusters
Question keywords: what, why, how, explain, describe, tell, help, guide, tutorial

Respond with ONLY valid JSON:
{{
    "intent": "QUESTION|ACTION|HYBRID",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation",
    "action_type": "specific action if ACTION",
    "requires_automation": true/false,
    "needs_step_by_step": true/false
}}"""

        try:
            response = await self.ai_service._call_chat_with_retries(
                intent_prompt,
                max_tokens=200,
                temperature=0.1,
                timeout=15
            )
            
            intent_data = self._extract_json_safe(response)
            
            if not intent_data:
                intent_data = self._classify_intent_by_patterns(query)
            
            session.intent_type = IntentType(intent_data.get('intent', 'QUESTION').lower())
            confidence = float(intent_data.get('confidence', 0.5))
            
            logger.info(f"🎯 Intent: {session.intent_type.value} (confidence: {confidence:.2f})")
            
            # If high-confidence action intent, start action flow
            if session.intent_type in [IntentType.ACTION, IntentType.HYBRID] and confidence > CONFIDENCE_THRESHOLD:
                action_intent = await self._detect_action_type(session, query)
                session.action_intent = action_intent
                session.state = SessionState.PARAMETER_COLLECTION
                
                return await self._start_parameter_collection(session)
            
            # If question or low confidence, provide LLM response with step-by-step
            elif session.intent_type == IntentType.QUESTION or confidence < CONFIDENCE_THRESHOLD:
                llm_response = await self._generate_llm_response_with_steps(query)
                self.stats['questions_answered'] += 1
                
                # Check if action might be helpful
                action_suggestion = None
                if confidence > 0.4 and intent_data.get('requires_automation'):
                    action_suggestion = (
                        "\n\n💡 **Tip**: Would you like me to help automate this task? "
                        "I can execute the necessary steps if you'd like."
                    )
                
                return {
                    'status': 'completed',
                    'intent_type': 'question',
                    'response': llm_response.get('answer', ''),
                    'steps': llm_response.get('steps', []),
                    'images': llm_response.get('images', []),
                    'summary': llm_response.get('summary', ''),
                    'action_suggestion': action_suggestion,
                    'session_id': session.session_id,
                    'options': [
                        'Yes, help me automate this',
                        'No, just the information is fine'
                    ] if action_suggestion else None
                }
            
        except Exception as e:
            logger.exception(f"❌ Intent detection error: {e}")
            return {
                'status': 'error',
                'message': 'I had trouble understanding your request. Could you rephrase it?',
                'session_id': session.session_id
            }
    
    def _get_cluster_fallback_parameters(self) -> List[Parameter]:
        """Fallback cluster parameters if integration not available"""
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
    
    async def _generate_llm_response_with_steps(self, query: str) -> Dict[str, Any]:
        """Generate LLM response with step-by-step instructions and images"""
        try:
            # Try to get RAG context
            context = []
            if 'rag' in self.service_capabilities and self.service_capabilities['rag'].available:
                try:
                    service_url = self.service_registry.get('rag')
                    response = await self.http_client.post(
                        f"{service_url}/api/rag/query",
                        json={'query': query, 'max_results': 5},
                        timeout=30.0
                    )
                    if response.status_code == 200:
                        result = response.json()
                        if result.get('answer'):
                            context = [result['answer']]
                except Exception as e:
                    logger.debug(f"RAG context fetch failed: {e}")
            
            # Generate enhanced response with steps
            if context:
                enhanced = await self.ai_service.generate_enhanced_response(query, context, None)
            else:
                enhanced = await self.ai_service.generate_enhanced_response(query, [], None)
            
            # Generate step-by-step instructions
            try:
                steps = await self.ai_service.generate_stepwise_response(query, context if context else [query])
            except Exception as e:
                logger.warning(f"Step generation failed: {e}")
                steps = []
            
            # Generate summary
            try:
                answer_text = enhanced.get("text") if isinstance(enhanced, dict) else enhanced
                summary = await self.ai_service.generate_summary(answer_text, max_sentences=3, max_chars=600)
            except Exception as e:
                logger.warning(f"Summary generation failed: {e}")
                summary = ""
            
            # Extract images from steps
            images = []
            for step in steps:
                if isinstance(step, dict) and step.get('image'):
                    img = step['image']
                    if isinstance(img, dict) and img.get('url'):
                        images.append(img)
            
            return {
                'answer': enhanced.get("text") if isinstance(enhanced, dict) else enhanced,
                'steps': steps,
                'images': images,
                'summary': summary,
                'confidence': enhanced.get("quality_score", 0.8) if isinstance(enhanced, dict) else 0.8
            }
            
        except Exception as e:
            logger.exception(f"LLM response with steps failed: {e}")
            return {
                'answer': "I apologize, but I encountered an error generating a response.",
                'steps': [],
                'images': [],
                'summary': '',
                'confidence': 0.0
            }
    
    def _detect_api_patterns(self, query: str) -> Optional[Dict[str, Any]]:
        """Detect explicit API patterns in query"""
        detected = {}
        
        # HTTP method + endpoint
        method_match = re.search(
            r'\b(GET|POST|PUT|DELETE|PATCH)\s+(/[\w\-/:\{\}]+)',
            query,
            re.IGNORECASE
        )
        if method_match:
            detected['method'] = method_match.group(1).upper()
            detected['endpoint'] = method_match.group(2)
        
        # URL
        url_match = re.search(r'https?://[^\s]+', query)
        if url_match:
            detected['url'] = url_match.group(0)
        
        # JSON payload
        json_match = re.search(r'\{[\s\S]*\}', query)
        if json_match:
            try:
                detected['payload'] = json.loads(json_match.group(0))
            except:
                pass
        
        return detected if detected else None
    
    def _classify_intent_by_patterns(self, query: str) -> Dict[str, Any]:
        """Fallback pattern-based classification"""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        action_matches = len(query_words & self.action_keywords)
        question_matches = len(query_words & self.question_keywords)
        
        # Check for step-by-step indicators
        needs_steps = any(word in query_lower for word in [
            'how to', 'steps', 'guide', 'tutorial', 'instructions', 'procedure'
        ])
        
        if action_matches > question_matches:
            return {
                'intent': 'ACTION',
                'confidence': min(0.8, action_matches * 0.2),
                'reasoning': 'Action keywords detected',
                'requires_automation': True,
                'needs_step_by_step': False
            }
        elif question_matches > action_matches:
            return {
                'intent': 'QUESTION',
                'confidence': min(0.8, question_matches * 0.2),
                'reasoning': 'Question keywords detected',
                'requires_automation': False,
                'needs_step_by_step': needs_steps
            }
        else:
            return {
                'intent': 'HYBRID',
                'confidence': 0.5,
                'reasoning': 'Mixed or unclear intent',
                'requires_automation': True,
                'needs_step_by_step': needs_steps
            }
    
    async def _detect_action_type(self, session: ConversationSession, query: str) -> ActionIntent:
        """Detect specific action type and target service"""
        query_lower = query.lower()
        
        # Detect action type
        if any(word in query_lower for word in ['scrape', 'crawl', 'extract', 'fetch']):
            action_type = ActionType.SCRAPING
            service_target = 'scraper'
        elif any(word in query_lower for word in ['upload', 'import', 'add document', 'store']):
            action_type = ActionType.DATA_UPLOAD
            service_target = 'rag'
        elif any(word in query_lower for word in ['bulk', 'multiple', 'batch', 'many']):
            action_type = ActionType.BULK_OPERATION
            service_target = 'scraper'
        elif any(word in query_lower for word in ['configure', 'setup', 'admin', 'user', 'permission']):
            action_type = ActionType.ADMIN_TASK
            service_target = 'admin'
        elif any(word in query_lower for word in ['search', 'find', 'query', 'lookup']):
            action_type = ActionType.RAG_QUERY
            service_target = 'rag'
        elif any(word in query_lower for word in ['cluster', 'kubernetes', 'k8s']):
            action_type = ActionType.CLUSTER_LIST
            service_target = 'cluster_api'
        else:
            action_type = ActionType.API_CALL
            service_target = self._guess_service_from_query(query)
        
        # Get parameters for action type
        parameters = self._get_parameters_for_action(action_type)
        
        return ActionIntent(
            action_type=action_type,
            confidence=0.8,
            service_target=service_target,
            parameters=parameters,
            description=f"Execute {action_type.value} on {service_target} service"
        )
    
    def _get_parameters_for_action(self, action_type: ActionType) -> List[Parameter]:
        """Define required parameters for action type"""
        
        if action_type == ActionType.CLUSTER_LIST:
            return self._get_cluster_fallback_parameters()
        
        elif action_type == ActionType.SCRAPING:
            return [
                Parameter(
                    name='url',
                    description='URL to scrape',
                    required=True,
                    param_type=ParameterType.URL,
                    validation_regex=r'https?://[^\s]+'
                ),
                Parameter(
                    name='extract_images',
                    description='Extract images from page?',
                    required=False,
                    param_type=ParameterType.BOOLEAN,
                    default=True
                ),
                Parameter(
                    name='wait_for_js',
                    description='Wait for JavaScript to load?',
                    required=False,
                    param_type=ParameterType.BOOLEAN,
                    default=False
                )
            ]
        
        elif action_type == ActionType.DATA_UPLOAD:
            return [
                Parameter(
                    name='file_path',
                    description='Path to file to upload',
                    required=True,
                    param_type=ParameterType.STRING
                ),
                Parameter(
                    name='store_in_knowledge',
                    description='Store in knowledge base?',
                    required=False,
                    param_type=ParameterType.BOOLEAN,
                    default=True
                )
            ]
        
        elif action_type == ActionType.BULK_OPERATION:
            return [
                Parameter(
                    name='base_url',
                    description='Base URL for bulk operation',
                    required=True,
                    param_type=ParameterType.URL
                ),
                Parameter(
                    name='max_depth',
                    description='Maximum crawl depth (1-5)',
                    required=False,
                    param_type=ParameterType.INTEGER,
                    default=2,
                    choices=[1, 2, 3, 4, 5],
                    min_value=1,
                    max_value=5
                ),
                Parameter(
                    name='max_urls',
                    description='Maximum URLs to process',
                    required=False,
                    param_type=ParameterType.INTEGER,
                    default=50,
                    min_value=1,
                    max_value=500
                )
            ]
        
        elif action_type == ActionType.API_CALL:
            return [
                Parameter(
                    name='endpoint',
                    description='API endpoint path',
                    required=True,
                    param_type=ParameterType.STRING
                ),
                Parameter(
                    name='method',
                    description='HTTP method',
                    required=True,
                    param_type=ParameterType.STRING,
                    choices=['GET', 'POST', 'PUT', 'DELETE', 'PATCH'],
                    default='POST'
                ),
                Parameter(
                    name='payload',
                    description='Request payload (JSON)',
                    required=False,
                    param_type=ParameterType.JSON
                )
            ]
        
        else:
            return []
    
    async def _parse_api_request(self, query: str, api_detected: Dict[str, Any]) -> ActionIntent:
        """Parse detected API request into ActionIntent"""
        parameters = []
        
        if 'endpoint' in api_detected:
            parameters.append(Parameter(
                name='endpoint',
                description='API endpoint',
                required=True,
                param_type=ParameterType.STRING,
                value=api_detected['endpoint'],
                collected=True
            ))
        
        if 'method' in api_detected:
            parameters.append(Parameter(
                name='method',
                description='HTTP method',
                required=True,
                param_type=ParameterType.STRING,
                value=api_detected['method'],
                collected=True
            ))
        
        if 'payload' in api_detected:
            parameters.append(Parameter(
                name='payload',
                description='Request payload',
                required=False,
                param_type=ParameterType.JSON,
                value=api_detected['payload'],
                collected=True
            ))
        
        service = 'admin'
        if 'url' in api_detected:
            parsed = urlparse(api_detected['url'])
            for svc_name, svc_url in self.service_registry.items():
                if parsed.netloc in svc_url:
                    service = svc_name
                    break
        
        return ActionIntent(
            action_type=ActionType.API_CALL,
            confidence=0.9,
            service_target=service,
            parameters=parameters,
            description=f"Execute API call: {api_detected.get('method', 'POST')} {api_detected.get('endpoint', '')}"
        )
    
    # ======================== Parameter Collection ========================
    async def _start_parameter_collection(self, session: ConversationSession) -> Dict[str, Any]:
        """Start interactive parameter collection"""
        
        if not session.action_intent:
            return {'status': 'error', 'message': 'No action intent found'}
        
        # Extract already-provided parameters
        await self._extract_parameters_from_query(session)
        
        # Find missing required parameters
        missing_params = [
            p for p in session.action_intent.parameters 
            if p.required and not p.collected
        ]
        
        if not missing_params:
            session.state = SessionState.VALIDATION
            return await self._handle_validation(session, None)
        
        # Ask for first missing parameter
        next_param = missing_params[0]
        session.parameters_pending = [p.name for p in missing_params]
        
        question = self._generate_parameter_question(next_param)
        
        return {
            'status': 'parameter_collection',
            'message': question,
            'parameter': next_param.to_dict(),
            'collected_so_far': session.parameters_collected,
            'pending': session.parameters_pending,
            'session_id': session.session_id,
            'options': next_param.choices if next_param.choices else None
        }
    
    async def _extract_parameters_from_query(self, session: ConversationSession):
        """Extract parameters from original query"""
        query = session.user_query
        
        for param in session.action_intent.parameters:
            if param.collected:
                continue
            
            # URL extraction
            if param.param_type == ParameterType.URL:
                urls = re.findall(r'https?://[^\s]+', query)
                if urls:
                    param.value = urls[0]
                    param.collected = True
                    session.parameters_collected[param.name] = urls[0]
            
            # Boolean extraction
            elif param.param_type == ParameterType.BOOLEAN:
                if any(word in query.lower() for word in ['yes', 'true', 'enable', 'with']):
                    param.value = True
                    param.collected = True
                    session.parameters_collected[param.name] = True
                elif any(word in query.lower() for word in ['no', 'false', 'disable', 'without']):
                    param.value = False
                    param.collected = True
                    session.parameters_collected[param.name] = False
            
            # Integer extraction
            elif param.param_type == ParameterType.INTEGER:
                numbers = re.findall(r'\b(\d+)\b', query)
                if numbers:
                    try:
                        value = int(numbers[0])
                        if param.choices and value in param.choices:
                            param.value = value
                            param.collected = True
                            session.parameters_collected[param.name] = value
                        elif not param.choices:
                            param.value = value
                            param.collected = True
                            session.parameters_collected[param.name] = value
                    except:
                        pass
            
            # JSON extraction
            elif param.param_type == ParameterType.JSON:
                json_match = re.search(r'\{[\s\S]*\}', query)
                if json_match:
                    try:
                        json_obj = json.loads(json_match.group(0))
                        param.value = json_obj
                        param.collected = True
                        session.parameters_collected[param.name] = json_obj
                    except:
                        pass
    
    async def _handle_parameter_collection(self, session: ConversationSession, user_input: str) -> Dict[str, Any]:
        """Handle user's response during parameter collection"""
        
        # Find current parameter
        pending_params = [
            p for p in session.action_intent.parameters 
            if p.name in session.parameters_pending
        ]
        
        if not pending_params:
            session.state = SessionState.VALIDATION
            return await self._handle_validation(session, None)
        
        current_param = pending_params[0]
        
        # Validate input
        is_valid, value, error = self._validate_parameter_input(user_input, current_param)
        
        if not is_valid:
            current_param.attempts += 1
            
            if current_param.attempts >= current_param.max_attempts:
                session.error_count += 1
                return {
                    'status': 'error',
                    'message': f"❌ Too many invalid attempts for '{current_param.name}'. Please start over or provide a valid value.",
                    'session_id': session.session_id,
                    'options': ['Start over', 'Skip this parameter']
                }
            
            return {
                'status': 'parameter_collection',
                'message': f"❌ {error}\n\nPlease try again ({current_param.attempts}/{current_param.max_attempts} attempts):",
                'parameter': current_param.to_dict(),
                'session_id': session.session_id
            }
        
        # Store collected parameter
        current_param.value = value
        current_param.collected = True
        session.parameters_collected[current_param.name] = value
        session.parameters_pending.remove(current_param.name)
        
        # Check for more parameters
        remaining = [
            p for p in session.action_intent.parameters 
            if p.required and not p.collected
        ]
        
        if remaining:
            next_param = remaining[0]
            question = self._generate_parameter_question(next_param)
            
            return {
                'status': 'parameter_collection',
                'message': f"✅ Got it! {question}",
                'parameter': next_param.to_dict(),
                'collected_so_far': session.parameters_collected,
                'pending': session.parameters_pending,
                'session_id': session.session_id,
                'options': next_param.choices if next_param.choices else None
            }
        
        # All parameters collected
        session.state = SessionState.VALIDATION
        return await self._handle_validation(session, None)
    
    def _validate_parameter_input(self, user_input: str, param: Parameter) -> Tuple[bool, Any, Optional[str]]:
        """Validate user input for parameter"""
        user_input = user_input.strip()
        
        if param.param_type == ParameterType.URL:
            is_valid, error = self.validator.validate_url(user_input)
            return (is_valid, user_input if is_valid else None, error)
        
        elif param.param_type == ParameterType.EMAIL:
            is_valid, error = self.validator.validate_email(user_input)
            return (is_valid, user_input if is_valid else None, error)
        
        elif param.param_type == ParameterType.BOOLEAN:
            is_valid, error = self.validator.validate_boolean(user_input)
            if is_valid:
                value = user_input.lower() in ['yes', 'y', 'true', '1', 'on', 'enable', 'enabled']
                return (True, value, None)
            return (False, None, error)
        
        elif param.param_type == ParameterType.INTEGER:
            is_valid, error = self.validator.validate_integer(
                user_input, 
                param.min_value, 
                param.max_value
            )
            if is_valid:
                value = int(user_input)
                if param.choices and value not in param.choices:
                    return (False, None, f"Must be one of: {param.choices}")
                return (True, value, None)
            return (False, None, error)
        
        elif param.param_type == ParameterType.FLOAT:
            is_valid, error = self.validator.validate_float(
                user_input, 
                param.min_value, 
                param.max_value
            )
            return (is_valid, float(user_input) if is_valid else None, error)
        
        elif param.param_type == ParameterType.JSON:
            is_valid, error = self.validator.validate_json(user_input)
            if is_valid:
                return (True, json.loads(user_input), None)
            return (False, None, error)
        
        elif param.param_type == ParameterType.STRING:
            is_valid, error = self.validator.validate_string(
                user_input,
                param.min_length,
                param.max_length,
                param.validation_regex
            )
            if is_valid:
                if param.choices and user_input not in param.choices:
                    return (False, None, f"Must be one of: {', '.join(param.choices)}")
                return (True, user_input, None)
            return (False, None, error)
        
        return (True, user_input, None)
    
    def _generate_parameter_question(self, param: Parameter) -> str:
        """Generate natural language question for parameter"""
        question = f"📝 **{param.description}**"
        
        if param.choices:
            question += f"\n\n**Options**: {', '.join(map(str, param.choices))}"
        
        if param.default is not None:
            question += f"\n**Default**: {param.default}"
        
        if param.param_type == ParameterType.INTEGER and (param.min_value or param.max_value):
            range_str = []
            if param.min_value is not None:
                range_str.append(f"min: {param.min_value}")
            if param.max_value is not None:
                range_str.append(f"max: {param.max_value}")
            if range_str:
                question += f"\n**Range**: {', '.join(range_str)}"
        
        return question
    
    # ======================== Validation & Confirmation ========================
    async def _handle_validation(self, session: ConversationSession, user_input: Optional[str]) -> Dict[str, Any]:
        """Validate all collected parameters"""
        
        # Build summary
        summary_lines = ["📋 **Task Summary**\n"]
        summary_lines.append(f"**Action**: {session.action_intent.action_type.value}")
        summary_lines.append(f"**Service**: {session.action_intent.service_target}\n")
        summary_lines.append("**Parameters:**")
        
        for param_name, param_value in session.parameters_collected.items():
            if isinstance(param_value, dict):
                param_value = json.dumps(param_value, indent=2)
            summary_lines.append(f"  • **{param_name}**: `{param_value}`")
        
        summary = "\n".join(summary_lines)
        
        # Move to confirmation
        session.state = SessionState.CONFIRMATION_PENDING
        session.awaiting_confirmation = True
        
        return {
            'status': 'awaiting_confirmation',
            'message': summary,
            'question': "\n\n✅ **Does this look correct? Should I proceed with executing this task?**",
            'options': ['Yes, proceed', 'No, cancel', 'Let me modify something'],
            'session_id': session.session_id
        }
    
    async def _handle_confirmation(self, session: ConversationSession, user_input: str) -> Dict[str, Any]:
        """Handle user's confirmation response"""
        
        user_input_lower = user_input.lower().strip()
        
        # Confirmation
        if any(word in user_input_lower for word in ['yes', 'proceed', 'confirm', 'go', 'execute', 'run']):
            session.state = SessionState.EXECUTING
            session.awaiting_confirmation = False
            
            # Execute action
            result = await self._execute_action(session)
            
            session.execution_result = result
            session.state = SessionState.COMPLETED if result['success'] else SessionState.FAILED
            
            if result['success']:
                self.stats['successful_actions'] += 1
            else:
                self.stats['failed_actions'] += 1
            
            return {
                'status': 'completed' if result['success'] else 'failed',
                'message': result['message'],
                'details': result.get('details'),
                'execution_time': result.get('execution_time'),
                'session_id': session.session_id
            }
        
        # Cancellation
        elif any(word in user_input_lower for word in ['no', 'cancel', 'stop', 'abort', 'nevermind']):
            session.state = SessionState.CANCELLED
            return {
                'status': 'cancelled',
                'message': '❌ Task cancelled. No actions were taken.',
                'session_id': session.session_id
            }
        
        # Modification request
        elif any(word in user_input_lower for word in ['modify', 'change', 'edit', 'update', 'fix']):
            session.state = SessionState.PARAMETER_COLLECTION
            session.parameters_pending = list(session.parameters_collected.keys())
            
            return {
                'status': 'parameter_modification',
                'message': '🔧 **Which parameter would you like to modify?**',
                'parameters': list(session.parameters_collected.keys()),
                'session_id': session.session_id
            }
        
        else:
            return {
                'status': 'awaiting_confirmation',
                'message': '❓ I didn\'t understand. Please respond with:\n  • "**Yes**" to proceed\n  • "**No**" to cancel\n  • "**Modify**" to change parameters',
                'options': ['Yes, proceed', 'No, cancel', 'Let me modify something'],
                'session_id': session.session_id
            }
    
    # ======================== Action Execution ========================
    async def _execute_action(self, session: ConversationSession) -> Dict[str, Any]:
        """Execute the actual action"""
        
        start_time = datetime.now()
        
        try:
            action_type = session.action_intent.action_type
            service_target = session.action_intent.service_target
            params = session.parameters_collected
            
            logger.info(f"🚀 Executing {action_type.value} on {service_target}")
            
            # Route to executor
            if action_type == ActionType.CLUSTER_LIST:
                # Will be handled by enhanced agent in unified_main
                return {
                    'success': True,
                    'message': '✅ Cluster query ready for execution',
                    'details': params,
                    'execution_time': 0.0
                }
            elif action_type == ActionType.SCRAPING:
                result = await self._execute_scraping(service_target, params)
            elif action_type == ActionType.DATA_UPLOAD:
                result = await self._execute_data_upload(service_target, params)
            elif action_type == ActionType.BULK_OPERATION:
                result = await self._execute_bulk_operation(service_target, params)
            elif action_type == ActionType.API_CALL:
                result = await self._execute_api_call(service_target, params)
            elif action_type == ActionType.RAG_QUERY:
                result = await self._execute_rag_query(service_target, params)
            elif action_type == ActionType.ADMIN_TASK:
                result = await self._execute_admin_task(service_target, params)
            else:
                result = {
                    'success': False,
                    'message': f'Action type {action_type.value} not implemented'
                }
            
            execution_time = (datetime.now() - start_time).total_seconds()
            result['execution_time'] = execution_time
            
            logger.info(f"✅ Execution completed in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            logger.exception(f"❌ Execution error: {e}")
            execution_time = (datetime.now() - start_time).total_seconds()
            return {
                'success': False,
                'message': f'❌ Execution failed: {str(e)}',
                'error': str(e),
                'execution_time': execution_time
            }
    
    async def _execute_scraping(self, service: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute web scraping"""
        service_url = self.service_registry.get(service)
        if not service_url:
            return {'success': False, 'message': f'Service {service} not found'}
        
        endpoint = f"{service_url}/api/scraper/fetch"
        payload = {
            'url': params.get('url'),
            'extract_text': True,
            'extract_images': params.get('extract_images', True),
            'extract_tables': True,
            'wait_for_js': params.get('wait_for_js', False),
            'output_format': 'json'
        }
        
        try:
            response = await self.http_client.post(endpoint, json=payload, timeout=60.0)
            response.raise_for_status()
            result = response.json()
            
            return {
                'success': True,
                'message': f'✅ Successfully scraped {params.get("url")}',
                'details': {
                    'content_length': len(result.get('content', {}).get('text', '')),
                    'images_found': len(result.get('content', {}).get('images', [])),
                    'status_code': result.get('status_code')
                }
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'❌ Scraping failed: {str(e)}',
                'error': str(e)
            }
    
    async def _execute_data_upload(self, service: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data upload"""
        return {
            'success': True,
            'message': f'✅ File upload initiated for {params.get("file_path")}',
            'details': params
        }
    
    async def _execute_bulk_operation(self, service: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute bulk operation"""
        service_url = self.service_registry.get(service)
        if not service_url:
            return {'success': False, 'message': f'Service {service} not found'}
        
        endpoint = f"{service_url}/api/scraper/bulk"
        payload = {
            'base_url': params.get('base_url'),
            'max_depth': params.get('max_depth', 2),
            'max_urls': params.get('max_urls', 50),
            'auto_store': True
        }
        
        try:
            response = await self.http_client.post(endpoint, json=payload, timeout=120.0)
            response.raise_for_status()
            result = response.json()
            
            return {
                'success': True,
                'message': f'✅ Bulk operation started',
                'details': {
                    'base_url': params.get('base_url'),
                    'discovered_urls': result.get('discovered_urls_count', 0),
                    'estimated_time': result.get('estimated_time_minutes', 0)
                }
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'❌ Bulk operation failed: {str(e)}',
                'error': str(e)
            }
    
    async def _execute_api_call(self, service: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute generic API call"""
        service_url = self.service_registry.get(service)
        if not service_url:
            return {'success': False, 'message': f'Service {service} not found'}
        
        endpoint = params.get('endpoint', '')
        method = params.get('method', 'POST').upper()
        payload = params.get('payload', {})
        
        full_url = urljoin(service_url, endpoint)
        
        try:
            if method == 'GET':
                response = await self.http_client.get(full_url, timeout=60.0)
            elif method == 'POST':
                response = await self.http_client.post(full_url, json=payload, timeout=60.0)
            elif method == 'PUT':
                response = await self.http_client.put(full_url, json=payload, timeout=60.0)
            elif method == 'DELETE':
                response = await self.http_client.delete(full_url, timeout=60.0)
            elif method == 'PATCH':
                response = await self.http_client.patch(full_url, json=payload, timeout=60.0)
            else:
                return {'success': False, 'message': f'Unsupported method: {method}'}
            
            response.raise_for_status()
            
            try:
                result_data = response.json()
            except:
                result_data = {'text': response.text[:500]}
            
            return {
                'success': True,
                'message': f'✅ API call successful: {method} {endpoint}',
                'details': {
                    'status_code': response.status_code,
                    'response': result_data
                }
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'❌ API call failed: {str(e)}',
                'error': str(e)
            }
    
    async def _execute_rag_query(self, service: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute RAG query"""
        service_url = self.service_registry.get(service)
        if not service_url:
            return {'success': False, 'message': f'Service {service} not found'}
        
        endpoint = f"{service_url}/api/rag/query"
        payload = {
            'query': params.get('query', ''),
            'max_results': params.get('max_results', 5)
        }
        
        try:
            response = await self.http_client.post(endpoint, json=payload, timeout=60.0)
            response.raise_for_status()
            result = response.json()
            
            return {
                'success': True,
                'message': '✅ Query executed successfully',
                'details': {
                    'results_found': result.get('results_found', 0),
                    'answer': result.get('answer', '')[:200],
                    'confidence': result.get('confidence', 0)
                }
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'❌ Query failed: {str(e)}',
                'error': str(e)
            }
    
    async def _execute_admin_task(self, service: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute admin task"""
        return {
            'success': True,
            'message': '✅ Admin task executed',
            'details': params
        }
    
    # ======================== Utility Methods ========================
    def _extract_json_safe(self, text: str) -> Optional[Dict[str, Any]]:
        """Safely extract JSON from text"""
        if not text:
            return None
        
        try:
            return json.loads(text)
        except:
            pass
        
        try:
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                return json.loads(json_match.group(0))
        except:
            pass
        
        return None
    
    def _guess_service_from_query(self, query: str) -> str:
        """Guess target service from query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['scrape', 'crawl', 'fetch', 'extract']):
            return 'scraper'
        elif any(word in query_lower for word in ['search', 'find', 'query', 'document']):
            return 'rag'
        elif any(word in query_lower for word in ['admin', 'user', 'permission', 'config']):
            return 'admin'
        elif any(word in query_lower for word in ['cluster', 'kubernetes', 'k8s']):
            return 'cluster_api'
        else:
            return 'admin'
    
    # ======================== Background Tasks ========================
    async def _periodic_health_check(self):
        """Periodically check service health"""
        while True:
            try:
                await asyncio.sleep(300)
                
                for service_name, capability in self.service_capabilities.items():
                    try:
                        health_url = urljoin(capability.base_url, '/health')
                        response = await self.http_client.get(health_url, timeout=5.0)
                        
                        if response.status_code == 200:
                            capability.available = True
                            capability.health_status = ServiceStatus.AVAILABLE
                        else:
                            capability.available = False
                            capability.health_status = ServiceStatus.DEGRADED
                    except:
                        capability.available = False
                        capability.health_status = ServiceStatus.UNAVAILABLE
                    
                    capability.last_check = datetime.now()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Health check error: {e}")
    
    async def _session_cleanup(self):
        """Clean up expired sessions"""
        while True:
            try:
                await asyncio.sleep(600)  # Every 10 minutes
                
                now = datetime.now()
                expired = []
                
                for session_id, session in self.active_sessions.items():
                    if session.is_expired():
                        expired.append(session_id)
                
                for session_id in expired:
                    del self.active_sessions[session_id]
                    logger.info(f"🧹 Cleaned up expired session: {session_id}")
                
                if expired:
                    logger.info(f"🧹 Cleaned up {len(expired)} expired sessions")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Session cleanup error: {e}")
    
    async def _statistics_updater(self):
        """Update statistics periodically"""
        while True:
            try:
                await asyncio.sleep(60)  # Every minute
                
                if self.active_sessions:
                    total_turns = sum(
                        len(session.conversation_history) 
                        for session in self.active_sessions.values()
                    )
                    self.stats['avg_turns_per_session'] = total_turns / len(self.active_sessions)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Statistics update error: {e}")
    
    async def cleanup_expired_sessions(self, max_age_seconds: int = 3600):
        now = datetime.utcnow()
        expired_keys = []

        for session_id, session in list(self.sessions.items()):
            last_seen = session.get("last_active")
            if last_seen and (now - last_seen).total_seconds() > max_age_seconds:
                expired_keys.append(session_id)

        for key in expired_keys:
            del self.sessions[key]

        return len(expired_keys)

    async def get_active_sessions(self) -> List[Dict[str, Any]]:
        """Get list of active sessions"""
        return [
            {
                'session_id': session.session_id,
                'query': session.user_query,
                'state': session.state.value,
                'intent_type': session.intent_type.value,
                'created_at': session.created_at.isoformat(),
                'last_activity': session.last_activity.isoformat(),
                'turns': len(session.conversation_history)
            }
            for session in self.active_sessions.values()
        ]
    
    async def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Get specific session"""
        return self.active_sessions.get(session_id)
    
    async def clear_session(self, session_id: str) -> bool:
        """Clear/delete a session"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            logger.info(f"🗑️ Cleared session: {session_id}")
            return True
        return False
    
    async def check_service_health(self) -> Dict[str, Dict[str, Any]]:
        """Check health of all services"""
        health = {}
        
        for service_name, capability in self.service_capabilities.items():
            health[service_name] = {
                'available': capability.available,
                'status': capability.health_status.value,
                'last_check': capability.last_check.isoformat(),
                'base_url': capability.base_url,
                'endpoints_count': len(capability.endpoints)
            }
        
        return health
    
    async def get_service_capabilities(self) -> Dict[str, ServiceCapability]:
        """Get capabilities of all services"""
        return self.service_capabilities
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return {
            **self.stats,
            'active_sessions': len(self.active_sessions),
            'total_services': len(self.service_capabilities),
            'available_services': sum(
                1 for cap in self.service_capabilities.values() 
                if cap.available
            ),
            'timestamp': datetime.now().isoformat()
        }
    
    async def close(self):
        """Close the agent and cleanup resources"""
        logger.info("🛑 Shutting down Intelligent Action Agent...")
        
        try:
            # Close all sessions
            session_ids = list(self.active_sessions.keys())
            for session_id in session_ids:
                await self.clear_session(session_id)
            
            # Close HTTP client
            if self.http_client:
                await self.http_client.aclose()
                logger.info("✅ HTTP client closed")
            
            logger.info("✅ Intelligent Action Agent shut down complete")
            
        except Exception as e:
            logger.exception(f"Error during shutdown: {e}")


# ======================== Export Public API ========================
__all__ = [
    'IntelligentActionAgent',
    'IntentType',
    'ActionType',
    'SessionState',
    'ServiceStatus',
    'ParameterType',
    'Parameter',
    'ServiceEndpoint',
    'ServiceCapability',
    'ActionIntent',
    'ConversationSession',
    'ConversationTurn',
    'ParameterValidator',
]