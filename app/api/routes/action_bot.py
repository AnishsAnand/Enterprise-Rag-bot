import asyncio
import json
import logging
import re
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Set, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field
from urllib.parse import urlparse, urljoin
import httpx
from collections import defaultdict
import time
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
    routing_info: Dict[str, Any] = field(default_factory=dict)

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

# ======================== Enhanced Cluster Intelligence ========================
class ClusterIntelligence:
    """Advanced cluster query intelligence with NLP-like understanding"""
    
    def __init__(self):
        # Enhanced location mapping with aliases and endpoint IDs
        self.location_mapping = {
            'mumbai': {
                'id': 11, 
                'name': 'Mumbai-BKC', 
                'code': 'EP_V2_MUM_BKC',
                'aliases': ['mumbai', 'mum', 'bombay', 'bkc', 'mumbai-bkc'],
                'region': 'India'
            },
            'delhi': {
                'id': 12, 
                'name': 'Delhi', 
                'code': 'EP_V2_DEL',
                'aliases': ['delhi', 'del', 'ncr', 'new delhi', 'new-delhi'],
                'region': 'India'
            },
            'bengaluru': {
                'id': 14, 
                'name': 'Bengaluru', 
                'code': 'EP_V2_BL',
                'aliases': ['bengaluru', 'bangalore', 'blr', 'bl', 'bang'],
                'region': 'India'
            },
            'chennai': {
                'id': 162, 
                'name': 'Chennai-AMB', 
                'code': 'EP_V2_CHN_AMB',
                'aliases': ['chennai', 'chn', 'madras', 'amb', 'chennai-amb'],
                'region': 'India'
            },
            'cressex': {
                'id': 204, 
                'name': 'Cressex', 
                'code': 'EP_V2_UKCX',
                'aliases': ['cressex', 'uk', 'ukcx', 'united kingdom', 'london'],
                'region': 'UK'
            },
        }
        
        self.all_endpoint_ids = [11, 12, 14, 162, 204]
        self.supported_k8s_versions = ['1.24', '1.25', '1.26', '1.27', '1.28', '1.29', '1.30']
        
        # Enhanced pattern recognition
        self.cluster_patterns = {
            'list_all': [
                r'(?:list|show|get|display|fetch|find|give\s+me|tell\s+me\s+about)\s+(?:all\s+)?(?:the\s+)?(?:kubernetes\s+|k8s\s+)?clusters?',
                r'(?:what|which)\s+clusters?\s+(?:do\s+)?(?:we\s+|i\s+)?(?:have|exist)',
                r'clusters?\s+(?:overview|summary|status|list)',
                r'show\s+(?:me\s+)?(?:my\s+)?clusters?',
                r'all\s+clusters?',
            ],
            'location_specific': [
                r'clusters?\s+(?:in|at|from|on|for|located\s+in)\s+(\w+(?:[- ]\w+)*)',
                r'(\w+(?:[- ]\w+)*)\s+(?:location\s+)?clusters?',
                r'(?:get|show|list|fetch)\s+(\w+(?:[- ]\w+)*)\s+clusters?',
            ],
            'status_filter': [
                r'(?:show|list|get|find)\s+(?:only\s+)?(?:the\s+)?(healthy|unhealthy|draft)\s+clusters?',
                r'clusters?\s+(?:that\s+are\s+|which\s+are\s+|with\s+status\s+)?(healthy|unhealthy|draft)',
                r'(healthy|unhealthy|draft)\s+clusters?',
                r'clusters?\s+(?:in\s+)?(healthy|unhealthy|draft)\s+(?:state|status)',
            ],
            'type_filter': [
                r'(?:show|list|get|find)\s+(?:only\s+)?(?:the\s+)?(mgmt|management|app|application)\s+clusters?',
                r'(mgmt|management|app|application)\s+(?:type\s+)?clusters?',
                r'clusters?\s+(?:of\s+)?(?:type\s+)?(mgmt|management|app|application)',
            ],
            'version_query': [
                r'clusters?\s+(?:running|with|on|using)\s+(?:kubernetes\s+|k8s\s+)?(?:version\s+)?v?(\d+\.\d+(?:\.\d+)?)',
                r'(?:version|ver|v)\s+(\d+\.\d+(?:\.\d+)?)\s+clusters?',
                r'k8s\s+(?:version\s+)?(\d+\.\d+(?:\.\d+)?)',
                r'kubernetes\s+(\d+\.\d+(?:\.\d+)?)',
            ],
            'count_query': [
                r'how\s+many\s+clusters?',
                r'total\s+(?:number\s+of\s+)?clusters?',
                r'count\s+(?:of\s+)?clusters?',
                r'number\s+of\s+clusters?',
            ],
            'detailed_query': [
                r'(?:detailed|full|complete|verbose|comprehensive)\s+(?:info|information|details|view)',
                r'show\s+(?:me\s+)?(?:all\s+)?(?:the\s+)?details?',
                r'give\s+(?:me\s+)?(?:complete|full)\s+(?:info|information)',
            ],
            'region_query': [
                r'clusters?\s+(?:in|from)\s+(india|uk|united\s+kingdom)',
                r'(india|uk|united\s+kingdom)\s+clusters?',
            ]
        }
        
        # Status synonyms with expanded variations
        self.status_synonyms = {
            'healthy': ['healthy', 'good', 'ok', 'running', 'active', 'up', 'working', 'operational', 'online'],
            'unhealthy': ['unhealthy', 'bad', 'down', 'failed', 'error', 'problem', 'issue', 'broken', 'offline', 'degraded'],
            'draft': ['draft', 'pending', 'incomplete', 'unfinished', 'in-progress', 'setup']
        }
        
        # Type synonyms
        self.type_synonyms = {
            'MGMT': ['mgmt', 'management', 'admin', 'control', 'master'],
            'APP': ['app', 'application', 'workload', 'service', 'worker']
        }
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Perform deep analysis of cluster query
        Returns comprehensive intent and parameters
        """
        query_lower = query.lower().strip()
        analysis = {
            'intent': 'cluster_list',
            'confidence': 0.0,
            'endpoint_ids': None,
            'filter_status': None,
            'filter_type': None,
            'filter_k8s_version': None,
            'show_details': False,
            'needs_confirmation': False,
            'auto_executable': False,
            'reasoning': [],
            'query_type': 'standard'
        }
        
        # Check for "list all" intent
        for pattern in self.cluster_patterns['list_all']:
            if re.search(pattern, query_lower):
                analysis['endpoint_ids'] = self.all_endpoint_ids
                analysis['confidence'] = 0.95
                analysis['auto_executable'] = True
                analysis['reasoning'].append('Detected "list all clusters" intent')
                break
        
        # Extract locations with improved matching
        locations_found = self._extract_locations(query_lower)
        if locations_found:
            analysis['endpoint_ids'] = locations_found
            analysis['confidence'] = max(analysis['confidence'], 0.9)
            analysis['auto_executable'] = True
            location_names = [self._get_location_name(loc_id) for loc_id in locations_found]
            analysis['reasoning'].append(f'Detected locations: {", ".join(location_names)}')
        
        # Extract region-based query
        region_endpoints = self._extract_region(query_lower)
        if region_endpoints:
            analysis['endpoint_ids'] = region_endpoints
            analysis['confidence'] = max(analysis['confidence'], 0.88)
            analysis['auto_executable'] = True
            analysis['reasoning'].append(f'Detected region-based query')
        
        # Extract status filter
        status = self._extract_status(query_lower)
        if status:
            analysis['filter_status'] = status
            analysis['confidence'] = max(analysis['confidence'], 0.85)
            analysis['reasoning'].append(f'Detected status filter: {status}')
        
        # Extract type filter
        cluster_type = self._extract_type(query_lower)
        if cluster_type:
            analysis['filter_type'] = cluster_type
            analysis['confidence'] = max(analysis['confidence'], 0.85)
            analysis['reasoning'].append(f'Detected type filter: {cluster_type}')
        
        # Extract K8s version with validation
        version = self._extract_version(query_lower)
        if version:
            # Validate version exists
            if version in self.supported_k8s_versions:
                analysis['filter_k8s_version'] = version
                analysis['confidence'] = max(analysis['confidence'], 0.85)
                analysis['reasoning'].append(f'Detected K8s version: {version}')
            else:
                # Version not supported
                analysis['filter_k8s_version'] = version
                analysis['confidence'] = max(analysis['confidence'], 0.60)
                analysis['query_type'] = 'unsupported_version'
                analysis['reasoning'].append(f'Unsupported K8s version requested: {version}')
        
        # Check for count query
        if self._is_count_query(query_lower):
            analysis['query_type'] = 'count'
            analysis['reasoning'].append('Count query detected')
        
        # Check for detailed output request
        if self._wants_details(query_lower):
            analysis['show_details'] = True
            analysis['reasoning'].append('Detailed output requested')
        
        # Determine if auto-executable
        if analysis['endpoint_ids'] or any([
            analysis['filter_status'],
            analysis['filter_type'],
            analysis['filter_k8s_version']
        ]):
            analysis['auto_executable'] = True
        
        # Set default endpoint_ids if not specified but filters are
        if not analysis['endpoint_ids'] and analysis['auto_executable']:
            analysis['endpoint_ids'] = self.all_endpoint_ids
            analysis['reasoning'].append('Defaulting to all locations')
        
        # Determine if confirmation needed (only for ambiguous or potentially destructive)
        analysis['needs_confirmation'] = not analysis['auto_executable'] or analysis['confidence'] < 0.7
        
        logger.info(f"📊 Query analysis: {' | '.join(analysis['reasoning'])}")
        return analysis
    
    def _extract_locations(self, query: str) -> Optional[List[int]]:
        """Extract location endpoint IDs from query with fuzzy matching"""
        endpoint_ids = set()
        
        # Check each location and its aliases
        for loc_key, loc_info in self.location_mapping.items():
            for alias in loc_info['aliases']:
                # Use word boundary for exact matching
                pattern = r'\b' + re.escape(alias.replace('-', r'[-\s]?')) + r'\b'
                if re.search(pattern, query, re.IGNORECASE):
                    endpoint_ids.add(loc_info['id'])
                    logger.debug(f"Matched location: {loc_info['name']} (alias: {alias})")
                    break
        
        # Check for "all" keyword
        if re.search(r'\b(?:all|every|each|entire)\b.*\bclusters?\b', query):
            return self.all_endpoint_ids
        
        return list(endpoint_ids) if endpoint_ids else None
    
    def _extract_region(self, query: str) -> Optional[List[int]]:
        """Extract clusters by region"""
        for pattern in self.cluster_patterns['region_query']:
            match = re.search(pattern, query)
            if match:
                region = match.group(1).lower()
                if 'india' in region:
                    return [11, 12, 14, 162]  # All India locations
                elif 'uk' in region or 'united kingdom' in region:
                    return [204]  # UK location
        return None
    
    def _extract_status(self, query: str) -> Optional[str]:
        """Extract status filter from query with synonym matching"""
        for status, synonyms in self.status_synonyms.items():
            for synonym in synonyms:
                if re.search(r'\b' + re.escape(synonym) + r'\b', query):
                    logger.debug(f"Matched status: {status} (synonym: {synonym})")
                    return status
        return None
    
    def _extract_type(self, query: str) -> Optional[str]:
        """Extract cluster type from query"""
        for cluster_type, synonyms in self.type_synonyms.items():
            for synonym in synonyms:
                if re.search(r'\b' + re.escape(synonym) + r'\b', query):
                    logger.debug(f"Matched type: {cluster_type} (synonym: {synonym})")
                    return cluster_type
        return None
    
    def _extract_version(self, query: str) -> Optional[str]:
        """Extract K8s version from query with normalization"""
        for pattern in self.cluster_patterns['version_query']:
            match = re.search(pattern, query)
            if match:
                version = match.group(1)
                # Normalize to major.minor format
                parts = version.split('.')
                normalized = '.'.join(parts[:2])
                logger.debug(f"Extracted version: {normalized}")
                return normalized
        return None
    
    def _is_count_query(self, query: str) -> bool:
        """Check if this is a count query"""
        for pattern in self.cluster_patterns['count_query']:
            if re.search(pattern, query):
                return True
        return False
    
    def _wants_details(self, query: str) -> bool:
        """Check if user wants detailed output"""
        for pattern in self.cluster_patterns['detailed_query']:
            if re.search(pattern, query):
                return True
        
        # Check for detail keywords
        detail_keywords = ['detail', 'detailed', 'verbose', 'full', 'complete', 'comprehensive', 'info', 'information']
        return any(keyword in query for keyword in detail_keywords)
    
    def _get_location_name(self, endpoint_id: int) -> str:
        """Get location name from endpoint ID"""
        for loc_info in self.location_mapping.values():
            if loc_info['id'] == endpoint_id:
                return loc_info['name']
        return f"Endpoint {endpoint_id}"
    
    def generate_natural_response(self, 
                                 clusters: List[Dict[str, Any]], 
                                 params: Dict[str, Any],
                                 stats: Dict[str, Any],
                                 query_analysis: Dict[str, Any]) -> str:
        """Generate natural language response for cluster data"""
        
        # Handle unsupported version
        if query_analysis.get('query_type') == 'unsupported_version':
            version = params.get('filter_k8s_version')
            response = f"❌ **Kubernetes version {version} is not available**\n\n"
            response += "**Supported versions:**\n"
            for v in self.supported_k8s_versions:
                response += f"  • v{v}\n"
            response += "\n💡 Please try searching with one of the supported versions."
            return response
        
        if not clusters:
            response = "🔍 **No clusters found** matching your criteria.\n\n"
            
            # Provide context-aware suggestions
            if params.get('filter_k8s_version'):
                response += f"💡 **Note**: No clusters with Kubernetes version {params['filter_k8s_version']} found.\n"
                if stats.get('kubernetes_versions'):
                    available = ', '.join(sorted(stats['kubernetes_versions'].keys()))
                    response += f"   **Available versions**: {available}\n"
            
            if params.get('filter_status'):
                response += f"\n💡 **Tip**: Try removing the '{params['filter_status']}' status filter.\n"
            
            if params.get('filter_type'):
                response += f"\n💡 **Tip**: Try removing the '{params['filter_type']}' type filter.\n"
            
            if params.get('endpoint_ids') and len(params['endpoint_ids']) < len(self.all_endpoint_ids):
                response += f"\n💡 **Tip**: Try searching across all locations.\n"
            
            return response
        
        # Build natural response
        total = len(clusters)
        
        # Handle count queries specifically
        if query_analysis.get('query_type') == 'count':
            response = f"📊 **Total Clusters**: {total}\n\n"
            if stats.get('by_location'):
                response += "**Breakdown by Location:**\n"
                for location, count in sorted(stats['by_location'].items()):
                    response += f"  • {location}: {count}\n"
            return response
        
        response = [f"📊 **Found {total} cluster{'s' if total != 1 else ''}**"]
        
        # Add context
        context_parts = []
        if params.get('endpoint_ids') and len(params['endpoint_ids']) < len(self.all_endpoint_ids):
            location_names = [self._get_location_name(eid) for eid in params['endpoint_ids']]
            context_parts.append(f"in {', '.join(location_names)}")
        
        if params.get('filter_status'):
            context_parts.append(f"with status **{params['filter_status']}**")
        
        if params.get('filter_type'):
            context_parts.append(f"of type **{params['filter_type']}**")
        
        if params.get('filter_k8s_version'):
            context_parts.append(f"running Kubernetes **v{params['filter_k8s_version']}**")
        
        if context_parts:
            response.append(' '.join(context_parts))
        
        response_text = ' '.join(response) + "\n\n"
        
        # Add statistics
        if stats:
            response_text += "### 📈 Summary Statistics\n"
            response_text += f"• **Total Clusters**: {stats.get('total', 0)}\n"
            response_text += f"• **Total Nodes**: {stats.get('total_nodes', 0)}\n"
            
            if stats.get('by_location'):
                response_text += "\n**By Location**:\n"
                for location, count in sorted(stats['by_location'].items()):
                    response_text += f"  • {location}: {count} cluster{'s' if count != 1 else ''}\n"
            
            if stats.get('by_status'):
                response_text += "\n**By Status**:\n"
                for status, count in stats['by_status'].items():
                    icon = "✅" if status.lower() == "healthy" else "⚠️" if status.lower() == "draft" else "❌"
                    response_text += f"  {icon} {status}: {count}\n"
            
            if stats.get('by_type'):
                response_text += "\n**By Type**:\n"
                for ctype, count in stats['by_type'].items():
                    response_text += f"  • {ctype}: {count}\n"
            
            if stats.get('kubernetes_versions'):
                response_text += "\n**Kubernetes Versions**:\n"
                for version, count in sorted(stats['kubernetes_versions'].items()):
                    response_text += f"  • v{version}: {count} cluster{'s' if count != 1 else ''}\n"
        
        # Add detailed cluster list if requested
        if params.get('show_details') and total <= 10:
            response_text += "\n### 📋 Detailed Cluster Information\n"
            for idx, cluster in enumerate(clusters, 1):
                response_text += f"\n**{idx}. {cluster.get('name', 'Unknown')}**\n"
                response_text += f"   • **Status**: {cluster.get('status', 'N/A')}\n"
                response_text += f"   • **Type**: {cluster.get('type', 'N/A')}\n"
                response_text += f"   • **K8s Version**: v{cluster.get('k8s_version', 'N/A')}\n"
                response_text += f"   • **Nodes**: {cluster.get('node_count', 0)}\n"
                response_text += f"   • **Location**: {cluster.get('location', 'N/A')}\n"
        elif total > 10:
            response_text += f"\n💡 Showing summary for {total} clusters. Use detailed view for individual cluster information."
        
        return response_text

# ======================== Enhanced Intelligent Action Agent ========================
class IntelligentActionAgent:
    """
    Production-grade intelligent action orchestrator with enhanced cluster intelligence
    """
    
    def __init__(self, ai_service, service_registry: Dict[str, str]):
        self.ai_service = ai_service
        self.service_registry = service_registry
        self.service_capabilities: Dict[str, ServiceCapability] = {}
        self.active_sessions: Dict[str, ConversationSession] = {}
        self.http_client: Optional[httpx.AsyncClient] = None
        self.validator = ParameterValidator()
        self.cluster_intel = ClusterIntelligence()
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
        
        # Statistics tracking
        self.stats = {
            'total_sessions': 0,
            'successful_actions': 0,
            'failed_actions': 0,
            'questions_answered': 0,
            'avg_turns_per_session': 0.0,
            'cluster_queries': 0,
            'auto_executed_queries': 0
        }
        
        logger.info("🤖 Enhanced Intelligent Action Agent initialized")
    
    # ======================== Initialization ========================
    async def initialize(self):
        """Initialize agent with service discovery"""
        logger.info("🚀 Initializing Enhanced Intelligent Action Agent...")
        
        try:
            # Initialize HTTP client with retry configuration
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
            
            logger.info("✅ Enhanced Intelligent Action Agent ready")
            
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
        Main entry point - handle user query intelligently with enhanced cluster support
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
    
    # ======================== Enhanced Cluster Query Handler ========================
    async def handle_cluster_query(self, query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Specialized handler for cluster queries with intelligent analysis
        """
        try:
            # Analyze the query using cluster intelligence
            analysis = self.cluster_intel.analyze_query(query)
            
            self.stats['cluster_queries'] += 1
            
            # If auto-executable, proceed directly to API call
            if analysis['auto_executable']:
                self.stats['auto_executed_queries'] += 1
                
                # Build API payload
                payload = {
                    'endpoints': analysis.get('endpoint_ids', self.cluster_intel.all_endpoint_ids)
                }
                
                # Execute cluster API call
                result = await self._execute_cluster_api(payload)
                
                if not result['success']:
                    return {
                        'status': 'error',
                        'message': result['message'],
                        'session_id': session_id
                    }
                
                # Parse and filter results
                clusters = result.get('clusters', [])
                filtered_clusters, stats = self._filter_clusters(
                    clusters,
                    analysis.get('filter_status'),
                    analysis.get('filter_type'),
                    analysis.get('filter_k8s_version')
                )
                
                # Generate natural language response
                params = {
                    'endpoint_ids': analysis.get('endpoint_ids'),
                    'filter_status': analysis.get('filter_status'),
                    'filter_type': analysis.get('filter_type'),
                    'filter_k8s_version': analysis.get('filter_k8s_version'),
                    'show_details': analysis.get('show_details', False)
                }
                
                response = self.cluster_intel.generate_natural_response(
                    filtered_clusters,
                    params,
                    stats,
                    analysis
                )
                
                return {
                    'status': 'completed',
                    'message': response,
                    'data': {
                        'clusters': filtered_clusters[:20] if not params['show_details'] else filtered_clusters,
                        'stats': stats,
                        'total_count': len(filtered_clusters)
                    },
                    'session_id': session_id,
                    'execution_time': result.get('execution_time', 0)
                }
            
            # If needs confirmation, ask user
            else:
                return {
                    'status': 'needs_clarification',
                    'message': self._generate_clarification_message(analysis),
                    'session_id': session_id,
                    'suggested_params': {
                        'endpoint_ids': analysis.get('endpoint_ids'),
                        'filter_status': analysis.get('filter_status'),
                        'filter_type': analysis.get('filter_type'),
                        'filter_k8s_version': analysis.get('filter_k8s_version')
                    }
                }
        
        except Exception as e:
            logger.exception(f"❌ Cluster query error: {e}")
            return {
                'status': 'error',
                'message': f'Failed to process cluster query: {str(e)}',
                'session_id': session_id
            }
    
    async def _execute_cluster_api(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute cluster API call with retry logic"""
        service_url = self.service_registry.get('cluster_api')
        if not service_url:
            return {
                'success': False,
                'message': 'Cluster API service not configured'
            }
        
        endpoint = f"{service_url}/clusterlist"
        
        start_time = datetime.now()
        retry_count = 0
        
        while retry_count < MAX_RETRIES:
            try:
                logger.info(f"🔄 Calling cluster API (attempt {retry_count + 1}/{MAX_RETRIES})")
                logger.debug(f"Payload: {json.dumps(payload, indent=2)}")
                
                response = await self.http_client.post(
                    endpoint,
                    json=payload,
                    timeout=60.0
                )
                
                response.raise_for_status()
                result = response.json()
                
                execution_time = (datetime.now() - start_time).total_seconds()
                
                logger.info(f"✅ Cluster API call successful ({execution_time:.2f}s)")
                
                return {
                    'success': True,
                    'clusters': result.get('clusters', []),
                    'execution_time': execution_time
                }
                
            except httpx.HTTPStatusError as e:
                logger.error(f"❌ HTTP {e.response.status_code}: {e.response.text[:200]}")
                retry_count += 1
                if retry_count < MAX_RETRIES:
                    await asyncio.sleep(RETRY_BACKOFF ** retry_count)
                else:
                    return {
                        'success': False,
                        'message': f'Cluster API error: HTTP {e.response.status_code}'
                    }
            
            except httpx.TimeoutException:
                logger.error(f"❌ Cluster API timeout")
                retry_count += 1
                if retry_count < MAX_RETRIES:
                    await asyncio.sleep(RETRY_BACKOFF ** retry_count)
                else:
                    return {
                        'success': False,
                        'message': 'Cluster API request timed out'
                    }
            
            except Exception as e:
                logger.exception(f"❌ Cluster API error: {e}")
                return {
                    'success': False,
                    'message': f'Cluster API error: {str(e)}'
                }
        
        return {
            'success': False,
            'message': 'Max retries exceeded'
        }
    
    def _filter_clusters(self, 
                        clusters: List[Dict[str, Any]], 
                        filter_status: Optional[str],
                        filter_type: Optional[str],
                        filter_version: Optional[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Filter clusters and generate statistics"""
        
        filtered = clusters
        
        # Apply status filter
        if filter_status:
            filtered = [c for c in filtered if c.get('status', '').lower() == filter_status.lower()]
        
        # Apply type filter
        if filter_type:
            filtered = [c for c in filtered if c.get('type', '').upper() == filter_type.upper()]
        
        # Apply version filter
        if filter_version:
            filtered = [c for c in filtered if c.get('k8s_version', '').startswith(filter_version)]
        
        # Generate statistics
        stats = {
            'total': len(filtered),
            'total_nodes': sum(c.get('node_count', 0) for c in filtered),
            'by_status': defaultdict(int),
            'by_type': defaultdict(int),
            'by_location': defaultdict(int),
            'kubernetes_versions': defaultdict(int)
        }
        
        for cluster in filtered:
            stats['by_status'][cluster.get('status', 'Unknown')] += 1
            stats['by_type'][cluster.get('type', 'Unknown')] += 1
            stats['by_location'][cluster.get('location', 'Unknown')] += 1
            
            version = cluster.get('k8s_version', '')
            if version:
                # Get major.minor version
                version_parts = version.split('.')[:2]
                normalized_version = '.'.join(version_parts)
                stats['kubernetes_versions'][normalized_version] += 1
        
        # Convert defaultdict to dict
        stats['by_status'] = dict(stats['by_status'])
        stats['by_type'] = dict(stats['by_type'])
        stats['by_location'] = dict(stats['by_location'])
        stats['kubernetes_versions'] = dict(stats['kubernetes_versions'])
        
        return filtered, stats
    
    def _generate_clarification_message(self, analysis: Dict[str, Any]) -> str:
        """Generate clarification message for ambiguous queries"""
        message = "🤔 **I need a bit more information**\n\n"
        
        if not analysis.get('endpoint_ids'):
            message += "**Which location(s)** would you like to query?\n"
            message += "Available locations:\n"
            for loc_key, loc_info in self.cluster_intel.location_mapping.items():
                message += f"  • {loc_info['name']} ({loc_info['code']})\n"
            message += "  • All locations\n\n"
        
        if analysis.get('confidence', 0) < 0.7:
            message += "**What would you like to know about the clusters?**\n"
            message += "  • List all clusters\n"
            message += "  • Filter by status (healthy/unhealthy/draft)\n"
            message += "  • Filter by type (MGMT/APP)\n"
            message += "  • Filter by Kubernetes version\n"
        
        return message
    
    # Remaining methods from original code...
    # (Intent detection, parameter collection, validation, confirmation, execution, etc.)
    # These remain largely unchanged from the original implementation
    
    async def _handle_intent_detection(self, session: ConversationSession, query: str) -> Dict[str, Any]:
        """Detect user intent with cluster-first approach"""
        
        # Check for cluster-specific patterns first
        if self._is_cluster_query(query):
            logger.info(f"🎯 Detected cluster query, routing to specialized handler")
            return await self.handle_cluster_query(query, session.session_id)
        
        # Continue with original intent detection for non-cluster queries
        # ... (rest of original _handle_intent_detection code)
        return {'status': 'processing', 'message': 'Processing query...', 'session_id': session.session_id}
    
    def _is_cluster_query(self, query: str) -> bool:
        """Check if query is cluster-related"""
        cluster_keywords = [
            'cluster', 'kubernetes', 'k8s', 'node', 'endpoint',
            'mumbai', 'delhi', 'bengaluru', 'chennai', 'cressex',
            'mgmt', 'management', 'app', 'application'
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in cluster_keywords)
    
    # ======================== Background Tasks ========================
    async def _periodic_health_check(self):
        """Periodically check service health"""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
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
                
                expired = []
                for session_id, session in self.active_sessions.items():
                    if session.is_expired():
                        expired.append(session_id)
                
                for session_id in expired:
                    del self.active_sessions[session_id]
                
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
    
    # ======================== Utility Methods ========================
    async def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive agent statistics"""
        return {
            **self.stats,
            'active_sessions': len(self.active_sessions),
            'total_services': len(self.service_capabilities),
            'available_services': sum(
                1 for cap in self.service_capabilities.values() 
                if cap.available
            ),
            'cluster_query_rate': (
                self.stats['cluster_queries'] / max(self.stats['total_sessions'], 1)
            ),
            'auto_execution_rate': (
                self.stats['auto_executed_queries'] / max(self.stats['cluster_queries'], 1)
            ),
            'timestamp': datetime.now().isoformat()
        }
    
    async def clear_session(self, session_id: str) -> bool:
        """Clear/delete a session"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            logger.info(f"🗑️ Cleared session: {session_id}")
            return True
        return False
    
    async def close(self):
        """Close the agent and cleanup resources"""
        logger.info("🛑 Shutting down Enhanced Intelligent Action Agent...")
        
        try:
            # Close all sessions
            session_ids = list(self.active_sessions.keys())
            for session_id in session_ids:
                await self.clear_session(session_id)
            
            # Close HTTP client
            if self.http_client:
                await self.http_client.aclose()
                logger.info("✅ HTTP client closed")
            
            logger.info("✅ Enhanced Intelligent Action Agent shutdown complete")
            
        except Exception as e:
            logger.exception(f"Error during shutdown: {e}")

    async def check_service_health(self) -> Dict[str, Any]:
        services = {}

        for name, cap in self.service_capabilities.items():
            services[name] = {
                "available": bool(cap.available),
                "status": (
                    cap.health_status.name
                    if cap.health_status
                    else "UNKNOWN"
                ),
                "last_check": (
                    cap.last_check.isoformat()
                    if cap.last_check
                    else None
                 )
                }

        unhealthy = [
            name for name, svc in services.items()
            if not svc["available"]
    ]

        overall_status = "healthy" if not unhealthy else "degraded"

        return {
        "status": overall_status,
        "services": services
    }

    async def cleanup_expired_sessions(self, max_age_seconds: int = 3600) -> int:
        """
        Public cleanup hook expected by background scheduler.
        """
        expired = []

        now = datetime.now()
        for session_id, session in self.active_sessions.items():
            if session.is_expired():
                expired.append(session_id)

        for session_id in expired:
            del self.active_sessions[session_id]

        if expired:
            logger.info(f"🧹 Cleanup hook removed {len(expired)} expired sessions")

        return len(expired)


# ======================== Export Public API ========================
__all__ = [
    'IntelligentActionAgent',
    'ClusterIntelligence',
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