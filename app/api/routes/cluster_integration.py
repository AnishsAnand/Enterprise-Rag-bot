# cluster_integration.py - Production-ready Cluster API Integration
"""
Integration for Tata Communications PaaS Cluster API
Provides automated cluster information retrieval and management
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum
try:
    from backup.action_bot import Parameter, ParameterType
    PARAMS_AVAILABLE = True
except ImportError:
    PARAMS_AVAILABLE = False
    logging.warning("⚠️ Parameter classes not available, using fallback")

logger = logging.getLogger(__name__)

# ======================== Configuration ========================
CLUSTER_API_BASE_URL = "https://ipcloud.tatacommunications.com/paasservice/paas"
CLUSTER_LIST_ENDPOINT = "/clusterlist"

# ======================== Enums ========================
class ClusterType(Enum):
    """Cluster types"""
    MGMT = "MGMT"
    APP = "APP"

class ClusterStatus(Enum):
    """Cluster health status"""
    HEALTHY = "Healthy"
    DRAFT = "Draft"
    UNHEALTHY = "Unhealthy"
    UNKNOWN = "Unknown"

class LocationEndpoint(Enum):
    """Available location endpoints"""
    MUMBAI_BKC = ("EP_V2_MUM_BKC", "Mumbai-BKC", 11)
    DELHI = ("EP_V2_DEL", "Delhi", 12)
    BENGALURU = ("EP_V2_BL", "Bengaluru", 14)
    CHENNAI_AMB = ("EP_V2_CHN_AMB", "Chennai-AMB", 162)
    CRESSEX = ("EP_V2_UKCX", "Cressex", 204)
    
    def __init__(self, code: str, display_name: str, endpoint_id: int):
        self.code = code
        self.display_name = display_name
        self.endpoint_id = endpoint_id

    class Parameter:
        def __init__(self, name, description, required, param_type, default=None, choices=None, **kwargs):
            self.name = name
            self.description = description
            self.required = required
            self.param_type = param_type
            self.default = default
            self.choices = choices
            self.collected = False
            self.value = None
            self.attempts = 0
            self.max_attempts = 3
        
        def to_dict(self):
            return {
                'name': self.name,
                'description': self.description,
                'required': self.required,
                'type': self.param_type.value if hasattr(self.param_type, 'value') else str(self.param_type),
                'default': self.default,
                'choices': self.choices,
                'collected': self.collected,
                'value': self.value}
        
# ======================== Data Models ========================
class ClusterInfo:
    """Cluster information model"""
    
    def __init__(
        self,
        cluster_id: int,
        cluster_name: str,
        location: str,
        display_name_endpoint: str,
        status: str,
        cluster_type: str,
        nodes_count: int,
        kubernetes_version: Optional[str],
        is_backup_enabled: bool,
        created_time: str,
        ci_master_id: int):
        self.cluster_id = cluster_id
        self.cluster_name = cluster_name
        self.location = location
        self.display_name_endpoint = display_name_endpoint
        self.status = status
        self.cluster_type = cluster_type
        self.nodes_count = nodes_count
        self.kubernetes_version = kubernetes_version
        self.is_backup_enabled = is_backup_enabled
        self.created_time = created_time
        self.ci_master_id = ci_master_id
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'cluster_id': self.cluster_id,
            'cluster_name': self.cluster_name,
            'location': self.location,
            'endpoint': self.display_name_endpoint,
            'status': self.status,
            'type': self.cluster_type,
            'nodes': self.nodes_count,
            'k8s_version': self.kubernetes_version or 'N/A',
            'backup_enabled': self.is_backup_enabled,
            'created': self.created_time,
            'ci_master_id': self.ci_master_id}

    def to_summary(self) -> str:
        """Generate human-readable summary"""
        return (
            f"**{self.cluster_name}** ({self.cluster_type})\n"
            f"  • Location: {self.display_name_endpoint}\n"
            f"  • Status: {self.status}\n"
            f"  • Nodes: {self.nodes_count}\n"
            f"  • K8s Version: {self.kubernetes_version or 'N/A'}\n"
            f"  • Backup: {'✅ Enabled' if self.is_backup_enabled else '❌ Disabled'}\n"
            f"  • Cluster ID: {self.cluster_id}")

# ======================== Parameter Definitions ========================
def get_cluster_list_parameters() -> List[Parameter]:
    """
    Get parameters for cluster list action
    Compatible with action_bot.py Parameter class
    """
    return [
        Parameter(
            name='endpoints',
            description='Location endpoints to query (comma-separated IDs or names)',
            required=False,
            param_type=ParameterType.STRING,
            default='all',
            choices=['all', 'mumbai', 'delhi', 'bengaluru', 'chennai', 'cressex', 'custom']
        ),
        Parameter(
            name='custom_endpoint_ids',
            description='Custom endpoint IDs (comma-separated numbers, e.g., 11,12,14)',
            required=False,
            param_type=ParameterType.STRING,
            validation_regex=r'^[\d,\s]+$'
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
            default=False)]

# ======================== Cluster API Handler ========================
class ClusterAPIHandler:
    """Handler for cluster API operations"""
    
    def __init__(self, http_client, project_id: str):
        self.http_client = http_client
        self.project_id = project_id
        self.base_url = f"{CLUSTER_API_BASE_URL}/{project_id}"
        logger.info(f"🔗 Cluster API Handler initialized: {self.base_url}")
    
    async def get_clusters(self,endpoint_ids: Optional[List[int]] = None,filter_status: Optional[str] = None,filter_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetch cluster list from API
        Args:
            endpoint_ids: List of endpoint IDs to query
            filter_status: Filter by status (healthy, draft, unhealthy)
            filter_type: Filter by type (MGMT, APP)
        Returns:
            Dictionary with status, clusters, and metadata
        """
        try:
            # Default to all endpoints if none specified
            if not endpoint_ids:
                endpoint_ids = [11, 12, 14, 162, 204]
            url = f"{self.base_url}{CLUSTER_LIST_ENDPOINT}"
            payload = {"endpoints": endpoint_ids}
            logger.info(f"🔍 Fetching clusters for endpoints: {endpoint_ids}")
            response = await self.http_client.post(url,json=payload,timeout=30.0)
            response.raise_for_status()
            result = response.json()
            if result.get('status') != 'success':
                return {'success': False,'message': f"API returned status: {result.get('status')}",'error': result.get('message')}
            # Parse and filter clusters
            clusters = self._parse_clusters(result.get('data', []))
            if filter_status and filter_status != 'all':
                clusters = self._filter_by_status(clusters, filter_status)
            if filter_type and filter_type != 'all':
                clusters = self._filter_by_type(clusters, filter_type)
            # Generate statistics
            stats = self._generate_statistics(clusters)
            logger.info(f"✅ Retrieved {len(clusters)} clusters")
            return {
                'success': True,
                'clusters': [c.to_dict() for c in clusters],
                'total_count': len(clusters),
                'statistics': stats,
                'endpoint_ids': endpoint_ids,
                'timestamp': datetime.now().isoformat()}    
        except Exception as e:
            logger.exception(f"❌ Error fetching clusters: {e}")
            return {'success': False,'message': f"Failed to fetch clusters: {str(e)}",'error': str(e)}
    
    def _parse_clusters(self, data: List[Dict[str, Any]]) -> List[ClusterInfo]:
        """Parse raw cluster data into ClusterInfo objects"""
        clusters = []
        for item in data:
            try:
                # Parse backup enabled
                is_backup = str(item.get('isIksBackupEnabled', 'false')).lower() == 'true'
                cluster = ClusterInfo(
                    cluster_id=int(item.get('clusterId', 0)),
                    cluster_name=item.get('clusterName', 'Unknown'),
                    location=item.get('location', 'Unknown'),
                    display_name_endpoint=item.get('displayNameEndpoint', 'Unknown'),
                    status=item.get('status', 'Unknown'),
                    cluster_type=item.get('type', 'APP'),
                    nodes_count=int(item.get('nodescount', 0)),
                    kubernetes_version=item.get('kubernetesVersion'),
                    is_backup_enabled=is_backup,
                    created_time=item.get('createdTime', ''),
                    ci_master_id=int(item.get('ciMasterId', 0)))
                clusters.append(cluster)
            except Exception as e:
                logger.warning(f"⚠️ Failed to parse cluster: {e}")
                continue
        return clusters
    
    def _filter_by_status(self, clusters: List[ClusterInfo], status: str) -> List[ClusterInfo]:
        """Filter clusters by status"""
        status_lower = status.lower()
        if status_lower == 'healthy':
            return [c for c in clusters if c.status == ClusterStatus.HEALTHY.value]
        elif status_lower == 'draft':
            return [c for c in clusters if c.status == ClusterStatus.DRAFT.value]
        elif status_lower == 'unhealthy':
            return [c for c in clusters if c.status == ClusterStatus.UNHEALTHY.value]
        return clusters
    
    def _filter_by_type(self, clusters: List[ClusterInfo], cluster_type: str) -> List[ClusterInfo]:
        """Filter clusters by type"""
        cluster_type_upper = cluster_type.upper()
        return [c for c in clusters if c.cluster_type == cluster_type_upper]
    
    def _generate_statistics(self, clusters: List[ClusterInfo]) -> Dict[str, Any]:
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
            'kubernetes_versions': {}}
        for cluster in clusters:
            # Count by status
            status_key = cluster.status
            stats['by_status'][status_key] = stats['by_status'].get(status_key, 0) + 1
            # Count by type
            type_key = cluster.cluster_type
            stats['by_type'][type_key] = stats['by_type'].get(type_key, 0) + 1
            # Count by location
            location_key = cluster.display_name_endpoint
            stats['by_location'][location_key] = stats['by_location'].get(location_key, 0) + 1
            # Total nodes
            stats['total_nodes'] += cluster.nodes_count
            # Backup enabled count
            if cluster.is_backup_enabled:
                stats['backup_enabled_count'] += 1
            # Kubernetes versions
            if cluster.kubernetes_version:
                version_key = cluster.kubernetes_version
                stats['kubernetes_versions'][version_key] = stats['kubernetes_versions'].get(version_key, 0) + 1
        return stats
    
    def format_clusters_output(self,clusters: List[ClusterInfo],show_details: bool = False) -> str:
        """Format clusters for display"""
        if not clusters:
            return "ℹ️ No clusters found matching your criteria."
        output = [f"📊 **Found {len(clusters)} Clusters**\n"]
        if show_details:
            # Detailed view
            for i, cluster in enumerate(clusters, 1):
                output.append(f"\n**{i}. {cluster.to_summary()}**")
        else:
            # Summary view grouped by location
            by_location = {}
            for cluster in clusters:
                location = cluster.display_name_endpoint
                if location not in by_location:
                    by_location[location] = []
                by_location[location].append(cluster)
            for location, loc_clusters in sorted(by_location.items()):
                output.append(f"\n### 📍 {location} ({len(loc_clusters)} clusters)")
                for cluster in loc_clusters:
                    status_icon = "✅" if cluster.status == ClusterStatus.HEALTHY.value else "⚠️"
                    output.append(
                        f"  {status_icon} **{cluster.cluster_name}** "
                        f"({cluster.cluster_type}) - "
                        f"{cluster.nodes_count} nodes - "
                        f"{cluster.kubernetes_version or 'N/A'}")
        return "\n".join(output)

# ======================== Helper Functions ========================

def parse_endpoint_selection(endpoint_param: str) -> Optional[List[int]]:
    """
    Parse endpoint selection parameter into list of endpoint IDs
    
    Args:
        endpoint_param: String like 'all', 'mumbai', 'delhi', or 'custom'
    Returns:
        List of endpoint IDs or None
    """
    endpoint_map = {
        'all': [11, 12, 14, 162, 204],
        'mumbai': [11],
        'delhi': [12],
        'bengaluru': [14],
        'chennai': [162],
        'cressex': [204]}
    return endpoint_map.get(endpoint_param.lower())

def parse_custom_endpoint_ids(custom_ids: str) -> List[int]:
    """
    Parse custom endpoint IDs from comma-separated string
    Args:
        custom_ids: String like "11,12,14"
    
    Returns:
        List of integer endpoint IDs
    """
    try:
        return [int(x.strip()) for x in custom_ids.split(',') if x.strip()]
    except ValueError:
        logger.warning(f"Invalid custom endpoint IDs: {custom_ids}")
        return []

# ======================== Integration Helper ========================
def create_cluster_handler(http_client, project_id: str = "1923") -> ClusterAPIHandler:
    return ClusterAPIHandler(http_client, project_id)
# ======================== Export Public API ========================
__all__ = [
    'ClusterAPIHandler',
    'ClusterInfo',
    'ClusterType',
    'ClusterStatus',
    'LocationEndpoint',
    'get_cluster_list_parameters',
    'parse_endpoint_selection',
    'parse_custom_endpoint_ids',
    'create_cluster_handler',
]