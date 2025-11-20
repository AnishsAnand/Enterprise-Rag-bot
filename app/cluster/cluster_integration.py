# cluster_api_integration.py - Production-ready cluster API integration
"""
Integration for Tata Communications PaaS cluster API
Provides automated cluster information retrieval and management
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum
from dataclasses import dataclass

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

# ======================== Data Models ========================
@dataclass
class ClusterInfo:
    """Cluster information model"""
    cluster_id: int
    cluster_name: str
    location: str
    display_name_endpoint: str
    status: ClusterStatus
    cluster_type: ClusterType
    nodes_count: int
    kubernetes_version: Optional[str]
    is_backup_enabled: bool
    created_time: str
    ci_master_id: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'cluster_id': self.cluster_id,
            'cluster_name': self.cluster_name,
            'location': self.location,
            'endpoint': self.display_name_endpoint,
            'status': self.status.value,
            'type': self.cluster_type.value,
            'nodes': self.nodes_count,
            'k8s_version': self.kubernetes_version or 'N/A',
            'backup_enabled': self.is_backup_enabled,
            'created': self.created_time
        }
    
    def to_summary(self) -> str:
        """Generate human-readable summary"""
        return (
            f"**{self.cluster_name}** ({self.cluster_type.value})\n"
            f"  • Location: {self.display_name_endpoint}\n"
            f"  • Status: {self.status.value}\n"
            f"  • Nodes: {self.nodes_count}\n"
            f"  • K8s Version: {self.kubernetes_version or 'N/A'}\n"
            f"  • Backup: {'✅ Enabled' if self.is_backup_enabled else '❌ Disabled'}\n"
            f"  • Cluster ID: {self.cluster_id}"
        )

# ======================== Action Type Extension ========================
# Add to existing ActionType enum in action_agent.py:
"""
class ActionType(Enum):
    # ... existing types ...
    CLUSTER_LIST = "cluster_list"          # New
    CLUSTER_DETAILS = "cluster_details"    # New
    CLUSTER_FILTER = "cluster_filter"      # New
"""

# ======================== Parameter Definitions ========================
def get_cluster_list_parameters() -> List:
    """Get parameters for cluster list action"""
    from action_agent import Parameter, ParameterType
    
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
            default=False
        )
    ]

# ======================== Cluster API Handler ========================
class ClusterAPIHandler:
    """Handler for cluster API operations"""
    
    def __init__(self, http_client, project_id: str):
        self.http_client = http_client
        self.project_id = project_id
        self.base_url = f"{CLUSTER_API_BASE_URL}/{project_id}"
    
    async def get_clusters(
        self,
        endpoint_ids: Optional[List[int]] = None,
        filter_status: Optional[str] = None,
        filter_type: Optional[str] = None
    ) -> Dict[str, Any]:
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
            
            response = await self.http_client.post(
                url,
                json=payload,
                timeout=30.0
            )
            response.raise_for_status()
            
            result = response.json()
            
            if result.get('status') != 'success':
                return {
                    'success': False,
                    'message': f"API returned status: {result.get('status')}",
                    'error': result.get('message')
                }
            
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
                'clusters': clusters,
                'total_count': len(clusters),
                'statistics': stats,
                'endpoint_ids': endpoint_ids,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.exception(f"❌ Error fetching clusters: {e}")
            return {
                'success': False,
                'message': f"Failed to fetch clusters: {str(e)}",
                'error': str(e)
            }
    
    def _parse_clusters(self, data: List[Dict[str, Any]]) -> List[ClusterInfo]:
        """Parse raw cluster data into ClusterInfo objects"""
        clusters = []
        
        for item in data:
            try:
                # Parse status
                status_str = item.get('status', 'Unknown')
                try:
                    status = ClusterStatus(status_str)
                except ValueError:
                    status = ClusterStatus.UNKNOWN
                
                # Parse type
                type_str = item.get('type', 'APP')
                try:
                    cluster_type = ClusterType(type_str)
                except ValueError:
                    cluster_type = ClusterType.APP
                
                # Parse backup enabled
                is_backup = str(item.get('isIksBackupEnabled', 'false')).lower() == 'true'
                
                cluster = ClusterInfo(
                    cluster_id=int(item.get('clusterId', 0)),
                    cluster_name=item.get('clusterName', 'Unknown'),
                    location=item.get('location', 'Unknown'),
                    display_name_endpoint=item.get('displayNameEndpoint', 'Unknown'),
                    status=status,
                    cluster_type=cluster_type,
                    nodes_count=int(item.get('nodescount', 0)),
                    kubernetes_version=item.get('kubernetesVersion'),
                    is_backup_enabled=is_backup,
                    created_time=item.get('createdTime', ''),
                    ci_master_id=int(item.get('ciMasterId', 0))
                )
                
                clusters.append(cluster)
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to parse cluster: {e}")
                continue
        
        return clusters
    
    def _filter_by_status(self, clusters: List[ClusterInfo], status: str) -> List[ClusterInfo]:
        """Filter clusters by status"""
        status_map = {
            'healthy': ClusterStatus.HEALTHY,
            'draft': ClusterStatus.DRAFT,
            'unhealthy': ClusterStatus.UNHEALTHY
        }
        
        target_status = status_map.get(status.lower())
        if not target_status:
            return clusters
        
        return [c for c in clusters if c.status == target_status]
    
    def _filter_by_type(self, clusters: List[ClusterInfo], cluster_type: str) -> List[ClusterInfo]:
        """Filter clusters by type"""
        try:
            target_type = ClusterType(cluster_type.upper())
            return [c for c in clusters if c.cluster_type == target_type]
        except ValueError:
            return clusters
    
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
            'kubernetes_versions': {}
        }
        
        for cluster in clusters:
            # Count by status
            status_key = cluster.status.value
            stats['by_status'][status_key] = stats['by_status'].get(status_key, 0) + 1
            
            # Count by type
            type_key = cluster.cluster_type.value
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
    
    def format_clusters_output(
        self,
        clusters: List[ClusterInfo],
        show_details: bool = False
    ) -> str:
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
                    status_icon = "✅" if cluster.status == ClusterStatus.HEALTHY else "⚠️"
                    output.append(
                        f"  {status_icon} **{cluster.cluster_name}** "
                        f"({cluster.cluster_type.value}) - "
                        f"{cluster.nodes_count} nodes - "
                        f"{cluster.kubernetes_version or 'N/A'}"
                    )
        
        return "\n".join(output)

# ======================== Integration with Action Agent ========================

def integrate_cluster_api_with_action_agent():
    """
    Instructions for integrating cluster API with action_agent.py
    
    Add this to action_agent.py in the ActionType enum:
    """
    integration_code = '''
# In action_agent.py, update ActionType enum:

class ActionType(Enum):
    # ... existing types ...
    CLUSTER_LIST = "cluster_list"
    CLUSTER_DETAILS = "cluster_details"
    CLUSTER_FILTER = "cluster_filter"

# In _get_parameters_for_action method, add:

elif action_type == ActionType.CLUSTER_LIST:
    return get_cluster_list_parameters()

# In _detect_action_type method, add detection logic:

elif any(word in query_lower for word in ['cluster', 'clusters', 'kubernetes', 'k8s']):
    if any(word in query_lower for word in ['list', 'show', 'get', 'fetch']):
        action_type = ActionType.CLUSTER_LIST
        service_target = 'cluster_api'
    else:
        action_type = ActionType.CLUSTER_LIST
        service_target = 'cluster_api'

# In _execute_action method, add execution handler:

elif action_type == ActionType.CLUSTER_LIST:
    result = await self._execute_cluster_list(service_target, params)

# Add new execution method:

async def _execute_cluster_list(self, service: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute cluster list retrieval"""
    try:
        # Parse endpoint selection
        endpoints_param = params.get('endpoints', 'all')
        endpoint_ids = None
        
        if endpoints_param == 'all':
            endpoint_ids = [11, 12, 14, 162, 204]  # All locations
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
                        'message': '❌ Invalid endpoint IDs format'
                    }
        
        # Get project ID from environment or config
        project_id = os.getenv('CLUSTER_API_PROJECT_ID', '1923')
        
        # Create handler
        handler = ClusterAPIHandler(self.http_client, project_id)
        
        # Fetch clusters
        result = await handler.get_clusters(
            endpoint_ids=endpoint_ids,
            filter_status=params.get('filter_status'),
            filter_type=params.get('filter_type')
        )
        
        if not result['success']:
            return result
        
        # Format output
        clusters = result['clusters']
        show_details = params.get('show_details', False)
        formatted_output = handler.format_clusters_output(clusters, show_details)
        
        # Generate statistics summary
        stats = result['statistics']
        stats_summary = [
            "\\n📈 **Statistics:**",
            f"  • Total Clusters: {stats.get('total', 0)}",
            f"  • Total Nodes: {stats.get('total_nodes', 0)}",
            f"  • Backup Enabled: {stats.get('backup_enabled_count', 0)}",
            "\\n**By Status:**"
        ]
        
        for status, count in stats.get('by_status', {}).items():
            stats_summary.append(f"  • {status}: {count}")
        
        stats_summary.append("\\n**By Type:**")
        for cluster_type, count in stats.get('by_type', {}).items():
            stats_summary.append(f"  • {cluster_type}: {count}")
        
        return {
            'success': True,
            'message': f'✅ Successfully retrieved cluster information\\n\\n{formatted_output}\\n{"".join(stats_summary)}',
            'details': {
                'total_clusters': len(clusters),
                'clusters': [c.to_dict() for c in clusters],
                'statistics': stats
            }
        }
        
    except Exception as e:
        logger.exception(f"❌ Cluster list execution failed: {e}")
        return {
            'success': False,
            'message': f'❌ Failed to retrieve clusters: {str(e)}',
            'error': str(e)
        }
'''
    return integration_code

# ======================== Environment Configuration ========================
def get_env_configuration():
    """Environment variables needed"""
    return '''
# Add to .env file:

# Cluster API Configuration
CLUSTER_API_PROJECT_ID=1923
CLUSTER_API_BASE_URL=https://ipcloud.tatacommunications.com/paasservice/paas
CLUSTER_API_TIMEOUT=30

# Optional: Authentication if required
CLUSTER_API_KEY=your-api-key-here
CLUSTER_API_AUTH_TOKEN=your-token-here
'''

# ======================== Usage Examples ========================
USAGE_EXAMPLES = '''
# Example 1: List all clusters
User: "Show me all kubernetes clusters"
Agent: [Detects cluster_list action]
Agent: "Which locations would you like to query?"
User: "all"
Agent: [Executes and displays formatted cluster list with statistics]

# Example 2: Filter by location
User: "List clusters in Mumbai and Delhi"
Agent: [Auto-detects endpoints]
Agent: [Shows clusters from Mumbai-BKC and Delhi]

# Example 3: Filter by status
User: "Show healthy clusters only"
Agent: [Filters and displays only healthy clusters]

# Example 4: Detailed view
User: "Get detailed information about all clusters"
Agent: "Show detailed cluster information?"
User: "yes"
Agent: [Shows comprehensive details for each cluster]

# Example 5: Custom endpoints
User: "Query endpoints 11, 14, 162"
Agent: [Uses custom endpoint IDs]
Agent: [Retrieves and displays results]
'''

if __name__ == "__main__":
    print("🔧 Cluster API Integration Module")
    print("="*60)
    print("\n📝 Integration Instructions:")
    print(integrate_cluster_api_with_action_agent())
    print("\n⚙️ Environment Configuration:")
    print(get_env_configuration())
    print("\n💡 Usage Examples:")
    print(USAGE_EXAMPLES)