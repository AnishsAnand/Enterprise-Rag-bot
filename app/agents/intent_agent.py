"""
Intent Agent - Detects user intent and extracts parameters from user input.
Identifies what resource and operation the user wants to perform.
"""

from typing import Any, Dict, List, Optional
from langchain.tools import Tool
import logging
import json
import re

from app.agents.base_agent import BaseAgent
from app.services.api_executor_service import api_executor_service

logger = logging.getLogger(__name__)


class IntentAgent(BaseAgent):
    """
    Agent specialized in detecting user intent and extracting parameters.
    Identifies resource type, operation, and extracts relevant parameters from user input.
    """
    
    def __init__(self):
        super().__init__(
            agent_name="IntentAgent",
            agent_description=(
                "Detects user intent for CRUD operations on cloud resources. "
                "Extracts parameters from natural language input."
            ),
            temperature=0.1  # Low temperature for consistent intent detection
        )
        
        # Load resource schema
        self.resource_schema = api_executor_service.resource_schema
        
        # Setup agent
        self.setup_agent()
    
    def get_system_prompt(self) -> str:
        """Return system prompt for intent agent."""
        resources_info = self._get_resources_info()
        
        prompt = """You are the Intent Agent, specialized in detecting user intent for cloud resource operations.

**Available Resources:**
""" + resources_info + """

**Your tasks:**
1. **Identify the resource type** the user wants to work with (k8s_cluster, firewall, kafka, gitlab, container_registry, jenkins, postgres, documentdb, etc.)
2. **Identify the operation** (create, read, update, delete, list)
3. **Extract parameters** from the user's message
4. **Return structured JSON** with your findings

**Output Format:**
Always respond with a JSON object containing:
- intent_detected: boolean (true/false)
- resource_type: string (k8s_cluster, firewall, kafka, gitlab, container_registry, jenkins, postgres, documentdb, etc.)
- operation: string (create, read, update, delete, list)
- extracted_params: object with extracted parameters
- confidence: number (0.0 to 1.0)
- ambiguities: array of unclear things
- clarification_needed: string question if needed, or null

**Examples:**

User: "Create a new Kubernetes cluster named prod-cluster"
→ intent_detected: true, resource_type: k8s_cluster, operation: create, extracted_params with Cluster Name: prod-cluster

User: "Delete the firewall rule"  
→ intent_detected: true, resource_type: firewall, operation: delete, ambiguities: Which firewall rule?

User: "Show me all clusters" or "List clusters"
→ intent_detected: true, resource_type: k8s_cluster, operation: list, extracted_params: empty

User: "What are the clusters in Mumbai?" or "What clusters are available in Delhi?"
→ intent_detected: true, resource_type: k8s_cluster, operation: list, extracted_params: empty

User: "How many clusters in Chennai?" or "Count clusters in Bengaluru"
→ intent_detected: true, resource_type: k8s_cluster, operation: list, extracted_params: empty

User: "Tell me about clusters in Mumbai and Chennai"
→ intent_detected: true, resource_type: k8s_cluster, operation: list, extracted_params: empty

User: "What are the available clusters?" or "What k8s clusters do we have?"
→ intent_detected: true, resource_type: k8s_cluster, operation: list, extracted_params: empty

**Cluster Filtering by BU/Environment/Zone Examples:**

User: "List clusters in business unit XYZ" or "Show clusters for BU ABC"
→ intent_detected: true, resource_type: k8s_cluster, operation: list, extracted_params: empty

User: "Filter clusters by department TATA" or "Show clusters in department test"
→ intent_detected: true, resource_type: k8s_cluster, operation: list, extracted_params: empty

User: "List clusters in environment production" or "Show clusters for env staging"
→ intent_detected: true, resource_type: k8s_cluster, operation: list, extracted_params: empty

User: "Filter clusters by zone XYZ" or "Show clusters in zone test"
→ intent_detected: true, resource_type: k8s_cluster, operation: list, extracted_params: empty

User: "What clusters are in the TATA COMMUNICATIONS business unit?"
→ intent_detected: true, resource_type: k8s_cluster, operation: list, extracted_params: empty

**Cluster Info/Lookup Examples (Reverse Mapping - Zone/Env/BU lookup):**

User: "Which zone is cluster blr-paas in?" or "What zone is blr-paas cluster in?"
→ intent_detected: true, resource_type: k8s_cluster, operation: read, extracted_params with cluster_name: "blr-paas"

User: "Which environment is cluster my-cluster in?" or "What env does my-cluster belong to?"
→ intent_detected: true, resource_type: k8s_cluster, operation: read, extracted_params with cluster_name: "my-cluster"

User: "Which business unit is cluster prod-app in?" or "What BU does prod-app belong to?"
→ intent_detected: true, resource_type: k8s_cluster, operation: read, extracted_params with cluster_name: "prod-app"

User: "Tell me about cluster test-cluster" or "Info about cluster dev-cluster"
→ intent_detected: true, resource_type: k8s_cluster, operation: read, extracted_params with cluster_name: "test-cluster"

User: "Find cluster staging-app" or "Lookup cluster production-web"
→ intent_detected: true, resource_type: k8s_cluster, operation: read, extracted_params with cluster_name: "staging-app"

User: "Where is cluster my-app located?" or "What is the hierarchy for cluster test?"
→ intent_detected: true, resource_type: k8s_cluster, operation: read, extracted_params with cluster_name: "my-app"

**Cluster Firewall Lookup Examples (find firewall associated with a cluster):**

User: "Which firewall is cluster blr-paas associated to?" or "What firewall is blr-paas using?"
→ intent_detected: true, resource_type: k8s_cluster, operation: read, extracted_params with cluster_name: "blr-paas"

User: "What firewall does cluster my-cluster use?" or "Show firewall for cluster test-app"
→ intent_detected: true, resource_type: k8s_cluster, operation: read, extracted_params with cluster_name: "my-cluster"

User: "Find the edge gateway for cluster prod-cluster"
→ intent_detected: true, resource_type: k8s_cluster, operation: read, extracted_params with cluster_name: "prod-cluster"

**NOTE:** When user asks about a firewall FOR a specific cluster, route to k8s_cluster read operation, NOT firewall read.

**Kafka Service Examples:**

User: "List Kafka services" or "Show me Kafka" or "What Kafka services do we have?"
→ intent_detected: true, resource_type: kafka, operation: list, extracted_params: empty

User: "Show Kafka in Mumbai" or "List Kafka services in Delhi"
→ intent_detected: true, resource_type: kafka, operation: list, extracted_params: empty

User: "How many Kafka services?" or "Count Kafka instances"
→ intent_detected: true, resource_type: kafka, operation: list, extracted_params: empty

**GitLab Service Examples:**

User: "List GitLab services" or "Show me GitLab" or "What GitLab services do we have?"
→ intent_detected: true, resource_type: gitlab, operation: list, extracted_params: empty

User: "Show GitLab in Chennai" or "List GitLab services in Bengaluru"
→ intent_detected: true, resource_type: gitlab, operation: list, extracted_params: empty

User: "How many GitLab instances?" or "Count GitLab services"
→ intent_detected: true, resource_type: gitlab, operation: list, extracted_params: empty

**Container Registry Service Examples:**

User: "List container registries" or "Show me container registry" or "What registries do we have?"
→ intent_detected: true, resource_type: container_registry, operation: list, extracted_params: empty

User: "Show docker registry in Mumbai" or "List registries in Delhi"
→ intent_detected: true, resource_type: container_registry, operation: list, extracted_params: empty

User: "How many container registries?" or "Count registry services"
→ intent_detected: true, resource_type: container_registry, operation: list, extracted_params: empty

**Jenkins Service Examples:**

User: "List Jenkins services" or "Show me Jenkins" or "What Jenkins instances do we have?"
→ intent_detected: true, resource_type: jenkins, operation: list, extracted_params: empty

User: "Show Jenkins in Chennai" or "List CI/CD services in Bengaluru"
→ intent_detected: true, resource_type: jenkins, operation: list, extracted_params: empty

User: "How many Jenkins servers?" or "Count Jenkins instances"
→ intent_detected: true, resource_type: jenkins, operation: list, extracted_params: empty

**PostgreSQL Service Examples:**

User: "List PostgreSQL services" or "Show me Postgres" or "What Postgres databases do we have?"
→ intent_detected: true, resource_type: postgres, operation: list, extracted_params: empty

User: "Show Postgres in Mumbai" or "List PostgreSQL services in Delhi"
→ intent_detected: true, resource_type: postgres, operation: list, extracted_params: empty

User: "How many Postgres instances?" or "Count PostgreSQL databases"
→ intent_detected: true, resource_type: postgres, operation: list, extracted_params: empty

**DocumentDB Service Examples:**

User: "List DocumentDB services" or "Show me DocumentDB" or "What MongoDB services do we have?"
→ intent_detected: true, resource_type: documentdb, operation: list, extracted_params: empty

User: "Show DocumentDB in Chennai" or "List NoSQL databases in Bengaluru"
→ intent_detected: true, resource_type: documentdb, operation: list, extracted_params: empty

User: "How many DocumentDB instances?" or "Count MongoDB services"
→ intent_detected: true, resource_type: documentdb, operation: list, extracted_params: empty

**Virtual Machine (VM) Examples:**

User: "List VMs" or "Show me virtual machines" or "What VMs do we have?"
→ intent_detected: true, resource_type: vm, operation: list, extracted_params: empty

User: "Show all servers" or "List instances" or "What virtual machines are running?"
→ intent_detected: true, resource_type: vm, operation: list, extracted_params: empty

User: "How many VMs?" or "Count virtual machines" or "Show me all instances"
→ intent_detected: true, resource_type: vm, operation: list, extracted_params: empty

User: "List VMs in Mumbai" or "Show virtual machines in Delhi endpoint"
→ intent_detected: true, resource_type: vm, operation: list, extracted_params: {{endpoint: Mumbai}}

User: "Show VMs in zone XYZ" or "List virtual machines in department ABC"
→ intent_detected: true, resource_type: vm, operation: list, extracted_params: {{zone: XYZ}} or {{department: ABC}}

**Firewall Examples:**

User: "List firewalls" or "Show me firewalls" or "What firewalls do we have?"
→ intent_detected: true, resource_type: firewall, operation: list, extracted_params: empty

User: "Show firewalls in Mumbai" or "List network firewalls in Delhi"
→ intent_detected: true, resource_type: firewall, operation: list, extracted_params: empty

User: "How many firewalls?" or "Count firewalls" or "Show all Vayu firewalls"
→ intent_detected: true, resource_type: firewall, operation: list, extracted_params: empty

**Reports Examples:**

User: "Show common cluster report" or "Open the common cluster report"
→ intent_detected: true, resource_type: reports, operation: list, extracted_params: {{report_type: common_cluster}}

User: "Show cluster inventory report" or "Open the cluster report"
→ intent_detected: true, resource_type: reports, operation: list, extracted_params: {{report_type: cluster_inventory}}

User: "Show cluster compute report" or "Open the cluster compute report"
→ intent_detected: true, resource_type: reports, operation: list, extracted_params: {{report_type: cluster_compute}}

User: "Show storage inventory report" or "Open the PVC report"
→ intent_detected: true, resource_type: reports, operation: list, extracted_params: {{report_type: storage_inventory}}

User: "List reports" or "Show reports table"
→ intent_detected: true, resource_type: reports, operation: list, extracted_params: empty

**RBAC (Role-Based Access Control) Examples:**

User: "List role bindings" or "Show user access" or "Who has access?"
→ intent_detected: true, resource_type: rbac, operation: list, extracted_params: empty

User: "Show access for user aramalin" or "What access does aramalin have?"
→ intent_detected: true, resource_type: rbac, operation: read, extracted_params: {{username: aramalin}}

User: "Who has admin access to blr-paas cluster?" or "List admins for cluster demo-caas"
→ intent_detected: true, resource_type: rbac, operation: list, extracted_params: {{cluster_name: blr-paas, role: admin}}

User: "Show all admins" or "List users with admin role"
→ intent_detected: true, resource_type: rbac, operation: list, extracted_params: {{role: admin}}

User: "What clusters can user cishan access?" or "Show cishan's permissions"
→ intent_detected: true, resource_type: rbac, operation: read, extracted_params: {{username: cishan}}

**CRITICAL: Role Bindings vs Environments - READ CAREFULLY:**
- "role bindings" = operation: list (NOT list_env!) - shows user access/permissions
- "environments" = operation: list_env - shows available environments

User: "List users in environment 5273" or "Show role bindings in env 5345"
→ intent_detected: true, resource_type: rbac, operation: list, extracted_params: {{env_id: 5273}}

User: "Tell me the role bindings for env TCL-IKS-PAAS-UAT-PROD-ENV-DND"
→ intent_detected: true, resource_type: rbac, operation: list, extracted_params: {{env_name: TCL-IKS-PAAS-UAT-PROD-ENV-DND}}

User: "Show role bindings in IKS_PAAS_BLR_DEMO_ENV"
→ intent_detected: true, resource_type: rbac, operation: list, extracted_params: {{env_name: IKS_PAAS_BLR_DEMO_ENV}}

User: "List role bindings" or "Show user access" or "Who has access?"
→ intent_detected: true, resource_type: rbac, operation: list, extracted_params: empty

User: "Who has exceptdelete role?" or "List users with except delete access"
→ intent_detected: true, resource_type: rbac, operation: list, extracted_params: {{role: exceptdelete}}

User: "What access does user aramalin have?" or "Show access for user bharatm"
→ intent_detected: true, resource_type: rbac, operation: read, extracted_params: {{username: aramalin}}

**Environment Listing (NOT role bindings):**

User: "Show environments in IKS_PAAS_BLR_BU" or "List environments in BU"
→ intent_detected: true, resource_type: rbac, operation: list_env, extracted_params: {{bu_name: IKS_PAAS_BLR_BU}}

User: "What environments are available in TCL-IKS-PAAS-BU?"
→ intent_detected: true, resource_type: rbac, operation: list_env, extracted_params: {{bu_name: TCL-IKS-PAAS-BU}}

User: "What are the environments available in business unit NSX AutoScale BU - DND?"
→ intent_detected: true, resource_type: rbac, operation: list_env, extracted_params: {{bu_name: NSX AutoScale BU - DND}}

User: "What are the envs available in bu NSX AutoScale BU - DND?"
→ intent_detected: true, resource_type: rbac, operation: list_env, extracted_params: {{bu_name: NSX AutoScale BU - DND}}

User: "List envs in business unit TATA COMMUNICATIONS"
→ intent_detected: true, resource_type: rbac, operation: list_env, extracted_params: {{bu_name: TATA COMMUNICATIONS}}

User: "Show me environments in VCD-DELHI-BU"
→ intent_detected: true, resource_type: rbac, operation: list_env, extracted_params: {{bu_name: VCD-DELHI-BU}}

User: "What envs are in BU AI Cloud BU?"
→ intent_detected: true, resource_type: rbac, operation: list_env, extracted_params: {{bu_name: AI Cloud BU}}

User: "Show all environments" or "List environments"
→ intent_detected: true, resource_type: rbac, operation: list_env, extracted_params: empty

User: "List business units" or "Show BUs" or "What BUs are available?"
→ intent_detected: true, resource_type: rbac, operation: list_bu, extracted_params: empty

User: "What business units exist?" or "Show me all business units"
→ intent_detected: true, resource_type: rbac, operation: list_bu, extracted_params: empty

User: "List all BUs" or "Show available business units"
→ intent_detected: true, resource_type: rbac, operation: list_bu, extracted_params: empty

**Endpoint/Datacenter Listing Examples:**

User: "What are the available endpoints?" or "List endpoints"
→ intent_detected: true, resource_type: endpoint, operation: list, extracted_params: empty

User: "Show me all datacenters" or "What datacenters are available?"
→ intent_detected: true, resource_type: endpoint, operation: list, extracted_params: empty

User: "What DCs do we have?" or "List all DCs"
→ intent_detected: true, resource_type: endpoint, operation: list, extracted_params: empty

User: "Show me the locations" or "What locations are available?"
→ intent_detected: true, resource_type: endpoint, operation: list, extracted_params: empty

User: "Where can I deploy?" or "What data centers can I use?"
→ intent_detected: true, resource_type: endpoint, operation: list, extracted_params: empty

User: "List all available data centers" or "Show available locations"
→ intent_detected: true, resource_type: endpoint, operation: list, extracted_params: empty

**Important Notes:**
- For "list" operation on k8s_cluster, kafka, gitlab, container_registry, jenkins, postgres, documentdb, firewall: "endpoints" parameter is required (data center selection)
- For "list" operation on vm: NO parameters required (lists all VMs), but can optionally extract endpoint, zone, or department for filtering
- For "list" operation on endpoint (or aliases: datacenter, dc, data center, location), just fetch all available endpoints
- k8s_cluster list supports FILTERING by BU/Environment/Zone - if user asks to "filter clusters by BU", "filter clusters by environment", or "filter clusters by zone", detect as k8s_cluster list operation
- IMPORTANT: "Show environments in BU" or "List environments" is NOT k8s_cluster list! This is rbac.list_env operation
- "Show environments" = rbac.list_env (listing environments)
- "Show clusters" or "Filter clusters" = k8s_cluster.list (listing/filtering clusters)
- CRITICAL RBAC DISTINCTION:
  * "role bindings" or "user access" = rbac.list (shows who has access)
  * "environments" = rbac.list_env (shows available environments)
  * "Show role bindings in ENV_NAME" = rbac.list with env_name param (NOT list_env!)
  * "Show environments in BU_NAME" = rbac.list_env with bu_name param
- Do NOT extract location names (like "Mumbai", "Delhi") for cluster/service/firewall operations - the ValidationAgent will handle matching locations to endpoint IDs
- Do NOT extract BU/Environment/Zone names for filtering - the K8sClusterAgent will extract and match them from the user query
- For VM operations, you CAN extract location/zone/department names as they are used as filters, not required parameters
- Just detect the intent and operation; ValidationAgent/K8sClusterAgent will intelligently match locations/filters from the user query
- ANY query asking about viewing/counting/listing actual resources (not concepts) should be detected as a list operation
- "What are the clusters?" = list operation (showing actual clusters)
- "What is a cluster?" = NOT a list operation (this would be a documentation question, but you won't see it as it's routed elsewhere)
- Endpoint aliases: datacenter, dc, data center, location, datacenters, data centers, locations, endpoints, dcs
- Kafka aliases: kafka, kafka service, kafka services, apache kafka
- GitLab aliases: gitlab, gitlab service, gitlab services, git lab
- Container Registry aliases: container registry, registry, registries, docker registry, image registry
- Jenkins aliases: jenkins, jenkins service, jenkins services, ci cd, continuous integration
- PostgreSQL aliases: postgres, postgresql, postgres service, postgresql database, pg
- DocumentDB aliases: documentdb, document db, mongodb, mongo, nosql database
- VM aliases: vm, vms, virtual machine, virtual machines, instance, instances, server, servers
- Firewall aliases: firewall, firewalls, fw, vayu firewall, network firewall
- Reports aliases: report, reports, common cluster report, common cluster, cluster inventory report, cluster report, cluster inventory, cluster compute report, compute report, cluster compute, storage inventory report, storage report, pvc report
- RBAC aliases: rbac, role binding, role bindings, access control, user access, permissions, cluster roles, user roles, who has access, access for user

Be precise in detecting intent and operation. Only extract parameters that you can accurately determine (like names, counts, versions) - do NOT extract parameters that require lookup or matching (like endpoints or locations)."""
        
        return prompt
    
    def _get_resources_info(self) -> str:
        """Get formatted information about available resources."""
        resources = self.resource_schema.get("resources", {})
        info_lines = []
        
        for resource_type, config in resources.items():
            operations = config.get("operations", [])
            info_lines.append(f"- {resource_type}: {', '.join(operations)}")
        
        return "\n".join(info_lines) if info_lines else "No resources configured"
    
    def get_tools(self) -> List[Tool]:
        """Return tools for intent agent."""
        return [
            Tool(
                name="get_resource_schema",
                func=self._get_resource_schema,
                description=(
                    "Get the schema for a specific resource type including "
                    "required and optional parameters. Input: resource_type"
                )
            ),
            Tool(
                name="extract_parameters",
                func=self._extract_parameters,
                description=(
                    "Extract parameters from user input text. "
                    "Input: JSON with user_text and resource_type"
                )
            ),
            Tool(
                name="validate_operation",
                func=self._validate_operation,
                description=(
                    "Check if an operation is valid for a resource type. "
                    "Input: JSON with resource_type and operation"
                )
            )
        ]
    
    def _get_resource_schema(self, resource_type: str) -> str:
        """Get schema for a resource type."""
        try:
            config = api_executor_service.get_resource_config(resource_type)
            if not config:
                return f"Resource type '{resource_type}' not found"
            
            return json.dumps(config, indent=2)
        except Exception as e:
            return f"Error getting resource schema: {str(e)}"
    
    def _extract_parameters(self, input_json: str) -> str:
        """Extract parameters from user input."""
        try:
            data = json.loads(input_json)
            user_text = data.get("user_text", "")
            resource_type = data.get("resource_type", "")
            
            # Get parameter definitions
            config = api_executor_service.get_resource_config(resource_type)
            if not config:
                return json.dumps({"error": "Resource type not found"})
            
            # Simple parameter extraction (can be enhanced with NER)
            extracted = {}
            
            # Extract common patterns
            # Name patterns: "named X", "name is X", "called X"
            name_match = re.search(r'(?:named|name is|called)\s+([a-zA-Z0-9-_]+)', user_text, re.IGNORECASE)
            if name_match:
                extracted["name"] = name_match.group(1)
            
            # Count/number patterns: "3 nodes", "5 workers"
            count_match = re.search(r'(\d+)\s+(?:nodes?|workers?|instances?)', user_text, re.IGNORECASE)
            if count_match:
                extracted["node_count"] = int(count_match.group(1))
            
            # Region patterns: "in us-east-1", "region us-west-2"
            region_match = re.search(r'(?:in|region)\s+([a-z]+-[a-z]+-\d+)', user_text, re.IGNORECASE)
            if region_match:
                extracted["region"] = region_match.group(1)
            
            return json.dumps(extracted, indent=2)
            
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    def _validate_operation(self, input_json: str) -> str:
        """Validate if operation is supported for resource."""
        try:
            data = json.loads(input_json)
            resource_type = data.get("resource_type", "")
            operation = data.get("operation", "")
            
            config = api_executor_service.get_resource_config(resource_type)
            if not config:
                return json.dumps({"valid": False, "error": "Resource type not found"})
            
            operations = config.get("operations", [])
            valid = operation in operations
            
            return json.dumps({
                "valid": valid,
                "supported_operations": operations
            })
            
        except Exception as e:
            return json.dumps({"valid": False, "error": str(e)})
    
    async def execute(
        self,
        input_text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute intent detection on user input.
        
        Args:
            input_text: User's message
            context: Additional context including session info
            
        Returns:
            Dict with intent detection result
        """
        try:
            logger.info(f"🎯 IntentAgent analyzing: {input_text[:100]}...")
            
            # Call parent execute to use LLM
            result = await super().execute(input_text, context)
            
            # Parse the LLM output as JSON
            output_text = result.get("output", "")
            
            # Try to extract JSON from output
            intent_data = self._parse_intent_output(output_text)
            
            # Add intent data to result
            result["intent_detected"] = intent_data.get("intent_detected", False)
            result["intent_data"] = intent_data
            
            # If intent detected, get required parameters from schema
            if intent_data.get("intent_detected"):
                resource_type = intent_data.get("resource_type")
                operation = intent_data.get("operation")
                
                # Handle multi-resource: if resource_type is a list, convert to string
                # For param lookup, use the first resource type
                if isinstance(resource_type, list):
                    logger.info(f"🔧 Multi-resource detected: {resource_type}")
                    # For parameter schema, use the first resource
                    lookup_resource_type = resource_type[0] if resource_type else None
                else:
                    lookup_resource_type = resource_type
                
                if lookup_resource_type and operation:
                    operation_config = api_executor_service.get_operation_config(
                        lookup_resource_type, operation
                    )
                    
                    if operation_config:
                        params = operation_config.get("parameters", {})
                        intent_data["required_params"] = params.get("required", [])
                        intent_data["optional_params"] = params.get("optional", [])
                        
                        logger.info(
                            f"✅ Intent detected: {operation} {resource_type} | "
                            f"Required params: {intent_data['required_params']}"
                        )
                    else:
                        logger.warning(f"⚠️ No operation config found for {lookup_resource_type}.{operation}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Intent detection failed: {str(e)}")
            return {
                "agent_name": self.agent_name,
                "success": False,
                "error": str(e),
                "intent_detected": False,
                "output": f"Failed to detect intent: {str(e)}"
            }
    
    def _parse_intent_output(self, output_text: str) -> Dict[str, Any]:
        """
        Parse intent data from LLM output.
        
        Args:
            output_text: LLM output text
            
        Returns:
            Parsed intent data dict
        """
        try:
            # Try to find JSON objects in the output (non-greedy approach)
            # Find all potential JSON objects
            json_objects = []
            depth = 0
            start_idx = None
            
            for i, char in enumerate(output_text):
                if char == '{':
                    if depth == 0:
                        start_idx = i
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0 and start_idx is not None:
                        json_str = output_text[start_idx:i+1]
                        try:
                            obj = json.loads(json_str)
                            # Look for the main intent JSON (has intent_detected field)
                            if isinstance(obj, dict) and 'intent_detected' in obj:
                                logger.info(f"✅ Found valid intent JSON with intent_detected={obj.get('intent_detected')}")
                                return obj
                            json_objects.append(obj)
                        except json.JSONDecodeError:
                            pass
                        start_idx = None
            
            # If we found any JSON with resource_type and operation, use it
            for obj in json_objects:
                if isinstance(obj, dict) and obj.get('resource_type') and obj.get('operation'):
                    # Set intent_detected to True if we have resource_type and operation
                    obj['intent_detected'] = True
                    logger.info(f"✅ Found JSON with resource_type={obj.get('resource_type')}, operation={obj.get('operation')}")
                    return obj
            
            # Fallback: Try to parse text format (e.g., "Intent Detected: true\nResource Type: rbac\n...")
            logger.info("📝 No valid intent JSON found, trying text format parsing")
            intent_data = self._parse_text_format(output_text)
            if intent_data.get("intent_detected") or (intent_data.get("resource_type") and intent_data.get("operation")):
                intent_data["intent_detected"] = True  # Ensure it's set
                logger.info(f"✅ Parsed text format: resource={intent_data.get('resource_type')}, op={intent_data.get('operation')}")
                return intent_data
            
            # Final fallback: return default structure
            return {
                "intent_detected": False,
                "resource_type": None,
                "operation": None,
                "extracted_params": {},
                "confidence": 0.0,
                "ambiguities": [],
                "clarification_needed": None
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to parse intent output: {str(e)}, trying text format")
            return self._parse_text_format(output_text)
    
    def _parse_text_format(self, output_text: str) -> Dict[str, Any]:
        """
        Parse intent data from text format output.
        Handles formats like:
        - "Intent Detected: true"
        - "**Intent Detected:** true" (markdown)
        - "- **intent_detected:** true" (bullet points)
        - "Resource Type: rbac"
        - "Operation: list"
        """
        result = {
                "intent_detected": False,
                "resource_type": None,
                "operation": None,
                "extracted_params": {},
                "confidence": 0.0,
            "ambiguities": [],
                "clarification_needed": None
            }
        
        # Strip markdown formatting first
        text_clean = re.sub(r'\*\*([^*]+)\*\*', r'\1', output_text)  # **bold** -> bold
        text_clean = re.sub(r'\*([^*]+)\*', r'\1', text_clean)       # *italic* -> italic
        text_clean = re.sub(r'^[-•]\s*', '', text_clean, flags=re.MULTILINE)  # Remove bullet points
        text_clean = re.sub(r'_([^_]+)_', r'\1', text_clean)         # _underline_ -> underline
        text_lower = text_clean.lower()
        
        logger.debug(f"🔍 Parsing text format (cleaned): {text_lower[:200]}...")
        
        # Check for intent detected - multiple patterns
        intent_patterns = [
            r'intent[_\s]*detected[:\s]*(\w+)',
            r'detected[:\s]*(\w+)',
        ]
        for pattern in intent_patterns:
            intent_match = re.search(pattern, text_lower)
            if intent_match and intent_match.group(1) in ('true', 'yes', '1'):
                result["intent_detected"] = True
                logger.info(f"✅ Text format: intent_detected=True")
                break
        
        # Extract resource type - multiple patterns
        resource_patterns = [
            r'resource[_\s]*type[:\s]*(\w+)',
            r'resource[:\s]*(\w+)',
        ]
        for pattern in resource_patterns:
            resource_match = re.search(pattern, text_lower)
            if resource_match:
                resource = resource_match.group(1)
                # Filter out common false positives
                if resource not in ['type', 'the', 'is', 'a', 'an']:
                    result["resource_type"] = resource
                    logger.info(f"✅ Text format: resource_type={result['resource_type']}")
                    break
        
        # Extract operation
        operation_match = re.search(r'operation[:\s]*(\w+)', text_lower)
        if operation_match:
            result["operation"] = operation_match.group(1)
            logger.info(f"✅ Text format: operation={result['operation']}")
        
        # Extract extracted_params - look for {key: value} patterns
        params_patterns = [
            r'extracted[_\s]*params[:\s]*\{([^}]+)\}',
            r'params[:\s]*\{([^}]+)\}',
        ]
        for pattern in params_patterns:
            params_match = re.search(pattern, text_lower)
            if params_match:
                params_str = params_match.group(1)
                # Parse simple key: value pairs
                for pair in params_str.split(','):
                    if ':' in pair:
                        key, value = pair.split(':', 1)
                        key = key.strip().strip('"\'')
                        value = value.strip().strip('"\'')
                        if key and value:
                            result["extracted_params"][key] = value
                if result["extracted_params"]:
                    logger.info(f"✅ Text format: extracted_params={result['extracted_params']}")
                break
        
        # Extract confidence
        confidence_match = re.search(r'confidence[:\s]*([\d.]+)', text_lower)
        if confidence_match:
            try:
                result["confidence"] = float(confidence_match.group(1))
            except ValueError:
                result["confidence"] = 1.0 if result["intent_detected"] else 0.0
        elif result["intent_detected"]:
            result["confidence"] = 1.0
        
        return result
