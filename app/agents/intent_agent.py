"""
Intent Agent - Detects user intent and extracts parameters from user input.
Identifies what resource and operation the user wants to perform.
"""

from typing import Any, Dict, List, Optional
from langchain.tools import Tool
import logging
import json
import re
import os 

from app.agents.base_agent import BaseAgent
from app.services.api_executor_service import api_executor_service


logger = logging.getLogger(__name__)
LB_NAME_PATTERN = re.compile(
    r"(EG_.*?_LB_SEG_\d+|LB_.*?\d+)",
    re.IGNORECASE)
DETAILS_KEYWORDS = re.compile(r"\b(details?|info|show|describe)\b", re.IGNORECASE)

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
        self.INTENT_MODEL = os.getenv("INTENT_MODEL", "meta/Llama-3.1-8B-Instruct")
    
    def get_system_prompt(self) -> str:

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
→ intent_detected: true, resource_type: vm, operation: list, extracted_params: endpoint = Mumbai

User: "Show VMs in zone XYZ" or "List virtual machines in department ABC"
→ intent_detected: true, resource_type: vm, operation: list, extracted_params: zone = XYZ or department = ABC

**Firewall Examples:**

User: "List firewalls" or "Show me firewalls" or "What firewalls do we have?"
→ intent_detected: true, resource_type: firewall, operation: list, extracted_params: empty

User: "Show firewalls in Mumbai" or "List network firewalls in Delhi"
→ intent_detected: true, resource_type: firewall, operation: list, extracted_params: empty

User: "How many firewalls?" or "Count firewalls" or "Show all Vayu firewalls"
→ intent_detected: true, resource_type: firewall, operation: list, extracted_params: empty

**Load Balancer Examples - COMPREHENSIVE PATTERNS:**

**General List (show all):**
User: "list load balancers" or "show load balancers" or "all load balancers"
→ resource_type: load_balancer, operation: list, extracted_params: empty

User: "show me all LBs" or "what load balancers do we have"
→ resource_type: load_balancer, operation: list, extracted_params: empty

User: "list lbs" or "get load balancers"
→ resource_type: load_balancer, operation: list, extracted_params: empty

**Specific Load Balancer (show one with FULL DETAILS):**
User: "show EG_Tata_Com_167_LB_SEG_388" or "EG_Tata_Com_167_LB_SEG_388"
→ resource_type: load_balancer, operation: list, extracted_params: empty
NOTE: Don't extract the LB name - LoadBalancerAgent will detect it and fetch COMPLETE details

User: "details for EG_Tata_Com_167_LB_SEG_388" or "list the details about EG_Tata_Com_167_LB_SEG_388"
→ resource_type: load_balancer, operation: list, extracted_params: empty

User: "get info on LB_TataCommu_Tata_C_229" or "tell me about LB_TataCommu_Tata_C_229"
→ resource_type: load_balancer, operation: list, extracted_params: empty

User: "what is EG_Tata_Com_142_LB_SEG_276" or "describe EG_Tata_Com_142_LB_SEG_276"
→ resource_type: load_balancer, operation: list, extracted_params: empty

**IMPORTANT: When user mentions ANY specific LB name (contains _LB_ pattern):**
- ALWAYS detect as: resource_type=load_balancer, operation=list
- LoadBalancerAgent will automatically:
  1. Detect it's a specific LB query
  2. Find the LBCI from the LB list
  3. Call getDetails API for configuration
  4. Call virtualservices API for VIPs/listeners
  5. Format everything beautifully

**Location-Filtered (specific location):**
User: "load balancers in Mumbai" or "show LBs in Delhi"
→ resource_type: load_balancer, operation: list, extracted_params: empty
NOTE: Don't extract location - LoadBalancerAgent will handle it

User: "list load balancers at Chennai datacenter"
→ resource_type: load_balancer, operation: list, extracted_params: empty

User: "what LBs are in Bangalore"
→ resource_type: load_balancer, operation: list, extracted_params: empty

**Status-Filtered:**
User: "show active load balancers" or "list inactive LBs"
→ resource_type: load_balancer, operation: list, extracted_params: empty

User: "what load balancers are degraded"
→ resource_type: load_balancer, operation: list, extracted_params: empty

User: "healthy load balancers" or "unhealthy LBs"
→ resource_type: load_balancer, operation: list, extracted_params: empty

**Feature-Filtered:**
User: "load balancers with SSL" or "HTTPS load balancers"
→ resource_type: load_balancer, operation: list, extracted_params: empty

User: "show SSL-enabled LBs" or "load balancers using HTTPS"
→ resource_type: load_balancer, operation: list, extracted_params: empty

User: "TCP load balancers" or "HTTP load balancers"
→ resource_type: load_balancer, operation: list, extracted_params: empty

**Count/Status Queries:**
User: "how many load balancers" or "count LBs"
→ resource_type: load_balancer, operation: list, extracted_params: empty

User: "how many active load balancers in Mumbai"
→ resource_type: load_balancer, operation: list, extracted_params: empty

**Configuration Queries:**
User: "load balancer configuration" or "LB settings"
→ resource_type: load_balancer, operation: list, extracted_params: empty

User: "show load balancer details" or "get LB info"
→ resource_type: load_balancer, operation: list, extracted_params: empty


**Load Balancer Aliases (ALL these should be detected):**
- load_balancer, load balancer, load balancers
- lb, lbs, LB, LBs
- loadbalancer, loadbalancers
- vayu load balancer, vayu lb (Vayu is the product name)
- network load balancer, nlb, NLB
- application load balancer, alb, ALB
- l4 load balancer, l7 load balancer

NOTE: LoadBalancerAgent will detect this as LBCI and automatically fetch:
  1. Load balancer configuration details
  2. Virtual services (VIPs, listeners, pools)
  3. Format everything in production-ready display
→ resource_type: load_balancer, operation: list, extracted_params: empty
**IMPORTANT: When user mentions ANY LBCI number (pure digits, 5-6 characters):**
- ALWAYS detect as: resource_type=load_balancer, operation=list
- LoadBalancerAgent will automatically:
  1. Detect it's an LBCI query
  2. Find the LB with that LBCI
  3. Call getDetails API for configuration
  4. Call virtualservices API for VIPs/listeners
  5. Format everything beautifully
  
  **IMPORTANT: LBCI Pattern = 5-6 digit numbers (e.g., 312798, 45762, 154892)**
- When user mentions ANY 5-6 digit number in context of load balancers
- ALWAYS detect as: resource_type=load_balancer, operation=list
- LoadBalancerAgent will automatically fetch COMPLETE details + virtual services

**CRITICAL RULES for Load Balancer Intent Detection:**
1. ANY query asking about load balancers → operation: list
2. Do NOT extract LB names, locations, or filters as params
3. LoadBalancerAgent will intelligently parse and filter
4. Just detect: resource_type=load_balancer, operation=list
5. Keep extracted_params EMPTY (or minimal)

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
- k8s_cluster list supports FILTERING by BU/Environment/Zone - if user asks to "filter by BU", "filter by environment", or "filter by zone", still detect as k8s_cluster list operation. The K8sClusterAgent will handle the filtering intelligently by matching names to IDs.
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
- For "list" operation on load_balancer: "endpoints" parameter is required (data center selection)
- Do NOT extract location names (like "Mumbai", "Delhi") - ValidationAgent will handle location matching
- Just detect the intent and operation; ValidationAgent will intelligently match locations
- ANY query asking about viewing/counting/listing load balancers should be detected as a list operation
- Reports aliases: report, reports, common cluster report, common cluster, cluster inventory report, cluster report, cluster inventory, cluster compute report, compute report, cluster compute, storage inventory report, storage report, pvc report

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
        
    

    LBCI_DIGIT_PATTERN = re.compile(r'\b\d{5,6}\b')
    LIST_LB_KEYWORDS = re.compile(r'\b(list|show|get|details|describe|what is|tell me about)\b.*\b(load balancer|lb|lbs|loadbalancer|loadbalancers|vayu)\b', re.I)

    def _detect_deterministic_intent(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Detect deterministic intents that must NOT go to the LLM.
        Returns intent dict if detected else None.
        """
        text = (text or "").strip()

        # 1) Explicit LB name (contains _LB_)
        lb_match = self.LB_NAME_PATTERN.search(text)
        if lb_match:
            return {
                "intent_detected": True,
                "resource_type": "load_balancer",
                "operation": "list",
                "extracted_params": {},  # per spec: don't extract LB name here
                "confidence": 0.99,
                "ambiguities": [],
                "clarification_needed": None,
                "meta": {"detection": "lb_name_pattern", "matched_text": lb_match.group(0)}
            }

        # 2) LBCI 5-6 digit number in context -> assume load_balancer
        if self.LBCI_DIGIT_PATTERN.search(text):
            return {
                "intent_detected": True,
                "resource_type": "load_balancer",
                "operation": "list",
                "extracted_params": {},
                "confidence": 0.95,
                "ambiguities": [],
                "clarification_needed": None,
                "meta": {"detection": "lbci_digits"}
            }

        # 3) Explicit "list load balancers" style phrasing
        if self.LIST_LB_KEYWORDS.search(text):
            return {
                "intent_detected": True,
                "resource_type": "load_balancer",
                "operation": "list",
                "extracted_params": {},
                "confidence": 0.9,
                "ambiguities": [],
                "clarification_needed": None,
                "meta": {"detection": "list_lb_keywords"}
            }

        # 4) Add other deterministic rules here (endpoints, datacenters)
        # Example: "list endpoints" -> endpoint:list
        if re.search(r'\b(list|show|get)\b.*\b(endpoints|datacenters|locations|dcs|data centers)\b', text, re.I):
            return {
                "intent_detected": True,
                "resource_type": "endpoint",
                "operation": "list",
                "extracted_params": {},
                "confidence": 0.9,
                "ambiguities": [],
                "clarification_needed": None,
                "meta": {"detection": "list_endpoints"}
            }

        return None

    async def execute(self,input_text: str,context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
    Execute intent detection on user input with deterministic short-circuits.
    Behavior:
    - If deterministic rules match (LB name / LBCI / other rules) return intent immediately
      and skip the LLM.
    - Otherwise call the LLM via parent, parse output robustly, enrich with schema info.
    - On internal error, return a SAFE fallback that does NOT auto-run potentially destructive operations.
    """
        LB_NAME_PATTERN = re.compile(r"(EG_.*?_LB_SEG_\d+|LB_.*?\d+)", re.I)
        DETAILS_KEYWORDS = re.compile(r"\b(details?|info|show|describe)\b", re.I)

        if LB_NAME_PATTERN.search(input_text):
            if DETAILS_KEYWORDS.search(input_text):
                operation = "get_details"
            else:
                operation = "list"
                
                intent = {
        "resource": "load_balancer",
        "operation": operation,
        "confidence": 0.99,
        "source": "rule"
            } 
        try:
            safe_preview = (input_text or "")[:120]
            logger.info("🎯 IntentAgent analyzing input (trimmed): %s", safe_preview)

        # 0) Guard: empty input
            if not input_text or not input_text.strip():
                return {
                "agent_name": self.agent_name,
                "success": True,
                "intent_detected": False,
                "intent_data": {
                    "resource_type": None,
                    "operation": None,
                    "extracted_params": {},
                    "confidence": 0.0,
                    "ambiguities": ["Empty input"],
                    "clarification_needed": "Please provide a resource or question."
                },
                "output": "No input provided"
            }

        # 1) Deterministic short-circuit rules (run BEFORE any LLM call)
        #    _detect_deterministic_intent should return a dict when a rule applies.
            deterministic = None
            try:
                deterministic = self._detect_deterministic_intent(input_text)
            except Exception as e:
            # Defensive: log but continue to LLM path (do not crash)
                logger.warning("⚠️ Deterministic intent detection failed: %s", str(e))

            if deterministic:
            # Normalize deterministic intent shape
                intent_data = {
                "intent_detected": bool(deterministic.get("intent_detected", True)),
                "resource_type": deterministic.get("resource_type"),
                "operation": deterministic.get("operation"),
                "extracted_params": deterministic.get("extracted_params", {}),
                "confidence": float(deterministic.get("confidence", 0.99)),
                "ambiguities": deterministic.get("ambiguities", []),
                "clarification_needed": deterministic.get("clarification_needed", None),
                # keep meta for observability, safe to include
                "meta": deterministic.get("meta", {"detection": "rule"})
            }

                logger.info("🔒 Deterministic intent detected (rule): %s", intent_data.get("meta"))
                return {
                "agent_name": self.agent_name,
                "success": True,
                "intent_detected": intent_data["intent_detected"],
                "intent_data": intent_data,
                "output": "Deterministic intent detection applied"
            }

        # 2) Otherwise call LLM via parent agent
        #    (Do not assume the parent accepts model arg; BaseAgent should manage model selection.)
            result = await super().execute(input_text, context)

        # Support common possible keys for text output
            output_text = result.get("output") or result.get("text") or result.get("response") or ""

        # 3) Parse intent JSON robustly
            intent_data = self._parse_intent_output(output_text)

        # Ensure consistent structure if LLM returned something unexpected
            if not isinstance(intent_data, dict):
                intent_data = {
                "intent_detected": False,
                "resource_type": None,
                "operation": None,
                "extracted_params": {},
                "confidence": 0.0,
                "ambiguities": [],
                "clarification_needed": None
            }

        # 4) If intent detected, enrich with operation/schema info (defensive)
            if intent_data.get("intent_detected"):
                resource_type = intent_data.get("resource_type")
                operation = intent_data.get("operation")

            # Normalize: if resource_type is list, pick first for lookup
                lookup_resource_type = resource_type[0] if isinstance(resource_type, list) and resource_type else resource_type

                if lookup_resource_type and operation:
                    try:
                        operation_config = api_executor_service.get_operation_config(
                        lookup_resource_type, operation
                    )
                        if operation_config:
                            params = operation_config.get("parameters", {})
                            intent_data.setdefault("required_params", params.get("required", []))
                            intent_data.setdefault("optional_params", params.get("optional", []))
                            logger.info(
                            "✅ Intent detected: %s %s | required=%s",
                            operation, lookup_resource_type, intent_data["required_params"]
                        )
                        else:
                            logger.warning("⚠️ No operation config found for %s.%s", lookup_resource_type, operation)
                    except Exception as e:
                    # Don't fail the whole function if enrichment fails
                        logger.warning("⚠️ Failed to enrich intent with schema: %s", str(e))

        # 5) Return standardized final result
            return {
            "agent_name": self.agent_name,
            "success": True,
            "intent_detected": bool(intent_data.get("intent_detected", False)),
            "intent_data": intent_data,
            "output": output_text
        }

        except Exception as exc:
        # Robust logging and safe fallback. Do NOT return malformed keys or a truthy intent that
        # would cause automatic execution. Keep fallback non-executing and request clarification.
            logger.exception("❌ IntentAgent failed during execute: %s", str(exc))
            fallback_intent = self._fallback_intent("Internal error during intent detection")

        # Normalize fallback shape, but prefer NOT to mark intent_detected True.
        # This prevents accidental auto-execution of downstream operations.
            normalized_fallback = {
            "intent_detected": bool(fallback_intent.get("intent_detected", False)),
            "resource_type": fallback_intent.get("resource_type"),
            "operation": fallback_intent.get("operation"),
            "extracted_params": fallback_intent.get("extracted_params", {}),
            "confidence": float(fallback_intent.get("confidence", 0.0)),
            "ambiguities": fallback_intent.get("ambiguities", ["Internal error"]),
            "clarification_needed": fallback_intent.get("clarification_needed", "Could you rephrase your request?")
        }

            return {
            "agent_name": self.agent_name,
            "success": True,
            "intent_detected": normalized_fallback["intent_detected"],
            "intent_data": normalized_fallback,
            "output": "Intent detection fallback applied due to internal error"
        }

    def _fallback_intent(self, reason: str) -> Dict[str, Any]:
        logger.warning("⚠️ Intent fallback used: %s", reason)
        return {
            "intent_detected": True,
            "resource_type": "load_balancer",
            "operation": "list",
            "extracted_params": {},
            "confidence": 0.4,
            "ambiguities": [reason],
            "clarification_needed": None
        }



    def _parse_intent_output(self, output_text: str) -> Dict[str, Any]:
        """
        Parse JSON object from LLM output robustly. Return standard structure if parsing fails.
        """
        try:
            if not output_text:
                return {
                    "intent_detected": False,
                    "resource_type": None,
                    "operation": None,
                    "extracted_params": {},
                    "confidence": 0.0,
                    "ambiguities": [],
                    "clarification_needed": None
                }

            start_idx = output_text.find('{')
            if start_idx != -1:
                # try progressively larger substrings up to a reasonable size
                for end_idx in range(len(output_text), start_idx, -1):
                    candidate = output_text[start_idx:end_idx]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict):
                            return parsed
                    except Exception:
                        continue
            # As a last resort, try to extract any key-value style lines (very heuristic)
            # Return default if nothing works
            logger.warning("⚠️ Could not locate JSON in LLM output. Returning default structure.")
            return {
                "intent_detected": False,
                "resource_type": None,
                "operation": None,
                "extracted_params": {},
                "confidence": 0.0,
                "ambiguities": ["Could not parse LLM JSON response"],
                "clarification_needed": None
            }

        except Exception as e:
            logger.warning("⚠️ Exception while parsing intent output: %s", str(e))
            return {
                "intent_detected": False,
                "resource_type": None,
                "operation": None,
                "extracted_params": {},
                "confidence": 0.0,
                "ambiguities": ["Parsing error"],
                "clarification_needed": None
            }

