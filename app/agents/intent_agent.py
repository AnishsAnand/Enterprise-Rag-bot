"""
Intent Agent - Detects user intent and extracts parameters from user input.
Identifies what resource and operation the user wants to perform.

Phase 2: RAG-driven API discovery - queries search_api_specs() for API context
and uses it to augment intent detection and param enrichment.
"""

from typing import Any, Dict, List, Optional, Tuple
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

# Module-level cache: resource_type -> list of operations (populated from RAG at execute time)
_rag_resource_index: Dict[str, List[str]] = {}

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
            temperature=0.1)
        # Load resource schema (empty when using RAG-only Phase 3)
        self.resource_schema = api_executor_service.resource_schema
        # Setup agent
        self.setup_agent()
        self.INTENT_MODEL = os.getenv("INTENT_MODEL", "meta/Llama-3.1-8B-Instruct")
    
    def get_system_prompt(self) -> str:
        resources_info = self._get_resources_info()
        prompt = """You are the Intent Agent, specialized in detecting user intent for cloud resource operations.
**Available Resources:**
""" + resources_info + """

**When RAG API specs are provided in Context:** Use them to inform your intent detection. The specs describe available APIs, operations, parameters, and workflows. Prefer matching the user's request to the RAG specs when relevant.

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

**CRITICAL - Standardized Parameter Names:**
Always use these EXACT parameter names in extracted_params (never use synonyms):
- **size**: For number of records/rows/items to return (NOT: limit, count, record_count, num_records, rows, total)
- **page**: For pagination page number (NOT: page_number, offset)
- **cluster_name**: For cluster names (NOT: clusterName, name, cluster)
- **report_type**: For report types (use: common_cluster, cluster_inventory, cluster_compute, storage_inventory)
- **endpoint**: For datacenter/location (NOT: location, datacenter, dc)

Example: User says "show 30 records" or "limit to 50" or "display 100 rows"
→ ALWAYS extract as: extracted_params: {{size: 30}} or {{size: 50}} or {{size: 100}}

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

User: "Show 30 records of cluster report" or "Display 50 rows of common cluster report"
→ intent_detected: true, resource_type: reports, operation: list, extracted_params: {{report_type: cluster_inventory, size: 30}} or {{report_type: common_cluster, size: 50}}
NOTE: ALWAYS use "size" for record count, never "limit", "count", "record_count", etc.

**Engagement Listing Examples (NOT business_unit):**

User: "List engagements" or "Show engagements" or "What engagements do I have?"
→ intent_detected: true, resource_type: engagement, operation: list, extracted_params: empty

User: "List my engagements" or "Show all engagements" or "Which engagements are available?"
→ intent_detected: true, resource_type: engagement, operation: list, extracted_params: empty

NOTE: "engagement" = account/tenant selection (e.g. Tata Communications, Vayu Cloud). "business_unit" = department/BU within an engagement. Do NOT confuse "list engagements" with "list BUs".

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
        """Get formatted information about available resources (schema or RAG-derived)."""
        resources = self.resource_schema.get("resources", {})
        if resources:
            info_lines = []
            for resource_type, config in resources.items():
                operations = config.get("operations", [])
                info_lines.append(f"- {resource_type}: {', '.join(operations)}")
            return "\n".join(info_lines)
        # Phase 3: RAG-only mode - use index populated from RAG at execute time
        if _rag_resource_index:
            return "\n".join(
                f"- {r}: {', '.join(ops)}" for r, ops in sorted(_rag_resource_index.items())
            )
        return "Resources are discovered from RAG API specs at runtime. See Context for available resources and operations."

    def _parse_resource_operation_from_chunk(self, content: str) -> Optional[Tuple[str, str]]:
        """Extract (resource, operation) from RAG chunk content with **Resource:** and **Operation:**."""
        if not content:
            return None
        res = re.search(r"\*\*Resource:\*\*\s*([a-zA-Z0-9_]+)", content, re.IGNORECASE)
        op = re.search(r"\*\*Operation:\*\*\s*([a-zA-Z0-9_]+)", content, re.IGNORECASE)
        if res and op:
            return (res.group(1).lower(), op.group(1).lower())
        return None

    async def _ensure_rag_resource_index(self) -> None:
        """
        Populate RAG resource index from API spec chunks (no hardcoded resources).
        Called at execute start so tools have resource→operations mapping.
        """
        global _rag_resource_index
        if _rag_resource_index:
            return
        try:
            from app.services.postgres_service import postgres_service
            if not postgres_service.pool:
                await postgres_service.initialize()
            if not postgres_service.pool:
                return
            results = await postgres_service.search_api_specs(
                "API specification resource list create read update delete", n_results=50
            )
            index: Dict[str, List[str]] = {}
            for r in results:
                content = r.get("content", "")
                parsed = self._parse_resource_operation_from_chunk(content)
                if parsed:
                    resource, op = parsed
                    if resource not in index:
                        index[resource] = []
                    if op not in index[resource]:
                        index[resource].append(op)
            _rag_resource_index = index
            if index:
                logger.info(f"📚 RAG resource index populated: {len(index)} resources")
        except Exception as e:
            logger.debug(f"RAG resource index failed: {e}")

    async def _fetch_rag_api_specs(self, user_input: str) -> List[Dict[str, Any]]:
        """
        Query RAG for API specs relevant to user input (Phase 2).
        Returns list of search results with content, metadata.
        """
        try:
            from app.services.postgres_service import postgres_service
            if not postgres_service.pool:
                await postgres_service.initialize()
            if postgres_service.pool:
                # Build query: user input + "API" to bias toward API spec chunks
                query = f"{user_input} API" if user_input.strip() else "API"
                results = await postgres_service.search_api_specs(query, n_results=5)
                if results:
                    logger.info(f"📚 RAG retrieved {len(results)} API spec chunks for intent")
                return results or []
        except Exception as e:
            logger.debug(f"RAG API spec fetch skipped: {e}")
        return []

    def _parse_params_from_rag_content(self, content: str) -> Dict[str, List[str]]:
        """Extract required and optional params from RAG markdown content."""
        result = {"required": [], "optional": []}
        if not content:
            return result
        req_section = re.search(r"## Required Parameters\s*\n(.*?)(?=##|$)", content, re.DOTALL | re.IGNORECASE)
        if req_section:
            result["required"] = re.findall(r"`([a-zA-Z0-9_]+)`", req_section.group(1))
        opt_section = re.search(r"## Optional Parameters\s*\n(.*?)(?=##|$)", content, re.DOTALL | re.IGNORECASE)
        if opt_section:
            result["optional"] = re.findall(r"`([a-zA-Z0-9_]+)`", opt_section.group(1))
        return result

    def _find_matching_rag_chunk(self, results: List[Dict], resource_type: str, operation: str) -> Optional[Dict]:
        """Find RAG chunk that matches resource_type and operation."""
        resource_type = (resource_type or "").lower()
        operation = (operation or "").lower()
        for r in results:
            content = r.get("content", "")
            meta = r.get("metadata", {})
            title = (meta.get("title") or "").lower()
            if resource_type in content.lower() and operation in content.lower():
                return r
            if resource_type in title and operation in title:
                return r
        return results[0] if results else None

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
                    "Input: JSON with resource_type and operation")) ]

    def _get_resource_schema(self, resource_type: str) -> str:
        """Get schema for a resource type (from schema or RAG)."""
        try:
            config = api_executor_service.get_resource_config(resource_type)
            if config:
                return json.dumps(config, indent=2)
            # Phase 3: Schema empty - API specs are in RAG; return helpful message
            return f"Resource '{resource_type}' - API specs are loaded from RAG. Use intent detection for operation and params."
        except Exception as e:
            return f"Error getting resource schema: {str(e)}"
    
    def _extract_parameters(self, input_json: str) -> str:
        """Extract parameters from user input."""
        try:
            data = json.loads(input_json)
            user_text = data.get("user_text", "")
            resource_type = data.get("resource_type", "")
            config = api_executor_service.get_resource_config(resource_type)
            extracted = {}
            # Simple parameter extraction (can be enhanced with NER)
            # Extract common patterns
            # Name patterns
            name_match = re.search(r'(?:named|name is|called)\s+([a-zA-Z0-9-_]+)', user_text, re.IGNORECASE)
            if name_match:
                extracted["name"] = name_match.group(1)
            # Count/number patterns
            count_match = re.search(r'(\d+)\s+(?:nodes?|workers?|instances?)', user_text, re.IGNORECASE)
            if count_match:
                extracted["node_count"] = int(count_match.group(1))
            # Region patterns
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
            resource_type = (data.get("resource_type", "") or "").lower()
            operation = (data.get("operation", "") or "").lower()
            config = api_executor_service.get_resource_config(resource_type)
            if not config:
                # Phase 3: Use RAG-derived operations (no hardcoded list)
                operations = _rag_resource_index.get(resource_type, [])
                # engagement: list and get are equivalent (same API returns engagements list)
                if resource_type == "engagement" and operation == "list" and "get" in operations:
                    valid = True
                else:
                    valid = operation in operations if operations else False
                return json.dumps({"valid": valid, "supported_operations": operations})
            operations = config.get("operations", [])
            valid = operation in operations
            return json.dumps({
                "valid": valid,
                "supported_operations": operations})
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

        # 1) Explicit LB name (contains _LB_) - use module-level pattern
        lb_match = LB_NAME_PATTERN.search(text)
        if lb_match:
            return {
                "intent_detected": True,
                "resource_type": "load_balancer",
                "operation": "list",
                "extracted_params": {}, 
                "confidence": 0.99,
                "ambiguities": [],
                "clarification_needed": None,
                "meta": {"detection": "lb_name_pattern", "matched_text": lb_match.group(0)}}

        if self.LBCI_DIGIT_PATTERN.search(text):
            return {
                "intent_detected": True,
                "resource_type": "load_balancer",
                "operation": "list",
                "extracted_params": {},
                "confidence": 0.95,
                "ambiguities": [],
                "clarification_needed": None,
                "meta": {"detection": "lbci_digits"} }

        if self.LIST_LB_KEYWORDS.search(text):
            return {
                "intent_detected": True,
                "resource_type": "load_balancer",
                "operation": "list",
                "extracted_params": {},
                "confidence": 0.9,
                "ambiguities": [],
                "clarification_needed": None,
                "meta": {"detection": "list_lb_keywords"}}

        if re.search(r'\b(list|show|get)\b.*\b(endpoints|datacenters|locations|dcs|data centers)\b', text, re.I):
            return {
                "intent_detected": True,
                "resource_type": "endpoint",
                "operation": "list",
                "extracted_params": {},
                "confidence": 0.9,
                "ambiguities": [],
                "clarification_needed": None,
                "meta": {"detection": "list_endpoints"}}
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
        await self._ensure_rag_resource_index()
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
                "meta": deterministic.get("meta", {"detection": "rule"})}
                logger.info("🔒 Deterministic intent detected (rule): %s", intent_data.get("meta"))
                return {
                "agent_name": self.agent_name,
                "success": True,
                "intent_detected": intent_data["intent_detected"],
                "intent_data": intent_data,
                "output": "Deterministic intent detection applied"}

        # 2) RAG-first: fetch API specs to augment intent (Phase 2)
            rag_results = await self._fetch_rag_api_specs(input_text)
            exec_context = dict(context) if context else {}
            if rag_results:
                rag_snippets = []
                for i, r in enumerate(rag_results[:5]):
                    content = (r.get("content") or "")[:1200]
                    if content:
                        rag_snippets.append(f"--- API Spec {i+1} ---\n{content}")
                if rag_snippets:
                    exec_context["rag_api_specs"] = "\n\n".join(rag_snippets)
                    logger.info(f"📚 Including {len(rag_snippets)} RAG API specs in intent prompt")
            # 3) Call LLM via parent agent with RAG context
            result = await super().execute(input_text, exec_context)
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
                "clarification_needed": None }
        # 4) If intent detected, enrich with params from RAG (preferred) or schema (fallback)
            if intent_data.get("intent_detected"):
                resource_type = intent_data.get("resource_type")
                operation = intent_data.get("operation")
                lookup_resource_type = resource_type[0] if isinstance(resource_type, list) and resource_type else resource_type
                if lookup_resource_type and operation:
                    # Prefer RAG when we have matching API spec
                    rag_chunk = self._find_matching_rag_chunk(rag_results, lookup_resource_type, operation) if rag_results else None
                    if rag_chunk:
                        content = rag_chunk.get("content", "")
                        parsed = self._parse_params_from_rag_content(content)
                        if parsed["required"] or parsed["optional"]:
                            intent_data.setdefault("required_params", parsed["required"])
                            intent_data.setdefault("optional_params", parsed["optional"])
                            intent_data["api_spec"] = content[:2000]  # Store for downstream agents
                            logger.info(
                                "✅ Intent enriched from RAG: %s %s | required=%s",
                                operation, lookup_resource_type, intent_data["required_params"])
                    # Fallback to schema when RAG has no params
                    if not intent_data.get("required_params") and not intent_data.get("optional_params"):
                        try:
                            operation_config = api_executor_service.get_operation_config(
                                lookup_resource_type, operation)
                            if operation_config:
                                params = operation_config.get("parameters", {})
                                intent_data.setdefault("required_params", params.get("required", []))
                                intent_data.setdefault("optional_params", params.get("optional", []))
                                logger.info(
                                    "✅ Intent enriched from schema: %s %s | required=%s",
                                    operation, lookup_resource_type, intent_data["required_params"])
                            else:
                                logger.warning("⚠️ No operation config found for %s.%s", lookup_resource_type, operation)
                        except Exception as e:
                            logger.warning("⚠️ Failed to enrich intent with schema: %s", str(e))
        # 5) Return standardized final result
            return {
            "agent_name": self.agent_name,
            "success": True,
            "intent_detected": bool(intent_data.get("intent_detected", False)),
            "intent_data": intent_data,
            "output": output_text}
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
            "clarification_needed": fallback_intent.get("clarification_needed", "Could you rephrase your request?") }
            return {
            "agent_name": self.agent_name,
            "success": True,
            "intent_detected": normalized_fallback["intent_detected"],
            "intent_data": normalized_fallback,
            "output": "Intent detection fallback applied due to internal error"}

    def _fallback_intent(self, reason: str) -> Dict[str, Any]:
        """
        Return a safe fallback when intent detection fails.
        IMPORTANT: Do NOT default to any specific resource type - this was causing
        random LB queries for unrelated resources like k8s_cluster.
        """
        logger.warning("⚠️ Intent fallback used: %s", reason)
        return {
            "intent_detected": False,  # Changed from True - don't auto-execute on failure
            "resource_type": None,     # Changed from "load_balancer" - don't assume resource
            "operation": None,         # Changed from "list" - don't assume operation
            "extracted_params": {},
            "confidence": 0.0,         # Changed from 0.4 - no confidence on fallback
            "ambiguities": [reason],
            "clarification_needed": "I couldn't understand your request. Could you please rephrase?"
        }



    def _parse_intent_output(self, output_text: str) -> Dict[str, Any]:
        """
        Parse JSON object from LLM output robustly. Return standard structure if parsing fails.
        Handles JSON wrapped in markdown code blocks (```json ... ```)
        """
        default_result = {
            "intent_detected": False,
            "resource_type": None,
            "operation": None,
            "extracted_params": {},
            "confidence": 0.0,
            "ambiguities": [],
            "clarification_needed": None
        }
        
        try:
            if not output_text:
                return default_result

            # Method 1: Try to extract JSON from markdown code blocks first
            # Handles ```json { ... } ``` or ``` { ... } ```
            code_block_pattern = re.compile(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', re.IGNORECASE)
            code_match = code_block_pattern.search(output_text)
            if code_match:
                try:
                    parsed = json.loads(code_match.group(1))
                    if isinstance(parsed, dict):
                        logger.debug("✅ Parsed JSON from markdown code block")
                        return parsed
                except json.JSONDecodeError:
                    pass  # Continue to other methods

            # Method 2: Find balanced braces - look for complete JSON object
            # More efficient than iterating backwards character by character
            brace_depth = 0
            start_idx = -1
            for i, char in enumerate(output_text):
                if char == '{':
                    if brace_depth == 0:
                        start_idx = i
                    brace_depth += 1
                elif char == '}':
                    brace_depth -= 1
                    if brace_depth == 0 and start_idx != -1:
                        candidate = output_text[start_idx:i+1]
                        try:
                            parsed = json.loads(candidate)
                            # IMPORTANT: Validate parsed JSON has expected intent keys
                            # This prevents matching empty {} or unrelated JSON fragments
                            if isinstance(parsed, dict) and ("intent_detected" in parsed or "resource_type" in parsed):
                                logger.debug("✅ Parsed JSON using brace matching")
                                return parsed
                        except json.JSONDecodeError:
                            pass
                        # Reset and continue looking for another JSON block
                        start_idx = -1
                        continue

            # Method 3: Last resort - try original approach with { to end
            start_idx = output_text.find('{')
            if start_idx != -1:
                # Find the last } and try that substring
                end_idx = output_text.rfind('}')
                if end_idx > start_idx:
                    candidate = output_text[start_idx:end_idx+1]
                    try:
                        parsed = json.loads(candidate)
                        # IMPORTANT: Validate parsed JSON has expected intent keys
                        if isinstance(parsed, dict) and ("intent_detected" in parsed or "resource_type" in parsed):
                            logger.debug("✅ Parsed JSON using first-last brace method")
                            return parsed
                    except json.JSONDecodeError:
                        pass

            # Method 4: Parse markdown bullet point format (fallback for non-JSON LLM responses)
            # Handles output like:
            # - intent_detected: true
            # - resource_type: k8s_cluster
            # - operation: list
            logger.info("🔍 JSON parsing failed, trying bullet point format fallback...")
            bullet_result = self._parse_bullet_format(output_text)
            if bullet_result and bullet_result.get("intent_detected"):
                logger.info(f"✅ Parsed intent from markdown bullet format: {bullet_result.get('resource_type')}/{bullet_result.get('operation')}")
                return bullet_result

            logger.warning("⚠️ Could not locate JSON in LLM output. Returning default structure.")
            default_result["ambiguities"] = ["Could not parse LLM JSON response"]
            return default_result

        except Exception as e:
            logger.warning("⚠️ Exception while parsing intent output: %s", str(e))
            default_result["ambiguities"] = ["Parsing error"]
            return default_result

    def _parse_bullet_format(self, output_text: str) -> Optional[Dict[str, Any]]:
        """
        Parse markdown bullet point format as fallback when LLM doesn't return JSON.
        Handles output like:
        - **intent_detected**: true
        - **resource_type**: k8s_cluster
        - **operation**: list
        Or plain format:
        - intent_detected: true
        - resource_type: k8s_cluster
        """
        try:
            result = {
                "intent_detected": False,
                "resource_type": None,
                "operation": None,
                "extracted_params": {},
                "confidence": 0.0,
                "ambiguities": [],
                "clarification_needed": None
            }
            
            # Look for bullet point patterns - handle both bold (**key**:) and plain (key:) formats
            # Pattern explanation: -?\s* = optional dash and whitespace, \*{0,2} = 0-2 asterisks for bold
            patterns = {
                "intent_detected": r'-?\s*\*{0,2}intent_detected\*{0,2}[:\s]+(\w+)',
                "resource_type": r'-?\s*\*{0,2}resource_type\*{0,2}[:\s]+(\w+)',
                "operation": r'-?\s*\*{0,2}operation\*{0,2}[:\s]+(\w+)',
                "confidence": r'-?\s*\*{0,2}confidence\*{0,2}[:\s]+([\d.]+)',
            }
            
            for key, pattern in patterns.items():
                match = re.search(pattern, output_text, re.IGNORECASE)
                if match:
                    value = match.group(1).strip()
                    if key == "intent_detected":
                        result[key] = value.lower() == "true"
                    elif key == "confidence":
                        try:
                            result[key] = float(value)
                        except ValueError:
                            result[key] = 0.8 if result.get("intent_detected") else 0.0
                    else:
                        result[key] = value
            
            # Check for extracted_params: {} or similar - handle bold format too
            params_match = re.search(r'\*{0,2}extracted_params\*{0,2}[:\s]*\{([^}]*)\}', output_text, re.IGNORECASE)
            if params_match:
                params_str = params_match.group(1).strip()
                if params_str:
                    # Try to parse as JSON-like key:value pairs
                    # First, try to fix common issues: unquoted keys/values
                    try:
                        result["extracted_params"] = json.loads("{" + params_str + "}")
                    except json.JSONDecodeError:
                        # Try to fix unquoted keys: cluster_name: "blr-paas" -> "cluster_name": "blr-paas"
                        fixed_params = re.sub(r'(\w+):', r'"\1":', params_str)
                        # Also quote unquoted string values
                        fixed_params = re.sub(r':\s*([a-zA-Z][\w-]*)\s*([,}]|$)', r': "\1"\2', fixed_params)
                        try:
                            result["extracted_params"] = json.loads("{" + fixed_params + "}")
                        except json.JSONDecodeError:
                            logger.debug(f"⚠️ Could not parse extracted_params: {params_str}")
            
            # Only return if we found meaningful data
            if result.get("resource_type") and result.get("operation"):
                logger.info(f"✅ Bullet parse result: resource={result['resource_type']}, op={result['operation']}, params={result.get('extracted_params')}")
                return result
            
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ Bullet format parsing failed: {e}")
            return None

