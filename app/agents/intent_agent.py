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
1. **Identify the resource type** the user wants to work with (k8s_cluster, firewall, kafka, gitlab, etc.)
2. **Identify the operation** (create, read, update, delete, list)
3. **Extract parameters** from the user's message
4. **Return structured JSON** with your findings

**Output Format:**
Always respond with a JSON object containing:
- intent_detected: boolean (true/false)
- resource_type: string (k8s_cluster, firewall, kafka, gitlab, etc.)
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
- For "list" operation on k8s_cluster, kafka, gitlab: "endpoints" parameter is required (data center selection)
- For "list" operation on endpoint (or aliases: datacenter, dc, data center, location), just fetch all available endpoints
- Do NOT extract location names (like "Mumbai", "Delhi") - the ValidationAgent will handle matching locations to endpoint IDs
- Just detect the intent and operation; ValidationAgent will intelligently match locations from the user query
- ANY query asking about viewing/counting/listing actual resources (not concepts) should be detected as a list operation
- "What are the clusters?" = list operation (showing actual clusters)
- "What is a cluster?" = NOT a list operation (this would be a documentation question, but you won't see it as it's routed elsewhere)
- Endpoint aliases: datacenter, dc, data center, location, datacenters, data centers, locations, endpoints, dcs
- Kafka aliases: kafka, kafka service, kafka services, apache kafka
- GitLab aliases: gitlab, gitlab service, gitlab services, git lab

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
                
                if resource_type and operation:
                    operation_config = api_executor_service.get_operation_config(
                        resource_type, operation
                    )
                    
                    if operation_config:
                        params = operation_config.get("parameters", {})
                        intent_data["required_params"] = params.get("required", [])
                        intent_data["optional_params"] = params.get("optional", [])
                        
                        logger.info(
                            f"✅ Intent detected: {operation} {resource_type} | "
                            f"Required params: {len(intent_data['required_params'])}"
                        )
            
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
            # Try to find JSON in the output
            json_match = re.search(r'\{[\s\S]*\}', output_text)
            if json_match:
                intent_data = json.loads(json_match.group(0))
                return intent_data
            
            # Fallback: return default structure
            return {
                "intent_detected": False,
                "resource_type": None,
                "operation": None,
                "extracted_params": {},
                "confidence": 0.0,
                "ambiguities": [],
                "clarification_needed": None
            }
            
        except json.JSONDecodeError as e:
            logger.warning(f"⚠️ Failed to parse intent JSON: {str(e)}")
            return {
                "intent_detected": False,
                "resource_type": None,
                "operation": None,
                "extracted_params": {},
                "confidence": 0.0,
                "ambiguities": ["Failed to parse intent"],
                "clarification_needed": None
            }

