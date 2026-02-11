"""
Validation Agent - Validates parameters and collects missing information.
Ensures all required parameters are present and valid before execution.
"""

from typing import Any, Dict, List, Optional
from langchain.tools import Tool
import logging
import json
import re
from app.agents.base_agent import BaseAgent
from app.agents.state.conversation_state import conversation_state_manager, ConversationStatus
from app.services.api_executor_service import api_executor_service
from app.services.ai_service import ai_service
from app.agents.handlers.cluster_creation_handler import ClusterCreationHandler
from app.agents.tools.parameter_extraction import ParameterExtractor

logger = logging.getLogger(__name__)

class ValidationAgent(BaseAgent):
    """
    Agent specialized in parameter validation and collection.
    Validates parameters against schema, collects missing parameters conversationally.
    """

    def __init__(self):
        super().__init__(
            agent_name="ValidationAgent",
            agent_description=(
                "Validates parameters for CRUD operations and collects missing information. "
                "Ensures data quality and completeness before execution."
            ),
            temperature=0.2
        )
        # Initialize handlers and tools
        self.cluster_creation_handler = ClusterCreationHandler()
        self.param_extractor = ParameterExtractor()
        # Store auth token and user type for API calls
        self._current_auth_token = None
        self._current_user_type = None
        self._current_user_id = None
        self._current_engagement_id = None
        self.setup_agent()

    def get_system_prompt(self) -> str:
        return """You are the Validation Agent, responsible for ensuring all parameters are correct and complete.

**Your responsibilities:**
1. **Validate collected parameters** against schema rules
2. **Identify missing required parameters**
3. **Ask for missing information** in a conversational way
4. **Extract parameters** from user responses
5. **Provide helpful guidance** on parameter format and requirements
6. **Fetch available options dynamically** (endpoints, versions, etc.)
7. **Match user's natural language** to actual option values

**NEW CAPABILITIES:**
- Use `fetch_available_options` to get current data centers, versions, etc. from APIs
- Use `match_user_selection_to_options` to map user input like "delhi dc" to actual endpoint IDs
- NEVER hardcode location names or options - always fetch dynamically!
- Present actual available options to users - don't guess!

**Validation rules:**
- Check data types (string, integer, boolean, etc.)
- Validate string lengths (min/max)
- Check numeric ranges (min/max)
- Validate enum values
- Check regex patterns
- Ensure required parameters are present

**When asking for parameters:**
- Be conversational and friendly
- For options like data centers, FIRST fetch available options, THEN present them to user
- Explain why the parameter is needed
- Provide examples when helpful
- Ask for one or a few related parameters at a time (don't overwhelm user)
- If user provides partial information, acknowledge it and ask for remaining items

**Example interactions:**

Missing data center (SMART WAY):
"Let me check which data centers are available..."
[fetches endpoints dynamically]
"I found 5 data centers available:
- Delhi
- Bengaluru
- Mumbai-BKC
- Chennai-AMB
- Cressex

Which one would you like to use? You can also say 'all' to list clusters across all data centers."

User responds: "delhi dc"
[matches "delhi dc" to "Delhi" endpoint]
"Perfect! I'll use the Delhi data center."

Missing name:
"I'll help you create that Kubernetes cluster. What would you like to name it?
(Use lowercase letters, numbers, and hyphens only, 3-63 characters)"

Invalid parameter:
"The cluster name 'My_Cluster!' contains invalid characters.
Please use only lowercase letters, numbers, and hyphens (e.g., 'my-cluster-01')."

**IMPORTANT:**
- Always fetch current options - don't assume
- Match user input intelligently - "dc" could mean "data center"
- Ask for clarification when ambiguous
- Be helpful, patient, and guide users

Remember: You have tools to fetch real-time data and match user input intelligently. Use them!"""

    def get_tools(self) -> List[Tool]:
        return [
            Tool(
                name="validate_parameters",
                func=self._validate_parameters,
                description=(
                    "Validate parameters against schema. "
                    "Input: JSON with resource_type, operation, and params"
                )
            ),
            Tool(
                name="get_missing_params",
                func=self._get_missing_params,
                description=(
                    "Get list of missing required parameters for current operation. "
                    "Input: session_id"
                )
            ),
            Tool(
                name="extract_params_from_response",
                func=self._extract_params_from_response,
                description=(
                    "Extract parameter values from user's response. "
                    "Input: JSON with user_text and expected_params"
                )
            ),
            Tool(
                name="update_conversation_params",
                func=self._update_conversation_params,
                description=(
                    "Update conversation state with collected parameters. "
                    "Input: JSON with session_id and params"
                )
            ),
            Tool(
                name="fetch_available_options",
                func=self._fetch_available_options,
                description=(
                    "Fetch available options for a parameter from the API. "
                    "For example, fetch available data centers/endpoints dynamically. "
                    "Input: JSON with option_type (e.g., 'endpoints', 'zones', 'flavors')"
                )
            ),
            Tool(
                name="match_user_selection_to_options",
                func=self._match_user_selection,
                description=(
                    "Match user's natural language selection to available options. "
                    "For example, match 'delhi dc' or 'bengaluru' to actual endpoint names. "
                    "Input: JSON with user_text and available_options list"))]

    def _validate_parameters(self, input_json: str) -> str:
        """Validate parameters against schema."""
        try:
            data = json.loads(input_json)
            resource_type = data.get("resource_type")
            operation = data.get("operation")
            params = data.get("params", {})

            validation_result = api_executor_service.validate_parameters(
                resource_type, operation, params)
            return json.dumps(validation_result, indent=2)
        except Exception as e:
            return json.dumps({"valid": False, "errors": [str(e)]})
        
    def _get_missing_params(self, session_id: str) -> str:
        """Get missing parameters from conversation state."""
        try:
            state = conversation_state_manager.get_session(session_id)
            if not state:
                return json.dumps({"error": "Session not found"})

            return json.dumps({
                "missing_params": list(state.missing_params),
                "collected_params": list(state.collected_params.keys()),
                "invalid_params": state.invalid_params
            }, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    def _extract_params_from_response(self, input_json: str) -> str:
        """Extract parameters from user response."""
        try:
            data = json.loads(input_json)
            user_text = data.get("user_text", "")
            expected_params = data.get("expected_params", [])
            # Simple extraction logic (can be enhanced with NER/LLM)
            extracted = {}
            # Try to match expected parameters in user text
            for param in expected_params:
                param_lower = param.lower()
                import re
                pattern = rf'{param_lower}[:\s]+([^\s,]+)'
                match = re.search(pattern, user_text.lower())

                if match:
                    extracted[param] = match.group(1)
                elif len(expected_params) == 1:
                    extracted[param] = user_text.strip()
            return json.dumps(extracted, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})
    def _update_conversation_params(self, input_json: str) -> str:
        """Update conversation state with parameters."""
        try:
            data = json.loads(input_json)
            session_id = data.get("session_id")
            params = data.get("params", {})
            state = conversation_state_manager.get_session(session_id)
            if not state:
                return json.dumps({"success": False, "error": "Session not found"})
            state.add_parameters(params)
            conversation_state_manager.update_session(state)
            ready = state.is_ready_to_execute()
            return json.dumps({
                "success": True,
                "ready_to_execute": ready,
                "missing_params": list(state.missing_params)
            }, indent=2)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})
    async def _fetch_available_options(self, option_type: str, auth_token: str = None, user_type: str = None) -> str:
        """Fetch available options dynamically from API."""
        try:
            option_type_lower = option_type.lower().strip()
            
            # Use provided auth_token/user_type or fall back to instance variables
            token = auth_token or self._current_auth_token
            utype = user_type or self._current_user_type
            user_id = getattr(self, '_current_user_id', None)
            engagement_id = getattr(self, '_current_engagement_id', None)

            if option_type_lower in ["endpoints", "endpoint", "data_centers", "datacenters", "dc", "locations"]:
                # Fetch available endpoints/data centers
                # Pass user_id and engagement_id to ensure we use the selected engagement
                logger.info(f"🔍 Fetching endpoints with user_id={user_id}, engagement_id={engagement_id}")
                endpoints = await api_executor_service.get_endpoints(
                    auth_token=token, 
                    user_type=utype,
                    user_id=user_id,
                    engagement_id=engagement_id
                )

                if endpoints:
                    formatted_options = []
                    for ep in endpoints:
                        formatted_options.append({
                            "id": ep.get("endpointId"),
                            "name": ep.get("endpointDisplayName"),
                            "type": ep.get("endpointType", ""),
                            "description": f"Data Center: {ep.get('endpointDisplayName')}"
                        })

                    return json.dumps({
                        "option_type": "endpoints",
                        "count": len(formatted_options),
                        "options": formatted_options,
                        "prompt_suggestion": (
                            f"I found {len(formatted_options)} available data centers:\n" +
                            "\n".join([f"- {opt['name']} (ID: {opt['id']})" for opt in formatted_options]) +
                            "\n\nWhich one would you like to use? You can say the name or 'all'."
                        )
                    }, indent=2)
                else:
                    # No endpoints returned - check if engagement selection is needed
                    # Use cached engagements from get_engagement_id to avoid double API call
                    engagements = api_executor_service.get_cached_pending_engagements()
                    if not engagements:
                        # Fallback: fetch if not cached
                        engagements = await api_executor_service.get_engagements_list(auth_token=token)
                    
                    if engagements:
                        # CUS users: Auto-select first engagement (they typically have one)
                        if utype == "CUS":
                            logger.info(f"✅ CUS user - auto-selecting first engagement")
                            # Auto-select and retry fetching endpoints
                            first_eng = engagements[0]
                            eng_id = first_eng.get("id")
                            # This should be handled by the caller, but for now return empty
                            return json.dumps({
                                "option_type": "endpoints",
                                "auto_selected_engagement": eng_id,
                                "error": "CUS user - auto-selected engagement, please retry",
                                "count": 0,
                                "options": []
                            }, indent=2)
                        
                        # Multiple engagements AND not CUS → Must select (ENG or unknown user_type)
                        # This is the safe default - if we don't know the user type, ask them to select
                        elif len(engagements) > 1:
                            logger.info(f"🔄 User has {len(engagements)} engagements (user_type={utype}) - selection required")
                            return json.dumps({
                                "option_type": "engagement_selection_required",
                                "engagements": engagements,
                                "count": len(engagements),
                                "error": "Please select an engagement first",
                                "prompt_suggestion": self._format_engagement_selection_prompt(engagements)
                            }, indent=2)
                        
                        # Single engagement - auto-select (any user type)
                        else:
                            logger.info(f"✅ Auto-selecting single engagement (user_type={utype}, count={len(engagements)})")
                            return json.dumps({
                                "option_type": "endpoints",
                                "auto_selected_engagement": engagements[0].get("id"),
                                "error": "Auto-selected engagement, please retry",
                                "count": 0,
                                "options": []
                            }, indent=2)
                    
                    return json.dumps({
                        "option_type": "endpoints",
                        "error": "No engagements or endpoints available",
                        "count": 0,
                        "options": []
                    })

            elif option_type_lower in ["k8s_versions", "kubernetes_versions", "versions"]:
                # Fetch available Kubernetes versions
                versions = ["v1.27.16", "v1.28.15", "v1.29.12", "v1.30.14"]
                formatted_options = [{"id": v, "name": v, "description": f"Kubernetes {v}"} for v in versions]
                return json.dumps({
                    "option_type": "k8s_versions",
                    "count": len(formatted_options),
                    "options": formatted_options,
                    "prompt_suggestion": (
                        "Available Kubernetes versions:\n" +
                        "\n".join([f"- {v}" for v in versions]) +
                        "\n\nWhich version would you like?"
                    )
                }, indent=2)
            else:
                return json.dumps({
                    "error": f"Unknown option type: {option_type}",
                    "supported_types": ["endpoints", "k8s_versions"]
                })
        except Exception as e:
            logger.error(f"Error fetching options: {e}")
            return json.dumps({"error": str(e)})
    
    def _format_engagement_selection_prompt(self, engagements: list) -> str:
        """Format a user-friendly engagement selection prompt."""
        prompt = "Before I can proceed, please select an engagement to work with:\n\n"
        for i, eng in enumerate(engagements, 1):
            name = eng.get("engagementName") or eng.get("name") or "Unknown"
            eng_id = eng.get("id")
            prompt += f"**{i}. {name}** (ID: {eng_id})\n"
        prompt += "\nYou can say the number, name, or ID. You can change this later by saying 'switch engagement'."
        return prompt

    async def _extract_location_from_query_json(self, input_json: str) -> str:
        """Use LLM to intelligently extract location/endpoint from user query (JSON input version)."""
        try:
            data = json.loads(input_json)
            user_query = data.get("user_query", "")
            available_options = data.get("available_options", [])

            if not user_query or not available_options:
                return json.dumps({"extracted": False, "error": "Missing user_query or available_options"})

            options_str = "\n".join([f"- {opt['name']}" for opt in available_options])

            prompt = f"""You are a location extraction specialist. Extract the data center/endpoint name(s) from the user's query.

Available Data Centers:
{options_str}

User Query: "{user_query}"

Instructions:
1. If user mentions SPECIFIC data center(s), return them comma-separated:
   - Single: "delhi" → LOCATION: Delhi
   - Multiple: "delhi and bengaluru" → LOCATION: Delhi, Bengaluru
2. If user says "all", "all dc", "all datacenters", "all locations", "in all", etc. → return "all"
3. If no specific location mentioned and no "all" → return "none"

Examples:
- "list clusters in delhi" → LOCATION: Delhi
- "show clusters in delhi and bengaluru" → LOCATION: Delhi, Bengaluru
- "list all clusters" → LOCATION: all
- "show all" → LOCATION: all
- "list clusters" → LOCATION: none
- "clusters in mumbai and chennai" → LOCATION: Mumbai-BKC, Chennai-AMB
- "list container registry in all dc" → LOCATION: all
- "show kafka in all locations" → LOCATION: all
- "vms in all datacenters" → LOCATION: all

Respond with ONLY ONE of these formats:
- LOCATION: Delhi
- LOCATION: Delhi, Bengaluru
- LOCATION: all
- LOCATION: none"""

            response = await ai_service._call_chat_with_retries(
                prompt=prompt,
                max_tokens=50,
                temperature=0.0
            )

            result = response.strip()
            logger.info(f"🤖 LLM extraction result: '{result}'")

            # Parse the response
            if result.startswith("LOCATION:"):
                location = result.replace("LOCATION:", "").strip()
                if location.lower() == "none":
                    return json.dumps({"extracted": False})
                else:
                    return json.dumps({"extracted": True, "location": location})

            return json.dumps({"extracted": False})

        except Exception as e:
            logger.error(f"Error in LLM extraction: {e}")
            return json.dumps({"extracted": False, "error": str(e)})

    async def _match_user_selection_json(self, input_json: str) -> str:
        """Use LLM to intelligently match user's response to API results (JSON input version)."""
        try:
            data = json.loads(input_json)
            user_text = data.get("user_text", "").strip()
            available_options = data.get("available_options", [])

            if not user_text or not available_options:
                return json.dumps({"matched": False, "error": "Missing user_text or available_options"})

            # === LLM-BASED MATCHING (NO PRIMITIVE PATTERNS!) ===
            options_list = "\n".join([f"  {i+1}. {opt['name']} (ID: {opt['id']})" for i, opt in enumerate(available_options)])

            matching_prompt = f"""Match the user's response to the correct data center(s) from the API response.

Available Data Centers (from API):
{options_list}

User's Response: "{user_text}"

Instructions:
1. CRITICAL: If the user's response does NOT contain any location/city/datacenter name, return {{"matched": false}}
   - "list clusters" → NO location mentioned → {{"matched": false}}
   - "show me clusters" → NO location mentioned → {{"matched": false}}
   - "what clusters are there" → NO location mentioned → {{"matched": false}}
2. If user says "all" or "all of them" or "every" or "everywhere" → return ALL IDs
3. If user mentions MULTIPLE locations (comma-separated or "and"), match ALL of them:
   - "Delhi, Bengaluru" → match both Delhi and Bengaluru
   - "delhi and mumbai" → match Delhi and Mumbai-BKC
4. Match user input to the correct data center (handle typos, abbreviations, spaces/hyphens):
   - "delhi" → Delhi
   - "chennai amb" → Chennai-AMB
   - "mumbai bkc" or "mumbai" → Mumbai-BKC
   - "bengaluru" or "bangalore" or "blr" → Bengaluru

IMPORTANT: Only return matched=true if the user EXPLICITLY mentions a location name. Generic queries like "list clusters" or "show me" do NOT match any location.

Respond in JSON format ONLY:
{{
  "matched": true,
  "matched_ids": [11],
  "matched_names": ["Delhi"]
}}

For multiple locations:
{{
  "matched": true,
  "matched_ids": [11, 12],
  "matched_names": ["Delhi", "Bengaluru"]
}}

If NO location is mentioned in user's response:
{{
  "matched": false
}}"""

            # Call LLM to do the matching
            llm_response = await ai_service._call_chat_with_retries(
                prompt=matching_prompt,
                max_tokens=150,
                temperature=0.0
            )

            logger.info(f"🤖 LLM matching response: {llm_response[:150]}")

            # Extract JSON from LLM response
            import re
            json_match = re.search(r'\{[^{}]*\}', llm_response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                if result.get("matched"):
                    matched_ids = result.get("matched_ids", [])
                    matched_names = result.get("matched_names", [])
                    
                    if len(matched_ids) == 1:
                        return json.dumps({
                            "matched": True,
                            "selection": "single",
                            "matched_ids": matched_ids,
                            "matched_names": matched_names
                        }, indent=2)
                    elif len(matched_ids) > 1:
                        return json.dumps({
                            "matched": True,
                            "selection": "multiple",
                            "matched_ids": matched_ids,
                            "matched_names": matched_names
                        }, indent=2)

            # No match
            return json.dumps({"matched": False, "no_match": True}, indent=2)

        except Exception as e:
            logger.error(f"Error matching selection: {e}")
            return json.dumps({"matched": False, "error": str(e)})
    
    async def _handle_cluster_creation(self,input_text: str,state: Any) -> Dict[str, Any]:
        """
        Delegate cluster creation workflow to specialized handler.
        Args:
            input_text: User's current input
            state: Conversation state
        Returns:
            Dict with next prompt or ready_to_execute flag
        """
        logger.info(f"🎯 Delegating to ClusterCreationHandler")
        return await self.cluster_creation_handler.handle(input_text, state)

    async def _extract_location_from_query(self,user_query: str,available_endpoints: List[Dict[str, Any]]) -> Optional[str]:
        """
        Delegate to ParameterExtractor tool.
        """
        return await self.param_extractor.extract_location_from_query(user_query, available_endpoints)
    
    async def _match_user_selection(self,input_text: str,available_options: List[Dict[str, Any]]) -> str:
        return await self.param_extractor.match_user_selection(input_text, available_options)
    def _fallback_pattern_match(self,user_text: str,available_options: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Fallback pattern matching when LLM fails.
        Uses simple string matching for common cases.
        """
        user_lower = user_text.lower().strip()
        
        # Check for "all" (word-boundary match to avoid "firewall")
        all_phrases = [
            r"\ball\b",
            r"\ball of them\b",
            r"\ball datacenters\b",
            r"\ball endpoints\b",
            r"\ball dc\b",
            r"\ball locations\b",
            r"\beverywhere\b",
        ]
        if any(re.search(pattern, user_lower) for pattern in all_phrases):
            matched_ids = [opt.get("id") for opt in available_options if opt.get("id")]
            matched_names = [opt.get("name") for opt in available_options if opt.get("name")]
            logger.info(f"✅ Pattern matched 'all' keywords to {len(matched_ids)} endpoints")
            return {
                "matched": True,
                "all": True,
                "matched_ids": matched_ids,
                "matched_names": matched_names}
        # Split by common delimiters
        parts = re.split(r'[,;]|\band\b|\bor\b', user_lower)
        parts = [p.strip() for p in parts if p.strip()]
        matched_ids = []
        matched_names = []
        # Match against available options only - no hardcoded location names
        for part in parts:
            part_clean = re.sub(r'[^\w\s-]', '', part).strip()
            if not part_clean:
                continue
            for opt in available_options:
                opt_name = (opt.get("name") or "").lower()
                opt_id = opt.get("id")
                # Substring match: user input in option name or option name in user input
                if part_clean in opt_name or opt_name in part_clean:
                    if opt_id and opt_id not in matched_ids:
                        matched_ids.append(opt_id)
                        matched_names.append(opt.get("name"))
                        logger.info(f"✅ Pattern matched '{part}' to '{opt.get('name')}'")
                        break
        if matched_ids:
            return {
                "matched": True,
                "matched_ids": matched_ids,
                "matched_names": matched_names
            }
        logger.info(f"❌ No pattern match found for '{user_text}'")
        return {"matched": False}

    async def execute(self,input_text: str,context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute validation and parameter collection with INTELLIGENT tools.
        Args:
            input_text: User's response
            context: Context including session_id, conversation_state, and auth_token
        Returns:
            Dict with validation result
        """
        try:
            logger.info(f"✅ ValidationAgent processing: {input_text[:100]}...")
            # Get conversation state
            session_id = context.get("session_id") if context else None
            state = conversation_state_manager.get_session(session_id) if session_id else None
            if not state:
                return {
                    "agent_name": self.agent_name,
                    "success": False,
                    "error": "No conversation state found",
                    "output": "I couldn't find our conversation. Let's start over."}
            
            # Extract auth_token from context or state
            auth_token = context.get("auth_token") if context else None
            if not auth_token and state:
                auth_token = state.auth_token
            # Extract user_type from context or state
            user_type = context.get("user_type") if context else None
            if not user_type and state:
                user_type = state.user_type
            # Store in instance variables for tool methods to access
            self._current_auth_token = auth_token
            self._current_user_type = user_type
            self._current_user_id = state.user_id if state else None
            self._current_engagement_id = state.selected_engagement_id if state else None
            # USE INTELLIGENT TOOLS for parameter collection
            # SPECIAL HANDLING FOR K8S CLUSTER CREATION (CUSTOMER WORKFLOW)
            # This must be checked BEFORE missing_params check because cluster creation
            if state.operation == "create" and state.resource_type == "k8s_cluster":
                logger.info("🎯 Routing to customer cluster creation workflow")
                return await self._handle_cluster_creation(input_text, state)
            # Engagement list: no params needed, proceed directly
            if state.operation == "list" and state.resource_type == "engagement":
                logger.info("🎯 Engagement listing - no params needed, ready to execute")
                return {
                    "agent_name": self.agent_name,
                    "success": True,
                    "output": "Fetching available engagements...",
                    "ready_to_execute": True
                }
            if state.missing_params:
                # SPECIAL HANDLING FOR ENDPOINT LISTING 
                if state.operation == "list" and state.resource_type == "endpoint":
                    logger.info("🎯 Endpoint listing - fetching endpoints directly (no user selection needed)")
                    # For endpoint listing, we just need engagement_id which is fetched automatically
                    # Mark engagement_id as collected 
                    state.add_parameter("engagement_id", "auto", is_valid=True)
                    # Persist state
                    conversation_state_manager.update_session(state)
                    return {
                        "agent_name": self.agent_name,
                        "success": True,
                        "output": "Fetching available endpoints...",
                        "ready_to_execute": True }
                # SPECIAL HANDLING FOR OTHER CREATE OPERATIONS
                elif state.operation == "create":
                    # Check what's missing and prioritize logical order
                    param_priority = ["clusterName", "name", "endpoint_id", "endpoints"]
                    # Find the first missing param in priority order
                    next_param_to_collect = None
                    for priority_param in param_priority:
                        if priority_param in state.missing_params:
                            next_param_to_collect = priority_param
                            break
                    # If user provided input, try to extract the parameter value using LLM
                    if input_text and next_param_to_collect and "endpoint" not in next_param_to_collect.lower():
                        logger.info(f"🤖 Using LLM to extract {next_param_to_collect} from: '{input_text}'")
                        # Use LLM to understand if user provided the value
                        extraction_prompt = f"""User was asked to provide '{next_param_to_collect}' for creating a Kubernetes cluster.
User's response: "{input_text}"

Is the user providing a value for {next_param_to_collect}? Extract it.

Respond with ONLY ONE of these formats:
- VALUE: <extracted_value>
- UNCLEAR: <reason>

Examples:
User response: "tchl-paas-dev-vcp" → VALUE: tchl-paas-dev-vcp
User response: "I want to name it myCluster" → VALUE: myCluster  
User response: "something" → VALUE: something
User response: "what should I name it?" → UNCLEAR: User is asking a question
"""
                 
                        try:
                            llm_response = await ai_service._call_chat_with_retries(
                                prompt=extraction_prompt,
                                max_tokens=100,
                                temperature=0.0)
                            result = llm_response.strip()
                            logger.info(f"🤖 LLM extraction result: '{result}'")
                            if result.startswith("VALUE:"):
                                extracted_value = result.replace("VALUE:", "").strip()
                                logger.info(f"✅ Extracted {next_param_to_collect} = '{extracted_value}'")

                                state.add_parameter(next_param_to_collect, extracted_value, is_valid=True)
                                conversation_state_manager.update_session(state)
                                # Continue to next missing param or endpoint collection
                        except Exception as e:
                            logger.error(f"Error in LLM extraction: {e}")
                    # If still missing the same param (LLM couldn't extract), ask again
                    if next_param_to_collect and next_param_to_collect in state.missing_params and "endpoint" not in next_param_to_collect.lower():
                        logger.info(f"🔍 CREATE workflow: Still need {next_param_to_collect}")
                        # Ask for the parameter conversationally
                        response = f"Great! Let's create a new Kubernetes cluster.\n\n"
                        if "name" in next_param_to_collect.lower():
                            response += "What would you like to name your cluster?"
                        else:
                            response += f"Please provide: {next_param_to_collect}"
                        return {
                            "agent_name": self.agent_name,
                            "success": True,
                            "output": response,
                            "ready_to_execute": False,
                            "missing_params": list(state.missing_params),
                            "next_param": next_param_to_collect
                        }
                # Check if we need endpoint/datacenter parameter
                if "endpoints" in state.missing_params or "endpoint_id" in state.missing_params or "endpoint_ids" in state.missing_params:
                    # SPECIAL CHECK: If user wants to filter by BU/Environment/Zone, 
                    # skip endpoint prompt and go directly to execution (which will show filter options)
                    # IMPORTANT: Use input_text FIRST (current request), then fall back to state.user_query
                    user_query = input_text or state.user_query or ""
                    query_lower = user_query.lower()
                    
                    logger.info(f"🔍 Checking for BU/Env/Zone filter in query: '{query_lower}' (resource_type: {state.resource_type})")
                    
                    # Check for filter type keywords - these indicate user wants to filter by BU/Env/Zone
                    # The endpoint will come FROM the filter selection, not from user
                    bu_keywords = ["bu", "business unit", "business units", "department", "dept"]
                    env_keywords = ["environment", "environments", "env"]
                    zone_keywords = ["zone", "zones"]
                    
                    # Check if query mentions filtering by any of these
                    has_filter_intent = "filter" in query_lower or "by" in query_lower
                    has_bu = any(kw in query_lower for kw in bu_keywords)
                    has_env = any(kw in query_lower for kw in env_keywords)
                    has_zone = any(kw in query_lower for kw in zone_keywords)
                    
                    is_filter_request = has_filter_intent and (has_bu or has_env or has_zone)
                    logger.info(f"🔍 Filter check: has_filter_intent={has_filter_intent}, has_bu={has_bu}, has_env={has_env}, has_zone={has_zone}, is_filter_request={is_filter_request}")
                    
                    if is_filter_request and state.resource_type == "k8s_cluster":
                        # Determine filter type
                        if has_bu:
                            filter_type = "bu"
                        elif has_env:
                            filter_type = "environment"
                        else:
                            filter_type = "zone"
                        
                        logger.info(f"🎯 Detected {filter_type} filter request - skipping endpoint prompt, routing to execution")
                        
                        # Get all endpoints temporarily (execution agent will show filter options)
                        endpoints_json = await self._fetch_available_options("endpoints")
                        endpoints_data = json.loads(endpoints_json)
                        
                        # CHECK: If engagement selection is required (ENG users with multiple engagements)
                        if endpoints_data.get("option_type") == "engagement_selection_required":
                            logger.info("🏢 Engagement selection required - prompting user")
                            engagements = endpoints_data.get("engagements", [])
                            
                            # Store engagements in state for later selection
                            state.pending_engagements = engagements
                            state.status = ConversationStatus.AWAITING_ENGAGEMENT_SELECTION
                            conversation_state_manager.update_session(state)
                            
                            return {
                                "agent_name": self.agent_name,
                                "success": True,
                                "output": endpoints_data.get("prompt_suggestion", "Please select an engagement."),
                                "ready_to_execute": False,
                                "engagement_selection_required": True
                            }
                        
                        if endpoints_data.get("options"):
                            all_endpoint_ids = [opt.get("id") for opt in endpoints_data["options"] if opt.get("id")]
                            all_endpoint_names = [opt.get("name") for opt in endpoints_data["options"] if opt.get("name")]
                            
                            # Temporarily mark endpoints as collected (execution will handle filter)
                            state.add_parameter("endpoints", all_endpoint_ids, is_valid=True)
                            state.add_parameter("endpoint_names", all_endpoint_names, is_valid=True)
                            
                            # Persist state
                            conversation_state_manager.update_session(state)
                            
                            # Mark ready to execute - the K8sClusterAgent will detect the filter request
                            # and show the BU/Env/Zone options instead of listing clusters
                            return {
                                "agent_name": self.agent_name,
                                "success": True,
                                "output": "Let me show you the available filter options...",
                                "ready_to_execute": True
                            }
                    
                    logger.info("🔍 Collecting endpoint parameter using intelligent tools")
                    # Fetch available endpoints dynamically
                    endpoints_json = await self._fetch_available_options("endpoints")
                    endpoints_data = json.loads(endpoints_json)
                    
                    # REUSE: If we have saved_endpoints and user input doesn't match any endpoint name, reuse
                    available_options = endpoints_data.get("options", [])
                    saved_eps = getattr(state, "saved_endpoints", None)
                    user_input = (input_text or state.user_query or "").strip().lower()
                    if saved_eps and available_options and state.resource_type in ("k8s_cluster", "vm", "firewall", "load_balancer", "lb"):
                        endpoint_names = [str(o.get("name", "")).lower() for o in available_options if o.get("name")]
                        input_looks_like_endpoint = user_input and (
                            user_input in endpoint_names or
                            any(user_input in en for en in endpoint_names) or
                            any(en in user_input for en in endpoint_names if len(user_input) >= 2)
                        )
                        if not input_looks_like_endpoint:
                            logger.info(f"♻️ Reusing saved endpoints: {getattr(state, 'saved_endpoint_names', saved_eps)} (input doesn't match endpoint names)")
                            state.add_parameter("endpoints", saved_eps, is_valid=True)
                            state.add_parameter("endpoint_names", getattr(state, "saved_endpoint_names", []), is_valid=True)
                            conversation_state_manager.update_session(state)
                            if state.is_ready_to_execute():
                                return {
                                    "agent_name": self.agent_name,
                                    "success": True,
                                    "output": f"Using {state.saved_endpoint_names or 'saved'} data center(s)...",
                                    "ready_to_execute": True
                                }
                    
                    # CHECK: If engagement selection is required (ENG users with multiple engagements)
                    if endpoints_data.get("option_type") == "engagement_selection_required":
                        logger.info("🏢 Engagement selection required - prompting user")
                        engagements = endpoints_data.get("engagements", [])
                        
                        # Store engagements in state for later selection
                        state.pending_engagements = engagements
                        state.status = ConversationStatus.AWAITING_ENGAGEMENT_SELECTION
                        conversation_state_manager.update_session(state)
                        
                        return {
                            "agent_name": self.agent_name,
                            "success": True,
                            "output": endpoints_data.get("prompt_suggestion", "Please select an engagement."),
                            "ready_to_execute": False,
                            "engagement_selection_required": True
                        }
                    
                    if endpoints_data.get("options"):
                        available_options = endpoints_data["options"]
                        # ALWAYS use input_text for NEW requests (not cached state.user_query!)
                        # Only use state.user_query if this is a follow-up response to "Which endpoint?"
                        # We detect follow-up by checking if we already asked a question
                        is_follow_up = len(state.conversation_history) > 1 and any(
                            "which one would you like" in msg.get("content", "").lower()
                            for msg in state.conversation_history if msg.get("role") == "assistant")
                        text_to_analyze = input_text if not is_follow_up else input_text
                        logger.info(f"🔍 Analyzing query: '{text_to_analyze}' for location extraction (is_follow_up: {is_follow_up})")
                        # USE LLM FOR INTELLIGENT EXTRACTION (not primitive pattern matching!)
                        try:
                            extraction_result_json = await self._extract_location_from_query_json(json.dumps({
                                "user_query": text_to_analyze,
                                "available_options": available_options
                            }))
                            extraction_result = json.loads(extraction_result_json) if extraction_result_json else {}
                        except Exception as e:
                            logger.warning(f"⚠️ Location extraction failed: {e}")
                            extraction_result = {}
                        if extraction_result.get("extracted"):
                            # LLM extracted a location!
                            extracted_location = extraction_result.get("location", "")
                            logger.info(f"🤖 LLM extracted location: '{extracted_location}'")
                            # SPECIAL CASE: If location is "all", immediately match all endpoints
                            if extracted_location.lower().strip() == "all":
                                logger.info(f"🌍 User requested ALL data centers!")
                                matched_ids = [opt.get("id") for opt in available_options if opt.get("id")]
                                matched_names = [opt.get("name") for opt in available_options if opt.get("name")]
                                # Add to state
                                state.add_parameter("endpoints", matched_ids, is_valid=True)
                                state.add_parameter("endpoint_names", matched_names, is_valid=True)
                                # Persist state after parameter collection
                                conversation_state_manager.update_session(state)
                                # Check if ready to execute now that we've collected endpoints
                                if state.is_ready_to_execute():
                                    logger.info("✅ All parameters collected, ready to execute")
                                    return {
                                        "agent_name": self.agent_name,
                                        "success": True,
                                        "output": f"Great! Fetching data from all {len(matched_ids)} data centers...",
                                        "ready_to_execute": True}
                            text_to_match = extracted_location
                        else:
                            text_to_match = input_text
                            logger.info(f"🔍 No location in query, using current input: '{text_to_match}'")
                        # Try to match user input to available endpoints (LLM-based!)
                        try:
                            match_result_json = await self._match_user_selection_json(json.dumps({
                                "user_text": text_to_match,
                                "available_options": available_options
                            }))
                            match_result = json.loads(match_result_json) if match_result_json else {}
                        except Exception as e:
                            logger.warning(f"⚠️ LLM matching failed: {e}")
                            match_result = {}
                        # FALLBACK: If LLM failed, try simple pattern matching
                        if not match_result.get("matched"):
                            logger.info("🔄 LLM matching failed, trying fallback pattern matching...")
                            match_result = self._fallback_pattern_match(text_to_match, available_options)
                        if match_result.get("matched"):
                            # Successfully matched!
                            matched_ids = match_result.get("matched_ids", [])
                            matched_names = match_result.get("matched_names", [])
                            logger.info(f"✅ Matched '{input_text}' to {matched_names} (IDs: {matched_ids})")
                            # Add to state
                            state.add_parameter("endpoints", matched_ids, is_valid=True)
                            state.add_parameter("endpoint_names", matched_names, is_valid=True)
                            # Persist state after parameter collection
                            conversation_state_manager.update_session(state)
                            # Check if ready to execute now that we've collected endpoints
                            if state.is_ready_to_execute():
                                logger.info("✅ All parameters collected, ready to execute")
                                return {
                                    "agent_name": self.agent_name,
                                    "success": True,
                                    "output": f"Great! Fetching clusters from {', '.join(matched_names)}...",
                                    "ready_to_execute": True}
                            # If still missing params, continue to collect them
                        elif match_result.get("ambiguous"):
                            # Multiple matches - ask for clarification
                            clarification = match_result.get("clarification_needed", "")
                            # Set status to AWAITING_SELECTION - we need user to pick one
                            state.status = ConversationStatus.AWAITING_SELECTION
                            # Persist state so it's available for next request
                            conversation_state_manager.update_session(state)

                            return {
                                "agent_name": self.agent_name,
                                "success": True,
                                "output": clarification,
                                "ready_to_execute": False,
                                "missing_params": list(state.missing_params)}
                        else:
                            # No match - show available options
                            prompt = endpoints_data.get("prompt_suggestion", "")
                            if not prompt:
                                options_list = "\n".join([f"- {opt['name']}" for opt in available_options])
                                prompt = f"I found {len(available_options)} available data centers:\n{options_list}\n\nWhich one would you like? You can also say 'all'."

                            # CRITICAL: Set status to AWAITING_SELECTION so orchestrator knows we're waiting for user selection
                            state.status = ConversationStatus.AWAITING_SELECTION
                            logger.info(f"🔄 Set conversation status to AWAITING_SELECTION for session {state.session_id}")
                            # Persist state so it's available for next request
                            conversation_state_manager.update_session(state)
                            return {
                                "agent_name": self.agent_name,
                                "success": True,
                                "output": prompt,
                                "ready_to_execute": False,
                                "missing_params": list(state.missing_params),
                                "available_options": available_options}
                    else:
                        return {
                            "agent_name": self.agent_name,
                            "success": False,
                            "output": "I couldn't fetch the available data centers. Please try again.",
                            "error": endpoints_data.get("error")}
                # For other parameters, use simple extraction
                else:
                    extracted_params = self._simple_param_extraction(input_text, state.missing_params)
                    if extracted_params:
                        # Validate extracted parameters
                        validation_result = api_executor_service.validate_parameters(
                            state.resource_type,
                            state.operation,
                            {**state.collected_params, **extracted_params}
                        )
                        # Add valid parameters to state
                        for param_name, param_value in extracted_params.items():
                            if validation_result["valid"] or param_name not in validation_result.get("errors", []):
                                state.add_parameter(param_name, param_value, is_valid=True)
                            else:
                                param_errors = [e for e in validation_result.get("errors", []) if param_name in e]
                                error_msg = param_errors[0] if param_errors else "Invalid value"
                                state.mark_parameter_invalid(param_name, error_msg)
            # Check if ready to execute
            if state.is_ready_to_execute():
                response = (
                    f"Perfect! I have all the information needed to {state.operation} the {state.resource_type}.\n\n"
                    f"**Summary:**\n"
                )
                for param, value in state.collected_params.items():
                    response += f"- {param}: {value}\n"
                response += "\nShall I proceed with this operation?"
                return {
                    "agent_name": self.agent_name,
                    "success": True,
                    "output": response,
                    "ready_to_execute": True
                }
            # Generate response asking for missing parameters
            response = self._generate_collection_message(state)
            return {
                "agent_name": self.agent_name,
                "success": True,
                "output": response,
                "ready_to_execute": False,
                "missing_params": list(state.missing_params)
            }
        except Exception as e:
            logger.error(f"❌ Validation failed: {str(e)}")
            return {
                "agent_name": self.agent_name,
                "success": False,
                "error": str(e),
                "output": f"I encountered an error while validating: {str(e)}"}

    def _simple_param_extraction(self,user_text: str,expected_params: set) -> Dict[str, Any]:
        """
        Simple parameter extraction from user text.
        Args:
            user_text: User's input text
            expected_params: Set of expected parameter names
        Returns:
            Dict of extracted parameters
        """
        extracted = {}
        # If only one parameter expected and input is simple, use entire input
        if len(expected_params) == 1:
            param_name = list(expected_params)[0]
            extracted[param_name] = user_text.strip()
        else:
            import re
            for param in expected_params:
                pattern = rf'{re.escape(param)}[:\s=]+([^\n,]+)'
                match = re.search(pattern, user_text, re.IGNORECASE)

                if match:
                    extracted[param] = match.group(1).strip()
        return extracted
    
    def _generate_collection_message(self, state) -> str:
        """
        Generate a message asking for missing parameters.
        Args:
            state: Conversation state
        Returns:
            Message string
        """
        if not state.missing_params:
            return "All parameters collected!"
        # Show invalid parameters first
        message = ""
        if state.invalid_params:
            message += "I noticed some issues with the information provided:\n"
            for param, error in state.invalid_params.items():
                message += f"- {param}: {error}\n"
            message += "\n"
        # Ask for missing parameters
        missing_list = sorted(state.missing_params)
        if len(missing_list) == 1:
            param = missing_list[0]
            message += f"I need one more thing: **{param}**. Could you provide that?"
        elif len(missing_list) <= 3:
            message += "I need a few more details:\n"
            for param in missing_list:
                message += f"- {param}\n"
            message += "\nYou can provide these all at once or one at a time."
        else:
            # Show first 3 parameters
            message += "I need several more details. Let's start with these:\n"
            for param in missing_list[:3]:
                message += f"- {param}\n"
            message += f"\n(Plus {len(missing_list) - 3} more after these)"
        return message
