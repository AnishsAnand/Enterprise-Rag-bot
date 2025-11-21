"""
Validation Agent - Validates parameters and collects missing information.
Ensures all required parameters are present and valid before execution.
"""

from typing import Any, Dict, List, Optional
from langchain.tools import Tool
import logging
import json

from app.agents.base_agent import BaseAgent
from app.agents.state.conversation_state import conversation_state_manager
from app.services.api_executor_service import api_executor_service

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
        
        # Setup agent
        self.setup_agent()
    
    def get_system_prompt(self) -> str:
        """Return system prompt for validation agent."""
        return """You are the Validation Agent, responsible for ensuring all parameters are correct and complete.

**Your responsibilities:**
1. **Validate collected parameters** against schema rules
2. **Identify missing required parameters**
3. **Ask for missing information** in a conversational way
4. **Extract parameters** from user responses
5. **Provide helpful guidance** on parameter format and requirements

**Validation rules:**
- Check data types (string, integer, boolean, etc.)
- Validate string lengths (min/max)
- Check numeric ranges (min/max)
- Validate enum values
- Check regex patterns
- Ensure required parameters are present

**When asking for parameters:**
- Be conversational and friendly
- Explain why the parameter is needed
- Provide examples when helpful
- Ask for one or a few related parameters at a time (don't overwhelm user)
- If user provides partial information, acknowledge it and ask for remaining items

**Example interactions:**

Missing name:
"I'll help you create that Kubernetes cluster. What would you like to name it? 
(Use lowercase letters, numbers, and hyphens only, 3-63 characters)"

Missing multiple params:
"Great! To create the cluster, I need a few more details:
- Cluster name (e.g., 'prod-cluster-01')
- Data Center location
- Kubernetes version you'd like to use

You can provide these all at once or one at a time."

Invalid parameter:
"The cluster name 'My_Cluster!' contains invalid characters. 
Please use only lowercase letters, numbers, and hyphens (e.g., 'my-cluster-01')."

Be helpful, patient, and guide users to provide valid information."""
    
    def get_tools(self) -> List[Tool]:
        """Return tools for validation agent."""
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
            )
        ]
    
    def _validate_parameters(self, input_json: str) -> str:
        """Validate parameters against schema."""
        try:
            data = json.loads(input_json)
            resource_type = data.get("resource_type")
            operation = data.get("operation")
            params = data.get("params", {})
            
            validation_result = api_executor_service.validate_parameters(
                resource_type, operation, params
            )
            
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
                
                # Look for patterns like "name: value" or "name is value"
                import re
                pattern = rf'{param_lower}[:\s]+([^\s,]+)'
                match = re.search(pattern, user_text.lower())
                
                if match:
                    extracted[param] = match.group(1)
                elif len(expected_params) == 1:
                    # If only one parameter expected, assume entire input is the value
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
            
            # Add parameters to state
            state.add_parameters(params)
            
            # Check if ready to execute
            ready = state.is_ready_to_execute()
            
            return json.dumps({
                "success": True,
                "ready_to_execute": ready,
                "missing_params": list(state.missing_params)
            }, indent=2)
            
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})
    
    async def execute(
        self,
        input_text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute validation and parameter collection.
        
        Args:
            input_text: User's response
            context: Context including session_id and conversation_state
            
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
                    "output": "I couldn't find our conversation. Let's start over."
                }
            
            # Try to extract parameters from user input
            if input_text and state.missing_params:
                # Simple parameter extraction
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
                            # Find specific error for this parameter
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
                "output": f"I encountered an error while validating: {str(e)}"
            }
    
    def _simple_param_extraction(
        self,
        user_text: str,
        expected_params: set
    ) -> Dict[str, Any]:
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
            # Try to extract multiple parameters
            import re
            for param in expected_params:
                # Look for "param: value" or "param = value" patterns
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

