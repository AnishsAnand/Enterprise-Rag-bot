"""
Cluster Creation Handler - Manages the multi-step workflow for creating Kubernetes clusters.

This handler encapsulates the 17-step customer workflow for cluster creation,
making it easier to maintain and test independently.

Enhanced Features:
- AI-powered intent understanding during workflow
- Intelligent detection of off-topic requests, change requests, and general conversation
- Graceful handling of workflow interruptions with user choice
- Natural conversation support (greetings, questions about capabilities)
"""
import logging
import re
from typing import Any, Dict, Optional, List, Tuple
import json
from app.services.api_executor_service import api_executor_service
from app.services.ai_service import ai_service
from app.agents.tools.parameter_extraction import ParameterExtractor
from app.agents.state.conversation_state import conversation_state_manager, ConversationStatus

logger = logging.getLogger(__name__)

class ClusterCreationHandler:
    """
    Handles the complete workflow for customer cluster creation.
    
    Implements the 17-step process defined in resource_schema.json,
    collecting parameters conversationally and validating them.
    """
    
    def __init__(self):
        self.param_extractor = ParameterExtractor()
        # Define the parameter collection workflow order for customers
        self.workflow = [
            "clusterName",
            "datacenter", 
            "k8sVersion",
            "cniDriver",
            "businessUnit",
            "environment",
            "zone",
            "operatingSystem",
            "workerPoolName",
            "nodeType",
            "flavor",
            "additionalStorage", 
            "replicaCount",
            "enableAutoscaling",  
            "maxReplicas",
            "addMoreWorkerPools",  # Ask if user wants to add more worker pools
            "tags" 
        ]
        
        # Worker pool specific params that get repeated for each pool
        self.worker_pool_params = [
            "workerPoolName",
            "nodeType", 
            "flavor",
            "additionalStorage",
            "replicaCount",
            "enableAutoscaling",
            "maxReplicas"
        ]
    
    def _check_cancel_intent(self, input_text: str, state: Any) -> Optional[Dict[str, Any]]:
        """
        Check if user wants to cancel the cluster creation workflow.
        
        Only triggers on explicit cancel keywords/phrases.
        
        Args:
            input_text: User's current input
            state: Conversation state
            
        Returns:
            Dict with cancel response if cancel detected, None otherwise
        """
        if not input_text:
            return None
        
        input_lower = input_text.lower().strip()
        
        # Only trigger on explicit cancel keywords
        cancel_keywords = [
            "cancel", "abort", "stop", "quit", "exit", "nevermind", "never mind",
            "forget it", "forget about it", "don't want to", "dont want to",
            "i don't want", "i dont want", "stop this", "cancel this",
            "cancel creation", "abort creation", "stop creating"
        ]
        
        is_cancel = any(keyword in input_lower for keyword in cancel_keywords)
        
        if not is_cancel:
            return None  # Not a cancel intent
        
        logger.info(f"🚫 Processing cancel request: '{input_text}'")
        
        # Clear the conversation state
        state.status = ConversationStatus.CANCELLED
        state.collected_params = {}
        state.missing_params = []
        if hasattr(state, 'last_asked_param'):
            state.last_asked_param = None
        
        # Clear any workflow flags
        state._workflow_interrupted = False
        state._workflow_paused = False
        state._pending_off_topic_input = None
        
        # Persist the cancelled state
        conversation_state_manager.update_session(state)
        
        return {
            "agent_name": "ValidationAgent",
            "success": True,
            "cancelled": True,
            "output": "No problem! I've cancelled the cluster creation.\n\nWhenever you're ready, just let me know what you'd like to do. I can help you:\n• Create a new cluster\n• View existing clusters\n• Check other resources\n• Answer questions\n\nHow can I help?"
        }
    
    async def _understand_user_intent(self, input_text: str, state: Any) -> Dict[str, Any]:
        """
        Use AI to intelligently understand what the user wants during the workflow.
        
        This replaces pattern-based detection with true AI understanding.
        Handles:
        - Valid answers to the current question
        - Requests to change/modify previous selections
        - Off-topic requests (wanting to do something else entirely)
        - General conversation (greetings, questions about capabilities)
        - Cancel/abort requests
        
        Args:
            input_text: User's current input
            state: Conversation state
            
        Returns:
            Dict with intent classification:
            - intent_type: "answer" | "change" | "off_topic" | "greeting" | "cancel" | "help"
            - details: Additional context about the intent
        """
        if not input_text or not input_text.strip():
            return {"intent_type": "answer", "details": "empty input"}
        
        last_param = getattr(state, 'last_asked_param', None)
        current_step = self._get_param_display_name(last_param) if last_param else "starting cluster creation"
        
        # Get list of collected params for context
        collected_summary = []
        for param in self.workflow:
            if param in state.collected_params:
                value = state.collected_params[param]
                display_value = value.get('name') if isinstance(value, dict) else str(value)
                collected_summary.append(f"{self._get_param_display_name(param)}: {display_value}")
        
        collected_context = "\n".join(collected_summary) if collected_summary else "None yet"
        
        try:
            prompt = f"""You are an intelligent assistant helping a user create a Kubernetes cluster.
The user is in the middle of a multi-step workflow.

**Current Context:**
- Workflow: Creating a Kubernetes cluster
- Current Step: Asking for "{current_step}"
- Parameters already collected:
{collected_context}

**User's Input:** "{input_text}"

**Your task:** Classify what the user wants. Choose ONE of these categories:

1. **ANSWER** - The user is providing an answer to the current question
   Examples: "delhi", "v1.30", "calico", "4", "yes", "ubuntu", "general purpose"

2. **CHANGE** - The user wants to modify a previously selected parameter
   Examples: "I want to change the datacenter", "can we use a different zone", "go back to kubernetes version", "actually make it mumbai instead"

3. **OFF_TOPIC** - The user wants to do something completely different (not related to this cluster creation)
   Examples: "what clusters are in delhi", "show me load balancers", "list all VMs", "check firewall status"

4. **GREETING** - The user is greeting or asking about capabilities
   Examples: "hi", "hello", "what can you do", "help me", "who are you"

5. **CANCEL** - The user wants to stop/abort the cluster creation entirely
   Examples: "cancel", "stop this", "I don't want to create a cluster anymore", "abort"

**Important considerations:**
- If user says a city/location name like "delhi" or "mumbai" alone, it's likely an ANSWER (selecting datacenter)
- If user asks "what clusters are in delhi", it's OFF_TOPIC (querying existing clusters)
- If user mentions changing, modifying, or going back to something, it's CHANGE
- Brief acknowledgments or clarifications about the current question are ANSWER

Respond with a JSON object:
{{"intent_type": "answer|change|off_topic|greeting|cancel", "details": "brief explanation", "change_target": "parameter name if CHANGE"}}"""

            response = await ai_service._call_chat_with_retries(
                prompt=prompt,
                temperature=0.1,
                max_tokens=150
            )
            
            # Parse JSON response
            response_text = response.strip()
            
            # Extract JSON from response
            json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                logger.info(f"🧠 AI understood intent: {result.get('intent_type')} - {result.get('details', '')[:50]}")
                return result
            else:
                # Fallback: try to parse the whole response
                result = json.loads(response_text)
                return result
                
        except json.JSONDecodeError as e:
            logger.warning(f"⚠️ Failed to parse AI response: {e}, defaulting to ANSWER")
            return {"intent_type": "answer", "details": "parse error fallback"}
        except Exception as e:
            logger.warning(f"⚠️ Intent understanding failed: {e}, defaulting to ANSWER")
            return {"intent_type": "answer", "details": "error fallback"}
    
    def _handle_greeting_in_workflow(self, input_text: str, state: Any) -> Dict[str, Any]:
        """
        Handle greetings during the cluster creation workflow.
        
        Acknowledges briefly and guides back to the current step.
        Global greeting handling is done at the orchestrator level.
        
        Args:
            input_text: User's greeting
            state: Conversation state
            
        Returns:
            Dict with brief acknowledgment and continuation
        """
        last_param = getattr(state, 'last_asked_param', None)
        current_step = self._get_param_display_name(last_param) if last_param else None
        current_step_num = self._get_workflow_step_number(last_param)
        total_steps = len(self.workflow)
        
        response = f"👋 Hi there! We're currently on step {current_step_num} of {total_steps} for your cluster creation."
        
        if current_step:
            response += f"\n\nLet's continue - I was asking about **{current_step}**."
        
        return {
            "agent_name": "ValidationAgent",
            "success": True,
            "greeting_handled": True,
            "output": response
        }
    
    async def _check_resume_intent(self, input_text: str) -> bool:
        """
        Use AI to check if user wants to resume the paused cluster creation.
        
        Args:
            input_text: User's input
            
        Returns:
            True if user wants to resume, False otherwise
        """
        try:
            prompt = f"""The user previously paused a cluster creation workflow.
            
User's message: "{input_text}"

Is the user trying to resume/continue the cluster creation they paused earlier?

Respond with ONLY: YES or NO"""

            response = await ai_service._call_chat_with_retries(
                prompt=prompt,
                temperature=0.0,
                max_tokens=10
            )
            
            return response.strip().upper() == "YES"
            
        except Exception as e:
            logger.warning(f"⚠️ Resume intent check failed: {e}")
            # Fallback to simple check
            lower = input_text.lower()
            return any(word in lower for word in ["resume", "continue cluster", "back to cluster"])
    
    async def _handle_change_request(self, change_target: str, state: Any) -> Optional[Dict[str, Any]]:
        """
        Intelligently handle a request to change a previous parameter.
        
        Uses AI understanding of what parameter the user wants to change,
        rather than exact keyword matching.
        
        Args:
            change_target: AI-detected parameter the user wants to change
            state: Conversation state
            
        Returns:
            Dict with response if change handled, None otherwise
        """
        if not change_target:
            return None
        
        change_target_lower = change_target.lower().strip()
        
        # Map common terms to parameter names using AI understanding
        # This is a fallback - the AI should have already identified the target
        param_mappings = {
            "cluster name": "clusterName",
            "name": "clusterName",
            "datacenter": "datacenter",
            "data center": "datacenter",
            "location": "datacenter",
            "dc": "datacenter",
            "kubernetes version": "k8sVersion",
            "k8s version": "k8sVersion",
            "version": "k8sVersion",
            "cni": "cniDriver",
            "cni driver": "cniDriver",
            "network driver": "cniDriver",
            "business unit": "businessUnit",
            "bu": "businessUnit",
            "department": "businessUnit",
            "environment": "environment",
            "env": "environment",
            "zone": "zone",
            "network zone": "zone",
            "operating system": "operatingSystem",
            "os": "operatingSystem",
            "worker pool": "workerPoolName",
            "pool name": "workerPoolName",
            "node type": "nodeType",
            "type": "nodeType",
            "flavor": "flavor",
            "size": "flavor",
            "cpu": "flavor",
            "memory": "flavor",
            "ram": "flavor",
            "storage": "additionalStorage",
            "disk": "additionalStorage",
            "replica": "replicaCount",
            "replicas": "replicaCount",
            "count": "replicaCount",
            "nodes": "replicaCount",
            "autoscaling": "enableAutoscaling",
            "auto scaling": "enableAutoscaling",
            "max replicas": "maxReplicas",
            "maximum": "maxReplicas",
            "tags": "tags",
        }
        
        # Find the parameter to change
        target_param = None
        
        # First, check if it's an exact parameter name
        if change_target_lower in [p.lower() for p in self.workflow]:
            target_param = next(p for p in self.workflow if p.lower() == change_target_lower)
        else:
            # Check mappings
            for term, param in param_mappings.items():
                if term in change_target_lower or change_target_lower in term:
                    target_param = param
                    break
        
        if not target_param:
            # Couldn't identify the parameter - ask for clarification
            collected_params = [p for p in self.workflow if p in state.collected_params]
            if collected_params:
                params_list = ", ".join([self._get_param_display_name(p) for p in collected_params])
                return {
                    "agent_name": "ValidationAgent",
                    "success": True,
                    "output": f"I'd be happy to help you change something. Which parameter would you like to modify?\n\n**Currently set:** {params_list}"
                }
            return None
        
        # Check if parameter was actually collected
        if target_param not in state.collected_params:
            return {
                "agent_name": "ValidationAgent",
                "success": True,
                "output": f"**{self._get_param_display_name(target_param)}** hasn't been set yet. Let's continue with the current step and you can provide it when we get there."
            }
        
        # Get old value for reference
        old_value = state.collected_params.get(target_param)
        old_display = old_value.get('name') if isinstance(old_value, dict) else str(old_value)
        
        # Clear the parameter and dependents
        self._clear_parameter_and_dependents(target_param, state)
        
        state.last_asked_param = None
        conversation_state_manager.update_session(state)
        
        # Ask for new value
        response = f"No problem! Let's change the **{self._get_param_display_name(target_param)}**.\n\n*Previous value: {old_display}*\n\n"
        
        next_question = await self._ask_for_parameter(target_param, state)
        response += next_question["output"]
        
        return {
            "agent_name": "ValidationAgent",
            "success": True,
            "parameter_changed": target_param,
            "output": response
        }
    
    def _build_workflow_interruption_prompt(self, input_text: str, state: Any) -> Dict[str, Any]:
        """
        Build a conversational prompt when user wants to do something else mid-workflow.
        
        Args:
            input_text: User's off-topic input
            state: Conversation state
            
        Returns:
            Dict with friendly options to handle the situation
        """
        last_param = getattr(state, 'last_asked_param', None)
        current_step = self._get_param_display_name(last_param) if last_param else "cluster creation"
        
        # Get summary of collected params so far
        collected_count = len([p for p in state.collected_params if not p.startswith('_')])
        total_steps = len(self.workflow)
        
        # Store the pending intent for later processing
        state._pending_off_topic_input = input_text
        state._workflow_interrupted = True
        conversation_state_manager.update_session(state)
        
        output = f"""I noticed you're asking about something different while we're creating your cluster.

**Your question:** "{input_text}"

You're currently {collected_count} steps into the cluster creation (out of {total_steps}). I don't want you to lose your progress!

**How would you like to proceed?**

• **"abort"** - I'll cancel the cluster creation and help with your question
• **"continue"** - Let's finish the cluster creation first, then I'll help
• **"save"** - I'll save your progress, help with your question, and you can say "resume cluster" later to continue

What would you prefer?"""
        
        return {
            "agent_name": "ValidationAgent",
            "success": True,
            "workflow_interrupted": True,
            "output": output
        }
    
    def _get_workflow_step_number(self, param_name: str) -> int:
        """Get the step number for a parameter in the workflow."""
        if not param_name or param_name.startswith('_'):
            return 1
        try:
            return self.workflow.index(param_name) + 1
        except ValueError:
            return 1
    
    async def _handle_workflow_interruption_response(self, input_text: str, state: Any) -> Optional[Dict[str, Any]]:
        """
        Handle user's response to workflow interruption prompt using AI understanding.
        
        Args:
            input_text: User's response
            state: Conversation state
            
        Returns:
            Dict with appropriate response based on user choice
        """
        # Check if workflow was interrupted
        if not getattr(state, '_workflow_interrupted', False):
            return None
        
        pending_input = getattr(state, '_pending_off_topic_input', '')
        
        # Use AI to understand user's choice
        try:
            prompt = f"""The user was asked how to handle an interruption during cluster creation.
They were given these options:
- "abort" - Cancel cluster creation and handle their other request
- "continue" - Finish cluster creation first
- "save" - Save progress and handle other request, resume later

User's response: "{input_text}"

What did the user choose? Respond with ONLY one word: ABORT, CONTINUE, or SAVE
If unclear, respond with UNCLEAR."""

            response = await ai_service._call_chat_with_retries(
                prompt=prompt,
                temperature=0.0,
                max_tokens=20
            )
            
            choice = response.strip().upper()
            logger.info(f"🧠 User's interruption choice: {choice}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to understand interruption response: {e}")
            choice = "UNCLEAR"
        
        # Handle ABORT
        if choice == "ABORT":
            logger.info(f"🚫 User chose to abort cluster creation to handle: '{pending_input}'")
            
            # Clear workflow state
            state.status = ConversationStatus.IDLE
            state.collected_params = {}
            state.missing_params = []
            state.last_asked_param = None
            state._workflow_interrupted = False
            state._pending_off_topic_input = None
            conversation_state_manager.update_session(state)
            
            return {
                "agent_name": "ValidationAgent",
                "success": True,
                "workflow_aborted": True,
                "pending_request": pending_input,
                "output": f"Got it! I've cancelled the cluster creation.\n\nLet me help you with: \"{pending_input}\""
            }
        
        # Handle CONTINUE
        elif choice == "CONTINUE":
            logger.info(f"▶️ User chose to continue cluster creation")
            
            # Clear interruption flag
            state._workflow_interrupted = False
            state._pending_off_topic_input = None
            conversation_state_manager.update_session(state)
            
            # Re-ask current parameter
            last_param = state.last_asked_param
            response_text = "Great, let's continue with the cluster creation.\n\n"
            
            if last_param and last_param != "_confirmation":
                next_q = await self._ask_for_parameter(last_param, state)
                next_q["output"] = response_text + next_q["output"]
                return next_q
            elif last_param == "_confirmation":
                summary = self._build_summary(state)
                summary["output"] = response_text + summary["output"]
                return summary
            else:
                next_param = self._find_next_parameter(state)
                if next_param:
                    next_q = await self._ask_for_parameter(next_param, state)
                    next_q["output"] = response_text + next_q["output"]
                    return next_q
                return self._build_summary(state)
        
        # Handle SAVE
        elif choice == "SAVE":
            logger.info(f"💾 User chose to save progress and handle: '{pending_input}'")
            
            # Keep collected params but clear interruption
            state._workflow_interrupted = False
            state._workflow_paused = True
            state._pending_off_topic_input = None
            conversation_state_manager.update_session(state)
            
            collected_summary = self._format_collected_params_summary(state)
            
            return {
                "agent_name": "ValidationAgent",
                "success": True,
                "workflow_paused": True,
                "pending_request": pending_input,
                "output": f"I've saved your cluster creation progress.\n\n{collected_summary}\n\nJust say **\"resume cluster\"** when you want to continue.\n\nNow let me help with: \"{pending_input}\""
            }
        
        # Unclear response - ask again more simply
        else:
            return {
                "agent_name": "ValidationAgent",
                "success": True,
                "output": f"I'm not sure what you'd like to do. Just to clarify:\n\n• Say **\"abort\"** to cancel cluster creation\n• Say **\"continue\"** to finish creating the cluster first\n• Say **\"save\"** to pause and come back later\n\nWhat would you prefer?"
            }
    
    def _log_cluster_payload(self, state: Any, step: str) -> None:
        """
        Log the current cluster creation payload for debugging.
        Args:
            state: Conversation state
            step: Current step/action name
        """
        params = state.collected_params.copy()
        
        # Build a readable payload representation
        payload = {}
        for key, value in params.items():
            # Skip internal params starting with _
            if key.startswith('_'):
                continue
            # Extract readable values from dict objects
            if isinstance(value, dict):
                payload[key] = value.get('name') or value.get('display_name') or value
            else:
                payload[key] = value
        logger.info(f"📦 PAYLOAD [{step}] Collected {len(payload)} params: {json.dumps(payload, indent=2)}")
        # Log the internal IDs separately for reference
        internal_params = {k: v for k, v in params.items() if k.startswith('_')}
        if internal_params:
            logger.debug(f"🔐 Internal IDs: {internal_params}")
    
    async def handle(self, input_text: str, state: Any) -> Dict[str, Any]:
        """
        Main entry point for handling cluster creation workflow.
        
        Enhanced with:
        - Mid-workflow interruption detection
        - Go-back option available at every step
        - Abort/Update/Continue options when off-topic detected
        
        Args:
            input_text: User's current input
            state: Conversation state
            
        Returns:
            Dict with next prompt or ready_to_execute flag
        """
        # Check for cancel/exit intent first
        cancel_result = self._check_cancel_intent(input_text, state)
        if cancel_result:
            return cancel_result
        logger.info(f"🎯 Cluster creation handler - collected params: {list(state.collected_params.keys())}")
        # Log current payload state at entry
        self._log_cluster_payload(state, "ENTRY")
        
        # Check if workflow was previously interrupted and we're handling the response
        if getattr(state, '_workflow_interrupted', False):
            interruption_response = await self._handle_workflow_interruption_response(input_text, state)
            if interruption_response:
                return interruption_response
            # If response was unrecognized, fall through to normal processing
        
        # Check for resume request (when workflow was paused)
        if getattr(state, '_workflow_paused', False):
            # Use AI to understand if user wants to resume
            is_resume = await self._check_resume_intent(input_text)
            if is_resume:
                logger.info("▶️ Resuming paused cluster creation workflow")
                state._workflow_paused = False
                conversation_state_manager.update_session(state)
                
                # Resume from where we left off
                next_param = self._find_next_parameter(state)
                if next_param:
                    summary = self._format_collected_params_summary(state)
                    result = await self._ask_for_parameter(next_param, state)
                    if summary:
                        result["output"] = f"Welcome back! Let's continue with your cluster.\n\n{summary}\n\n---\n\n{result['output']}"
                    else:
                        result["output"] = f"Welcome back! Let's continue with your cluster.\n\n{result['output']}"
                    return result
                return self._build_summary(state)
        
        # If user provided input AND we previously asked for a parameter, process it
        if input_text and hasattr(state, 'last_asked_param') and state.last_asked_param:
            logger.info(f"📥 Processing response for: {state.last_asked_param}")
            
            # Use AI to understand what the user wants
            user_intent = await self._understand_user_intent(input_text, state)
            intent_type = user_intent.get("intent_type", "answer")
            
            logger.info(f"🧠 User intent: {intent_type} - {user_intent.get('details', '')[:50]}")
            
            # Handle different intent types
            if intent_type == "greeting":
                # Handle greeting and guide back to workflow
                greeting_response = self._handle_greeting_in_workflow(input_text, state)
                # Re-ask the current question
                last_param = state.last_asked_param
                if last_param and last_param != "_confirmation":
                    next_question = await self._ask_for_parameter(last_param, state)
                    greeting_response["output"] += "\n\n" + next_question["output"]
                return greeting_response
            
            elif intent_type == "cancel":
                # User wants to cancel - delegate to cancel handler
                return self._check_cancel_intent(input_text, state) or await self._process_user_input(input_text, state)
            
            elif intent_type == "off_topic":
                # User wants to do something else - offer options
                logger.info(f"🔀 Off-topic intent detected: {user_intent.get('details')}")
                return self._build_workflow_interruption_prompt(input_text, state)
            
            elif intent_type == "change":
                # User wants to change a previous parameter
                change_target = user_intent.get("change_target")
                if change_target:
                    # Try to find the parameter to change
                    change_result = await self._handle_change_request(change_target, state)
                    if change_result:
                        return change_result
                # Fall through to special commands handler if not found
            
            # Default: treat as answer to current question
            result = await self._process_user_input(input_text, state)
            
            # Log payload after processing
            self._log_cluster_payload(state, f"AFTER_{state.last_asked_param}")
            if result:
                # Check if we should continue to next parameter (success with feedback)
                if result.get("continue_workflow"):
                    # Get the next parameter after this success
                    next_param = self._find_next_parameter(state)
                    if next_param is None:
                        return self._build_summary(state)
                    # Add summary of collected params
                    summary = self._format_collected_params_summary(state)
                    if summary:
                        result["output"] = result["output"] + "\n\n" + summary
                    # Combine success message with next question
                    next_question = await self._ask_for_parameter(next_param, state)
                    result["output"] = result["output"] + "\n\n" + next_question["output"]
                    return result
                # Error or validation failure - ask again
                return result
            else:
                # Successful processing without explicit result - continue to next param
                # Add summary of collected params
                summary = self._format_collected_params_summary(state)
                next_param = self._find_next_parameter(state)
                if next_param is None:
                    return self._build_summary(state)
                
                next_question = await self._ask_for_parameter(next_param, state)
                if summary:
                    next_question["output"] = summary + "\n\n" + next_question["output"]
                return next_question
        
        # On first entry (no last_asked_param), try to extract params from user's initial response
        # This handles the case where user provides params in response to IntentAgent's clarification
        elif input_text and not hasattr(state, 'last_asked_param'):
            logger.info(f"🔍 First entry - attempting to extract params from: '{input_text[:100]}...'")
            extracted = await self._extract_initial_params(input_text, state)
            if extracted:
                logger.info(f"✅ Extracted initial params: {list(extracted.keys())}")
        # Find the next parameter to collect (after processing or on initial call)
        next_param = self._find_next_parameter(state)
        # If all parameters collected, mark as ready
        if next_param is None:
            return self._build_summary(state)
        # Ask for the next parameter
        return await self._ask_for_parameter(next_param, state)
    
    async def _extract_initial_params(self, input_text: str, state: Any) -> Dict[str, Any]:
        """
        Extract parameters from user's initial response (before any specific param was asked).
        This handles the case where user provides params in response to IntentAgent's clarification
        Args:
            input_text: User's input text
            state: Conversation state
            
        Returns:
            Dict of extracted parameters
        """
        extracted = {}
        text_lower = input_text.lower()
        # First, normalize any existing params from IntentAgent
        self._normalize_collected_params(state)
        # If we already have clusterName from IntentAgent, validate it
        if "clusterName" in state.collected_params and not state.collected_params.get("_clusterName_validated"):
            cluster_name = state.collected_params["clusterName"]
            logger.info(f"🔍 Validating clusterName from IntentAgent: '{cluster_name}'")
            # Validate format
            if not re.match(r'^[a-zA-Z][a-zA-Z0-9-]{2,17}$', cluster_name):
                logger.info(f"❌ Name format invalid: '{cluster_name}'")
                del state.collected_params["clusterName"]
            else:
                # Check availability
                check_result = await api_executor_service.execute_operation(
                    resource_type="k8s_cluster",
                    operation="check_cluster_name",
                    params={"clusterName": cluster_name})
                # Parse response: empty data = available
                is_available = not check_result.get("data", {}).get("data", {})
                check_result = {"available": is_available, "success": check_result.get("success", False)}
                if check_result.get("available"):
                    state.collected_params["_clusterName_validated"] = True
                    extracted["clusterName"] = cluster_name
                    logger.info(f"✅ clusterName '{cluster_name}' validated and available")
                    conversation_state_manager.update_session(state)
                else:
                    logger.info(f"❌ clusterName '{cluster_name}' is not available")
                    del state.collected_params["clusterName"]
        # If we already have clusterName validated, skip extraction
        if "clusterName" not in state.collected_params:
            # Try to extract cluster name
            # Patterns: "name is X", "X as the name", "name: X", "called X", "name it X"
            name_patterns = [
                r'(?:name\s+is|named?)\s+["\']?([a-zA-Z][a-zA-Z0-9-]{2,17})["\']?',
                r'["\']?([a-zA-Z][a-zA-Z0-9-]{2,17})["\']?\s+(?:as\s+)?(?:the\s+)?name',
                r'name[:\s]+["\']?([a-zA-Z][a-zA-Z0-9-]{2,17})["\']?',
                r'(?:call|name)\s+it\s+["\']?([a-zA-Z][a-zA-Z0-9-]{2,17})["\']?']
            
            cluster_name = None
            for pattern in name_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    potential_name = match.group(1)
                    # Validate format
                    if re.match(r'^[a-zA-Z][a-zA-Z0-9-]{2,17}$', potential_name):
                        cluster_name = potential_name
                        logger.info(f"🔍 Extracted cluster name from pattern: '{cluster_name}'")
                        break
            # If we found a cluster name, validate and store it
            if cluster_name:
                logger.info(f"🔍 Checking availability for extracted name: '{cluster_name}'")
                check_result = await api_executor_service.execute_operation(
                    resource_type="k8s_cluster",
                    operation="check_cluster_name",
                    params={"clusterName": cluster_name})
                # Parse response: empty data = available
                is_available = not check_result.get("data", {}).get("data", {})
                check_result = {"available": is_available, "success": check_result.get("success", False)}
                if check_result.get("available"):
                    state.collected_params["clusterName"] = cluster_name
                    extracted["clusterName"] = cluster_name
                    logger.info(f"✅ Stored extracted clusterName: '{cluster_name}'")
                    conversation_state_manager.update_session(state)
                else:
                    logger.info(f"⚠️ Extracted name '{cluster_name}' is not available, will ask user")
        else:
            logger.info(f"✅ clusterName already collected from IntentAgent: '{state.collected_params['clusterName']}'")
            extracted["clusterName"] = state.collected_params["clusterName"]
        # Try to extract datacenter/location
        # Common location keywords that map to datacenters
        location_keywords = {
            'bengaluru': 'bengaluru',
            'bangalore': 'bengaluru',
            'blr': 'bengaluru',
            'delhi': 'delhi',
            'del': 'delhi',
            'mumbai': 'mumbai',
            'mum': 'mumbai',
            'bkc': 'mumbai-bkc',
            'chennai': 'chennai',
            'amb': 'chennai-amb',
            'cressex': 'cressex'}
        
        # Check for location mentions
        if not hasattr(state, '_detected_location') or not state._detected_location:
            detected_location = None
            for keyword, location in location_keywords.items():
                if keyword in text_lower:
                    detected_location = location
                    logger.info(f"🔍 Detected location keyword '{keyword}' -> '{location}'")
                    break
            # Store detected location for later use in datacenter selection
            if detected_location:
                state._detected_location = detected_location
                extracted["_detected_location"] = detected_location
                logger.info(f"📍 Stored detected location hint: '{detected_location}'")
                conversation_state_manager.update_session(state)
        return extracted
    
    def _find_next_parameter(self, state: Any) -> Optional[str]:
        """
        Find the next parameter that needs to be collected.
        Args:
            state: Conversation state
            
        Returns:
            Next parameter name or None if all collected
        """
        # First, normalize any alternate param names that might be in state
        self._normalize_collected_params(state)
        for param in self.workflow:
            if param not in state.collected_params:
                # Skip optional params based on user choice
                if param == "maxReplicas" and not state.collected_params.get("enableAutoscaling"):
                    continue 
                return param
        return None
    
    def _normalize_collected_params(self, state: Any) -> None:
        """
        Normalize alternate parameter names in collected_params.
        This handles cases where IntentAgent extracted params with different naming.
        Args:
            state: Conversation state (modified in place)
        """
        # Mapping of alternate names to canonical names
        aliases = {
            "cluster_name": "clusterName",
            "clustername": "clusterName",
            "name": "clusterName",
            "k8s_version": "k8sVersion",
            "kubernetes_version": "k8sVersion",
            "version": "k8sVersion",
            "data_center": "datacenter",
            "location": "_detected_location",
            "endpoint": "_detected_location",
            "worker_pool_name": "workerPoolName",
            "pool_name": "workerPoolName",
            "node_type": "nodeType",
            "replica_count": "replicaCount",
            "node_count": "replicaCount",
            "replicas": "replicaCount",
            "cni_driver": "cniDriver",
            "business_unit": "businessUnit",
            "operating_system": "operatingSystem",
            "enable_autoscaling": "enableAutoscaling",
            "max_replicas": "maxReplicas"}
    
        params_to_normalize = []
        for key in list(state.collected_params.keys()):
            canonical = aliases.get(key.lower())
            if canonical and canonical != key and canonical not in state.collected_params:
                params_to_normalize.append((key, canonical))
        for old_key, new_key in params_to_normalize:
            value = state.collected_params.pop(old_key)
            state.collected_params[new_key] = value
            logger.info(f"🔄 Normalized param in state: {old_key} → {new_key} = {value}")
    
    def _build_summary(self, state: Any) -> Dict[str, Any]:
        """
        Build final summary when all parameters are collected.
        Args:
            state: Conversation state
        Returns:
            Dict with summary and awaiting confirmation
        """
        logger.info(f"✅ All cluster creation parameters collected!")
        
        # Log final payload
        self._log_cluster_payload(state, "COMPLETE")
        # Set status to COLLECTING_PARAMS (we're awaiting confirmation)
        # Use last_asked_param = "_confirmation" to track that we're in confirmation step
        state.status = ConversationStatus.COLLECTING_PARAMS
        state.last_asked_param = "_confirmation"  # Special marker for confirmation step
        state.missing_params = []
        conversation_state_manager.update_session(state)
        params = state.collected_params
        
        # Helper to safely get name from dict or string
        def get_name(val, key='name'):
            if isinstance(val, dict):
                return val.get(key, val.get('display_name', str(val)))
            return str(val) if val else 'N/A'
        
        def format_node_type(nt):
            if not nt:
                return 'N/A'
            return nt.replace('generalPurpose', 'General Purpose').replace('computeOptimized', 'Compute Optimized').replace('memoryOptimized', 'Memory Optimized')
        
        # Build worker pools section
        worker_pools = params.get("worker_pools", [])
        
        if worker_pools:
            # Multiple worker pools
            worker_pools_section = "**Worker Pools:**\n"
            for i, pool in enumerate(worker_pools, 1):
                autoscaling = ""
                if pool.get("enableAutoscaling"):
                    autoscaling = f" (autoscaling to {pool.get('maxReplicas', pool.get('replicaCount', 1))})"
                
                worker_pools_section += f"""
**Pool {i}: `{pool.get('workerPoolName', f'pool{i}')}`**
  - Type: {format_node_type(pool.get('nodeType'))}
  - Flavor: {pool.get('flavorName', get_name(pool.get('flavor')))}
  - Storage: {pool.get('additionalStorage') or pool.get('flavor', {}).get('disk_gb', 'default')} GB
  - Nodes: {pool.get('replicaCount', 1)}{autoscaling}
"""
        else:
            # Single worker pool (legacy/fallback)
            autoscaling_info = ""
            if params.get("enableAutoscaling"):
                autoscaling_info = f" (autoscaling up to {params.get('maxReplicas', 8)} nodes)"
            
            worker_pools_section = f"""**Worker Pool:**
- **Pool Name**: `{params.get('workerPoolName', 'N/A')}`
- **Node Type**: {format_node_type(params.get('nodeType'))}
- **Flavor**: {get_name(params.get('flavor'))}
- **Storage**: {params.get('additionalStorage') or params.get('flavor', {}).get('disk_gb', 'N/A')} GB
- **Node Count**: {params.get('replicaCount', 'N/A')}{autoscaling_info}
"""
        
        summary = f"""
**🎉 Cluster Configuration Complete!**

**Basic Configuration:**
- **Cluster Name**: `{params.get('clusterName', 'N/A')}`
- **Datacenter**: {get_name(params.get('datacenter'))}
- **Kubernetes Version**: {params.get('k8sVersion', 'N/A')}
- **CNI Driver**: {params.get('cniDriver', 'N/A')}

**Network Setup:**
- **Business Unit**: {get_name(params.get('businessUnit'))}
- **Environment**: {get_name(params.get('environment'))}
- **Zone**: {get_name(params.get('zone'))}

**Operating System**: {get_name(params.get('operatingSystem'), 'display_name')}

{worker_pools_section}

**Master Nodes** (auto-configured):
- **Type**: Virtual Control Plane
- **Mode**: High Availability (3x D8 nodes)

---

⏱️ **Estimated creation time**: 5-10 minutes

**Please review the configuration above.**

Reply with:
- **"yes"** or **"proceed"** to create the cluster
- **"change [parameter]"** to modify a specific parameter (e.g., "change cluster name")
- **"cancel"** to abort cluster creation
"""
        return {
            "agent_name": "ValidationAgent",
            "success": True,
            "ready_to_execute": False,  # Not ready yet - awaiting confirmation
            "output": summary}
    
    def _check_for_special_commands(self, input_text: str, state: Any) -> Optional[Dict[str, Any]]:
        """
        Fallback handler for special situations that AI might miss.
        
        This is a safety net - most intent understanding is done by AI.
        Only handles clear cancellation requests as a backup.
        
        Args:
            input_text: User's input
            state: Conversation state 
        Returns:
            Dict with response if special command detected, None otherwise
        """
        # This method is now minimal since AI handles most understanding
        # It serves as a backup for explicit cancellation
        return None
    
    def _get_param_value_display(self, value: Any) -> str:
        """Get display string for a parameter value."""
        if value is None:
            return "Not set"
        if isinstance(value, dict):
            return value.get('name') or value.get('display_name') or str(value)
        if isinstance(value, bool):
            return "Yes" if value else "No"
        return str(value)
    
    def _clear_parameter_and_dependents(self, param_name: str, state: Any) -> None:
        """
        Clear a parameter and all its dependent parameters based on dependency chain.
        Dependency chain:
        - datacenter → businessUnit → environment → zone → operatingSystem, nodeType, flavor
        - businessUnit → environment → zone → operatingSystem, nodeType, flavor
        - environment → zone → operatingSystem, nodeType, flavor
        - zone → operatingSystem, nodeType, flavor
        - nodeType → flavor
        - k8sVersion → cniDriver (and potentially OS/flavors if they depend on version)
        Args:
            param_name: Name of parameter to clear
            state: Conversation state (modified in place)
        """
        logger.info(f"🧹 Clearing parameter '{param_name}' and its dependents")
        
        # Define dependency mapping: param -> list of dependent params to clear
        dependencies = {
            "datacenter": [
                "businessUnit", "environment", "zone", 
                "operatingSystem", "nodeType", "flavor", "additionalStorage"
            ],
            "businessUnit": [
                "environment", "zone", 
                "operatingSystem", "nodeType", "flavor", "additionalStorage"
            ],
            "environment": [
                "zone", "operatingSystem", "nodeType", "flavor", "additionalStorage"
            ],
            "zone": [
                "operatingSystem", "nodeType", "flavor", "additionalStorage"
            ],
            "k8sVersion": [
                "cniDriver"  # CNI driver depends on k8s version
            ],
            "nodeType": [
                "flavor", "additionalStorage"  # Flavor depends on node type
            ],
            "operatingSystem": [
                "nodeType", "flavor", "additionalStorage"  # Node type/flavor depend on OS
            ]
        }
        # Get list of params to clear (including the param itself)
        params_to_clear = [param_name]
        if param_name in dependencies:
            params_to_clear.extend(dependencies[param_name])
        # Clear each parameter and its internal tracking params
        for param in params_to_clear:
            if param in state.collected_params:
                del state.collected_params[param]
                logger.info(f"  ✓ Cleared: {param}")
            # Clear internal tracking params
            internal_keys = [
                f"_{param}_id",
                f"_{param}_name",
                f"_{param}_validated"]
            for key in internal_keys:
                if key in state.collected_params:
                    del state.collected_params[key]
        # Clear cached options that depend on the cleared params
        if param_name == "datacenter":
            if hasattr(state, '_business_units'):
                delattr(state, '_business_units')
            if hasattr(state, '_department_details'):
                delattr(state, '_department_details')
        elif param_name == "businessUnit":
            if hasattr(state, '_business_units'):
                delattr(state, '_business_units')
        elif param_name == "environment":
            # No specific cache to clear
            pass
        elif param_name == "zone":
            if hasattr(state, '_os_options'):
                delattr(state, '_os_options')
            if hasattr(state, '_node_types'):
                delattr(state, '_node_types')
            if hasattr(state, '_all_flavors'):
                delattr(state, '_all_flavors')
        elif param_name == "k8sVersion":
            if hasattr(state, '_k8s_versions'):
                delattr(state, '_k8s_versions')
            if hasattr(state, '_cni_drivers'):
                delattr(state, '_cni_drivers')
        elif param_name == "nodeType":
            if hasattr(state, '_node_types'):
                delattr(state, '_node_types')
            if hasattr(state, '_all_flavors'):
                delattr(state, '_all_flavors')
        logger.info(f"✅ Cleared {param_name} and {len(params_to_clear) - 1} dependent parameter(s)")
    
    async def _safe_match_selection(self,input_text: str,available_options: List[Dict[str, Any]],param_name: str) -> Optional[Dict[str, Any]]:
        """
        Safely match user selection with proper error handling.
        Args:
            input_text: User's input
            available_options: List of available options to match against
            param_name: Name of parameter being matched (for error messages)
            
        Returns:
            Dict with matched_item if successful, None if no match or error
        """
        try:
            if not available_options:
                logger.error(f"❌ No {param_name} options available")
                return None
            
            matched = await self.param_extractor.match_user_selection(input_text, available_options)
            matched_data = json.loads(matched)
            
            if not matched_data or not matched_data.get("matched"):
                logger.info(f"❌ No match found for {param_name}: '{input_text}'")
                return None
            matched_item = matched_data.get("matched_item")
            if not matched_item:
                logger.error(f"❌ LLM returned matched=true but matched_item is None for {param_name}")
                return None
            if not isinstance(matched_item, dict) or "id" not in matched_item:
                logger.error(f"❌ matched_item missing 'id' field for {param_name}: {matched_item}")
                return None
            logger.info(f"✅ Matched {param_name}: {matched_item.get('name')} (ID: {matched_item['id']})")
            return matched_item
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse LLM matching response for {param_name}: {e}")
            logger.error(f"   Raw response: {matched[:200] if 'matched' in locals() and matched else 'None'}")
            return None
        except Exception as e:
            logger.error(f"❌ Error matching {param_name}: {e}", exc_info=True)
            return None
    
    def _get_param_display_name(self, param_name: str) -> str:
        """Get user-friendly display name for parameter."""
        display_names = {
            "clusterName": "Cluster Name",
            "datacenter": "Datacenter",
            "k8sVersion": "Kubernetes Version",
            "cniDriver": "CNI Driver",
            "businessUnit": "Business Unit",
            "environment": "Environment",
            "zone": "Zone",
            "operatingSystem": "Operating System",
            "workerPoolName": "Worker Pool Name",
            "nodeType": "Node Type",
            "flavor": "Flavor",
            "additionalStorage": "Additional Storage",
            "replicaCount": "Replica Count",
            "enableAutoscaling": "Enable Autoscaling",
            "maxReplicas": "Max Replicas",
            "tags": "Tags"}
        return display_names.get(param_name, param_name)
    
    def _get_workflow_navigation_hint(self, state: Any, current_step: int) -> str:
        """
        Get a subtle hint about workflow flexibility.
        
        Kept minimal to maintain natural conversation flow.
        Users can naturally ask to change things without explicit commands.
        
        Args:
            state: Conversation state
            current_step: Current step number in workflow
            
        Returns:
            Minimal hint string or empty
        """
        # Only show a hint after a few steps to not overwhelm new users
        if current_step >= 4:
            return "\n\n*Feel free to ask if you want to change anything we've discussed.*"
        return ""
    
    def _format_collected_params_summary(self, state: Any) -> str:
        """
        Format a pretty-printed summary of collected parameters for user display.
        
        Args:
            state: Conversation state
            
        Returns:
            Formatted string with collected parameters
        """
        if not state.collected_params:
            return ""
        # Build summary
        summary_lines = ["📋 **Current Configuration:**"]
        for param in self.workflow:
            if param in state.collected_params and not param.startswith('_'):
                value = state.collected_params[param]
                display_name = self._get_param_display_name(param)
                # Format value based on type
                if isinstance(value, dict):
                    # For complex objects like flavor, show the name
                    display_value = value.get('name') or value.get('display_name') or str(value)
                elif isinstance(value, bool):
                    display_value = "Yes" if value else "No"
                else:
                    display_value = str(value)
                summary_lines.append(f"  ✓ **{display_name}**: {display_value}")
        return "\n".join(summary_lines)
    
    async def _process_user_input(self, input_text: str, state: Any) -> Optional[Dict[str, Any]]:
        """
        Process user's input for the last asked parameter.
        Args:
            input_text: User's response
            state: Conversation state
            
        Returns:
            Dict with result or None to continue
        """
        # Check for special commands first (exit, cancel, go back, etc.)
        special_cmd = self._check_for_special_commands(input_text, state)
        if special_cmd:
            return special_cmd
        last_param = state.last_asked_param
        logger.info(f"📝 Processing user input for: {last_param}")
        # Special handler for confirmation step
        if last_param == "_confirmation":
            return await self._handle_confirmation(input_text, state)
        # Delegate to specific handler methods
        handlers = {
            "clusterName": self._handle_cluster_name,
            "datacenter": self._handle_datacenter,
            "k8sVersion": self._handle_k8s_version,
            "cniDriver": self._handle_cni_driver,
            "businessUnit": self._handle_business_unit,
            "environment": self._handle_environment,
            "zone": self._handle_zone,
            "operatingSystem": self._handle_operating_system,
            "workerPoolName": self._handle_worker_pool_name,
            "nodeType": self._handle_node_type,
            "flavor": self._handle_flavor,
            "additionalStorage": self._handle_additional_storage,
            "replicaCount": self._handle_replica_count,
            "enableAutoscaling": self._handle_autoscaling,
            "maxReplicas": self._handle_max_replicas,
            "addMoreWorkerPools": self._handle_add_more_worker_pools,
            "tags": self._handle_tags
        }
        
        handler = handlers.get(last_param)
        if handler:
            return await handler(input_text, state)
        return None
    
    async def _handle_cluster_name(self, input_text: str, state: Any) -> Optional[Dict[str, Any]]:
        """Validate and collect cluster name."""
        cluster_name = input_text.strip()
        logger.info(f"🔍 Validating cluster name: '{cluster_name}'")
        # Validate format
        if not re.match(r'^[a-zA-Z][a-zA-Z0-9-]{2,17}$', cluster_name):
            logger.info(f"❌ Name format invalid")
            return {
                "agent_name": "ValidationAgent",
                "success": True,
                "output": "❌ Cluster name must start with a letter and be 3-18 characters (letters, numbers, hyphens). Please try again:"
            }
        logger.info(f"✅ Name format valid, checking availability...")
        # Check availability
        check_result = await api_executor_service.check_cluster_name_available(cluster_name)
        logger.info(f"📋 Availability check result: {check_result}")
        if not check_result.get("available"):
            logger.info(f"❌ Name already taken")
            return {
                "agent_name": "ValidationAgent",
                "success": True,
                "output": f"❌ Cluster name '{cluster_name}' is already taken. Please choose another name:"}
        
        logger.info(f"✅ Name is available, storing...")
        state.collected_params["clusterName"] = cluster_name
        logger.info(f"✅✅ Stored clusterName = '{cluster_name}', collected params now: {list(state.collected_params.keys())}")
        # Persist state after collecting parameter
        conversation_state_manager.update_session(state)
        # Return success message to user (don't return None - that skips feedback)
        return {
            "agent_name": "ValidationAgent",
            "success": True,
            "output": f"✅ Great! Cluster name **`{cluster_name}`** is available and reserved.\n\nLet me continue with the next step...",
            "continue_workflow": True }  # Signal to continue to next parameter
    
    async def _handle_datacenter(self, input_text: str, state: Any) -> Optional[Dict[str, Any]]:
        """Fetch and match datacenter selection."""
        # Fetch datacenters if not cached
        if not hasattr(state, '_datacenter_options'):
            engagement_id = await api_executor_service.get_engagement_id()
            # Get IPC engagement ID first
            ipc_engagement_id = await api_executor_service.get_ipc_engagement_id(engagement_id)
            
            # Get IKS images with datacenters
            dc_result = await api_executor_service.execute_operation(
                resource_type="k8s_cluster",
                operation="get_iks_images",
                params={"ipc_engagement_id": ipc_engagement_id})
            # Extract datacenters from images
            if dc_result.get("success") and dc_result.get("data"):
                api_data = dc_result["data"]
                if api_data.get("status") == "success" and api_data.get("data"):
                    all_images = []
                    for category, images in api_data["data"].items():
                        if isinstance(images, list):
                            all_images.extend(images)
                    # Extract unique datacenters
                    datacenters = {}
                    for img in all_images:
                        dc_id = img.get("endpointId")
                        if dc_id and dc_id not in datacenters:
                            datacenters[dc_id] = {
                                "id": dc_id,
                                "name": img.get("endpointName", f"DC-{dc_id}"),
                                "endpoint": img.get("endpoint", "")}
                    dc_result = {
                        "success": True,
                        "datacenters": list(datacenters.values()),
                        "images": all_images
                    }
                else:
                    dc_result = {"success": False, "datacenters": [], "images": []}
            else:
                dc_result = {"success": False, "datacenters": [], "images": []}
            state._datacenter_options = dc_result.get("datacenters", [])
            state._all_images = dc_result.get("images", [])
        # Check if we have datacenter options
        if not state._datacenter_options:
            logger.error("❌ No datacenter options available")
            return {
                "agent_name": "ValidationAgent",
                "success": True,
                "output": "❌ No datacenters are currently available. Please contact your administrator."}
        # Match user selection using LLM (with safe error handling)
        dc_info = await self._safe_match_selection(input_text, state._datacenter_options, "datacenter")
        if dc_info:
            state.collected_params["datacenter"] = dc_info
            state.collected_params["_datacenter_id"] = dc_info["id"]
            conversation_state_manager.update_session(state)
            return None
        else:
            return {
                "agent_name": "ValidationAgent",
                "success": True,
                "output": "❌ I couldn't match that datacenter. Please choose from the list above:"}
    
    async def _handle_k8s_version(self, input_text: str, state: Any) -> Optional[Dict[str, Any]]:
        """Match Kubernetes version selection using intelligent LLM matching."""
        if not hasattr(state, '_k8s_versions'):
            dc_id = state.collected_params["_datacenter_id"]
            # Extract k8s versions from images for this datacenter
            dc_images = [img for img in state._all_images if img.get("endpointId") == dc_id]
            import re
            version_set = set()
            for img in dc_images:
                match = re.search(r'v\d+\.\d+\.\d+', img.get("ImageName", ""))
                if match:
                    version_set.add(match.group(0))
            # Sort semantically 
            versions = sorted(list(version_set), key=lambda v: [int(x) for x in v[1:].split('.')], reverse=True)
            state._k8s_versions = versions
        # Clean input 
        cleaned_input = input_text.strip().lstrip('•·-*').strip()
        # Direct match first
        if cleaned_input in state._k8s_versions:
            state.collected_params["k8sVersion"] = cleaned_input
            conversation_state_manager.update_session(state)
            return None
        # Use LLM for intelligent matching (e.g., "1.30" → "v1.30.9")
        version_options = [{"id": v, "name": v} for v in state._k8s_versions]
        matched = await self.param_extractor.match_user_selection(cleaned_input, version_options)
        matched_data = json.loads(matched)
        if matched_data.get("matched") and matched_data.get("matched_item"):
            matched_version = matched_data["matched_item"]["id"]
            state.collected_params["k8sVersion"] = matched_version
            logger.info(f"✅ LLM matched k8s version: '{cleaned_input}' → '{matched_version}'")
            conversation_state_manager.update_session(state)
            return None
        else:
            versions_list = ", ".join(state._k8s_versions[:5])  
            return {
                "agent_name": "ValidationAgent",
                "success": True,
                "output": f"❌ Could not match '{cleaned_input}' to available versions.\n\nAvailable: {versions_list}..."}
    
    async def _handle_cni_driver(self, input_text: str, state: Any) -> Optional[Dict[str, Any]]:
        """Match CNI driver selection using intelligent LLM matching."""
        if not hasattr(state, '_cni_drivers'):
            dc_id = state.collected_params["_datacenter_id"]
            k8s_version = state.collected_params["k8sVersion"]
            
            logger.info(f"🌐 Fetching CNI drivers for endpoint {dc_id}, k8s version {k8s_version}")
            
            driver_result = await api_executor_service.execute_operation(
                resource_type="k8s_cluster",
                operation="get_network_list",
                params={"endpointId": dc_id, "k8sVersion": k8s_version})
            # Parse response to extract drivers list
            drivers = []
            if driver_result.get("success") and driver_result.get("data"):
                api_data = driver_result["data"]
                if api_data.get("status") == "success" and api_data.get("data"):
                    inner_data = api_data["data"]
                    if isinstance(inner_data, dict):
                        drivers = inner_data.get("data", [])
                    elif isinstance(inner_data, list):
                        drivers = inner_data
            
            logger.info(f"✅ Extracted {len(drivers)} CNI drivers: {drivers}")
            state._cni_drivers = drivers
        # Clean input (remove bullet chars, extra whitespace)
        cleaned_input = input_text.strip().lstrip('•·-*').strip()
        # Direct match first
        if cleaned_input in state._cni_drivers:
            state.collected_params["cniDriver"] = cleaned_input
            conversation_state_manager.update_session(state)
            return None
        # Use LLM for intelligent matching
        driver_options = [{"id": d, "name": d} for d in state._cni_drivers]
        matched = await self.param_extractor.match_user_selection(cleaned_input, driver_options)
        matched_data = json.loads(matched)
        if matched_data.get("matched") and matched_data.get("matched_item"):
            matched_driver = matched_data["matched_item"]["id"]
            state.collected_params["cniDriver"] = matched_driver
            logger.info(f"✅ LLM matched CNI driver: '{cleaned_input}' → '{matched_driver}'")
            conversation_state_manager.update_session(state)
            return None
        else:
            driver_list = ", ".join(state._cni_drivers)
            return {
                "agent_name": "ValidationAgent",
                "success": True,
                "output": f"❌ Could not match '{cleaned_input}' to available drivers.\n\nPlease select one of: {driver_list}"}
    
    async def _handle_business_unit(self, input_text: str, state: Any) -> Optional[Dict[str, Any]]:
        """Fetch and match business unit selection (filtered by selected datacenter endpoint)."""
        if not hasattr(state, '_department_details') or not state._department_details:
            # Fetch full department hierarchy (BU -> Environment -> Zone)
            logger.info(f"🏢 Fetching department details with nested hierarchy...")
            # Get IPC engagement ID
            ipc_engagement_id = await api_executor_service.get_ipc_engagement_id()
            # Get department details with nested hierarchy
            dept_result = await api_executor_service.execute_operation(
                resource_type="k8s_cluster",
                operation="get_department_details",
                params={"ipc_engagement_id": ipc_engagement_id})
            # Parse response
            if dept_result.get("success") and dept_result.get("data"):
                api_data = dept_result["data"]
                if api_data.get("status") == "success":
                    dept_data = api_data.get("data", {})
                    dept_result = {
                        "success": True,
                        "data": dept_data,
                        "departmentList": dept_data.get("departmentList", [])
                    }
                else:
                    dept_result = {"success": False, "departmentList": []}
            else:
                dept_result = {"success": False, "departmentList": []}
            logger.info(f"🏢 API result: success={dept_result.get('success')}, departments count={len(dept_result.get('departmentList', []))}")
            if not dept_result.get("success"):
                logger.error(f"❌ Failed to fetch department details: {dept_result.get('error')}")
            # Store full hierarchy for later use (environments, zones)
            state._department_details = dept_result.get("departmentList", [])
            # Filter by selected datacenter's endpoint ID
            selected_endpoint_id = state.collected_params.get("_datacenter_id")
            logger.info(f"🏢 Filtering departments for endpoint ID: {selected_endpoint_id}")
            # Filter departments that match the selected endpoint
            filtered_bus = []
            for dept in state._department_details:
                dept_endpoint_id = dept.get("endpointId")
                if dept_endpoint_id == selected_endpoint_id:
                    filtered_bus.append({
                        "id": dept["departmentId"],
                        "name": dept["departmentName"],
                        "endpoint_id": dept_endpoint_id,
                        "environmentList": dept.get("environmentList", [])})  # Keep nested data
            
            logger.info(f"🏢 Found {len(filtered_bus)} business units for endpoint {selected_endpoint_id}")
            state._business_units = filtered_bus
        # Match user selection using LLM 
        bu_info = await self._safe_match_selection(input_text, state._business_units, "businessUnit")
        if bu_info:
            # Find the full BU data with environmentList
            full_bu = next((bu for bu in state._business_units if bu["id"] == bu_info["id"]), bu_info)
            state.collected_params["businessUnit"] = full_bu
            logger.info(f"✅ Selected business unit: {full_bu['name']} (ID: {full_bu['id']}, {len(full_bu.get('environmentList', []))} environments)")
            conversation_state_manager.update_session(state)
            return None
        else:
            return {
                "agent_name": "ValidationAgent",
                "success": True,
                "output": "❌ I couldn't match that business unit. Please choose from the list above:"}
    
    async def _handle_environment(self, input_text: str, state: Any) -> Optional[Dict[str, Any]]:
        """Filter and match environment selection using nested data from selected business unit."""
        import traceback
        try:
            bu_value = state.collected_params.get("businessUnit")
            logger.info(f"🔍 _handle_environment: bu_value type={type(bu_value)}")
            # Get environments from the nested BU data
            if isinstance(bu_value, dict):
                bu_id = bu_value.get("id")
                bu_name = bu_value.get("name")
                # Get environments directly from the BU's nested data
                env_list = bu_value.get("environmentList", [])
                logger.info(f"🏢 BU '{bu_name}' has {len(env_list)} environments in nested data")
            else:
                logger.error(f"❌ businessUnit is not a dict: {bu_value}")
                return {
                    "agent_name": "ValidationAgent",
                    "success": True,
                    "output": "❌ Error: Business unit data not found. Please go back and select a business unit."}
            if not env_list:
                logger.warning(f"⚠️ No environments in BU '{bu_name}'")
                return {
                    "agent_name": "ValidationAgent",
                    "success": True,
                    "output": f"❌ No environments found for business unit '{bu_name}'. Please contact your administrator."}
            # Build options list for matching - use environmentId and environmentName
            env_options = [
                {"id": env["environmentId"], "name": env["environmentName"]} 
                for env in env_list]
            logger.info(f"🔍 Matching '{input_text}' against {len(env_options)} environments: {[e['name'] for e in env_options]}")
            matched = await self.param_extractor.match_user_selection(input_text, env_options)
            matched_data = json.loads(matched)
            if matched_data.get("matched"):
                env_info = matched_data.get("matched_item")
                # Find the full environment data including zoneList
                full_env = next((e for e in env_list if e["environmentId"] == env_info["id"]), None)
                if full_env:
                    state.collected_params["environment"] = {
                        "id": full_env["environmentId"], 
                        "name": full_env["environmentName"],
                        "zoneList": full_env.get("zoneList", [])}  # Keep zones for next step
                    state.collected_params["_environment_name"] = full_env["environmentName"]
                    state.collected_params["_department_id"] = bu_id
                    logger.info(f"✅ Selected environment: {full_env['environmentName']} (ID: {full_env['environmentId']}, {len(full_env.get('zoneList', []))} zones)")
                    conversation_state_manager.update_session(state)
                    return None
            
            return {
                "agent_name": "ValidationAgent",
                "success": True,
                "output": "❌ I couldn't match that environment. Please choose from the list above:"}
        except Exception as e:
            logger.error(f"❌ _handle_environment error: {e}")
            logger.error(f"🔍 Traceback:\n{traceback.format_exc()}")
            raise
    
    async def _handle_zone(self, input_text: str, state: Any) -> Optional[Dict[str, Any]]:
        """Match zone selection using nested data from selected environment."""
        env_value = state.collected_params.get("environment")
        logger.info(f"🔍 _handle_zone: env_value type={type(env_value)}")
        # Get zones from the nested environment data
        if isinstance(env_value, dict):
            env_name = env_value.get("name")
            # Get zones directly from the environment's nested data
            zone_list = env_value.get("zoneList", [])
            logger.info(f"🗺️ Environment '{env_name}' has {len(zone_list)} zones in nested data")
        else:
            logger.error(f"❌ environment is not a dict: {env_value}")
            return {
                "agent_name": "ValidationAgent",
                "success": True,
                "output": "❌ Error: Environment data not found. Please go back and select an environment."}
        
        if not zone_list:
            logger.warning(f"⚠️ No zones in environment '{env_name}'")
            return {
                "agent_name": "ValidationAgent",
                "success": True,
                "output": f"❌ No zones found for environment '{env_name}'. Please contact your administrator."}

        # Build options list for matching - use zoneId and zoneName
        zone_options = [
            {"id": zone["zoneId"], "name": zone["zoneName"]} 
            for zone in zone_list]
        logger.info(f"🔍 Matching '{input_text}' against {len(zone_options)} zones: {[z['name'] for z in zone_options]}")
        matched = await self.param_extractor.match_user_selection(input_text, zone_options)
        matched_data = json.loads(matched)
        if matched_data.get("matched"):
            zone_info = matched_data.get("matched_item")
            state.collected_params["zone"] = zone_info
            state.collected_params["_zone_id"] = zone_info["id"]
            logger.info(f"✅ Selected zone: {zone_info['name']} (ID: {zone_info['id']})")
            conversation_state_manager.update_session(state)
            return None
        else:
            return {
                "agent_name": "ValidationAgent",
                "success": True,
                "output": "❌ I couldn't match that zone. Please choose from the list above:"}
    
    async def _handle_operating_system(self, input_text: str, state: Any) -> Optional[Dict[str, Any]]:
        """Fetch and match operating system selection."""
        if not hasattr(state, '_os_options'):
            zone_id = state.collected_params["_zone_id"]
            k8s_version = state.collected_params["k8sVersion"]
            # Get OS images for zone
            os_result = await api_executor_service.execute_operation(
                resource_type="k8s_cluster",
                operation="get_os_images",
                params={"zoneId": zone_id})
            # Parse and filter by k8s version
            if os_result.get("success") and os_result.get("data"):
                api_data = os_result["data"]
                if api_data.get("status") == "success" and api_data.get("data"):
                    images = api_data["data"].get("image", {}).get("options", [])
                    # Filter by k8s version
                    version_patterns = [k8s_version]
                    if k8s_version.startswith("v"):
                        version_patterns.append(k8s_version[1:])
                    filtered = [
                        img for img in images 
                        if any(p in (img.get("label", "") or img.get("ImageName", "")) for p in version_patterns)]
                    # Group by osMake + osVersion
                    grouped = {}
                    for img in filtered:
                        os_make = img.get('osMake', 'Unknown')
                        os_version = img.get('osVersion', '')
                        key = f"{os_make} {os_version}".strip()
                        if key not in grouped:
                            grouped[key] = {
                                "display_name": key,
                                "os_id": img.get("id"),
                                "os_make": os_make,
                                "os_model": img.get("osModel"),
                                "os_version": os_version,
                                "hypervisor": img.get("hypervisor"),
                                "image_id": img.get("IMAGEID"),
                                "image_name": img.get("ImageName")}
                    os_result = {"success": True, "os_options": list(grouped.values())}
                else:
                    os_result = {"success": False, "os_options": []}
            else:
                os_result = {"success": False, "os_options": []}
            state._os_options = os_result.get("os_options", [])
        os_options = [{"id": i, "name": opt["display_name"]} for i, opt in enumerate(state._os_options)]
        matched = await self.param_extractor.match_user_selection(input_text, os_options)
        matched_data = json.loads(matched)
        if matched_data.get("matched"):
            os_idx = matched_data.get("matched_item")["id"]
            state.collected_params["operatingSystem"] = state._os_options[os_idx]
            conversation_state_manager.update_session(state)
            return None
        else:
            return {
                "agent_name": "ValidationAgent",
                "success": True,
                "output": "❌ I couldn't match that OS. Please choose from the list above:"}
    
    async def _handle_worker_pool_name(self, input_text: str, state: Any) -> Optional[Dict[str, Any]]:
        """Validate worker pool name format."""
        pool_name = input_text.strip().lower()
        if not re.match(r'^[a-z0-9]{1,5}$', pool_name):
            return {
                "agent_name": "ValidationAgent",
                "success": True,
                "output": "❌ Worker pool name must be 1-5 lowercase alphanumeric characters. Please try again:"}
        state.collected_params["workerPoolName"] = pool_name
        conversation_state_manager.update_session(state)
        return None
    
    def _get_node_type_display(self, node_type: str) -> str:
        """Get user-friendly display name for node type (like UI does)."""
        if not node_type:
            return node_type
        # Map API values to display names 
        display_map = {
            'generalPurpose': 'General Purpose',
            'memoryOptimized': 'Memory Optimized',
            'computeOptimized': 'Compute Optimized'}
        return display_map.get(node_type, node_type)
    
    async def _handle_node_type(self, input_text: str, state: Any) -> Optional[Dict[str, Any]]:
        """Fetch flavors and match node type selection using intelligent LLM matching."""
        if not hasattr(state, '_node_types') or not state._node_types:
            zone_id = state.collected_params["_zone_id"]
            os_model = state.collected_params["operatingSystem"].get("os_model", "ubuntu")
            # Get flavors for zone
            flavor_result = await api_executor_service.execute_operation(
                resource_type="k8s_cluster",
                operation="get_flavors",
                params={"zoneId": zone_id})
            # Parse and filter flavors
            if flavor_result.get("success") and flavor_result.get("data"):
                api_data = flavor_result["data"]
                if api_data.get("status") == "success" and api_data.get("data"):
                    all_flavors = api_data["data"].get("flavor", [])
                    # Filter by applicationType = Container (strict match)
                    container_flavors = [f for f in all_flavors if f.get("applicationType") == "Container"]
                    # Filter by OS model if provided
                    if os_model and container_flavors:
                        os_model_lower = os_model.lower()
                        filtered = [f for f in container_flavors 
                                   if os_model_lower in f.get("osModel", "").lower()]
                        if filtered:
                            container_flavors = filtered
                    # Extract unique node types (raw flavorCategory values - no normalization)
                    node_types = list(set(f.get("flavorCategory") for f in container_flavors if f.get("flavorCategory")))
                    # Format flavors
                    formatted_flavors = []
                    for flavor in container_flavors:
                        vcpu = flavor.get("vCpu", 0)
                        vram_mb = flavor.get("vRam", 0)
                        vram_gb = vram_mb // 1024 if vram_mb else 0
                        vdisk = flavor.get("vDisk", 0)
                        formatted_flavors.append({
                            "id": flavor.get("artifactId"),
                            "name": f"{vcpu} vCPU / {vram_gb} GB RAM / {vdisk} GB Storage",
                            "display_name": flavor.get("display_name", flavor.get("FlavorName")),
                            "flavor_name": flavor.get("FlavorName"),
                            "sku_code": flavor.get("skuCode"),
                            "circuit_id": flavor.get("circuitId"),
                            "vcpu": vcpu,
                            "vram_gb": vram_gb,
                            "disk_gb": vdisk,
                            "node_type": flavor.get("flavorCategory"),
                            "storage_type": flavor.get("storageType"),
                            "os_model": flavor.get("osModel")})
                    flavor_result = {
                        "success": True,
                        "node_types": node_types,
                        "flavors": formatted_flavors}
                else:
                    flavor_result = {"success": False, "node_types": [], "flavors": []}
            else:
                flavor_result = {"success": False, "node_types": [], "flavors": []}
            # Use raw node types from API (no transformation)
            state._node_types = flavor_result.get("node_types", [])
            state._all_flavors = flavor_result.get("flavors", [])
            logger.info(f"📋 Node types from API: {state._node_types}")
        # Clean input (remove bullet chars, extra whitespace)
        cleaned_input = input_text.strip().lstrip('•·-*').strip()
        # Build options - use RAW node type names (no pretty display)
        type_options = [
            {"id": nt, "name": nt}  # Use raw name as-is from API
            for nt in state._node_types]
        # Use LLM for intelligent matching
        matched = await self.param_extractor.match_user_selection(cleaned_input, type_options)
        matched_data = json.loads(matched)
        if matched_data.get("matched") and matched_data.get("matched_item"):
            matched_type = matched_data["matched_item"]["id"]
            state.collected_params["nodeType"] = matched_type
            logger.info(f"✅ LLM matched node type: '{cleaned_input}' → '{matched_type}'")
            conversation_state_manager.update_session(state)
            return None
        else:
            # Show raw node types in error message
            types_list = ", ".join(state._node_types) if state._node_types else "General Purpose, Compute Optimized, or Memory Optimized"
            return {
                "agent_name": "ValidationAgent",
                "success": True,
                "output": f"❌ Could not match '{cleaned_input}' to available types.\n\nPlease choose one of: {types_list}"}
    
    async def _handle_flavor(self, input_text: str, state: Any) -> Optional[Dict[str, Any]]:
        """
        Filter and match flavor selection by node type.
        
        Enhanced with AI-powered criteria-based selection:
        - "lowest" / "smallest" / "minimum" - selects smallest resources
        - "highest" / "largest" / "maximum" - selects largest resources
        - "cheapest" / "least expensive" - selects smallest (assumed cheapest)
        - Specific values like "8 cpu" or "32gb ram"
        """
        node_type = state.collected_params["nodeType"]
        # Filter flavors by selected node type
        filtered_flavors = [f for f in state._all_flavors if f.get("node_type") == node_type]
        logger.info(f"🔍 Filtering flavors for node type '{node_type}': {len(filtered_flavors)} flavors")
        
        if not filtered_flavors:
            return {
                "agent_name": "ValidationAgent",
                "success": True,
                "output": f"❌ No flavors available for {node_type}. Please go back and select a different node type."
            }
        
        # Check if user is using criteria-based selection
        criteria_result = await self._select_flavor_by_criteria(input_text, filtered_flavors)
        
        if criteria_result:
            state.collected_params["flavor"] = criteria_result
            logger.info(f"✅ Selected flavor by criteria: {criteria_result['name']} (ID: {criteria_result['id']})")
            conversation_state_manager.update_session(state)
            return {
                "agent_name": "ValidationAgent",
                "success": True,
                "continue_workflow": True,
                "output": f"✅ Selected **{criteria_result['name']}** based on your criteria."
            }
        
        # Fall back to standard matching
        flavor_options = [{"id": f["id"], "name": f["name"]} for f in filtered_flavors]
        matched = await self.param_extractor.match_user_selection(input_text, flavor_options)
        matched_data = json.loads(matched)
        if matched_data.get("matched"):
            flavor_id = matched_data.get("matched_item")["id"]
            flavor_info = next((f for f in filtered_flavors if f["id"] == flavor_id), None)
            if flavor_info:
                state.collected_params["flavor"] = flavor_info
                logger.info(f"✅ Selected flavor: {flavor_info['name']} (ID: {flavor_id})")
                conversation_state_manager.update_session(state)
                return None
        return {
            "agent_name": "ValidationAgent",
            "success": True,
            "output": "I couldn't find a matching flavor. Please select from the list above, or describe what you need (e.g., 'smallest option', 'at least 16GB RAM', 'lowest CPU')."
        }
    
    async def _select_flavor_by_criteria(self, user_input: str, flavors: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Use AI to select a flavor based on user's criteria.
        
        Handles criteria like:
        - "lowest" / "smallest" / "minimum" / "cheapest"
        - "highest" / "largest" / "maximum" 
        - "at least X CPU" / "minimum X GB RAM"
        - "around X vCPU" / "approximately X GB"
        
        Args:
            user_input: User's selection criteria
            flavors: List of available flavor options
            
        Returns:
            Selected flavor dict or None if not criteria-based
        """
        try:
            # Format flavors with full details for AI
            flavor_details = []
            for i, f in enumerate(flavors):
                flavor_details.append(
                    f"Option {i+1}: ID={f['id']}, {f['vcpu']} vCPU, {f['vram_gb']} GB RAM, {f['disk_gb']} GB Storage"
                )
            flavors_str = "\n".join(flavor_details)
            
            prompt = f"""You are helping select a compute flavor based on user criteria.

Available Flavors:
{flavors_str}

User's request: "{user_input}"

Analyze if the user is:
1. Using CRITERIA to select (e.g., "lowest", "smallest", "minimum", "cheapest", "least", "at least X", "around X")
2. Or just naming a specific option

If using CRITERIA:
- "lowest" / "smallest" / "minimum" / "cheapest" / "least" → Select the option with LOWEST resources (smallest vCPU, then smallest RAM)
- "highest" / "largest" / "maximum" / "most" → Select the option with HIGHEST resources
- "at least X vCPU" or "minimum X GB" → Select the SMALLEST option that meets the requirement
- "around X" / "approximately X" → Select the closest match

Respond with JSON:
- If criteria-based selection: {{"criteria_based": true, "selected_index": <1-based index>, "reason": "brief explanation"}}
- If NOT criteria-based (user naming specific option): {{"criteria_based": false}}"""

            response = await ai_service._call_chat_with_retries(
                prompt=prompt,
                temperature=0.1,
                max_tokens=200
            )
            
            # Parse response
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                
                if result.get("criteria_based") and result.get("selected_index"):
                    idx = result["selected_index"] - 1  # Convert to 0-based
                    if 0 <= idx < len(flavors):
                        logger.info(f"🎯 AI criteria selection: {result.get('reason', 'matched criteria')}")
                        return flavors[idx]
            
            return None
            
        except Exception as e:
            logger.warning(f"⚠️ Criteria-based flavor selection failed: {e}")
            return None
    
    async def _handle_additional_storage(self, input_text: str, state: Any) -> Optional[Dict[str, Any]]:
        """Handle optional additional disk storage."""
        user_response = input_text.lower().strip()
        # Get minimum storage from selected flavor
        flavor = state.collected_params.get("flavor", {})
        min_storage = flavor.get("disk_gb", 50)
        # Check for skip/no responses
        if any(word in user_response for word in ["no", "skip", "default", "none"]):
            state.collected_params["additionalStorage"] = None
            logger.info(f"⏭️ Skipping additional storage, using default: {min_storage} GB")
            conversation_state_manager.update_session(state)
            return None
        # Try to parse a number
        try:
            # Extract number from input
            import re
            numbers = re.findall(r'\d+', user_response)
            if numbers:
                storage = int(numbers[0])
                if storage > min_storage:
                    state.collected_params["additionalStorage"] = storage
                    logger.info(f"✅ Additional storage set to: {storage} GB")
                    conversation_state_manager.update_session(state)
                    return None
                else:
                    return {
                        "agent_name": "ValidationAgent",
                        "success": True,
                        "output": f"❌ Storage must be greater than the flavor's default ({min_storage} GB). Please enter a larger value or type 'skip':"}
            else:
                return {
                    "agent_name": "ValidationAgent",
                    "success": True,
                    "output": f"❌ Please enter a number greater than {min_storage} GB, or type 'skip' to use the default:"}
        except ValueError:
            return {
                "agent_name": "ValidationAgent",
                "success": True,
                "output": f"❌ Please enter a valid number greater than {min_storage} GB, or type 'skip':"}
    
    async def _handle_replica_count(self, input_text: str, state: Any) -> Optional[Dict[str, Any]]:
        """Validate replica count."""
        try:
            count = int(input_text.strip())
            if 1 <= count <= 8:
                state.collected_params["replicaCount"] = count
                conversation_state_manager.update_session(state)
                return None
            else:
                return {
                    "agent_name": "ValidationAgent",
                    "success": True,
                    "output": "❌ Replica count must be between 1 and 8. Please try again:"}
        except ValueError:
            return {
                "agent_name": "ValidationAgent",
                "success": True,
                "output": "❌ Please enter a number between 1 and 8:"}
    
    async def _handle_autoscaling(self, input_text: str, state: Any) -> Optional[Dict[str, Any]]:
        """Handle autoscaling yes/no."""
        user_response = input_text.lower().strip()
        if "yes" in user_response or "enable" in user_response:
            state.collected_params["enableAutoscaling"] = True
            conversation_state_manager.update_session(state)
            return None
        elif "no" in user_response or "skip" in user_response:
            state.collected_params["enableAutoscaling"] = False
            conversation_state_manager.update_session(state)
            return None
        else:
            return {
                "agent_name": "ValidationAgent",
                "success": True,
                "output": "❌ Please answer 'yes' or 'no' for autoscaling:"}
    
    async def _handle_max_replicas(self, input_text: str, state: Any) -> Optional[Dict[str, Any]]:
        """Validate max replicas."""
        try:
            max_count = int(input_text.strip())
            min_count = state.collected_params["replicaCount"]
            if min_count <= max_count <= 8:
                state.collected_params["maxReplicas"] = max_count
                conversation_state_manager.update_session(state)
                return None
            else:
                return {
                    "agent_name": "ValidationAgent",
                    "success": True,
                    "output": f"❌ Max replicas must be between {min_count} and 8. Please try again:"}
        except ValueError:
            min_count = state.collected_params["replicaCount"]
            return {
                "agent_name": "ValidationAgent",
                "success": True,
                "output": f"❌ Please enter a number between {min_count} and 8:"
            }
    
    async def _handle_add_more_worker_pools(self, input_text: str, state: Any) -> Optional[Dict[str, Any]]:
        """
        Handle the option to add more worker pools.
        
        If user wants to add more:
        - Save current worker pool configuration
        - Clear worker pool params from collected_params
        - Restart worker pool collection
        
        If user says no:
        - Continue to tags
        """
        # Use AI to understand user's intent
        try:
            prompt = f"""The user was asked if they want to add another worker pool to their Kubernetes cluster.

User's response: "{input_text}"

Does the user want to add another worker pool?
- "yes", "add another", "one more", "add more" → YES
- "no", "that's it", "done", "no more", "continue" → NO

Respond with ONLY: YES or NO"""

            response = await ai_service._call_chat_with_retries(
                prompt=prompt,
                temperature=0.0,
                max_tokens=10
            )
            
            wants_more = response.strip().upper() == "YES"
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to understand add more pools response: {e}")
            # Fallback to simple check
            wants_more = any(word in input_text.lower() for word in ["yes", "add", "another", "more"])
        
        if wants_more:
            # Save current worker pool to the list
            current_pool = self._extract_current_worker_pool(state)
            
            # Initialize worker_pools list if not exists
            if "worker_pools" not in state.collected_params:
                state.collected_params["worker_pools"] = []
            
            state.collected_params["worker_pools"].append(current_pool)
            pool_count = len(state.collected_params["worker_pools"])
            
            logger.info(f"✅ Saved worker pool #{pool_count}: {current_pool.get('workerPoolName')}")
            
            # Clear worker pool params to collect new pool
            for param in self.worker_pool_params:
                if param in state.collected_params:
                    del state.collected_params[param]
            
            # Clear cached options that might differ per pool
            if hasattr(state, '_node_types'):
                delattr(state, '_node_types')
            if hasattr(state, '_all_flavors'):
                delattr(state, '_all_flavors')
            
            conversation_state_manager.update_session(state)
            
            return {
                "agent_name": "ValidationAgent",
                "success": True,
                "output": f"✅ **Worker Pool #{pool_count} saved!**\n\nLet's configure the next worker pool."
            }
        else:
            # User doesn't want more pools - save current pool if not already saved
            if "worker_pools" not in state.collected_params:
                state.collected_params["worker_pools"] = []
            
            current_pool = self._extract_current_worker_pool(state)
            state.collected_params["worker_pools"].append(current_pool)
            
            # Mark addMoreWorkerPools as complete (set to False to skip in workflow)
            state.collected_params["addMoreWorkerPools"] = False
            
            pool_count = len(state.collected_params["worker_pools"])
            logger.info(f"✅ Worker pool configuration complete. Total pools: {pool_count}")
            
            conversation_state_manager.update_session(state)
            return None  # Continue to next param (tags)
    
    def _extract_current_worker_pool(self, state: Any) -> Dict[str, Any]:
        """
        Extract current worker pool configuration from state.
        
        Returns:
            Dict with worker pool configuration
        """
        pool = {}
        
        # Extract worker pool params
        pool["workerPoolName"] = state.collected_params.get("workerPoolName", "pool1")
        pool["nodeType"] = state.collected_params.get("nodeType")
        
        # Get flavor details
        flavor = state.collected_params.get("flavor", {})
        pool["flavor"] = flavor
        pool["flavorId"] = flavor.get("id")
        pool["flavorName"] = flavor.get("name")
        
        pool["additionalStorage"] = state.collected_params.get("additionalStorage")
        pool["replicaCount"] = state.collected_params.get("replicaCount", 1)
        pool["enableAutoscaling"] = state.collected_params.get("enableAutoscaling", False)
        
        if pool["enableAutoscaling"]:
            pool["maxReplicas"] = state.collected_params.get("maxReplicas", pool["replicaCount"])
        
        return pool
    
    async def _handle_tags(self, input_text: str, state: Any) -> Optional[Dict[str, Any]]:
        """Handle tags (skip for MVP)."""
        user_response = input_text.lower().strip()
        if "no" in user_response or "skip" in user_response:
            state.collected_params["tags"] = []
            conversation_state_manager.update_session(state)
            return None
        else:
            # For MVP, just skip tags
            state.collected_params["tags"] = []
            conversation_state_manager.update_session(state)
            return None
    
    async def _handle_confirmation(self, input_text: str, state: Any) -> Optional[Dict[str, Any]]:
        """
        Handle user's confirmation or edit request after review.
        Args:
            input_text: User's response (yes/no/change X/cancel)
            state: Conversation state
        Returns:
            Dict with result or None to continue
        """
        user_response = input_text.lower().strip()
        # Check for confirmation
        if any(word in user_response for word in ["yes", "proceed", "confirm", "create", "go ahead"]):
            logger.info("✅ User confirmed cluster creation")
            state.status = ConversationStatus.READY_TO_EXECUTE
            state.last_asked_param = None
            conversation_state_manager.update_session(state)
            return {
                "agent_name": "ValidationAgent",
                "success": True,
                "ready_to_execute": True,
                "output": "🚀 Creating your cluster... This will take 15-30 minutes."}
        # Check for cancellation
        elif any(word in user_response for word in ["cancel", "abort", "stop", "no"]):
            logger.info("❌ User cancelled cluster creation")
            state.status = ConversationStatus.CANCELLED
            state.last_asked_param = None
            conversation_state_manager.update_session(state)
            return {
                "agent_name": "ValidationAgent",
                "success": True,
                "ready_to_execute": False,
                "output": "❌ Cluster creation cancelled. No resources were created."}
        # Check for edit/change request
        elif "change" in user_response or "edit" in user_response or "modify" in user_response:
            # Extract which parameter to change
            param_map = {
                "cluster name": "clusterName",
                "name": "clusterName",
                "datacenter": "datacenter",
                "data center": "datacenter",
                "location": "datacenter",
                "kubernetes": "k8sVersion",
                "k8s": "k8sVersion",
                "version": "k8sVersion",
                "cni": "cniDriver",
                "network": "cniDriver",
                "business unit": "businessUnit",
                "bu": "businessUnit",
                "environment": "environment",
                "env": "environment",
                "zone": "zone",
                "operating system": "operatingSystem",
                "os": "operatingSystem",
                "worker pool": "workerPoolName",
                "pool name": "workerPoolName",
                "node type": "nodeType",
                "flavor": "flavor",
                "storage": "additionalStorage",
                "replica": "replicaCount",
                "count": "replicaCount",
                "autoscaling": "enableAutoscaling"}
            # Find which parameter user wants to change
            param_to_change = None
            for key, value in param_map.items():
                if key in user_response:
                    param_to_change = value
                    break
            if param_to_change:
                logger.info(f"🔄 User wants to change: {param_to_change}")
                # Clear the parameter and all dependent parameters
                self._clear_parameter_and_dependents(param_to_change, state)
                # Reset status to collecting
                state.status = ConversationStatus.COLLECTING_PARAMS
                state.last_asked_param = None
                conversation_state_manager.update_session(state)
                # Ask for the parameter again
                return await self._ask_for_parameter(param_to_change, state)
            else:
                return {
                    "agent_name": "ValidationAgent",
                    "success": True,
                    "output": "❌ I couldn't identify which parameter you want to change. Please specify (e.g., 'change cluster name', 'change datacenter', etc.)"}
        
        else:
            return {
                "agent_name": "ValidationAgent",
                "success": True,
                "output": "❌ Please reply with 'yes' to proceed, 'change [parameter]' to modify something, or 'cancel' to abort."}
    
    async def _ask_for_parameter(self, param_name: str, state: Any) -> Dict[str, Any]:
        """
        Ask user for a specific parameter with context and available options.
        Args:
            param_name: Name of parameter to ask for
            state: Conversation state
        Returns:
            Dict with prompt for user
        """
        logger.info(f"❓ Asking for parameter: {param_name}")
        state.last_asked_param = param_name
        # Persist state with last_asked_param so it survives restarts
        conversation_state_manager.update_session(state)
        # Build prompts with available options
        if param_name == "clusterName":
            output = "**Step 1/17**: What would you like to name your cluster?\n\n📝 Requirements: Start with a letter, 3-18 characters (letters, numbers, hyphens)"
        
        elif param_name == "datacenter":
            if not hasattr(state, '_datacenter_options'):
                engagement_id = await api_executor_service.get_engagement_id()
                # Get IPC engagement ID first
                ipc_engagement_id = await api_executor_service.get_ipc_engagement_id(engagement_id)
                # Get IKS images with datacenters
                dc_result = await api_executor_service.execute_operation(
                    resource_type="k8s_cluster",
                    operation="get_iks_images",
                    params={"ipc_engagement_id": ipc_engagement_id})
                # Extract datacenters from images
                if dc_result.get("success") and dc_result.get("data"):
                    api_data = dc_result["data"]
                    if api_data.get("status") == "success" and api_data.get("data"):
                        all_images = []
                        for category, images in api_data["data"].items():
                            if isinstance(images, list):
                                all_images.extend(images)
                        # Extract unique datacenters
                        datacenters = {}
                        for img in all_images:
                            dc_id = img.get("endpointId")
                            if dc_id and dc_id not in datacenters:
                                datacenters[dc_id] = {
                                    "id": dc_id,
                                    "name": img.get("endpointName", f"DC-{dc_id}"),
                                    "endpoint": img.get("endpoint", "")}
                        dc_result = {
                            "success": True,
                            "datacenters": list(datacenters.values()),
                            "images": all_images}
                    else:
                        dc_result = {"success": False, "datacenters": [], "images": []}
                else:
                    dc_result = {"success": False, "datacenters": [], "images": []}
                # Store in state (moved outside the if/else blocks)
                state._datacenter_options = dc_result.get("datacenters", [])
                state._all_images = dc_result.get("images", [])
            # Check if we have a detected location from initial extraction
            if hasattr(state, '_detected_location') and state._detected_location:
                detected = state._detected_location.lower()
                logger.info(f"🔍 Trying to auto-match detected location: '{detected}'")
                
                for dc in state._datacenter_options:
                    dc_name_lower = dc.get('name', '').lower()
                    if detected in dc_name_lower or dc_name_lower.startswith(detected):
                        # Found a match! Auto-select it
                        state.collected_params["datacenter"] = dc
                        state.collected_params["_datacenter_id"] = dc["id"]
                        # Clear the hint
                        del state._detected_location
                        conversation_state_manager.update_session(state)
                        logger.info(f"✅ Auto-matched datacenter: {dc['name']}")
                        # Ask for the next param instead
                        next_param = self._find_next_parameter(state)
                        if next_param:
                            return await self._ask_for_parameter(next_param, state)
                        else:
                            return self._build_summary(state)
            dc_list = "\n".join([f"  • {dc['name']}" for dc in state._datacenter_options])
            output = f"**Step 2/17**: Which data center would you like to deploy the cluster in?\n\n📍 **Available data centers:**\n{dc_list}"
        
        elif param_name == "k8sVersion":
            dc_id = state.collected_params["_datacenter_id"]
            # Extract k8s versions from images for this datacenter
            dc_images = [img for img in state._all_images if img.get("endpointId") == dc_id]
            import re
            version_set = set()
            for img in dc_images:
                match = re.search(r'v\d+\.\d+\.\d+', img.get("ImageName", ""))
                if match:
                    version_set.add(match.group(0))
            # Sort semantically (latest first)
            versions = sorted(list(version_set), key=lambda v: [int(x) for x in v[1:].split('.')], reverse=True)
            state._k8s_versions = versions
            version_list = "\n".join([f"  • {v}" for v in versions])
            output = f"**Step 3/17**: Which Kubernetes version would you like to use?\n\n🎯 **Available versions:**\n{version_list}"
        
        elif param_name == "cniDriver":
            dc_id = state.collected_params["_datacenter_id"]
            k8s_version = state.collected_params["k8sVersion"]
            logger.info(f"🌐 Fetching CNI drivers for endpoint {dc_id}, k8s version {k8s_version}")
            driver_result = await api_executor_service.execute_operation(
                resource_type="k8s_cluster",
                operation="get_network_list",
                params={"endpointId": dc_id, "k8sVersion": k8s_version})
            logger.info(f"📦 CNI API response: {driver_result}")
            # Parse response to extract drivers list
            drivers = []
            if driver_result.get("success") and driver_result.get("data"):
                api_data = driver_result["data"]
                if api_data.get("status") == "success" and api_data.get("data"):
                    inner_data = api_data["data"]
                    if isinstance(inner_data, dict):
                        drivers = inner_data.get("data", [])
                    elif isinstance(inner_data, list):
                        drivers = inner_data
            logger.info(f"✅ Extracted {len(drivers)} CNI drivers: {drivers}")
            state._cni_drivers = drivers
            driver_list = "\n".join([f"  • {d}" for d in state._cni_drivers])
            output = f"**Step 4/17**: Which CNI (Container Network Interface) driver?\n\n🌐 **Available drivers:**\n{driver_list}"
        
        elif param_name == "businessUnit":
            # Fetch full department hierarchy 
            logger.info(f"🏢 Fetching department details with nested hierarchy...")
            # Get IPC engagement ID
            ipc_engagement_id = await api_executor_service.get_ipc_engagement_id()
            # Get department details with nested hierarchy
            dept_result = await api_executor_service.execute_operation(
                resource_type="k8s_cluster",
                operation="get_department_details",
                params={"ipc_engagement_id": ipc_engagement_id})
            # Parse response
            if dept_result.get("success") and dept_result.get("data"):
                api_data = dept_result["data"]
                if api_data.get("status") == "success":
                    dept_data = api_data.get("data", {})
                    dept_result = {
                        "success": True,
                        "data": dept_data,
                        "departmentList": dept_data.get("departmentList", [])}
                else:
                    dept_result = {"success": False, "departmentList": []}
            else:
                dept_result = {"success": False, "departmentList": []}
            logger.info(f"🏢 API result: success={dept_result.get('success')}, departments count={len(dept_result.get('departmentList', []))}")
            
            if not dept_result.get("success"):
                logger.error(f"❌ Failed to fetch department details: {dept_result.get('error')}")
            # Store full hierarchy for later use (environments, zones)
            state._department_details = dept_result.get("departmentList", [])
            # Filter by selected datacenter's endpoint ID
            selected_endpoint_id = state.collected_params.get("_datacenter_id")
            datacenter_name = state.collected_params.get("datacenter", "selected")
            if isinstance(datacenter_name, dict):
                datacenter_name = datacenter_name.get("name", "selected")
            logger.info(f"🏢 Filtering departments for endpoint ID: {selected_endpoint_id}")
            # Filter departments that match the selected endpoint
            filtered_bus = []
            for dept in state._department_details:
                dept_endpoint_id = dept.get("endpointId")
                if dept_endpoint_id == selected_endpoint_id:
                    filtered_bus.append({
                        "id": dept["departmentId"],
                        "name": dept["departmentName"],
                        "endpoint_id": dept_endpoint_id,
                        "environmentList": dept.get("environmentList", [])  # Keep nested data
                    })
            
            logger.info(f"🏢 Found {len(filtered_bus)} business units for endpoint {selected_endpoint_id}")
            state._business_units = filtered_bus
            
            if filtered_bus:
                bu_list = "\n".join([f"  • {bu['name']}" for bu in filtered_bus])
                output = f"**Step 5/17**: Which business unit should this cluster belong to?\n\n🏢 **Available business units** (for {datacenter_name}):\n{bu_list}"
            else:
                output = f"**Step 5/17**: ⚠️ No business units found for datacenter {datacenter_name}.\n\nPlease contact your administrator to create a business unit for this location."
        
        elif param_name == "environment":
            # Get environments from the nested BU data
            bu_value = state.collected_params.get("businessUnit")
            logger.info(f"🔍 Asking for environment. BU value type: {type(bu_value)}")
            
            if isinstance(bu_value, dict):
                bu_name = bu_value.get("name")
                # Get environments directly from the BU's nested data
                env_list = bu_value.get("environmentList", [])
                logger.info(f"🏢 BU '{bu_name}' has {len(env_list)} environments in nested data")
            else:
                logger.error(f"❌ businessUnit is not a dict: {bu_value}")
                output = f"**Step 6/17**: ⚠️ Error: Business unit data not found. Please go back and select a business unit."
                return {"agent_name": "ValidationAgent", "success": True, "output": output}
            
            if env_list:
                env_names = "\n".join([f"  • {env['environmentName']}" for env in env_list])
                output = f"**Step 6/17**: Which environment is this cluster for?\n\n🔧 **Available environments** (for {bu_name}):\n{env_names}"
            else:
                output = f"**Step 6/17**: ⚠️ No environments found for business unit '{bu_name}'.\n\nPlease contact your administrator to create an environment for this business unit."
        
        elif param_name == "zone":
            # Get zones from the nested environment data
            env_value = state.collected_params.get("environment")
            logger.info(f"🔍 Asking for zone. Environment value type: {type(env_value)}")
            
            if isinstance(env_value, dict):
                env_name = env_value.get("name")
                # Get zones directly from the environment's nested data
                zone_list = env_value.get("zoneList", [])
                logger.info(f"🗺️ Environment '{env_name}' has {len(zone_list)} zones in nested data")
            else:
                logger.error(f"❌ environment is not a dict: {env_value}")
                output = f"**Step 7/17**: ⚠️ Error: Environment data not found. Please go back and select an environment."
                return {"agent_name": "ValidationAgent", "success": True, "output": output}
            
            if zone_list:
                zone_names = "\n".join([f"  • {z['zoneName']}" for z in zone_list])
                output = f"**Step 7/17**: Which network zone (VLAN) should the cluster use?\n\n🗺️ **Available zones** (for {env_name}):\n{zone_names}"
            else:
                output = f"**Step 7/17**: ⚠️ No zones found for environment '{env_name}'.\n\nPlease contact your administrator to create a zone for this environment."
        
        elif param_name == "operatingSystem":
            zone_id = state.collected_params["_zone_id"]
            k8s_version = state.collected_params["k8sVersion"]
            
            # Get OS images for zone
            os_result = await api_executor_service.execute_operation(
                resource_type="k8s_cluster",
                operation="get_os_images",
                params={"zoneId": zone_id}
            )
            
            # Parse and filter by k8s version
            if os_result.get("success") and os_result.get("data"):
                api_data = os_result["data"]
                if api_data.get("status") == "success" and api_data.get("data"):
                    images = api_data["data"].get("image", {}).get("options", [])
                    
                    # Filter by k8s version
                    version_patterns = [k8s_version]
                    if k8s_version.startswith("v"):
                        version_patterns.append(k8s_version[1:])
                    
                    filtered = [
                        img for img in images 
                        if any(p in (img.get("label", "") or img.get("ImageName", "")) for p in version_patterns)
                    ]
                    
                    # Group by osMake + osVersion
                    grouped = {}
                    for img in filtered:
                        os_make = img.get('osMake', 'Unknown')
                        os_version = img.get('osVersion', '')
                        key = f"{os_make} {os_version}".strip()
                        
                        if key not in grouped:
                            grouped[key] = {
                                "display_name": key,
                                "os_id": img.get("id"),
                                "os_make": os_make,
                                "os_model": img.get("osModel"),
                                "os_version": os_version,
                                "hypervisor": img.get("hypervisor"),
                                "image_id": img.get("IMAGEID"),
                                "image_name": img.get("ImageName")
                            }
                    
                    os_result = {"success": True, "os_options": list(grouped.values())}
                else:
                    os_result = {"success": False, "os_options": []}
            else:
                os_result = {"success": False, "os_options": []}
            state._os_options = os_result.get("os_options", [])
            
            os_list = "\n".join([f"  • {opt['display_name']}" for opt in state._os_options])
            output = f"**Step 8/17**: Which operating system for the worker nodes?\n\n💿 **Available OS options:**\n{os_list}"
        
        elif param_name == "workerPoolName":
            output = "**Step 9/17**: What would you like to name this worker node pool?\n\n📝 Requirements: 1-5 lowercase alphanumeric characters"
        
        elif param_name == "nodeType":
            zone_id = state.collected_params["_zone_id"]
            os_model = state.collected_params["operatingSystem"].get("os_model", "ubuntu")
            # Get flavors for zone
            flavor_result = await api_executor_service.execute_operation(
                resource_type="k8s_cluster",
                operation="get_flavors",
                params={"zoneId": zone_id}
            )
            
            # Parse and filter flavors
            if flavor_result.get("success") and flavor_result.get("data"):
                api_data = flavor_result["data"]
                if api_data.get("status") == "success" and api_data.get("data"):
                    all_flavors = api_data["data"].get("flavor", [])
                    
                    # Filter by applicationType = Container (strict match)
                    container_flavors = [f for f in all_flavors if f.get("applicationType") == "Container"]
                    
                    # Filter by OS model if provided
                    if os_model and container_flavors:
                        os_model_lower = os_model.lower()
                        filtered = [f for f in container_flavors 
                                   if os_model_lower in f.get("osModel", "").lower()]
                        if filtered:
                            container_flavors = filtered
                    
                    # Extract unique node types (raw flavorCategory values - no normalization)
                    node_types = list(set(f.get("flavorCategory") for f in container_flavors if f.get("flavorCategory")))
                    
                    # Format flavors
                    formatted_flavors = []
                    for flavor in container_flavors:
                        vcpu = flavor.get("vCpu", 0)
                        vram_mb = flavor.get("vRam", 0)
                        vram_gb = vram_mb // 1024 if vram_mb else 0
                        vdisk = flavor.get("vDisk", 0)
                        
                        formatted_flavors.append({
                            "id": flavor.get("artifactId"),
                            "name": f"{vcpu} vCPU / {vram_gb} GB RAM / {vdisk} GB Storage",
                            "display_name": flavor.get("display_name", flavor.get("FlavorName")),
                            "flavor_name": flavor.get("FlavorName"),
                            "sku_code": flavor.get("skuCode"),
                            "circuit_id": flavor.get("circuitId"),
                            "vcpu": vcpu,
                            "vram_gb": vram_gb,
                            "disk_gb": vdisk,
                            "node_type": flavor.get("flavorCategory"),
                            "storage_type": flavor.get("storageType"),
                            "os_model": flavor.get("osModel")
                        })
                    
                    flavor_result = {
                        "success": True,
                        "node_types": node_types,
                        "flavors": formatted_flavors
                    }
                else:
                    flavor_result = {"success": False, "node_types": [], "flavors": []}
            else:
                flavor_result = {"success": False, "node_types": [], "flavors": []}
            
            # Use raw node types from API (no transformation)
            state._node_types = flavor_result.get("node_types", [])
            state._all_flavors = flavor_result.get("flavors", [])
            
            logger.info(f"📋 Node types from API: {state._node_types}")
            
            if state._node_types:
                # Show RAW node types as-is from API (no pretty display)
                type_list = "\n".join([f"  • {t}" for t in state._node_types])
            else:
                # Fallback if API returned no node types
                type_list = "  • No node types available - please check zone/OS selection"
                logger.warning("⚠️ No node types returned from flavor API!")
            
            output = f"**Step 10/17**: What type of worker nodes do you need?\n\n💻 **Available types:**\n{type_list}"
        
        elif param_name == "flavor":
            node_type = state.collected_params["nodeType"]
            filtered_flavors = [f for f in state._all_flavors if f.get("node_type") == node_type]
            
            logger.info(f"📋 Flavors for node type '{node_type}': {len(filtered_flavors)}")
            
            # Format: "8 vCPU / 32 GB RAM / 100 GB Storage"
            flavor_list = "\n".join([f"  • {f['name']}" for f in filtered_flavors])
            output = f"**Step 11/17**: Which compute configuration?\n\n⚡ **Available flavors** ({node_type}):\n{flavor_list}"
        
        elif param_name == "additionalStorage":
            flavor = state.collected_params.get("flavor", {})
            default_storage = flavor.get("disk_gb", 50)
            output = f"**Step 12/17**: Would you like additional disk storage?\n\n💾 Default storage: **{default_storage} GB**\n\nEnter a number greater than {default_storage} GB, or type 'skip' to use default:"
        
        elif param_name == "replicaCount":
            output = "**Step 13/17**: How many worker nodes for this pool?\n\n📊 Enter a number between 1 and 8:"
        
        elif param_name == "enableAutoscaling":
            output = "**Step 14/17**: Would you like to enable autoscaling for this pool?\n\n🔄 This allows automatic scaling based on load. Answer 'yes' or 'no':"
        
        elif param_name == "maxReplicas":
            min_count = state.collected_params["replicaCount"]
            output = f"**Step 15/17**: What should be the maximum number of replicas when autoscaling?\n\n📈 Enter a number between {min_count} and 8:"
        
        elif param_name == "addMoreWorkerPools":
            # Get count of existing pools
            existing_pools = state.collected_params.get("worker_pools", [])
            pool_count = len(existing_pools) + 1  # +1 for current pool being configured
            
            # Show summary of current pool
            current_pool = self._extract_current_worker_pool(state)
            pool_summary = f"""
**Worker Pool #{pool_count} Configuration:**
• Name: `{current_pool.get('workerPoolName')}`
• Type: {current_pool.get('nodeType')}
• Flavor: {current_pool.get('flavorName', 'N/A')}
• Nodes: {current_pool.get('replicaCount', 1)}
• Autoscaling: {'Yes' if current_pool.get('enableAutoscaling') else 'No'}
"""
            
            output = f"**Step 16/17**: Worker Pool Configuration\n{pool_summary}\n\n🔄 **Would you like to add another worker pool?**\n\nYou can have multiple worker pools with different configurations (e.g., one for general workloads, another for memory-intensive tasks).\n\nAnswer 'yes' to add another pool, or 'no' to continue."
        
        elif param_name == "tags":
            output = "**Step 17/17**: Would you like to add any tags (key-value pairs for organization)?\n\n🏷️ Answer 'yes' or 'no' (you can skip for now):"
        
        else:
            output = f"Please provide: {param_name}"
        
        # Add navigation hint to help users know they can go back
        current_step = self._get_workflow_step_number(param_name)
        nav_hint = self._get_workflow_navigation_hint(state, current_step)
        output += nav_hint
        
        return {
            "agent_name": "ValidationAgent",
            "success": True,
            "output": output
        }
