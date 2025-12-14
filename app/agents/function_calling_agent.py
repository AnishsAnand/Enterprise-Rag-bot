"""
Function Calling Agent - Modern AI agent using OpenAI function calling pattern.
This agent uses tools/functions instead of multi-step intent→validation→execution flow.
"""

from typing import Any, Dict, List, Optional
import logging
import json

from app.agents.base_agent import BaseAgent
from app.services.ai_service import ai_service
from app.services.function_calling_service import function_calling_service

logger = logging.getLogger(__name__)


class FunctionCallingAgent(BaseAgent):
    """
    Modern agent that uses function calling (tool use) pattern.
    
    Instead of:
        User → IntentAgent → ValidationAgent → ExecutionAgent → API
    
    We have:
        User → FunctionCallingAgent → LLM decides → Tool Call → API
    
    Benefits:
    - LLM decides WHEN to call functions
    - Automatic parameter extraction
    - Sees API responses and can react
    - Supports parallel tool calls
    """
    
    def __init__(self):
        super().__init__(
            agent_name="FunctionCallingAgent",
            agent_description=(
                "Modern agent using OpenAI function calling. "
                "Let's LLM decide when to call tools and extract parameters automatically."
            ),
            temperature=0.1  # Low temperature for reliable function calling
        )
        
        # Don't setup traditional agent - we handle everything custom
        # self.setup_agent()
        logger.info("✅ FunctionCallingAgent initialized")
    
    async def execute(
        self,
        input_text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute agent with function calling.
        
        Flow:
        1. Build messages with system prompt + conversation history
        2. Call LLM with available tools
        3. If LLM returns tool calls → execute them → feed results back to LLM
        4. If LLM returns text → return to user
        5. Support multi-turn tool calling (ReAct pattern)
        
        Args:
            input_text: User's query
            context: Context with session_id, user_roles, conversation_history, etc.
            
        Returns:
            Dict with agent response
        """
        context = context or {}
        session_id = context.get("session_id", "default")
        user_roles = context.get("user_roles", [])
        
        try:
            logger.info(f"🎯 FunctionCallingAgent executing: {input_text[:100]}...")
            
            # Step 1: Build system prompt with available tools
            system_prompt = self._build_system_prompt()
            
            # Step 2: Build message history
            messages = self._build_messages(input_text, context, system_prompt)
            
            # Step 3: Get available tools
            tools = function_calling_service.get_tools_for_llm()
            
            # Step 4: Call LLM with tools (with retry for tool calling)
            max_iterations = 15  # Increased to support listing all managed services (9+ types)
            iteration = 0
            
            while iteration < max_iterations:
                iteration += 1
                logger.info(f"🔄 Function calling iteration {iteration}/{max_iterations}")
                
                llm_response = await ai_service.chat_with_function_calling(
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    max_tokens=2000,
                    temperature=0.1
                )
                
                if not llm_response.get("success"):
                    return {
                        "success": False,
                        "error": llm_response.get("error", "LLM call failed"),
                        "output": f"I encountered an error: {llm_response.get('error')}"
                    }
                
                # Check if LLM made tool calls
                tool_calls = llm_response.get("tool_calls")
                
                if tool_calls:
                    # LLM wants to call functions!
                    logger.info(f"🔧 LLM requested {len(tool_calls)} tool call(s)")
                    
                    # Add assistant message with tool calls to history
                    messages.append(llm_response["message"])
                    
                    # Execute each tool call
                    tool_results = []
                    for tool_call in tool_calls:
                        func_name = tool_call["function"]["name"]
                        func_args_str = tool_call["function"]["arguments"]
                        
                        try:
                            func_args = json.loads(func_args_str)
                        except json.JSONDecodeError:
                            logger.error(f"❌ Failed to parse tool arguments: {func_args_str}")
                            func_args = {}
                        
                        # Execute the function
                        tool_result = await function_calling_service.execute_function(
                            function_name=func_name,
                            arguments=func_args,
                            context={
                                "session_id": session_id,
                                "user_roles": user_roles
                            }
                        )
                        
                        tool_results.append({
                            "tool_call_id": tool_call.get("id"),
                            "function_name": func_name,
                            "result": tool_result
                        })
                        
                        # Add tool result to messages
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.get("id"),
                            "name": func_name,
                            "content": json.dumps(tool_result, indent=2)
                        })
                        
                        logger.info(f"✅ Tool {func_name} executed: {tool_result.get('success')}")
                    
                    # Continue loop - LLM will see tool results and respond
                    continue
                
                else:
                    # LLM returned final text response
                    content = llm_response.get("content", "")
                    logger.info(f"✅ FunctionCallingAgent completed with text response")
                    
                    return {
                        "success": True,
                        "output": content,
                        "agent_name": self.agent_name,
                        "iterations": iteration,
                        "function_calls_made": [
                            msg for msg in messages 
                            if msg.get("role") == "assistant" and msg.get("tool_calls")
                        ]
                    }
            
            # Max iterations reached
            logger.warning(f"⚠️ Max iterations ({max_iterations}) reached")
            return {
                "success": False,
                "error": "Max iterations reached",
                "output": "I apologize, but I couldn't complete the request within the iteration limit."
            }
            
        except Exception as e:
            logger.error(f"❌ FunctionCallingAgent failed: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "output": f"I encountered an error: {str(e)}"
            }
    
    def _build_system_prompt(self) -> str:
        """Build system prompt with tool descriptions."""
        tools_desc = function_calling_service.get_function_schemas_text()
        
        prompt = f"""You are an intelligent cloud resource management assistant with access to powerful tools.

**Your capabilities:**
- You can call functions/tools to perform actual operations on cloud infrastructure
- You can see the results of tool calls and react accordingly
- You think step-by-step about what tools to use and when
- You always provide clear, helpful responses to users

**Available Tools:**
{tools_desc}

**Guidelines:**
1. **When user asks about clusters/resources:** Use the appropriate tool to fetch real data
2. **Handle locations intelligently:** 
   - If user says "Delhi", "Mumbai", etc. → pass as location_names to list_k8s_clusters
   - If user doesn't specify location → call get_datacenters first to show options
3. **After tool calls:** Analyze the results and provide a clear summary to the user
4. **Error handling:** If a tool fails, explain the issue clearly
5. **Multi-step reasoning:** You can call multiple tools in sequence if needed

**Example interaction:**
User: "List clusters in Delhi"
You: [Call list_k8s_clusters with location_names=["Delhi"]]
[See results]
You: "I found 3 clusters in Delhi datacenter:
- prod-cluster-01 (status: running)
- dev-cluster-02 (status: stopped)
- test-cluster-03 (status: running)"

**ReAct Pattern (Reasoning + Acting):**
Think step-by-step:
1. What does the user want?
2. What tool(s) do I need to call?
3. What are the arguments?
4. [Call tool]
5. What did the tool return?
6. How should I respond to the user?

Be helpful, accurate, and conversational!"""
        
        return prompt
    
    def _build_messages(
        self,
        input_text: str,
        context: Dict[str, Any],
        system_prompt: str
    ) -> List[Dict[str, str]]:
        """Build message history for LLM."""
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add conversation history if available
        conversation_history = context.get("conversation_history", [])
        if conversation_history:
            # Add last N messages for context
            for msg in conversation_history[-10:]:  # Last 10 messages
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role in ["user", "assistant"] and content:
                    messages.append({"role": role, "content": content})
        
        # Add current user message
        messages.append({"role": "user", "content": input_text})
        
        return messages
    
    def get_system_prompt(self) -> str:
        """Return system prompt (for base class compatibility)."""
        return self._build_system_prompt()
    
    def get_tools(self) -> List:
        """Return empty list (we use function calling, not langchain tools)."""
        return []


# Global instance
function_calling_agent = FunctionCallingAgent()

