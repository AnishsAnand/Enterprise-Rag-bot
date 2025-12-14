# Function Calling Implementation - Modern AI Agent Pattern

## 🎯 Overview

This document describes the new **Function Calling** implementation - a modern AI agent pattern that simplifies the multi-agent workflow and makes it more intelligent.

## 📊 Before vs After

### Traditional Flow (Old)
```
User: "List clusters in Delhi"
    ↓
OrchestratorAgent → IntentAgent (detect intent: list + k8s_cluster)
    ↓
IntentAgent → ValidationAgent (collect missing params: endpoints)
    ↓
ValidationAgent asks LLM to match "Delhi" → endpoint ID 11
    ↓
ValidationAgent → ExecutionAgent
    ↓
ExecutionAgent → API call → Response
```

**Issues:**
- 4 agents involved (Orchestrator → Intent → Validation → Execution)
- Each agent needs separate LLM calls
- Rigid parameter collection flow
- Hard to handle edge cases

### Modern Function Calling Flow (New)
```
User: "List clusters in Delhi"
    ↓
OrchestratorAgent → FunctionCallingAgent
    ↓
LLM with tools available:
  - list_k8s_clusters(location_names: ["Delhi"])
  - get_datacenters()
  - create_k8s_cluster(...)
    ↓
LLM decides: "I'll call list_k8s_clusters with location_names=['Delhi']"
    ↓
Function executes:
  1. Fetch datacenters
  2. Match "Delhi" → endpoint ID 11
  3. Call API
  4. Return results
    ↓
LLM sees results → Formats response for user
```

**Benefits:**
- ✅ Single agent handles everything
- ✅ LLM decides WHEN to call functions
- ✅ Automatic parameter extraction
- ✅ LLM sees API responses and can react
- ✅ Supports parallel tool calls
- ✅ ReAct pattern (reasoning traces)

## 🏗️ Architecture

### New Components

#### 1. **FunctionCallingService** (`app/services/function_calling_service.py`)

Manages function definitions and execution:

```python
# Define a function
function_calling_service.register_function(
    FunctionDefinition(
        name="list_k8s_clusters",
        description="List Kubernetes clusters in specified locations",
        parameters={
            "type": "object",
            "properties": {
                "location_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Datacenter locations (Delhi, Mumbai, etc.)"
                }
            }
        },
        handler=self._list_k8s_clusters_handler
    )
)
```

**Features:**
- OpenAI-compatible tool definitions
- Function registry
- Automatic execution with context
- Error handling

#### 2. **FunctionCallingAgent** (`app/agents/function_calling_agent.py`)

The modern agent that uses function calling:

```python
async def execute(self, input_text: str, context: Dict):
    # 1. Build system prompt with available tools
    system_prompt = self._build_system_prompt()
    
    # 2. Call LLM with tools
    response = await ai_service.chat_with_function_calling(
        messages=[...],
        tools=function_calling_service.get_tools_for_llm(),
        tool_choice="auto"
    )
    
    # 3. If LLM returns tool calls → execute them
    if response.tool_calls:
        for tool_call in response.tool_calls:
            result = await function_calling_service.execute_function(...)
            # Feed result back to LLM
    
    # 4. LLM returns final response to user
    return final_response
```

**Features:**
- Multi-turn function calling (ReAct pattern)
- Conversation history support
- Iteration limit to prevent loops
- Detailed logging

#### 3. **AIService Enhancement** (`app/services/ai_service.py`)

Added function calling support:

```python
async def chat_with_function_calling(
    self,
    messages: List[Dict],
    tools: List[Dict],
    tool_choice: str = "auto"
) -> Dict:
    """Chat with function calling support."""
    response = self.grok_client.chat.completions.create(
        model=self.current_chat_model,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice
    )
    
    # Parse tool_calls from response
    return parsed_response
```

#### 4. **Orchestrator Updates** (`app/agents/orchestrator_agent.py`)

Added routing to function calling agent:

```python
# Feature flag
self.use_function_calling = True  # Enable modern approach

# In routing logic
if self.use_function_calling and self.function_calling_agent:
    if is_resource_operation:
        return {"route": "function_calling"}
```

## 🔧 Available Functions

### 1. `list_k8s_clusters`

List Kubernetes clusters in specified locations.

**Parameters:**
- `location_names` (array, optional): Location names like ["Delhi", "Mumbai"]

**Intelligence:**
- Automatically fetches available datacenters
- Fuzzy matches location names to endpoint IDs
- If no location specified, queries all datacenters

**Example:**
```
User: "List clusters in Delhi"
→ Calls: list_k8s_clusters(location_names=["Delhi"])
→ Returns: {success: true, clusters: [...], total_count: 3}
```

### 2. `get_datacenters`

Get list of available datacenter locations.

**Parameters:** None

**Use case:**
- When user doesn't specify location
- To show available options

**Example:**
```
User: "What datacenters are available?"
→ Calls: get_datacenters()
→ Returns: {success: true, datacenters: [{id: 11, name: "Delhi"}, ...]}
```

### 3. `create_k8s_cluster`

Create a new Kubernetes cluster.

**Parameters:**
- `cluster_name` (string, required): Cluster name
- `location_name` (string, required): Datacenter location
- `cluster_size` (string, required): "small", "medium", or "large"

**Example:**
```
User: "Create a small cluster named test-cluster in Mumbai"
→ Calls: create_k8s_cluster(cluster_name="test-cluster", location_name="Mumbai", cluster_size="small")
```

## 🧪 Testing

### Run the Test Script

```bash
cd /home/unixlogin/Vayu/Enterprise-Rag-bot
python test_function_calling.py
```

### Test Cases Included

1. ✅ "List clusters in Delhi" - Location-specific query
2. ✅ "Show me clusters in Mumbai" - Alternative phrasing
3. ✅ "List all clusters" - Query all locations
4. ✅ "How many clusters are in Chennai?" - Count query
5. ✅ "What clusters do we have?" - General query

### Expected Output

```
🎯 FunctionCallingAgent executing: List clusters in Delhi...
🔄 Function calling iteration 1/5
🔧 LLM requested 1 tool call(s)
  - Tool: list_k8s_clusters with args: {"location_names": ["Delhi"]}
✅ Tool list_k8s_clusters executed: True
✅ FunctionCallingAgent completed with text response

Response: "I found 3 clusters in Delhi datacenter:
- prod-cluster-01 (status: running)
- dev-cluster-02 (status: stopped)
- test-cluster-03 (status: running)"
```

## 🎨 ReAct Pattern (Reasoning + Acting)

The system prompt includes explicit reasoning instructions:

```
**ReAct Pattern (Reasoning + Acting):**
Think step-by-step:
1. What does the user want?
2. What tool(s) do I need to call?
3. What are the arguments?
4. [Call tool]
5. What did the tool return?
6. How should I respond to the user?
```

This makes the LLM's decision-making process more transparent and reliable.

## 🔄 How to Switch Between Modes

### Use Function Calling (Default)

```python
# In orchestrator_agent.py
self.use_function_calling = True
```

### Use Traditional Flow

```python
# In orchestrator_agent.py
self.use_function_calling = False
```

## 📈 Performance Comparison

| Metric | Traditional Flow | Function Calling |
|--------|-----------------|------------------|
| **Agents involved** | 4 (Orch → Intent → Val → Exec) | 2 (Orch → FunctionCalling) |
| **LLM calls** | 3-5+ | 2-3 |
| **Parameter extraction** | Manual validation loops | Automatic |
| **Error handling** | Rigid state machine | LLM sees errors and adapts |
| **Code complexity** | High (state management) | Low (LLM decides) |
| **Extensibility** | Add new agent | Add new function |

## 🚀 Adding New Functions

### Step 1: Define the Function

```python
# In function_calling_service.py
self.register_function(
    FunctionDefinition(
        name="delete_firewall_rule",
        description="Delete a firewall rule by ID",
        parameters={
            "type": "object",
            "properties": {
                "rule_id": {
                    "type": "string",
                    "description": "Firewall rule ID to delete"
                }
            },
            "required": ["rule_id"]
        },
        handler=self._delete_firewall_handler
    )
)
```

### Step 2: Implement the Handler

```python
async def _delete_firewall_handler(
    self,
    arguments: Dict[str, Any],
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """Handler for delete_firewall_rule function."""
    from app.services.api_executor_service import api_executor_service
    
    rule_id = arguments.get("rule_id")
    
    result = await api_executor_service.execute_operation(
        resource_type="firewall",
        operation="delete",
        params={"rule_id": rule_id},
        user_roles=context.get("user_roles", [])
    )
    
    return result
```

### Step 3: Test

```python
# User query
"Delete firewall rule fw-12345"

# LLM automatically calls
delete_firewall_rule(rule_id="fw-12345")
```

That's it! No need to modify Intent/Validation/Execution agents.

## 🎯 Future Extensions

### 1. Add More Resources

Following the same pattern, add functions for:
- Kafka services
- GitLab services
- Container registries
- Jenkins instances
- PostgreSQL databases
- DocumentDB instances

### 2. LangGraph Integration (Recommended)

```python
from langgraph.graph import StateGraph

workflow = StateGraph(ConversationState)
workflow.add_node("function_calling", function_calling_agent.execute)
workflow.add_conditional_edges(...)
```

Benefits:
- Visual workflow
- Checkpointing (save conversation state)
- Easy debugging

### 3. Streaming Responses

```python
async def execute_streaming(self, input_text: str):
    yield {"type": "thinking", "text": "Analyzing your request..."}
    yield {"type": "tool_call", "name": "list_k8s_clusters"}
    yield {"type": "result", "data": clusters}
```

### 4. Parallel Tool Calls

LLM can request multiple tools at once:

```python
# User: "List clusters in Delhi and Mumbai"
tool_calls = [
    {name: "list_k8s_clusters", args: {location_names: ["Delhi"]}},
    {name: "list_k8s_clusters", args: {location_names: ["Mumbai"]}}
]

# Execute in parallel
results = await asyncio.gather(*[execute_tool(tc) for tc in tool_calls])
```

## 📚 Related Files

- `/app/services/function_calling_service.py` - Function definitions & registry
- `/app/agents/function_calling_agent.py` - Modern agent implementation
- `/app/services/ai_service.py` - Added `chat_with_function_calling()`
- `/app/agents/orchestrator_agent.py` - Added function calling routing
- `/app/agents/agent_manager.py` - Wired function calling agent
- `/test_function_calling.py` - Test script

## 🏁 Conclusion

The function calling pattern is a **modern, scalable approach** that:
- ✅ Reduces complexity (fewer agents)
- ✅ Improves intelligence (LLM decides when to act)
- ✅ Easier to extend (add functions, not agents)
- ✅ Better error handling (LLM sees results)
- ✅ Supports advanced patterns (ReAct, parallel calls)

**Status:** ✅ Implemented and ready for production use!

## 🔗 References

- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)
- [ReAct Paper](https://arxiv.org/abs/2210.03629)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)

---

**Last Updated:** 2024-12-13  
**Author:** AI Agent Implementation Team

