# Agent Architecture Explained

## Table of Contents
1. [Why Are They Called "Agents"?](#why-agents)
2. [Where LangChain Is Used](#langchain-usage)
3. [Complete Flow: User Query to Results](#complete-flow)

---

## 1. Why Are They Called "Agents"? {#why-agents}

### What is an "Agent" in AI/ML Context?

In AI terminology, an **"agent"** is a system that can:
- **Perceive** its environment (user input, conversation state, API responses)
- **Reason** about what actions to take (using LLM-based decision making)
- **Act** autonomously (execute tools, make API calls, route to other agents)
- **Learn** from context (maintain conversation history, adapt responses)

### Why Your System Uses "Agents"

Your system uses **LangChain's Agent Framework**, which provides:

1. **Autonomous Decision Making**: Each agent uses an LLM to decide:
   - Which tools to use
   - What parameters to extract
   - How to respond to the user
   - When to hand off to another agent

2. **Tool Usage**: Agents can call "tools" (functions) like:
   - `validate_parameters` - Check if data is valid
   - `fetch_available_options` - Get data from APIs
   - `execute_api_operation` - Make actual API calls
   - `query_knowledge_base` - Search documentation

3. **Multi-Agent Collaboration**: Agents work together:
   - **OrchestratorAgent** decides which agent should handle a request
   - **IntentAgent** detects what the user wants
   - **ValidationAgent** collects and validates parameters
   - **ExecutionAgent** performs the actual operation
   - **RAGAgent** answers documentation questions

4. **State Management**: Agents maintain conversation context:
   - Track collected parameters
   - Remember what's been asked
   - Manage multi-turn conversations

### Key Difference from Simple Functions

**Without Agents (Simple Functions):**
```python
def create_cluster(name, region):
    # Just executes - no reasoning
    api.create_cluster(name, region)
```

**With Agents:**
```python
# Agent reasons about:
# - Is the name valid?
# - What region options are available?
# - Does the user have permissions?
# - Should I ask for more information?
# - How should I format the response?
```

---

## 2. Where LangChain Is Used {#langchain-usage}

LangChain is used throughout the agent system, primarily in these areas:

### 2.1 BaseAgent (`app/agents/base_agent.py`)

**Core LangChain Components:**

```python
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
```

**What Each Component Does:**

1. **`ChatOpenAI`** (Lines 55-62):
   - Wraps your LLM (GPT-OSS-120B via Grok API)
   - Provides standardized interface for LLM calls
   - Handles API communication, retries, error handling

2. **`ChatPromptTemplate`** (Lines 112-117):
   - Structures prompts with system messages, user input, and conversation history
   - Ensures consistent prompt formatting across agents

3. **`create_openai_functions_agent`** (Lines 120-124):
   - Creates an agent that can use tools (functions)
   - Enables agents to decide when to call tools vs. respond directly
   - Handles tool selection and execution

4. **`AgentExecutor`** (Lines 127-135):
   - Orchestrates agent execution
   - Manages tool calls, LLM interactions, and error handling
   - Limits iterations to prevent infinite loops
   - Returns intermediate steps for debugging

5. **`ConversationBufferMemory`** (Lines 74-78):
   - Maintains conversation history
   - Stores previous messages for context
   - Enables multi-turn conversations

6. **`Tool`** (Used in all agents):
   - Wraps Python functions as callable tools
   - Provides descriptions for LLM to understand when to use them
   - Enables agents to interact with external systems

### 2.2 Agent-Specific LangChain Usage

#### IntentAgent (`app/agents/intent_agent.py`)
- Uses LangChain's agent framework to:
  - Detect user intent from natural language
  - Extract parameters using LLM reasoning
  - Call tools like `get_resource_schema`, `extract_parameters`

#### ValidationAgent (`app/agents/validation_agent.py`)
- Uses LangChain tools for:
  - `validate_parameters` - Schema validation
  - `fetch_available_options` - Dynamic option fetching
  - `match_user_selection_to_options` - Intelligent matching
  - LLM-based parameter extraction from user responses

#### ExecutionAgent (`app/agents/execution_agent.py`)
- Uses LangChain tools for:
  - `execute_api_operation` - API calls
  - `list_k8s_clusters` - Cluster listing
  - `format_success_response` - Response formatting

#### RAGAgent (`app/agents/rag_agent.py`)
- Uses LangChain tools for:
  - `query_knowledge_base` - RAG queries
  - Integrates with existing Milvus-based RAG system

#### OrchestratorAgent (`app/agents/orchestrator_agent.py`)
- Uses LangChain for:
  - Routing decisions (LLM-based routing)
  - Tool-based agent coordination
  - Conversation flow management

### 2.3 What LangChain Provides vs. What You Built

**LangChain Provides:**
- Agent framework (decision-making structure)
- Tool integration (function calling)
- Prompt templates (consistent formatting)
- Memory management (conversation history)
- LLM abstraction (standardized API calls)

**You Built:**
- Agent logic (what each agent does)
- Business rules (validation, parameter collection)
- API integration (executing operations)
- State management (conversation state)
- Routing logic (which agent handles what)

---

## 3. Complete Flow: User Query to Results {#complete-flow}

### High-Level Flow Diagram

```
User Query
    ↓
FastAPI Endpoint (/api/agent/chat)
    ↓
AgentManager.process_request()
    ↓
OrchestratorAgent.orchestrate()
    ↓
[Routing Decision - LLM-based]
    ↓
┌─────────────────────────────────────┐
│  Route: RESOURCE_OPERATIONS         │
│  OR                                 │
│  Route: DOCUMENTATION               │
└─────────────────────────────────────┘
    ↓
┌──────────────┬──────────────────────┐
│              │                      │
IntentAgent    │              RAGAgent│
    ↓          │                      │
Validation    │              [RAG Query]
Agent          │                      │
    ↓          │                      │
Execution     │              [Return Answer]
Agent          │                      │
    ↓          │                      │
[API Call]     │                      │
    ↓          │                      │
[Result]       │                      │
└──────────────┴──────────────────────┘
    ↓
Response to User
```

### Detailed Step-by-Step Flow

#### Step 1: User Sends Query

**User Input:**
```
"Create a Kubernetes cluster named prod-cluster"
```

**API Call:**
```bash
POST /api/agent/chat
{
  "message": "Create a Kubernetes cluster named prod-cluster",
  "user_id": "user123",
  "user_roles": ["admin"]
}
```

#### Step 2: FastAPI Receives Request

**File:** `app/api/routes/agent_chat.py` (Lines 60-117)

```python
@router.post("/chat")
async def agent_chat(request: AgentChatRequest):
    # Generate session ID
    session_id = request.session_id or str(uuid.uuid4())
    
    # Get agent manager
    manager = get_agent_manager(...)
    
    # Process request
    result = await manager.process_request(...)
    
    return AgentChatResponse(...)
```

**What Happens:**
- Creates/retrieves session ID for conversation tracking
- Gets AgentManager instance (initializes if needed)
- Passes request to agent system

#### Step 3: AgentManager Routes to Orchestrator

**File:** `app/agents/agent_manager.py` (Lines 95-151)

```python
async def process_request(...):
    # Process through orchestrator
    result = await self.orchestrator.orchestrate(
        user_input=user_input,
        session_id=session_id,
        user_id=user_id,
        user_roles=user_roles
    )
```

**What Happens:**
- Increments request counter
- Logs request metadata
- Delegates to OrchestratorAgent

#### Step 4: Orchestrator Makes Routing Decision

**File:** `app/agents/orchestrator_agent.py` (Lines 254-342)

**4a. Check Conversation State:**
```python
# If already collecting parameters → ValidationAgent
if state.status == ConversationStatus.COLLECTING_PARAMS:
    return {"route": "validation"}

# If ready to execute → ExecutionAgent
if state.status == ConversationStatus.READY_TO_EXECUTE:
    return {"route": "execution"}
```

**4b. LLM-Based Routing (New Conversations):**
```python
routing_prompt = """
Determine if query is:
A) RESOURCE_OPERATIONS (create, list, delete, etc.)
B) DOCUMENTATION (how-to questions, explanations)
"""

llm_response = await ai_service._call_chat_with_retries(...)

if "RESOURCE_OPERATIONS" in llm_response:
    return {"route": "intent"}
elif "DOCUMENTATION" in llm_response:
    return {"route": "rag"}
```

**Example Routing:**
- "Create cluster" → `RESOURCE_OPERATIONS` → IntentAgent
- "How do I create?" → `DOCUMENTATION` → RAGAgent
- "List clusters" → `RESOURCE_OPERATIONS` → IntentAgent

#### Step 5A: Intent Detection (Resource Operations)

**File:** `app/agents/intent_agent.py` (Lines 209-260)

**5a.1. LLM Analyzes Query:**
```python
# System prompt tells LLM to detect:
# - Resource type (k8s_cluster, firewall, etc.)
# - Operation (create, list, update, delete)
# - Extract parameters (name, region, etc.)

result = await super().execute(input_text, context)
```

**5a.2. Parse Intent:**
```python
intent_data = {
    "intent_detected": True,
    "resource_type": "k8s_cluster",
    "operation": "create",
    "extracted_params": {"name": "prod-cluster"},
    "required_params": ["clusterName", "endpoints", "k8sVersion", ...]
}
```

**5a.3. Update Conversation State:**
```python
state.set_intent(
    resource_type="k8s_cluster",
    operation="create",
    required_params=[...]
)
state.add_parameters({"name": "prod-cluster"})
```

**5a.4. Check Missing Parameters:**
```python
if state.missing_params:
    # Route to ValidationAgent
    state.status = ConversationStatus.COLLECTING_PARAMS
    validation_result = await validation_agent.execute(...)
```

#### Step 5B: Validation & Parameter Collection

**File:** `app/agents/validation_agent.py` (Lines 519-792)

**5b.1. Check Missing Parameters:**
```python
if state.missing_params:
    # Example: missing ["endpoints", "k8sVersion", "replicaCount"]
```

**5b.2. Fetch Available Options (Dynamic):**
```python
# For endpoints parameter:
endpoints_json = await self._fetch_available_options("endpoints")
# Returns: [{"id": 11, "name": "Delhi"}, {"id": 12, "name": "Bengaluru"}, ...]
```

**5b.3. Extract from User Query (LLM-based):**
```python
# If user said "clusters in delhi":
extraction_result = await self._extract_location_from_query_json(...)
# Returns: {"extracted": True, "location": "Delhi"}

# Match to actual endpoint:
match_result = await self._match_user_selection_json(...)
# Returns: {"matched": True, "matched_ids": [11], "matched_names": ["Delhi"]}
```

**5b.4. Update State:**
```python
state.add_parameter("endpoints", [11], is_valid=True)
state.add_parameter("endpoint_names", ["Delhi"], is_valid=True)
```

**5b.5. Ask for Missing Parameters:**
```python
if state.missing_params:
    # Generate conversational prompt
    response = "I need a few more details:\n"
    response += "- Kubernetes version\n"
    response += "- Number of worker nodes\n"
    return {"output": response, "ready_to_execute": False}
```

**5b.6. Check Ready to Execute:**
```python
if state.is_ready_to_execute():
    return {
        "output": "Perfect! I have all the information. Shall I proceed?",
        "ready_to_execute": True
    }
```

#### Step 5C: Execution

**File:** `app/agents/execution_agent.py` (Lines 407-542)

**5c.1. Verify Ready:**
```python
if not state.is_ready_to_execute():
    return {"error": "Not ready to execute"}
```

**5c.2. Build Payload (for create operations):**
```python
if state.operation == "create":
    payload = await self._build_cluster_create_payload(state)
    # Constructs complete API payload from collected parameters
```

**5c.3. Execute API Call:**
```python
execution_result = await api_executor_service.execute_operation(
    resource_type=state.resource_type,
    operation=state.operation,
    params=payload,
    user_roles=user_roles
)
```

**5c.4. Format Response:**
```python
if execution_result.get("success"):
    response = self._format_success_message(state, execution_result)
    # "✅ Successfully created Kubernetes cluster 'prod-cluster'..."
else:
    response = self._format_error_message(state, execution_result)
    # "❌ I couldn't complete the operation because..."
```

**5c.5. Update State:**
```python
state.set_execution_result(execution_result)
state.status = ConversationStatus.COMPLETED
```

#### Step 5D: RAG Query (Documentation)

**File:** `app/agents/rag_agent.py` (Lines 158-230)

**5d.1. Create RAG Request:**
```python
widget_req = WidgetQueryRequest(
    query=input_text,
    max_results=5,
    include_sources=True
)
```

**5d.2. Query Knowledge Base:**
```python
result = await widget_query(widget_req, background_tasks)
# Uses existing Milvus vector database
# Performs semantic search
# Generates answer from retrieved documents
```

**5d.3. Format Response:**
```python
answer = result.get("answer", "")
sources = result.get("sources", [])

# Add source citations
if sources:
    answer += "\n\n**Sources:**\n"
    for source in sources[:3]:
        answer += f"- {source['title']}\n"
```

#### Step 6: Response to User

**File:** `app/api/routes/agent_chat.py` (Lines 102-110)

```python
return AgentChatResponse(
    success=True,
    response="✅ Successfully created Kubernetes cluster...",
    session_id=session_id,
    routing="execution",
    execution_result={...},
    metadata={...}
)
```

**Final Response:**
```json
{
  "success": true,
  "response": "✅ Successfully created Kubernetes cluster 'prod-cluster'...",
  "session_id": "abc-123-def",
  "routing": "execution",
  "execution_result": {
    "success": true,
    "data": {
      "cluster_id": "cls-abc123",
      "status": "Provisioning"
    }
  }
}
```

### Multi-Turn Conversation Example

**Turn 1:**
```
User: "Create a Kubernetes cluster"
Bot: "I'll help you create a Kubernetes cluster. I need some information:
     - Cluster Name
     - Data Center location
     - Kubernetes version
     ..."
```

**Turn 2 (Same Session):**
```
User: "Name it prod-cluster-01, version 1.28, in Delhi"
Bot: "Great! I've collected:
     - Cluster Name: prod-cluster-01
     - Kubernetes version: 1.28
     - Data Center: Delhi
     I still need: Number of worker nodes..."
```

**Turn 3:**
```
User: "3 worker nodes"
Bot: "Perfect! I have all the information. Shall I proceed?"
```

**Turn 4:**
```
User: "Yes"
Bot: "✅ Successfully created Kubernetes cluster!
     - Name: prod-cluster-01
     - Status: Provisioning
     - Cluster ID: cls-abc123"
```

### State Management Throughout Flow

**ConversationState tracks:**
- `session_id` - Unique conversation identifier
- `user_id` - User identifier
- `resource_type` - What resource (k8s_cluster, firewall, etc.)
- `operation` - What operation (create, list, delete, etc.)
- `collected_params` - Parameters collected so far
- `missing_params` - Parameters still needed
- `status` - Current state (COLLECTING_PARAMS, READY_TO_EXECUTE, EXECUTING, COMPLETED)
- `conversation_history` - All messages in the conversation

**State Transitions:**
```
NEW → COLLECTING_PARAMS → READY_TO_EXECUTE → EXECUTING → COMPLETED
                                    ↓
                                 FAILED
```

---

## Summary

### Why "Agents"?
- They use LangChain's agent framework for autonomous decision-making
- They can reason about actions, use tools, and collaborate
- They maintain state and context across conversations

### LangChain Usage
- **BaseAgent**: Core agent framework, LLM integration, tool system
- **All Agents**: Use LangChain tools for function calling
- **Orchestrator**: Uses LangChain for routing decisions
- **Memory**: LangChain's ConversationBufferMemory for history

### Complete Flow
1. User query → FastAPI endpoint
2. AgentManager → OrchestratorAgent
3. Routing decision (LLM-based)
4. IntentAgent → detects intent and extracts params
5. ValidationAgent → collects missing params (multi-turn)
6. ExecutionAgent → executes API operation
7. Response → formatted and returned to user

**OR** (for documentation queries):
3. Routing → RAGAgent
4. RAGAgent → queries Milvus knowledge base
5. Response → formatted answer with sources

The system is designed to handle complex, multi-step operations conversationally, with intelligent parameter collection and validation at each step.


