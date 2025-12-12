# Enterprise RAG Bot - Multi-Agent Architecture

## 🎯 Overview

The Enterprise RAG Bot uses a **multi-agent orchestration system** where specialized agents work together to handle user requests. Each agent has a specific responsibility and the flow is **sequential** based on the conversation state.

---

## 🏗️ System Architecture - The Real Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Request                             │
│                    "List clusters in Delhi"                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    📋 ORCHESTRATOR AGENT                         │
│              The Main Coordinator (Entry Point)                  │
│                                                                  │
│  Decides: What type of request is this?                         │
│  • Resource operation? → Route to Intent Agent                  │
│  • Documentation question? → Route to RAG Agent                 │
│  • Collecting parameters? → Route to Validation Agent           │
│  • Ready to execute? → Route to Execution Agent                 │
└────────────────────────────┬────────────────────────────────────┘
                             │
                ┌────────────┴────────────┐
                │                         │
        Resource Operation        Documentation Question
                │                         │
                ▼                         ▼
┌───────────────────────────┐   ┌──────────────────────┐
│   🎯 INTENT AGENT         │   │   📚 RAG AGENT       │
│                           │   │                      │
│  Detects:                 │   │  Uses vector DB to   │
│  • Resource type          │   │  answer questions    │
│  • Operation (list/create)│   │  from documentation  │
│  • Extract parameters     │   │                      │
│                           │   │  Returns answer      │
│  Returns:                 │   │  directly to user    │
│  {                        │   └──────────────────────┘
│    resource: "k8s_cluster"│            │
│    operation: "list"      │            ▼
│    params: {}             │        [END]
│    missing: ["endpoints"] │
│  }                        │
└────────────┬──────────────┘
             │
             │ If missing params found
             ▼
┌─────────────────────────────────────────────────────────────────┐
│              ✅ VALIDATION AGENT                                 │
│              Collects & Validates Parameters                     │
│                                                                  │
│  Tasks:                                                          │
│  1. Identify missing parameters                                 │
│  2. Fetch available options (e.g., list datacenters)            │
│  3. Ask user conversationally                                   │
│  4. Match user's response to valid options                      │
│  5. Validate collected parameters                               │
│                                                                  │
│  Example:                                                        │
│  "I found 5 data centers: Delhi, Mumbai, Chennai...             │
│   Which one would you like?"                                    │
│                                                                  │
│  User: "delhi"                                                   │
│                                                                  │
│  → Matches "delhi" to endpoint ID: 11                           │
│  → Adds to state: endpoints = [11]                              │
│  → Checks: All params collected? YES                            │
│  → Returns: ready_to_execute = True                             │
└────────────┬────────────────────────────────────────────────────┘
             │
             │ When ready_to_execute = True
             ▼
┌─────────────────────────────────────────────────────────────────┐
│              ⚡ EXECUTION AGENT                                  │
│              Executes the Operation                              │
│                                                                  │
│  Tasks:                                                          │
│  1. Read collected parameters from state                        │
│  2. Call APIExecutorService                                     │
│  3. Format results beautifully                                  │
│  4. Return to user                                              │
│                                                                  │
│  Example:                                                        │
│  → Calls: api_executor_service.list_clusters(endpoints=[11])    │
│  → Gets: 17 clusters in Delhi                                   │
│  → Formats: "✅ Found 17 clusters in Delhi..."                  │
│  → Updates state: COMPLETED                                     │
└────────────┬────────────────────────────────────────────────────┘
             │
             ▼
        User sees result
          [END]
```

---

## 🤖 Agent Responsibilities (What Each Agent Actually Does)

### 1. 🎭 Orchestrator Agent

**Location**: `app/agents/orchestrator_agent.py`

**Core Responsibility**: **Route requests to the right agent based on conversation state**

**System Prompt (Actual Instructions)**:
```
You are the Orchestrator Agent, the main coordinator in a multi-agent system.

Your responsibilities:
1. Route user requests to appropriate specialized agents
2. Manage conversation flow and track state
3. Decide which agent handles each step

Decision making:
- If user asks a question about documentation → RAGAgent
- If user wants to perform an action → IntentAgent → ValidationAgent → ExecutionAgent
- If unclear intent → Ask clarifying questions
- If missing parameters → Collect them conversationally
```

**When it routes to each agent**:

| Situation | Routes To | Why |
|-----------|-----------|-----|
| New resource operation request | **Intent Agent** | Need to detect what resource and operation |
| Conversation state = `COLLECTING_PARAMS` | **Validation Agent** | User is providing missing parameters |
| Conversation state = `READY_TO_EXECUTE` | **Execution Agent** | All params collected, time to execute |
| Documentation question detected | **RAG Agent** | Not a resource operation, use knowledge base |

**Example Flow in Code**:
```python
async def orchestrate(user_input, session_id, user_roles):
    # Get conversation state
    state = get_session(session_id)
    
    # ROUTING LOGIC:
    
    # Case 1: Collecting parameters?
    if state.status == COLLECTING_PARAMS and state.missing_params:
        return route_to("validation")
    
    # Case 2: Ready to execute?
    if state.status == READY_TO_EXECUTE:
        return route_to("execution")
    
    # Case 3: New request - use LLM to decide
    routing = await decide_routing(user_input)  # LLM decides: resource_op vs documentation
    
    if routing == "resource_operation":
        return route_to("intent")  # Detect intent first
    else:
        return route_to("rag")  # Answer from docs
```

---

### 2. 🎯 Intent Agent

**Location**: `app/agents/intent_agent.py`

**Core Responsibility**: **Detect what the user wants to do**

**System Prompt (Actual Instructions)**:
```
You are the Intent Agent, specialized in detecting user intent for cloud resource operations.

Your tasks:
1. Identify the resource type (k8s_cluster, firewall, etc.)
2. Identify the operation (create, read, update, delete, list)
3. Extract parameters from the user's message
4. Return structured JSON with your findings

Output Format:
{
  "intent_detected": true,
  "resource_type": "k8s_cluster",
  "operation": "list",
  "extracted_params": {},
  "confidence": 0.99
}

Examples:
- "List clusters" → resource: k8s_cluster, operation: list
- "Create a firewall rule" → resource: firewall, operation: create
- "What clusters are in Delhi?" → resource: k8s_cluster, operation: list
```

**What it does**:
1. Analyzes user input with LLM
2. Extracts structured intent data
3. Looks up required parameters from `resource_schema.json`
4. Returns intent + required params list

**Example**:
```python
User: "list clusters in delhi"

Intent Agent returns:
{
  "intent_detected": true,
  "resource_type": "k8s_cluster",
  "operation": "list",
  "required_params": ["endpoints"],  # From schema
  "extracted_params": {},  # Didn't extract endpoint IDs yet
  "confidence": 0.99
}

Orchestrator sees missing_params = ["endpoints"]
→ Routes to Validation Agent to collect endpoints
```

---

### 3. ✅ Validation Agent

**Location**: `app/agents/validation_agent.py`

**Core Responsibility**: **Collect and validate ALL missing parameters**

**System Prompt (Actual Instructions)**:
```
You are the Validation Agent, responsible for ensuring all parameters are correct and complete.

Your responsibilities:
1. Validate collected parameters against schema rules
2. Identify missing required parameters
3. Ask for missing information conversationally
4. Extract parameters from user responses
5. Fetch available options dynamically (endpoints, versions, etc.)
6. Match user's natural language to actual option values

When asking for parameters:
- Fetch available options first (e.g., call get_endpoints API)
- Present actual options to user
- Match user input intelligently (e.g., "delhi" → endpoint ID 11)
- Validate and add to conversation state

Example:
Missing data center:
"Let me check which data centers are available..."
[fetches endpoints dynamically]
"I found 5 data centers:
- Delhi
- Bengaluru
- Mumbai-BKC
- Chennai-AMB
- Cressex

Which one would you like?"

User: "delhi dc"
[matches "delhi dc" to Delhi endpoint]
"Perfect! I'll use the Delhi data center."
→ Returns ready_to_execute = True
```

**What it does**:
1. Checks what parameters are missing from conversation state
2. For "endpoints" parameter:
   - Calls `api_executor_service.get_endpoints()` to fetch available datacenters
   - Presents options to user
   - Waits for user response
   - Uses LLM to match user input ("delhi") to actual endpoint (ID: 11)
   - Adds to state: `endpoints = [11]`
3. For other parameters:
   - Asks conversationally
   - Extracts from user response
   - Validates against schema
4. When all params collected → Returns `ready_to_execute = True`

**Example Flow**:
```python
User: "list clusters in delhi"

IntentAgent → missing_params = ["endpoints"]

ValidationAgent:
1. Fetches available endpoints from API
2. Sees user's original query mentioned "delhi"
3. Matches "delhi" to endpoint ID 11
4. Adds to state: endpoints = [11]
5. Checks: All params collected? YES
6. Returns: {"ready_to_execute": True}

Orchestrator sees ready_to_execute = True
→ Routes to Execution Agent
```

---

### 4. ⚡ Execution Agent

**Location**: `app/agents/execution_agent.py`

**Core Responsibility**: **Execute the operation and format results**

**System Prompt (Actual Instructions)**:
```
You are the Execution Agent, responsible for executing validated operations on cloud resources.

Your responsibilities:
1. Execute API calls for CRUD operations
2. Handle execution results (success and errors)
3. Provide clear feedback to users
4. Format responses in a user-friendly way
5. Handle errors gracefully with helpful messages

For listing Kubernetes clusters:
- Use list_k8s_clusters tool
- Format results beautifully grouped by datacenter
- Show cluster status, versions, node count
- Add helpful emojis and formatting

When reporting success:
- Confirm what was done
- Provide key details
- Use clear formatting
```

**What it does**:
1. Reads collected parameters from conversation state
2. Calls appropriate method on `api_executor_service`
3. Formats results beautifully for user
4. Updates conversation state to `COMPLETED`

**Example**:
```python
State has:
- resource_type = "k8s_cluster"
- operation = "list"
- collected_params = {"endpoints": [11]}

Execution Agent:
1. Calls: api_executor_service.list_clusters(endpoint_ids=[11])
2. Gets: 17 clusters in Delhi
3. Formats beautiful response:

"✅ Found 17 Kubernetes Clusters
Across 1 data center

📍 Delhi (17 clusters)
✅ prod-cluster-01 | Healthy | 5 nodes | v1.28
✅ dev-cluster-02 | Healthy | 3 nodes | v1.27
..."

4. Returns to user
5. Updates state.status = COMPLETED
```

---

### 5. 📚 RAG Agent

**Location**: `app/agents/rag_agent.py`

**Core Responsibility**: **Answer documentation questions using vector database**

**When it's used**:
- User asks "How do I create a cluster?"
- User asks "What is Kubernetes?"
- User asks "Why did my deployment fail?"

**What it does**:
1. Searches vector database (Milvus) for relevant docs
2. Retrieves top-k relevant chunks
3. Uses LLM to generate answer with context
4. Returns answer with source citations

**Not used for**:
- Actual resource operations (listing, creating, etc.)
- That's handled by Intent → Validation → Execution flow

---

## 🔄 Complete Flow Example: "List clusters in Delhi"

```
Step 1: User sends request
├─ Input: "list clusters in delhi"
│
Step 2: Orchestrator receives request
├─ Checks state: No active conversation (new request)
├─ Uses LLM routing: Detects "RESOURCE_OPERATIONS"
├─ Routes to: Intent Agent
│
Step 3: Intent Agent analyzes
├─ Detects: resource="k8s_cluster", operation="list"
├─ Looks up required params: ["endpoints"]
├─ Extracted params: {} (no endpoint IDs extracted yet)
├─ Returns: {intent_detected: true, missing_params: ["endpoints"]}
│
Step 4: Orchestrator sees missing params
├─ Updates state.status = COLLECTING_PARAMS
├─ Routes to: Validation Agent
│
Step 5: Validation Agent collects parameters
├─ Fetches available endpoints from API
├─ Sees original query mentioned "delhi"
├─ Uses LLM to match "delhi" → endpoint ID 11
├─ Adds to state: endpoints = [11]
├─ Checks: All params collected? YES
├─ Returns: {ready_to_execute: true}
│
Step 6: Orchestrator sees ready_to_execute
├─ Updates state.status = EXECUTING
├─ Routes to: Execution Agent
│
Step 7: Execution Agent executes
├─ Reads state: endpoints = [11]
├─ Calls: api_executor_service.list_clusters(endpoint_ids=[11])
├─ Gets: 17 clusters
├─ Formats beautiful response with emojis and tables
├─ Updates state.status = COMPLETED
├─ Returns formatted response
│
Step 8: User sees result
└─ "✅ Found 17 Kubernetes Clusters in Delhi..."
```

---

## 🎯 Why This Sequential Flow?

### Orchestrator Doesn't Call ValidationAgent or ExecutionAgent Directly at Start

**The flow is ALWAYS**:
```
Orchestrator → Intent → (ValidationAgent if needed) → ExecutionAgent
```

**Never**:
```
Orchestrator → ValidationAgent directly  ❌
Orchestrator → ExecutionAgent directly  ❌
```

**Why?**
1. **Intent Agent** must detect what operation user wants **first**
2. Only after intent is known can we determine **which** parameters are needed
3. **ValidationAgent** can only collect parameters if it knows what operation is being performed
4. **ExecutionAgent** only executes when ALL parameters are validated and ready

**Exception**: If conversation is already in progress:
- State.status = `COLLECTING_PARAMS` → Skip Intent, go to Validation
- State.status = `READY_TO_EXECUTE` → Skip Intent/Validation, go to Execution

---

## 📊 Conversation State Management

**Location**: `app/agents/state/conversation_state.py`

**Tracks**:
```python
class ConversationState:
    session_id: str
    user_id: str
    status: ConversationStatus  # INITIATED, COLLECTING_PARAMS, READY_TO_EXECUTE, EXECUTING, COMPLETED
    resource_type: str  # "k8s_cluster"
    operation: str  # "list"
    required_params: Set[str]  # {"endpoints"}
    collected_params: Dict[str, Any]  # {"endpoints": [11]}
    missing_params: Set[str]  # {} (empty when all collected)
    conversation_history: List[Dict]
```

**State Transitions**:
```
INITIATED (new conversation)
    │
    ▼ (Intent detected)
COLLECTING_PARAMS (missing params found)
    │
    ▼ (All params collected)
READY_TO_EXECUTE
    │
    ▼ (Execution started)
EXECUTING
    │
    ▼ (Operation completed)
COMPLETED
```

---

## 🔌 API Executor Service

**Location**: `app/services/api_executor_service.py`

**Purpose**: Execute actual API calls (NOT agent logic)

**Agents call this service, never the other way around**

**Example Methods**:
```python
async def list_clusters(endpoint_ids, engagement_id)
async def get_endpoints()
async def list_endpoints()
async def execute_operation(resource_type, operation, params)
```

**Uses**: `resource_schema.json` for API endpoint URLs and configurations

---

## ✅ Summary

| Agent | When Orchestrator Calls It | Purpose |
|-------|----------------------------|---------|
| **IntentAgent** | New resource operation request | Detect resource type, operation, extract params |
| **ValidationAgent** | When state.status = `COLLECTING_PARAMS` | Collect missing parameters conversationally |
| **ExecutionAgent** | When state.status = `READY_TO_EXECUTE` | Execute the operation and format results |
| **RAGAgent** | Documentation question detected | Answer from knowledge base |

**The orchestrator routes intelligently based on conversation state, NOT in parallel!**

