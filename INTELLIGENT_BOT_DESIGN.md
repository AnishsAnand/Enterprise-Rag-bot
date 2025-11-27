# Intelligent Bot Design - Dynamic & Conversational

## 🎯 **Philosophy: No Hardcoding, Always Ask**

The bot is now designed to be **intelligent and dynamic** rather than relying on hardcoded mappings. It:

1. **Fetches available options** from APIs in real-time
2. **Asks clarifying questions** when information is ambiguous or missing
3. **Matches user input intelligently** to actual values
4. **Maintains conversation context** across multiple turns
5. **Learns from available data** rather than pre-programmed responses

---

## 🚫 **What We Removed**

### **Before (Hardcoded):**
```python
# ❌ HARDCODED - Not scalable!
location_mapping = {
    "delhi": "Delhi",
    "bengaluru": "Bengaluru",
    "mumbai": "Mumbai-BKC",
    # ... manually maintained list
}
```

**Problems:**
- ❌ Requires code changes for new data centers
- ❌ Doesn't handle typos or variations
- ❌ No awareness of actual available options
- ❌ Can't ask for clarification
- ❌ Single-turn only (no conversation)

### **After (Dynamic & Intelligent):**
```python
# ✅ DYNAMIC - Fetches from API!
async def _fetch_available_options(self, option_type: str):
    endpoints = await api_executor_service.get_endpoints()
    # Returns actual, current data from API
```

**Benefits:**
- ✅ Always shows current options
- ✅ Handles any input intelligently
- ✅ Asks for clarification when needed
- ✅ Multi-turn conversations
- ✅ No code changes for new datacenters

---

## 🧠 **How It Works**

### **1. User Query Arrives**

```
User: "cluster in delhi dc"
```

### **2. Intent Detection** (IntentAgent)

```python
{
  "resource_type": "k8s_cluster",
  "operation": "list",
  "extracted_params": {
    "location_hint": "delhi dc"  # Natural language hint, not validated yet
  }
}
```

### **3. Parameter Validation & Collection** (ValidationAgent)

**Step 1: Fetch Available Options**
```python
# ValidationAgent calls its tool
endpoints = await self._fetch_available_options("endpoints")

# Returns:
{
  "options": [
    {"id": 11, "name": "Delhi"},
    {"id": 12, "name": "Bengaluru"},
    {"id": 14, "name": "Mumbai-BKC"},
    {"id": 162, "name": "Chennai-AMB"},
    {"id": 204, "name": "Cressex"}
  ]
}
```

**Step 2: Match User Input**
```python
# ValidationAgent matches "delhi dc" to available options
result = self._match_user_selection({
    "user_text": "delhi dc",
    "available_options": endpoints["options"]
})

# Returns:
{
  "matched": true,
  "matched_ids": [11],
  "matched_names": ["Delhi"]
}
```

**Step 3: If No Match - Ask User**
```
Bot: "I couldn't find a data center matching 'delhii'. 
Available data centers:
- Delhi
- Bengaluru
- Mumbai-BKC
- Chennai-AMB
- Cressex

Which one did you mean?"
```

### **4. Execution** (ExecutionAgent)

```python
# Now with validated endpoint ID
await api_executor_service.list_clusters(endpoint_ids=[11])
```

---

## 🔧 **New Agent Tools**

### **1. fetch_available_options**

**Purpose:** Dynamically fetch current options from APIs

**Usage:**
```python
result = await validation_agent._fetch_available_options("endpoints")
```

**Supported Types:**
- `endpoints` / `data_centers` / `dc` → Fetches current data centers
- `k8s_versions` → Fetches available Kubernetes versions
- More can be added easily!

**Output:**
```json
{
  "option_type": "endpoints",
  "count": 5,
  "options": [
    {
      "id": 11,
      "name": "Delhi",
      "description": "Data Center: Delhi"
    },
    ...
  ],
  "prompt_suggestion": "I found 5 available data centers:\n- Delhi (ID: 11)\n..."
}
```

### **2. match_user_selection_to_options**

**Purpose:** Intelligently match user's natural language to actual values

**Usage:**
```python
result = validation_agent._match_user_selection({
    "user_text": "delhi dc",
    "available_options": [...]
})
```

**Matching Logic:**
1. Check for "all" selection → Returns all IDs
2. Exact match → Returns single match
3. Partial match → Returns single or multiple
4. No match → Asks for clarification

**Examples:**

| User Input | Matches To | Result |
|------------|------------|---------|
| `"delhi"` | Delhi | ✅ Single match → ID 11 |
| `"delhi dc"` | Delhi | ✅ Partial match → ID 11 |
| `"blr"` | Bengaluru | ✅ Partial match → ID 12 |
| `"all"` | All endpoints | ✅ All IDs: [11,12,14,162,204] |
| `"delhii"` (typo) | No match | ❓ Asks for clarification |
| `"mum"` | Mumbai-BKC | ⚠️ Partial match → ID 14 |
| `"ban"` | Bengaluru, Bangalore | ⚠️ Ambiguous → Asks which one |

---

## 💬 **Conversation Flow Examples**

### **Example 1: Simple Query**

```
User: "list clusters in delhi"

Bot: [Fetches endpoints, matches "delhi" to Delhi endpoint]
     ✅ Found **13 Kubernetes clusters** in **Delhi**:
     
     📍 **Delhi** (13 clusters)
       ✅ tchl-paas-dev-vcp - 8 nodes, K8s v1.27.16
       ✅ del-bkp-dnd - 5 nodes, K8s v1.26.15
       ...
```

### **Example 2: Ambiguous Query**

```
User: "cluster in dc"

Bot: "I need to know which data center. Let me show you the available options..."
     [Fetches endpoints dynamically]
     "I found 5 data centers:
     - Delhi
     - Bengaluru
     - Mumbai-BKC
     - Chennai-AMB
     - Cressex
     
     Which one would you like? You can also say 'all'."

User: "bengaluru"

Bot: [Matches "bengaluru" to Bengaluru endpoint]
     ✅ Found **17 Kubernetes clusters** in **Bengaluru**:
     ...
```

### **Example 3: Typo/Invalid Input**

```
User: "show clusters in delhii"

Bot: "I couldn't find a data center matching 'delhii'.
     Available options: Delhi, Bengaluru, Mumbai-BKC, Chennai-AMB, Cressex
     
     Did you mean 'Delhi'?"

User: "yes"

Bot: ✅ Found **13 Kubernetes clusters** in **Delhi**:
     ...
```

### **Example 4: Multi-Step (Create Cluster)**

```
User: "create a kubernetes cluster"

Bot: "I'll help you create a Kubernetes cluster! First, which data center?"
     [Fetches and shows available data centers]

User: "mumbai"

Bot: "Great! Mumbai-BKC selected. What would you like to name your cluster?
     (Use lowercase letters, numbers, and hyphens only)"

User: "prod-api-cluster"

Bot: "Perfect name! Which Kubernetes version?"
     [Fetches available versions]
     "Available versions:
     - v1.27.16 (stable)
     - v1.28.15 (recommended)
     - v1.29.12 (latest)
     - v1.30.14"

User: "1.28.15"

Bot: "Got it! v1.28.15. Would you like High Availability (3 masters) or Standard (1 master)?"

... continues ...
```

---

## 🎨 **Key Design Principles**

### **1. Dynamic Over Static**
- ✅ Fetch from APIs
- ❌ Never hardcode

### **2. Ask, Don't Assume**
- ✅ Present options
- ❌ Guess what user meant

### **3. Intelligent Matching**
- ✅ Handle variations ("dc" = "data center")
- ✅ Fuzzy matching ("deli" → "Delhi")
- ✅ Ask for clarification when ambiguous

### **4. Conversational**
- ✅ Multi-turn conversations
- ✅ Context retention
- ✅ Natural language

### **5. Self-Documenting**
- ✅ Bot explains what it needs
- ✅ Provides examples
- ✅ Shows available options

---

## 🔄 **How to Extend**

### **Adding New Option Types**

Want to fetch "flavors" or "zones" dynamically?

```python
# In validation_agent.py, add to _fetch_available_options:

elif option_type_lower in ["flavors", "sizes"]:
    flavors = await api_executor_service.get_flavors()  # New API call
    formatted_options = []
    for flavor in flavors:
        formatted_options.append({
            "id": flavor["id"],
            "name": flavor["name"],
            "description": f"{flavor['cpu']} CPU, {flavor['ram']} RAM"
        })
    return json.dumps({
        "option_type": "flavors",
        "options": formatted_options,
        "prompt_suggestion": "Available compute flavors:\n..."
    })
```

### **Adding New Resources**

Want to add "firewall" or "load balancer" operations?

1. **Update `resource_schema.json`** with resource definition
2. **Add API calls** to `api_executor_service.py`
3. **Done!** Agents handle the rest automatically

No need to update:
- ❌ rag_widget.py (routing is generic)
- ❌ Intent detection (uses schema)
- ❌ Validation (uses schema)
- ❌ Execution (uses schema)

---

## 📊 **Comparison: Before vs After**

| Aspect | Before (Hardcoded) | After (Intelligent) |
|--------|-------------------|---------------------|
| **Data Sources** | Hardcoded in code | Fetched from APIs |
| **Scalability** | Requires code changes | Automatically adapts |
| **User Experience** | Single-turn | Multi-turn conversations |
| **Ambiguity** | Fails or guesses | Asks for clarification |
| **Typos** | Fails | Suggests corrections |
| **New Datacenters** | Code update needed | Works immediately |
| **Variations** | Limited ("delhi", "Delhi") | Unlimited ("delhi", "Delhi", "delhi dc", "DELHI", etc.) |
| **Maintainability** | High (manual sync) | Low (self-updating) |

---

## 🧪 **Testing the New System**

### **Test 1: Natural Query**
```bash
curl -X POST http://localhost:8001/api/chat/query \
  -H "Content-Type: application/json" \
  -d '{"query": "show me clusters in bengaluru"}'
```

**Expected:** Bot fetches endpoints, matches "bengaluru", shows clusters

### **Test 2: Ambiguous Query**
```bash
curl -X POST http://localhost:8001/api/chat/query \
  -H "Content-Type: application/json" \
  -d '{"query": "cluster in dc"}'
```

**Expected:** Bot asks which data center, shows all available options

### **Test 3: Typo**
```bash
curl -X POST http://localhost:8001/api/chat/query \
  -H "Content-Type: application/json" \
  -d '{"query": "clusters in delhii"}'
```

**Expected:** Bot suggests "Delhi" as correction

### **Test 4: "All" Selection**
```bash
curl -X POST http://localhost:8001/api/chat/query \
  -H "Content-Type: application/json" \
  -d '{"query": "list all clusters"}'
```

**Expected:** Bot shows clusters from all endpoints

---

## 📝 **Implementation Checklist**

- [x] **ValidationAgent Tools**
  - [x] `fetch_available_options`
  - [x] `match_user_selection_to_options`

- [x] **Removed Hardcoding**
  - [x] Location mappings in rag_widget.py
  - [x] Implicit detection (now generic)

- [x] **Updated Prompts**
  - [x] ValidationAgent system prompt
  - [x] Mentions dynamic capabilities

- [ ] **Multi-Turn Conversation** (Next Phase)
  - [ ] Conversation state persistence
  - [ ] Follow-up question handling
  - [ ] Context retention across turns

- [ ] **Create Cluster Implementation** (Next Phase)
  - [ ] Step-by-step parameter collection
  - [ ] Dynamic zone/flavor fetching
  - [ ] Confirmation flow

---

## 🎓 **For Developers**

### **Key Files**

1. **`app/agents/validation_agent.py`**
   - NEW: `_fetch_available_options()` - Fetch from APIs
   - NEW: `_match_user_selection()` - Intelligent matching

2. **`app/api/routes/rag_widget.py`**
   - REMOVED: Hardcoded `location_mapping`
   - SIMPLIFIED: Generic routing logic

3. **`app/services/api_executor_service.py`**
   - `get_endpoints()` - Fetch available data centers
   - `list_clusters(endpoint_ids)` - List with specific endpoints

4. **`app/config/resource_schema.json`**
   - Defines all resources and operations
   - Agents read this to understand what's possible

### **Adding New Dynamic Options**

Template:
```python
# 1. Add API call (if needed)
async def get_new_option(self):
    # Call API
    return formatted_data

# 2. Add to _fetch_available_options
elif option_type_lower in ["new_option", "alias"]:
    data = await self.get_new_option()
    return json.dumps({
        "option_type": "new_option",
        "options": data,
        "prompt_suggestion": "Available options:\n..."
    })

# 3. Done! Agents use it automatically
```

---

## 🚀 **Future Enhancements**

1. **ML-Based Matching**
   - Use embeddings for better fuzzy matching
   - Learn from user corrections

2. **Context-Aware Suggestions**
   - "You used Delhi last time, want to use it again?"
   - Remember user preferences

3. **Proactive Validation**
   - Check resource availability before asking
   - "Delhi is at capacity, I recommend Bengaluru"

4. **Natural Language Extraction**
   - "I need a 3-node cluster in mumbai with k8s 1.28"
   - Extract all parameters in one go

---

**Status:** ✅ Core infrastructure complete  
**Next:** Multi-turn conversation state management  
**Goal:** Truly conversational, zero-hardcoding bot  

---

*"The bot should be smart enough to ask, not assume!"*

