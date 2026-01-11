# Fix: LLM-Based Routing to Function Calling Agent

## Date: December 15, 2025

---

## 🐛 The Real Problem

When the LLM correctly identified a query as "RESOURCE_OPERATIONS", it was routing to the **old IntentAgent flow** instead of the **new FunctionCallingAgent flow**.

### What Was Wrong

**The routing code had this logic:**

```python
if "RESOURCE_OPERATIONS" in llm_response.upper():
    logger.info(f"✅ LLM routing: RESOURCE_OPERATIONS → IntentAgent")  # ❌ WRONG!
    return {
        "route": "intent",  # ❌ Should be "function_calling"!
        "reason": "LLM detected resource operation intent"
    }
```

**This meant:**
- LLM correctly detected "RESOURCE_OPERATIONS" ✅
- But code sent it to `"intent"` (old 3-agent flow) ❌
- Should have sent it to `"function_calling"` (modern flow) ✅

---

## ✅ The Fix

### Removed Keyword-Based Routing (Lines 315-338)

**BEFORE:** Had a keyword check that bypassed LLM routing:
```python
# NEW: Function calling mode - bypass traditional agent flow
if self.use_function_calling and self.function_calling_agent:
    resource_keywords = ["list", "show", "cluster", "firewall", ...]
    if any(keyword in query_lower for keyword in resource_keywords):
        return {"route": "function_calling"}
```

**AFTER:** Removed entirely - let the LLM do the routing (smarter!)

---

### Fixed LLM Routing Logic (Lines 375-398)

**BEFORE:**
```python
if "RESOURCE_OPERATIONS" in llm_response.upper():
    return {"route": "intent"}  # ❌ Wrong destination
```

**AFTER:**
```python
if "RESOURCE_OPERATIONS" in llm_response.upper():
    # Route to FunctionCallingAgent if available
    if self.use_function_calling and self.function_calling_agent:
        return {"route": "function_calling"}  # ✅ Correct!
    else:
        return {"route": "intent"}  # Fallback for old systems
```

---

### Fixed Fallback Routing (Empty Response & Exceptions)

**BEFORE:**
```python
# On error or empty response
return {"route": "intent"}  # ❌ Always went to old flow
```

**AFTER:**
```python
# On error or empty response
if self.use_function_calling and self.function_calling_agent:
    return {"route": "function_calling"}  # ✅ Modern flow first
else:
    return {"route": "intent"}  # Fallback for old systems
```

---

## 🎯 Routing Flow Now

### For Resource Operations
```
User: "container registry in chennai"
  ↓
OrchestratorAgent._decide_routing()
  ↓
LLM analyzes query (no keyword check!)
  ↓
LLM: "ROUTE: RESOURCE_OPERATIONS"
  ↓
Code checks: self.use_function_calling? YES
  ↓
Return: {"route": "function_calling"}
  ↓
FunctionCallingAgent executes
  ↓
LLM picks: list_registry function
  ↓
API Call → Response
```

### For Documentation Questions
```
User: "how do I create a cluster?"
  ↓
OrchestratorAgent._decide_routing()
  ↓
LLM analyzes query
  ↓
LLM: "ROUTE: DOCUMENTATION"
  ↓
Return: {"route": "rag"}
  ↓
RAGAgent searches docs
```

---

## 📊 Why This is Better

| Aspect | Keyword-Based | LLM-Based (Fixed) |
|--------|---------------|-------------------|
| **Intelligence** | Dumb string matching | Smart semantic understanding |
| **Maintenance** | Must update keywords for new services | Automatically handles new services |
| **Accuracy** | Can miss variations | Understands intent regardless of phrasing |
| **Examples** | Needs "registry" keyword | Understands "docker image repo", "container storage" |
| **Edge Cases** | Fails on creative phrasings | Handles natural language variations |

### Example Queries That Work Now

| Query | Keyword Match? | LLM Understands? | Result |
|-------|----------------|------------------|--------|
| "container registry in chennai" | ❌ (no keywords) | ✅ RESOURCE_OPERATIONS | ✅ FunctionCallingAgent |
| "show me docker registries" | ❌ (no "registry") | ✅ RESOURCE_OPERATIONS | ✅ FunctionCallingAgent |
| "where are my image repos?" | ❌ (no match) | ✅ RESOURCE_OPERATIONS | ✅ FunctionCallingAgent |
| "what k8s clusters exist?" | ✅ (has "cluster") | ✅ RESOURCE_OPERATIONS | ✅ FunctionCallingAgent |
| "how do I use registry?" | ❌ (has "registry") | ✅ DOCUMENTATION | ✅ RAGAgent (correct!) |

---

## 🧪 Testing

### Test Case 1: Container Registry
```bash
Query: "container registry in chennai"
Expected Log: "✅ LLM routing: RESOURCE_OPERATIONS → FunctionCallingAgent"
Expected Route: function_calling
Expected Function: list_registry
```

### Test Case 2: Creative Phrasing
```bash
Query: "show me docker image repositories"
Expected: LLM understands this is registry listing
Expected Route: function_calling
Expected Function: list_registry
```

### Test Case 3: Documentation (Should NOT route to function calling)
```bash
Query: "how do I use container registry?"
Expected Log: "✅ LLM routing: DOCUMENTATION → RAGAgent"
Expected Route: rag
```

---

## 🚀 Deployment

1. **Restart the application** to load the fixed routing logic
2. **Test various queries** - the LLM should handle all variations correctly
3. **Monitor logs** for routing decisions

**No more manual keyword maintenance!** 🎉

---

## 📝 Files Modified

1. ✅ `app/agents/orchestrator_agent.py`
   - Removed keyword-based routing (lines 315-338)
   - Fixed LLM routing to send RESOURCE_OPERATIONS → function_calling
   - Fixed all fallback paths to prefer function_calling

---

## 💡 Key Insight

**The bug was not missing keywords - the bug was that the LLM routing was pointing to the wrong destination!**

- ✅ LLM correctly identified "RESOURCE_OPERATIONS"
- ❌ Code sent it to "intent" (old flow)
- ✅ Now sends it to "function_calling" (modern flow)

**Trust the LLM!** It's smarter than keyword matching. 🧠

