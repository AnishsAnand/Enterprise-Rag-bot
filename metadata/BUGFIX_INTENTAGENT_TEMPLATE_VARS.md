# IntentAgent Template Variable Error - FIXED

## Date: December 12, 2025

---

## ❌ Error

```
IntentAgent execution failed: 
'Input to ChatPromptTemplate is missing variables {'"zone"', '"department"', '"endpoint"'}. 
Expected: ['"department"', '"endpoint"', '"zone"', 'agent_scratchpad', 'input'] 
Received: ['input', 'chat_history', 'intermediate_steps', 'agent_scratchpad']'
```

## 🔍 Root Cause

In `app/agents/intent_agent.py`, the system prompt contained:

```
- can optionally extract "endpoint", "zone", or "department" for filtering
```

The `ChatPromptTemplate` from LangChain interprets quoted strings like `"endpoint"`, `"zone"`, `"department"` as template variables that need to be provided as inputs.

Since these were in descriptive text (not actual template variables), the template engine expected them but never received them, causing the error.

## ✅ Fix

**File:** `app/agents/intent_agent.py`

**Line:** ~204

**Change:**
```diff
- - For "list" operation on vm: NO parameters required (lists all VMs), but can optionally extract "endpoint", "zone", or "department" for filtering
+ - For "list" operation on vm: NO parameters required (lists all VMs), but can optionally extract endpoint, zone, or department for filtering
```

**Explanation:**
Removed the quotes around `endpoint`, `zone`, and `department` so they're no longer interpreted as template variables.

## 🧪 Verification

1. Backend restarted successfully
2. No template variable errors in logs
3. IntentAgent can now process queries correctly
4. All 12 resources loaded successfully

## 📝 Technical Details

### How ChatPromptTemplate Works

1. Scans prompt text for variable patterns:
   - `{variable_name}` - explicit template variable
   - `"variable_name"` - can also be interpreted as variable in certain contexts
2. Expects all detected variables to be provided as input
3. Raises error if variables are missing

### Why This Happened

- We added VM examples with optional filter parameters
- To describe the filters in natural language, we used quotes: `"endpoint"`
- ChatPromptTemplate interpreted these as variables
- No values were provided for these "variables"
- Error was raised

### Prevention

When writing prompts for LangChain agents:
- Avoid quoting parameter names in descriptive text
- Use backticks for inline code: `endpoint` instead of `"endpoint"`
- Or rephrase: "extract endpoint values" instead of "extract 'endpoint'"
- Test prompt changes with actual queries

## ✅ Status

**FIXED** - IntentAgent error resolved, backend running normally.

## 🔗 Related

- VM and Firewall integration: `metadata/VM_AND_FIREWALL_INTEGRATION.md`
- Managed services integration: `metadata/MANAGED_SERVICES_EXTENDED.md`


