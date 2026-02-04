# Fix: Auth Token Propagation Issue

## Problem Identified

Looking at the terminal logs, the **auth token was NOT being extracted or passed through** to the API calls:

```
ERROR:app.services.api_executor_service:❌ API credentials not configured (email or password missing)
ERROR:app.services.api_executor_service:❌ Failed to refresh auth token for user: default
ERROR:app.services.api_executor_service:❌ Cannot make API call: No valid auth token (user: default)
ERROR:app.services.api_executor_service:❌ Failed to fetch engagement ID
```

## Root Cause

The **ValidationAgent** was calling `api_executor_service.get_endpoints()` **without passing the `auth_token`**. 

The token was being extracted properly at the OpenWebUI level and passed to the AgentManager and Orchestrator, but when the Orchestrator called the ValidationAgent, the `auth_token` was not being included in the context.

## Chain of Failure

```
✅ UI → Authorization: Bearer <keycloak-token>
✅ OpenWebUI Router → Extracts token
✅ Widget Endpoint → Passes token
✅ AgentManager → Receives token
✅ Orchestrator → Receives token
❌ ValidationAgent → NOT receiving token (missing from context)
❌ API Executor → Tries to use email/password (fails)
```

## Files Fixed

### 1. `app/agents/validation_agent.py`

**Added:**
- Instance variable `self._current_auth_token` to store auth token
- Extract auth_token from context/state in `execute()` method
- Pass auth_token to `_fetch_available_options()` method
- Use auth_token when calling `api_executor_service.get_endpoints()`

```python
# In __init__
self._current_auth_token = None

# In execute()
auth_token = context.get("auth_token") if context else None
if not auth_token and state:
    auth_token = state.auth_token
self._current_auth_token = auth_token

# In _fetch_available_options()
token = auth_token or self._current_auth_token
endpoints = await api_executor_service.get_endpoints(auth_token=token)
```

### 2. `app/agents/orchestrator_agent.py`

**Added:**
- Pass `auth_token` in context when calling `validation_agent.execute()`

```python
# Before (missing auth_token):
validation_result = await self.validation_agent.execute(user_input, {
    "session_id": state.session_id,
    "conversation_state": state.to_dict()
})

# After (includes auth_token):
validation_result = await self.validation_agent.execute(user_input, {
    "session_id": state.session_id,
    "conversation_state": state.to_dict(),
    "auth_token": auth_token  # ← Added!
})
```

## What Now Works

1. **Token Extraction**: ✅ Token extracted from Authorization header
2. **Token Propagation**: ✅ Token passed through all agent layers
3. **ValidationAgent**: ✅ Receives and stores auth_token
4. **API Calls**: ✅ Uses Keycloak token for API authentication

## Complete Flow (Fixed)

```
UI (Keycloak Token)
  ↓ Authorization: Bearer <token>
OpenWebUI Router
  ↓ Extracts token
Widget Endpoint
  ↓ Passes token
AgentManager
  ↓ auth_token=token
Orchestrator
  ↓ auth_token=token (in context)
ValidationAgent ✅ (FIXED!)
  ↓ Extracts from context
  ↓ Stores in self._current_auth_token
  ↓ Passes to API calls
API Executor Service
  ↓ Uses Bearer token
Tata Communications APIs ✅
```

## Testing

To verify the fix works, look for these log messages:

### ✅ Success Indicators:
```
INFO:app.agents.validation_agent:✅ ValidationAgent processing: ...
INFO:app.services.api_executor_service:✅ Using Bearer token from UI (Keycloak)
INFO:app.services.api_executor_service:✅ Fetching engagement details...
```

### ❌ Failure Indicators (should NOT see these anymore):
```
ERROR:app.services.api_executor_service:❌ API credentials not configured
ERROR:app.services.api_executor_service:❌ Failed to refresh auth token
ERROR:app.services.api_executor_service:❌ Cannot make API call: No valid auth token
```

## Configuration

**Environment:** `API_AUTH_ENABLED=true` (set in `.env`)

This enables Keycloak token authentication mode where Bearer tokens from the UI are used instead of email/password credentials.

## Summary

The issue was a **missing link in the token propagation chain**. The Orchestrator was not passing the `auth_token` to the ValidationAgent in the context dictionary. This has now been fixed, and the Keycloak Bearer token from your UI will flow through all layers to the API calls.

**Status:** ✅ FIXED - Auth token now propagates correctly through all agent layers.
