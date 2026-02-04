# API Authentication Flow

## Overview

The system now supports two authentication modes:

1. **Development Mode** (`API_AUTH_ENABLED=false`): Authentication is completely bypassed
2. **Production Mode** (`API_AUTH_ENABLED=true`): Uses Bearer tokens from Keycloak via the UI

## Configuration

### `.env` File

```env
# Set to 'false' to disable authentication (development mode)
API_AUTH_ENABLED=false

# API authentication credentials (only needed if API_AUTH_ENABLED=true)
# When true, the system uses Bearer tokens from the UI instead of these credentials
# API_AUTH_EMAIL=your-email@example.com
# API_AUTH_PASSWORD=your-password
```

## Authentication Flow (Production Mode)

When `API_AUTH_ENABLED=true`:

```
UI (Keycloak) 
  → Authorization: Bearer <token>
    → FastAPI Route (/api/widget/query)
      → Extract token from header
        → AgentManager.process_request(auth_token=token)
          → OrchestratorAgent.orchestrate(auth_token=token)
            → ValidationAgent/ExecutionAgent (auth_token=token)
              → APIExecutorService.execute_operation(auth_token=token)
                → _make_api_call(auth_token=token)
                  → _get_auth_headers(auth_token=token)
                    → Uses provided token in Authorization header
```

## Key Changes

### 1. `APIExecutorService` (`app/services/api_executor_service.py`)

- Added `auth_enabled` flag (reads from `API_AUTH_ENABLED` env var)
- Updated `_ensure_valid_token()`: Bypasses token validation when auth is disabled
- Updated `_get_auth_headers()`: Accepts `auth_token` parameter and uses it directly when provided
- Updated `_make_api_call()`: Accepts `auth_token` parameter and skips validation if provided
- Updated `execute_operation()`: Accepts `auth_token` parameter and passes it through
- Updated helper methods (`get_engagement_id`, `get_ipc_engagement_id`, `get_endpoints`): Accept and pass `auth_token`

### 2. `AgentManager` (`app/agents/agent_manager.py`)

- Updated `process_request()`: Accepts `auth_token` parameter
- Passes token through to orchestrator

### 3. Route Handler (`app/api/routes/rag_widget.py`)

- Extracts Bearer token from Authorization header
- Passes token to `_handle_agent_routing()`
- `_handle_agent_routing()` passes token to agent manager

## Usage

### Development (No Authentication)

```bash
# In .env
API_AUTH_ENABLED=false
```

No Authorization header needed. All API calls proceed without authentication.

### Production (Keycloak Authentication)

```bash
# In .env
API_AUTH_ENABLED=true
```

The UI must provide a Keycloak Bearer token in the Authorization header:

```http
POST /api/widget/query
Authorization: Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...
Content-Type: application/json

{
  "query": "list all clusters",
  "user_id": "user@example.com"
}
```

The backend extracts the token and uses it for all downstream API calls to the Tata Communications platform.

## Benefits

1. **Development Mode**: No credentials needed for local testing
2. **Production Mode**: Uses secure Keycloak tokens from UI
3. **No credential storage**: Email/password no longer required in environment variables
4. **Token pass-through**: UI authentication propagates to all backend API calls
5. **Flexible**: Easy to switch between modes via single environment variable

## Migration Notes

If you previously used `API_AUTH_EMAIL` and `API_AUTH_PASSWORD`:

1. Set `API_AUTH_ENABLED=false` for development
2. Set `API_AUTH_ENABLED=true` for production with Keycloak
3. Remove or comment out `API_AUTH_EMAIL` and `API_AUTH_PASSWORD` from `.env`

The system will automatically use the Bearer token from the Authorization header when enabled.
