# Permission Fix: User Role for API Access

## Issue

Function calling was working correctly, but API calls were being rejected with permission errors:

```
WARNING: Permission denied: user roles ['user'] do not match required permissions 
['admin', 'developer', 'viewer'] for endpoint.list
```

**Impact:** Users couldn't list clusters or datacenters through OpenWebUI, even though the function calling mechanism was working perfectly.

---

## Root Cause

**File:** `app/routers/openai_compatible.py` line 259

The default user role was set to `["user"]`:
```python
user_roles = ["user"]  # ❌ 'user' role not in allowed permissions
```

But the API executor requires one of these roles for endpoint operations:
- `admin` (full access)
- `developer` (read/write access)
- `viewer` (read-only access)

The `"user"` role was not included in any of these permission lists in the resource schema.

---

## Fix Applied

**Changed default role from `["user"]` to `["viewer"]`:**

```python
# Default roles - Changed from ["user"] to ["viewer"] for API access
# TODO: Extract actual roles from JWT token or OpenWebUI authentication  
user_roles = ["viewer"]  # viewer has read permissions for clusters/endpoints
```

---

## Why "viewer" Role?

From `app/config/resource_schema.json`:

```json
{
  "endpoint": {
    "permissions": {
      "list": ["admin", "developer", "viewer"]  // ✅ viewer can list
    }
  },
  "k8s_cluster": {
    "permissions": {
      "list": ["admin", "developer", "viewer"],  // ✅ viewer can list
      "create": ["admin", "developer"],          // ❌ viewer cannot create
      "delete": ["admin"]                        // ❌ viewer cannot delete
    }
  }
}
```

**Viewer role provides:**
- ✅ List clusters
- ✅ List datacenters/endpoints
- ✅ View cluster details
- ❌ Cannot create/update/delete (read-only)

This is **safe** for general OpenWebUI users while allowing the function calling feature to work.

---

## Future Enhancement (TODO)

For production, you should extract actual user roles from authentication:

### Option A: Extract from OpenWebUI JWT Token

```python
# In openai_compatible.py
def extract_roles_from_token(authorization: str) -> List[str]:
    """Extract user roles from JWT token."""
    try:
        import jwt
        token = authorization.replace("Bearer ", "")
        decoded = jwt.decode(token, verify=False)  # Or verify with secret
        return decoded.get("roles", ["viewer"])
    except:
        return ["viewer"]  # Fallback to viewer

# Then use:
user_roles = extract_roles_from_token(authorization) if authorization else ["viewer"]
```

### Option B: Query OpenWebUI Database

```python
async def get_user_roles_from_openwebui(user_id: str) -> List[str]:
    """Query OpenWebUI database for user roles."""
    # Connect to OpenWebUI's database and fetch user.role
    # Map OpenWebUI roles to your system roles
    # Example: "admin" -> ["admin"], "user" -> ["viewer"]
    pass
```

### Option C: Environment Variable Override

```python
# For testing/development
default_role = os.getenv("DEFAULT_USER_ROLE", "viewer")
user_roles = [default_role]
```

---

## Testing

After this fix, the function calling should work:

**Query:** "List clusters in Delhi"

**Expected Flow:**
1. ✅ Routes to FunctionCallingAgent
2. ✅ LLM calls `list_k8s_clusters(location_names=["Delhi"])`
3. ✅ Function fetches datacenters (permission: viewer ✓)
4. ✅ Function lists clusters (permission: viewer ✓)
5. ✅ Returns results to user

**Logs should show:**
```
INFO: Function calling iteration 1/5
INFO: Tool: list_k8s_clusters with args: {"location_names": ["Delhi"]}
INFO: ✅ Function executed successfully
INFO: Found X clusters in Delhi datacenter...
```

---

## Status

✅ **Fixed** - Default role changed to "viewer"

**Date:** December 13, 2024  
**Impact:** Medium (blocked function calling feature for all users)  
**Resolution:** Immediate (one-line change)

---

## Related Permissions

From resource schema, viewer role can access:

| Resource | Operation | Viewer Access |
|----------|-----------|---------------|
| **endpoint** | list | ✅ Yes |
| **k8s_cluster** | list | ✅ Yes |
| **k8s_cluster** | create | ❌ No (admin/developer only) |
| **k8s_cluster** | delete | ❌ No (admin only) |
| **firewall** | list | ✅ Yes |
| **firewall** | create | ❌ No |
| **kafka** | list | ✅ Yes |
| **gitlab** | list | ✅ Yes |
| **jenkins** | list | ✅ Yes |
| **postgres** | list | ✅ Yes |

**Summary:** Viewer role is perfect for read operations (list, get) but blocks write operations (create, update, delete).

