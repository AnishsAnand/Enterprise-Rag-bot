# Bug Fix: Missing _get_auth_headers Method

## Issue
**Error:** `'APIExecutorService' object has no attribute '_get_auth_headers'`

**When:** Trying to list Kafka or GitLab services

## Root Cause
The `list_managed_services()` method in `api_executor_service.py` was calling `self._get_auth_headers()` on line 493, but this method was never implemented.

```python
# Line 493 in api_executor_service.py
headers = await self._get_auth_headers()  # ❌ Method didn't exist!
```

## Solution
Added the missing `_get_auth_headers()` method to the `APIExecutorService` class:

```python
async def _get_auth_headers(self) -> Dict[str, str]:
    """
    Get authentication headers with current token.
    
    Returns:
        Dictionary of headers including authorization
    """
    headers = {
        "Content-Type": "application/json"
    }
    
    # Ensure we have a valid token
    await self._ensure_valid_token()
    
    # Add authentication with dynamically fetched token
    if self.auth_token:
        headers["Authorization"] = f"Bearer {self.auth_token}"
        logger.debug("✅ Using dynamically fetched auth token")
    else:
        logger.warning("⚠️ No auth token available for API call")
    
    return headers
```

## What This Method Does
1. **Creates base headers** with Content-Type: application/json
2. **Ensures valid token** by calling `_ensure_valid_token()` which:
   - Checks if current token is still valid
   - Fetches a new token if needed/expired
3. **Adds Bearer authentication** to the headers
4. **Returns complete headers** ready for API calls

## Files Modified
- `app/services/api_executor_service.py` - Added `_get_auth_headers()` method after line 168

## Testing
After this fix, Kafka and GitLab listing works correctly:
- ✅ "list kafka services"
- ✅ "list gitlab services"  
- ✅ Authentication headers properly included
- ✅ Tokens automatically refreshed

## Related
- This fix enables the managed services integration added for Kafka and GitLab
- See `metadata/MANAGED_SERVICES_INTEGRATION.md` for full feature documentation
