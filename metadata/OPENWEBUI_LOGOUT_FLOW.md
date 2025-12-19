# 🚪 OpenWebUI Logout Flow - What Happens When You Click Sign Out

## Current Issue

When you click "Sign Out" in OpenWebUI, you're seeing a broken page with "OI" text at `localhost:57695/auth`. This happens because:

1. **OpenWebUI** is trying to redirect to an `/auth` endpoint
2. **Your backend** doesn't have a route handler for `/auth`
3. The frontend is showing a fallback/error page

## What Should Happen

### Normal Logout Flow:

1. **User clicks "Sign Out"** in OpenWebUI
2. **OpenWebUI clears its session** (localStorage, cookies)
3. **Redirects to login page** (usually `/login` or `/auth/login`)
4. **User can log in again** with different credentials

### Current Problem:

OpenWebUI is redirecting to `/auth` which doesn't exist in your backend, causing the broken page.

## 🔧 Solutions

### Option 1: Add Auth Route Handler (Recommended)

Add a route handler that redirects to the login page:

```python
# In app/main.py, add this route:

@app.get("/auth")
@app.get("/auth/login")
async def auth_redirect():
    """Redirect /auth to login page for OpenWebUI compatibility."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/login", status_code=302)
```

### Option 2: Configure OpenWebUI Redirect

Configure OpenWebUI to redirect to the correct login URL:

1. **Check OpenWebUI settings** for logout redirect URL
2. **Set it to** `/login` or your actual login page URL
3. **Or disable** the redirect and handle logout client-side

### Option 3: Use OpenWebUI's Built-in Logout

OpenWebUI handles logout internally. The issue might be:
- A misconfigured redirect URL
- A frontend routing problem
- Missing route handler

## 📋 Step-by-Step: What Happens Now

### When You Click "Sign Out" in OpenWebUI:

1. ✅ **OpenWebUI clears its session**
   - Removes tokens from localStorage
   - Clears cookies
   - Resets user state

2. ❌ **Redirect fails**
   - Tries to go to `/auth` 
   - Backend doesn't have this route
   - Shows broken page

3. ✅ **You're actually logged out**
   - Even though the page looks broken
   - Your session is cleared
   - Just need to navigate to login manually

## 🛠️ Quick Fix

### Immediate Solution:

1. **After clicking Sign Out**, manually navigate to:
   ```
   http://localhost:3000/login
   ```
   or
   ```
   http://localhost:8000/login
   ```

2. **Or refresh the page** and go to the login URL

### Permanent Fix:

Add the route handler mentioned in Option 1 above.

## 🔍 Debugging Steps

1. **Check browser console** for errors:
   ```javascript
   // Open DevTools (F12) → Console tab
   // Look for redirect errors
   ```

2. **Check network tab**:
   - See what requests are made on logout
   - Check redirect responses

3. **Check OpenWebUI logs**:
   ```bash
   docker logs enterprise-rag-openwebui -f
   ```

4. **Check backend logs**:
   ```bash
   # Look for 404 errors on /auth
   tail -f /path/to/your/logs
   ```

## ✅ Verification

After implementing the fix:

1. **Click Sign Out** in OpenWebUI
2. **Should redirect** to login page (not broken page)
3. **Can log in** as different user
4. **Each user's credentials** work independently

## 🎯 Expected Behavior

```
User clicks "Sign Out"
    ↓
OpenWebUI clears session
    ↓
Redirects to /login (or /auth/login)
    ↓
Login page loads correctly
    ↓
User can log in as different user
    ↓
New user's API credentials are used
```

## 📝 Additional Notes

- **JWT tokens are stateless**: Logout is primarily client-side
- **OpenWebUI manages its own sessions**: Your backend just validates tokens
- **Each user needs their own credentials**: Set via `/api/user/credentials` endpoint
- **Admin access issue**: Should be resolved with the auth system fixes


