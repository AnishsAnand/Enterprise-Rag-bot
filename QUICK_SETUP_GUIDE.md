# Quick Setup Guide - Enable Signup

## Current Status
OpenWebUI is running but signup is disabled by default.

## Solution: Create First Admin User

When OpenWebUI starts fresh (no existing users), the **first user to register becomes admin** automatically.

### Steps:

1. **Access OpenWebUI**:
   ```
   http://localhost:3000/
   ```

2. **You should see a signup form** (since there are no users yet)

3. **Create your admin account**:
   - Email: `admin@tatacommunications.com`
   - Password: (choose a strong password)
   - Name: Admin User

4. **You're now admin!** The first user is automatically admin.

5. **Enable signup for others**:
   - Click your profile icon (top right)
   - Go to "Admin Settings" or "Settings"
   - Find "Enable Signup" toggle
   - Turn it ON
   - Now others can create accounts

## Two-Tier Access (Simple Version - No SSO)

Once signup is enabled, here's how to handle two types of users:

### Option 1: Email Domain Check (Recommended - Simplest)

I'll update the backend to automatically check email domains:

```python
# Tata Communications employees get full access
if user_email.endswith("@tatacommunications.com"):
    user_roles = ["admin", "developer", "viewer"]  # Full access
else:
    user_roles = ["viewer"]  # Read-only
```

**Pros**:
- ✅ Automatic - no manual work
- ✅ Works immediately
- ✅ Tata employees just need to use their Tata email

**Cons**:
- ⚠️ Someone could potentially fake a Tata email (though unlikely)

### Option 2: Manual Approval (Most Control)

1. Anyone can signup
2. New users get "viewer" role (read-only)
3. Admin reviews users
4. Admin manually upgrades Tata employees to "developer" role

**Pros**:
- ✅ Complete control
- ✅ Verify each user

**Cons**:
- ❌ Manual work for each user

### Option 3: SSO with Tata Communications (Most Secure - Requires Setup)

This is what I explained in `SSO_EXPLANATION.md`.

**Pros**:
- ✅ Most secure
- ✅ Automatic verification
- ✅ Enterprise-grade

**Cons**:
- ⚠️ Requires IT coordination
- ⚠️ Need Client ID/Secret from Tata IT

## My Recommendation

**Start with Option 1 (Email Domain Check)** because:
1. Works immediately
2. No manual approval needed
3. Good enough for internal use
4. Can add SSO later if needed

## Implementation

Once you create the admin account and enable signup, I'll implement the email domain check in the backend. This way:

- Users with `@tatacommunications.com` → Full access (can perform actions)
- Other users → Read-only access (can chat, cannot perform actions)

## Next Steps

1. Visit `http://localhost:3000/`
2. Create your admin account (first user)
3. Enable signup in admin settings
4. Let me know, and I'll implement the email domain check

**Questions?**
- Want Option 1 (Email check)? → I'll implement it now
- Want Option 2 (Manual approval)? → I'll show you how
- Want Option 3 (SSO)? → Read `SSO_EXPLANATION.md` first

