# ✅ Model Visibility Fix - Models Now Available to All Users!

## Problem

**Issue**: Models (Vayu Maya, Vayu Maya v2) were only visible to admin users, not regular users.

**Root Cause**: OpenWebUI has a model permission system where models need to be explicitly added to the database with proper access control settings. By default, models fetched from external APIs are not automatically visible to all users.

## Solution Applied

### 1. Disabled Model Filter
```json
{
  "ui": {
    "enable_model_filter": false
  }
}
```

### 2. Added Models to Database

Added both models to OpenWebUI's internal database with:
- ✅ **Public access** (no user restrictions)
- ✅ **Active status**
- ✅ **Proper metadata**

```sql
INSERT INTO model (id, user_id, base_model_id, name, params, meta, is_active, access_control)
VALUES (
  'Vayu Maya',
  '<admin_id>',
  'Vayu Maya',
  'Vayu Maya',
  '{}',
  '{"description": "AI Cloud Assistant for Tata Communications", ...}',
  1,
  '{"read": {"group_ids": [], "user_ids": []}, "write": {"group_ids": [], "user_ids": []}}'
)
```

**Key Point**: Empty `user_ids` arrays = **public access** for all users!

## Current Status

| Model | Status | Access | Visible To |
|-------|--------|--------|------------|
| Vayu Maya | ✅ Active | 🌐 Public | All users |
| Vayu Maya v2 | ✅ Active | 🌐 Public | All users |

## Verification

### Check Models in Database:

```bash
docker cp enterprise-rag-openwebui:/app/backend/data/webui.db /tmp/webui.db
python3 << 'EOF'
import sqlite3, json
conn = sqlite3.connect('/tmp/webui.db')
cursor = conn.cursor()
cursor.execute("SELECT id, name, is_active, access_control FROM model")
for m in cursor.fetchall():
    access = json.loads(m[3])
    public = len(access['read']['user_ids']) == 0
    print(f"{m[1]}: Active={m[2]}, Public={public}")
conn.close()
EOF
```

Expected output:
```
Vayu Maya: Active=1, Public=True
Vayu Maya v2: Active=1, Public=True
```

## Testing

### Test as Regular User:

1. **Logout** from admin account
2. **Login** as regular user (e.g., `cdemoipc@gmail.com`)
3. **Click** "Select a model" dropdown
4. **You should see**:
   - ✅ Vayu Maya
   - ✅ Vayu Maya v2

### Test as Admin:

1. **Login** as admin
2. **Go to** Admin Panel → Settings → Models
3. **Verify** both models are listed and active

## How OpenWebUI Model Visibility Works

### Model Sources:

OpenWebUI can get models from two sources:

1. **External API** (your backend at `/api/v1/models`)
   - Fetched dynamically
   - **Not automatically visible to all users**
   - Need to be added to database

2. **Internal Database** (SQLite `model` table)
   - Stored permanently
   - Access control applied
   - Visible based on permissions

### Access Control:

```json
{
  "read": {
    "group_ids": [],     // Empty = all groups
    "user_ids": []       // Empty = all users
  },
  "write": {
    "group_ids": [],
    "user_ids": []
  }
}
```

- **Empty arrays** = Public access ✅
- **Non-empty arrays** = Restricted to specific users/groups ❌

## If Models Disappear Again

### Quick Fix Script:

```bash
cd /home/unixlogin/Vayu/Enterprise-Rag-bot
./scripts/make_models_public.sh
```

### Manual Fix:

```bash
# 1. Copy database
docker cp enterprise-rag-openwebui:/app/backend/data/webui.db /tmp/webui.db

# 2. Add models (Python script above)

# 3. Restore database
docker stop enterprise-rag-openwebui
docker cp /tmp/webui.db enterprise-rag-openwebui:/app/backend/data/webui.db
docker start enterprise-rag-openwebui
```

## Admin Panel Settings

### To Manage Model Visibility (Admin Only):

1. **Login** as admin
2. **Go to** Admin Panel (gear icon)
3. **Navigate to** Settings → Models
4. **You'll see** all models with options to:
   - Enable/disable models
   - Set access permissions
   - Configure model parameters

### To Make a Model Public:

1. Click on the model
2. Go to "Access Control"
3. Ensure **no specific users** are selected
4. Save

## Architecture

```
Regular User Login
    ↓
Request: GET /api/models
    ↓
OpenWebUI checks:
    1. Is user authenticated? ✅
    2. Get models from database
    3. Filter by access_control
    4. Return visible models
    ↓
Models with empty user_ids[] = Visible ✅
Models with specific user_ids[] = Hidden ❌
```

## Why This Approach?

### Benefits:

✅ **Centralized control**: Admin can manage model visibility  
✅ **Granular permissions**: Can restrict specific models if needed  
✅ **Persistent**: Settings survive restarts  
✅ **Secure**: Prevents unauthorized access to restricted models  

### Trade-offs:

⚠️ **Manual setup**: Models need to be added to database  
⚠️ **Sync required**: If backend models change, database needs update  

## Best Practices

### For Production:

1. **Add all models** to database with public access
2. **Disable model filter** in config
3. **Document** which models are available
4. **Monitor** model usage via logs

### For Development:

1. **Test with multiple users** (admin + regular)
2. **Verify** model visibility for each role
3. **Check** access control settings

## Troubleshooting

### Issue: Models still not visible to regular users

**Check**:
1. Are models in database?
   ```bash
   docker exec enterprise-rag-openwebui sqlite3 /app/backend/data/webui.db "SELECT COUNT(*) FROM model"
   ```

2. Is access control set to public?
   ```bash
   # Check access_control field (should have empty arrays)
   ```

3. Is model filter disabled?
   ```bash
   curl -s http://localhost:3000/api/config | grep enable_model_filter
   # Should show: "enable_model_filter": false
   ```

### Issue: Models visible but can't use them

**Check**:
1. Is backend running?
   ```bash
   curl http://localhost:8001/api/v1/models
   ```

2. Can OpenWebUI reach backend?
   ```bash
   docker exec enterprise-rag-openwebui curl http://host.docker.internal:8001/api/v1/models
   ```

3. Check OpenWebUI logs:
   ```bash
   docker logs enterprise-rag-openwebui | grep -i error
   ```

## Summary

✅ **Models added** to OpenWebUI database  
✅ **Public access** configured (empty user_ids)  
✅ **Model filter** disabled  
✅ **Both models** (Vayu Maya, Vayu Maya v2) visible to all users  
✅ **Persistent** across restarts  

**Now all users (admin and regular) can see and use both models!** 🎉

## Quick Reference

| Action | Command |
|--------|---------|
| Check models | `docker exec enterprise-rag-openwebui sqlite3 /app/backend/data/webui.db "SELECT id, name FROM model"` |
| Make models public | `./scripts/make_models_public.sh` |
| Restart OpenWebUI | `docker restart enterprise-rag-openwebui` |
| Check logs | `docker logs -f enterprise-rag-openwebui` |

**Test now**: Login as regular user and select a model! ✅

