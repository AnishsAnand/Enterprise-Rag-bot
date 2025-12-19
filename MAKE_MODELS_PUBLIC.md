# 🌐 Make Models Public for All Users

## Current Situation

✅ Connection configured correctly  
✅ Models showing in Admin Panel (Vayu Maya, Vayu Maya v2)  
❌ Models not visible to regular users  

## The Issue

Models need to be set as **PUBLIC** so all users can see them.

## Solution: Set Model Visibility

### In the Admin Panel:

1. **Go to**: Settings → **Models** (you're already there based on screenshot)
2. **Click on each model** (Vayu Maya, Vayu Maya v2)
3. Look for one of these options:
   - **"Visibility"** setting
   - **"Access Control"** button
   - **"Permissions"** section
   - **"Available to"** dropdown
4. Set to:
   - **"Public"** or
   - **"All Users"** or
   - **"Everyone"**
5. **Save** changes

### Alternative: Check Model Settings

In your second screenshot, I see a **Settings** panel with:
- Reorder Models
- Default Models
- Default Pinned Models

**Try this**:
1. Click on **"Vayu Maya"** in the models list
2. Look for a **gear icon** or **settings button** next to the model name
3. Click it to open model-specific settings
4. Look for **"Visibility"** or **"Access"** option
5. Set to **"Public"**

### Or Check General Settings:

1. Go to Settings → **General**
2. Look for:
   - **"Model Filter"** - should be OFF/disabled
   - **"Enable Model Filter"** - should be unchecked
   - **"Default User Role"** - should be "user"
   - **"Allow users to access all models"** - should be ON

## Quick Database Fix (If UI doesn't work)

If you can't find the visibility settings in the UI, I can fix it directly in the database:

```bash
# This will make all models public
python3 << 'EOF'
import sqlite3, json

conn = sqlite3.connect('/tmp/webui.db')
cursor = conn.cursor()

# Get all models
cursor.execute("SELECT id, name, access_control FROM model")
models = cursor.fetchall()

for model_id, name, access in models:
    # Set public access (empty user_ids = everyone)
    public_access = {
        "read": {"group_ids": [], "user_ids": []},
        "write": {"group_ids": [], "user_ids": []}
    }
    
    cursor.execute(
        "UPDATE model SET access_control=? WHERE id=?",
        (json.dumps(public_access), model_id)
    )
    print(f"✅ Made {name} public")

conn.commit()
conn.close()
EOF

# Copy back and restart
docker stop enterprise-rag-openwebui
docker cp /tmp/webui.db enterprise-rag-openwebui:/app/backend/data/webui.db
docker start enterprise-rag-openwebui
```

## Test After Changes

1. **Logout** from admin account
2. **Login** as regular user: `cdemoipc@gmail.com`
3. Click **"Select a model"** dropdown
4. You should see: Vayu Maya, Vayu Maya v2

## What To Look For

In the Admin Panel → Models section, look for:
- ⚙️ **Settings icon** next to each model
- 👁️ **Visibility icon** or **eye icon**
- 🔒 **Lock icon** (means restricted)
- 🌐 **Globe icon** (means public)
- **Three dots menu** (⋮) next to model name

Click on these to access model-specific settings.

## Common Locations

| Setting | Location |
|---------|----------|
| Model Visibility | Settings → Models → Click Model → Visibility |
| Access Control | Settings → Models → Click Model → Access Control |
| Model Permissions | Settings → Models → Right-click Model → Permissions |
| Model Settings | Settings → Models → Gear Icon → Settings |

## If Nothing Works

Let me know and I'll:
1. Fix it directly in the database
2. Or we can check OpenWebUI logs to see why models aren't showing for regular users

**Try clicking on the model names in your Models list and see if a settings panel opens!**

