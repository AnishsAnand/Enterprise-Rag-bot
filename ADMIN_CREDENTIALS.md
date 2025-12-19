# 🔑 OpenWebUI Admin Credentials

## Admin Login

**URL**: http://localhost:3000/

**Email**: `admin@localhost`

**Password**: `Admin@123`

## What To Do After Login

### Step 1: Access Admin Panel

1. Click the **gear icon** (⚙️) in the top right
2. This opens the Admin Panel

### Step 2: Configure OpenAI API Connection

1. In Admin Panel, go to **Settings**
2. Navigate to **Connections** or **External APIs** or **Models**
3. Look for **OpenAI API** section
4. Click **Add Connection** or **Configure**

### Step 3: Add Your Backend

**Configuration**:
- **API Base URL**: `http://host.docker.internal:8001/api/v1`
- **API Key**: `sk-dummy-key` (or any value, your backend doesn't validate it)
- **Name**: `Vayu Maya Backend` (optional)
- **Enable**: ✅ Yes

### Step 4: Save and Test

1. Click **Save**
2. Click **Test Connection** (if available)
3. You should see: ✅ Connected
4. Models should appear: `Vayu Maya`, `Vayu Maya v2`

### Step 5: Make Models Available to All Users

1. In Admin Panel → Settings → **Models**
2. Find your models: `Vayu Maya`, `Vayu Maya v2`
3. For each model:
   - Click on it
   - Go to **Access Control** or **Permissions**
   - Set to **Public** or **All Users**
   - Save

### Step 6: Enable Signup (if needed)

1. Admin Panel → Settings → **General**
2. Find **Enable Signup**
3. Toggle to **ON**
4. Save

## Troubleshooting

### If you can't find "Connections" in Admin Panel:

Try these sections:
- **Settings → Admin Settings → Connections**
- **Settings → External APIs**
- **Settings → Models → Add Model Source**
- **Workspace → Connections**

### If models still don't appear:

1. Check OpenWebUI logs:
   ```bash
   docker logs enterprise-rag-openwebui | grep -i "error\|openai"
   ```

2. Check if backend is being called:
   ```bash
   tail -f /tmp/user_main.log | grep "GET /api/v1/models"
   ```

3. Test connection manually:
   ```bash
   docker exec enterprise-rag-openwebui curl http://host.docker.internal:8001/api/v1/models
   ```

### If connection test fails:

**Check backend is running**:
```bash
ps aux | grep "uvicorn.*8001"
curl http://localhost:8001/api/v1/models
```

**Check Docker networking**:
```bash
docker exec enterprise-rag-openwebui ping -c 2 host.docker.internal
```

## Alternative: Configure via Database

If Admin Panel doesn't have the option, we can add it directly to the database:

```bash
# Stop OpenWebUI
docker stop enterprise-rag-openwebui

# Copy database
docker cp enterprise-rag-openwebui:/app/backend/data/webui.db /tmp/webui.db

# Add configuration (Python script)
python3 << 'EOF'
import sqlite3, json
conn = sqlite3.connect('/tmp/webui.db')
cursor = conn.cursor()

# Get config
cursor.execute("SELECT id, data FROM config")
config_id, config_json = cursor.fetchone()
config = json.loads(config_json)

# Add OpenAI connection
config['openai'] = {
    'api_base_urls': ['http://host.docker.internal:8001/api/v1'],
    'api_keys': ['sk-dummy-key'],
    'enable': True
}

# Save
cursor.execute("UPDATE config SET data=? WHERE id=?", (json.dumps(config), config_id))
conn.commit()
conn.close()
print("✅ Configuration added")
EOF

# Restore database
docker cp /tmp/webui.db enterprise-rag-openwebui:/app/backend/data/webui.db

# Start OpenWebUI
docker start enterprise-rag-openwebui
```

## Summary

| Item | Value |
|------|-------|
| **URL** | http://localhost:3000/ |
| **Email** | admin@localhost |
| **Password** | Admin@123 |
| **Backend URL** | http://host.docker.internal:8001/api/v1 |
| **API Key** | sk-dummy-key |

## Next Steps

1. ✅ Login with admin credentials
2. ✅ Configure backend connection in Admin Panel
3. ✅ Make models public for all users
4. ✅ Test with regular user account

**Good luck! Let me know if you need help navigating the Admin Panel!**

