#!/bin/bash
# Script to make all models available to all users in OpenWebUI

echo "🔧 Making models available to all users..."

# Copy database from container
docker cp enterprise-rag-openwebui:/app/backend/data/webui.db /tmp/webui.db

# Update database to make models public
python3 << 'EOF'
import sqlite3
import json

db_path = '/tmp/webui.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get current config
cursor.execute("SELECT id, data FROM config")
result = cursor.fetchone()

if result:
    config_id, config_json = result
    config = json.loads(config_json)
    
    print("Current config keys:", list(config.keys()))
    
    # Set model visibility settings
    if 'ui' not in config:
        config['ui'] = {}
    
    # Enable model visibility for all users
    config['ui']['enable_model_filter'] = False  # Disable model filtering
    
    # Update config
    new_config_json = json.dumps(config)
    cursor.execute("UPDATE config SET data = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?", 
                   (new_config_json, config_id))
    conn.commit()
    print("✅ Model filter disabled - all models visible to all users")
else:
    print("⚠️  No config found")

# Check if there are any model-specific permissions
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%model%'")
model_tables = cursor.fetchall()
print(f"\nModel-related tables: {[t[0] for t in model_tables]}")

conn.close()
EOF

# Copy database back and restart
docker stop enterprise-rag-openwebui
docker cp /tmp/webui.db enterprise-rag-openwebui:/app/backend/data/webui.db
docker start enterprise-rag-openwebui

echo "✅ OpenWebUI restarted"
echo ""
echo "🌐 Models should now be visible to all users!"
echo "📝 Login as a regular user and check 'Select a model'"

