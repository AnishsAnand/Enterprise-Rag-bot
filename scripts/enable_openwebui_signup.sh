#!/bin/bash
# Script to permanently enable signup in OpenWebUI

echo "🔧 Enabling signup in OpenWebUI..."

# Copy database from container
docker cp enterprise-rag-openwebui:/app/backend/data/webui.db /tmp/webui.db

# Update database
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
    
    # Enable signup
    if 'ui' not in config:
        config['ui'] = {}
    config['ui']['enable_signup'] = True
    
    # Update config
    new_config_json = json.dumps(config)
    cursor.execute("UPDATE config SET data = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?", 
                   (new_config_json, config_id))
    conn.commit()
    print("✅ Signup enabled in database")
else:
    print("⚠️  No config found in database")

conn.close()
EOF

# Copy database back and restart
docker stop enterprise-rag-openwebui
docker cp /tmp/webui.db enterprise-rag-openwebui:/app/backend/data/webui.db
docker start enterprise-rag-openwebui

echo "✅ OpenWebUI restarted with signup enabled"
echo ""
echo "🌐 Access OpenWebUI at: http://localhost:3000/"
echo "📝 Signup is now permanently enabled!"

