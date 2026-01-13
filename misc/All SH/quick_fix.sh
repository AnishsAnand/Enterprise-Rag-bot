#!/bin/bash
# Quick emergency fix script

echo "Applying emergency fixes..."

# 1. Fix database immediately
docker-compose exec -T postgres psql -U ragbot -d ragbot_db << EOF
ALTER TABLE users ADD COLUMN IF NOT EXISTS full_name VARCHAR(255);
ALTER TABLE users ADD COLUMN IF NOT EXISTS avatar_url VARCHAR(500);
ALTER TABLE users ADD COLUMN IF NOT EXISTS bio TEXT;
ALTER TABLE users ADD COLUMN IF NOT EXISTS theme VARCHAR(50) DEFAULT 'light';
ALTER TABLE users ADD COLUMN IF NOT EXISTS language VARCHAR(10) DEFAULT 'en';
EOF

echo "✅ Database schema fixed"

# 2. Restart services
docker-compose restart bot user

echo "✅ Services restarted"

# 3. Check health
sleep 5
curl -f http://localhost:8000/health && echo "✅ Bot OK" || echo "❌ Bot failed"
curl -f http://localhost:8001/health && echo "✅ User OK" || echo "❌ User failed"

echo "Done! Check logs if errors persist:"
echo "docker-compose logs -f bot user"