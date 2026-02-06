#!/bin/bash
# Restart backend container with latest code changes

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🔄 Restarting enterprise-rag-backend..."
echo ""

# Check if container exists
if docker ps -a --format "{{.Names}}" | grep -q "^enterprise-rag-backend$"; then
    echo "📦 Container exists, restarting..."
    docker-compose -f docker-compose.prod.yml -p enterprise-rag-bot-main restart backend
    
    echo ""
    echo "⏳ Waiting for backend to start..."
    sleep 5
    
    echo ""
    echo "📊 Container status:"
    docker ps --filter "name=enterprise-rag-backend" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    
    echo ""
    echo "🔍 Checking health..."
    if curl -sf http://localhost:8000/health/liveness > /dev/null 2>&1; then
        echo "✅ Backend is healthy and responding"
    else
        echo "⚠️  Backend is starting up (health check may take a moment)"
        echo "💡 Check logs with: docker logs -f enterprise-rag-backend"
    fi
else
    echo "❌ Backend container not found!"
    echo "💡 Start it with: ./start-backend.sh"
    exit 1
fi

echo ""
echo "📋 Recent logs:"
docker logs enterprise-rag-backend --tail 10 2>&1 | tail -5
