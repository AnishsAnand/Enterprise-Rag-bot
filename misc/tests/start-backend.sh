#!/bin/bash
# Start backend from Enterprise-Rag-bot directory
# This script starts the backend without removing orphan containers from other projects

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🚀 Starting enterprise-rag-backend from $(pwd)"
echo "📋 Using project name: enterprise-rag-bot-main"

# Check if backend container already exists
if docker ps -a --format "{{.Names}}" | grep -q "^enterprise-rag-backend$"; then
    if docker ps --format "{{.Names}}" | grep -q "^enterprise-rag-backend$"; then
        echo "✅ Backend container already running"
    else
        echo "ℹ️  Backend container exists but stopped, starting it..."
        docker start enterprise-rag-backend
    fi
else
    # Check if infrastructure containers exist - if they do, start backend without recreating them
    if docker ps --format "{{.Names}}" | grep -q "^enterprise-rag-postgres$"; then
        echo "ℹ️  Infrastructure containers already exist, creating backend only..."
        docker-compose -f docker-compose.prod.yml -p enterprise-rag-bot-main up -d --no-deps backend
    else
        echo "ℹ️  Starting backend with infrastructure..."
        docker-compose -f docker-compose.prod.yml -p enterprise-rag-bot-main up -d backend
    fi
fi

echo "✅ Backend started successfully!"
echo "📊 Container status:"
docker ps --filter "name=enterprise-rag-backend" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
