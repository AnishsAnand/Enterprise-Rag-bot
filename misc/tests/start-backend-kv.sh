#!/bin/bash
# Start backend-kv from kuber/Enterprise-Rag-bot directory
# This script starts the backend-kv without removing orphan containers from other projects

set -e

KUBER_DIR="/home/unixlogin/kuber/Enterprise-Rag-bot"

if [ ! -f "$KUBER_DIR/docker-compose.prod.yml" ]; then
    echo "❌ Error: docker-compose.prod.yml not found at $KUBER_DIR"
    exit 1
fi

cd "$KUBER_DIR"

echo "🚀 Starting enterprise-rag-backend-kv from $(pwd)"
echo "📋 Using project name: enterprise-rag-bot-kv"

# Use explicit project name to avoid conflicts with other docker-compose projects
docker-compose -f docker-compose.prod.yml -p enterprise-rag-bot-kv up -d backend

echo "✅ Backend-kv started successfully!"
echo "📊 Container status:"
docker ps --filter "name=enterprise-rag-backend-kv" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
