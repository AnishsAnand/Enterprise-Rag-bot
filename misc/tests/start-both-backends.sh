#!/bin/bash
# Start both backend containers without conflicts
# This script ensures both backends run simultaneously without removing each other
# Can be run from either Enterprise-Rag-bot or kuber/Enterprise-Rag-bot directory

set -e

MAIN_DIR="/home/unixlogin/Vayu/Enterprise-Rag-bot"
KUBER_DIR="/home/unixlogin/kuber/Enterprise-Rag-bot"

# Verify both directories exist
if [ ! -f "$MAIN_DIR/docker-compose.prod.yml" ]; then
    echo "❌ Error: docker-compose.prod.yml not found at $MAIN_DIR"
    exit 1
fi

if [ ! -f "$KUBER_DIR/docker-compose.prod.yml" ]; then
    echo "❌ Error: docker-compose.prod.yml not found at $KUBER_DIR"
    exit 1
fi

echo "🚀 Starting both backend containers..."
echo ""

# Start first backend
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1️⃣ Starting enterprise-rag-backend (port 8000)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cd "$MAIN_DIR"

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
        # Start backend without depends_on check since infrastructure is already running
        docker-compose -f docker-compose.prod.yml -p enterprise-rag-bot-main up -d --no-deps backend
    else
        echo "ℹ️  Starting backend with infrastructure..."
        docker-compose -f docker-compose.prod.yml -p enterprise-rag-bot-main up -d backend
    fi
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2️⃣ Starting enterprise-rag-backend-kv (port 8004)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cd "$KUBER_DIR"

# Check if KV backend container already exists
if docker ps -a --format "{{.Names}}" | grep -q "^enterprise-rag-backend-kv$"; then
    if docker ps --format "{{.Names}}" | grep -q "^enterprise-rag-backend-kv$"; then
        echo "✅ KV Backend container already running"
    else
        echo "ℹ️  KV Backend container exists but stopped, starting it..."
        docker start enterprise-rag-backend-kv
    fi
else
    # Check if KV infrastructure containers exist - if they do, start backend without recreating them
    if docker ps --format "{{.Names}}" | grep -q "^enterprise-rag-postgres-kv$"; then
        echo "ℹ️  KV infrastructure containers already exist, creating backend only..."
        # Start backend without depends_on check since infrastructure is already running
        docker-compose -f docker-compose.prod.yml -p enterprise-rag-bot-kv up -d --no-deps backend
    else
        echo "ℹ️  Starting KV backend with infrastructure..."
        docker-compose -f docker-compose.prod.yml -p enterprise-rag-bot-kv up -d backend
    fi
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Both backends started successfully!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📊 Container status:"
docker ps --filter "name=enterprise-rag-backend" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
