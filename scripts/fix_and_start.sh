#!/bin/bash
# fix_and_start.sh - PRODUCTION STARTUP SCRIPT
# Run this script to fix all issues and start the application

set -e
set -o pipefail

echo "🚀 Enterprise RAG Bot - Production Startup & Fix Script"
echo "========================================================"
echo ""

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
COMPOSE_FILE="docker-compose.prod.yml"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# -------------------------------------------------------------------
# ROOT CHECK
# -------------------------------------------------------------------
if [ "$EUID" -eq 0 ]; then
    echo -e "${YELLOW}⚠️  Warning: Running as root. Non-root user is recommended.${NC}"
fi

# -------------------------------------------------------------------
# 1. ENVIRONMENT VARIABLES
# -------------------------------------------------------------------
echo "1️⃣  Checking environment variables..."

if [ ! -f .env ]; then
    echo -e "${RED}❌ .env file not found!${NC}"
    if [ -f .env.example ]; then
        cp .env.example .env
        echo -e "${GREEN}✅ Created .env from .env.example${NC}"
        echo -e "${YELLOW}⚠️  Please edit .env and add required API keys${NC}"
    else
        echo -e "${RED}❌ .env.example not found. Create .env manually.${NC}"
        exit 1
    fi
fi

if ! grep -q "^GROK_API_KEY=" .env || grep -Eq "^GROK_API_KEY=($|\"\")" .env; then
    echo -e "${RED}❌ GROK_API_KEY is not set in .env${NC}"
    echo "Add: GROK_API_KEY=your_key_here"
    exit 1
fi

echo -e "${GREEN}✅ Environment variables OK${NC}"
echo ""

# -------------------------------------------------------------------
# 2. DIRECTORIES & PERMISSIONS
# -------------------------------------------------------------------
echo "2️⃣  Creating directories..."

mkdir -p backend/outputs nginx/ssl postgres_data

chmod -R 755 backend/outputs nginx/ssl 2>/dev/null || true

echo -e "${GREEN}✅ Directories ready${NC}"
echo ""

# -------------------------------------------------------------------
# 3. SSL CHECK
# -------------------------------------------------------------------
echo "3️⃣  Checking SSL certificates..."

if [ ! -f nginx/ssl/cert.pem ] || [ ! -f nginx/ssl/key.pem ]; then
    echo -e "${YELLOW}⚠️  SSL certificates not found. Running HTTP only.${NC}"
    echo "To enable HTTPS later: ./scripts/generate_ssl_certs.sh"
else
    echo -e "${GREEN}✅ SSL certificates found${NC}"
fi

echo ""

# -------------------------------------------------------------------
# 4. DOCKER CHECK
# -------------------------------------------------------------------
echo "4️⃣  Checking Docker..."

if ! command -v docker &>/dev/null; then
    echo -e "${RED}❌ Docker not installed${NC}"
    exit 1
fi

if ! command -v docker-compose &>/dev/null && ! docker compose version &>/dev/null; then
    echo -e "${RED}❌ Docker Compose not installed${NC}"
    exit 1
fi

if [ ! -f "$COMPOSE_FILE" ]; then
    echo -e "${RED}❌ $COMPOSE_FILE not found in project root${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Docker OK${NC}"
echo ""

# -------------------------------------------------------------------
# 5. STOP EXISTING CONTAINERS
# -------------------------------------------------------------------
echo "5️⃣  Stopping existing containers..."

docker-compose -f $COMPOSE_FILE down 2>/dev/null || \
docker compose -f $COMPOSE_FILE down 2>/dev/null || true

echo -e "${GREEN}✅ Containers stopped${NC}"
echo ""

# -------------------------------------------------------------------
# 6. DATABASE INIT CHECK
# -------------------------------------------------------------------
echo "6️⃣  Preparing database initialization..."

if [ -f init_postgres.sql ]; then
    echo -e "${GREEN}✅ Database init script found${NC}"
else
    echo -e "${YELLOW}⚠️  init_postgres.sql not found${NC}"
fi

echo ""

# -------------------------------------------------------------------
# 7. BUILD & START SERVICES
# -------------------------------------------------------------------
echo "7️⃣  Building and starting services..."
echo "This may take a few minutes..."

export USER_ID=$(id -u)
export GROUP_ID=$(id -g)

if command -v docker-compose &>/dev/null; then
    docker-compose -f $COMPOSE_FILE up -d --build
else
    docker compose -f $COMPOSE_FILE up -d --build
fi

echo -e "${GREEN}✅ Services started${NC}"
echo ""

# -------------------------------------------------------------------
# 8. HEALTH CHECKS
# -------------------------------------------------------------------
echo "8️⃣  Waiting for services to be healthy..."
echo "This may take up to 60 seconds..."

# PostgreSQL
echo "Waiting for PostgreSQL..."
for i in {1..30}; do
    if docker-compose -f $COMPOSE_FILE exec -T postgres pg_isready -U ragbot &>/dev/null || \
       docker compose -f $COMPOSE_FILE exec -T postgres pg_isready -U ragbot &>/dev/null; then
        echo -e "${GREEN}✅ PostgreSQL is ready${NC}"
        break
    fi
    [ "$i" -eq 30 ] && { echo -e "${RED}❌ PostgreSQL failed to start${NC}"; exit 1; }
    sleep 2
done

# Bot
echo "Waiting for Bot service..."
for i in {1..30}; do
    if curl -fs http://localhost:8000/health &>/dev/null; then
        echo -e "${GREEN}✅ Bot service ready${NC}"
        break
    fi
    sleep 2
done

# -------------------------------------------------------------------
# DONE
# -------------------------------------------------------------------
echo ""
echo "========================================================"
echo -e "${GREEN}🎉 Startup Complete!${NC}"
echo "========================================================"
echo ""
echo "📊 Service URLs:"
echo "   - Backend API:   http://localhost:8000"
echo "   - User Frontend: http://localhost:4201"
echo "   - OpenWebUI:     http://localhost:3000"
echo "   - Nginx (HTTP):  http://localhost:80"
echo "   - PostgreSQL:    localhost:5432"
echo ""
echo "✨ Application is now running."
