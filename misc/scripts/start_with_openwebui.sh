#!/bin/bash

#############################################################################
# Start Enterprise RAG Bot with Open WebUI
# This script starts all services including Open WebUI integration
#############################################################################

set -e

echo "🚀 Starting Enterprise RAG Bot with Open WebUI Integration"
echo "======================================================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${RED}❌ .env file not found!${NC}"
    echo "Please create .env from env.openwebui.template"
    exit 1
fi

echo -e "${GREEN}✅ .env file found${NC}"

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}❌ docker-compose not found!${NC}"
    echo "Please install docker-compose first"
    exit 1
fi

echo -e "${GREEN}✅ docker-compose found${NC}"

# Load environment variables
source .env

echo ""
echo "📦 Starting services with docker-compose..."
echo "======================================================================="

# Start services
docker-compose -f docker-compose.openwebui.yml up -d

echo ""
echo -e "${YELLOW}⏳ Waiting for services to be ready...${NC}"
sleep 10

echo ""
echo "🏥 Health Checks"
echo "======================================================================="

# Check Open WebUI
if curl -s http://localhost:3000 > /dev/null; then
    echo -e "${GREEN}✅ Open WebUI is running (http://localhost:3000)${NC}"
else
    echo -e "${RED}❌ Open WebUI is not responding${NC}"
fi

# Check Backend
if curl -s http://localhost:8000/health > /dev/null; then
    echo -e "${GREEN}✅ Backend API is running (http://localhost:8000)${NC}"
else
    echo -e "${RED}❌ Backend API is not responding${NC}"
fi

# Check OpenAI endpoints
if curl -s http://localhost:8000/api/v1/models > /dev/null; then
    echo -e "${GREEN}✅ OpenAI-compatible endpoints working${NC}"
else
    echo -e "${YELLOW}⚠️  OpenAI endpoints may not be ready yet${NC}"
fi

echo ""
echo "📊 Service Status"
echo "======================================================================="
docker-compose -f docker-compose.openwebui.yml ps

echo ""
echo "🎉 Deployment Complete!"
echo "======================================================================="
echo ""
echo "Access your services:"
echo "  🌐 Open WebUI:        http://localhost:3000"
echo "  🔧 Backend API:       http://localhost:8000"
echo "  📚 API Docs:          http://localhost:8000/docs"
echo "  💾 MinIO Console:     http://localhost:9001"
echo ""
echo "Next steps:"
echo "  1. Open http://localhost:3000 in your browser"
echo "  2. Create a user account"
echo "  3. Select 'enterprise-rag-bot' model"
echo "  4. Start chatting!"
echo ""
echo "View logs:"
echo "  docker-compose -f docker-compose.openwebui.yml logs -f"
echo ""
echo "Stop services:"
echo "  docker-compose -f docker-compose.openwebui.yml down"
echo ""
echo "======================================================================="

