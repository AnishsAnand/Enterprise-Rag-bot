#!/bin/bash
# ==============================================================================
# QUICK DEPLOYMENT SCRIPT - Enterprise RAG Bot
# One command to deploy entire production system
# ==============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         Enterprise RAG Bot - Quick Deployment                  ║"
echo "║         Production Fixed v2.1.0                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check prerequisites
echo -e "${YELLOW}📋 Checking prerequisites...${NC}"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Docker found: $(docker --version)${NC}"

# Check Docker Compose
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED}❌ Docker Compose is not installed. Please install Docker Compose first.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Docker Compose found${NC}"

# Check .env file
if [ ! -f .env ]; then
    echo -e "${YELLOW}⚠️  .env file not found. Creating from template...${NC}"
    if [ -f .env.example ]; then
        cp .env.example .env
        echo -e "${GREEN}✓ Created .env from template${NC}"
        echo -e "${YELLOW}⚠️  IMPORTANT: Edit .env and set your API keys before continuing!${NC}"
        echo -e "${YELLOW}   Required variables:${NC}"
        echo -e "${YELLOW}   - GROK_API_KEY${NC}"
        echo -e "${YELLOW}   - POSTGRES_PASSWORD${NC}"
        echo -e "${YELLOW}   - OPENWEBUI_API_KEY${NC}"
        echo -e "${YELLOW}   - WEBUI_SECRET_KEY${NC}"
        echo ""
        read -p "Press Enter after updating .env file..."
    else
        echo -e "${RED}❌ .env.example not found. Please create .env manually.${NC}"
        exit 1
    fi
fi

# Validate critical environment variables
echo -e "${YELLOW}🔍 Validating environment variables...${NC}"

source .env

if [ -z "$GROK_API_KEY" ] || [ "$GROK_API_KEY" = "your_actual_api_key_here" ]; then
    echo -e "${RED}❌ GROK_API_KEY not set in .env file${NC}"
    exit 1
fi
echo -e "${GREEN}✓ GROK_API_KEY is set${NC}"

if [ -z "$POSTGRES_PASSWORD" ] || [ "$POSTGRES_PASSWORD" = "ragbot_secret_2024" ]; then
    echo -e "${YELLOW}⚠️  Using default POSTGRES_PASSWORD (not recommended for production)${NC}"
fi

if [ -z "$OPENWEBUI_API_KEY" ] || [ "$OPENWEBUI_API_KEY" = "rag-bot-api-key-change-this" ]; then
    echo -e "${YELLOW}⚠️  Using default OPENWEBUI_API_KEY (not recommended for production)${NC}"
fi

# Check required environment variables are consistent
echo -e "${YELLOW}🔧 Verifying configuration consistency...${NC}"

EXPECTED_RELEVANCE="0.08"
ACTUAL_RELEVANCE="${MILVUS_MIN_RELEVANCE:-0.15}"

if [ "$ACTUAL_RELEVANCE" != "$EXPECTED_RELEVANCE" ]; then
    echo -e "${YELLOW}⚠️  MILVUS_MIN_RELEVANCE mismatch!${NC}"
    echo -e "${YELLOW}   Expected: $EXPECTED_RELEVANCE, Found: $ACTUAL_RELEVANCE${NC}"
    echo -e "${YELLOW}   Updating .env file...${NC}"
    
    if grep -q "MILVUS_MIN_RELEVANCE=" .env; then
        sed -i.bak "s/MILVUS_MIN_RELEVANCE=.*/MILVUS_MIN_RELEVANCE=$EXPECTED_RELEVANCE/" .env
    else
        echo "MILVUS_MIN_RELEVANCE=$EXPECTED_RELEVANCE" >> .env
    fi
    echo -e "${GREEN}✓ Fixed MILVUS_MIN_RELEVANCE${NC}"
fi

# Stop any existing containers
echo ""
echo -e "${YELLOW}🛑 Stopping existing containers (if any)...${NC}"
docker-compose down 2>/dev/null || true
echo -e "${GREEN}✓ Cleanup complete${NC}"

# Pull latest images
echo ""
echo -e "${YELLOW}📥 Pulling base images...${NC}"
docker-compose pull postgres etcd minio open-webui

# Build application
echo ""
echo -e "${YELLOW}🔨 Building application images...${NC}"
echo -e "${BLUE}   This may take 5-10 minutes on first build...${NC}"
docker-compose build --no-cache

# Start infrastructure services first
echo ""
echo -e "${YELLOW}🚀 Starting infrastructure services...${NC}"
docker-compose up -d postgres etcd minio

echo -e "${BLUE}   Waiting for infrastructure to be ready...${NC}"
sleep 10

# Start Milvus
echo ""
echo -e "${YELLOW}🚀 Starting Milvus vector database...${NC}"
docker-compose up -d milvus

echo -e "${BLUE}   Waiting for Milvus to initialize (60 seconds)...${NC}"
sleep 60

# Start RAG bot
echo ""
echo -e "${YELLOW}🚀 Starting Enterprise RAG Bot...${NC}"
docker-compose up -d enterprise-rag-bot

echo -e "${BLUE}   Waiting for RAG Bot to initialize (60 seconds)...${NC}"
sleep 60

# Start Open WebUI
echo ""
echo -e "${YELLOW}🚀 Starting Open WebUI...${NC}"
docker-compose up -d open-webui

echo -e "${BLUE}   Waiting for Open WebUI to initialize (30 seconds)...${NC}"
sleep 30

# Show status
echo ""
echo -e "${YELLOW}📊 Checking service status...${NC}"
docker-compose ps

# Run verification
echo ""
echo -e "${YELLOW}🔍 Running health checks...${NC}"

# Make verify script executable
chmod +x verify_startup.sh 2>/dev/null || true

if [ -f verify_startup.sh ]; then
    ./verify_startup.sh
else
    echo -e "${YELLOW}⚠️  verify_startup.sh not found, running basic checks...${NC}"
    
    # Basic health checks
    echo "Checking RAG Bot API..."
    if curl -f http://localhost:8001/health >/dev/null 2>&1; then
        echo -e "${GREEN}✓ RAG Bot API is healthy${NC}"
    else
        echo -e "${RED}✗ RAG Bot API is not responding${NC}"
    fi
    
    echo "Checking Open WebUI..."
    if curl -f http://localhost:3000/health >/dev/null 2>&1; then
        echo -e "${GREEN}✓ Open WebUI is healthy${NC}"
    else
        echo -e "${RED}✗ Open WebUI is not responding${NC}"
    fi
fi

# Display summary
echo ""
echo -e "${GREEN}"
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║              🎉 Deployment Complete!                           ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "${BLUE}🌐 Access your services:${NC}"
echo ""
echo -e "  ${GREEN}Open WebUI${NC}           http://localhost:3000"
echo -e "  ${GREEN}Admin Frontend${NC}       http://localhost:4200"
echo -e "  ${GREEN}User Frontend${NC}        http://localhost:4201"
echo -e "  ${GREEN}Admin API Docs${NC}       http://localhost:8000/docs"
echo -e "  ${GREEN}User API Docs${NC}        http://localhost:8001/docs"
echo ""

echo -e "${BLUE}📚 Next Steps:${NC}"
echo ""
echo "1. Open Open WebUI: http://localhost:3000"
echo "2. Create an account and log in"
echo "3. Verify 'Vayu Maya' model is available"
echo "4. Train the system with documents:"
echo "   - Via Admin UI: http://localhost:4200"
echo "   - Via API: See PRODUCTION_DEPLOYMENT.md"
echo "5. Test queries in Open WebUI"
echo ""

echo -e "${YELLOW}📊 Monitoring:${NC}"
echo ""
echo "  View logs:     docker-compose logs -f"
echo "  Check health:  curl http://localhost:8001/health"
echo "  Check stats:   curl http://localhost:8001/api/rag-widget/widget/knowledge-stats"
echo ""

echo -e "${YELLOW}⚠️  Important Notes:${NC}"
echo ""
echo "  • The system starts with an EMPTY knowledge base"
echo "  • You MUST add documents before getting meaningful responses"
echo "  • Refer to PRODUCTION_DEPLOYMENT.md for training guide"
echo "  • Monitor logs for any errors: docker-compose logs -f"
echo ""

echo -e "${GREEN}✓ Deployment script completed successfully!${NC}"
echo ""