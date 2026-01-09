#!/bin/bash
# PostgreSQL Setup Script for Enterprise RAG Bot
# Run this script to set up PostgreSQL with pgvector

set -e  # Exit on error

echo "=========================================="
echo "PostgreSQL + pgvector Setup Script"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as root
if [ "$EUID" -eq 0 ]; then 
    echo -e "${RED}❌ Please do not run as root${NC}"
    exit 1
fi

# Detect OS
OS="unknown"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
    if [ -f /etc/debian_version ]; then
        DISTRO="debian"
    elif [ -f /etc/redhat-release ]; then
        DISTRO="redhat"
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
fi

echo "🖥️  Detected OS: $OS"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if Docker is available
if command_exists docker && command_exists docker-compose; then
    echo -e "${GREEN}✅ Docker found${NC}"
    DOCKER_AVAILABLE=true
else
    echo -e "${YELLOW}⚠️  Docker not found${NC}"
    DOCKER_AVAILABLE=false
fi

# Ask user for installation method
echo ""
echo "Choose installation method:"
echo "1) Docker (recommended - easier setup)"
echo "2) Native PostgreSQL installation"
echo ""
read -p "Enter choice [1-2]: " INSTALL_METHOD

if [ "$INSTALL_METHOD" == "1" ]; then
    if [ "$DOCKER_AVAILABLE" = false ]; then
        echo -e "${RED}❌ Docker not available. Please install Docker first.${NC}"
        exit 1
    fi
    
    echo ""
    echo "🐳 Starting PostgreSQL with Docker..."
    
    # Create docker-compose.yml if it doesn't exist
    if [ ! -f "docker-compose.yml" ]; then
        echo -e "${YELLOW}⚠️  docker-compose.yml not found. Creating...${NC}"
        cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  postgres:
    image: pgvector/pgvector:pg16
    container_name: enterprise_rag_postgres
    environment:
      POSTGRES_DB: enterprise_rag
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: changeme123
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres -d enterprise_rag"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

volumes:
  postgres_data:
EOF
    fi
    
    # Start Docker containers
    docker-compose up -d postgres
    
    # Wait for PostgreSQL to be ready
    echo "⏳ Waiting for PostgreSQL to be ready..."
    for i in {1..30}; do
        if docker-compose exec -T postgres pg_isready -U postgres > /dev/null 2>&1; then
            echo -e "${GREEN}✅ PostgreSQL is ready!${NC}"
            break
        fi
        echo -n "."
        sleep 2
    done
    
    # Enable pgvector extension
    echo "🔧 Enabling pgvector extension..."
    docker-compose exec -T postgres psql -U postgres -d enterprise_rag -c "CREATE EXTENSION IF NOT EXISTS vector;" > /dev/null 2>&1
    
    echo -e "${GREEN}✅ PostgreSQL with pgvector is running!${NC}"
    echo ""
    echo "Connection details:"
    echo "  Host: localhost"
    echo "  Port: 5432"
    echo "  Database: enterprise_rag"
    echo "  User: postgres"
    echo "  Password: changeme123"
    
elif [ "$INSTALL_METHOD" == "2" ]; then
    echo ""
    echo "📦 Installing PostgreSQL natively..."
    
    if [ "$OS" == "linux" ]; then
        if [ "$DISTRO" == "debian" ]; then
            echo "Installing PostgreSQL 16 on Debian/Ubuntu..."
            sudo apt-get update
            sudo apt-get install -y postgresql-16 postgresql-contrib-16
            
            # Install pgvector
            echo "Installing pgvector..."
            sudo apt-get install -y postgresql-16-pgvector
            
        elif [ "$DISTRO" == "redhat" ]; then
            echo "Installing PostgreSQL 16 on Red Hat/CentOS..."
            sudo dnf install -y postgresql16-server postgresql16-contrib
            sudo postgresql-setup --initdb
            sudo systemctl enable postgresql
            sudo systemctl start postgresql
            
            # Install pgvector (may need to build from source)
            echo -e "${YELLOW}⚠️  pgvector installation on Red Hat requires building from source${NC}"
            echo "See: https://github.com/pgvector/pgvector#installation"
        fi
        
    elif [ "$OS" == "macos" ]; then
        if command_exists brew; then
            echo "Installing PostgreSQL via Homebrew..."
            brew install postgresql@16
            brew install pgvector
            
            # Start PostgreSQL
            brew services start postgresql@16
        else
            echo -e "${RED}❌ Homebrew not found. Please install Homebrew first.${NC}"
            exit 1
        fi
    fi
    
    # Create database and enable extension
    echo "🔧 Setting up database..."
    sleep 3
    
    sudo -u postgres psql -c "CREATE DATABASE enterprise_rag;" 2>/dev/null || echo "Database may already exist"
    sudo -u postgres psql -d enterprise_rag -c "CREATE EXTENSION IF NOT EXISTS vector;" 2>/dev/null
    
    echo -e "${GREEN}✅ PostgreSQL installed and configured!${NC}"
    
else
    echo -e "${RED}❌ Invalid choice${NC}"
    exit 1
fi

# Update .env file
echo ""
echo "📝 Updating .env file..."

if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚠️  .env file not found. Creating...${NC}"
    touch .env
fi

# Check if PostgreSQL vars already exist
if grep -q "POSTGRES_HOST" .env; then
    echo -e "${YELLOW}⚠️  PostgreSQL configuration already exists in .env${NC}"
else
    echo "" >> .env
    echo "# PostgreSQL Configuration (Migration from Milvus)" >> .env
    echo "POSTGRES_HOST=localhost" >> .env
    echo "POSTGRES_PORT=5432" >> .env
    echo "POSTGRES_DB=enterprise_rag" >> .env
    echo "POSTGRES_USER=postgres" >> .env
    echo "POSTGRES_PASSWORD=changeme123" >> .env
    echo "POSTGRES_TABLE=enterprise_rag" >> .env
    echo "" >> .env
    echo "# Connection Pool" >> .env
    echo "POSTGRES_POOL_MIN=2" >> .env
    echo "POSTGRES_POOL_MAX=10" >> .env
    echo "" >> .env
    echo "# Search Configuration" >> .env
    echo "POSTGRES_MIN_RELEVANCE=0.08" >> .env
    echo "POSTGRES_MAX_INITIAL_RESULTS=200" >> .env
    echo "POSTGRES_RERANK_TOP_K=100" >> .env
    echo "POSTGRES_QUERY_TOP_K=100" >> .env
    echo "POSTGRES_ENABLE_QUERY_EXPANSION=true" >> .env
    echo "POSTGRES_ENABLE_SEMANTIC_RERANK=true" >> .env
    echo "POSTGRES_ENABLE_CONTEXT_ENRICHMENT=true" >> .env
    
    echo -e "${GREEN}✅ .env file updated${NC}"
fi

# Install Python dependencies
echo ""
echo "📦 Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    pip install asyncpg psycopg2-binary
    echo -e "${GREEN}✅ Python dependencies installed${NC}"
else
    echo -e "${YELLOW}⚠️  requirements.txt not found. Please install asyncpg manually:${NC}"
    echo "   pip install asyncpg psycopg2-binary"
fi

echo ""
echo "=========================================="
echo -e "${GREEN}✅ Setup Complete!${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Review .env file and update password if needed"
echo "2. Copy postgres_service.py to app/services/"
echo "3. Run migration script: python migrate_milvus_to_postgres.py"
echo "4. Update imports in your code files"
echo "5. Test the application"
echo ""
echo "Useful commands:"
echo "  - Test connection: python test_postgres_connection.py"
echo "  - View logs: docker-compose logs -f postgres"
echo "  - Stop: docker-compose down"
echo ""