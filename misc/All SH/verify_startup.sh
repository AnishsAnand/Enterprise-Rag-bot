#!/bin/bash
# ==============================================================================
# PRODUCTION STARTUP VERIFICATION SCRIPT
# Verifies all services are healthy after docker-compose up
# ==============================================================================

set -e

echo "🚀 Enterprise RAG Bot - Production Startup Verification"
echo "========================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Service URLs
POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
POSTGRES_PORT="${POSTGRES_PORT:-5435}"
REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"
MILVUS_HOST="${MILVUS_HOST:-localhost}"
MILVUS_PORT="${MILVUS_PORT:-19530}"
RAG_ADMIN_HOST="${RAG_ADMIN_HOST:-localhost}"
RAG_ADMIN_PORT="${RAG_ADMIN_PORT:-8000}"
RAG_USER_HOST="${RAG_USER_HOST:-localhost}"
RAG_USER_PORT="${RAG_USER_PORT:-8001}"
OPENWEBUI_HOST="${OPENWEBUI_HOST:-localhost}"
OPENWEBUI_PORT="${OPENWEBUI_PORT:-3000}"

# Counter for checks
PASSED=0
FAILED=0
TOTAL=0

check_service() {
    local name=$1
    local host=$2
    local port=$3
    local timeout=${4:-5}
    
    TOTAL=$((TOTAL + 1))
    echo -n "Checking $name ($host:$port)... "
    
    if timeout $timeout bash -c "cat < /dev/null > /dev/tcp/$host/$port" 2>/dev/null; then
        echo -e "${GREEN}✓ OK${NC}"
        PASSED=$((PASSED + 1))
        return 0
    else
        echo -e "${RED}✗ FAILED${NC}"
        FAILED=$((FAILED + 1))
        return 1
    fi
}

check_http_endpoint() {
    local name=$1
    local url=$2
    local expected_code=${3:-200}
    local timeout=${4:-10}
    
    TOTAL=$((TOTAL + 1))
    echo -n "Checking $name ($url)... "
    
    response=$(curl -s -o /dev/null -w "%{http_code}" --max-time $timeout "$url" 2>/dev/null || echo "000")
    
    if [ "$response" = "$expected_code" ]; then
        echo -e "${GREEN}✓ OK (HTTP $response)${NC}"
        PASSED=$((PASSED + 1))
        return 0
    else
        echo -e "${RED}✗ FAILED (HTTP $response, expected $expected_code)${NC}"
        FAILED=$((FAILED + 1))
        return 1
    fi
}

check_postgres() {
    TOTAL=$((TOTAL + 1))
    echo -n "Checking PostgreSQL readiness... "
    
    if docker exec enterprise-rag-postgres pg_isready -U ragbot -d ragbot_sessions >/dev/null 2>&1; then
        echo -e "${GREEN}✓ OK${NC}"
        PASSED=$((PASSED + 1))
        return 0
    else
        echo -e "${RED}✗ FAILED${NC}"
        FAILED=$((FAILED + 1))
        return 1
    fi
}

check_redis() {
    TOTAL=$((TOTAL + 1))
    echo -n "Checking Redis ping... "
    
    if docker exec enterprise-rag-redis redis-cli ping | grep -q PONG; then
        echo -e "${GREEN}✓ OK${NC}"
        PASSED=$((PASSED + 1))
        return 0
    else
        echo -e "${RED}✗ FAILED${NC}"
        FAILED=$((FAILED + 1))
        return 1
    fi
}

check_milvus_health() {
    TOTAL=$((TOTAL + 1))
    echo -n "Checking Milvus health endpoint... "
    
    response=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "http://$MILVUS_HOST:9091/metrics" 2>/dev/null || echo "000")
    
    if [ "$response" = "200" ]; then
        echo -e "${GREEN}✓ OK${NC}"
        PASSED=$((PASSED + 1))
        return 0
    else
        echo -e "${RED}✗ FAILED (HTTP $response)${NC}"
        FAILED=$((FAILED + 1))
        return 1
    fi
}

check_rag_bot_health() {
    TOTAL=$((TOTAL + 1))
    echo -n "Checking RAG Bot health endpoint... "
    
    response=$(curl -s --max-time 10 "http://$RAG_USER_HOST:$RAG_USER_PORT/health" 2>/dev/null || echo "")
    
    if echo "$response" | grep -q '"status"'; then
        echo -e "${GREEN}✓ OK${NC}"
        echo "   Response: $response" | head -c 100
        PASSED=$((PASSED + 1))
        return 0
    else
        echo -e "${RED}✗ FAILED${NC}"
        echo "   Response: $response" | head -c 100
        FAILED=$((FAILED + 1))
        return 1
    fi
}

check_openwebui_health() {
    TOTAL=$((TOTAL + 1))
    echo -n "Checking Open WebUI health... "
    
    response=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "http://$OPENWEBUI_HOST:$OPENWEBUI_PORT/health" 2>/dev/null || echo "000")
    
    if [ "$response" = "200" ]; then
        echo -e "${GREEN}✓ OK${NC}"
        PASSED=$((PASSED + 1))
        return 0
    else
        echo -e "${RED}✗ FAILED (HTTP $response)${NC}"
        FAILED=$((FAILED + 1))
        return 1
    fi
}

check_milvus_collection() {
    TOTAL=$((TOTAL + 1))
    echo -n "Checking Milvus collection stats... "
    
    response=$(curl -s --max-time 15 "http://$RAG_USER_HOST:$RAG_USER_PORT/api/rag-widget/widget/knowledge-stats" 2>/dev/null || echo "")
    
    if echo "$response" | grep -q '"document_count"'; then
        doc_count=$(echo "$response" | grep -o '"document_count":[0-9]*' | grep -o '[0-9]*')
        echo -e "${GREEN}✓ OK (Documents: $doc_count)${NC}"
        PASSED=$((PASSED + 1))
        return 0
    else
        echo -e "${RED}✗ FAILED${NC}"
        FAILED=$((FAILED + 1))
        return 1
    fi
}

check_ai_service() {
    TOTAL=$((TOTAL + 1))
    echo -n "Checking AI service embedding test... "
    
    # Test via health endpoint which includes AI service check
    response=$(curl -s --max-time 20 "http://$RAG_USER_HOST:$RAG_USER_PORT/health" 2>/dev/null || echo "")
    
    if echo "$response" | grep -q '"ai_services"'; then
        echo -e "${GREEN}✓ OK${NC}"
        PASSED=$((PASSED + 1))
        return 0
    else
        echo -e "${RED}✗ FAILED${NC}"
        FAILED=$((FAILED + 1))
        return 1
    fi
}

# Main execution
echo "Starting health checks..."
echo ""

echo "=== Infrastructure Services ==="
check_service "PostgreSQL" "$POSTGRES_HOST" "$POSTGRES_PORT"
check_postgres
check_service "Redis" "$REDIS_HOST" "$REDIS_PORT"
check_redis
check_service "Etcd" "localhost" "2379"
check_service "MinIO" "localhost" "9000"
check_milvus_health

echo ""
echo "=== Application Services ==="
check_service "RAG Bot Admin API" "$RAG_ADMIN_HOST" "$RAG_ADMIN_PORT"
check_service "RAG Bot User API" "$RAG_USER_HOST" "$RAG_USER_PORT"
check_rag_bot_health
check_service "Open WebUI" "$OPENWEBUI_HOST" "$OPENWEBUI_PORT"
check_openwebui_health

echo ""
echo "=== Functional Checks ==="
check_milvus_collection
check_ai_service

# OpenAI API compatibility check
check_http_endpoint "OpenAI Models Endpoint" "http://$RAG_USER_HOST:$RAG_USER_PORT/api/v1/models" 200

echo ""
echo "========================================================"
echo "Summary:"
echo -e "  Passed: ${GREEN}$PASSED${NC}"
echo -e "  Failed: ${RED}$FAILED${NC}"
echo -e "  Total:  $TOTAL"
echo "========================================================"

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed! System is ready.${NC}"
    echo ""
    echo "🎉 Services are available at:"
    echo "   - Admin Frontend: http://localhost:4200"
    echo "   - User Frontend: http://localhost:4201"
    echo "   - Open WebUI: http://localhost:3000"
    echo "   - Admin API: http://localhost:8000/docs"
    echo "   - User API: http://localhost:8001/docs"
    exit 0
else
    echo -e "${RED}✗ Some checks failed. Please review the logs.${NC}"
    echo ""
    echo "Troubleshooting tips:"
    echo "  1. Check Docker logs: docker-compose logs -f"
    echo "  2. Verify environment variables in .env"
    echo "  3. Ensure all volumes are properly mounted"
    echo "  4. Check port availability: netstat -tuln | grep -E '(3000|4200|4201|8000|8001|19530)'"
    exit 1
fi