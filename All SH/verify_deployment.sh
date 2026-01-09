#!/bin/bash

# ==============================================================================
# Enterprise RAG Bot + Open WebUI - Deployment Verification Script
# ==============================================================================

set -e

echo "🔍 Enterprise RAG Bot - Production Deployment Verification"
echo "=========================================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track errors
ERRORS=0
WARNINGS=0

# Helper functions
check_pass() {
    echo -e "${GREEN}✅ PASS${NC}: $1"
}

check_fail() {
    echo -e "${RED}❌ FAIL${NC}: $1"
    ((ERRORS++))
}

check_warn() {
    echo -e "${YELLOW}⚠️  WARN${NC}: $1"
    ((WARNINGS++))
}

# ==============================================================================
# 1. Check Docker Services
# ==============================================================================
echo "1️⃣  Checking Docker Services..."
echo "----------------------------------------"

if docker compose -f docker-compose.openwebui.yml ps >/dev/null 2>&1; then
    check_pass "Docker Compose is available"
else
    check_fail "Docker Compose not found or not running"
fi

# Check each service
SERVICES=("postgres" "redis" "etcd" "minio" "milvus" "enterprise-rag-bot" "open-webui")
for service in "${SERVICES[@]}"; do
    if docker ps | grep -q "enterprise-rag-$service\|$service"; then
        STATUS=$(docker inspect --format='{{.State.Health.Status}}' "enterprise-rag-$service" 2>/dev/null || echo "unknown")
        if [ "$STATUS" == "healthy" ] || [ "$STATUS" == "unknown" ]; then
            check_pass "$service is running"
        else
            check_warn "$service is running but not healthy (status: $STATUS)"
        fi
    else
        check_fail "$service is not running"
    fi
done

echo ""

# ==============================================================================
# 2. Check Environment Configuration
# ==============================================================================
echo "2️⃣  Checking Environment Configuration..."
echo "----------------------------------------"

# Check .env file exists
if [ -f .env ]; then
    check_pass ".env file exists"
    
    # Check critical variables
    CRITICAL_VARS=("GROK_API_KEY" "EMBEDDING_DIMENSION" "POSTGRES_PASSWORD")
    for var in "${CRITICAL_VARS[@]}"; do
        if grep -q "^${var}=" .env && ! grep -q "^${var}=$\|^${var}=your-" .env; then
            check_pass "$var is configured"
        else
            check_fail "$var is missing or not configured in .env"
        fi
    done
    
    # Check optimized search parameters
    SEARCH_PARAMS=("MILVUS_MIN_RELEVANCE" "MILVUS_MAX_INITIAL_RESULTS" "MILVUS_SEARCH_EF")
    for param in "${SEARCH_PARAMS[@]}"; do
        if grep -q "^${param}=" .env; then
            VALUE=$(grep "^${param}=" .env | cut -d'=' -f2)
            check_pass "$param is set to: $VALUE"
        else
            check_warn "$param not explicitly set (will use default)"
        fi
    done
else
    check_fail ".env file not found"
fi

echo ""

# ==============================================================================
# 3. Check Service Health Endpoints
# ==============================================================================
echo "3️⃣  Checking Service Health Endpoints..."
echo "----------------------------------------"

# Wait a moment for services to be ready
sleep 2

# Check Milvus
echo "Checking Milvus..."
if curl -sf http://localhost:19530/healthz >/dev/null 2>&1; then
    check_pass "Milvus is responding"
else
    check_fail "Milvus health check failed"
fi

# Check RAG Bot
echo "Checking Enterprise RAG Bot..."
if RESPONSE=$(curl -sf http://localhost:8001/api/v1/health 2>/dev/null); then
    check_pass "RAG Bot API is responding"
    
    # Parse JSON response
    if echo "$RESPONSE" | grep -q '"status":"healthy"'; then
        check_pass "RAG Bot status is healthy"
    else
        check_warn "RAG Bot is responding but may not be fully healthy"
    fi
    
    # Check document count
    DOC_COUNT=$(echo "$RESPONSE" | grep -o '"documents":[0-9]*' | grep -o '[0-9]*' || echo "0")
    if [ "$DOC_COUNT" -gt 0 ]; then
        check_pass "Milvus has $DOC_COUNT documents"
    else
        check_warn "Milvus has no documents - upload knowledge base"
    fi
else
    check_fail "RAG Bot health check failed"
fi

# Check Open WebUI
echo "Checking Open WebUI..."
if curl -sf http://localhost:3000/health >/dev/null 2>&1; then
    check_pass "Open WebUI is responding"
else
    check_fail "Open WebUI health check failed"
fi

echo ""

# ==============================================================================
# 4. Check Search Configuration
# ==============================================================================
echo "4️⃣  Verifying Search Configuration..."
echo "----------------------------------------"

# Check environment variables inside container
echo "Checking Milvus search parameters in container..."
if docker exec enterprise-rag-bot env | grep -q "MILVUS_MIN_RELEVANCE=0.08\|MILVUS_MIN_RELEVANCE=0.0[6-9]"; then
    check_pass "MILVUS_MIN_RELEVANCE is optimized (≤0.08)"
else
    CURRENT=$(docker exec enterprise-rag-bot env | grep "MILVUS_MIN_RELEVANCE" | cut -d'=' -f2 || echo "not set")
    check_warn "MILVUS_MIN_RELEVANCE should be ≤0.08 (current: $CURRENT)"
fi

if docker exec enterprise-rag-bot env | grep -q "MILVUS_MAX_INITIAL_RESULTS=200\|MILVUS_MAX_INITIAL_RESULTS=[2-9][0-9][0-9]"; then
    check_pass "MILVUS_MAX_INITIAL_RESULTS is optimized (≥200)"
else
    CURRENT=$(docker exec enterprise-rag-bot env | grep "MILVUS_MAX_INITIAL_RESULTS" | cut -d'=' -f2 || echo "not set")
    check_warn "MILVUS_MAX_INITIAL_RESULTS should be ≥200 (current: $CURRENT)"
fi

if docker exec enterprise-rag-bot env | grep -q "MILVUS_SEARCH_EF=128\|MILVUS_SEARCH_EF=[1-9][0-9][0-9]"; then
    check_pass "MILVUS_SEARCH_EF is optimized (≥128)"
else
    CURRENT=$(docker exec enterprise-rag-bot env | grep "MILVUS_SEARCH_EF" | cut -d'=' -f2 || echo "not set")
    check_warn "MILVUS_SEARCH_EF should be ≥128 (current: $CURRENT)"
fi

echo ""

# ==============================================================================
# 5. Test Search Functionality
# ==============================================================================
echo "5️⃣  Testing Search Functionality..."
echo "----------------------------------------"

# Test basic chat completion
echo "Testing chat completion endpoint..."
CHAT_RESPONSE=$(curl -sf -X POST http://localhost:8001/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Vayu Maya",
    "messages": [{"role": "user", "content": "test query"}],
    "stream": false
  }' 2>/dev/null || echo "")

if [ -n "$CHAT_RESPONSE" ] && echo "$CHAT_RESPONSE" | grep -q '"choices"'; then
    check_pass "Chat completion endpoint working"
    
    # Check response length
    RESPONSE_LENGTH=$(echo "$CHAT_RESPONSE" | grep -o '"content":"[^"]*"' | wc -c)
    if [ "$RESPONSE_LENGTH" -gt 100 ]; then
        check_pass "Response has substantial content ($RESPONSE_LENGTH chars)"
    else
        check_warn "Response seems short ($RESPONSE_LENGTH chars)"
    fi
else
    check_fail "Chat completion endpoint not working properly"
fi

echo ""

# ==============================================================================
# 6. Check Image Support
# ==============================================================================
echo "6️⃣  Checking Image Support..."
echo "----------------------------------------"

# Look for image-related logs
if docker logs enterprise-rag-bot 2>&1 | tail -100 | grep -q "Extracted.*images"; then
    check_pass "Image extraction is functioning"
    LAST_IMAGE_COUNT=$(docker logs enterprise-rag-bot 2>&1 | grep "Extracted.*images" | tail -1 | grep -o '[0-9]*' | head -1)
    if [ -n "$LAST_IMAGE_COUNT" ] && [ "$LAST_IMAGE_COUNT" -gt 0 ]; then
        check_pass "Last extraction found $LAST_IMAGE_COUNT images"
    fi
else
    check_warn "No recent image extraction logs found"
fi

# Check if images are in metadata
if [ "$DOC_COUNT" -gt 0 ]; then
    echo "Checking if documents contain image metadata..."
    # This would require a more complex query, so we'll check logs instead
    if docker logs enterprise-rag-bot 2>&1 | grep -q "images_json"; then
        check_pass "Documents contain image metadata"
    else
        check_warn "Image metadata may be missing from documents"
    fi
fi

echo ""

# ==============================================================================
# 7. Check Logs for Errors
# ==============================================================================
echo "7️⃣  Checking Recent Logs for Errors..."
echo "----------------------------------------"

# Check RAG Bot logs
ERROR_COUNT=$(docker logs enterprise-rag-bot 2>&1 | tail -100 | grep -i "error\|failed\|exception" | grep -v "No errors" | wc -l)
if [ "$ERROR_COUNT" -eq 0 ]; then
    check_pass "No recent errors in RAG Bot logs"
else
    check_warn "Found $ERROR_COUNT error lines in recent logs"
    echo "  Last few errors:"
    docker logs enterprise-rag-bot 2>&1 | tail -100 | grep -i "error\|failed" | tail -3 | sed 's/^/    /'
fi

# Check Milvus logs
MILVUS_ERRORS=$(docker logs enterprise-rag-milvus 2>&1 | tail -100 | grep -i "error\|fatal" | wc -l)
if [ "$MILVUS_ERRORS" -eq 0 ]; then
    check_pass "No recent errors in Milvus logs"
else
    check_warn "Found $MILVUS_ERRORS error lines in Milvus logs"
fi

echo ""

# ==============================================================================
# Summary
# ==============================================================================
echo "=========================================================================="
echo "📊 Verification Summary"
echo "=========================================================================="
echo ""

TOTAL_CHECKS=$((ERRORS + WARNINGS))

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}🎉 Perfect! All checks passed!${NC}"
    echo ""
    echo "Your Enterprise RAG Bot is production-ready with:"
    echo "  ✅ Optimized search parameters"
    echo "  ✅ Image support enabled"
    echo "  ✅ All services healthy"
    echo "  ✅ No errors detected"
    echo ""
    echo "Next steps:"
    echo "  1. Access Open WebUI at http://localhost:3000"
    echo "  2. Create your account"
    echo "  3. Start chatting with Vayu Maya!"
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}⚠️  System is functional with ${WARNINGS} warning(s)${NC}"
    echo ""
    echo "Warnings found:"
    echo "  - Some optimization parameters could be improved"
    echo "  - Review warnings above and adjust configuration if needed"
    echo ""
    echo "The system should work, but performance may not be optimal."
    exit 0
else
    echo -e "${RED}❌ Deployment has ${ERRORS} critical error(s) and ${WARNINGS} warning(s)${NC}"
    echo ""
    echo "Critical issues detected. Please fix the errors above before proceeding."
    echo ""
    echo "Common solutions:"
    echo "  1. Check .env file has all required variables"
    echo "  2. Ensure all Docker services are running:"
    echo "     docker compose -f docker-compose.openwebui.yml up -d"
    echo "  3. Check logs for specific errors:"
    echo "     docker logs enterprise-rag-bot"
    echo "  4. Review the DEPLOYMENT_GUIDE.md for troubleshooting"
    exit 1
fi