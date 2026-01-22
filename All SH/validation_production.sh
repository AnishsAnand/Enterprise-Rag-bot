#!/bin/bash
# Complete Production Validation Script
# Tests all components including image loading and Open WebUI integration

set -e

echo "=============================================="
echo "🔍 Production Validation - Enterprise RAG Bot"
echo "=============================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

ERRORS=0
WARNINGS=0
TESTS_PASSED=0
TESTS_FAILED=0

# ==============================================================================
# Helper Functions
# ==============================================================================

test_passed() {
    echo -e "${GREEN}✅ $1${NC}"
    TESTS_PASSED=$((TESTS_PASSED + 1))
}

test_failed() {
    echo -e "${RED}❌ $1${NC}"
    TESTS_FAILED=$((TESTS_FAILED + 1))
    ERRORS=$((ERRORS + 1))
}

test_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
    WARNINGS=$((WARNINGS + 1))
}

# ==============================================================================
# Test 1: Container Health
# ==============================================================================

echo -e "${BLUE}TEST 1: Container Health Checks${NC}"
echo "-------------------------------------------"

services=(
    "enterprise-rag-bot:8001"
    "enterprise-rag-postgres:5432"
    "enterprise-rag-milvus:9091"
    "enterprise-rag-minio:9000"
    "enterprise-rag-openwebui:8080"
)

for service in "${services[@]}"; do
    container=$(echo $service | cut -d: -f1)
    port=$(echo $service | cut -d: -f2)
    
    if docker ps | grep -q "$container"; then
        if docker ps | grep "$container" | grep -q "(healthy)"; then
            test_passed "$container is healthy"
        else
            test_warning "$container is running but not healthy yet"
        fi
    else
        test_failed "$container is not running"
    fi
done

echo ""

# ==============================================================================
# Test 2: HTTP Endpoints
# ==============================================================================

echo -e "${BLUE}TEST 2: HTTP Endpoint Checks${NC}"
echo "-------------------------------------------"

# User backend health
if curl -sf http://localhost:8001/health > /dev/null 2>&1; then
    test_passed "User backend health endpoint (8001)"
else
    test_failed "User backend health endpoint unreachable"
fi

# Admin backend health
if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    test_passed "Admin backend health endpoint (8000)"
else
    test_failed "Admin backend health endpoint unreachable"
fi

# Open WebUI health
if curl -sf http://localhost:3000 > /dev/null 2>&1; then
    test_passed "Open WebUI is accessible (3000)"
else
    test_failed "Open WebUI is not accessible"
fi

# Milvus metrics
if curl -sf http://localhost:9091/metrics > /dev/null 2>&1; then
    test_passed "Milvus metrics endpoint (9091)"
else
    test_failed "Milvus is not responding"
fi

echo ""

# ==============================================================================
# Test 3: OpenAI-Compatible Endpoints
# ==============================================================================

echo -e "${BLUE}TEST 3: OpenAI-Compatible API${NC}"
echo "-------------------------------------------"

# List models
models_response=$(curl -sf http://localhost:8001/api/v1/models 2>/dev/null || echo "{}")
if echo "$models_response" | grep -q "enterprise-rag-bot"; then
    test_passed "OpenAI models endpoint returns enterprise-rag-bot"
else
    test_failed "OpenAI models endpoint not working"
    echo "Response: $models_response"
fi

# Test chat completion (simple)
completion_response=$(curl -sf -X POST http://localhost:8001/api/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "enterprise-rag-bot",
        "messages": [{"role": "user", "content": "test"}],
        "stream": false
    }' 2>/dev/null || echo "{}")

if echo "$completion_response" | grep -q '"role": "assistant"'; then
    test_passed "OpenAI chat completions endpoint working"
else
    test_failed "OpenAI chat completions endpoint not working"
    echo "Response: ${completion_response:0:200}"
fi

echo ""

# ==============================================================================
# Test 4: Image Loading Test
# ==============================================================================

echo -e "${BLUE}TEST 4: Image Loading Verification${NC}"
echo "-------------------------------------------"

# Test with query that should return images
image_test_response=$(curl -sf -X POST http://localhost:8001/api/chat/query \
    -H "Content-Type: application/json" \
    -d '{
        "query": "kubernetes documentation",
        "max_results": 5,
        "include_images": true
    }' 2>/dev/null || echo "{}")

if echo "$image_test_response" | grep -q '"images"'; then
    test_passed "Response includes images field"
    
    # Count images
    image_count=$(echo "$image_test_response" | grep -o '"url"' | wc -l)
    if [ "$image_count" -gt 0 ]; then
        test_passed "Found $image_count image(s) in response"
    else
        test_warning "Images field present but empty (may be expected if no docs with images)"
    fi
    
    # Validate image URLs
    if echo "$image_test_response" | grep -q '"url": "http'; then
        test_passed "Image URLs are properly formatted (http/https)"
    else
        test_warning "No valid image URLs found"
    fi
else
    test_failed "Response missing images field"
    echo "Response structure: $(echo "$image_test_response" | jq 'keys' 2>/dev/null || echo 'Invalid JSON')"
fi

echo ""

# ==============================================================================
# Test 5: Step Images Test
# ==============================================================================

echo -e "${BLUE}TEST 5: Step Images Verification${NC}"
echo "-------------------------------------------"

if echo "$image_test_response" | grep -q '"steps"'; then
    test_passed "Response includes steps field"
    
    # Check if steps have images
    if echo "$image_test_response" | grep -A 5 '"steps"' | grep -q '"image"'; then
        test_passed "Steps include image field"
    else
        test_warning "Steps present but no images attached (may be expected)"
    fi
else
    test_warning "Response missing steps field"
fi

echo ""

# ==============================================================================
# Test 6: Response Quality Test
# ==============================================================================

echo -e "${BLUE}TEST 6: Response Quality Checks${NC}"
echo "-------------------------------------------"

if echo "$image_test_response" | grep -q '"answer"'; then
    test_passed "Response includes answer field"
    
    answer_length=$(echo "$image_test_response" | jq -r '.answer | length' 2>/dev/null || echo "0")
    if [ "$answer_length" -gt 50 ]; then
        test_passed "Answer length is adequate ($answer_length chars)"
    else
        test_warning "Answer is short ($answer_length chars)"
    fi
else
    test_failed "Response missing answer field"
fi

if echo "$image_test_response" | grep -q '"confidence"'; then
    confidence=$(echo "$image_test_response" | jq -r '.confidence' 2>/dev/null || echo "0")
    test_passed "Confidence score present: $confidence"
else
    test_warning "Confidence score missing"
fi

echo ""

# ==============================================================================
# Test 7: Milvus Collection Status
# ==============================================================================

echo -e "${BLUE}TEST 7: Milvus Collection Verification${NC}"
echo "-------------------------------------------"

# Test Milvus connection from within container
milvus_test=$(docker exec enterprise-rag-bot python -c "
import asyncio
import json
from app.services.milvus_service import milvus_service

async def test():
    try:
        await milvus_service.initialize()
        stats = await milvus_service.get_collection_stats()
        print(json.dumps(stats))
    except Exception as e:
        print(json.dumps({'error': str(e)}))

asyncio.run(test())
" 2>/dev/null || echo '{"error": "Failed to run test"}')

if echo "$milvus_test" | grep -q '"status"'; then
    status=$(echo "$milvus_test" | jq -r '.status' 2>/dev/null || echo "unknown")
    doc_count=$(echo "$milvus_test" | jq -r '.document_count' 2>/dev/null || echo "0")
    
    test_passed "Milvus collection status: $status"
    test_passed "Documents in collection: $doc_count"
    
    if [ "$doc_count" -eq 0 ]; then
        test_warning "No documents in collection yet - upload docs via admin panel"
    fi
else
    test_failed "Could not connect to Milvus collection"
    echo "Error: $milvus_test"
fi

echo ""

# ==============================================================================
# Test 8: File Structure Validation
# ==============================================================================

echo -e "${BLUE}TEST 8: Critical Files Check${NC}"
echo "-------------------------------------------"

critical_files=(
    "main.py"
    "app/user_main.py"
    "Dockerfile"
    "docker-compose.openwebui.yml"
    ".env"
    "misc/config/supervisord.conf"
    "misc/config/default.conf"
    "misc/scripts/entrypoint.sh"
)

for file in "${critical_files[@]}"; do
    if docker exec enterprise-rag-bot test -f "/app/$file" 2>/dev/null || test -f "$file"; then
        test_passed "$file exists"
    else
        test_failed "$file is missing"
    fi
done

echo ""

# ==============================================================================
# Test 9: Log Files Check
# ==============================================================================

echo -e "${BLUE}TEST 9: Application Logs${NC}"
echo "-------------------------------------------"

# Check if log files exist and have recent entries
if docker exec enterprise-rag-bot test -f /app/logs/user_backend.log 2>/dev/null; then
    test_passed "User backend log file exists"
    
    # Check for recent errors
    error_count=$(docker exec enterprise-rag-bot grep -c ERROR /app/logs/user_backend.log 2>/dev/null || echo "0")
    if [ "$error_count" -eq 0 ]; then
        test_passed "No errors in user backend log"
    else
        test_warning "$error_count errors found in user backend log"
    fi
else
    test_warning "User backend log file not found"
fi

if docker exec enterprise-rag-bot test -f /app/logs/admin_backend.log 2>/dev/null; then
    test_passed "Admin backend log file exists"
else
    test_warning "Admin backend log file not found"
fi

echo ""

# ==============================================================================
# Test 10: Performance Test
# ==============================================================================

echo -e "${BLUE}TEST 10: Performance Test${NC}"
echo "-------------------------------------------"

start_time=$(date +%s%3N)
perf_response=$(curl -sf -X POST http://localhost:8001/api/chat/query \
    -H "Content-Type: application/json" \
    -d '{"query": "test performance", "max_results": 5}' 2>/dev/null || echo "{}")
end_time=$(date +%s%3N)

response_time=$((end_time - start_time))

if echo "$perf_response" | grep -q '"answer"'; then
    if [ "$response_time" -lt 5000 ]; then
        test_passed "Response time: ${response_time}ms (excellent)"
    elif [ "$response_time" -lt 10000 ]; then
        test_passed "Response time: ${response_time}ms (acceptable)"
    else
        test_warning "Response time: ${response_time}ms (slow)"
    fi
else
    test_failed "Performance test failed - no response"
fi

echo ""

# ==============================================================================
# Summary
# ==============================================================================

echo "=============================================="
echo "📊 Validation Summary"
echo "=============================================="
echo ""
echo -e "${GREEN}Tests Passed: $TESTS_PASSED${NC}"
echo -e "${RED}Tests Failed: $TESTS_FAILED${NC}"
echo -e "${YELLOW}Warnings: $WARNINGS${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}=============================================="
    echo "✅ ALL TESTS PASSED - PRODUCTION READY"
    echo "==============================================
${NC}"
    echo ""
    echo "Access your services:"
    echo "  🌐 Open WebUI:    http://localhost:3000"
    echo "  📱 User Frontend: http://localhost:4201"
    echo "  ⚙️  Admin Panel:   http://localhost:4200"
    echo "  📚 API Docs:      http://localhost:8001/docs"
    echo ""
    exit 0
else
    echo -e "${RED}=============================================="
    echo "❌ SOME TESTS FAILED - REVIEW REQUIRED"
    echo "==============================================
${NC}"
    echo ""
    echo "Check logs:"
    echo "  docker-compose -f docker-compose.openwebui.yml logs -f"
    echo ""
    exit 1
fi