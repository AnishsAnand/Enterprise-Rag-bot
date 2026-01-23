#!/bin/bash

#############################################################################
# Test OpenAI-Compatible Endpoints
# Tests the /api/v1/models and /api/v1/chat/completions endpoints
#############################################################################

set -e

echo "🧪 Testing OpenAI-Compatible Endpoints"
echo "======================================================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

BASE_URL="http://localhost:8001"

echo ""
echo -e "${BLUE}Test 1: Health Check${NC}"
echo "-----------------------------------------------------------------------"
echo "GET $BASE_URL/health"
curl -s $BASE_URL/health | jq '.' || echo -e "${RED}❌ Health check failed${NC}"
echo ""

echo ""
echo -e "${BLUE}Test 2: List Models${NC}"
echo "-----------------------------------------------------------------------"
echo "GET $BASE_URL/api/v1/models"
MODELS_RESPONSE=$(curl -s $BASE_URL/api/v1/models)
echo "$MODELS_RESPONSE" | jq '.'
if echo "$MODELS_RESPONSE" | jq -e '.data[0].id' > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Models endpoint working${NC}"
else
    echo -e "${RED}❌ Models endpoint failed${NC}"
fi
echo ""

echo ""
echo -e "${BLUE}Test 3: Simple Chat Completion (Non-Streaming)${NC}"
echo "-----------------------------------------------------------------------"
echo "POST $BASE_URL/api/v1/chat/completions"
CHAT_RESPONSE=$(curl -s -X POST $BASE_URL/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "enterprise-rag-bot",
    "messages": [
      {"role": "user", "content": "Hello, can you help me?"}
    ],
    "stream": false
  }')

echo "$CHAT_RESPONSE" | jq '.'
if echo "$CHAT_RESPONSE" | jq -e '.choices[0].message.content' > /dev/null 2>&1; then
    RESPONSE_TEXT=$(echo "$CHAT_RESPONSE" | jq -r '.choices[0].message.content')
    echo -e "${GREEN}✅ Chat completion working${NC}"
    echo -e "${YELLOW}Response: ${NC}$RESPONSE_TEXT"
else
    echo -e "${RED}❌ Chat completion failed${NC}"
fi
echo ""

echo ""
echo -e "${BLUE}Test 4: Chat with RAG Question${NC}"
echo "-----------------------------------------------------------------------"
echo "POST $BASE_URL/api/v1/chat/completions"
RAG_RESPONSE=$(curl -s -X POST $BASE_URL/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "enterprise-rag-bot",
    "messages": [
      {"role": "user", "content": "How do I create a Kubernetes cluster?"}
    ],
    "stream": false
  }')

echo "$RAG_RESPONSE" | jq '.'
if echo "$RAG_RESPONSE" | jq -e '.choices[0].message.content' > /dev/null 2>&1; then
    RAG_TEXT=$(echo "$RAG_RESPONSE" | jq -r '.choices[0].message.content')
    echo -e "${GREEN}✅ RAG query working${NC}"
    echo -e "${YELLOW}Response: ${NC}$RAG_TEXT"
else
    echo -e "${RED}❌ RAG query failed${NC}"
fi
echo ""

echo ""
echo -e "${BLUE}Test 5: Cluster Creation Intent${NC}"
echo "-----------------------------------------------------------------------"
echo "POST $BASE_URL/api/v1/chat/completions"
CLUSTER_RESPONSE=$(curl -s -X POST $BASE_URL/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "enterprise-rag-bot",
    "messages": [
      {"role": "user", "content": "Create a new Kubernetes cluster"}
    ],
    "stream": false
  }')

echo "$CLUSTER_RESPONSE" | jq '.'
if echo "$CLUSTER_RESPONSE" | jq -e '.choices[0].message.content' > /dev/null 2>&1; then
    CLUSTER_TEXT=$(echo "$CLUSTER_RESPONSE" | jq -r '.choices[0].message.content')
    echo -e "${GREEN}✅ Cluster creation intent working${NC}"
    echo -e "${YELLOW}Response: ${NC}$CLUSTER_TEXT"
else
    echo -e "${RED}❌ Cluster creation failed${NC}"
fi
echo ""

echo ""
echo -e "${BLUE}Test 6: Streaming Response (SSE)${NC}"
echo "-----------------------------------------------------------------------"
echo "POST $BASE_URL/api/v1/chat/completions (stream=true)"
echo "First 20 lines of streaming response:"
curl -s -X POST $BASE_URL/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "enterprise-rag-bot",
    "messages": [
      {"role": "user", "content": "Tell me about your capabilities"}
    ],
    "stream": true
  }' | head -20

echo ""
echo -e "${GREEN}✅ Streaming test complete${NC}"
echo ""

echo ""
echo "📊 Summary"
echo "======================================================================="
echo "All tests completed. Review the results above."
echo ""
echo "If all tests passed:"
echo "  ✅ Your OpenAI-compatible API is working"
echo "  ✅ Ready to connect with Open WebUI"
echo "  ✅ Agent system is responding"
echo ""
echo "Next step:"
echo "  Open http://localhost:3000 and test with Open WebUI!"
echo ""
echo "======================================================================="

