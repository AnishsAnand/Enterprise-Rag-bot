#!/bin/bash

#############################################################################
# Test Backend Standalone (Port 8001)
# Tests the backend without Docker, directly with uvicorn
#############################################################################

set -e

echo "🧪 Testing Enterprise RAG Bot Backend (Standalone on Port 8001)"
echo "======================================================================="

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

cd /home/unixlogin/vayuMaya/Enterprise-Rag-bot

echo ""
echo -e "${BLUE}Step 1: Clearing Python cache${NC}"
echo "-----------------------------------------------------------------------"
find app -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find app -name "*.pyc" -delete 2>/dev/null || true
echo -e "${GREEN}✅ Cache cleared${NC}"

echo ""
echo -e "${BLUE}Step 2: Starting backend on port 8001${NC}"
echo "-----------------------------------------------------------------------"
pkill -9 -f "uvicorn app.main:app" 2>/dev/null || true
sleep 2

echo "Starting uvicorn..."
nohup uvicorn app.main:app --host 0.0.0.0 --port 8001 > backend_test.log 2>&1 &
BACKEND_PID=$!
echo "Backend started with PID: $BACKEND_PID"
echo "Waiting 15 seconds for startup..."
sleep 15

echo ""
echo -e "${BLUE}Step 3: Checking if backend is running${NC}"
echo "-----------------------------------------------------------------------"
if ps -p $BACKEND_PID > /dev/null; then
    echo -e "${GREEN}✅ Backend process is running (PID: $BACKEND_PID)${NC}"
else
    echo -e "${RED}❌ Backend process died. Check logs:${NC}"
    tail -50 backend_test.log
    exit 1
fi

echo ""
echo -e "${BLUE}Step 4: Testing Health Endpoint${NC}"
echo "-----------------------------------------------------------------------"
HEALTH_RESPONSE=$(curl -s http://localhost:8001/health 2>&1)
if echo "$HEALTH_RESPONSE" | jq . > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Health endpoint responding${NC}"
    echo "$HEALTH_RESPONSE" | jq .
else
    echo -e "${RED}❌ Health endpoint not responding properly${NC}"
    echo "Response: $HEALTH_RESPONSE"
    echo ""
    echo "Backend logs:"
    tail -50 backend_test.log
    exit 1
fi

echo ""
echo -e "${BLUE}Step 5: Testing OpenAI Models Endpoint${NC}"
echo "-----------------------------------------------------------------------"
MODELS_RESPONSE=$(curl -s http://localhost:8001/api/v1/models 2>&1)
if echo "$MODELS_RESPONSE" | jq . > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Models endpoint responding${NC}"
    echo "$MODELS_RESPONSE" | jq .
else
    echo -e "${RED}❌ Models endpoint failed${NC}"
    echo "Response: $MODELS_RESPONSE"
fi

echo ""
echo -e "${BLUE}Step 6: Testing Chat Completions${NC}"
echo "-----------------------------------------------------------------------"
CHAT_RESPONSE=$(curl -s -X POST http://localhost:8001/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "enterprise-rag-bot",
    "messages": [
      {"role": "user", "content": "Hello, test message"}
    ],
    "stream": false
  }' 2>&1)

if echo "$CHAT_RESPONSE" | jq . > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Chat completions working${NC}"
    RESPONSE_TEXT=$(echo "$CHAT_RESPONSE" | jq -r '.choices[0].message.content' 2>/dev/null)
    if [ ! -z "$RESPONSE_TEXT" ]; then
        echo -e "${YELLOW}Bot Response:${NC} $RESPONSE_TEXT"
    fi
    echo ""
    echo "Full response:"
    echo "$CHAT_RESPONSE" | jq .
else
    echo -e "${RED}❌ Chat completions failed${NC}"
    echo "Response: $CHAT_RESPONSE"
    echo ""
    echo "Backend logs (last 30 lines):"
    tail -30 backend_test.log
fi

echo ""
echo "========================================================================="
echo -e "${GREEN}✅ Backend Testing Complete${NC}"
echo "========================================================================="
echo ""
echo "Backend is running on: http://localhost:8001"
echo "API Docs: http://localhost:8001/docs"
echo "Health Check: http://localhost:8001/health"
echo ""
echo "View logs: tail -f backend_test.log"
echo "Stop backend: pkill -f 'uvicorn app.main:app'"
echo ""
echo "========================================================================="

