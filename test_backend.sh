#!/bin/bash
# Quick backend test script

echo "🧪 Testing Enterprise RAG Bot Backend..."
echo "========================================"
echo ""

# Test 1: List all clusters
echo "📋 Test 1: List all clusters"
curl -s -X POST http://localhost:8001/api/widget/query \
  -H "Content-Type: application/json" \
  -d '{"query": "list all clusters"}' | jq '.response' | head -20

echo ""
echo "---"
echo ""

# Test 2: List clusters in specific location
echo "📍 Test 2: List clusters in Delhi"
curl -s -X POST http://localhost:8001/api/widget/query \
  -H "Content-Type: application/json" \
  -d '{"query": "list clusters in delhi"}' | jq '.response' | head -20

echo ""
echo "---"
echo ""

# Test 3: Start cluster creation
echo "🏗️ Test 3: Start cluster creation"
curl -s -X POST http://localhost:8001/api/widget/query \
  -H "Content-Type: application/json" \
  -d '{"query": "create a cluster"}' | jq '.response' | head -30

echo ""
echo "========================================"
echo "✅ Tests complete!"
echo ""
echo "💡 Check /tmp/user_main.log for detailed logs"
echo "💡 Use the widget on port 4201 for full testing"

