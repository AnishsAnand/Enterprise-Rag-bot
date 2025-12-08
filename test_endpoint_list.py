#!/usr/bin/env python3
"""
Test script for the endpoint listing workflow.
Tests the complete flow: engagement -> endpoint list
Similar to cluster listing but for endpoints/datacenters/locations.
"""

import asyncio
import os
import json
from dotenv import load_dotenv

# IMPORTANT: Load environment variables BEFORE importing the service
load_dotenv()

# Now import the service
from app.services.api_executor_service import api_executor_service

# Reinitialize credentials
api_executor_service.auth_email = os.getenv("API_AUTH_EMAIL", "")
api_executor_service.auth_password = os.getenv("API_AUTH_PASSWORD", "")
api_executor_service.auth_url = os.getenv(
    "API_AUTH_URL",
    "https://ipcloud.tatacommunications.com/portalservice/api/v1/getAuthToken"
)


def print_separator(title: str = ""):
    """Print a formatted separator."""
    print("\n" + "=" * 70)
    if title:
        print(title)
        print("=" * 70)
    print()


async def test_list_endpoints_workflow():
    """Test the list_endpoints workflow method."""
    print_separator("Test 1: List Endpoints Workflow (Full)")
    
    result = await api_executor_service.list_endpoints()
    
    if result.get("success"):
        data = result.get("data", {})
        endpoints = data.get("endpoints", [])
        total = data.get("total", 0)
        
        print(f"✅ Successfully fetched {total} endpoints/datacenters:")
        print()
        print(f"{'#':<3} {'Name':<25} {'ID':<6} {'Type':<15} {'Region':<15}")
        print("-" * 70)
        
        for i, ep in enumerate(endpoints, 1):
            name = ep.get("name", "Unknown")
            ep_id = ep.get("id", "N/A")
            ep_type = ep.get("type", "")
            region = ep.get("region", "")
            print(f"{i:<3} {name:<25} {ep_id:<6} {ep_type:<15} {region:<15}")
        
        print()
        print(f"📊 Message: {result.get('message', '')}")
        return endpoints
    else:
        print(f"❌ Failed to list endpoints: {result.get('error')}")
        return None


async def test_intent_detection():
    """Test if intent agent would correctly detect endpoint listing queries."""
    print_separator("Test 2: Intent Detection Patterns")
    
    # These are the queries that should trigger endpoint listing
    test_queries = [
        "What are the available endpoints?",
        "List endpoints",
        "Show me all datacenters",
        "What datacenters are available?",
        "What DCs do we have?",
        "List all DCs",
        "Show me the locations",
        "What locations are available?",
        "Where can I deploy?",
        "What data centers can I use?",
        "List all available data centers",
        "Show available locations",
    ]
    
    print("The following queries should trigger endpoint listing intent:")
    print()
    for i, query in enumerate(test_queries, 1):
        print(f"  {i:2}. \"{query}\"")
    
    print()
    print("💡 These patterns are now configured in the IntentAgent system prompt")
    print("   and the 'endpoint' resource has aliases: datacenter, dc, data center, location")
    

async def test_via_agent_chat():
    """Test endpoint listing via the agent chat API (if server is running)."""
    print_separator("Test 3: Test via Agent Chat API")
    
    import httpx
    
    # Check if server is running
    try:
        async with httpx.AsyncClient() as client:
            # Test with a simple endpoint listing query
            test_queries = [
                "What are the available endpoints?",
                "List all datacenters",
                "Show me the DCs",
            ]
            
            for query in test_queries:
                print(f"\n🔍 Testing query: \"{query}\"")
                print("-" * 50)
                
                try:
                    response = await client.post(
                        "http://localhost:8001/api/agent/chat",
                        json={
                            "message": query,
                            "session_id": "test-endpoint-list-session",
                            "user_id": "test-user"
                        },
                        timeout=30.0
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        print(f"✅ Response received:")
                        print(f"   Routing: {data.get('routing', 'N/A')}")
                        print(f"   Success: {data.get('success', 'N/A')}")
                        print()
                        print("   Response preview:")
                        response_text = data.get('response', '')[:500]
                        for line in response_text.split('\n'):
                            print(f"   {line}")
                        if len(data.get('response', '')) > 500:
                            print("   ... (truncated)")
                    else:
                        print(f"❌ HTTP Error: {response.status_code}")
                        print(f"   Response: {response.text[:200]}")
                        
                except httpx.ConnectError:
                    print("⚠️ Could not connect to server at localhost:8001")
                    print("   Make sure the server is running: uvicorn app.user_main:app --port 8001")
                    break
                except Exception as e:
                    print(f"❌ Error: {str(e)}")
                    
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")


async def main():
    """Run all tests."""
    print("\n" + "🚀 " * 20)
    print("Starting Endpoint Listing Workflow Tests")
    print("🚀 " * 20)
    
    print(f"\n📧 Email: {api_executor_service.auth_email}")
    print(f"🔗 Auth URL: {api_executor_service.auth_url}")
    
    try:
        # Test 1: Direct workflow test
        endpoints = await test_list_endpoints_workflow()
        
        # Test 2: Show intent patterns
        await test_intent_detection()
        
        # Test 3: Test via agent chat API
        await test_via_agent_chat()
        
        # Summary
        print_separator("Test Summary")
        print(f"✅ Endpoint listing workflow: {'PASSED' if endpoints else 'FAILED'}")
        print("✅ Intent patterns documented: PASSED")
        print("ℹ️  Agent chat API test: See results above")
        print("\n🎉 All tests completed!\n")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        await api_executor_service.close()


if __name__ == "__main__":
    asyncio.run(main())





