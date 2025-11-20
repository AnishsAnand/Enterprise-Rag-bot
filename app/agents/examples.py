"""
Example Usage of Chatbot Agent System
Production-ready examples for common use cases
"""

import asyncio
from app.agents.captain_chatbot_agent import captain_chatbot_agent


async def example_enable_service():
    """Example: Enable a Kubernetes service"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Enable Kubernetes Service")
    print("="*70)
    
    response = await captain_chatbot_agent.process_user_query(
        "Enable the api-service"
    )
    
    print(f"Query: Enable the api-service")
    print(f"Intent: {response.get('intent')}")
    print(f"Summary: {response.get('summary')}")
    print(f"Actions Taken: {response.get('actions_taken')}")
    print(f"Status: {response.get('status')}")


async def example_search_documentation():
    """Example: Search documentation"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Search Documentation")
    print("="*70)
    
    response = await captain_chatbot_agent.process_user_query(
        "How do I enable Kubernetes services?"
    )
    
    print(f"Query: How do I enable Kubernetes services?")
    print(f"Intent: {response.get('intent')}")
    print(f"Summary: {response.get('summary')}")
    print(f"Status: {response.get('status')}")


async def example_get_service_health():
    """Example: Analyze service health"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Analyze Service Health")
    print("="*70)
    
    response = await captain_chatbot_agent.process_user_query(
        "Analyze the health of the worker service"
    )
    
    print(f"Query: Analyze the health of the worker service")
    print(f"Intent: {response.get('intent')}")
    print(f"Summary: {response.get('summary')}")
    print(f"Next Steps: {response.get('next_steps')}")
    print(f"Status: {response.get('status')}")


async def example_get_service_logs():
    """Example: Get service logs"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Get Service Logs")
    print("="*70)
    
    response = await captain_chatbot_agent.process_user_query(
        "Show me the logs for the database service"
    )
    
    print(f"Query: Show me the logs for the database service")
    print(f"Intent: {response.get('intent')}")
    print(f"Summary: {response.get('summary')}")
    print(f"Status: {response.get('status')}")


async def example_api_call():
    """Example: Call external API"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Call External API")
    print("="*70)
    
    response = await captain_chatbot_agent.process_user_query(
        "Call the payment API",
        context={
            "url": "https://api.example.com/payment",
            "method": "POST",
            "api_key": "sk_test_123456",
            "payload": {"amount": 100, "currency": "USD"}
        }
    )
    
    print(f"Query: Call the payment API")
    print(f"Intent: {response.get('intent')}")
    print(f"Summary: {response.get('summary')}")
    print(f"Status: {response.get('status')}")


async def example_agent_status():
    """Example: Get agent status"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Get Agent Status")
    print("="*70)
    
    status = captain_chatbot_agent.get_all_agents_status()
    
    print(f"Captain Agent Status: {status['captain']['status']}")
    print(f"Total Agents: {len(status['agents'])}")
    
    for agent_name, agent_status in status['agents'].items():
        print(f"\n{agent_name.upper()}:")
        print(f"  - Role: {agent_status['role']}")
        print(f"  - Status: {agent_status['status']}")
        print(f"  - Messages: {agent_status['message_count']}")
        print(f"  - Tool Calls: {agent_status['tool_calls']}")


async def run_all_examples():
    """Run all examples"""
    print("\n" + "="*70)
    print("CHATBOT AGENT SYSTEM - EXAMPLES")
    print("="*70)
    
    await example_enable_service()
    await example_search_documentation()
    await example_get_service_health()
    await example_get_service_logs()
    await example_api_call()
    await example_agent_status()
    
    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETED")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(run_all_examples())
