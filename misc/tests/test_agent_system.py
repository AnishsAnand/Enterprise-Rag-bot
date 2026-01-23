"""
Test script for the multi-agent system.
Run this after installing dependencies to verify the system works.
"""

import asyncio
import sys
import os

# Add app to path
sys.path.insert(0, os.path.dirname(__file__))


async def test_agent_imports():
    """Test that all agent modules can be imported."""
    print("🧪 Testing agent imports...")
    
    try:
        from app.agents import (
            AgentManager,
            get_agent_manager,
            OrchestratorAgent,
            IntentAgent,
            ValidationAgent,
            ExecutionAgent,
            RAGAgent,
            BaseAgent
        )
        print("✅ All agent modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {str(e)}")
        return False


async def test_state_management():
    """Test conversation state management."""
    print("\n🧪 Testing state management...")
    
    try:
        from app.agents.state import (
            ConversationState,
            ConversationStatus,
            conversation_state_manager
        )
        
        # Create a test session
        state = conversation_state_manager.create_session("test-session", "test-user")
        
        # Set intent
        state.set_intent(
            resource_type="k8s_cluster",
            operation="create",
            required_params=["name", "region"],
            optional_params=["tags"]
        )
        
        # Add parameters
        state.add_parameter("name", "test-cluster")
        
        # Check state
        assert state.intent == "create_k8s_cluster"
        assert "name" in state.collected_params
        assert "region" in state.missing_params
        
        # Cleanup
        conversation_state_manager.delete_session("test-session")
        
        print("✅ State management working correctly")
        return True
    except Exception as e:
        print(f"❌ State management test failed: {str(e)}")
        return False


async def test_api_executor():
    """Test API executor service."""
    print("\n🧪 Testing API executor service...")
    
    try:
        from app.services.api_executor_service import api_executor_service
        
        # Test resource config loading
        config = api_executor_service.get_resource_config("k8s_cluster")
        if config:
            print(f"   ✓ Resource schema loaded: {list(config.keys())}")
        else:
            print("   ⚠️ No resources configured in schema")
        
        # Test parameter validation
        validation_result = api_executor_service.validate_parameters(
            resource_type="k8s_cluster",
            operation="create",
            params={"name": "test"}
        )
        
        print(f"   ✓ Validation working: {validation_result.get('valid', False)}")
        
        print("✅ API executor service working correctly")
        return True
    except Exception as e:
        print(f"❌ API executor test failed: {str(e)}")
        return False


async def test_agent_initialization():
    """Test agent initialization."""
    print("\n🧪 Testing agent initialization...")
    
    try:
        from app.agents import get_agent_manager
        
        # Initialize agent manager (without services for now)
        manager = get_agent_manager()
        
        # Check initialization
        assert manager.initialized, "Agent manager not initialized"
        assert manager.orchestrator is not None, "Orchestrator not initialized"
        assert manager.intent_agent is not None, "Intent agent not initialized"
        assert manager.validation_agent is not None, "Validation agent not initialized"
        assert manager.execution_agent is not None, "Execution agent not initialized"
        assert manager.rag_agent is not None, "RAG agent not initialized"
        
        print("✅ All agents initialized successfully")
        print(f"   ✓ Orchestrator: {manager.orchestrator.agent_name}")
        print(f"   ✓ Intent: {manager.intent_agent.agent_name}")
        print(f"   ✓ Validation: {manager.validation_agent.agent_name}")
        print(f"   ✓ Execution: {manager.execution_agent.agent_name}")
        print(f"   ✓ RAG: {manager.rag_agent.agent_name}")
        
        return True
    except Exception as e:
        print(f"❌ Agent initialization test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_langchain_imports():
    """Test LangChain package imports."""
    print("\n🧪 Testing LangChain imports...")
    
    try:
        import langchain
        print(f"   ✓ langchain {langchain.__version__}")
        
        from langchain_openai import ChatOpenAI
        print(f"   ✓ langchain_openai")
        
        from langchain_community import __version__ as lc_community_version
        print(f"   ✓ langchain_community")
        
        import langgraph
        print(f"   ✓ langgraph")
        
        print("✅ All LangChain packages available")
        return True
    except ImportError as e:
        print(f"❌ LangChain import failed: {str(e)}")
        print("\n💡 To fix, run:")
        print("   source .venv/bin/activate")
        print("   pip install langchain==0.1.0 langchain-openai langchain-community langgraph")
        return False


async def run_all_tests():
    """Run all tests."""
    print("=" * 70)
    print("🚀 Multi-Agent System Test Suite")
    print("=" * 70)
    
    results = []
    
    # Test 1: LangChain imports
    results.append(await test_langchain_imports())
    
    # Test 2: Agent imports
    results.append(await test_agent_imports())
    
    # Test 3: State management
    results.append(await test_state_management())
    
    # Test 4: API executor
    results.append(await test_api_executor())
    
    # Test 5: Agent initialization (only if LangChain is available)
    if results[0]:  # LangChain imports succeeded
        results.append(await test_agent_initialization())
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 Test Summary")
    print("=" * 70)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✅ All tests passed! The multi-agent system is ready to use.")
        print("\n🚀 Next steps:")
        print("   1. Start the application: uvicorn app.main:app --reload")
        print("   2. Visit http://localhost:8000/docs for API documentation")
        print("   3. Test the agent chat endpoint: POST /api/agent/chat")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please check the errors above.")
        
        if not results[0]:  # LangChain import failed
            print("\n💡 Install missing dependencies:")
            print("   source .venv/bin/activate")
            print("   pip install langchain==0.1.0 langchain-openai langchain-community langgraph")
    
    print("=" * 70)
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)

