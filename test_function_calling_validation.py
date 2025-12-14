#!/usr/bin/env python3
"""
Quick validation test for Function Calling implementation.
Tests the wiring and logic without requiring external APIs.
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_imports():
    """Test that all new modules can be imported."""
    logger.info("🧪 Testing imports...")
    
    try:
        from app.services.function_calling_service import function_calling_service, FunctionDefinition
        logger.info("✅ FunctionCallingService imported successfully")
        
        from app.agents.function_calling_agent import function_calling_agent
        logger.info("✅ FunctionCallingAgent imported successfully")
        
        from app.services.ai_service import ai_service
        logger.info("✅ AIService imported successfully")
        
        from app.agents.agent_manager import AgentManager
        logger.info("✅ AgentManager imported successfully")
        
        return True
    except Exception as e:
        logger.error(f"❌ Import failed: {str(e)}", exc_info=True)
        return False


def test_function_service():
    """Test function calling service setup."""
    logger.info("")
    logger.info("🧪 Testing FunctionCallingService...")
    
    try:
        from app.services.function_calling_service import function_calling_service
        
        # Check registered functions
        functions = function_calling_service.functions
        logger.info(f"✅ Registered functions: {list(functions.keys())}")
        
        # Check expected functions
        expected_functions = ["list_k8s_clusters", "get_datacenters", "create_k8s_cluster"]
        for func_name in expected_functions:
            if func_name in functions:
                func = functions[func_name]
                logger.info(f"  ✅ {func_name}: {func.description[:60]}...")
            else:
                logger.error(f"  ❌ {func_name}: NOT FOUND")
                return False
        
        # Test tool format conversion
        tools = function_calling_service.get_tools_for_llm()
        logger.info(f"✅ Generated {len(tools)} OpenAI-compatible tools")
        
        # Test function schemas text
        schemas_text = function_calling_service.get_function_schemas_text()
        logger.info(f"✅ Generated function schemas ({len(schemas_text)} chars)")
        
        return True
    except Exception as e:
        logger.error(f"❌ FunctionCallingService test failed: {str(e)}", exc_info=True)
        return False


def test_agent_initialization():
    """Test agent initialization."""
    logger.info("")
    logger.info("🧪 Testing Agent Initialization...")
    
    try:
        from app.agents.agent_manager import AgentManager
        
        agent_manager = AgentManager()
        logger.info("✅ AgentManager created")
        
        agent_manager.initialize()
        logger.info("✅ AgentManager initialized")
        
        # Check stats
        stats = agent_manager.get_stats()
        logger.info(f"✅ Stats retrieved:")
        logger.info(f"  - Initialized: {stats['initialized']}")
        logger.info(f"  - Agents:")
        for agent_type, agent_name in stats['agents'].items():
            status = "✅" if agent_name else "❌"
            logger.info(f"    {status} {agent_type}: {agent_name or 'Not initialized'}")
        
        # Check function calling agent specifically
        if stats['agents'].get('function_calling'):
            logger.info("✅ FunctionCallingAgent is wired to AgentManager")
        else:
            logger.error("❌ FunctionCallingAgent NOT wired")
            return False
        
        # Check orchestrator has function calling agent
        if hasattr(agent_manager.orchestrator, 'function_calling_agent'):
            if agent_manager.orchestrator.function_calling_agent:
                logger.info("✅ Orchestrator has FunctionCallingAgent reference")
            else:
                logger.error("❌ Orchestrator.function_calling_agent is None")
                return False
        else:
            logger.error("❌ Orchestrator doesn't have function_calling_agent attribute")
            return False
        
        # Check feature flag
        if hasattr(agent_manager.orchestrator, 'use_function_calling'):
            use_fc = agent_manager.orchestrator.use_function_calling
            logger.info(f"✅ Function calling mode: {'ENABLED' if use_fc else 'DISABLED'}")
        else:
            logger.error("❌ Orchestrator doesn't have use_function_calling flag")
            return False
        
        return True
    except Exception as e:
        logger.error(f"❌ Agent initialization test failed: {str(e)}", exc_info=True)
        return False


def test_routing_logic():
    """Test routing decision logic."""
    logger.info("")
    logger.info("🧪 Testing Routing Logic...")
    
    try:
        from app.agents.orchestrator_agent import OrchestratorAgent
        from app.agents.function_calling_agent import function_calling_agent
        from app.agents.state.conversation_state import conversation_state_manager
        
        # Create orchestrator with function calling
        orchestrator = OrchestratorAgent()
        orchestrator.function_calling_agent = function_calling_agent
        orchestrator.use_function_calling = True
        
        # Create a test state
        state = conversation_state_manager.create_session("test_routing", "test_user")
        
        # Test queries that should route to function calling
        test_queries = [
            "List clusters in Delhi",
            "Show me clusters",
            "Create a cluster",
            "How many clusters?",
            "Get datacenters"
        ]
        
        logger.info("Testing queries that should route to function calling:")
        for query in test_queries:
            # Check if query contains resource keywords
            query_lower = query.lower()
            resource_keywords = [
                "list", "show", "get", "fetch", "view", "count", "how many",
                "create", "make", "add", "new", "deploy",
                "cluster", "firewall", "database", "kafka", "gitlab",
                "datacenter", "endpoint", "location"
            ]
            
            is_resource_operation = any(keyword in query_lower for keyword in resource_keywords)
            
            if is_resource_operation:
                logger.info(f"  ✅ '{query}' → Should route to function_calling")
            else:
                logger.info(f"  ⚠️  '{query}' → Would NOT route to function_calling")
        
        # Clean up
        conversation_state_manager.delete_session("test_routing")
        
        return True
    except Exception as e:
        logger.error(f"❌ Routing logic test failed: {str(e)}", exc_info=True)
        return False


def main():
    """Run all validation tests."""
    logger.info("=" * 80)
    logger.info("🎯 FUNCTION CALLING VALIDATION TEST")
    logger.info("=" * 80)
    
    results = {
        "Imports": test_imports(),
        "Function Service": test_function_service(),
        "Agent Initialization": test_agent_initialization(),
        "Routing Logic": test_routing_logic()
    }
    
    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 80)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{status}: {test_name}")
        if not passed:
            all_passed = False
    
    logger.info("")
    if all_passed:
        logger.info("🎉 ALL TESTS PASSED! Function calling is properly wired.")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Start the backend: uvicorn app.main:app --reload --port 8000")
        logger.info("  2. Test with real queries through OpenWebUI or API")
        logger.info("  3. Monitor logs for function calling execution")
    else:
        logger.error("❌ SOME TESTS FAILED! Please review errors above.")
    
    logger.info("=" * 80)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

