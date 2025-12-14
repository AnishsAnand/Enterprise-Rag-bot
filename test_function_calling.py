#!/usr/bin/env python3
"""
Test script for Function Calling implementation.
Tests the new modern approach for cluster listing.
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.agents.agent_manager import AgentManager
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def test_function_calling():
    """Test function calling with cluster listing queries."""
    
    logger.info("=" * 80)
    logger.info("🧪 TESTING FUNCTION CALLING IMPLEMENTATION")
    logger.info("=" * 80)
    
    # Initialize agent manager
    agent_manager = AgentManager()
    agent_manager.initialize()
    
    # Test cases
    test_cases = [
        {
            "query": "List clusters in Delhi",
            "description": "Test location-based cluster listing (Delhi)"
        },
        {
            "query": "Show me clusters in Mumbai",
            "description": "Test location-based cluster listing (Mumbai)"
        },
        {
            "query": "List all clusters",
            "description": "Test listing all clusters"
        },
        {
            "query": "How many clusters are in Chennai?",
            "description": "Test cluster counting with location"
        },
        {
            "query": "What clusters do we have?",
            "description": "Test general cluster query"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        query = test_case["query"]
        description = test_case["description"]
        
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"TEST {i}/{len(test_cases)}: {description}")
        logger.info(f"Query: '{query}'")
        logger.info("=" * 80)
        
        try:
            # Process request
            result = await agent_manager.process_request(
                user_input=query,
                session_id=f"test_session_{i}",
                user_id="test_user",
                user_roles=["admin"]
            )
            
            # Display results
            logger.info("")
            logger.info("📊 RESULTS:")
            logger.info(f"  Success: {result.get('success')}")
            logger.info(f"  Routing: {result.get('routing', 'N/A')}")
            
            if result.get('metadata'):
                metadata = result['metadata']
                logger.info(f"  Duration: {metadata.get('duration_seconds', 0):.2f}s")
                if 'agent_type' in metadata:
                    logger.info(f"  Agent Type: {metadata.get('agent_type')}")
                    logger.info(f"  Modern Approach: {metadata.get('modern_approach', False)}")
            
            if result.get('iterations'):
                logger.info(f"  Iterations: {result.get('iterations')}")
            
            if result.get('function_calls_made'):
                logger.info(f"  Function Calls: {len(result.get('function_calls_made'))}")
            
            logger.info("")
            logger.info("💬 RESPONSE:")
            response = result.get('response', '')
            # Print first 500 chars of response
            if len(response) > 500:
                logger.info(f"  {response[:500]}...")
                logger.info(f"  ... (truncated, total length: {len(response)} chars)")
            else:
                logger.info(f"  {response}")
            
            if not result.get('success'):
                error = result.get('error', 'Unknown error')
                logger.error(f"  ❌ Error: {error}")
            else:
                logger.info("  ✅ Test passed!")
            
        except Exception as e:
            logger.error(f"❌ Test failed with exception: {str(e)}", exc_info=True)
    
    # Print summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("📈 TEST SUMMARY")
    logger.info("=" * 80)
    
    stats = agent_manager.get_stats()
    logger.info(f"Total requests processed: {stats['total_requests']}")
    logger.info(f"Active sessions: {stats['active_sessions']}")
    logger.info(f"")
    logger.info(f"Agents initialized:")
    for agent_type, agent_name in stats['agents'].items():
        status = "✅" if agent_name else "❌"
        logger.info(f"  {status} {agent_type}: {agent_name or 'Not initialized'}")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("🎉 TESTING COMPLETE!")
    logger.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_function_calling())

