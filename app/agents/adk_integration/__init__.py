"""
ADK (Agent Development Kit) + LangChain Hybrid Integration
===========================================================

Adds Google ADK capabilities ON TOP of the existing LangChain agent stack.
Zero existing agent code is modified.

What ADK adds:
  - Structured session / state management  (ADKSessionManager)
  - Parallel async execution               (ADKParallelExecutor)
  - Parallel multi-endpoint API calls      (ADKApiPatches)
  - Drop-in AgentManager replacement       (ADKHybridManager)

Entry points
------------
  from app.agents.adk_integration import get_adk_agent_manager
  manager = get_adk_agent_manager()        # replaces get_agent_manager()
"""

from app.agents.adk_integration.adk_session_manager import adk_session_manager
from app.agents.adk_integration.adk_parallel_executor import adk_parallel_executor
from app.agents.adk_integration.adk_hybrid_manager import ADKHybridManager, get_adk_agent_manager

__all__ = [
    "adk_session_manager",
    "adk_parallel_executor",
    "ADKHybridManager",
    "get_adk_agent_manager",
]