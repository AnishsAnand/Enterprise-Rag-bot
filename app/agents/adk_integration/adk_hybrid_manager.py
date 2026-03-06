"""
ADK Hybrid Manager
==================
Drop-in replacement for AgentManager.

Extends (not replaces) AgentManager with:
  1. ADK session state management          via ADKSessionManager
  2. Parallel intent + RAG fetch           via ADKParallelExecutor
  3. Enriched context injection            into OrchestratorAgent
  4. ADK session state updated post-turn   for future turns

What is NOT changed
───────────────────
  - All LangChain agents (Intent, Validation, Execution, RAG, Orchestrator)
  - api_executor_service methods (unless apply_adk_patches() is called)
  - Any route / widget code that calls process_request()
  - Authentication / token management

Usage (app/main.py or startup)
───────────────────────────────
    # Replace this:
    from app.agents.agent_manager import get_agent_manager
    manager = get_agent_manager()

    # With this:
    from app.agents.adk_integration.adk_hybrid_manager import get_adk_agent_manager
    manager = get_adk_agent_manager()

    # Optionally enable parallel API calls:
    from app.agents.adk_integration.adk_api_patches import apply_adk_patches
    apply_adk_patches()
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.agents.agent_manager import AgentManager
from app.agents.adk_integration.adk_session_manager import adk_session_manager
from app.agents.adk_integration.adk_parallel_executor import adk_parallel_executor

logger = logging.getLogger(__name__)


class ADKHybridManager(AgentManager):
    """
    AgentManager with ADK session state and parallel execution.

    Inherits all agent initialisation, stats, cleanup, and reset helpers.
    Overrides only the methods that benefit from ADK additions.
    """

    def __init__(self) -> None:
        super().__init__()
        self._adk_enabled = True
        logger.info("✅ ADKHybridManager ready (extends AgentManager)")

    # ------------------------------------------------------------------
    # OVERRIDE: process_request
    # ------------------------------------------------------------------

    async def process_request(
        self,
        user_input: str,
        session_id: str,
        user_id: str,
        user_roles: List[str] = None,
        auth_token: str = None,
        user_type: str = None,
    ) -> Dict[str, Any]:
        """
        Process a user request with ADK session state + parallel RAG/intent.

        Flow
        ────
        1. Get / create ADK session  (includes ConversationState bootstrap)
        2. Parallel: intent detection + RAG API-spec fetch
        3. Inject pre-fetched context into orchestration call
        4. Delegate to OrchestratorAgent (unchanged)
        5. Update ADK session state with turn outcome
        """
        if not self.initialized:
            self.initialize()

        self.total_requests += 1
        start_time = datetime.utcnow()

        logger.info(
            "📥 [ADK] Request #%d | session=%s | user=%s | type=%s",
            self.total_requests, session_id, user_id, user_type,
        )

        try:
            # ---- 1. ADK session ----------------------------------------
            adk_session = await adk_session_manager.get_or_create_session(
                session_id=session_id,
                user_id=user_id,
                initial_state={
                    "user_type": user_type or "CUS",
                    "user_roles": user_roles or [],
                    "auth_token_present": bool(auth_token),
                    "session_start": datetime.utcnow().isoformat(),
                },
            )
            prior_state = dict(adk_session.state)

            # ---- 2. Parallel intent + RAG fetch -------------------------
            from app.services.postgres_service import postgres_service

            intent_result, pre_fetched_rag = (
                await adk_parallel_executor.parallel_intent_and_rag(
                    intent_agent=self.intent_agent,
                    user_input=user_input,
                    context={
                        "session_id": session_id,
                        "user_id": user_id,
                        "user_type": user_type,
                        # Signal to IntentAgent that it can skip its own RAG
                        # fetch because we already have results incoming.
                        # IntentAgent checks this flag if it supports it;
                        # otherwise it just runs normally – harmless duplicate.
                        "_rag_pre_fetched": True,
                    },
                    postgres_service=postgres_service,
                )
            )

            # ---- 3. Build enriched context for orchestrator -------------
            #
            # OrchestratorAgent.orchestrate() signature is unchanged.
            # We attach extras as keyword args that the orchestrator can
            # consume if it's been updated, or safely ignore if not.
            #
            orchestrate_kwargs: Dict[str, Any] = dict(
                user_input=user_input,
                session_id=session_id,
                user_id=user_id,
                user_roles=user_roles or [],
                auth_token=auth_token,
                user_type=user_type,
            )

            # If the orchestrator accepts pre-computed data, pass it.
            # This avoids a second intent LLM call inside the orchestrator.
            # Orchestrators that don't accept these kwargs will simply ignore them.
            try:
                import inspect
                sig = inspect.signature(self.orchestrator.orchestrate)
                if "_pre_computed" in sig.parameters:
                    orchestrate_kwargs["_pre_computed"] = {
                        "intent_result": intent_result,
                        "rag_specs": pre_fetched_rag,
                        "adk_session_state": prior_state,
                    }
            except Exception:
                pass  # orchestrator signature inspection failed – carry on

            # ---- 4. Delegate to existing OrchestratorAgent --------------
            result = await self.orchestrator.orchestrate(**orchestrate_kwargs)

            # ---- 5. Update ADK session state ----------------------------
            intent_data = intent_result.get("intent_data", {})
            await adk_session_manager.update_session_state(
                session_id=session_id,
                user_id=user_id,
                updates={
                    "last_query": user_input[:500],
                    "last_intent": intent_data.get("resource_type"),
                    "last_operation": intent_data.get("operation"),
                    "last_success": result.get("success", False),
                    "last_updated": datetime.utcnow().isoformat(),
                    "turn_count": prior_state.get("turn_count", 0) + 1,
                },
            )

            # ---- Metadata -----------------------------------------------
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            result["metadata"] = {
                "request_number": self.total_requests,
                "duration_seconds": duration,
                "timestamp": end_time.isoformat(),
                "session_id": session_id,
                "user_id": user_id,
                "adk_enabled": True,
                "parallel_rag_chunks": len(pre_fetched_rag),
            }

            logger.info(
                "✅ [ADK] Request #%d done in %.2fs | success=%s | "
                "rag_chunks=%d | turns=%d",
                self.total_requests,
                duration,
                result.get("success"),
                len(pre_fetched_rag),
                prior_state.get("turn_count", 0) + 1,
            )
            return result

        except Exception as exc:
            logger.exception("❌ [ADK] Request processing failed: %s", exc)
            return {
                "success": False,
                "error": str(exc),
                "response": f"I encountered an error processing your request: {exc}",
                "metadata": {
                    "request_number": self.total_requests,
                    "timestamp": datetime.utcnow().isoformat(),
                    "session_id": session_id,
                    "user_id": user_id,
                    "adk_enabled": True,
                },
            }

    # ------------------------------------------------------------------
    # OVERRIDE: get_conversation_status
    # ------------------------------------------------------------------

    async def get_conversation_status(self, session_id: str) -> Dict[str, Any]:
        """Read status from ADK session first, fall back to ConversationStateManager."""
        try:
            state = await adk_session_manager.get_state(
                session_id=session_id, user_id="lookup"
            )
            if state:
                return {"found": True, "state": state, "source": "adk_session"}
        except Exception:
            pass
        return await super().get_conversation_status(session_id)

    # ------------------------------------------------------------------
    # OVERRIDE: reset_conversation
    # ------------------------------------------------------------------

    async def reset_conversation(
        self, session_id: str, user_id: str = "unknown"
    ) -> Dict[str, Any]:
        """Clear both ADK session and ConversationState."""
        await adk_session_manager.delete_session(session_id, user_id)
        return await super().reset_conversation(session_id)

    # ------------------------------------------------------------------
    # OVERRIDE: get_stats – add ADK layer info
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        stats = super().get_stats()
        stats["adk"] = {
            "enabled": self._adk_enabled,
            "active_adk_sessions": len(adk_session_manager.list_active_sessions()),
            "parallel_max_concurrency": adk_parallel_executor._max,
        }
        return stats


# ---------------------------------------------------------------------------
# Singleton factory – mirrors get_agent_manager() API exactly
# ---------------------------------------------------------------------------

_adk_manager: Optional[ADKHybridManager] = None


def get_adk_agent_manager(
    vector_service=None, ai_service=None
) -> ADKHybridManager:
    """
    Return (or create) the global ADKHybridManager singleton.

    Signature matches get_agent_manager() for a fully drop-in swap.
    """
    global _adk_manager
    if _adk_manager is None:
        _adk_manager = ADKHybridManager()
        _adk_manager.initialize()
        logger.info("🚀 ADKHybridManager singleton initialised")
    return _adk_manager