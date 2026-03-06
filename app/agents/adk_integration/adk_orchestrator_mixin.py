"""
ADK Orchestrator Mixin
======================
Thin mixin for OrchestratorAgent that wires in ADK pre-computed context
(parallel intent + RAG results from ADKHybridManager) without touching any
existing orchestrator logic.

Three changes to orchestrator_agent.py
───────────────────────────────────────
1. Import this mixin  (1 line)
2. Add it to the class declaration  (1 word)
3. Add ``_pre_computed=None`` to orchestrate() and call _apply_adk_context()
   just before _execute_routing()  (3 lines)

That is the complete integration.  Every existing code path is unchanged.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ADKOrchestratorMixin:
    """
    Mixin for OrchestratorAgent.

    Must appear BEFORE BaseAgent in the class MRO so Python's method
    resolution finds these helpers first:

        class OrchestratorAgent(ADKOrchestratorMixin, BaseAgent):
            ...
    """

    # ------------------------------------------------------------------
    # Public helpers – call these from inside orchestrate()
    # ------------------------------------------------------------------

    def _extract_pre_computed(
        self,
        pre_computed: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Safely unpack the ``_pre_computed`` dict injected by ADKHybridManager.

        Returns
        ───────
        intent_result : full intent-agent result dict, or None
        rag_specs     : list of RAG spec dicts (may be empty list)
        """
        if not pre_computed or not isinstance(pre_computed, dict):
            return None, []

        intent_result = pre_computed.get("intent_result")
        rag_specs = pre_computed.get("rag_specs") or []

        if intent_result:
            resource = intent_result.get("intent_data", {}).get("resource_type")
            operation = intent_result.get("intent_data", {}).get("operation")
            logger.info(
                "🔌 ADK pre-computed: %s.%s | rag_chunks=%d",
                resource, operation, len(rag_specs),
            )

        return intent_result, rag_specs

    def _should_skip_intent(
        self,
        pre_computed_intent: Optional[Dict[str, Any]],
        confidence_threshold: float = 0.7,
    ) -> bool:
        """
        Return True when the pre-computed intent is reliable enough to use
        directly, avoiding a second LLM round-trip.

        Low-confidence results (< threshold) fall through to the normal
        IntentAgent.execute() path – unchanged behaviour.
        """
        if not pre_computed_intent:
            return False
        if not pre_computed_intent.get("intent_detected"):
            return False
        confidence = pre_computed_intent.get("intent_data", {}).get("confidence", 0.0)
        skip = float(confidence) >= confidence_threshold
        if skip:
            logger.info(
                "⚡ Skipping IntentAgent LLM call – using ADK pre-computed "
                "intent (confidence=%.2f ≥ threshold=%.2f)",
                confidence, confidence_threshold,
            )
        return skip

    def _apply_adk_context(
        self,
        pre_computed: Optional[Dict[str, Any]],
        routing_decision: Dict[str, Any],
    ) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Main integration hook – call this once just before _execute_routing().

        Returns the same (intent_result, rag_specs) tuple and logs one
        coherent message so it is easy to spot in logs.

        Usage in orchestrate() just before _execute_routing():

            adk_intent, adk_rag = self._apply_adk_context(
                _pre_computed, routing_decision
            )

        Then pass ``adk_intent`` into _execute_routing so it can optionally
        skip the IntentAgent LLM call when route == "intent".
        """
        intent_result, rag_specs = self._extract_pre_computed(pre_computed)

        if routing_decision.get("route") == "intent" and intent_result:
            if self._should_skip_intent(intent_result):
                logger.info(
                    "🚀 ADK fast-path active for route='intent'"
                )
            else:
                logger.debug(
                    "ADK context present but confidence below threshold – "
                    "will call IntentAgent normally"
                )

        return intent_result, rag_specs

    def _build_intent_result(
        self,
        adk_intent: Optional[Dict[str, Any]],
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Decide whether to use the ADK pre-computed intent or call the LLM.

        Returns
        ───────
        used_adk  : True if the pre-computed result should be used
        result    : intent-agent result dict to use downstream, or None
                    (when None, caller must call intent_agent.execute() itself)
        """
        if self._should_skip_intent(adk_intent):
            logger.info("⚡ Using ADK pre-computed intent – no LLM call needed")
            return True, adk_intent
        return False, None