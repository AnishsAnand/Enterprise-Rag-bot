"""
ADK Parallel Executor
=====================
Provides parallel async execution that is fully compatible with ADK's async
model (asyncio) while leaving all existing LangChain agent code untouched.

Where the current code is sequential TODAY            → runs in parallel HERE
─────────────────────────────────────────────────────────────────────────────
list_firewalls()     for ep_id in endpoint_ids: …    → scatter/gather
list_managed_services for ep_id in endpoint_ids: …   → scatter/gather
ValidationAgent      param_check, perm_check, ep_lookup (sequential) → parallel
Intent + RAG fetch   intent.execute() then rag query  → both fire at t=0

No LangChain agent files are modified.  The executor calls their existing
.execute() / API methods inside asyncio.gather() wrappers.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ADKParallelExecutor:
    """
    Scatter-gather task runner.

    Design notes
    ────────────
    • ADK's ParallelAgent model is mirrored with asyncio.gather() so this
      works identically whether google-adk is installed or not.
    • A semaphore caps concurrent tasks to avoid overwhelming the Postgres /
      LLM endpoints.
    • All exceptions are captured per-task (return_exceptions=True) so one
      failing endpoint doesn't abort the rest.
    """

    def __init__(self, max_concurrency: int = 10) -> None:
        self._sem = asyncio.Semaphore(max_concurrency)
        self._max = max_concurrency
        logger.info("✅ ADKParallelExecutor ready (max_concurrency=%d)", max_concurrency)

    # ------------------------------------------------------------------
    # Core primitive
    # ------------------------------------------------------------------

    async def run_parallel(
        self,
        tasks: List[Tuple[str, Coroutine]],
        timeout: float = 30.0,
    ) -> Dict[str, Any]:
        """
        Execute (label, coroutine) pairs concurrently.

        Returns {label: result_or_exception_dict}.
        Never raises – exceptions are returned as {"error": ..., "success": False}.

        Args:
            tasks:   List of (name, awaitable) pairs.
            timeout: Wall-clock timeout for the entire gather.
        """
        if not tasks:
            return {}

        labels = [t[0] for t in tasks]
        guarded = [self._guarded(label, coro) for label, coro in tasks]

        t0 = time.perf_counter()
        try:
            raw = await asyncio.wait_for(
                asyncio.gather(*guarded, return_exceptions=True),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "⏱️ Parallel timeout after %.1fs | tasks=%s", timeout, labels
            )
            return {lbl: {"success": False, "error": "timeout"} for lbl in labels}

        elapsed = time.perf_counter() - t0
        results: Dict[str, Any] = {}
        for label, value in zip(labels, raw):
            if isinstance(value, Exception):
                logger.error("Parallel task '%s' exception: %s", label, value)
                results[label] = {"success": False, "error": str(value)}
            else:
                results[label] = value

        logger.info(
            "⚡ Parallel done in %.2fs | tasks=%s", elapsed, labels
        )
        return results

    # ------------------------------------------------------------------
    # High-level helpers used by ADKHybridManager and ADKApiPatches
    # ------------------------------------------------------------------

    async def scatter_endpoint_calls(
        self,
        call_fn: Callable[..., Coroutine],
        endpoint_ids: List[int],
        shared_kwargs: Dict[str, Any],
        timeout: float = 45.0,
    ) -> Dict[int, Any]:
        """
        Call `call_fn(endpoint_ids=[ep_id], **shared_kwargs)` for every
        endpoint_id in parallel and merge results.

        Replaces sequential for-loops in:
          - api_executor_service.list_firewalls()
          - api_executor_service.list_managed_services()
          - any other multi-endpoint pattern

        Returns {endpoint_id: result_dict}
        """
        async def _one(ep_id: int) -> Dict[str, Any]:
            return await call_fn(endpoint_ids=[ep_id], **shared_kwargs)

        tasks = [(str(ep_id), _one(ep_id)) for ep_id in endpoint_ids]
        raw = await self.run_parallel(tasks, timeout=timeout)
        return {int(ep_id): result for ep_id, result in raw.items()}

    async def parallel_validation_checks(
        self,
        checks: Dict[str, Coroutine],
        timeout: float = 15.0,
    ) -> Dict[str, Any]:
        """
        Run independent validation coroutines concurrently.

        Example – call from ValidationAgent wrapper:
            results = await parallel_executor.parallel_validation_checks({
                "permissions":  check_permissions_coro,
                "params":       validate_params_coro,
                "endpoint":     resolve_endpoint_coro,
            })

        Returns {check_name: result}
        """
        return await self.run_parallel(list(checks.items()), timeout=timeout)

    async def parallel_intent_and_rag(
        self,
        intent_agent: Any,
        user_input: str,
        context: Dict[str, Any],
        postgres_service: Any,
        n_rag_results: int = 5,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Fire intent detection and RAG API-spec lookup at the same time.

        Currently IntentAgent fetches RAG specs *inside* its own execute(),
        making it sequential:  intent LLM call → await RAG → return.

        Here both start at t=0.  The result is identical but ~30–50 % faster
        because the Postgres vector search overlaps with the LLM call.

        The RAG results are returned separately so the caller can inject them
        into the orchestration context, avoiding a second fetch downstream.

        Returns:
            (intent_result_dict, list_of_rag_spec_dicts)
        """
        async def _rag_fetch() -> List[Dict[str, Any]]:
            try:
                if not getattr(postgres_service, "pool", None):
                    await postgres_service.initialize()
                if postgres_service.pool:
                    return await postgres_service.search_api_specs(
                        f"{user_input} API", n_results=n_rag_results
                    ) or []
            except Exception as exc:
                logger.warning("Parallel RAG fetch failed: %s", exc)
            return []

        async def _intent_detect() -> Dict[str, Any]:
            # Pass a flag so IntentAgent can optionally skip its own RAG fetch
            ctx = {**context, "_parallel_rag_in_flight": True}
            return await intent_agent.execute(user_input, ctx)

        results = await self.run_parallel(
            [("intent", _intent_detect()), ("rag", _rag_fetch())],
            timeout=25.0,
        )

        intent_result: Dict[str, Any] = results.get("intent") or {}
        rag_specs: List[Dict[str, Any]] = results.get("rag") or []

        # Normalise – run_parallel wraps exceptions as dicts
        if not isinstance(intent_result, dict) or not intent_result.get("agent_name"):
            logger.warning("Intent parallel task returned unexpected: %s", type(intent_result))
            intent_result = {
                "success": False,
                "intent_detected": False,
                "intent_data": {},
                "output": "",
            }

        if not isinstance(rag_specs, list):
            rag_specs = []

        logger.info(
            "⚡ parallel_intent_and_rag: detected=%s, rag_chunks=%d",
            intent_result.get("intent_detected"),
            len(rag_specs),
        )
        return intent_result, rag_specs

    async def gather_results_from_endpoints(
        self,
        raw_endpoint_results: Dict[int, Any],
    ) -> Tuple[List[Any], Dict[int, str]]:
        """
        Merge per-endpoint parallel results into (all_items, errors).

        Works for clusters, services, firewalls – anything where each endpoint
        returns {"success": bool, "data": list, "error"?: str}.

        Returns:
            all_items – merged flat list of items across all endpoints
            errors    – {endpoint_id: error_message} for failed endpoints
        """
        all_items: List[Any] = []
        errors: Dict[int, str] = {}

        for ep_id, result in raw_endpoint_results.items():
            if isinstance(result, dict) and result.get("success"):
                data = result.get("data", [])
                if isinstance(data, list):
                    all_items.extend(data)
                elif data:
                    all_items.append(data)
            else:
                err = (
                    result.get("error", "unknown error")
                    if isinstance(result, dict)
                    else str(result)
                )
                errors[ep_id] = err
                logger.warning("Endpoint %d failed: %s", ep_id, err)

        return all_items, errors

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _guarded(self, label: str, coro: Coroutine) -> Any:
        """Rate-limit and isolate a single task."""
        async with self._sem:
            try:
                return await coro
            except Exception as exc:
                logger.error("Task '%s' raised: %s", label, exc, exc_info=True)
                raise  # let asyncio.gather capture it as an exception


# Singleton
adk_parallel_executor = ADKParallelExecutor(max_concurrency=10)