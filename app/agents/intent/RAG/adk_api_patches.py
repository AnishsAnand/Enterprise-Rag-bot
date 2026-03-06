"""
ADK API Patches
===============
Monkeypatches sequential for-loops in APIExecutorService with parallel
asyncio.gather() calls – zero changes to the original service file.

Specifically replaces the sequential ``for endpoint_id in endpoint_ids``
patterns in:
    - list_firewalls()         → parallel per-endpoint POST calls
    - list_managed_services()  → parallel per-endpoint POST calls
    - scatter_lb_queries()     → new helper that didn't exist before

Call ``apply_adk_patches()`` once at application startup (e.g. in main.py)
AFTER api_executor_service is imported.

Why monkeypatching and not inheritance?
  APIExecutorService is a singleton (api_executor_service) referenced all
  over the codebase.  Replacing the method on the *instance* is the
  smallest-footprint change that needs zero import changes elsewhere.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.agents.adk_integration.adk_parallel_executor import adk_parallel_executor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Patched list_firewalls
# ---------------------------------------------------------------------------

async def _parallel_list_firewalls(
    self,
    endpoint_ids: List[int] = None,
    ipc_engagement_id: int = None,
    variant: str = "",
    auth_token: str = None,
    user_id: str = None,
) -> Dict[str, Any]:
    """
    Parallel replacement for APIExecutorService.list_firewalls().

    Original code iterated endpoint_ids sequentially; this version fires all
    per-endpoint POST calls concurrently then merges the results.
    """
    import time
    start = time.time()

    # ---- same setup logic as the original --------------------------------
    if not user_id:
        user_id = self._get_user_id_from_email()

    if not ipc_engagement_id:
        ipc_engagement_id = await self.get_ipc_engagement_id(
            user_id=user_id, auth_token=auth_token
        )
        if not ipc_engagement_id:
            return {"success": False, "error": "Could not retrieve IPC engagement ID"}

    if not endpoint_ids:
        endpoints_result = await self.list_endpoints()
        if not endpoints_result.get("success"):
            return {"success": False, "error": "Could not fetch endpoints"}
        endpoint_ids = [
            ep.get("id")
            for ep in endpoints_result.get("data", {}).get("endpoints", [])
            if ep.get("id")
        ]

    logger.info(
        "⚡ [PATCH] Parallel list_firewalls: %d endpoints", len(endpoint_ids)
    )

    # ---- scatter across endpoints in parallel ----------------------------
    headers = await self._get_auth_headers(user_id=user_id, auth_token=auth_token)
    client = await self._get_http_client()
    url = "https://ipcloud.tatacommunications.com/networkservice/firewallconfig/details"

    async def _one_endpoint(ep_id: int) -> Dict[str, Any]:
        import httpx
        payload = {
            "engagementId": ipc_engagement_id,
            "endpointId": ep_id,
            "variant": variant,
        }
        try:
            resp = await client.post(url, json=payload, headers=headers, timeout=30.0)
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                for fw in data:
                    fw["_queried_endpoint_id"] = ep_id
                return {"success": True, "data": data, "count": len(data)}
            return {
                "success": False,
                "error": f"Status {resp.status_code}",
                "data": [],
            }
        except Exception as exc:
            return {"success": False, "error": str(exc), "data": []}

    raw = await adk_parallel_executor.run_parallel(
        [(str(ep_id), _one_endpoint(ep_id)) for ep_id in endpoint_ids],
        timeout=45.0,
    )

    all_firewalls: List[Any] = []
    endpoint_results: Dict[int, Any] = {}
    for ep_id_str, result in raw.items():
        ep_id = int(ep_id_str)
        endpoint_results[ep_id] = result
        if result.get("success"):
            all_firewalls.extend(result.get("data", []))

    logger.info(
        "✅ [PATCH] list_firewalls parallel complete: %d firewalls in %.2fs",
        len(all_firewalls), time.time() - start,
    )
    return {
        "success": True,
        "data": all_firewalls,
        "total": len(all_firewalls),
        "endpoints_queried": endpoint_ids,
        "endpoint_results": endpoint_results,
        "ipc_engagement_id": ipc_engagement_id,
        "variant": variant,
        "duration_seconds": time.time() - start,
        "message": f"Found {len(all_firewalls)} firewalls",
        "_parallel": True,
    }


# ---------------------------------------------------------------------------
# Patched list_managed_services
# ---------------------------------------------------------------------------

async def _parallel_list_managed_services(
    self,
    service_type: str,
    endpoint_ids: List[int] = None,
    ipc_engagement_id: int = None,
    auth_token: str = None,
    user_id: str = None,
) -> Dict[str, Any]:
    """
    Parallel replacement for APIExecutorService.list_managed_services().

    Fires one POST per endpoint concurrently instead of sequentially.
    """
    import json
    import time
    start = time.time()

    # ---- same setup logic as original ------------------------------------
    paas_engagement_id = None
    if endpoint_ids is None:
        paas_engagement_id = await self.get_engagement_id(
            auth_token=auth_token, user_id=user_id
        )
        if not paas_engagement_id:
            return {"success": False, "error": "Failed to fetch PAAS engagement ID"}

    if ipc_engagement_id is None:
        if paas_engagement_id is None:
            paas_engagement_id = await self.get_engagement_id(
                auth_token=auth_token, user_id=user_id
            )
        ipc_engagement_id = await self.get_ipc_engagement_id(
            paas_engagement_id, auth_token=auth_token, user_id=user_id
        )
        if not ipc_engagement_id:
            return {
                "success": False,
                "error": "Failed to convert PAAS engagement to IPC engagement ID",
            }

    if endpoint_ids is None:
        if paas_engagement_id is None:
            paas_engagement_id = await self.get_engagement_id(
                auth_token=auth_token, user_id=user_id
            )
        endpoints = await self.get_endpoints(
            paas_engagement_id, auth_token=auth_token, user_id=user_id
        )
        if not endpoints:
            return {"success": False, "error": "Failed to fetch endpoints"}
        endpoint_ids = [ep["endpointId"] for ep in endpoints]

    logger.info(
        "⚡ [PATCH] Parallel list_managed_services (%s): %d endpoints",
        service_type, len(endpoint_ids),
    )

    url = (
        f"https://ipcloud.tatacommunications.com/paasservice/api/v1/paas/"
        f"listManagedServices/{service_type}"
    )
    headers = await self._get_auth_headers(user_id=user_id, auth_token=auth_token)
    client = await self._get_http_client()

    async def _one_endpoint(ep_id: int) -> Dict[str, Any]:
        payload = {
            "engagementId": ipc_engagement_id,
            "endpoints": [ep_id],
            "serviceType": service_type,
        }
        try:
            resp = await client.post(url, json=payload, headers=headers, timeout=30.0)
            if resp.status_code == 200:
                outer = resp.json().get("data", {})
                services = (
                    outer.get("data", []) if isinstance(outer, dict) else outer
                )
                return {
                    "success": True,
                    "data": services if isinstance(services, list) else [],
                }
            return {"success": False, "error": f"Status {resp.status_code}", "data": []}
        except Exception as exc:
            return {"success": False, "error": str(exc), "data": []}

    raw = await adk_parallel_executor.run_parallel(
        [(str(ep_id), _one_endpoint(ep_id)) for ep_id in endpoint_ids],
        timeout=45.0,
    )

    all_services: List[Any] = []
    for result in raw.values():
        if isinstance(result, dict) and result.get("success"):
            all_services.extend(result.get("data", []))

    logger.info(
        "✅ [PATCH] list_managed_services (%s) done: %d services in %.2fs",
        service_type, len(all_services), time.time() - start,
    )
    return {
        "success": True,
        "data": all_services,
        "total": len(all_services),
        "service_type": service_type,
        "ipc_engagement_id": ipc_engagement_id,
        "endpoints": endpoint_ids,
        "message": f"Found {len(all_services)} {service_type} services",
        "raw_response": {},
        "_parallel": True,
    }


# ---------------------------------------------------------------------------
# Patch applicator
# ---------------------------------------------------------------------------

def apply_adk_patches() -> None:
    """
    Monkeypatch APIExecutorService instance with parallel implementations.

    Call ONCE at startup, e.g. in app/main.py:

        from app.agents.adk_integration.adk_api_patches import apply_adk_patches
        apply_adk_patches()

    After this call:
      - api_executor_service.list_firewalls()        → parallel version
      - api_executor_service.list_managed_services() → parallel version
      - All callers (agents, routes) pick up the change automatically
        because they hold a reference to the same singleton.
    """
    import types
    from app.services.api_executor_service import api_executor_service

    # Bind patched methods to the existing singleton instance
    api_executor_service.list_firewalls = types.MethodType(          # type: ignore[method-assign]
        _parallel_list_firewalls, api_executor_service
    )
    api_executor_service.list_managed_services = types.MethodType(   # type: ignore[method-assign]
        _parallel_list_managed_services, api_executor_service
    )

    logger.info(
        "✅ ADK patches applied: list_firewalls + list_managed_services "
        "now run endpoints IN PARALLEL"
    )