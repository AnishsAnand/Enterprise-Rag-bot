"""
app/agents/action_bot.py

ActionBot: an orchestration layer that:
 - Accepts an action description / payload (GET/POST/etc), target service and params.
 - Uses the project's ai_service (passed in) for intent/slot parsing when needed.
 - Probes services (health, versions) and will return "interactive questions" when
   critical information (e.g., version, required params) is missing.
 - Performs HTTP calls to internal endpoints to automate tasks and records a minimal task history.

Usage: instantiate with an ai_service instance and optionally a base internal registry URL.
"""

from typing import Any, Dict, List, Optional, Tuple
import asyncio
import logging
import json
import time
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger("action_bot")
logger.setLevel(logging.INFO)

@dataclass
class ActionResult:
    status: str  # "ok", "needs_input", "error", "partial"
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

class ActionBot:
    def __init__(
        self,
        ai_service,
        internal_registry_base: Optional[str] = None,
        http_timeout: int = 20,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        """
        ai_service: your existing ai_service object providing e.g. parse_intent or generate_response.
        internal_registry_base: optional base URL for service registry (e.g., "http://internal-registry:8000")
        """
        self.ai_service = ai_service
        self.internal_registry_base = internal_registry_base
        self.http_timeout = http_timeout
        self.client = http_client or httpx.AsyncClient(timeout=httpx.Timeout(http_timeout))
        self.task_history: List[Dict[str, Any]] = []

    async def close(self):
        await self.client.aclose()

    async def _probe_service(self, base_url: str) -> Dict[str, Any]:
        """Probe service for /health and /versions endpoints (best-effort)."""
        result = {}
        try:
            r = await self.client.get(f"{base_url.rstrip('/')}/health")
            result["health_status"] = r.status_code
            try:
                result["health_body"] = r.json()
            except Exception:
                result["health_body"] = (r.text or "")[:200]
        except Exception as e:
            result["health_error"] = str(e)

        try:
            r = await self.client.get(f"{base_url.rstrip('/')}/versions")
            result["versions_status"] = r.status_code
            try:
                result["versions_body"] = r.json()
            except Exception:
                result["versions_body"] = (r.text or "")[:400]
        except Exception as e:
            result["versions_error"] = str(e)

        return result

    async def _call_service(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        json_payload: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        stream: bool = False,
    ) -> Tuple[int, Dict[str, Any]]:
        """Perform HTTP call and return (status_code, parsed_body_or_text)."""
        method = method.lower()
        try:
            resp = await self.client.request(
                method, url, headers=headers, params=params, json=json_payload, data=data
            )
            body = None
            try:
                body = resp.json()
            except Exception:
                body = {"text": (resp.text or "")[:2000]}
            return resp.status_code, {"body": body, "headers": dict(resp.headers)}
        except Exception as e:
            logger.exception("HTTP call failed: %s %s", method, url)
            return 0, {"error": str(e)}

    async def _ask_for_missing_version(self, target_service: str) -> ActionResult:
        """Return a needs_input response asking the caller what version to use."""
        question = (
            f"I couldn't determine the version for service '{target_service}'. "
            "Which version should I operate on? (e.g. 'v1.2.3' or 'latest')"
        )
        return ActionResult(status="needs_input", message=question, details={"missing": "version"})

    async def _validate_payload(self, payload: Dict[str, Any]) -> Tuple[bool, Optional[ActionResult]]:
        """
        Validate presence of essential fields. If something is missing, return (False, ActionResult)
        """
        if "target" not in payload:
            return False, ActionResult(status="needs_input", message="Missing 'target' (service name or base url).")
        if "method" not in payload:
            return False, ActionResult(status="needs_input", message="Missing 'method' (GET|POST|PUT|DELETE).")
        return True, None

    async def handle_action_request(self, payload: Dict[str, Any]) -> ActionResult:
        """
        Main entrypoint.
        payload example:
        {
            "target": "user-service",               # or full URL "http://user-service:8000/api"
            "method": "POST",
            "path": "/api/v1/users/restart",
            "version": "v1.2.3",                    # optional
            "headers": {...},
            "params": {...},
            "json": {...},
            "interactive": True                     # prefer interactive responses
        }
        """
        start = time.time()
        ok, err = await self._validate_payload(payload)
        if not ok:
            return err

        target = payload["target"]
        method = payload["method"].upper()
        path = payload.get("path", "")
        version = payload.get("version")
        headers = payload.get("headers")
        params = payload.get("params")
        json_payload = payload.get("json")
        raw_base = None

        # If target looks like a URL, use it directly; otherwise attempt registry discovery
        if target.startswith("http://") or target.startswith("https://"):
            raw_base = target
        else:
            # Try to use internal registry base if provided, else treat target as hostname
            if self.internal_registry_base:
                raw_base = f"{self.internal_registry_base.rstrip('/')}/{target}"
            else:
                # best-effort assumption: target is reachable by schema http
                raw_base = f"http://{target}"

        # If there's no version provided, try to probe the service for versions
        if not version:
            probe = await self._probe_service(raw_base)
            # If probe discovered versions, try to parse common keys
            versions_body = probe.get("versions_body") or {}
            discovered_version = None
            if isinstance(versions_body, dict):
                # try to find keys like 'version' or 'versions' etc.
                for k in ("version", "versions", "app_version"):
                    if k in versions_body:
                        discovered_version = versions_body.get(k)
                        break
            if not discovered_version:
                # ask for version
                logger.info("version not found for %s; asking user", target)
                return await self._ask_for_missing_version(target)
            else:
                version = discovered_version

        # build final URL
        final_url = f"{raw_base.rstrip('/')}/{version.lstrip('/')}{path}" if version else f"{raw_base.rstrip('/')}{path}"
        # Normalize double slashes
        final_url = final_url.replace(":/", "://").replace("//", "/").replace(":/", "://")
        # small fix: ensure http(s):// is preserved
        if raw_base.startswith("http://") or raw_base.startswith("https://"):
            # when normalization removed the double slash after scheme fix it:
            if final_url.startswith("http:/") and not final_url.startswith("http://"):
                final_url = final_url.replace("http:/", "http://", 1)
            if final_url.startswith("https:/") and not final_url.startswith("https://"):
                final_url = final_url.replace("https:/", "https://", 1)

        # If interactive mode, attempt to validate required json fields using ai_service slot extraction if available
        if payload.get("interactive", True) and hasattr(self.ai_service, "parse_required_fields"):
            # ai_service.parse_required_fields should accept (method, path, json_payload) and return a dict e.g. {"missing":["field1"]}
            try:
                missing = await self.ai_service.parse_required_fields(method, path, json_payload)
                if missing and isinstance(missing, dict) and missing.get("missing"):
                    return ActionResult(status="needs_input", message="Missing required fields", details=missing)
            except Exception as e:
                logger.debug("ai_service parse_required_fields failed: %s", e)

        # Attempt the HTTP call
        status_code, body = await self._call_service(method, final_url, headers=headers, params=params, json_payload=json_payload)

        end = time.time()
        record = {
            "timestamp": time.time(),
            "target": target,
            "url": final_url,
            "method": method,
            "status_code": status_code,
            "duration_s": round(end - start, 3),
            "payload_summary": {"has_json": bool(json_payload), "params": bool(params)}
        }
        self.task_history.append(record)

        # Interpret results
        if status_code >= 200 and status_code < 300:
            return ActionResult(status="ok", message="Action executed successfully", details={"status_code": status_code, "result": body})
        elif status_code == 0:
            return ActionResult(status="error", message="Request failed", details=body)
        elif status_code >= 400 and status_code < 500:
            # client error -> ask for more info (if interactive)
            return ActionResult(status="partial", message=f"Client error {status_code}", details=body)
        else:
            return ActionResult(status="error", message=f"Server error {status_code}", details=body)
