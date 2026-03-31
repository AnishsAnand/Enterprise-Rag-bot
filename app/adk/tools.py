"""
ADK Function Tools wrapping existing service logic.

Each tool is a plain async function with typed parameters and a docstring.
ADK auto-wraps them as FunctionTool instances. The ToolContext parameter
is automatically injected by the framework and provides access to session
state (auth_token, engagement_id, user_id, user_type).
"""
import logging
from typing import Optional

from google.adk.tools import ToolContext

logger = logging.getLogger(__name__)


def _build_resource_context(tool_context: ToolContext) -> dict:
    """Build context dict expected by resource agent execute_operation()."""
    return {
        "auth_token": tool_context.state.get("auth_token"),
        "user_id": tool_context.state.get("user_id"),
        "user_type": tool_context.state.get("user_type"),
        "selected_engagement_id": tool_context.state.get("engagement_id"),
        "user_roles": tool_context.state.get("user_roles", ["admin", "developer", "viewer"]),
    }


async def _ensure_engagement(tool_context: ToolContext) -> Optional[int]:
    """Ensure engagement_id is set in state; fetch if missing. Returns engagement_id or None.

    Prefer api_executor's PAAS engagement (updated by set_engagement_id after Python or
    tool selection) over stale tool_context.state — ADK state can lag behind the executor
    after switching engagements, which caused list_* tools to use the wrong engagement.
    """
    from app.services.api_executor_service import api_executor_service

    auth_token = tool_context.state.get("auth_token")
    user_id = tool_context.state.get("user_id")
    user_type = tool_context.state.get("user_type")

    tool_eid = tool_context.state.get("engagement_id")

    resolved = await api_executor_service.get_engagement_id(
        auth_token=auth_token, user_id=user_id, user_type=user_type
    )

    if resolved is not None:
        if tool_eid != resolved:
            logger.info(
                "_ensure_engagement: syncing engagement_id tool_state=%s → executor=%s",
                tool_eid,
                resolved,
            )
        tool_context.state["engagement_id"] = resolved
        return resolved

    # Executor has no PAAS id — do not trust tool_state alone for ENG: it is often a
    # stale row index (e.g. 5) from an old LLM tool call, which breaks list_clusters.
    ut = (user_type or "").upper()
    if ut == "ENG":
        logger.warning(
            "_ensure_engagement: executor returned no PAAS id; ignoring tool_state engagement_id=%s",
            tool_eid,
        )
        return None

    if tool_eid:
        return tool_eid
    return None


# ---------------------------------------------------------------------------
# RAG / Knowledge Base Tools
# ---------------------------------------------------------------------------

async def search_knowledge_base(query: str, tool_context: ToolContext) -> dict:
    """Search the Vayu Maya platform knowledge base for documentation, guides,
    and general information about cloud infrastructure.

    Use this tool when the user asks general questions about how the platform
    works, concepts, how-to guides, or anything that is NOT a live resource
    operation (listing clusters, VMs, etc.).

    Args:
        query: The search query describing what information to find.

    Returns:
        dict with matching documentation snippets including source metadata.
    """
    from app.services.postgres_service import postgres_service

    try:
        results = await postgres_service.search_documents(query=query, n_results=8)
        if results:
            snippets = []
            for r in results:
                meta = r.get("metadata", {}) or {}
                snippets.append({
                    "content": r.get("content", "")[:3000],
                    "title": meta.get("title", ""),
                    "url": meta.get("url", ""),
                    "domain": meta.get("domain", ""),
                    "score": round(r.get("relevance_score", 0), 3),
                    "images": meta.get("images", []),
                })
            return {"status": "success", "results": snippets, "count": len(snippets)}
        return {"status": "no_results", "message": "No relevant documentation found for the query."}
    except Exception as e:
        logger.error(f"search_knowledge_base error: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


# ---------------------------------------------------------------------------
# Engagement Tools
# ---------------------------------------------------------------------------

async def get_engagements(tool_context: ToolContext) -> dict:
    """Get the list of cloud engagements (subscriptions/projects) available
    for the current user.

    Sets engagement_list_shown=True in session state so the Python pre-check
    can intercept the user's next numeric reply and resolve the row number to
    the real engagement_id (e.g. row "5" → engagement_id 1924).

    Returns:
        dict with the list of engagements and the currently active engagement_id (if any).
    """
    from app.services.api_executor_service import api_executor_service

    auth_token = tool_context.state.get("auth_token")
    user_id = tool_context.state.get("user_id")
    current_eid = tool_context.state.get("engagement_id")

    engagements = await api_executor_service.get_engagements_list(
        auth_token=auth_token, user_id=user_id
    )
    if not engagements:
        return {"status": "error", "message": "No engagements found for your account. Please contact support."}

    # Mark that the list has been shown so the Python pre-check can correctly
    # map the user's next number (a row index) to the real engagement_id.
    # Without this, "5" would be passed as engagement_id=5 instead of id=1924.
    tool_context.state["engagement_list_shown"] = True

    items = [{"id": e.get("id"), "name": e.get("engagementName", "")} for e in engagements]
    return {
        "status": "success",
        "current_engagement_id": current_eid,
        "engagements": items,
        "message": (
            f"Active engagement: {current_eid}. "
            if current_eid else
            "No engagement selected yet. Use select_engagement() to set one."
        ),
    }


async def select_engagement(engagement_id: int, tool_context: ToolContext) -> dict:
    """Select a specific engagement by its engagement_id (the real ID from the
    Engagement ID column, e.g. 1924 — NOT the row number from the list).

    Args:
        engagement_id: The real engagement_id to activate (e.g. 1924, 15787).

    Returns:
        dict confirming the selection.
    """
    from app.services.api_executor_service import api_executor_service

    auth_token = tool_context.state.get("auth_token")
    user_id = tool_context.state.get("user_id")

    # Safety check: if engagement_id looks like a small row index (≤ 100),
    # validate it against the actual engagements list.
    # The real engagement IDs are typically 3–5 digit numbers (e.g. 1924, 15787).
    real_id = engagement_id
    if engagement_id <= 100:
        engagements = await api_executor_service.get_engagements_list(
            auth_token=auth_token, user_id=user_id
        )
        if engagements and 1 <= engagement_id <= len(engagements):
            # Treat it as a row index and resolve to the real ID
            selected = engagements[engagement_id - 1]
            real_id = selected.get("id", engagement_id)
            logger.info(
                "select_engagement: row index %d → real engagement_id=%s name='%s'",
                engagement_id, real_id, selected.get("engagementName", ""),
            )
        # If it's small but doesn't match a valid row, still try as-is

    await api_executor_service.set_engagement_id(user_id=user_id, engagement_id=real_id)
    tool_context.state["engagement_id"] = real_id
    tool_context.state["engagement_user_selected"] = True
    # ADK's State object doesn't support .pop() — use None to signal deletion
    tool_context.state["engagement_list_shown"] = None

    return {
        "status": "success",
        "engagement_id": real_id,
        "message": f"Engagement {real_id} is now active.",
    }


# ---------------------------------------------------------------------------
# Kubernetes Cluster Tools
# ---------------------------------------------------------------------------

async def list_k8s_clusters(tool_context: ToolContext) -> dict:
    """List all Kubernetes clusters across all datacenters for the user's
    current engagement.

    Returns:
        dict with cluster data including names, status, node count, etc.
    """
    from app.agents.resource_agents.k8s_cluster_agent import K8sClusterAgent

    eid = await _ensure_engagement(tool_context)
    if not eid:
        return {"status": "error", "message": "No engagement found. Unable to list clusters."}

    ctx = _build_resource_context(tool_context)
    agent = K8sClusterAgent()
    result = await agent.execute_operation("list", {}, ctx)
    return _sanitize_result(result)


async def check_cluster_name(cluster_name: str, tool_context: ToolContext) -> dict:
    """Check if a Kubernetes cluster name is available for use.

    Args:
        cluster_name: The cluster name to check availability for.

    Returns:
        dict with availability status.
    """
    from app.services.api_executor_service import api_executor_service

    if not cluster_name:
        return {"status": "error", "message": "Please provide a cluster name to check."}

    try:
        result = await api_executor_service.check_cluster_name_available(cluster_name)
        if result.get("success"):
            available = result.get("available", False)
            msg = result.get("message", "")
            response = f"**Cluster Name Check:** `{cluster_name}`\n\n"
            if available:
                response += f"The cluster name `{cluster_name}` is **available** and can be used."
            else:
                existing = result.get("existing_cluster", {})
                response += f"The cluster name `{cluster_name}` is **already taken**."
                if existing:
                    response += f"\n- Existing Cluster ID: {existing.get('clusterId', 'N/A')}"
            return {"status": "success", "response": response, "available": available}
        return {"status": "error", "message": result.get("error", "Failed to check cluster name.")}
    except Exception as e:
        logger.error(f"check_cluster_name error: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


async def get_cluster_details(cluster_name: str, tool_context: ToolContext) -> dict:
    """Get detailed information about a specific Kubernetes cluster.

    Args:
        cluster_name: Name of the cluster to get details for.

    Returns:
        dict with detailed cluster information.
    """
    from app.agents.resource_agents.k8s_cluster_agent import K8sClusterAgent

    eid = await _ensure_engagement(tool_context)
    if not eid:
        return {"status": "error", "message": "No engagement found."}

    ctx = _build_resource_context(tool_context)
    ctx["user_query"] = f"details for cluster {cluster_name}"
    agent = K8sClusterAgent()
    result = await agent.execute_operation("read", {"cluster_name": cluster_name}, ctx)
    return _sanitize_result(result)


# ---------------------------------------------------------------------------
# VM Tools
# ---------------------------------------------------------------------------

async def list_vms(tool_context: ToolContext) -> dict:
    """List all virtual machines for the user's current engagement.

    Returns:
        dict with VM data including names, status, specs, etc.
    """
    from app.agents.resource_agents.virtual_machine_agent import VirtualMachineAgent

    eid = await _ensure_engagement(tool_context)
    if not eid:
        return {"status": "error", "message": "No engagement found. Unable to list VMs."}

    ctx = _build_resource_context(tool_context)
    agent = VirtualMachineAgent()
    result = await agent.execute_operation("list", {}, ctx)
    return _sanitize_result(result)


# ---------------------------------------------------------------------------
# Firewall Tools
# ---------------------------------------------------------------------------

async def list_firewalls(tool_context: ToolContext) -> dict:
    """List all firewalls for the user's current engagement.

    Returns:
        dict with firewall data including names, status, rules summary.
    """
    from app.agents.resource_agents.network_agent import NetworkAgent

    eid = await _ensure_engagement(tool_context)
    if not eid:
        return {"status": "error", "message": "No engagement found. Unable to list firewalls."}

    ctx = _build_resource_context(tool_context)
    ctx["resource_type"] = "firewall"
    agent = NetworkAgent()
    result = await agent.execute_operation("list", {}, ctx)
    return _sanitize_result(result)


# ---------------------------------------------------------------------------
# Load Balancer Tools
# ---------------------------------------------------------------------------

async def list_load_balancers(tool_context: ToolContext) -> dict:
    """List all load balancers for the user's current engagement.

    Returns:
        dict with load balancer data.
    """
    from app.agents.resource_agents.load_balancer_agent import LoadBalancerAgent

    eid = await _ensure_engagement(tool_context)
    if not eid:
        return {"status": "error", "message": "No engagement found."}

    ctx = _build_resource_context(tool_context)
    agent = LoadBalancerAgent()
    result = await agent.execute_operation("list", {}, ctx)
    return _sanitize_result(result)


async def get_load_balancer_details(lb_name: str, tool_context: ToolContext) -> dict:
    """Get detailed information about a specific load balancer.

    Args:
        lb_name: Name or identifier of the load balancer.

    Returns:
        dict with detailed load balancer information.
    """
    from app.agents.resource_agents.load_balancer_agent import LoadBalancerAgent

    eid = await _ensure_engagement(tool_context)
    if not eid:
        return {"status": "error", "message": "No engagement found."}

    ctx = _build_resource_context(tool_context)
    agent = LoadBalancerAgent()
    result = await agent.execute_operation("read", {"lb_name": lb_name}, ctx)
    return _sanitize_result(result)


# ---------------------------------------------------------------------------
# Managed Service Tools
# ---------------------------------------------------------------------------

async def list_managed_services(service_type: str, tool_context: ToolContext) -> dict:
    """List managed services of a specific type for the user's engagement.

    Args:
        service_type: Type of managed service. Must be one of:
            kafka, postgres, documentdb, gitlab, jenkins, container_registry.

    Returns:
        dict with managed service instances.
    """
    from app.agents.resource_agents.managed_services_agent import ManagedServicesAgent

    eid = await _ensure_engagement(tool_context)
    if not eid:
        return {"status": "error", "message": "No engagement found."}

    ctx = _build_resource_context(tool_context)
    ctx["resource_type"] = service_type
    agent = ManagedServicesAgent()
    result = await agent.execute_operation("list", {"service_type_hint": service_type}, ctx)
    return _sanitize_result(result)


# ---------------------------------------------------------------------------
# Infrastructure Tools (Zones, Environments, Business Units)
# ---------------------------------------------------------------------------

async def list_zones(tool_context: ToolContext) -> dict:
    """List all available zones/regions.

    Returns:
        dict with zone information.
    """
    from app.services.api_executor_service import api_executor_service

    auth_token = tool_context.state.get("auth_token")
    result = await api_executor_service.execute_operation(
        resource_type="zone", operation="list", params={}, auth_token=auth_token
    )
    return _sanitize_result(result)


async def list_environments(tool_context: ToolContext) -> dict:
    """List all available environments (e.g. production, staging, development).

    Returns:
        dict with environment information.
    """
    from app.services.api_executor_service import api_executor_service

    auth_token = tool_context.state.get("auth_token")
    result = await api_executor_service.execute_operation(
        resource_type="environment", operation="list", params={}, auth_token=auth_token
    )
    return _sanitize_result(result)


async def list_business_units(tool_context: ToolContext) -> dict:
    """List all available business units/departments.

    Returns:
        dict with business unit information.
    """
    from app.services.api_executor_service import api_executor_service

    auth_token = tool_context.state.get("auth_token")
    result = await api_executor_service.execute_operation(
        resource_type="business_unit", operation="list", params={}, auth_token=auth_token
    )
    return _sanitize_result(result)


# ---------------------------------------------------------------------------
# Reports Tool
# ---------------------------------------------------------------------------

async def get_cluster_report(report_type: str, tool_context: ToolContext) -> dict:
    """Generate a report about Kubernetes clusters.

    Args:
        report_type: Type of report. One of:
            common_cluster, cluster_inventory, cluster_compute, storage_inventory.

    Returns:
        dict with report data.
    """
    from app.agents.resource_agents.reports_agent import ReportsAgent

    eid = await _ensure_engagement(tool_context)
    if not eid:
        return {"status": "error", "message": "No engagement found."}

    ctx = _build_resource_context(tool_context)
    agent = ReportsAgent()
    result = await agent.execute_operation("read", {"report_type": report_type}, ctx)
    return _sanitize_result(result)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize_result(result: dict) -> dict:
    """Ensure the result dict is JSON-serializable and not too large for the LLM."""
    if not isinstance(result, dict):
        return {"status": "error", "message": "Unexpected result format."}

    sanitized = {
        "status": "success" if result.get("success") else "error",
    }

    if result.get("response"):
        sanitized["response"] = str(result["response"])[:8000]

    if result.get("error"):
        sanitized["error_message"] = str(result["error"])[:1000]

    if result.get("data"):
        data = result["data"]
        if isinstance(data, dict):
            if "data" in data and isinstance(data["data"], list):
                items = data["data"]
                sanitized["total_count"] = len(items)
                sanitized["items"] = _truncate_items(items, max_items=30)
            else:
                sanitized["data"] = _truncate_dict(data)
        elif isinstance(data, list):
            sanitized["total_count"] = len(data)
            sanitized["items"] = _truncate_items(data, max_items=30)
        else:
            sanitized["data"] = str(data)[:4000]

    return sanitized


def _truncate_items(items: list, max_items: int = 30) -> list:
    """Keep only key fields from list items to fit LLM context."""
    truncated = []
    for item in items[:max_items]:
        if isinstance(item, dict):
            summary = {}
            for key in list(item.keys())[:15]:
                val = item[key]
                if isinstance(val, str) and len(val) > 200:
                    val = val[:200] + "..."
                elif isinstance(val, (list, dict)):
                    val = str(val)[:200]
                summary[key] = val
            truncated.append(summary)
        else:
            truncated.append(str(item)[:200])
    if len(items) > max_items:
        truncated.append(f"... and {len(items) - max_items} more items")
    return truncated


def _truncate_dict(d: dict, max_keys: int = 20) -> dict:
    """Truncate a dict to fit LLM context."""
    result = {}
    for i, (key, val) in enumerate(d.items()):
        if i >= max_keys:
            result["_truncated"] = f"{len(d) - max_keys} more keys"
            break
        if isinstance(val, str) and len(val) > 500:
            val = val[:500] + "..."
        elif isinstance(val, list) and len(val) > 10:
            val = val[:10] + [f"... {len(val) - 10} more"]
        result[key] = val
    return result
