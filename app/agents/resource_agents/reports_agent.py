"""
Reports Agent - Handles report table retrieval and formatting.
"""

from typing import Any, Dict, List, Optional
from datetime import date, timedelta
import logging

from app.agents.resource_agents.base_resource_agent import BaseResourceAgent
from app.services.api_executor_service import api_executor_service

logger = logging.getLogger(__name__)


class ReportsAgent(BaseResourceAgent):
    """Agent for reports and analytics tables."""

    REPORT_DEFINITIONS = {
        "common_cluster": {
            "report_name": "CommonClusterReport",
            "display_name": "Common Cluster Report",
            "data_key": "vms",
            "filter_hint": (
                "You can filter by cluster name (via cluster list stream), "
                "datacenter, or date range (start/end)."
            ),
            "display_columns": [
                ("clusterName", "Cluster Name"),
                ("resourceStatus", "Resource Status"),
                ("location", "Location"),
                ("departmentName", "Department Name"),
                ("environment", "Environment"),
                ("zoneLabel", "Zone Label"),
                ("creationDate", "Creation Date"),
                ("activationDate", "Activation Date"),
                ("createdBy", "Created By"),
                ("deactivationDate", "Deactivation Date"),
                ("updatedDate", "Last Updated"),
                ("resourceType", "Resource Type"),
                ("orderId", "Order Id"),
                ("tags", "Tags")
            ]
        },
        "cluster_inventory": {
            "report_name": "clusterReport",
            "display_name": "Cluster Inventory Report",
            "data_key": "clusters",
            "alternate_data_keys": ["clusterList", "items", "data"],
            "filter_hint": (
                "You can filter by cluster name, datacenter, status, Kubernetes version, "
                "or date range (start/end)."
            ),
            "display_columns": [
                ("clusterName", "Cluster Name"),
                ("clusterId", "Cluster Id"),
                ("clusterStatus", "Cluster Status"),
                ("ingressIp", "Ingress IP"),
                ("clusterType", "Cluster Type"),
                ("region", "Region"),
                ("departmentName", "Department"),
                ("environment", "Environment"),
                ("creationDate", "Creation Date"),
                ("activationDate", "Activation Date"),
                ("k8sVersion", "K8s Version"),
                ("tags", "Tags")
            ]
        },
        "cluster_compute": {
            "report_name": "clusterComputeReport",
            "display_name": "Cluster Compute Report",
            "data_key": "clusters",
            "alternate_data_keys": ["clusterList", "items", "data"],
            "filter_hint": (
                "You can filter by cluster name, datacenter, worker node, "
                "or date range (start/end)."
            ),
            "display_columns": [
                ("nodePoolName", "Node Pool Name"),
                ("clusterName", "Cluster Name"),
                ("clusterId", "Cluster Id"),
                ("region", "Region"),
                ("nodeOs", "Node Os"),
                ("nodeStatus", "Node Status"),
                ("replica", "Replica"),
                ("vcpu", "Vcpu"),
                ("vram", "Vram"),
                ("activationDate", "Activation Date"),
                ("tags", "Tags")
            ]
        },
        "storage_inventory": {
            "report_name": "clusterPVCReport",
            "display_name": "Storage Inventory Report",
            "data_key": "vms",
            "alternate_data_keys": ["items", "data"],
            "filter_hint": (
                "You can filter by cluster name, datacenter, PVC type, "
                "or date range (start/end)."
            ),
            "display_columns": [
                ("pvcName", "Pvc Name"),
                ("clusterName", "Cluster Name"),
                ("clusterId", "Cluster Id"),
                ("pvcStatus", "Pvc Status"),
                ("pvcType", "Pvc Type"),
                ("storageAllocated", "Storage Allocated"),
                ("activationDate", "Activation Date"),
                ("tags", "Tags")
            ]
        }
    }

    def __init__(self):
        super().__init__(
            agent_name="ReportsAgent",
            agent_description=(
                "Specialized agent for reports and analytics tables. "
                "Uses LLM formatting to present report data clearly."
            ),
            resource_type="reports",
            temperature=0.2
        )

    def get_supported_operations(self) -> List[str]:
        return ["list"]

    async def execute_operation(
        self,
        operation: str,
        params: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            logger.info(f"📊 ReportsAgent executing: {operation}")

            if operation == "list":
                return await self._list_reports(params, context)

            return {
                "success": False,
                "error": f"Unsupported operation: {operation}",
                "response": f"I don't support the '{operation}' operation for reports yet."
            }
        except Exception as e:
            logger.error(f"❌ ReportsAgent error: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "response": f"An error occurred while fetching reports: {str(e)}"
            }

    async def _list_reports(
        self,
        params: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """List report data for the requested report type."""
        user_query = context.get("user_query", "")
        if self._wants_report_catalog(user_query, params):
            return self._get_report_catalog_response()

        report_key = self._detect_report_type(params, user_query)
        report_def = self.REPORT_DEFINITIONS.get(report_key)
        if not report_def:
            return {
                "success": False,
                "error": "Unknown report type",
                "response": "I couldn't determine which report you want. Please specify the report name."
            }

        page = self._safe_int(params.get("page", 0), default=0, minimum=0)
        # IntentAgent is instructed to always use "size" for record count
        # Keep "limit" as fallback for backward compatibility
        size_param = params.get("size") or params.get("limit")
        size = self._safe_int(size_param, default=5, minimum=1) if size_param else 5
        wants_all = self._wants_all(user_query, params)

        filter_request = self._detect_filter_request(user_query, params)
        if filter_request:
            return await self._get_filter_options(filter_request, report_def, context)

        engagement_id = await self.get_engagement_id(user_roles=context.get("user_roles", []))
        if not engagement_id:
            return {
                "success": False,
                "error": "Failed to get engagement ID",
                "response": "Unable to retrieve engagement information for reports."
            }

        api_payload = {
            "report_name": report_def["report_name"],
            "page": page,
            "size": size,
            "engagement_id": engagement_id
        }
        api_payload.update(self._extract_filters(params))

        result = await api_executor_service.execute_operation(
            resource_type="reports",
            operation="list",
            params=api_payload,
            user_roles=context.get("user_roles", [])
        )

        if not result.get("success"):
            return {
                "success": False,
                "error": result.get("error"),
                "response": f"Failed to fetch {report_def['display_name']}: {result.get('error')}"
            }

        raw = result.get("data") or {}
        data_block = raw.get("data") if isinstance(raw, dict) else {}
        data_block = data_block if isinstance(data_block, dict) else {}
        items = self._extract_report_items(data_block, report_def)
        total_count = data_block.get("totalCount")

        # If user asked for all results, re-fetch with totalCount as size
        if wants_all and isinstance(total_count, int) and total_count > size:
            api_payload["page"] = 0
            api_payload["size"] = total_count
            result = await api_executor_service.execute_operation(
                resource_type="reports",
                operation="list",
                params=api_payload,
                user_roles=context.get("user_roles", [])
            )
            if result.get("success"):
                raw = result.get("data") or {}
                data_block = raw.get("data") if isinstance(raw, dict) else {}
                data_block = data_block if isinstance(data_block, dict) else {}
                items = data_block.get(report_def["data_key"], items)
                total_count = data_block.get("totalCount", total_count)

        normalized_items = self._normalize_report_items(items, report_def)
        report_table = self._build_report_table(normalized_items, report_def)
        shown_count = len(normalized_items)
        response_text = f"## 📊 {report_def['display_name']}\n\n"
        if isinstance(total_count, int):
            response_text += f"Showing **{shown_count}** of **{total_count}** records.\n"
        else:
            response_text += f"Showing **{shown_count}** records.\n"
        if report_table:
            response_text += "\n" + report_table
        if isinstance(total_count, int) and shown_count < total_count:
            response_text += "\n\n💡 More records are available. Ask for a larger size or say “show all”."
        if report_def.get("filter_hint"):
            response_text += f"\n\n💡 {report_def['filter_hint']}"

        return {
            "success": True,
            "data": items,
            "response": response_text,
            "metadata": {
                "report": report_def["report_name"],
                "display_name": report_def["display_name"],
                "count": len(items) if isinstance(items, list) else 0,
                "total_count": total_count,
                "page": page,
                "size": total_count if wants_all and isinstance(total_count, int) else size,
                "resource_type": "reports"
            }
        }

    def _detect_report_type(self, params: Dict[str, Any], user_query: str) -> str:
        report_type = (params.get("report_type") or params.get("report") or "").strip().lower()
        if report_type in self.REPORT_DEFINITIONS:
            return report_type
        if "common" in report_type and "cluster" in report_type:
            return "common_cluster"
        if "cluster" in report_type and "inventory" in report_type:
            return "cluster_inventory"
        if "cluster" in report_type and "report" in report_type:
            return "cluster_inventory"
        if "cluster" in report_type and "compute" in report_type:
            return "cluster_compute"
        if "storage" in report_type and "inventory" in report_type:
            return "storage_inventory"
        if "pvc" in report_type and "report" in report_type:
            return "storage_inventory"

        query_lower = user_query.lower()
        if "common cluster" in query_lower:
            return "common_cluster"
        if "cluster inventory" in query_lower or "cluster report" in query_lower:
            return "cluster_inventory"
        if "cluster compute" in query_lower or "compute report" in query_lower:
            return "cluster_compute"
        if "storage inventory" in query_lower or "pvc report" in query_lower:
            return "storage_inventory"

        return "common_cluster"

    def _safe_int(self, value: Any, default: int = 0, minimum: Optional[int] = None) -> int:
        try:
            number = int(value)
        except (TypeError, ValueError):
            return default

        if minimum is not None and number < minimum:
            return minimum
        return number

    def _extract_filters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        for key in [
            "startDate",
            "endDate",
            "clusterName",
            "datacenter",
            "status",
            "k8sVersion",
            "workerNode",
            "pvcType"
        ]:
            value = params.get(key)
            if value:
                payload[key] = value
        return payload

    def _normalize_report_items(self, items: Any, report_def: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not isinstance(items, list):
            return []
        columns = report_def.get("display_columns", [])
        if not columns:
            return [item for item in items if isinstance(item, dict)]
        normalized = []
        for item in items:
            if not isinstance(item, dict):
                continue
            normalized.append({col_key: item.get(col_key, "") for col_key, _ in columns})
        return normalized

    def _extract_report_items(self, data_block: Dict[str, Any], report_def: Dict[str, Any]) -> List[Any]:
        if not isinstance(data_block, dict):
            return []
        items = data_block.get(report_def["data_key"], [])
        if isinstance(items, list) and items:
            return items
        for alt_key in report_def.get("alternate_data_keys", []):
            candidate = data_block.get(alt_key)
            if isinstance(candidate, list) and candidate:
                return candidate
        for value in data_block.values():
            if isinstance(value, list) and value:
                return value
            if isinstance(value, dict):
                nested = value.get(report_def["data_key"])
                if isinstance(nested, list) and nested:
                    return nested
        return items if isinstance(items, list) else []

    def _build_report_table(self, items: List[Dict[str, Any]], report_def: Dict[str, Any]) -> str:
        if not items:
            return ""
        columns = report_def.get("display_columns", [])
        if not columns:
            return ""

        header_labels = [label for _, label in columns]
        header = "| " + " | ".join(header_labels) + " |\n"
        separator = "| " + " | ".join(["---"] * len(columns)) + " |\n"

        rows = []
        for item in items:
            row_values = []
            for col_key, _ in columns:
                value = item.get(col_key, "")
                if isinstance(value, (dict, list)):
                    value = str(value)
                row_values.append(str(value))
            rows.append("| " + " | ".join(row_values) + " |")

        return header + separator + "\n".join(rows)

    def _detect_filter_request(self, user_query: str, params: Dict[str, Any]) -> Optional[str]:
        query_lower = (user_query or "").lower()
        if not query_lower:
            return None

        wants_filter = any(kw in query_lower for kw in ["filter", "filters", "by", "choose"])
        if not wants_filter:
            return None

        if ("datacenter" in query_lower or "data center" in query_lower or "location" in query_lower) and not params.get("datacenter"):
            return "report_datacenter"
        if ("date" in query_lower or "created date" in query_lower or "date range" in query_lower) and not (
            params.get("startDate") or params.get("endDate")
        ):
            return "report_dates"
        if self._wants_cluster_filter(query_lower) and not params.get("clusterName"):
            return "report_cluster"

        return None

    def _wants_cluster_filter(self, query_lower: str) -> bool:
        if not query_lower:
            return False
        explicit_phrases = [
            "cluster name",
            "cluster names",
            "cluster list",
            "cluster filter",
            "filter cluster",
            "filter by cluster",
            "by cluster",
            "cluster:",
            "cluster="
        ]
        return any(phrase in query_lower for phrase in explicit_phrases)

    async def _get_filter_options(
        self,
        filter_type: str,
        report_def: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        if filter_type == "report_cluster":
            options = await self._get_cluster_options(context)
            response = self._format_options_table(
                title="Available Cluster Names",
                columns=["Cluster Name"],
                rows=[[opt["name"]] for opt in options]
            )
            response += "\n\n💡 Reply with the **cluster name** or **number** to filter."
        elif filter_type == "report_datacenter":
            options = await self._get_datacenter_options(context)
            response = self._format_options_table(
                title="Available Datacenters",
                columns=["Datacenter"],
                rows=[[opt["name"]] for opt in options]
            )
            response += "\n\n💡 Reply with the **datacenter name** or **number** to filter."
        else:
            options = self._get_date_filter_options()
            response = self._format_options_table(
                title="Choose Created Date",
                columns=["Date Filter"],
                rows=[[opt["name"]] for opt in options]
            )
            response += "\n\n💡 Reply with the **option number**. Use 'Custom Date Range' to set dates."

        return {
            "success": True,
            "response": response,
            "set_filter_state": True,
            "filter_options_for_state": options,
            "filter_type_for_state": filter_type,
            "metadata": {
                "awaiting_filter_selection": True,
                "resource_type": "reports",
                "report": report_def.get("report_name")
            }
        }

    async def _get_cluster_options(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        result = await api_executor_service.list_clusters()
        if not result.get("success"):
            return []
        clusters = result.get("data", [])
        names = []
        for item in clusters:
            if isinstance(item, dict):
                name = item.get("clusterName")
                if name and name not in names:
                    names.append(name)
        return [{"id": idx + 1, "name": name} for idx, name in enumerate(names)]

    async def _get_datacenter_options(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        result = await api_executor_service.list_clusters()
        names = []
        if result.get("success"):
            clusters = result.get("data", [])
            for item in clusters:
                if isinstance(item, dict):
                    name = item.get("location") or item.get("datacenter")
                    normalized = self._normalize_datacenter_name(name)
                    if normalized and normalized not in names:
                        names.append(normalized)
        if not names:
            endpoints = await api_executor_service.get_endpoints()
            if not endpoints:
                return []
            for ep in endpoints:
                name = ep.get("endpointDisplayName")
                normalized = self._normalize_datacenter_name(name)
                if normalized and normalized not in names:
                    names.append(normalized)
        return [{"id": idx + 1, "name": name} for idx, name in enumerate(names)]

    def _get_date_filter_options(self) -> List[Dict[str, Any]]:
        today = date.today()
        options = [
            ("Last 24 Hours", today - timedelta(days=1), today),
            ("Last 7 Days", today - timedelta(days=7), today),
            ("Last 30 Days", today - timedelta(days=30), today),
            ("Last 90 Days", today - timedelta(days=90), today),
        ]
        results = []
        for idx, (label, start, end) in enumerate(options, 1):
            results.append({
                "id": idx,
                "name": label,
                "startDate": start.isoformat(),
                "endDate": end.isoformat()
            })
        results.append({"id": len(results) + 1, "name": "Custom Date Range"})
        return results

    def _normalize_datacenter_name(self, value: Any) -> Optional[str]:
        if not value:
            return None
        name = str(value).strip()
        if not name:
            return None
        if name.startswith("EP_V2_"):
            mapping = {
                "EP_V2_MUM_BKC": "Mumbai-BKC",
                "EP_V2_DEL": "Delhi",
                "EP_V2_BL": "Bengaluru",
                "EP_V2_CHN_AMB": "Chennai-AMB",
                "EP_V2_UKCX": "Cressex",
            }
            return mapping.get(name, name)
        if name.startswith("EP_GCC_"):
            suffix = name.replace("EP_GCC_", "").strip()
            gcc_map = {
                "DEL": "Delhi",
                "MUM": "Mumbai",
                "BL": "Bengaluru",
                "BLR": "Bengaluru",
                "CHN": "Chennai",
            }
            city = gcc_map.get(suffix, suffix)
            return f"GCC{city}"
        return name

    def _format_options_table(self, title: str, columns: List[str], rows: List[List[str]]) -> str:
        header = "| # | " + " | ".join(columns) + " |\n"
        separator = "|---| " + " | ".join(["---"] * len(columns)) + " |\n"
        body = "\n".join(
            f"| {idx} | " + " | ".join(row) + " |"
            for idx, row in enumerate(rows, 1)
        )
        return f"## {title}\n\n" + header + separator + body

    def _wants_all(self, user_query: str, params: Dict[str, Any]) -> bool:
        if isinstance(params.get("all"), bool) and params.get("all"):
            return True
        if str(params.get("size", "")).strip().lower() in ["all", "full", "everything"]:
            return True
        query_lower = (user_query or "").lower()
        return any(kw in query_lower for kw in ["all", "everything", "full report", "entire report"])

    def _wants_report_catalog(self, user_query: str, params: Dict[str, Any]) -> bool:
        if params.get("report_type") or params.get("report"):
            return False
        query_lower = (user_query or "").lower()
        if not query_lower:
            return False
        keywords = [
            "reports available",
            "different reports",
            "report options",
            "list reports",
            "show reports",
            "which reports",
            "available reports",
        ]
        return "report" in query_lower and any(kw in query_lower for kw in keywords)

    def _get_report_catalog_response(self) -> Dict[str, Any]:
        options = [
            {"id": 1, "name": "Common", "value": "common_cluster"},
            {"id": 2, "name": "Cluster", "value": "cluster_inventory"},
            {"id": 3, "name": "Compute", "value": "cluster_compute"},
            {"id": 4, "name": "Storage", "value": "storage_inventory"},
        ]
        response = self._format_options_table(
            title="Available Reports",
            columns=["Report"],
            rows=[[opt["name"]] for opt in options]
        )
        response += "\n\n💡 Reply with the **report name** or **number** to open it."

        return {
            "success": True,
            "response": response,
            "set_filter_state": True,
            "filter_options_for_state": options,
            "filter_type_for_state": "report_catalog",
            "metadata": {
                "awaiting_filter_selection": True,
                "resource_type": "reports"
            }
        }
