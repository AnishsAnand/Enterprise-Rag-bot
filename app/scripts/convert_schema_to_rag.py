#!/usr/bin/env python3
"""
Phase 1: Convert resource_schema.json to RAG-ready markdown chunks.

Each resource+operation becomes a document chunk with:
- Structured markdown content (LLM-parseable)
- source="api_spec" for filtering
- title and url for identification

Usage:
  python -m app.scripts.convert_schema_to_rag [--output-dir PATH] [--schema-path PATH]
"""

import json
import os
import argparse
from pathlib import Path
from typing import Any, Dict, List

# Default paths
DEFAULT_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "config" / "resource_schema.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "metadata" / "api_spec_chunks"


def _format_param_list(params: List[str]) -> str:
    """Format parameter list for markdown."""
    if not params:
        return "None"
    return "\n".join(f"- `{p}`" for p in params)


def _format_workflow_steps(workflow: Dict[str, Any], resource: str) -> str:
    """Format workflow steps for markdown."""
    lines = []
    for wk_name, wk_config in (workflow or {}).items():
        steps = wk_config.get("steps", [])
        if not steps:
            continue
        lines.append(f"### Workflow: {wk_name}")
        lines.append(wk_config.get("description", ""))
        for s in steps:
            action = s.get("action", "?")
            res = s.get("resource", "")
            op = s.get("operation", "")
            dep = s.get("depends_on", [])
            dep_str = f" (depends on: {', '.join(dep)})" if dep else ""
            lines.append(f"- Step {s.get('step', '?')}: {action} ({res}.{op}){dep_str}")
        lines.append("")
    return "\n".join(lines) if lines else "No workflow defined"


def resource_to_markdown_chunk(
    resource_type: str,
    operation: str,
    config: Dict[str, Any],
) -> str:
    """
    Convert a single resource+operation to markdown for RAG.
    """
    api_endpoints = config.get("api_endpoints", {})
    params_config = config.get("parameters", {})
    response_mapping = config.get("response_mapping", {})
    permissions = config.get("permissions", {})
    aliases = config.get("aliases", [])
    workflow = config.get("workflow", {})

    ep = api_endpoints.get(operation, {})
    method = ep.get("method", "GET")
    url = ep.get("url", "")
    description = ep.get("description", "")

    param_cfg = params_config.get(operation, {})
    required = param_cfg.get("required", [])
    optional = param_cfg.get("optional", [])
    param_desc = param_cfg.get("description", "")

    # Build markdown
    parts = [
        f"# API Specification: {resource_type} - {operation}",
        "",
        f"**Resource:** {resource_type}",
        f"**Operation:** {operation}",
    ]
    if aliases:
        parts.append(f"**Aliases:** {', '.join(aliases)}")
    parts.append("")

    parts.append("## Endpoint")
    parts.append(f"- **Method:** {method}")
    parts.append(f"- **URL:** {url}")
    parts.append("- **Auth:** Bearer token (from Keycloak)")
    parts.append(f"- **Description:** {description}")
    parts.append("")

    parts.append("## Required Parameters")
    parts.append(_format_param_list(required))
    if param_desc:
        parts.append(f"\n{param_desc}")
    parts.append("")

    parts.append("## Optional Parameters")
    parts.append(_format_param_list(optional))
    parts.append("")

    if response_mapping:
        parts.append("## Response Mapping")
        for key, path in response_mapping.items():
            parts.append(f"- `{key}`: {path}")
        parts.append("")

    if permissions:
        parts.append("## Permissions")
        parts.append(f"Roles: {', '.join(permissions.get(operation, []))}")
        parts.append("")

    workflow_str = _format_workflow_steps(workflow, resource_type)
    if workflow_str:
        parts.append("## Workflow Steps")
        parts.append(workflow_str)

    return "\n".join(parts)


def convert_schema_to_documents(schema_path: str) -> List[Dict[str, Any]]:
    """
    Convert resource_schema.json to list of RAG-ready documents.

    Returns list of dicts with: content, url, title, source, format, timestamp
    """
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    resources = schema.get("resources", {})
    documents = []

    for resource_type, config in resources.items():
        operations = config.get("operations", [])
        api_endpoints = config.get("api_endpoints", {})

        # Include all operations from api_endpoints (some may not be in operations list)
        all_ops = set(operations) | set(api_endpoints.keys())

        for op in sorted(all_ops):
            if op in api_endpoints or op in config.get("parameters", {}):
                content = resource_to_markdown_chunk(resource_type, op, config)
                doc_id = f"api_spec:{resource_type}:{op}"
                documents.append({
                    "content": content,
                    "url": f"internal://api_spec/{resource_type}/{op}",
                    "title": f"API {resource_type} {op}",
                    "source": "api_spec",
                    "format": "markdown",
                    "timestamp": None,  # Will use now() in ingestion
                })
    return documents


def main():
    parser = argparse.ArgumentParser(description="Convert resource_schema.json to RAG chunks")
    parser.add_argument(
        "--schema-path",
        default=str(DEFAULT_SCHEMA_PATH),
        help="Path to resource_schema.json",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to write markdown files (optional)",
    )
    parser.add_argument(
        "--write-files",
        action="store_true",
        help="Write markdown files to output-dir",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output documents as JSON to stdout",
    )
    args = parser.parse_args()

    if not os.path.exists(args.schema_path):
        print(f"Error: Schema not found at {args.schema_path}")
        return 1

    documents = convert_schema_to_documents(args.schema_path)
    print(f"Converted {len(documents)} API spec chunks")

    if args.write_files:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, doc in enumerate(documents):
            title = doc["title"].replace(" ", "_").lower()
            fpath = out_dir / f"{title}_{i}.md"
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(doc["content"])
            print(f"  Wrote {fpath}")
        print(f"Wrote {len(documents)} files to {out_dir}")

    if args.json:
        import json as j
        print(j.dumps(documents, indent=2))

    return 0


if __name__ == "__main__":
    exit(main())
