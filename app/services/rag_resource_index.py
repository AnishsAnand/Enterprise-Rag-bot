"""
RAG-driven resource index. No hardcoded resource/operation lists.
Populated from API spec chunks at runtime. Used by Orchestrator and IntentAgent.
"""
import logging
import re
from typing import Dict, List, Set, Tuple

logger = logging.getLogger(__name__)

# Module-level cache (populated from RAG)
_resource_index: Dict[str, List[str]] = {}
_skip_greeting_terms: Set[str] = set()
_initialized = False


def _parse_resource_operation_from_chunk(content: str) -> Tuple[str, str] | None:
    """Extract (resource, operation) from RAG chunk."""
    if not content:
        return None
    res = re.search(r"\*\*Resource:\*\*\s*([a-zA-Z0-9_]+)", content, re.IGNORECASE)
    op = re.search(r"\*\*Operation:\*\*\s*([a-zA-Z0-9_]+)", content, re.IGNORECASE)
    if res and op:
        return (res.group(1).lower(), op.group(1).lower())
    return None


def _parse_aliases_from_chunk(content: str) -> List[str]:
    """Extract **Aliases:** comma-separated values from RAG chunk."""
    if not content:
        return []
    m = re.search(r"\*\*Aliases:\*\*\s*([^\n]+)", content, re.IGNORECASE)
    if m:
        return [a.strip().lower() for a in m.group(1).split(",") if a.strip()]
    return []


async def ensure_initialized() -> None:
    """Populate resource index from RAG API specs. Call before using get_*."""
    global _resource_index, _skip_greeting_terms, _initialized
    if _initialized:
        return
    try:
        from app.services.postgres_service import postgres_service
        if not postgres_service.pool:
            await postgres_service.initialize()
        if not postgres_service.pool:
            return
        results = await postgres_service.search_api_specs(
            "API specification resource list create read update delete", n_results=50
        )
        index: Dict[str, List[str]] = {}
        terms: Set[str] = set()
        for r in results:
            content = r.get("content", "")
            parsed = _parse_resource_operation_from_chunk(content)
            if parsed:
                resource, op = parsed
                if resource not in index:
                    index[resource] = []
                if op not in index[resource]:
                    index[resource].append(op)
                terms.add(resource)
                terms.add(resource.replace("_", " "))
                for alias in _parse_aliases_from_chunk(content):
                    terms.add(alias)
                terms.add(op)
        _resource_index = index
        _skip_greeting_terms = terms
        _initialized = True
        if index:
            logger.info(f"📚 RAG resource index: {len(index)} resources, {len(terms)} skip-greeting terms")
    except Exception as e:
        logger.debug(f"RAG resource index failed: {e}")


def get_resource_index() -> Dict[str, List[str]]:
    """Resource type -> list of operations. Empty until ensure_initialized() runs."""
    return dict(_resource_index)


def get_skip_greeting_terms() -> Set[str]:
    """Terms from RAG (resource names, aliases, operations). If in user input, skip greeting check."""
    return set(_skip_greeting_terms)


def input_matches_rag_terms(user_input: str) -> bool:
    """
    True if user input contains any RAG-derived term (resource, alias, operation).
    Used by orchestrator to skip greeting check without hardcoded lists.
    """
    if not _skip_greeting_terms:
        return False
    lower = user_input.lower()
    for term in _skip_greeting_terms:
        if len(term) >= 2 and term in lower:
            return True
    return False
