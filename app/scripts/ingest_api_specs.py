#!/usr/bin/env python3
"""
Phase 1: Ingest API spec documents into RAG.

Converts resource_schema.json to markdown chunks and stores them
in the enterprise_rag table with source="api_spec".

Key guarantees (production-safe):
  - Stable doc_id per resource+operation → no duplicate rows on re-ingest
  - Content-hash deduplication → canonical spec wins when duplicates exist
  - Spec validation → broken chunks (empty URL / no params) are skipped with warnings
  - Recursive directory walk → catches specs in any subdirectory
  - Dry-run mode → preview what would be ingested without touching the DB

Usage:
  python -m app.scripts.ingest_api_specs                        # default dir
  python -m app.scripts.ingest_api_specs --schema-path PATH     # explicit path/dir
  python -m app.scripts.ingest_api_specs --dry-run              # no DB writes
  python -m app.scripts.ingest_api_specs --skip-validation      # ingest all (debug)

Requires: PostgreSQL running, POSTGRES_* env vars set.
"""

import asyncio
import hashlib
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Bootstrap project root on sys.path so app.* imports work when run directly
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.scripts.convert_schema_to_rag import convert_schema_to_documents  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
DEFAULT_CHUNKS_DIR = PROJECT_ROOT / "metadata" / "api_spec_chunks"

# ---------------------------------------------------------------------------
# Validation thresholds
# ---------------------------------------------------------------------------
# A spec is considered "broken" if it is missing ALL of these indicators.
_REQUIRED_FIELD_PATTERNS = [
    re.compile(r"\*\*Method:\*\*\s*\S+"),       # ## Endpoint -> Method
    re.compile(r"\*\*URL:\*\*\s*https?://\S+"),  # URL must be an actual http(s) URL
    re.compile(r"\*\*Resource:\*\*\s*\S+"),       # Resource header
    re.compile(r"\*\*Operation:\*\*\s*\S+"),      # Operation header
]

_PLACEHOLDER_PATTERNS = [
    re.compile(r"\{BASE_URL[^}]*\}"),   # e.g. {BASE_URL_PAAS_SERVICE}
    re.compile(r"`\{[^}]+\}`"),         # backtick-wrapped placeholders in URL
]

# Minimum content length to be worth ingesting (prevents empty/stub files)
_MIN_CONTENT_LENGTH = 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stable_doc_id(resource: str, operation: str) -> str:
    """
    Produce a deterministic doc_id for a given resource+operation pair.
    Using the same id on re-ingest allows the DB layer to UPSERT rather
    than INSERT, preventing row bloat.

    Format: api_spec:{resource}:{operation}
    """
    return f"api_spec:{resource}:{operation}"


def _content_hash(content: str) -> str:
    """SHA-256 hex digest of content for dedup comparison."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _extract_resource_operation(content: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse **Resource:** and **Operation:** from markdown content.
    Returns (resource, operation) or (None, None) if not found.
    """
    resource = operation = None
    for line in content.splitlines():
        if line.startswith("**Resource:**"):
            resource = line.split("**Resource:**", 1)[-1].strip()
        elif line.startswith("**Operation:**"):
            operation = line.split("**Operation:**", 1)[-1].strip()
        if resource and operation:
            break
    return resource, operation


def _validate_spec(doc: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate a RAG document dict before ingestion.

    Returns (is_valid, reason_if_invalid).
    """
    content = doc.get("content", "")

    # 1. Minimum length
    if len(content.strip()) < _MIN_CONTENT_LENGTH:
        return False, f"Content too short ({len(content.strip())} chars)"

    # 2. Placeholder URLs — these are unfilled template stubs
    for pat in _PLACEHOLDER_PATTERNS:
        if pat.search(content):
            return False, f"Contains placeholder URL pattern: {pat.pattern!r}"

    # 3. Must have required structured fields
    missing = [
        pat.pattern
        for pat in _REQUIRED_FIELD_PATTERNS
        if not pat.search(content)
    ]
    if missing:
        return False, f"Missing required fields: {missing}"

    return True, ""


def _score_spec(content: str) -> int:
    """
    Score a spec document so that when duplicates exist we keep the best one.
    Higher is better. Scoring heuristics:
      +10  has a real https:// URL
      +5   has Required Parameters (non-None)
      +5   has Optional Parameters (non-None)
      +3   has Response Mapping section
      +2   has Permissions section
      +1   has Workflow Steps section
    """
    score = 0
    if re.search(r"https?://\S+", content):
        score += 10
    if re.search(r"## Required Parameters\n(?!None)", content):
        score += 5
    if re.search(r"## Optional Parameters\n(?!None)", content):
        score += 5
    if "## Response Mapping" in content:
        score += 3
    if "## Permissions" in content:
        score += 2
    if "## Workflow Steps" in content:
        score += 1
    return score


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------

def _load_markdown_file(file_path: Path) -> Optional[Dict[str, Any]]:
    """
    Load a single .md / .txt file into a RAG document dict.
    Returns None on read failure.
    """
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        logger.error(f"Cannot read {file_path}: {exc}")
        return None

    if not content.strip():
        logger.warning(f"Skipping empty file: {file_path.name}")
        return None

    # Extract resource+operation from structured markdown if available
    resource, operation = _extract_resource_operation(content)

    # Build stable doc_id
    if resource and operation:
        doc_id = _stable_doc_id(resource, operation)
    else:
        # Fallback: hash of file stem so it remains stable across re-ingests
        doc_id = f"api_spec:file:{_content_hash(file_path.stem)[:16]}"
        logger.debug(
            f"{file_path.name}: could not extract resource/operation — "
            f"using stem-hash doc_id ({doc_id})"
        )

    return {
        "doc_id": doc_id,
        "content": content,
        "url": f"internal://api_spec_chunks/{file_path.name}",
        "title": file_path.stem.replace("_", " ").title(),
        "source": "api_spec",
        "format": "markdown",
        "timestamp": None,   # filled later
        "images": [],
        "_content_hash": _content_hash(content),
        "_source_file": str(file_path),
    }


def _load_json_schema_file(file_path: Path) -> List[Dict[str, Any]]:
    """
    Convert a resource_schema JSON file into RAG document dicts.
    """
    try:
        raw_docs = convert_schema_to_documents(str(file_path))
    except Exception as exc:
        logger.error(f"Failed converting JSON schema {file_path}: {exc}")
        return []

    enriched = []
    for doc in raw_docs:
        resource, operation = _extract_resource_operation(doc.get("content", ""))
        doc_id = (
            _stable_doc_id(resource, operation)
            if resource and operation
            else f"api_spec:json:{_content_hash(doc.get('content', ''))[:16]}"
        )
        doc["doc_id"] = doc_id
        doc["_content_hash"] = _content_hash(doc.get("content", ""))
        doc["_source_file"] = str(file_path)
        doc.setdefault("images", [])
        enriched.append(doc)

    return enriched


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _deduplicate_documents(
    documents: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    For each unique doc_id, keep only the highest-scored spec.
    Returns (kept_docs, dropped_docs).
    """
    # doc_id → (doc, score)
    registry: Dict[str, Tuple[Dict[str, Any], int]] = {}

    for doc in documents:
        doc_id = doc["doc_id"]
        score = _score_spec(doc["content"])

        if doc_id not in registry:
            registry[doc_id] = (doc, score)
        else:
            existing_doc, existing_score = registry[doc_id]
            if score > existing_score:
                logger.warning(
                    f"DUPLICATE doc_id={doc_id!r} — "
                    f"replacing {existing_doc['_source_file']} (score={existing_score}) "
                    f"with {doc['_source_file']} (score={score})"
                )
                registry[doc_id] = (doc, score)
            else:
                logger.warning(
                    f"DUPLICATE doc_id={doc_id!r} — "
                    f"keeping {existing_doc['_source_file']} (score={existing_score}), "
                    f"dropping {doc['_source_file']} (score={score})"
                )

    kept = [doc for doc, _ in registry.values()]
    dropped = [
        doc for doc in documents
        if doc not in [d for d, _ in registry.values()]
    ]
    return kept, dropped


# ---------------------------------------------------------------------------
# Validation pass
# ---------------------------------------------------------------------------

def _validate_documents(
    documents: List[Dict[str, Any]],
    skip_validation: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Filter out broken / incomplete spec documents.
    Returns (valid_docs, invalid_docs).
    """
    if skip_validation:
        logger.warning("Spec validation DISABLED — all documents will be ingested as-is")
        return documents, []

    valid, invalid = [], []
    for doc in documents:
        ok, reason = _validate_spec(doc)
        if ok:
            valid.append(doc)
        else:
            invalid.append(doc)
            logger.warning(
                f"INVALID spec skipped — {doc['_source_file']} "
                f"(doc_id={doc['doc_id']!r}): {reason}"
            )

    return valid, invalid


# ---------------------------------------------------------------------------
# Main ingestion entry point
# ---------------------------------------------------------------------------

async def ingest_api_specs(
    schema_path: Optional[str] = None,
    dry_run: bool = False,
    skip_validation: bool = False,
) -> int:
    """
    Load all API spec files from `schema_path` (file or directory),
    deduplicate, validate, and ingest into the RAG knowledge base.

    Args:
        schema_path:      Path to a directory containing .md/.txt/.json files,
                          or a single .json resource schema file.
                          Defaults to metadata/api_spec_chunks.
        dry_run:          If True, perform all processing but do NOT write to DB.
        skip_validation:  If True, bypass spec-quality validation (useful for
                          debugging but NOT recommended in production).

    Returns:
        Number of documents successfully ingested (0 on dry-run).
    """
    target = Path(schema_path) if schema_path else DEFAULT_CHUNKS_DIR

    # ------------------------------------------------------------------ #
    # 1. Collect raw documents
    # ------------------------------------------------------------------ #
    raw_documents: List[Dict[str, Any]] = []

    if target.is_file():
        if target.suffix.lower() == ".json":
            raw_documents = _load_json_schema_file(target)
        elif target.suffix.lower() in {".md", ".txt"}:
            doc = _load_markdown_file(target)
            if doc:
                raw_documents = [doc]
        else:
            logger.error(f"Unsupported single-file type: {target.suffix}")
            return 0

    elif target.is_dir():
        # Recursive walk — catches specs in any subdirectory
        all_files = sorted(
            f for f in target.rglob("*")
            if f.is_file() and not f.name.startswith(".")
        )

        if not all_files:
            logger.warning(f"No files found under {target}")
            return 0

        logger.info(f"Discovered {len(all_files)} file(s) under {target}")

        for file_path in all_files:
            suffix = file_path.suffix.lower()
            if suffix == ".json":
                raw_documents.extend(_load_json_schema_file(file_path))
            elif suffix in {".md", ".txt"}:
                doc = _load_markdown_file(file_path)
                if doc:
                    raw_documents.append(doc)
            else:
                logger.debug(f"Skipping unsupported type: {file_path.name}")

    else:
        logger.error(f"Path not found or not accessible: {target}")
        return 0

    logger.info(f"Loaded {len(raw_documents)} raw document(s)")

    if not raw_documents:
        logger.warning("Nothing to ingest.")
        return 0

    # ------------------------------------------------------------------ #
    # 2. Deduplicate — keep best spec per resource+operation
    # ------------------------------------------------------------------ #
    deduped, dropped_dupes = _deduplicate_documents(raw_documents)
    if dropped_dupes:
        logger.warning(
            f"Deduplication: dropped {len(dropped_dupes)} duplicate(s), "
            f"keeping {len(deduped)} unique spec(s)"
        )

    # ------------------------------------------------------------------ #
    # 3. Validate — reject broken / incomplete specs
    # ------------------------------------------------------------------ #
    valid_docs, invalid_docs = _validate_documents(deduped, skip_validation)
    if invalid_docs:
        logger.warning(
            f"Validation: rejected {len(invalid_docs)} broken spec(s), "
            f"{len(valid_docs)} spec(s) remain"
        )

    if not valid_docs:
        logger.error("No valid documents to ingest after dedup + validation.")
        return 0

    # ------------------------------------------------------------------ #
    # 4. Stamp timestamps (do not overwrite if already set)
    # ------------------------------------------------------------------ #
    now = datetime.now(timezone.utc).isoformat()
    for doc in valid_docs:
        if not doc.get("timestamp"):
            doc["timestamp"] = now
        # Remove internal-only bookkeeping keys before DB insert
        doc.pop("_content_hash", None)
        doc.pop("_source_file", None)

    # ------------------------------------------------------------------ #
    # 5. Dry-run: report and exit
    # ------------------------------------------------------------------ #
    if dry_run:
        logger.info("=== DRY RUN — no changes written to the database ===")
        logger.info(f"Would ingest {len(valid_docs)} document(s):")
        for doc in valid_docs:
            logger.info(f"  [{doc['doc_id']}]  {doc['title']}")
        logger.info(
            f"Summary: raw={len(raw_documents)}, "
            f"after_dedup={len(deduped)}, "
            f"after_validation={len(valid_docs)}, "
            f"dropped_dupes={len(dropped_dupes)}, "
            f"invalid={len(invalid_docs)}"
        )
        return 0

    # ------------------------------------------------------------------ #
    # 6. Ingest into PostgreSQL / pgvector
    # ------------------------------------------------------------------ #
    from app.services.postgres_service import postgres_service  # noqa: E402 (late import — avoids startup cost in dry-run)

    if not postgres_service.pool:
        await postgres_service.initialize()

    if not postgres_service.pool:
        logger.error(
            "PostgreSQL connection pool unavailable. "
            "Check POSTGRES_HOST / POSTGRES_DB / POSTGRES_USER / POSTGRES_PASSWORD env vars."
        )
        return 0

    logger.info(f"Ingesting {len(valid_docs)} document(s) into RAG knowledge base…")

    try:
        ids = await postgres_service.add_documents(valid_docs)
    except Exception as exc:
        logger.exception(f"Ingestion failed: {exc}")
        return 0

    ingested_count = len(ids) if ids else 0
    logger.info(
        f"✅ Ingestion complete — "
        f"ingested={ingested_count}, "
        f"dropped_dupes={len(dropped_dupes)}, "
        f"invalid={len(invalid_docs)}"
    )
    return ingested_count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Ingest API spec files into the RAG knowledge base.\n"
            "Supports .json (resource schema), .md, and .txt files. "
            "Recursively walks directories."
        )
    )
    parser.add_argument(
        "--schema-path",
        default=None,
        help=(
            "Path to a directory of spec files (.md/.txt/.json) "
            "or a single resource_schema.json. "
            f"Defaults to: {DEFAULT_CHUNKS_DIR}"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be ingested without writing to the database.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help=(
            "Bypass spec-quality validation. "
            "NOT recommended for production — use for debugging only."
        ),
    )
    args = parser.parse_args()

    count = asyncio.run(
        ingest_api_specs(
            schema_path=args.schema_path,
            dry_run=args.dry_run,
            skip_validation=args.skip_validation,
        )
    )

    if args.dry_run:
        return 0

    if count == 0:
        logger.error("No documents were ingested. Check warnings above.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())