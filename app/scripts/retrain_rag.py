#!/usr/bin/env python3
"""
Retrain RAG: Clear knowledge base, optionally bulk-scrape URLs, ingest API specs + md files.

BULK SCRAPE COMMANDS (most common):
  # Basic bulk scrape: clear KB + scrape docs + ingest API/md
  python -m app.scripts.retrain_rag --base-url https://docs.example.com

  # Full deep crawl (no URL limit, depth 15)
  python -m app.scripts.retrain_rag --base-url https://docs.example.com --no-limit

  # Scrape only (no API/md ingest)
  python -m app.scripts.retrain_rag --base-url https://docs.example.com --scrape-only

  # With seed URLs (guarantee critical pages are included)
  python -m app.scripts.retrain_rag --base-url https://ipcloud.tatacommunications.com/docs/docs/ \\
    --seed-urls https://ipcloud.tatacommunications.com/docs/docs/zones

  # Via backend API (server must be running)
  python -m app.scripts.retrain_rag --api-base http://localhost:8000 --base-url https://docs.example.com

  # Dry-run: preview what would be ingested without touching the DB
  python -m app.scripts.retrain_rag --dry-run

OTHER OPTIONS:
  # Full retrain: clear + API specs + md files (no scrape)
  python -m app.scripts.retrain_rag

  # No clear (append to existing KB)
  python -m app.scripts.retrain_rag --no-clear --base-url https://docs.example.com

  # Custom paths
  python -m app.scripts.retrain_rag --schema-path /path/to/schema.json --md-dir /path/to/md

Requires: PostgreSQL, POSTGRES_* env vars. For scrape: crawl4ai (pip install crawl4ai), Playwright.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_SCHEMA_PATH = PROJECT_ROOT / "metadata" / "resource_schema_backup.json"
DEFAULT_MD_DIR = PROJECT_ROOT / "metadata" / "api_spec_chunks"


# ---------------------------------------------------------------------------
# KB management
# ---------------------------------------------------------------------------

async def clear_knowledge_base(use_truncate: bool = True) -> None:
    """Clear the RAG table. TRUNCATE is faster; delete_collection drops and recreates."""
    from app.services.postgres_service import postgres_service

    if not postgres_service.pool:
        await postgres_service.initialize()
    if not postgres_service.pool:
        raise RuntimeError("PostgreSQL unavailable. Set POSTGRES_HOST, POSTGRES_DB, etc.")

    if use_truncate:
        await postgres_service.truncate_table()
    else:
        await postgres_service.delete_collection()
        await postgres_service.initialize()
    logger.info("✅ Knowledge base cleared")


async def clear_via_api(api_base: str) -> None:
    """Call backend clear-knowledge API. Server must be running."""
    try:
        import httpx
    except ImportError:
        raise RuntimeError("httpx required for --api-base. pip install httpx")

    url = f"{api_base.rstrip('/')}/rag-widget/widget/clear-knowledge"
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.delete(url)
        r.raise_for_status()
    logger.info("✅ Knowledge base cleared via API")


# ---------------------------------------------------------------------------
# Bulk scrape
# ---------------------------------------------------------------------------

async def run_bulk_scrape_via_api(
    api_base: str,
    base_url: str,
    max_depth: int = 15,
    max_urls: int | None = None,
    seed_urls: list[str] | None = None,
) -> int:
    """Call backend bulk-scrape API. Server must be running."""
    try:
        import httpx
    except ImportError:
        raise RuntimeError("httpx required for --api-base. pip install httpx")

    url = f"{api_base.rstrip('/')}/rag-widget/widget/bulk-scrape"
    payload = {
        "base_url": base_url,
        "max_depth": max_depth,
        "max_urls": max_urls or 5000,
        "auto_store": True,
        "extract_deep_images": True,
    }
    if seed_urls:
        payload["seed_urls"] = seed_urls
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
    count = data.get("discovered_urls_count", 0)
    logger.info(f"✅ Bulk scrape started: {count} URLs discovered (runs in background)")
    return count


async def run_bulk_scrape(
    base_url: str,
    max_depth: int = 15,
    max_urls: int | None = 2000,
    seed_urls: list[str] | None = None,
) -> int:
    """
    Run bulk scrape: discover URLs, scrape, store in RAG.
    Returns number of documents stored.
    """
    from app.services.scraper_service import scraper_service
    from app.api.routes.rag_widget import enhanced_bulk_scrape_task

    effective_max = max_urls if max_urls and max_urls > 0 else None
    logger.info(
        f"🌐 Discovering URLs from {base_url} "
        f"(depth={max_depth}, max_urls={effective_max or 'unlimited'})"
    )
    discovered_urls = await scraper_service.discover_url(base_url, max_depth, effective_max or 0)

    if seed_urls:
        seed_set = {u.strip() for u in seed_urls if u and u.strip()}
        discovered_urls = list(set(discovered_urls or []) | seed_set)
        logger.info(f"🌱 Merged {len(seed_set)} seed URL(s): {len(discovered_urls)} total")

    if not discovered_urls:
        logger.warning("No URLs discovered — check base_url and network access")
        return 0

    logger.info(f"📄 Found {len(discovered_urls)} URL(s), starting scrape…")
    await enhanced_bulk_scrape_task(
        discovered_urls,
        auto_store=True,
        max_depth=max_depth,
        extract_images=True,
    )
    return len(discovered_urls)


# ---------------------------------------------------------------------------
# MD file loading  (FIXED: recursive glob + stable doc_ids)
# ---------------------------------------------------------------------------

def _stable_md_doc_id(file_path: Path) -> str:
    """
    Produce a deterministic doc_id for a manually-authored markdown file.

    Strategy: mirror ingest_api_specs.py — try to extract Resource+Operation
    from the file header; fall back to a path-based stable key so that
    re-ingesting the same file never creates a second DB row.
    """
    import re
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        content = ""

    resource = operation = None
    for line in content.splitlines():
        if line.startswith("**Resource:**"):
            resource = line.split("**Resource:**", 1)[-1].strip()
        elif line.startswith("**Operation:**"):
            operation = line.split("**Operation:**", 1)[-1].strip()
        if resource and operation:
            break

    if resource and operation:
        return f"api_spec:{resource}:{operation}"

    # Stable fallback: relative path from the MD directory root
    return f"api_spec:file:{file_path.stem}"


def load_md_files_from_dir(md_dir: Path, source: str = "api_spec") -> list:
    """
    Load markdown/text files from `md_dir` into RAG document dicts.

    Changes from original:
      - Uses rglob("*.md") + rglob("*.txt") → RECURSIVE (catches subdirectories)
      - Attaches a stable doc_id per file so re-ingests UPSERT rather than INSERT
      - Logs a warning for empty files instead of silently skipping
    """
    if not md_dir.exists():
        logger.warning(f"MD directory not found: {md_dir}")
        return []

    # Collect .md and .txt recursively, sorted for deterministic ordering
    all_files = sorted(
        f
        for pattern in ("*.md", "*.txt")
        for f in md_dir.rglob(pattern)
        if f.is_file() and not f.name.startswith(".")
    )

    if not all_files:
        logger.warning(f"No .md / .txt files found under {md_dir}")
        return []

    documents = []
    now = datetime.now(timezone.utc).isoformat()

    for fpath in all_files:
        try:
            content = fpath.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            logger.warning(f"Cannot read {fpath}: {exc}")
            continue

        if not content.strip():
            logger.warning(f"Skipping empty file: {fpath.name}")
            continue

        documents.append({
            "doc_id": _stable_md_doc_id(fpath),
            "content": content,
            "url": f"internal://api_spec/{fpath.stem}",
            "title": f"API {fpath.stem.replace('_', ' ').title()}",
            "source": source,
            "format": "markdown",
            "timestamp": now,
            "images": [],
        })

    logger.info(f"Loaded {len(documents)} file(s) from {md_dir}")
    return documents


# ---------------------------------------------------------------------------
# Ingestion helpers
# ---------------------------------------------------------------------------

async def ingest_api_specs_from_schema(schema_path: Path) -> int:
    """Ingest API specs from resource schema JSON via the dedicated script."""
    from app.scripts.convert_schema_to_rag import convert_schema_to_documents
    import hashlib

    if not schema_path.exists():
        logger.warning(f"Schema not found: {schema_path}")
        return 0

    documents = convert_schema_to_documents(str(schema_path))
    if not documents:
        return 0

    now = datetime.now(timezone.utc).isoformat()
    for doc in documents:
        doc["timestamp"] = now
        doc.setdefault("images", [])

        # Attach stable doc_id (resource+operation extracted from content)
        resource = operation = None
        for line in doc.get("content", "").splitlines():
            if line.startswith("**Resource:**"):
                resource = line.split("**Resource:**", 1)[-1].strip()
            elif line.startswith("**Operation:**"):
                operation = line.split("**Operation:**", 1)[-1].strip()
            if resource and operation:
                break
        doc["doc_id"] = (
            f"api_spec:{resource}:{operation}"
            if resource and operation
            else f"api_spec:json:{hashlib.sha256(doc.get('content','').encode()).hexdigest()[:16]}"
        )

    from app.services.postgres_service import postgres_service

    if not postgres_service.pool:
        await postgres_service.initialize()
    if not postgres_service.pool:
        raise RuntimeError("PostgreSQL unavailable")

    ids = await postgres_service.add_documents(documents)
    logger.info(f"✅ Ingested {len(ids)} API spec document(s) from schema")
    return len(ids)


async def ingest_md_files_from_dir(md_dir: Path) -> int:
    """Ingest markdown files from directory using the fixed recursive loader."""
    documents = load_md_files_from_dir(md_dir)
    if not documents:
        return 0

    from app.services.postgres_service import postgres_service

    if not postgres_service.pool:
        await postgres_service.initialize()
    if not postgres_service.pool:
        raise RuntimeError("PostgreSQL unavailable")

    ids = await postgres_service.add_documents(documents)
    logger.info(f"✅ Ingested {len(ids)} md document(s) from {md_dir}")
    return len(ids)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

async def retrain(
    clear: bool = True,
    use_truncate: bool = True,
    base_url: str | None = None,
    scrape_only: bool = False,
    schema_path: Path | None = None,
    md_dir: Path | None = None,
    max_depth: int = 15,
    max_urls: int | None = 2000,
    seed_urls: list[str] | None = None,
    api_base: str | None = None,
    ingest_api_specs: bool = True,
    ingest_md_files: bool = True,
    dry_run: bool = False,
) -> dict:
    """
    Full retrain pipeline.

    Steps:
      1. (Optional) Clear knowledge base
      2. (Optional) Bulk scrape external docs
      3. (Optional) Ingest API spec JSON → markdown
      4. (Optional) Ingest .md/.txt files from directory

    Returns: {scraped: N, api_specs: N, md_files: N, total: N}
    """
    schema_path = schema_path or DEFAULT_SCHEMA_PATH
    md_dir = md_dir or DEFAULT_MD_DIR

    stats = {"scraped": 0, "api_specs": 0, "md_files": 0, "total": 0}

    if dry_run:
        logger.info("=== DRY RUN MODE — no changes will be written to the database ===")

    # ------------------------------------------------------------------ #
    # 1. Clear
    # ------------------------------------------------------------------ #
    if clear and not dry_run:
        if api_base:
            await clear_via_api(api_base)
            from app.services.postgres_service import postgres_service
            if not postgres_service.pool:
                await postgres_service.initialize()
        else:
            await clear_knowledge_base(use_truncate=use_truncate)
    elif clear and dry_run:
        logger.info("[DRY RUN] Would clear knowledge base")

    # ------------------------------------------------------------------ #
    # 2. Bulk scrape (optional)
    # ------------------------------------------------------------------ #
    if base_url:
        if dry_run:
            logger.info(f"[DRY RUN] Would bulk-scrape: {base_url}")
        elif api_base:
            stats["scraped"] = await run_bulk_scrape_via_api(
                api_base, base_url, max_depth, max_urls, seed_urls
            )
            if scrape_only:
                logger.info(
                    "⏳ Scrape running in background on server. "
                    "Check server logs for progress."
                )
                stats["total"] = stats["scraped"]
                return stats
            logger.info("⏳ Waiting 10 s for scrape to progress before ingesting API/md…")
            await asyncio.sleep(10)
        else:
            stats["scraped"] = await run_bulk_scrape(base_url, max_depth, max_urls, seed_urls)
            if scrape_only:
                stats["total"] = stats["scraped"]
                return stats

    # ------------------------------------------------------------------ #
    # 3. Ingest API specs from JSON schema
    # ------------------------------------------------------------------ #
    if not scrape_only and ingest_api_specs:
        if dry_run:
            logger.info(f"[DRY RUN] Would ingest API specs from: {schema_path}")
        else:
            stats["api_specs"] = await ingest_api_specs_from_schema(schema_path)

    # ------------------------------------------------------------------ #
    # 4. Ingest .md / .txt files (recursive)
    # ------------------------------------------------------------------ #
    if not scrape_only and ingest_md_files:
        if dry_run:
            docs = load_md_files_from_dir(md_dir)
            logger.info(f"[DRY RUN] Would ingest {len(docs)} md file(s) from {md_dir}")
        else:
            stats["md_files"] = await ingest_md_files_from_dir(md_dir)

    stats["total"] = stats["scraped"] + stats["api_specs"] + stats["md_files"]
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Retrain RAG: clear, bulk scrape, ingest API specs + md files"
    )
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="Do not clear KB before ingestion (append mode)",
    )
    parser.add_argument(
        "--drop-table",
        action="store_true",
        help="Use DROP TABLE + recreate instead of TRUNCATE when clearing",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        help="Base URL for bulk scrape (e.g. https://docs.example.com)",
    )
    parser.add_argument(
        "--scrape-only",
        action="store_true",
        help="Only run bulk scrape — skip API spec and md file ingestion",
    )
    parser.add_argument(
        "--schema-path",
        type=Path,
        default=DEFAULT_SCHEMA_PATH,
        help="Path to resource_schema.json",
    )
    parser.add_argument(
        "--md-dir",
        type=Path,
        default=DEFAULT_MD_DIR,
        help="Directory containing .md/.txt API spec files (searched recursively)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=15,
        help="Scrape discovery depth (1–20)",
    )
    parser.add_argument(
        "--max-urls",
        type=int,
        default=2000,
        help="Max URLs to scrape (0 = unlimited)",
    )
    parser.add_argument(
        "--no-limit",
        action="store_true",
        help="Crawl entire site with no URL cap (overrides --max-urls)",
    )
    parser.add_argument(
        "--seed-urls",
        type=str,
        nargs="+",
        help="Extra seed URLs to always include in scrape",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        help="Hit backend REST APIs instead of direct DB (e.g. http://localhost:8000)",
    )
    parser.add_argument(
        "--no-api-specs",
        action="store_true",
        help="Skip API spec ingestion from JSON schema",
    )
    parser.add_argument(
        "--no-md-files",
        action="store_true",
        help="Skip md file ingestion from directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview pipeline actions without writing to the database",
    )
    args = parser.parse_args()

    max_urls = 0 if args.no_limit else args.max_urls

    stats = asyncio.run(
        retrain(
            clear=not args.no_clear,
            use_truncate=not args.drop_table,
            base_url=args.base_url,
            scrape_only=args.scrape_only,
            schema_path=args.schema_path,
            md_dir=args.md_dir,
            max_depth=args.max_depth,
            max_urls=max_urls,
            seed_urls=args.seed_urls or None,
            api_base=args.api_base,
            ingest_api_specs=not args.no_api_specs,
            ingest_md_files=not args.no_md_files,
            dry_run=args.dry_run,
        )
    )

    if args.dry_run:
        logger.info("Dry run complete — no data was written")
        return 0

    logger.info(
        f"🎉 Retrain complete — "
        f"scraped={stats['scraped']}, "
        f"api_specs={stats['api_specs']}, "
        f"md_files={stats['md_files']}, "
        f"total={stats['total']}"
    )
    return 0 if stats["total"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())