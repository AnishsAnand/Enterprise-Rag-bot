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

OTHER OPTIONS:
  # Full retrain: clear + API specs + md files (no scrape)
  python -m app.scripts.retrain_rag

  # No clear (append to existing)
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_SCHEMA_PATH = PROJECT_ROOT / "metadata" / "resource_schema_backup.json"
DEFAULT_MD_DIR = PROJECT_ROOT / "metadata" / "api_spec_chunks"


async def clear_knowledge_base(use_truncate: bool = True) -> None:
    """Clear the RAG table. Truncate is faster; delete_collection drops and recreates."""
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
        f"🌐 Discovering URLs from {base_url} (depth={max_depth}, max_urls={effective_max or 'unlimited'})"
    )
    discovered_urls = await scraper_service.discover_url(base_url, max_depth, effective_max or 0)

    if seed_urls:
        seed_set = {u.strip() for u in seed_urls if u and u.strip()}
        discovered_urls = list(set(discovered_urls or []) | seed_set)
        logger.info(f"🌱 Merged {len(seed_set)} seed URLs: {len(discovered_urls)} total")

    if not discovered_urls:
        logger.warning("No URLs discovered")
        return 0

    logger.info(f"📄 Found {len(discovered_urls)} URLs, starting scrape...")
    await enhanced_bulk_scrape_task(
        discovered_urls,
        auto_store=True,
        max_depth=max_depth,
        extract_images=True,
    )
    return len(discovered_urls)


def load_md_files_from_dir(md_dir: Path, source: str = "api_spec") -> list:
    """Load markdown files from directory into RAG document format."""
    if not md_dir.exists():
        logger.warning(f"MD directory not found: {md_dir}")
        return []

    documents = []
    for fpath in sorted(md_dir.glob("*.md")):
        try:
            content = fpath.read_text(encoding="utf-8", errors="replace")
            if not content.strip():
                continue
            doc_id = f"api_spec:{fpath.stem}"
            documents.append({
                "content": content,
                "url": f"internal://api_spec/{fpath.stem}",
                "title": f"API {fpath.stem.replace('_', ' ').title()}",
                "source": source,
                "format": "markdown",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "images": [],
            })
        except Exception as e:
            logger.warning(f"Failed to load {fpath}: {e}")

    return documents


async def ingest_api_specs_from_schema(schema_path: Path) -> int:
    """Ingest API specs from schema JSON."""
    from app.scripts.convert_schema_to_rag import convert_schema_to_documents

    if not schema_path.exists():
        logger.warning(f"Schema not found: {schema_path}")
        return 0

    documents = convert_schema_to_documents(str(schema_path))
    if not documents:
        return 0

    now = datetime.now(timezone.utc).isoformat()
    for doc in documents:
        doc["timestamp"] = now
        doc["images"] = doc.get("images", [])

    from app.services.postgres_service import postgres_service
    if not postgres_service.pool:
        await postgres_service.initialize()
    if not postgres_service.pool:
        raise RuntimeError("PostgreSQL unavailable")

    ids = await postgres_service.add_documents(documents)
    logger.info(f"✅ Ingested {len(ids)} API spec documents from schema")
    return len(ids)


async def ingest_md_files_from_dir(md_dir: Path) -> int:
    """Ingest markdown files from directory."""
    documents = load_md_files_from_dir(md_dir)
    if not documents:
        return 0

    from app.services.postgres_service import postgres_service
    if not postgres_service.pool:
        await postgres_service.initialize()
    if not postgres_service.pool:
        raise RuntimeError("PostgreSQL unavailable")

    ids = await postgres_service.add_documents(documents)
    logger.info(f"✅ Ingested {len(ids)} md documents from {md_dir}")
    return len(ids)


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
) -> dict:
    """
    Full retrain pipeline.
    Returns: {scraped: N, api_specs: N, md_files: N, total: N}
    """
    schema_path = schema_path or DEFAULT_SCHEMA_PATH
    md_dir = md_dir or DEFAULT_MD_DIR

    stats = {"scraped": 0, "api_specs": 0, "md_files": 0, "total": 0}

    # 1. Clear (via API or direct)
    if clear:
        if api_base:
            await clear_via_api(api_base)
            # After API clear, we need postgres for ingest - ensure initialised
            from app.services.postgres_service import postgres_service
            if not postgres_service.pool:
                await postgres_service.initialize()
        else:
            await clear_knowledge_base(use_truncate=use_truncate)

    # 2. Bulk scrape (optional, via API or direct)
    if base_url:
        if api_base:
            stats["scraped"] = await run_bulk_scrape_via_api(
                api_base, base_url, max_depth, max_urls, seed_urls
            )
            # API returns immediately; scrape runs in background. Wait for completion if scrape_only.
            if scrape_only:
                logger.info("⏳ Scrape running in background on server. Check server logs for progress.")
                stats["total"] = stats["scraped"]
                return stats
            # Give backend time to start storing before we ingest
            logger.info("⏳ Waiting 10s for scrape to progress before ingesting API/md...")
            await asyncio.sleep(10)
        else:
            stats["scraped"] = await run_bulk_scrape(base_url, max_depth, max_urls, seed_urls)
            if scrape_only:
                stats["total"] = stats["scraped"]
                return stats

    # 3. Ingest API specs from schema
    if not scrape_only and ingest_api_specs:
        stats["api_specs"] = await ingest_api_specs_from_schema(schema_path)

    # 4. Ingest md files from directory (includes manually added like api_engagement_list)
    if not scrape_only and ingest_md_files:
        stats["md_files"] = await ingest_md_files_from_dir(md_dir)

    stats["total"] = stats["scraped"] + stats["api_specs"] + stats["md_files"]
    return stats


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Retrain RAG: clear, bulk scrape, ingest API specs + md files"
    )
    parser.add_argument("--no-clear", action="store_true", help="Do not clear KB (append mode)")
    parser.add_argument("--drop-table", action="store_true", help="Use DROP TABLE instead of TRUNCATE")
    parser.add_argument("--base-url", type=str, help="Base URL for bulk scrape (e.g. https://docs.example.com)")
    parser.add_argument("--scrape-only", action="store_true", help="Only run bulk scrape, skip API/md ingest")
    parser.add_argument("--schema-path", type=Path, default=DEFAULT_SCHEMA_PATH, help="Path to resource_schema.json")
    parser.add_argument("--md-dir", type=Path, default=DEFAULT_MD_DIR, help="Directory with .md files")
    parser.add_argument("--max-depth", type=int, default=15, help="Scrape discovery depth (1-20)")
    parser.add_argument("--max-urls", type=int, default=2000, help="Max URLs to scrape (0 = unlimited)")
    parser.add_argument("--no-limit", action="store_true", help="No URL limit (crawl entire site)")
    parser.add_argument(
        "--seed-urls",
        type=str,
        nargs="+",
        help="Extra URLs to always scrape (e.g. https://.../docs/zones https://.../docs/businessUnit)",
    )
    parser.add_argument("--api-base", type=str, help="Hit backend APIs (e.g. http://localhost:8000)")
    parser.add_argument("--no-api-specs", action="store_true", help="Skip API spec ingestion from schema")
    parser.add_argument("--no-md-files", action="store_true", help="Skip md files ingestion from directory")
    args = parser.parse_args()

    # Build ingest flags from negated args
    ingest_api_specs = not args.no_api_specs
    ingest_md_files = not args.no_md_files

    max_urls = 0 if args.no_limit else args.max_urls

    stats = asyncio.run(retrain(
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
        ingest_api_specs=ingest_api_specs,
        ingest_md_files=ingest_md_files,
    ))

    logger.info(
        f"🎉 Retrain complete: scraped={stats['scraped']}, "
        f"api_specs={stats['api_specs']}, md_files={stats['md_files']}, total={stats['total']}"
    )
    return 0 if stats["total"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
