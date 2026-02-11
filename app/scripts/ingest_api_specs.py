#!/usr/bin/env python3
"""
Phase 1: Ingest API spec documents into RAG.

Converts resource_schema.json to markdown chunks and stores them
in the enterprise_rag table with source="api_spec".

Usage:
  python -m app.scripts.ingest_api_specs [--schema-path PATH]

Requires: PostgreSQL running, POSTGRES_* env vars set.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.scripts.convert_schema_to_rag import convert_schema_to_documents

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def ingest_api_specs(schema_path: str = None) -> int:
    """
    Convert resource_schema.json to RAG documents and ingest.

    Returns: Number of documents ingested.
    """
    if not schema_path:
        schema_path = str(
            Path(__file__).resolve().parent.parent / "config" / "resource_schema.json"
        )

    if not os.path.exists(schema_path):
        logger.error(f"Schema not found: {schema_path}")
        return 0

    # Convert to documents
    documents = convert_schema_to_documents(schema_path)
    if not documents:
        logger.warning("No documents to ingest")
        return 0

    # Add timestamp to each
    now = datetime.now(timezone.utc).isoformat()
    for doc in documents:
        doc["timestamp"] = now
        doc["images"] = doc.get("images", [])

    # Import postgres service and ensure it's initialized
    from app.services.postgres_service import postgres_service

    # Ensure pool is connected
    if not postgres_service.pool:
        await postgres_service.initialize()
    if not postgres_service.pool:
        logger.error("PostgreSQL not available. Set POSTGRES_HOST, POSTGRES_DB, etc.")
        return 0

    logger.info(f"Ingesting {len(documents)} API spec documents...")
    ids = await postgres_service.add_documents(documents)
    logger.info(f"Successfully ingested {len(ids)} documents")
    return len(ids)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ingest API specs from resource_schema.json into RAG")
    parser.add_argument("--schema-path", default=None, help="Path to resource_schema.json")
    args = parser.parse_args()

    count = asyncio.run(ingest_api_specs(args.schema_path))
    return 0 if count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
