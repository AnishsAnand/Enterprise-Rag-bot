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
    Convert all files inside metadata/api_spec_chunks
    to RAG documents and ingest.

    Returns: Number of documents ingested.
    """

    if not schema_path:
        schema_path = (
            Path(__file__).resolve().parent.parent.parent
            / "metadata"
            / "api_spec_chunks"
        )

    schema_path = Path(schema_path)

    if not schema_path.exists() or not schema_path.is_dir():
        logger.error(f"Directory not found: {schema_path}")
        return 0

    # 🔥 Get ALL files recursively
    all_files = [
        f for f in schema_path.rglob("*")
        if f.is_file() and not f.name.startswith(".")
    ]

    if not all_files:
        logger.warning(f"No files found in {schema_path}")
        return 0

    logger.info(f"Found {len(all_files)} files in {schema_path}")

    all_documents = []

    for file_path in all_files:
        logger.info(f"Processing {file_path}")

        try:
            # -------- If JSON schema file → use converter --------
            if file_path.suffix.lower() == ".json":
                documents = convert_schema_to_documents(str(file_path))
                if documents:
                    all_documents.extend(documents)

        # -------- If Markdown → ingest directly --------
            elif file_path.suffix.lower() in [".md", ".txt"]:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                all_documents.append({
                "content": content,
                "url": f"internal://api_spec_chunks/{file_path.name}",
                "title": file_path.stem,
                "source": "api_spec",
                "format": "markdown",
                "timestamp": None,
                "images": [],
                })

            else:
                logger.info(f"Skipping unsupported file type: {file_path.name}")

        except Exception as e:
            logger.error(f"Failed processing {file_path}: {e}")

    # Add timestamp
    now = datetime.now(timezone.utc).isoformat()
    for doc in all_documents:
        doc["timestamp"] = now
        doc["images"] = doc.get("images", [])

    from app.services.postgres_service import postgres_service

    if not postgres_service.pool:
        await postgres_service.initialize()

    if not postgres_service.pool:
        logger.error("PostgreSQL not available.")
        return 0

    logger.info(f"Ingesting {len(all_documents)} documents...")
    ids = await postgres_service.add_documents(all_documents)
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
