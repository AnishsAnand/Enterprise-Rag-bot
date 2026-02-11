#!/usr/bin/env python3
"""
Test script for Phase 2 RAG-driven intent flow.

Run from project root with deps installed. Requires:
- PostgreSQL + enterprise_rag with ingested API specs (python3 -m app.scripts.ingest_api_specs)
- LLM endpoint (OPENAI_API_BASE or similar)
- POSTGRES_* env vars

Usage:
  python3 -m app.scripts.test_rag_intent "list clusters"
  python3 -m app.scripts.test_rag_intent "show me load balancers"
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


async def test_rag_search():
    """Test RAG API spec search only (no LLM)."""
    from app.services.postgres_service import postgres_service

    await postgres_service.initialize()
    if not postgres_service.pool:
        print("❌ PostgreSQL not available. Set POSTGRES_HOST, POSTGRES_DB, etc.")
        return False

    query = "list kubernetes clusters API"
    results = await postgres_service.search_api_specs(query, n_results=3)
    print(f"\n📚 RAG search for '{query}': {len(results)} results\n")
    for i, r in enumerate(results):
        content = (r.get("content") or "")[:400]
        score = r.get("relevance_score", 0)
        title = r.get("metadata", {}).get("title", "")
        print(f"--- Result {i+1} (score={score:.3f}) {title} ---")
        print(content)
        print()
    return True


async def test_intent_full(user_input: str):
    """Test full intent flow (RAG + LLM)."""
    from app.agents.intent_agent import IntentAgent

    agent = IntentAgent()
    result = await agent.execute(user_input, context={"session_id": "test-session"})

    print(f"\n🎯 Intent test: '{user_input}'")
    print("-" * 50)
    print("intent_detected:", result.get("intent_detected"))
    idata = result.get("intent_data", {})
    print("resource_type:", idata.get("resource_type"))
    print("operation:", idata.get("operation"))
    print("required_params:", idata.get("required_params"))
    print("optional_params:", idata.get("optional_params"))
    print("has_api_spec:", bool(idata.get("api_spec")))
    if idata.get("api_spec"):
        print("api_spec length:", len(idata["api_spec"]))
    print()

    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 -m app.scripts.test_rag_intent <query>")
        print("   or: python3 -m app.scripts.test_rag_intent --rag-only")
        print()
        print("Examples:")
        print('  python3 -m app.scripts.test_rag_intent "list clusters"')
        print('  python3 -m app.scripts.test_rag_intent "show load balancers"')
        print("  python3 -m app.scripts.test_rag_intent --rag-only  # Test RAG search only")
        return 1

    arg = sys.argv[1]
    if arg == "--rag-only":
        ok = asyncio.run(test_rag_search())
        return 0 if ok else 1

    asyncio.run(test_intent_full(arg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
