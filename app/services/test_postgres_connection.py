"""
Test PostgreSQL connection and basic operations.
Run this AFTER setting up PostgreSQL to verify everything works.
"""

import asyncio
import sys
from datetime import datetime

# Add app directory to path
sys.path.insert(0, '.')

from app.services.postgres_service import postgres_service


async def test_connection():
    """Test basic PostgreSQL connection."""
    print("=" * 70)
    print("🧪 Testing PostgreSQL Connection")
    print("=" * 70)
    print()
    
    try:
        print("1️⃣  Initializing PostgreSQL service...")
        await postgres_service.initialize()
        
        if not postgres_service._connection_established:
            print("❌ Failed to establish connection")
            return False
        
        print("✅ Connection established!")
        print()
        
        print("2️⃣  Getting collection stats...")
        stats = await postgres_service.get_collection_stats()
        
        print(f"   📊 Status: {stats.get('status')}")
        print(f"   📊 Table: {stats.get('collection_name')}")
        print(f"   📊 Documents: {stats.get('document_count')}")
        print(f"   📊 Embedding dimension: {stats.get('embedding_dimension')}")
        print(f"   📊 Indexes: {len(stats.get('indexes', []))}")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_document_operations():
    """Test document insertion and search."""
    print("=" * 70)
    print("🧪 Testing Document Operations")
    print("=" * 70)
    print()
    
    try:
        # Ensure initialized
        if not postgres_service._connection_established:
            await postgres_service.initialize()
        
        # Test document insertion
        print("1️⃣  Testing document insertion...")
        test_docs = [
            {
                "content": "PostgreSQL is a powerful open-source relational database with excellent vector search capabilities through pgvector extension.",
                "url": "https://example.com/postgres-test-1",
                "title": "PostgreSQL Vector Search",
                "format": "text",
                "source": "connection_test",
            },
            {
                "content": "Vector databases are essential for modern AI applications including RAG systems and semantic search.",
                "url": "https://example.com/postgres-test-2",
                "title": "Vector Databases for AI",
                "format": "text",
                "source": "connection_test",
            },
            {
                "content": "HNSW indexing provides fast approximate nearest neighbor search for high-dimensional vectors.",
                "url": "https://example.com/postgres-test-3",
                "title": "HNSW Indexing",
                "format": "text",
                "source": "connection_test",
            }
        ]
        
        ids = await postgres_service.add_documents(test_docs)
        
        if ids and len(ids) == len(test_docs):
            print(f"✅ Inserted {len(ids)} test documents")
            print(f"   Document IDs: {ids[:2]}...")
        else:
            print(f"⚠️  Expected {len(test_docs)} IDs, got {len(ids) if ids else 0}")
            return False
        
        print()
        
        # Test search
        print("2️⃣  Testing vector search...")
        search_queries = [
            "PostgreSQL database",
            "vector search",
            "HNSW algorithm"
        ]
        
        for query in search_queries:
            results = await postgres_service.search_documents(query, n_results=3)
            print(f"   🔍 Query: '{query}'")
            print(f"      Results: {len(results)}")
            
            if results:
                top_result = results[0]
                print(f"      Top result relevance: {top_result.get('relevance_score', 0):.3f}")
                print(f"      Content preview: {top_result.get('content', '')[:80]}...")
            print()
        
        # Get updated stats
        print("3️⃣  Updated statistics...")
        stats = await postgres_service.get_collection_stats()
        print(f"   📊 Total documents: {stats.get('document_count')}")
        print(f"   📊 Table size: {stats.get('table_size', 'unknown')}")
        print()
        
        print("✅ Document operations test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Document operations test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_performance():
    """Test search performance."""
    print("=" * 70)
    print("🧪 Testing Search Performance")
    print("=" * 70)
    print()
    
    try:
        if not postgres_service._connection_established:
            await postgres_service.initialize()
        
        import time
        
        test_queries = [
            "database optimization",
            "vector embeddings",
            "semantic search algorithms",
            "machine learning infrastructure",
            "data indexing strategies"
        ]
        
        print("Running 5 search queries...")
        start_time = time.time()
        
        for i, query in enumerate(test_queries, 1):
            query_start = time.time()
            results = await postgres_service.search_documents(query, n_results=10)
            query_time = (time.time() - query_start) * 1000
            
            print(f"   Query {i}: {query_time:.0f}ms ({len(results)} results)")
        
        total_time = (time.time() - start_time) * 1000
        avg_time = total_time / len(test_queries)
        
        print()
        print(f"📊 Performance Summary:")
        print(f"   Total time: {total_time:.0f}ms")
        print(f"   Average per query: {avg_time:.0f}ms")
        
        if avg_time < 1000:
            print(f"   ✅ Performance: Excellent (<1s)")
        elif avg_time < 3000:
            print(f"   ✅ Performance: Good (<3s)")
        else:
            print(f"   ⚠️  Performance: Slow (>3s) - Consider index optimization")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def cleanup_test_data():
    """Clean up test documents."""
    print()
    print("🧹 Cleaning up test data...")
    
    try:
        if not postgres_service.pool:
            return
        
        async with postgres_service.pool.acquire() as conn:
            delete_sql = f"""
            DELETE FROM {postgres_service.table_name}
            WHERE source = 'connection_test';
            """
            result = await conn.execute(delete_sql)
            print(f"✅ Cleaned up test data")
            
    except Exception as e:
        print(f"⚠️  Cleanup warning: {e}")


async def main():
    """Run all tests."""
    print()
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "PostgreSQL Connection Test Suite" + " " * 21 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    results = []
    
    # Test 1: Connection
    connection_ok = await test_connection()
    results.append(("Connection", connection_ok))
    
    if not connection_ok:
        print()
        print("❌ Connection test failed. Please check:")
        print("   1. PostgreSQL is running (docker-compose ps)")
        print("   2. .env file has correct credentials")
        print("   3. pgvector extension is installed")
        return
    
    # Test 2: Document Operations
    operations_ok = await test_document_operations()
    results.append(("Document Operations", operations_ok))
    
    # Test 3: Performance
    if operations_ok:
        performance_ok = await test_performance()
        results.append(("Performance", performance_ok))
    
    # Cleanup
    await cleanup_test_data()
    
    # Close connection
    await postgres_service.close()
    
    # Summary
    print()
    print("=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    
    all_passed = all(result[1] for result in results)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {test_name:.<40} {status}")
    
    print("=" * 70)
    
    if all_passed:
        print()
        print("🎉 All tests passed! PostgreSQL is ready for production.")
        print()
        print("Next steps:")
        print("   1. Run migration script: python migrate_milvus_to_postgres.py")
        print("   2. Update code imports from milvus_service to postgres_service")
        print("   3. Test your application thoroughly")
        print()
    else:
        print()
        print("❌ Some tests failed. Please review the errors above.")
        print()
    
    print(f"⏰ Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
