"""
Migration script to transfer data from Milvus to PostgreSQL.
Run this BEFORE switching to postgres_service in production.
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import List, Dict, Any

# Import BOTH services for migration
from app.services.milvus_service import milvus_service
from app.services.postgres_service import postgres_service
from app.services.ai_service import ai_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def export_from_milvus(batch_size: int = 100) -> List[Dict[str, Any]]:
    """Export all documents from Milvus."""
    logger.info("🔄 Exporting data from Milvus...")
    
    try:
        # Initialize Milvus
        await milvus_service.initialize()
        
        # Get collection stats
        stats = await milvus_service.get_collection_stats()
        total_docs = stats.get("document_count", 0)
        logger.info(f"📊 Total documents in Milvus: {total_docs}")
        
        if total_docs == 0:
            logger.warning("⚠️ No documents found in Milvus")
            return []
        
        # Search with high limit to get all documents
        # Note: Milvus requires a query, so we use a broad search
        all_results = await milvus_service.search_documents(
            query="*",  # Wildcard search
            n_results=min(total_docs, 10000)  # Limit for safety
        )
        
        if not all_results:
            logger.warning("⚠️ No results returned from Milvus search")
            return []
        
        # Convert Milvus results to document format
        documents = []
        for result in all_results:
            metadata = result.get("metadata", {})
            
            doc = {
                "content": result.get("content", ""),
                "url": metadata.get("url", ""),
                "title": metadata.get("title", ""),
                "format": metadata.get("format", "text"),
                "timestamp": metadata.get("timestamp", datetime.now().isoformat()),
                "source": metadata.get("source", "milvus_migration"),
                "images": metadata.get("images", []),
            }
            
            # Only add documents with content
            if doc["content"] and len(doc["content"].strip()) > 10:
                documents.append(doc)
        
        logger.info(f"✅ Exported {len(documents)} documents from Milvus")
        return documents
        
    except Exception as e:
        logger.exception(f"❌ Error exporting from Milvus: {e}")
        return []


async def import_to_postgres(documents: List[Dict[str, Any]], batch_size: int = 50) -> int:
    """Import documents to PostgreSQL in batches."""
    logger.info(f"🔄 Importing {len(documents)} documents to PostgreSQL...")
    
    try:
        # Initialize PostgreSQL
        await postgres_service.initialize()
        
        total_imported = 0
        
        # Process in batches to avoid overwhelming the system
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(documents) + batch_size - 1) // batch_size
            
            logger.info(f"📦 Processing batch {batch_num}/{total_batches} ({len(batch)} docs)...")
            
            try:
                ids = await postgres_service.add_documents(batch)
                total_imported += len(ids)
                logger.info(f"✅ Batch {batch_num} imported: {len(ids)} documents")
                
            except Exception as e:
                logger.error(f"❌ Batch {batch_num} failed: {e}")
                continue
            
            # Small delay between batches to avoid overwhelming the system
            if i + batch_size < len(documents):
                await asyncio.sleep(1)
        
        logger.info(f"✅ Total imported to PostgreSQL: {total_imported}/{len(documents)}")
        return total_imported
        
    except Exception as e:
        logger.exception(f"❌ Error importing to PostgreSQL: {e}")
        return 0


async def verify_migration() -> bool:
    """Verify that migration was successful."""
    logger.info("🔍 Verifying migration...")
    
    try:
        # Get stats from both systems
        milvus_stats = await milvus_service.get_collection_stats()
        postgres_stats = await postgres_service.get_collection_stats()
        
        milvus_count = milvus_stats.get("document_count", 0)
        postgres_count = postgres_stats.get("document_count", 0)
        
        logger.info(f"📊 Milvus documents: {milvus_count}")
        logger.info(f"📊 PostgreSQL documents: {postgres_count}")
        
        # Test search functionality
        test_query = "test search query"
        logger.info(f"🔍 Testing search with query: '{test_query}'")
        
        milvus_results = await milvus_service.search_documents(test_query, n_results=5)
        postgres_results = await postgres_service.search_documents(test_query, n_results=5)
        
        logger.info(f"📊 Milvus search returned: {len(milvus_results)} results")
        logger.info(f"📊 PostgreSQL search returned: {len(postgres_results)} results")
        
        # Consider migration successful if counts are close (within 5%)
        if postgres_count > 0:
            difference_pct = abs(milvus_count - postgres_count) / max(milvus_count, 1) * 100
            
            if difference_pct <= 5:
                logger.info(f"✅ Migration verified successfully! Difference: {difference_pct:.2f}%")
                return True
            else:
                logger.warning(f"⚠️ Document count difference: {difference_pct:.2f}%")
                return False
        else:
            logger.error("❌ PostgreSQL has no documents after migration")
            return False
            
    except Exception as e:
        logger.exception(f"❌ Verification failed: {e}")
        return False


async def save_backup(documents: List[Dict[str, Any]], filename: str = "milvus_backup.json"):
    """Save documents to JSON backup file."""
    logger.info(f"💾 Saving backup to {filename}...")
    
    try:
        backup_data = {
            "timestamp": datetime.now().isoformat(),
            "document_count": len(documents),
            "documents": documents
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(backup_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Backup saved: {filename}")
        
    except Exception as e:
        logger.exception(f"❌ Failed to save backup: {e}")


async def load_backup(filename: str = "milvus_backup.json") -> List[Dict[str, Any]]:
    """Load documents from JSON backup file."""
    logger.info(f"📂 Loading backup from {filename}...")
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            backup_data = json.load(f)
        
        documents = backup_data.get("documents", [])
        logger.info(f"✅ Loaded {len(documents)} documents from backup")
        return documents
        
    except Exception as e:
        logger.exception(f"❌ Failed to load backup: {e}")
        return []


async def main():
    """Main migration workflow."""
    logger.info("=" * 70)
    logger.info("🚀 Starting Milvus → PostgreSQL Migration")
    logger.info("=" * 70)
    
    start_time = datetime.now()
    
    # Step 1: Export from Milvus
    logger.info("\n📤 STEP 1: Export from Milvus")
    documents = await export_from_milvus()
    
    if not documents:
        logger.error("❌ No documents to migrate. Exiting.")
        return
    
    # Step 2: Save backup
    logger.info("\n💾 STEP 2: Save backup")
    await save_backup(documents)
    
    # Step 3: Import to PostgreSQL
    logger.info("\n📥 STEP 3: Import to PostgreSQL")
    imported_count = await import_to_postgres(documents)
    
    if imported_count == 0:
        logger.error("❌ Migration failed - no documents imported")
        return
    
    # Step 4: Verify migration
    logger.info("\n✅ STEP 4: Verify migration")
    success = await verify_migration()
    
    # Step 5: Cleanup
    logger.info("\n🧹 STEP 5: Cleanup")
    await milvus_service.close()
    await postgres_service.close()
    
    # Summary
    duration = (datetime.now() - start_time).total_seconds()
    logger.info("\n" + "=" * 70)
    logger.info("📊 MIGRATION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"✅ Status: {'SUCCESS' if success else 'FAILED'}")
    logger.info(f"📄 Documents exported: {len(documents)}")
    logger.info(f"📄 Documents imported: {imported_count}")
    logger.info(f"⏱️  Duration: {duration:.2f} seconds")
    logger.info("=" * 70)
    
    if success:
        logger.info("\n🎉 Migration completed successfully!")
        logger.info("\n📝 Next steps:")
        logger.info("1. Update .env file with PostgreSQL configuration")
        logger.info("2. Update import statements in code files")
        logger.info("3. Test the application thoroughly")
        logger.info("4. Keep Milvus backup for rollback if needed")
    else:
        logger.error("\n❌ Migration had issues. Review logs and retry.")


if __name__ == "__main__":
    asyncio.run(main())
