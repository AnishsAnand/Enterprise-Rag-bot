-- init_postgres.sql - PRODUCTION DATABASE INITIALIZATION
-- For Enterprise RAG Bot with pgvector
-- Database: enterprise_rag (NOT ragbot_db)
-- Last Updated: 2026-02-18
--
-- NOTE: Most tables are created by SQLAlchemy ORM on application startup.
-- This script handles:
--   1. pgvector extension installation
--   2. enterprise_rag table (vector storage) with proper indexes
--   3. Permissions setup
--
-- Tables created by SQLAlchemy (do NOT duplicate here):
--   - users, user_sessions, documents, document_chunks
--   - rag_queries, knowledge_bases, audit_logs
--   - chats, chat_files, folders, tags
--   - conversation_sessions, user_context_preferences

-- ============================================================================
-- STEP 1: Enable pgvector extension
-- ============================================================================
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================================
-- STEP 2: Create enterprise_rag table (main vector storage)
-- ============================================================================
-- This table stores document embeddings for RAG retrieval
-- Embedding dimension: 4096 (Voyage AI voyage-3 model)

CREATE TABLE IF NOT EXISTS enterprise_rag (
    id VARCHAR(100) PRIMARY KEY,
    embedding vector(4096),
    content TEXT NOT NULL,
    content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    url VARCHAR(2000),
    title VARCHAR(500),
    format VARCHAR(100),
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    source VARCHAR(100),
    content_length INTEGER,
    word_count INTEGER,
    image_count INTEGER DEFAULT 0,
    has_images BOOLEAN DEFAULT FALSE,
    domain VARCHAR(500),
    content_hash BIGINT,
    images_json JSONB DEFAULT '[]'::jsonb,
    key_terms TEXT[],
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- STEP 3: Create indexes for enterprise_rag
-- ============================================================================

-- HNSW index for fast vector similarity search (L2 distance)
-- Parameters: m=16 (connections per layer), ef_construction=200 (build quality)
CREATE INDEX IF NOT EXISTS enterprise_rag_embedding_hnsw_idx 
ON enterprise_rag 
USING hnsw (embedding vector_l2_ops)
WITH (m = 16, ef_construction = 200);

-- GIN index for full-text search on content
CREATE INDEX IF NOT EXISTS enterprise_rag_content_tsv_idx 
ON enterprise_rag 
USING gin (content_tsv);

-- B-tree indexes for common query patterns
CREATE INDEX IF NOT EXISTS enterprise_rag_url_idx ON enterprise_rag(url);
CREATE INDEX IF NOT EXISTS enterprise_rag_timestamp_idx ON enterprise_rag(timestamp);
CREATE INDEX IF NOT EXISTS enterprise_rag_source_idx ON enterprise_rag(source);
CREATE INDEX IF NOT EXISTS enterprise_rag_domain_idx ON enterprise_rag(domain);

-- ============================================================================
-- STEP 4: Grant permissions
-- ============================================================================
-- Note: Database and user are created by Docker environment variables
-- POSTGRES_USER=ragbot, POSTGRES_DB=enterprise_rag

-- Grant all privileges on current and future tables
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO ragbot;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO ragbot;

-- Set default privileges for future tables (created by SQLAlchemy)
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO ragbot;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO ragbot;

-- ============================================================================
-- STEP 5: Verify setup
-- ============================================================================
DO $$
BEGIN
    RAISE NOTICE '===========================================';
    RAISE NOTICE 'PostgreSQL initialization complete!';
    RAISE NOTICE '===========================================';
    RAISE NOTICE 'Database: enterprise_rag';
    RAISE NOTICE 'User: ragbot';
    RAISE NOTICE 'pgvector version: %', (SELECT extversion FROM pg_extension WHERE extname = 'vector');
    RAISE NOTICE '';
    RAISE NOTICE 'Tables created by this script:';
    RAISE NOTICE '  - enterprise_rag (vector storage)';
    RAISE NOTICE '';
    RAISE NOTICE 'Tables created by SQLAlchemy on app startup:';
    RAISE NOTICE '  - users, user_sessions, documents, document_chunks';
    RAISE NOTICE '  - rag_queries, knowledge_bases, audit_logs';
    RAISE NOTICE '  - chats, chat_files, folders, tags';
    RAISE NOTICE '  - conversation_sessions, user_context_preferences';
    RAISE NOTICE '===========================================';
END $$;

-- Show extension status
SELECT 'pgvector extension' as component, extname, extversion as version 
FROM pg_extension 
WHERE extname = 'vector';
