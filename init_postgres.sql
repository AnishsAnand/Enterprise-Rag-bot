-- init_postgres.sql - PRODUCTION DATABASE INITIALIZATION
-- Place this file in project root, it will be mounted to postgres container

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Fix users table schema (add missing columns if they don't exist)
DO $$ 
BEGIN
    -- Add full_name if missing
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name='users' AND column_name='full_name'
    ) THEN
        ALTER TABLE users ADD COLUMN full_name VARCHAR(255);
    END IF;
    
    -- Add avatar_url if missing
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name='users' AND column_name='avatar_url'
    ) THEN
        ALTER TABLE users ADD COLUMN avatar_url VARCHAR(500);
    END IF;
    
    -- Add bio if missing
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name='users' AND column_name='bio'
    ) THEN
        ALTER TABLE users ADD COLUMN bio TEXT;
    END IF;
    
    -- Add theme if missing
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name='users' AND column_name='theme'
    ) THEN
        ALTER TABLE users ADD COLUMN theme VARCHAR(50) DEFAULT 'light';
    END IF;
    
    -- Add language if missing
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name='users' AND column_name='language'
    ) THEN
        ALTER TABLE users ADD COLUMN language VARCHAR(10) DEFAULT 'en';
    END IF;
    
    -- Add timezone if missing
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name='users' AND column_name='timezone'
    ) THEN
        ALTER TABLE users ADD COLUMN timezone VARCHAR(50) DEFAULT 'UTC';
    END IF;
    
    -- Add notifications_enabled if missing
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name='users' AND column_name='notifications_enabled'
    ) THEN
        ALTER TABLE users ADD COLUMN notifications_enabled BOOLEAN DEFAULT TRUE;
    END IF;
    
    -- Add email_notifications if missing
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name='users' AND column_name='email_notifications'
    ) THEN
        ALTER TABLE users ADD COLUMN email_notifications BOOLEAN DEFAULT FALSE;
    END IF;
    
    -- Add last_login if missing
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name='users' AND column_name='last_login'
    ) THEN
        ALTER TABLE users ADD COLUMN last_login TIMESTAMP;
    END IF;
    
    -- Add login_count if missing
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name='users' AND column_name='login_count'
    ) THEN
        ALTER TABLE users ADD COLUMN login_count INTEGER DEFAULT 0;
    END IF;
    
    -- Add failed_login_attempts if missing
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name='users' AND column_name='failed_login_attempts'
    ) THEN
        ALTER TABLE users ADD COLUMN failed_login_attempts INTEGER DEFAULT 0;
    END IF;
    
    -- Add locked_until if missing
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name='users' AND column_name='locked_until'
    ) THEN
        ALTER TABLE users ADD COLUMN locked_until TIMESTAMP;
    END IF;
END $$;

-- Create vector table for enterprise_rag if it doesn't exist
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

-- Create HNSW index for vector similarity search
CREATE INDEX IF NOT EXISTS enterprise_rag_embedding_hnsw_idx 
ON enterprise_rag 
USING hnsw (embedding vector_l2_ops)
WITH (m = 16, ef_construction = 200);

-- Create other useful indexes
CREATE INDEX IF NOT EXISTS enterprise_rag_url_idx ON enterprise_rag(url);
CREATE INDEX IF NOT EXISTS enterprise_rag_timestamp_idx ON enterprise_rag(timestamp);
CREATE INDEX IF NOT EXISTS enterprise_rag_source_idx ON enterprise_rag(source);

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE ragbot_db TO ragbot;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO ragbot;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO ragbot;

-- Verify setup
SELECT 'PostgreSQL initialization complete!' as status;
SELECT extname FROM pg_extension WHERE extname = 'vector';