-- PostgreSQL initialization script for Enterprise RAG Bot
-- Creates pgvector extension and initial schema

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS pgvector;

-- Create schema for pgvector search
CREATE SCHEMA IF NOT EXISTS vector_store;

-- Create documents table with vector support
CREATE TABLE IF NOT EXISTS vector_store.documents (
    id SERIAL PRIMARY KEY,
    document_id TEXT NOT NULL,
    title VARCHAR(500) NOT NULL,
    source_url VARCHAR(500),
    chunk_text TEXT NOT NULL,
    embedding VECTOR(1536),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(document_id)
);

-- Create indexes for performance
CREATE INDEX idx_documents_embedding ON vector_store.documents USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX idx_documents_created_at ON vector_store.documents(created_at);
CREATE INDEX idx_documents_title ON vector_store.documents(title);

-- Grant permissions
GRANT ALL PRIVILEGES ON SCHEMA vector_store TO ragbot;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA vector_store TO ragbot;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA vector_store TO ragbot;
