CREATE TABLE IF NOT EXISTS enterprise_rag (
    id VARCHAR(100) PRIMARY KEY,
    embedding VECTOR(4096) NOT NULL,
    content TEXT NOT NULL,

    content_tsv tsvector
        GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,

    url VARCHAR(2000),
    title VARCHAR(500),
    format VARCHAR(100),
    source VARCHAR(100),
    domain VARCHAR(500),

    content_length INTEGER,
    word_count INTEGER,
    image_count INTEGER DEFAULT 0,
    has_images BOOLEAN DEFAULT FALSE,

    images_json JSONB DEFAULT '[]'::jsonb,
    key_terms TEXT[],

    content_hash BIGINT,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
