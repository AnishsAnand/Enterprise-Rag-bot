CREATE INDEX IF NOT EXISTS enterprise_rag_embedding_hnsw_idx
ON enterprise_rag
USING hnsw (embedding vector_l2_ops)
WITH (m = 16, ef_construction = 200);

CREATE INDEX IF NOT EXISTS enterprise_rag_url_idx ON enterprise_rag(url);
CREATE INDEX IF NOT EXISTS enterprise_rag_source_idx ON enterprise_rag(source);
CREATE INDEX IF NOT EXISTS enterprise_rag_domain_idx ON enterprise_rag(domain);

CREATE INDEX IF NOT EXISTS enterprise_rag_tsv_idx
ON enterprise_rag USING GIN (content_tsv);
