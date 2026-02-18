# Enterprise RAG Bot - Database Schema Documentation

**Generated from live PostgreSQL container:** `enterprise-rag-postgres`  
**Database:** `enterprise_rag`  
**PostgreSQL Version:** pgvector/pgvector:0.7.3-pg16  
**Last Updated:** 2026-02-18

---

## Overview

The database contains **14 tables**, **7 sequences**, and **59 indexes**. All functions are provided by the pgvector extension for vector similarity search operations.

### Extensions Installed

| Extension | Version | Description |
|-----------|---------|-------------|
| plpgsql | 1.0 | PL/pgSQL procedural language |
| vector | 0.7.3 | Vector data type with ivfflat and hnsw access methods |

---

## Tables

### 1. `users` (21 columns)

Core user authentication and profile table.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| id | integer | NOT NULL | nextval('users_id_seq') | Primary key |
| username | varchar(255) | NOT NULL | | Unique login identifier |
| email | varchar(255) | NOT NULL | | Unique email address |
| hashed_password | varchar(255) | NOT NULL | | Bcrypt hashed password |
| full_name | varchar(255) | | | Display name |
| avatar_url | varchar(500) | | | Profile image URL |
| bio | text | | | User biography |
| role | varchar(6) | NOT NULL | | User role (admin/user/viewer/editor) |
| is_active | boolean | NOT NULL | | Account enabled status |
| is_verified | boolean | NOT NULL | | Email verification status |
| theme | varchar(50) | | | UI theme preference |
| language | varchar(10) | | | Language preference |
| timezone | varchar(50) | | | Timezone setting |
| notifications_enabled | boolean | NOT NULL | | Push notifications enabled |
| email_notifications | boolean | NOT NULL | | Email notifications enabled |
| last_login | timestamptz | | | Last login timestamp |
| login_count | integer | NOT NULL | | Total successful logins |
| failed_login_attempts | integer | NOT NULL | | Failed login counter |
| locked_until | timestamptz | | | Account lock expiry |
| created_at | timestamptz | NOT NULL | now() | Account creation time |
| updated_at | timestamptz | NOT NULL | now() | Last update time |

**Indexes:**
- `users_pkey` - PRIMARY KEY (id)
- `ix_users_username` - UNIQUE (username)
- `ix_users_email` - UNIQUE (email)
- `ix_users_role` - btree (role)
- `ix_users_is_active` - btree (is_active)
- `ix_users_created_at` - btree (created_at)

**Referenced By:**
- `audit_logs.user_id` → ON DELETE SET NULL
- `documents.owner_id` → ON DELETE CASCADE
- `knowledge_bases.owner_id` → ON DELETE CASCADE
- `rag_queries.user_id` → ON DELETE SET NULL
- `user_sessions.user_id` → ON DELETE CASCADE

---

### 2. `user_sessions` (9 columns)

Active user session tracking for security and audit.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| id | integer | NOT NULL | nextval('user_sessions_id_seq') | Primary key |
| user_id | integer | NOT NULL | | Foreign key to users |
| session_token | varchar(500) | NOT NULL | | Unique session identifier |
| ip_address | varchar(45) | | | Client IP (IPv4/IPv6) |
| user_agent | varchar(500) | | | Browser/client info |
| is_active | boolean | NOT NULL | | Session active status |
| created_at | timestamptz | NOT NULL | now() | Session start time |
| last_activity | timestamptz | NOT NULL | now() | Last activity timestamp |
| expires_at | timestamptz | NOT NULL | | Session expiry time |

**Indexes:**
- `user_sessions_pkey` - PRIMARY KEY (id)
- `ix_user_sessions_session_token` - UNIQUE (session_token)
- `ix_user_sessions_user_id` - btree (user_id)

**Foreign Keys:**
- `user_id` → `users(id)` ON DELETE CASCADE

---

### 3. `enterprise_rag` (19 columns)

Main vector storage table for RAG content with embeddings.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| id | varchar(100) | NOT NULL | | Primary key (content hash) |
| embedding | vector(4096) | | | Vector embedding (4096 dimensions) |
| content | text | NOT NULL | | Full text content |
| content_tsv | tsvector | | GENERATED (to_tsvector) | Full-text search vector |
| url | varchar(2000) | | | Source URL |
| title | varchar(500) | | | Content title |
| format | varchar(100) | | | Content format type |
| timestamp | timestamptz | | now() | Ingestion timestamp |
| source | varchar(100) | | | Content source identifier |
| content_length | integer | | | Character count |
| word_count | integer | | | Word count |
| image_count | integer | | 0 | Number of images |
| has_images | boolean | | false | Contains images flag |
| domain | varchar(500) | | | Source domain |
| content_hash | bigint | | | Content hash for deduplication |
| images_json | jsonb | | '[]'::jsonb | Image metadata array |
| key_terms | text[] | | | Extracted key terms |
| created_at | timestamptz | | now() | Creation timestamp |
| updated_at | timestamptz | | now() | Last update timestamp |

**Indexes:**
- `enterprise_rag_pkey` - PRIMARY KEY (id)
- `enterprise_rag_content_tsv_idx` - GIN (content_tsv) for full-text search

**Note:** The HNSW index for vector similarity search is NOT present in the live database. See [Issues](#issues) section.

---

### 4. `chats` (11 columns)

Chat conversation storage (Open WebUI compatible format).

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| id | varchar(36) | NOT NULL | | UUID primary key |
| user_id | varchar(255) | NOT NULL | | User identifier |
| title | text | NOT NULL | | Chat title |
| chat | json | NOT NULL | | Full chat history JSON |
| created_at | bigint | NOT NULL | | Unix timestamp (ms) |
| updated_at | bigint | NOT NULL | | Unix timestamp (ms) |
| share_id | text | | | Public share identifier |
| archived | boolean | NOT NULL | | Archived status |
| pinned | boolean | | | Pinned status |
| meta | json | NOT NULL | | Additional metadata |
| folder_id | text | | | Parent folder reference |

**Indexes:**
- `chats_pkey` - PRIMARY KEY (id)
- `chats_share_id_key` - UNIQUE (share_id)
- `idx_chat_folder_id` - btree (folder_id)
- `idx_chat_updated_user` - btree (updated_at, user_id)
- `idx_chat_user_archived` - btree (user_id, archived)
- `idx_chat_user_id` - btree (user_id)
- `idx_chat_user_pinned` - btree (user_id, pinned)
- `ix_chats_user_id` - btree (user_id)

**Referenced By:**
- `chat_files.chat_id` → ON DELETE CASCADE

---

### 5. `chat_files` (7 columns)

Files attached to chat messages.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| id | varchar(36) | NOT NULL | | UUID primary key |
| user_id | varchar(255) | NOT NULL | | User identifier |
| chat_id | varchar(36) | NOT NULL | | Parent chat reference |
| message_id | varchar(255) | | | Associated message ID |
| file_id | varchar(255) | NOT NULL | | File storage identifier |
| created_at | bigint | NOT NULL | | Unix timestamp (ms) |
| updated_at | bigint | NOT NULL | | Unix timestamp (ms) |

**Indexes:**
- `chat_files_pkey` - PRIMARY KEY (id)
- `idx_chat_file_chat_id` - btree (chat_id)
- `ix_chat_files_chat_id` - btree (chat_id)

**Foreign Keys:**
- `chat_id` → `chats(id)` ON DELETE CASCADE

---

### 6. `folders` (6 columns)

Hierarchical folder structure for organizing chats.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| id | varchar(36) | NOT NULL | | UUID primary key |
| name | varchar(255) | NOT NULL | | Folder name |
| user_id | varchar(255) | NOT NULL | | Owner user ID |
| parent_id | varchar(36) | | | Parent folder (null = root) |
| created_at | bigint | NOT NULL | | Unix timestamp (ms) |
| updated_at | bigint | NOT NULL | | Unix timestamp (ms) |

**Indexes:**
- `folders_pkey` - PRIMARY KEY (id)
- `idx_folder_parent_id` - btree (parent_id)
- `idx_folder_user_id` - btree (user_id)
- `ix_folders_user_id` - btree (user_id)

---

### 7. `tags` (4 columns)

User-defined tags for categorization.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| id | varchar(255) | NOT NULL | | Primary key |
| name | varchar(255) | NOT NULL | | Tag name |
| user_id | varchar(255) | NOT NULL | | Owner user ID |
| created_at | bigint | NOT NULL | | Unix timestamp (ms) |

**Indexes:**
- `tags_pkey` - PRIMARY KEY (id)
- `idx_tag_user_id` - btree (user_id)
- `ix_tags_user_id` - btree (user_id)

---

### 8. `documents` (16 columns)

Document metadata and processing status tracking.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| id | integer | NOT NULL | nextval('documents_id_seq') | Primary key |
| owner_id | integer | NOT NULL | | Foreign key to users |
| title | varchar(500) | NOT NULL | | Document title |
| description | text | | | Document description |
| file_path | varchar(500) | | | Local file path |
| file_size | integer | | | File size in bytes |
| file_type | varchar(50) | | | MIME type |
| source_url | varchar(500) | | | Original source URL |
| source_type | varchar(50) | | | Source type identifier |
| status | varchar(10) | NOT NULL | | Processing status |
| processing_error | text | | | Error message if failed |
| vector_ids | json | | | Associated vector IDs |
| chunk_count | integer | NOT NULL | | Number of chunks created |
| created_at | timestamptz | NOT NULL | now() | Creation timestamp |
| updated_at | timestamptz | NOT NULL | now() | Last update timestamp |
| processed_at | timestamptz | | | Processing completion time |

**Indexes:**
- `documents_pkey` - PRIMARY KEY (id)
- `ix_documents_created_at` - btree (created_at)
- `ix_documents_owner_id` - btree (owner_id)
- `ix_documents_source_url` - btree (source_url)
- `ix_documents_status` - btree (status)
- `ix_documents_title` - btree (title)

**Foreign Keys:**
- `owner_id` → `users(id)` ON DELETE CASCADE

**Referenced By:**
- `document_chunks.document_id` → ON DELETE CASCADE

---

### 9. `document_chunks` (9 columns)

Individual text chunks from processed documents.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| id | integer | NOT NULL | nextval('document_chunks_id_seq') | Primary key |
| document_id | integer | NOT NULL | | Parent document reference |
| chunk_text | text | NOT NULL | | Chunk content |
| chunk_index | integer | NOT NULL | | Position in document |
| vector_id | varchar(100) | | | Reference to enterprise_rag |
| embedding_generated | boolean | NOT NULL | | Embedding status |
| relevance_score | double precision | | | Search relevance score |
| keywords | json | | | Extracted keywords |
| created_at | timestamptz | NOT NULL | now() | Creation timestamp |

**Indexes:**
- `document_chunks_pkey` - PRIMARY KEY (id)
- `ix_document_chunks_document_id` - btree (document_id)
- `ix_document_chunks_vector_id` - UNIQUE (vector_id)

**Foreign Keys:**
- `document_id` → `documents(id)` ON DELETE CASCADE

---

### 10. `rag_queries` (12 columns)

RAG query tracking for analytics and improvement.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| id | integer | NOT NULL | nextval('rag_queries_id_seq') | Primary key |
| user_id | integer | | | Foreign key to users |
| query_text | text | NOT NULL | | User query |
| session_id | varchar(100) | | | Session identifier |
| retrieved_chunks | integer | NOT NULL | | Chunks retrieved |
| response_text | text | | | Generated response |
| response_sources | json | | | Source references |
| query_latency_ms | double precision | | | Query latency |
| relevance_score | double precision | | | Relevance score |
| user_rating | integer | | | User feedback rating |
| user_feedback | text | | | User feedback text |
| created_at | timestamptz | NOT NULL | now() | Query timestamp |

**Indexes:**
- `rag_queries_pkey` - PRIMARY KEY (id)
- `ix_rag_queries_created_at` - btree (created_at)
- `ix_rag_queries_session_id` - btree (session_id)
- `ix_rag_queries_user_id` - btree (user_id)

**Foreign Keys:**
- `user_id` → `users(id)` ON DELETE SET NULL

---

### 11. `knowledge_bases` (10 columns)

Knowledge base collections for organizing documents.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| id | integer | NOT NULL | nextval('knowledge_bases_id_seq') | Primary key |
| name | varchar(255) | NOT NULL | | Knowledge base name |
| description | text | | | Description |
| owner_id | integer | NOT NULL | | Foreign key to users |
| is_public | boolean | NOT NULL | | Public visibility |
| auto_train | boolean | NOT NULL | | Auto-train on upload |
| total_documents | integer | NOT NULL | | Document count |
| total_chunks | integer | NOT NULL | | Total chunks |
| created_at | timestamptz | NOT NULL | now() | Creation timestamp |
| updated_at | timestamptz | NOT NULL | now() | Last update timestamp |

**Indexes:**
- `knowledge_bases_pkey` - PRIMARY KEY (id)
- `ix_knowledge_bases_name` - btree (name)
- `ix_knowledge_bases_owner_id` - btree (owner_id)

**Foreign Keys:**
- `owner_id` → `users(id)` ON DELETE CASCADE

---

### 12. `audit_logs` (10 columns)

Security and compliance audit trail.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| id | integer | NOT NULL | nextval('audit_logs_id_seq') | Primary key |
| user_id | integer | | | Foreign key to users |
| action | varchar(100) | NOT NULL | | Action performed |
| resource_type | varchar(50) | NOT NULL | | Resource type |
| resource_id | varchar(100) | | | Resource identifier |
| details | json | | | Additional context |
| ip_address | varchar(45) | | | Client IP address |
| status | varchar(20) | NOT NULL | | Action status |
| error_message | text | | | Error details |
| created_at | timestamptz | NOT NULL | now() | Timestamp |

**Indexes:**
- `audit_logs_pkey` - PRIMARY KEY (id)
- `idx_audit_created_status` - btree (created_at, status)
- `idx_audit_user_action` - btree (user_id, action)
- `ix_audit_logs_action` - btree (action)
- `ix_audit_logs_created_at` - btree (created_at)
- `ix_audit_logs_status` - btree (status)
- `ix_audit_logs_user_id` - btree (user_id)

**Foreign Keys:**
- `user_id` → `users(id)` ON DELETE SET NULL

---

### 13. `conversation_sessions` (23 columns)

Agentic conversation state management.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| session_id | varchar(64) | NOT NULL | | Primary key |
| user_id | varchar(128) | NOT NULL | | User identifier |
| created_at | timestamp | | | Session start |
| updated_at | timestamp | | | Last update |
| expires_at | timestamp | | | Session expiry |
| intent | varchar(128) | | | Detected intent |
| resource_type | varchar(64) | | | Target resource type |
| operation | varchar(32) | | | Operation type |
| user_query | text | | | Original user query |
| status | varchar(32) | | | Session status |
| active_agent | varchar(64) | | | Current agent |
| required_params | json | | | Required parameters |
| optional_params | json | | | Optional parameters |
| collected_params | json | | | Collected parameters |
| missing_params | json | | | Missing parameters |
| invalid_params | json | | | Invalid parameters |
| conversation_history | json | | | Full conversation |
| clarification_count | varchar(8) | | | Clarification counter |
| max_clarifications | varchar(8) | | | Max clarifications |
| execution_result | json | | | Execution result |
| error_message | text | | | Error message |
| agent_handoffs | json | | | Agent handoff history |
| extra_data | json | | | Additional data |

**Indexes:**
- `conversation_sessions_pkey` - PRIMARY KEY (session_id)
- `ix_conversation_sessions_session_id` - btree (session_id)
- `ix_conversation_sessions_user_id` - btree (user_id)

---

### 14. `user_context_preferences` (20 columns)

User default context preferences for cloud operations.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| user_id | varchar(255) | NOT NULL | | Primary key |
| default_engagement_id | integer | | | Default engagement |
| default_engagement_name | varchar(255) | | | Engagement name |
| default_ipc_engagement_id | integer | | | IPC engagement |
| default_datacenter_id | integer | | | Default datacenter |
| default_datacenter_name | varchar(255) | | | Datacenter name |
| default_endpoint_ids | json | | | Default endpoints |
| default_cluster_id | integer | | | Default cluster |
| default_cluster_name | varchar(255) | | | Cluster name |
| default_firewall_id | integer | | | Default firewall |
| default_firewall_name | varchar(255) | | | Firewall name |
| default_business_unit_id | integer | | | Default business unit |
| default_business_unit_name | varchar(255) | | | Business unit name |
| default_environment_id | integer | | | Default environment |
| default_environment_name | varchar(255) | | | Environment name |
| default_zone_id | integer | | | Default zone |
| default_zone_name | varchar(255) | | | Zone name |
| preferences | json | | | Additional preferences |
| created_at | timestamptz | NOT NULL | now() | Creation timestamp |
| updated_at | timestamptz | NOT NULL | now() | Last update timestamp |

**Indexes:**
- `user_context_preferences_pkey` - PRIMARY KEY (user_id)
- `idx_user_context_datacenter` - btree (default_datacenter_id)
- `idx_user_context_engagement` - btree (default_engagement_id)
- `ix_user_context_preferences_user_id` - btree (user_id)

---

## Sequences

| Sequence Name | Used By |
|---------------|---------|
| users_id_seq | users.id |
| user_sessions_id_seq | user_sessions.id |
| documents_id_seq | documents.id |
| document_chunks_id_seq | document_chunks.id |
| rag_queries_id_seq | rag_queries.id |
| knowledge_bases_id_seq | knowledge_bases.id |
| audit_logs_id_seq | audit_logs.id |

---

## Functions

All 114 functions in the database are provided by the **pgvector** extension for vector operations:

### Key Vector Functions

| Function | Description |
|----------|-------------|
| `cosine_distance(vector, vector)` | Cosine distance between vectors |
| `l2_distance(vector, vector)` | Euclidean (L2) distance |
| `inner_product(vector, vector)` | Inner product of vectors |
| `vector_dims(vector)` | Get vector dimensions |
| `binary_quantize(vector)` | Binary quantization |
| `subvector(vector, start, length)` | Extract subvector |

### Supported Vector Types

- `vector` - Standard float32 vectors
- `halfvec` - Half-precision (float16) vectors
- `sparsevec` - Sparse vectors

---

## Entity Relationship Diagram

```
┌─────────────────┐
│     users       │
│  (id, username) │
└────────┬────────┘
         │
    ┌────┼────────────────┬──────────────────┬──────────────────┐
    │    │                │                  │                  │
    ▼    ▼                ▼                  ▼                  ▼
┌───────────────┐  ┌─────────────┐  ┌────────────────┐  ┌──────────────┐
│ user_sessions │  │  documents  │  │ knowledge_bases│  │  audit_logs  │
│   (user_id)   │  │ (owner_id)  │  │   (owner_id)   │  │  (user_id)   │
└───────────────┘  └──────┬──────┘  └────────────────┘  └──────────────┘
                          │
                          ▼
                  ┌────────────────┐
                  │ document_chunks│
                  │ (document_id)  │
                  └───────┬────────┘
                          │
                          ▼
                  ┌────────────────┐
                  │ enterprise_rag │
                  │  (vector_id)   │
                  └────────────────┘

┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   chats     │────▶│ chat_files  │     │   folders   │
│  (user_id)  │     │  (chat_id)  │     │  (user_id)  │
└─────────────┘     └─────────────┘     └─────────────┘

┌─────────────────────┐     ┌──────────────────────────┐
│ conversation_sessions│     │ user_context_preferences │
│      (user_id)      │     │        (user_id)         │
└─────────────────────┘     └──────────────────────────┘
```

---

## Issues Found

### 1. Missing HNSW Index on enterprise_rag

The `init_postgres.sql` specifies an HNSW index for vector similarity search, but it's **NOT present** in the live database:

```sql
-- Expected but MISSING:
CREATE INDEX IF NOT EXISTS enterprise_rag_embedding_hnsw_idx 
ON enterprise_rag 
USING hnsw (embedding vector_l2_ops)
WITH (m = 16, ef_construction = 200);
```

**Impact:** Vector similarity searches will use sequential scan instead of the optimized HNSW index, significantly impacting query performance.

**Fix:** Run the index creation manually or update the init script.

### 2. Missing Indexes on enterprise_rag

The following indexes from `init_postgres.sql` are also missing:

```sql
CREATE INDEX IF NOT EXISTS enterprise_rag_url_idx ON enterprise_rag(url);
CREATE INDEX IF NOT EXISTS enterprise_rag_timestamp_idx ON enterprise_rag(timestamp);
CREATE INDEX IF NOT EXISTS enterprise_rag_source_idx ON enterprise_rag(source);
```

### 3. init_postgres.sql is Outdated

The current `init_postgres.sql` file:
- References wrong database name (`ragbot_db` instead of `enterprise_rag`)
- Only creates the `enterprise_rag` table
- Missing 13 other tables that exist in the live database
- Tables are created by SQLAlchemy ORM, not the init script

---

## Recommendations

1. **Update init_postgres.sql** to match the actual database schema or remove it entirely since SQLAlchemy handles table creation.

2. **Create missing indexes** for better query performance:
   ```sql
   -- HNSW index for vector search
   CREATE INDEX IF NOT EXISTS enterprise_rag_embedding_hnsw_idx 
   ON enterprise_rag USING hnsw (embedding vector_l2_ops)
   WITH (m = 16, ef_construction = 200);
   
   -- Additional useful indexes
   CREATE INDEX IF NOT EXISTS enterprise_rag_url_idx ON enterprise_rag(url);
   CREATE INDEX IF NOT EXISTS enterprise_rag_timestamp_idx ON enterprise_rag(timestamp);
   CREATE INDEX IF NOT EXISTS enterprise_rag_source_idx ON enterprise_rag(source);
   ```

3. **Backup strategy:** The database uses Docker volumes (`postgres_data`), ensure regular backups are configured.

---

## Connection Details

```
Host: localhost (or postgres from within Docker network)
Port: 5435 (external) / 5432 (internal)
Database: enterprise_rag
User: ragbot
Password: [see .env file]
```
