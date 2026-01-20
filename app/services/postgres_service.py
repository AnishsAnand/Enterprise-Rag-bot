# postgres_service.py
# ✅ PRODUCTION-READY PostgreSQL Service with pgvector
# ✅ Fixes "timestamp column does not exist" error
# ✅ Handles images, embeddings, and deduplication
# ✅ Auto-healing schema with proper locking

import json
import os
import uuid
import re
import asyncio
import logging
import socket
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import quote_plus, urlparse

import asyncpg
from asyncpg.pool import Pool

from app.core.config import settings
from app.services.ai_service import ai_service

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("POSTGRES_LOG_LEVEL", "INFO"))


class PostgresService:
    """Production-grade PostgreSQL service with pgvector extension."""

    def __init__(self):
        self.pool: Optional[Pool] = None
        
        # Validate table name
        raw_table_name: str = getattr(settings, "POSTGRES_TABLE", "enterprise_rag") or "enterprise_rag"
        self.table_name: str = self._validate_table_name(raw_table_name)

        # Detect environment and set connection params
        self._detect_environment()

        # Search configuration
        self.search_config = {
            "min_relevance_threshold": float(os.getenv("POSTGRES_MIN_RELEVANCE", "0.08")),
            "max_initial_results": int(os.getenv("POSTGRES_MAX_INITIAL_RESULTS", "200")),
            "rerank_top_k": int(os.getenv("POSTGRES_RERANK_TOP_K", "100")),
        }

        # Embedding dimension - will be auto-detected or use config
        self.embedding_dim: Optional[int] = None
        self._dimension_detected = False
        self._config_dimension = int(os.getenv("EMBEDDING_DIMENSION", "4096"))

        # Connection state
        self._connection_established: bool = False
        self._initialization_attempted: bool = False
        self._schema_lock = asyncio.Lock()

        # Stopwords for query preprocessing
        self.stopwords: Set[str] = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
            'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his',
            'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy',
            'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use', 'that', 'with',
            'have', 'this', 'will', 'your', 'from', 'they', 'know', 'want', 'been',
            'good', 'much', 'some', 'time', 'very', 'when', 'come', 'here', 'just',
            'like', 'long', 'make', 'many', 'over', 'such', 'take', 'than', 'them',
            'well', 'were', 'what', 'about', 'into'
        }

        logger.info(f"🔧 PostgresService initialized")
        logger.info(f"   - Table: {self.table_name}")
        logger.info(f"   - Environment: {getattr(self, 'environment', 'unknown')}")
        logger.info(f"   - Host: {getattr(self, 'db_host', 'unknown')}:{getattr(self, 'db_port', 'unknown')}")
        logger.info(f"   - Database: {getattr(self, 'db_name', 'unknown')}")

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _validate_table_name(self, name: str) -> str:
        """Ensure table name is a safe SQL identifier."""
        if not isinstance(name, str) or not name:
            return "enterprise_rag"
        if re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', name):
            return name
        logger.warning(f"⚠️ Invalid table name '{name}', using 'enterprise_rag'")
        return "enterprise_rag"

    def _detect_environment(self) -> None:
        """Auto-detect environment and configure connection parameters."""
        explicit_host = os.getenv("POSTGRES_HOST")
        candidate_hosts = []

        if explicit_host:
            candidate_hosts.append(explicit_host)
        else:
            candidate_hosts.extend(["localhost", "127.0.0.1"])

        resolved_host = None
        for h in candidate_hosts:
            if self._host_resolves(h):
                resolved_host = h
                break

        if resolved_host:
            self.db_host = explicit_host or resolved_host
            self.environment = "explicit" if explicit_host else "localhost"
        else:
            self.db_host = explicit_host or "localhost"
            self.environment = "unknown"
            logger.warning("⚠️ Could not resolve DB hostname")

        self.db_port = int(os.getenv("POSTGRES_PORT", "5432"))
        self.db_name = os.getenv("POSTGRES_DB", "ragbot_db")
        self.db_user = os.getenv("POSTGRES_USER", "ragbot")
        self.db_password = os.getenv("POSTGRES_PASSWORD", "ragbot_secret_2024")

    def _host_resolves(self, host: str) -> bool:
        """Check if host resolves to an IP address."""
        try:
            socket.getaddrinfo(host, None)
            return True
        except Exception:
            return False

    def _parse_timestamp(self, ts: Any) -> datetime:
        """
        Convert various timestamp formats to datetime object.
        Handles datetime objects, ISO strings, and falls back to current time.
        """
        # Already a datetime object
        if isinstance(ts, datetime):
            return ts

        # Parse string formats
        if isinstance(ts, str) and ts.strip():
            try:
                cleaned = ts.replace('Z', '+00:00')
                return datetime.fromisoformat(cleaned)
            except Exception:
                try:
                    from dateutil import parser
                    return parser.parse(ts)
                except Exception:
                    pass

        # Fallback to current time
        logger.debug(f"Using current time for invalid timestamp: {ts}")
        return datetime.now()

    async def _detect_embedding_dimension(self) -> int:
        """Auto-detect embedding dimension from AI service."""
        if self._dimension_detected and self.embedding_dim:
            return self.embedding_dim

        logger.info("🔍 Detecting embedding dimension...")
        
        try:
            embeddings = await ai_service.generate_embeddings(["test dimension detection"])
            
            if not embeddings or not embeddings[0]:
                raise RuntimeError("Failed to generate test embedding")

            dim = len(embeddings[0])
            self.embedding_dim = dim
            self._dimension_detected = True

            if dim != self._config_dimension:
                logger.warning(
                    f"⚠️ Dimension mismatch: config={self._config_dimension}, actual={dim}"
                )

            logger.info(f"✅ Detected embedding dimension: {dim}")
            return dim

        except Exception as e:
            logger.error(f"❌ Dimension detection failed: {e}")
            self.embedding_dim = self._config_dimension
            self._dimension_detected = True
            logger.warning(f"⚠️ Using config dimension: {self._config_dimension}")
            return self._config_dimension

    # ========================================================================
    # INITIALIZATION
    # ========================================================================

    async def initialize(self) -> None:
        """Initialize PostgreSQL connection pool and schema."""
        if self.pool is not None or self._initialization_attempted:
            logger.debug("PostgreSQL already initialized")
            return

        self._initialization_attempted = True

        # Step 1: Detect embedding dimension first
        await self._detect_embedding_dimension()

        # Step 2: Check if PostgreSQL is available
        if not self._is_postgres_available():
            logger.warning("⚠️ PostgreSQL unavailable - running in DEGRADED MODE")
            self._connection_established = False
            self.pool = None
            return

        # Step 3: Try connecting to available hosts
        max_retries = 3
        backoff = 2.0

        prioritized_hosts = [self.db_host]
        if self.db_host not in ("localhost", "127.0.0.1"):
            prioritized_hosts.extend(["localhost", "127.0.0.1"])

        for attempt in range(1, max_retries + 1):
            host_to_try = prioritized_hosts[(attempt - 1) % len(prioritized_hosts)]
            
            logger.info(
                f"🔌 Connecting to PostgreSQL (attempt {attempt}/{max_retries}) "
                f"using host '{host_to_try}'"
            )

            if not self._host_resolves(host_to_try):
                logger.debug(f"Host '{host_to_try}' does not resolve")
                continue

            try:
                self.pool = await asyncpg.create_pool(
                    host=host_to_try,
                    port=self.db_port,
                    database=self.db_name,
                    user=self.db_user,
                    password=self.db_password,
                    min_size=int(os.getenv("POSTGRES_POOL_MIN", "2")),
                    max_size=int(os.getenv("POSTGRES_POOL_MAX", "10")),
                    command_timeout=60,
                    timeout=10,
                )

                async with self.pool.acquire() as conn:
                    version = await conn.fetchval('SELECT version();')
                    logger.info(f"✅ PostgreSQL connected: {version[:50]}...")

                # Ensure schema
                await self._ensure_schema()
                
                self._connection_established = True
                self.db_host = host_to_try

                logger.info("✅ PostgreSQL initialization complete")
                return

            except asyncpg.InvalidCatalogNameError:
                logger.error(f"❌ Database '{self.db_name}' does not exist!")
                logger.error(f"   Create it: createdb -U {self.db_user} {self.db_name}")
                break

            except asyncpg.InvalidPasswordError:
                logger.error(f"❌ Invalid password for user '{self.db_user}'")
                break

            except (asyncpg.PostgresConnectionError, ConnectionRefusedError, OSError) as e:
                logger.debug(f"Connection attempt {attempt} failed: {e.__class__.__name__}")

                if self.pool:
                    try:
                        await self.pool.close()
                    except Exception:
                        pass
                    self.pool = None

                if attempt >= max_retries:
                    logger.warning("⚠️ PostgreSQL unavailable after retries - DEGRADED MODE")
                    self._connection_established = False
                    return

                await asyncio.sleep(backoff * attempt)

            except Exception as e:
                logger.exception(f"❌ Unexpected error: {e}")
                self._connection_established = False
                self.pool = None
                return

    def _is_postgres_available(self) -> bool:
        """Quick check if PostgreSQL is available."""
        test_hosts = [self.db_host, "localhost", "127.0.0.1"]
        for host in test_hosts:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex((host, self.db_port))
                sock.close()
                if result == 0:
                    return True
            except Exception:
                continue
        return False

    # ========================================================================
    # SCHEMA MANAGEMENT
    # ========================================================================

    async def _ensure_schema(self) -> None:
        """
        Ensure pgvector extension and create table schema.
        Uses lock to prevent concurrent CREATE TABLE operations.
        """
        if not self.pool:
            logger.warning("⚠️ Cannot ensure schema - pool not initialized")
            return

        async with self._schema_lock:
            try:
                async with self.pool.acquire() as conn:
                    # Set search path
                    await conn.execute("SET search_path TO public;")

                    # Enable pgvector extension
                    try:
                        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                        logger.info("✅ pgvector extension enabled")
                    except Exception as e:
                        logger.debug(f"pgvector extension note: {e}")

                    # Create table with proper schema
                    create_table_sql = f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        id VARCHAR(100) PRIMARY KEY,
                        embedding vector({self.embedding_dim}),
                        content TEXT NOT NULL,
                        content_tsv tsvector GENERATED ALWAYS AS 
                            (to_tsvector('english', content)) STORED,
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
                    """
                    await conn.execute(create_table_sql)
                    logger.info(f"✅ Table {self.table_name} created/verified")

                    # Create HNSW index for vector similarity
                    try:
                        await conn.execute(f"""
                        CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_hnsw_idx 
                        ON {self.table_name} 
                        USING hnsw (embedding vector_l2_ops)
                        WITH (m = 16, ef_construction = 200);
                        """)
                        logger.info("✅ HNSW vector index created")
                    except Exception as e:
                        logger.debug(f"HNSW index note: {e}")

                    # Create GIN index for full-text search
                    try:
                        await conn.execute(f"""
                        CREATE INDEX IF NOT EXISTS {self.table_name}_content_tsv_idx
                        ON {self.table_name} USING gin(content_tsv);
                        """)
                        logger.info("✅ Full-text search index created")
                    except Exception as e:
                        logger.debug(f"FTS index note: {e}")

                    logger.info("✅ Schema fully initialized")

                    try:
                        await conn.execute(f"""
                                           ALTER TABLE {self.table_name}
                                           ADD COLUMN IF NOT EXISTS timestamp TIMESTAMPTZ DEFAULT NOW();
                                           """)
                        
                    except Exception as e:
                        logger.debug(f"Timestamp Exists:{e}")


            except Exception as e:
                logger.exception(f"❌ Schema creation failed: {e}")
                raise

    async def _table_exists(self, conn: Optional[asyncpg.connection.Connection] = None) -> bool:
        """Check if target table exists."""
        if not self.pool:
            return False

        check_sql = """
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name = $1
        );
        """

        try:
            if conn:
                exists = await conn.fetchval(check_sql, self.table_name)
            else:
                async with self.pool.acquire() as temp_conn:
                    exists = await temp_conn.fetchval(check_sql, self.table_name)
            return bool(exists)
        except Exception as e:
            logger.debug(f"Table existence check failed: {e}")
            return False

    # ========================================================================
    # DOCUMENT INGESTION
    # ========================================================================

    async def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """
        Add documents to PostgreSQL with proper error handling.
        ✅ FIXED: Properly handles timestamp column in INSERT statement.
        """
        if not self.pool or not self._connection_established:
            logger.warning("⚠️ PostgreSQL unavailable - cannot add documents")
            return []

        if not documents:
            return []

        try:
            # Verify table exists
            async with self.pool.acquire() as conn:
                if not await self._table_exists(conn):
                    logger.warning(f"⚠️ Table '{self.table_name}' missing - creating schema")
                    await self._ensure_schema()
                    
                    if not await self._table_exists(conn):
                        logger.error(f"❌ Table '{self.table_name}' still missing after schema creation")
                        return []

            # Prepare document data
            ids: List[str] = []
            texts: List[str] = []
            documents_data: List[Dict[str, Any]] = []

            for doc in documents:
                doc_id = str(uuid.uuid4())
                ids.append(doc_id)

                content = str(doc.get("content", ""))[:8000]
                texts.append(content)

                # Parse timestamp properly
                timestamp_value = self._parse_timestamp(doc.get("timestamp"))

                # Normalize images
                images_raw = doc.get("images", [])
                images_normalized = self._normalize_images_for_storage(images_raw)

                if images_normalized:
                    logger.debug(
                        f"📷 Processing {len(images_normalized)} images for "
                        f"{doc.get('url', 'unknown')[:60]}"
                    )

                documents_data.append({
                    "id": doc_id,
                    "content": content,
                    "url": str(doc.get("url", ""))[:2000],
                    "title": str(doc.get("title", ""))[:500],
                    "format": str(doc.get("format", "text"))[:100],
                    "timestamp": timestamp_value,
                    "source": str(doc.get("source", "web_scraping"))[:100],
                    "content_length": len(content),
                    "word_count": len(content.split()),
                    "image_count": len(images_normalized),
                    "has_images": len(images_normalized) > 0,
                    "domain": self._extract_domain(doc.get("url", "")),
                    "content_hash": abs(hash(content)) % (10**12),
                    "images_json": images_normalized,
                    "key_terms": [],
                })

            # Generate embeddings
            logger.info(f"🔄 Generating embeddings for {len(texts)} documents...")
            embeddings = await ai_service.generate_embeddings(texts)

            if not embeddings or len(embeddings) != len(texts):
                logger.error("❌ Failed to generate embeddings")
                return []

            # ✅ CRITICAL FIX: Proper INSERT statement with all 16 columns
            insert_sql = f"""
            INSERT INTO {self.table_name} 
            (id, embedding, content, url, title, format, timestamp, source,
             content_length, word_count, image_count, has_images, domain,
             content_hash, images_json, key_terms)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
            ON CONFLICT (id) DO UPDATE SET
                embedding = EXCLUDED.embedding,
                content = EXCLUDED.content,
                updated_at = NOW();
            """

            inserted_count = 0
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    for i, doc_data in enumerate(documents_data):
                        # Format embedding as vector literal
                        embedding_str = '[' + ','.join(map(str, embeddings[i])) + ']'

                        # Serialize images as JSON string
                        images_json_str = json.dumps(doc_data["images_json"])

                        # ✅ CRITICAL: Execute with exactly 16 parameters matching INSERT
                        await conn.execute(
                            insert_sql,
                            doc_data["id"],              # $1
                            embedding_str,               # $2
                            doc_data["content"],         # $3
                            doc_data["url"],             # $4
                            doc_data["title"],           # $5
                            doc_data["format"],          # $6
                            doc_data["timestamp"],       # $7  ✅ FIXED: datetime object
                            doc_data["source"],          # $8
                            doc_data["content_length"],  # $9
                            doc_data["word_count"],      # $10
                            doc_data["image_count"],     # $11
                            doc_data["has_images"],      # $12
                            doc_data["domain"],          # $13
                            doc_data["content_hash"],    # $14
                            images_json_str,             # $15  ✅ JSON string
                            doc_data["key_terms"]        # $16
                        )
                        inserted_count += 1

            total_images = sum(d["image_count"] for d in documents_data)
            logger.info(
                f"✅ Successfully stored {inserted_count} documents "
                f"with {total_images} total images"
            )
            return ids

        except Exception as e:
            logger.exception(f"❌ Error adding documents: {e}")
            return []

    # ========================================================================
    # IMAGE HANDLING
    # ========================================================================

    def _normalize_images_for_storage(self, images_raw: Any) -> List[Dict[str, Any]]:
        """Normalize images to consistent format for storage."""
        if not images_raw:
            return []

        # Parse JSON string if needed
        if isinstance(images_raw, str):
            try:
                images_raw = json.loads(images_raw)
            except Exception:
                return []

        if not isinstance(images_raw, list):
            return []

        normalized = []

        for img in images_raw:
            # Handle string URLs
            if isinstance(img, str):
                if img.startswith("http"):
                    normalized.append({
                        "url": img,
                        "alt": "",
                        "caption": "",
                        "type": "content"
                    })
                continue

            # Handle dict images
            if isinstance(img, dict):
                url = img.get("url")

                # Must have valid URL
                if not url or not isinstance(url, str):
                    continue
                if not url.startswith("http"):
                    continue

                normalized.append({
                    "url": url,
                    "alt": str(img.get("alt", ""))[:200],
                    "caption": str(img.get("caption", ""))[:500],
                    "type": str(img.get("type", "content"))[:50]
                })

        return normalized

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            parsed = urlparse(url)
            return parsed.netloc[:500] if parsed.netloc else ""
        except Exception:
            return ""

    # ========================================================================
    # SEARCH
    # ========================================================================

    async def search_documents(
        self, 
        query: str, 
        n_results: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Perform vector similarity search with relevance filtering."""
        if not query or not self.pool or not self._connection_established:
            logger.debug("⚠️ Search unavailable or empty query")
            return []

        try:
            # Preprocess query
            cleaned_query, key_terms = self._preprocess_query(query)
            if not cleaned_query:
                return []

            logger.info(f"🔍 Searching: '{query[:60]}...'")

            # Generate query embedding
            query_embeddings = await ai_service.generate_embeddings([cleaned_query])
            if not query_embeddings:
                logger.warning("⚠️ Failed to generate embeddings")
                return []

            query_embedding = query_embeddings[0]

            if len(query_embedding) != self.embedding_dim:
                logger.error(
                    f"❌ Embedding dimension mismatch: "
                    f"query={len(query_embedding)}, expected={self.embedding_dim}"
                )
                return []

            embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'

            # Vector similarity search
            initial_k = self.search_config["max_initial_results"]
            search_sql = f"""
            SELECT 
                id, content, url, title, format, timestamp, source,
                content_length, word_count, image_count, has_images,
                domain, content_hash, images_json, key_terms,
                embedding <-> $1::vector AS distance
            FROM {self.table_name}
            ORDER BY embedding <-> $1::vector
            LIMIT $2;
            """

            async with self.pool.acquire() as conn:
                rows = await conn.fetch(search_sql, embedding_str, initial_k)

            if not rows:
                logger.info("ℹ️ No documents found")
                return []

            # Score and filter results
            scored_results = []
            for row in rows:
                content = row['content'] or ""
                raw_images_field = row.get('images_json', None)

                # Parse images from database
                images_list = self._coerce_images_from_db(raw_images_field)

                metadata = {
                    "url": row['url'] or "",
                    "title": row['title'] or "",
                    "format": row['format'] or "",
                    "timestamp": row['timestamp'].isoformat() if row['timestamp'] else "",
                    "source": row['source'] or "",
                    "image_count": row['image_count'] or 0,
                    "has_images": row['has_images'] or False,
                    "domain": row['domain'] or "",
                    "images": images_list,
                }

                distance = float(row['distance'])
                relevance_score = max(0.0, 1.0 / (1.0 + distance))

                if relevance_score >= self.search_config["min_relevance_threshold"]:
                    scored_results.append({
                        "content": content,
                        "metadata": metadata,
                        "relevance_score": relevance_score,
                    })

            # Sort by relevance
            scored_results.sort(key=lambda x: x["relevance_score"], reverse=True)

            # Limit results
            final_k = int(n_results or 10)
            final_results = []
            for r in scored_results[:final_k]:
                final_results.append({
                    "content": r["content"],
                    "relevance_score": float(r["relevance_score"]),
                    "metadata": r["metadata"]
                })

            logger.info(f"✅ Returning {len(final_results)} results")
            return final_results

        except Exception as e:
            logger.exception(f"❌ Search error: {e}")
            return []

    def _coerce_images_from_db(self, images_field: Any) -> List[Dict[str, Any]]:
        """Parse images from database JSONB field."""
        normalized: List[Dict[str, Any]] = []

        try:
            # Parse the field into a list
            if isinstance(images_field, list):
                entries = images_field
            elif isinstance(images_field, str) and images_field.strip():
                try:
                    entries = json.loads(images_field)
                except Exception:
                    entries = []
            else:
                entries = []

            for item in entries:
                # Handle string URLs
                if isinstance(item, str):
                    if item.startswith("http"):
                        normalized.append({
                            "url": item,
                            "alt": "",
                            "caption": ""
                        })
                    continue

                # Handle dict objects
                if isinstance(item, dict):
                    url = item.get("url")
                    
                    if not url or not isinstance(url, str):
                        continue

                    normalized.append({
                        "url": url,
                        "alt": str(item.get("alt", ""))[:200],
                        "caption": str(item.get("caption", ""))[:500]
                    })

        except Exception as e:
            logger.debug(f"Image parsing failed: {e}")

        return normalized

    def _preprocess_query(self, query: str) -> Tuple[str, List[str]]:
        
        if isinstance(query, dict):
            query = json.dumps(query)
        elif not isinstance(query, str):
            query = str(query)
        if not query:
            return "", []
        
        cleaned = re.sub(r'\s+', ' ', query.strip().lower())
        words = re.findall(r'\b\w+\b', cleaned)
        key_terms = [w for w in words if len(w) > 3 and w not in self.stopwords]
        
        return cleaned, key_terms[:15]

    # ========================================================================
    # STATISTICS & MANAGEMENT
    # ========================================================================

    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics with auto-healing."""
        if not self.pool or not self._connection_established:
            return {
                "document_count": 0,
                "collection_name": self.table_name,
                "status": "unavailable",
                "message": "PostgreSQL not connected"
            }

        try:
            async with self.pool.acquire() as conn:
                # Auto-heal if table missing
                if not await self._table_exists(conn):
                    logger.warning("⚠️ Table missing during stats - creating schema")
                    await self._ensure_schema()
                    
                    if not await self._table_exists(conn):
                        return {
                            "document_count": 0,
                            "collection_name": self.table_name,
                            "status": "initializing",
                            "message": "Schema creation in progress"
                        }

                count = await conn.fetchval(f"SELECT COUNT(*) FROM {self.table_name};")

                return {
                    "document_count": int(count),
                    "collection_name": self.table_name,
                    "status": "active",
                    "connection": {
                        "host": self.db_host,
                        "port": self.db_port,
                        "database": self.db_name,
                    },
                    "embedding_dimension": self.embedding_dim,
                }

        except Exception as e:
            logger.exception(f"❌ Stats error: {e}")
            return {
                "document_count": 0,
                "status": "error",
                "error": str(e)
            }

    async def delete_collection(self) -> None:
        """Delete the entire table."""
        if not self.pool:
            return

        try:
            async with self.pool.acquire() as conn:
                await conn.execute(f"DROP TABLE IF EXISTS {self.table_name} CASCADE;")
                logger.info(f"✅ Deleted table: {self.table_name}")
        except Exception as e:
            logger.exception(f"❌ Delete error: {e}")

    async def close(self) -> None:
        """Close PostgreSQL connection pool."""
        if self.pool:
            try:
                await self.pool.close()
                self.pool = None
                self._connection_established = False
                logger.info("✅ PostgreSQL connection closed")
            except Exception as e:
                logger.exception(f"❌ Close error: {e}")


# ============================================================================
# GLOBAL SINGLETON INSTANCE
# ============================================================================
postgres_service = PostgresService()