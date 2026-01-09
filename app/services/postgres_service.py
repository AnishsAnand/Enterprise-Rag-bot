# postgres_service.py - PRODUCTION FIX FOR TIMESTAMP & SCHEMA RACE ISSUES
# Key Changes:
# - Added robust images_json parsing and normalization
# - If image entry has 'image_prompt' but no URL: produce safe placeholder URL
# - Defensive table name validation
# - Schema lock to avoid race on CREATE TABLE

import json
import os
import uuid
import re
import asyncio
import logging
import socket
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict
from urllib.parse import quote_plus

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
        # Validate provided table name
        raw_table_name: str = getattr(settings, "POSTGRES_TABLE", "enterprise_rag") or "enterprise_rag"
        self.table_name: str = self._validate_table_name(raw_table_name)

        self._detect_environment()

        self.search_config = {
            "min_relevance_threshold": float(os.getenv("POSTGRES_MIN_RELEVANCE", "0.08")),
            "max_initial_results": int(os.getenv("POSTGRES_MAX_INITIAL_RESULTS", "200")),
            "rerank_top_k": int(os.getenv("POSTGRES_RERANK_TOP_K", "100")),
            "enable_query_expansion": os.getenv("POSTGRES_ENABLE_QUERY_EXPANSION", "true").lower() == "true",
            "enable_semantic_rerank": os.getenv("POSTGRES_ENABLE_SEMANTIC_RERANK", "true").lower() == "true",
            "enable_context_enrichment": os.getenv("POSTGRES_ENABLE_CONTEXT_ENRICHMENT", "true").lower() == "true",
            "diversity_factor": float(os.getenv("POSTGRES_DIVERSITY_FACTOR", "0.2")),
        }

        self.embedding_dim: int = int(os.getenv("EMBEDDING_DIMENSION", "4096"))

        logger.info(f"🔧 PostgresService PRODUCTION init:")
        logger.info(f"   - Environment: {getattr(self, 'environment', 'unknown')}")
        logger.info(f"   - Host: {getattr(self, 'db_host', 'unknown')}:{getattr(self, 'db_port', 'unknown')}")
        logger.info(f"   - Database: {getattr(self, 'db_name', 'unknown')}")
        logger.info(f"   - Table: {self.table_name}")
        logger.info(f"   - Embedding dimension: {self.embedding_dim}")

        self._connection_established: bool = False
        self._initialization_attempted: bool = False

        # Schema creation lock to avoid concurrent CREATE TABLE race conditions
        self._schema_lock = asyncio.Lock()

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

    def _validate_table_name(self, name: str) -> str:
        """Ensure provided table name is a safe SQL identifier. Fall back to 'enterprise_rag'."""
        if not isinstance(name, str) or not name:
            return "enterprise_rag"
        if re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', name):
            return name
        logger.warning(f"⚠️ Invalid POSTGRES_TABLE name '{name}', defaulting to 'enterprise_rag'")
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
            if explicit_host:
                self.db_host = explicit_host
                self.environment = "explicit"
            elif resolved_host in ("localhost", "127.0.0.1"):
                self.db_host = resolved_host
                self.environment = "localhost"
            else:
                self.db_host = resolved_host
                self.environment = "docker_or_network"
        else:
            self.db_host = explicit_host or "localhost"
            self.environment = "unknown"
            logger.warning("⚠️ Could not resolve DB hostname. Defaulting to 'localhost'.")

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

    # ✅ PRODUCTION FIX: New helper method to parse timestamps
    def _parse_timestamp(self, ts: Any) -> datetime:
        """
        Convert various timestamp formats to datetime object.

        Handles:
        - datetime objects (pass through)
        - ISO format strings (most common)
        - Other common date strings
        - Fallback to current time for invalid inputs

        CRITICAL: Returns datetime object, NOT string!
        This fixes the asyncpg DataError.
        """
        # Already a datetime object - pass through
        if isinstance(ts, datetime):
            return ts

        # Try parsing string formats
        if isinstance(ts, str) and ts.strip():
            try:
                # ISO format: "2026-01-09T06:08:05.421253"
                # Handle both with and without timezone
                cleaned = ts.replace('Z', '+00:00')
                return datetime.fromisoformat(cleaned)
            except Exception as e:
                logger.debug(f"ISO parse failed for '{ts}': {e}")

                # Try dateutil parser as fallback (more flexible)
                try:
                    from dateutil import parser
                    return parser.parse(ts)
                except Exception as e2:
                    logger.debug(f"Dateutil parse failed for '{ts}': {e2}")

        # Fallback: current time with warning
        logger.warning(f"⚠️ Invalid timestamp '{ts}', using current time")
        return datetime.now()

    async def initialize(self) -> None:
        """Initialize PostgreSQL connection pool."""
        if self.pool is not None:
            logger.debug("PostgreSQL pool already initialized")
            return

        if self._initialization_attempted:
            logger.debug("PostgreSQL initialization already attempted")
            return

        self._initialization_attempted = True

        if not self._is_postgres_available():
            logger.warning("⚠️ PostgreSQL is not available on any host")
            logger.warning("⚠️ Application will run in DEGRADED MODE without vector search")
            self._connection_established = False
            self.pool = None
            return

        max_retries = 3
        backoff = 2.0

        prioritized_hosts = [self.db_host]
        if self.db_host not in ("localhost", "127.0.0.1"):
            prioritized_hosts.extend(["localhost", "127.0.0.1"])

        for attempt in range(1, max_retries + 1):
            host_to_try = prioritized_hosts[(attempt - 1) % len(prioritized_hosts)]
            logger.info(f"🔌 Connecting to PostgreSQL (attempt {attempt}/{max_retries}) using host '{host_to_try}'")

            if not self._host_resolves(host_to_try):
                logger.debug(f"   Host '{host_to_try}' does not resolve. Trying next.")
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

                # Ensure schema BEFORE declaring connection established
                try:
                    await self._ensure_schema()
                except Exception as e:
                    logger.exception(f"❌ Schema ensure failed during initialize: {e}")
                    # Attempt cleanup and retry
                    if self.pool:
                        try:
                            await self.pool.close()
                        except Exception:
                            pass
                        self.pool = None
                    raise

                # Only mark fully established AFTER schema exists
                self._connection_established = True
                self.db_host = host_to_try

                logger.info(f"✅ PostgreSQL initialization complete (schema verified)")
                return

            except asyncpg.InvalidCatalogNameError:
                logger.error(f"❌ Database '{self.db_name}' does not exist!")
                logger.error(f"   Create it with: createdb -U {self.db_user} {self.db_name}")
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
                    logger.warning("⚠️ PostgreSQL connection failed after retries")
                    logger.warning("⚠️ Application will run in DEGRADED MODE without vector search")
                    self._connection_established = False
                    return

                await asyncio.sleep(backoff * attempt)

            except Exception as e:
                logger.exception(f"❌ Unexpected error: {e}")
                self._connection_established = False
                self.pool = None
                return

    def _is_postgres_available(self) -> bool:
        """Quick check if PostgreSQL is available on any host."""
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

    async def _ensure_schema(self) -> None:
        """Ensure pgvector extension and create table schema. Serialized by _schema_lock."""
        if not self.pool:
            logger.warning("⚠️ Cannot ensure schema - pool not initialized")
            return

        # Acquire lock to prevent concurrent CREATE TABLE from multiple coroutines
        async with self._schema_lock:
            try:
                async with self.pool.acquire() as conn:
                    # Ensure we're operating in public schema (avoid search_path surprises)
                    try:
                        await conn.execute("SET search_path TO public;")
                    except Exception:
                        # Non-fatal — continue
                        pass

                    # Ensure pgvector extension exists
                    try:
                        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                        logger.info("✅ pgvector extension enabled")
                    except Exception as e:
                        # Log and continue — may fail if user lacks rights
                        logger.debug(f"pgvector extension creation/log note: {e}")

                    # Create table if missing
                    create_table_sql = f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        id VARCHAR(100) PRIMARY KEY,
                        embedding vector({self.embedding_dim}),
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
                    """
                    await conn.execute(create_table_sql)
                    logger.info(f"✅ Table {self.table_name} created/verified")

                    # Create HNSW index — best-effort (some Postgres versions or pgvector versions might not accept options)
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

                    logger.info("✅ All indexes created/verified")

            except Exception as e:
                logger.exception(f"❌ Error ensuring schema: {e}")
                raise

    async def _table_exists(self, conn: Optional[asyncpg.connection.Connection] = None) -> bool:
        """
        Check if the target table exists. If a connection is not supplied, acquire one.
        Uses information_schema for reliability.
        """
        if not self.pool:
            return False

        # Use supplied conn if present
        if conn:
            try:
                exists = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT 1
                        FROM information_schema.tables
                        WHERE table_schema = 'public' AND table_name = $1
                    );
                """, self.table_name)
                return bool(exists)
            except Exception as e:
                logger.debug(f"_table_exists check failed with provided conn: {e}")
                return False

        # Acquire a connection temporarily
        try:
            async with self.pool.acquire() as conn2:
                exists = await conn2.fetchval("""
                    SELECT EXISTS (
                        SELECT 1
                        FROM information_schema.tables
                        WHERE table_schema = 'public' AND table_name = $1
                    );
                """, self.table_name)
                return bool(exists)
        except Exception as e:
            logger.debug(f"_table_exists check failed: {e}")
            return False

    async def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:

        if not self.pool or not self._connection_established:
            logger.warning("⚠️ PostgreSQL unavailable - cannot add documents")
            return []

        try:
        # Ensure table exists
            async with self.pool.acquire() as conn:
                if not await self._table_exists(conn):
                    logger.warning(
                    f"⚠️ Table '{self.table_name}' missing - attempting to create"
                    )
                    await self._ensure_schema()
                
                    if not await self._table_exists(conn):
                        logger.error(
                        f"❌ Table '{self.table_name}' still missing after schema creation"
                    )
                        return []

            ids: List[str] = []
            texts: List[str] = []
            documents_data: List[Dict[str, Any]] = []

            for doc in documents or []:
                doc_id = str(uuid.uuid4())
                ids.append(doc_id)

                content = str(doc.get("content", ""))[:8000]
                texts.append(content)

            # ✅ CRITICAL: Parse timestamp properly
                timestamp_value = self._parse_timestamp(doc.get("timestamp"))

            # ✅ CRITICAL: Process images field
                images_raw = doc.get("images", [])
                images_normalized = self._normalize_images_for_storage(images_raw)
            
            # Log image storage
                if images_normalized:
                    logger.info(
                    f"📷 Storing {len(images_normalized)} images for document "
                    f"{doc.get('url', 'unknown')[:60]}"
                )
                    logger.debug(f"Sample image URL: {images_normalized[0].get('url', 'N/A')[:80]}")

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
                "image_count": len(images_normalized),  # ✅ Set correct count
                "has_images": len(images_normalized) > 0,  # ✅ Set boolean
                "domain": self._extract_domain(doc.get("url", "")),
                "content_hash": abs(hash(content)) % (10**8),
                "images_json": images_normalized,  # ✅ Store normalized images
                "key_terms": [],
            })

        # Generate embeddings
            logger.info(f"🔄 Generating embeddings for {len(texts)} documents...")
            embeddings = await ai_service.generate_embeddings(texts)

            if not embeddings or len(embeddings) != len(texts):
                logger.error("❌ Failed to generate embeddings")
                return []

        # Insert into database
            insert_sql = f"""
            INSERT INTO {self.table_name} 
            (id, embedding, content, url, title, format, timestamp, source,
         content_length, word_count, image_count, has_images, domain,
         content_hash, images_json, key_terms)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
            """

            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    for i, doc_data in enumerate(documents_data):
                    # Prepare embedding as vector literal
                        embedding_str = '[' + ','.join(map(str, embeddings[i])) + ']'

                    # ✅ CRITICAL: Serialize images_json as JSON string
                        images_json_str = json.dumps(doc_data["images_json"])

                        await conn.execute(
                        insert_sql,
                        doc_data["id"],
                        embedding_str,
                        doc_data["content"],
                        doc_data["url"],
                        doc_data["title"],
                        doc_data["format"],
                        doc_data["timestamp"],
                        doc_data["source"],
                        doc_data["content_length"],
                        doc_data["word_count"],
                        doc_data["image_count"],
                        doc_data["has_images"],
                        doc_data["domain"],
                        doc_data["content_hash"],
                        images_json_str,  # ✅ JSON string
                        doc_data["key_terms"]
                        )

            logger.info(
            f"✅ Successfully added {len(ids)} documents "
            f"with {sum(d['image_count'] for d in documents_data)} total images"
        )
            return ids

        except Exception as e:
            logger.exception(f"❌ Error adding documents: {e}")
            return []
        
    def _normalize_images_for_storage(self, images_raw: Any) -> List[Dict[str, Any]]:

        if not images_raw:
            return []
    
    # Ensure it's a list
        if isinstance(images_raw, str):
            try:
                images_raw = json.loads(images_raw)
            except:
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
                    "type": "content",
                    "text": ""
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
                "type": str(img.get("type", "content"))[:50],
                "text": str(img.get("text", ""))[:800]
            })
    
        return normalized
    def _extract_domain(self, url: str) -> str:

        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc[:500] if parsed.netloc else ""
        except:
            return ""

    async def search_documents(self, query: str, n_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """Perform vector similarity search."""
        if not query:
            return []

        if not self.pool or not self._connection_established:
            logger.debug("⚠️ PostgreSQL unavailable - returning empty results")
            return []

        try:
            cleaned_query, key_terms = self._preprocess_query(query)
            if not cleaned_query:
                return []

            logger.info(f"🔍 Search: '{query[:60]}...'")

            query_embeddings = await ai_service.generate_embeddings([cleaned_query])
            if not query_embeddings:
                logger.warning("⚠️ Failed to generate embeddings")
                return []

            query_embedding = query_embeddings[0]

            if len(query_embedding) != self.embedding_dim:
                logger.error(f"❌ Embedding dimension mismatch")
                return []

            embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'

            initial_k = 100
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

            scored_results = []
            for row in rows:
                content = row['content'] or ""
                # Parse images field directly from the row (asyncpg will decode json into Python)
                raw_images_field = row.get('images_json', None)

                # Normalize metadata
                metadata = {
                    "url": row['url'] or "",
                    "title": row['title'] or "",
                    "format": row['format'] or "",
                    "timestamp": row['timestamp'].isoformat() if row['timestamp'] else "",
                    "source": row['source'] or "",
                    "image_count": row['image_count'] or 0,
                    "has_images": row['has_images'] or False,
                    "domain": row['domain'] or "",
                    # images_json left raw here for debugging, but we'll provide normalized images below
                    "images_json": raw_images_field if raw_images_field is not None else [],
                }

                # Convert raw images field into normalized list of {url, alt, caption}
                images_list = self._coerce_images_from_db(raw_images_field)
                # Attach normalized images
                metadata["images"] = images_list

                distance = float(row['distance'])
                relevance_score = max(0.0, 1.0 / (1.0 + distance))

                if relevance_score >= self.search_config["min_relevance_threshold"]:
                    scored_results.append({
                        "content": content,
                        "metadata": metadata,
                        "relevance_score": relevance_score,
                    })

            scored_results.sort(key=lambda x: x["relevance_score"], reverse=True)

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

        normalized: List[Dict[str, Any]] = []

        try:
        # Parse the field into a list
            if isinstance(images_field, list):
                entries = images_field
            elif isinstance(images_field, str) and images_field.strip():
                try:
                    entries = json.loads(images_field)
                except Exception:
                # Maybe comma-separated URLs
                    entries = [s.strip() for s in images_field.split(",") if s.strip()]
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
                    else:
                    # Treat as prompt / label → generate placeholder
                        placeholder_url = self._placeholder_for_prompt(item)
                        normalized.append({
                        "url": placeholder_url,
                        "alt": item[:80],
                        "caption": ""
                    })
                    continue

            # Handle dict-like objects
                if isinstance(item, dict):
                    url = item.get("url") or item.get("image_url") or None
                    alt = item.get("alt") or item.get("title") or item.get("caption") or ""
                    caption = item.get("caption") or item.get("title") or ""

                # CRITICAL: Handle image_prompt conversion
                    if not url and item.get("image_prompt"):
                        prompt = item.get("image_prompt")
                        url = self._placeholder_for_prompt(prompt)
                        alt = f"Visual Guide: {prompt[:60]}"
                        logger.debug(f"Converted image_prompt to placeholder URL: {url[:60]}...")

                # Skip if still no URL
                    if not url:
                        logger.debug(f"Skipping image entry with no URL or prompt: {item}")
                        continue

                    normalized.append({
                    "url": url,
                    "alt": alt[:200] if isinstance(alt, str) else "",
                    "caption": caption[:500] if isinstance(caption, str) else ""
                })
                    continue

        except Exception as e:
            logger.error(f"⚠️ _coerce_images_from_db failed: {e}", exc_info=True)

        logger.debug(f"Normalized {len(normalized)} images from DB field")
        return normalized

    def _placeholder_for_prompt(self, prompt: str, size: Tuple[int, int] = (900, 400)) -> str:
    
        try:
        # Clean prompt text
            text = prompt.strip()[:120]
        
        # Remove special characters
            text = re.sub(r'[^\w\s-]', '', text)
            text = re.sub(r'\s+', ' ', text)
        
        # Ensure non-empty
            if not text:
                text = "Visual Guide"
        
        # Truncate if too long
            if len(text) > 50:
                text = text[:47] + "..."
        
        # URL encode
            encoded = quote_plus(text)
        
            width, height = size
        
        # Professional color scheme
            bg_color = "4A90E2"  # Professional blue
            text_color = "FFFFFF"  # White text
        
        # Generate URL
            url = (
            f"https://via.placeholder.com/{width}x{height}/{bg_color}/{text_color}"
            f"?text={encoded}"
            )
        
            logger.debug(f"Generated placeholder URL: {url[:80]}...")
            return url
        
        except Exception as e:
            logger.error(f"Placeholder generation error: {e}")
            return "https://via.placeholder.com/900x400/4A90E2/FFFFFF?text=Image"

    def _preprocess_query(self, query: str) -> Tuple[str, List[str]]:
        """Preprocess query and extract key terms."""
        if not query:
            return "", []
        cleaned = re.sub(r'\s+', ' ', query.strip().lower())
        words = re.findall(r'\b\w+\b', cleaned)
        key_terms = [w for w in words if len(w) > 3 and w not in self.stopwords]
        return cleaned, key_terms[:15]

    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics. Auto-attempts schema creation if table missing."""
        if not self.pool or not self._connection_established:
            return {
                "document_count": 0,
                "collection_name": "unavailable",
                "status": "unavailable",
                "message": "PostgreSQL not connected.",
            }

        try:
            async with self.pool.acquire() as conn:
                # If table missing, attempt to ensure schema (self-healing)
                if not await self._table_exists(conn):
                    logger.warning(f"⚠️ Table '{self.table_name}' missing when getting stats - attempting to ensure schema")
                    await self._ensure_schema()
                    # If still missing, return initializing state
                    if not await self._table_exists(conn):
                        logger.error(f"❌ Table '{self.table_name}' still missing after _ensure_schema()")
                        return {
                            "document_count": 0,
                            "collection_name": self.table_name,
                            "status": "initializing",
                            "message": "Schema creation in progress or failed."
                        }

                count_sql = f"SELECT COUNT(*) FROM {self.table_name};"
                count = await conn.fetchval(count_sql)

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
            logger.exception(f"❌ Error getting stats: {e}")
            return {
                "document_count": 0,
                "status": "error",
                "error": str(e),
            }

    async def delete_collection(self) -> None:
        """Delete the table."""
        try:
            if not self.pool:
                return
            async with self.pool.acquire() as conn:
                await conn.execute(f"DROP TABLE IF EXISTS {self.table_name} CASCADE;")
                logger.info(f"✅ Deleted table: {self.table_name}")
        except Exception as e:
            logger.exception(f"❌ Error deleting table: {e}")

    async def close(self) -> None:
        """Close PostgreSQL connection pool."""
        try:
            if self.pool:
                await self.pool.close()
                self.pool = None
                self._connection_established = False
                logger.info(f"✅ Closed PostgreSQL connection pool")
        except Exception as e:
            logger.exception(f"❌ Error closing connection: {e}")


# Global instance
postgres_service = PostgresService()
