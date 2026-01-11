# milvus_service.py - PRODUCTION FINAL VERSION
# Combines optimized search parameters with complete implementation
# Best of both: aggressive search + comprehensive features

import json
import os
import urllib.parse
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Set
import re
import math
import asyncio
import socket
import logging
from collections import defaultdict

from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
)
from pymilvus.exceptions import MilvusException, DataNotMatchException

from app.core.config import settings
from app.services.ai_service import ai_service

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("MILVUS_LOG_LEVEL", "INFO"))


class MilvusService:
    """
    PRODUCTION-GRADE Milvus service with OPTIMIZED search and complete features.
    
    Key optimizations:
    - AGGRESSIVE search limits (200+ initial results)
    - LOWER relevance threshold (0.08) for better recall
    - HIGHER search parameters (ef=128, nprobe=64)
    - Enhanced image extraction and metadata handling
    - Comprehensive error recovery and connection stability
    - Query expansion and semantic reranking
    - Diversity filtering for result quality
    """

    def __init__(self):
        self.collection: Optional[Collection] = None
        self.collection_name: str = getattr(settings, "MILVUS_COLLECTION", "enterprise_rag")

    # Connection params
        self.milvus_uri: Optional[str] = os.getenv("MILVUS_URI")
        self._env_host = os.getenv("MILVUS_HOST", "milvus")
        self._env_port = os.getenv("MILVUS_PORT", "19530")
        self.milvus_user: str = os.getenv("MILVUS_USER", "")
        self.milvus_password: str = os.getenv("MILVUS_PASSWORD", "")
        self.milvus_alias: str = os.getenv("MILVUS_ALIAS", "default")

        self.debug_dump: bool = os.getenv("MILVUS_DEBUG_DUMP", "false").lower() == "true"

    # Index configuration
        self.index_type = os.getenv("MILVUS_INDEX_TYPE", "HNSW").upper()
        self.default_index_params = {
        "HNSW": {
            "index_type": "HNSW",
            "metric_type": "L2",
            "params": {
                "M": int(os.getenv("MILVUS_HNSW_M", "16")),
                "efConstruction": int(os.getenv("MILVUS_HNSW_EFC", "200"))
            }
        },
        "IVF_FLAT": {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": int(os.getenv("MILVUS_IVF_NLIST", "2048"))}
        },
    }

    # FIXED: Use environment variables for search parameters
        self.search_ef = int(os.getenv("MILVUS_SEARCH_EF", "128"))
        self.search_nprobe = int(os.getenv("MILVUS_NPROBE", "64"))

    # FIXED: Ensure all search config uses environment variables
        self.search_config = {
        "min_relevance_threshold": float(os.getenv("MILVUS_MIN_RELEVANCE", "0.08")),
        "max_initial_results": int(os.getenv("MILVUS_MAX_INITIAL_RESULTS", "200")),
        "rerank_top_k": int(os.getenv("MILVUS_RERANK_TOP_K", "100")),
        "enable_query_expansion": os.getenv("MILVUS_ENABLE_QUERY_EXPANSION", "true").lower() == "true",
        "enable_semantic_rerank": os.getenv("MILVUS_ENABLE_SEMANTIC_RERANK", "true").lower() == "true",
        "enable_context_enrichment": os.getenv("MILVUS_ENABLE_CONTEXT_ENRICHMENT", "true").lower() == "true",
        "diversity_factor": float(os.getenv("MILVUS_DIVERSITY_FACTOR", "0.2")),
    }

        self.embedding_dim: int = int(os.getenv("EMBEDDING_DIMENSION", "4096"))
    
    # Log configuration for verification
        logger.info(f"🔧 MilvusService PRODUCTION init:")
        logger.info(f"   - Embedding dimension: {self.embedding_dim}")
        logger.info(f"   - Index type: {self.index_type}")
        logger.info(f"   - Search EF: {self.search_ef}")
        logger.info(f"   - Search nprobe: {self.search_nprobe}")
        logger.info(f"   - Min relevance: {self.search_config['min_relevance_threshold']}")
        logger.info(f"   - Max initial results: {self.search_config['max_initial_results']}")
        logger.info(f"   - Rerank top K: {self.search_config['rerank_top_k']}")

        self._connection_established: bool = False

    # Enhanced stopwords
        self.stopwords: Set[str] = {
        'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one',
        'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see',
        'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use', 'that',
        'with', 'have', 'this', 'will', 'your', 'from', 'they', 'know', 'want', 'been', 'good',
        'much', 'some', 'time', 'very', 'when', 'come', 'here', 'just', 'like', 'long', 'make',
        'many', 'over', 'such', 'take', 'than', 'them', 'well', 'were', 'what', 'about', 'into'
    }

    # ========== Connection Helpers ==========
    def _try_socket(self, host: str, port: int, timeout: float = 1.0) -> bool:
        """Test socket connectivity."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            sock.connect((host, port))
            sock.close()
            return True
        except Exception:
            return False

    def _choose_host_candidate(self) -> Tuple[Optional[str], Optional[int]]:
        """Select host candidate intelligently."""
        if self.milvus_uri:
            return None, None

        candidates = []
        if self._env_host:
            candidates.append(self._env_host)
        candidates.extend(["milvus", "127.0.0.1", "localhost"])

        try:
            port_int = int(self._env_port)
        except Exception:
            port_int = 19530

        for host in candidates:
            try:
                if self._try_socket(host, port_int, timeout=0.8):
                    logger.debug(f"Socket probe succeeded: {host}:{port_int}")
                    return host, port_int
            except Exception:
                continue

        return self._env_host if self._env_host else "127.0.0.1", port_int

    # ========== Lifecycle ==========
    async def initialize(self) -> None:
        """Connect to Milvus and initialize collection."""
        if self.collection is not None:
            return

        max_retries = int(os.getenv("MILVUS_CONNECT_RETRIES", "5"))
        backoff = float(os.getenv("MILVUS_CONNECT_BACKOFF", "2"))

        for attempt in range(1, max_retries + 1):
            try:
                connect_kwargs = {"alias": self.milvus_alias}
                if self.milvus_uri:
                    connect_kwargs["uri"] = self.milvus_uri
                    logger.info("Using MILVUS_URI for connection")
                else:
                    host, port = self._choose_host_candidate()
                    connect_kwargs["host"] = host
                    connect_kwargs["port"] = int(port)

                if self.milvus_user:
                    connect_kwargs["user"] = self.milvus_user
                if self.milvus_password:
                    connect_kwargs["password"] = self.milvus_password

                logger.info(f"Connecting to Milvus (attempt {attempt}/{max_retries})")
                connections.connect(**connect_kwargs)
                self._connection_established = True
                logger.info(f"✅ Connected to Milvus (alias={self.milvus_alias})")

                # Ensure collection exists
                if utility.has_collection(self.collection_name, using=self.milvus_alias):
                    self.collection = Collection(self.collection_name, using=self.milvus_alias)
                    logger.info(f"✅ Found existing collection: {self.collection_name}")
                    
                    # Check dimension consistency
                    try:
                        schema = getattr(self.collection, "schema", None)
                        detected_dim = None
                        if schema:
                            for f in getattr(schema, "fields", []):
                                if getattr(f, "name", "") == "embedding":
                                    params = getattr(f, "params", None) or getattr(f, "type_params", None)
                                    if isinstance(params, dict):
                                        detected_dim = params.get("dim") or params.get("dimension")
                                    else:
                                        detected_dim = getattr(f, "dim", None)
                                    break

                        if detected_dim:
                            detected_dim = int(detected_dim)
                            logger.info(f"🔍 Detected collection dim={detected_dim}, expected={self.embedding_dim}")
                            if detected_dim != self.embedding_dim:
                                count = 0
                                try:
                                    count = self.collection.num_entities
                                except Exception:
                                    pass
                                if count == 0:
                                    logger.warning("Dimension mismatch but collection empty — rebuilding")
                                    utility.drop_collection(self.collection_name, using=self.milvus_alias)
                                    await self._create_collection()
                                    self.collection = Collection(self.collection_name, using=self.milvus_alias)
                                else:
                                    raise ValueError(
                                        f"Embedding dimension mismatch: collection={detected_dim}, "
                                        f"env={self.embedding_dim} (count={count}). Manual migration required."
                                    )
                    except ValueError:
                        raise
                    except Exception as e:
                        logger.warning(f"Schema inspection warning: {e}")
                else:
                    logger.info(f"📦 Creating collection: {self.collection_name}")
                    await self._create_collection()
                    self.collection = Collection(self.collection_name, using=self.milvus_alias)

                # Ensure index exists
                try:
                    existing_indexes = getattr(self.collection, "indexes", []) or []
                    if not existing_indexes:
                        idx_conf = self.default_index_params.get(self.index_type)
                        self.collection.create_index(field_name="embedding", index_params=idx_conf)
                        logger.info(f"✅ Created index: {idx_conf}")
                    else:
                        logger.debug("Index already present on collection")
                except Exception as e:
                    logger.warning(f"⚠️ Index check/create failed: {e}")

                # Load collection with retries
                for load_attempt in range(6):
                    try:
                        self.collection.load()
                        logger.info(f"✅ Collection loaded (attempt {load_attempt + 1})")
                        break
                    except Exception as e:
                        logger.warning(f"⚠️ Load attempt {load_attempt+1}/6 failed: {e}")
                        await asyncio.sleep(2 ** load_attempt)
                else:
                    logger.error(f"❌ Failed to load collection after retries")

                break  # Success

            except Exception as e:
                logger.exception(f"❌ Init attempt {attempt}/{max_retries} failed: {e}")
                self._connection_established = False
                self.collection = None
                if attempt >= max_retries:
                    logger.error("❌ Initialization aborted after max retries")
                    return
                await asyncio.sleep(backoff * attempt)

    async def _create_collection(self) -> None:
        """Create collection with optimized schema."""
        logger.info(f"🔧 Creating collection schema (dim={self.embedding_dim})")
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="format", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="timestamp", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="content_length", dtype=DataType.INT64),
            FieldSchema(name="word_count", dtype=DataType.INT64),
            FieldSchema(name="image_count", dtype=DataType.INT64),
            FieldSchema(name="has_images", dtype=DataType.BOOL),
            FieldSchema(name="domain", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="content_hash", dtype=DataType.INT64),
            FieldSchema(name="images_json", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="key_terms", dtype=DataType.VARCHAR, max_length=2000),
        ]

        schema = CollectionSchema(
            fields=fields,
            description=f"Enterprise RAG - Optimized Search - Dim: {self.embedding_dim}",
            auto_id=False
        )

        # Drop if exists
        try:
            if utility.has_collection(self.collection_name, using=self.milvus_alias):
                logger.warning(f"⚠️ Dropping existing collection: {self.collection_name}")
                utility.drop_collection(self.collection_name, using=self.milvus_alias)
        except Exception as e:
            logger.debug(f"No existing collection to drop: {e}")

        self.collection = Collection(name=self.collection_name, schema=schema, using=self.milvus_alias)
        logger.info(f"✅ Collection created")

        # Create optimized index
        try:
            idx_conf = self.default_index_params.get(self.index_type)
            self.collection.create_index(field_name="embedding", index_params=idx_conf)
            logger.info(f"✅ Index created: {idx_conf}")
        except Exception as e:
            logger.warning(f"⚠️ Index creation warning: {e}")

    # ========== Query Processing ==========
    def _preprocess_query(self, query: str) -> Tuple[str, List[str]]:
        """Preprocess query and extract key terms."""
        if not query:
            return "", []
        
        original_query = query.strip()
        cleaned_query = re.sub(r'\s+', ' ', original_query.lower())
        
        key_terms: List[str] = []
        
        # Extract technical patterns
        technical_patterns = [
            r'\b[A-Z]{2,}\b',
            r'\b\w+(?:ing|ion|tion|sion|ness|ment|able|ible|ance|ence)\b',
            r'\b\w+[-_]\w+(?:[-_]\w+)*\b',
            r'\b\d+(?:\.\d+)*\w*\b',
            r'\b(?:how|what|where|when|why|which)\s+\w+\b',
            r'\b[a-z]+\.[a-z]+\b',
        ]
        
        for pattern in technical_patterns:
            matches = re.findall(pattern, original_query)
            key_terms.extend(matches)
        
        # Extract significant words
        words = re.findall(r'\b\w+\b', cleaned_query)
        key_terms.extend([w for w in words if len(w) > 3 and w not in self.stopwords])
        
        # Deduplicate
        seen = set()
        unique_terms = []
        for term in key_terms:
            tl = term.lower()
            if tl not in seen:
                seen.add(tl)
                unique_terms.append(term)
        
        return cleaned_query, unique_terms[:15]

    def _calculate_enhanced_relevance(self, document: str, metadata: Dict,
                                     query: str, key_terms: List[str],
                                     distance: float) -> float:
        """Calculate comprehensive relevance score."""
        if not document:
            return 0.0

        doc_lower = document.lower()
        query_lower = query.lower()

        # Semantic score
        semantic_score = max(0.0, 1.0 / (1.0 + distance))

        # Term matching with position awareness
        term_score = 0.0
        if key_terms:
            term_matches = 0
            early_matches = 0
            doc_len = len(doc_lower) if doc_lower else 0
            for term in key_terms:
                term_lower = term.lower()
                if term_lower in doc_lower:
                    term_matches += 1
                    pos = doc_lower.find(term_lower)
                    if pos >= 0 and doc_len > 0 and pos < doc_len * 0.3:
                        early_matches += 1
            term_score = min(1.0, (term_matches / len(key_terms)) + (early_matches * 0.1))

        # Phrase matching
        phrase_score = 0.0
        query_words = query_lower.split()
        if len(query_words) > 1:
            if query_lower in doc_lower:
                phrase_score = 0.5
            else:
                bigram_matches = 0
                for i in range(len(query_words) - 1):
                    bigram = f"{query_words[i]} {query_words[i+1]}"
                    if bigram in doc_lower:
                        bigram_matches += 1
                if bigram_matches > 0:
                    phrase_score = min(0.4, bigram_matches * 0.2)

        # Quality score
        quality_score = 0.0
        doc_length = len(document)
        word_count = len(document.split())
        if 200 <= doc_length <= 5000:
            quality_score += 0.2
        elif 100 <= doc_length < 200:
            quality_score += 0.1
        elif 5000 < doc_length <= 8000:
            quality_score += 0.15
        
        if word_count > 0:
            unique_words = len(set(document.lower().split()))
            diversity = unique_words / word_count
            if 0.4 <= diversity <= 0.8:
                quality_score += 0.1

        # Title score
        title_score = 0.0
        if metadata.get("title"):
            title_lower = str(metadata["title"]).lower()
            title_matches = sum(1 for term in key_terms if term.lower() in title_lower)
            if title_matches > 0:
                title_score = min(0.25, title_matches * 0.15)
            if query_lower in title_lower:
                title_score += 0.15

        # URL score
        url_score = 0.0
        if metadata.get("url"):
            url_lower = str(metadata["url"]).lower()
            url_matches = sum(1 for term in key_terms if term.lower() in url_lower)
            if url_matches > 0:
                url_score = min(0.15, url_matches * 0.08)

        # Recency score
        recency_score = 0.0
        if metadata.get("timestamp"):
            try:
                doc_date = datetime.fromisoformat(str(metadata["timestamp"]).replace("Z", "+00:00"))
                days_old = (datetime.now().replace(tzinfo=doc_date.tzinfo) - doc_date).days
                if days_old < 7:
                    recency_score = 0.10
                elif days_old < 30:
                    recency_score = 0.07
                elif days_old < 90:
                    recency_score = 0.04
                elif days_old < 180:
                    recency_score = 0.02
            except Exception:
                pass

        # Format score
        format_score = 0.0
        doc_format = metadata.get("format", "").lower()
        if doc_format in ["pdf", "docx", "html"]:
            format_score = 0.05

        # Image bonus
        image_score = 0.0
        if metadata.get("has_images") or metadata.get("image_count", 0) > 0:
            image_score = 0.05

        # Weighted final score
        final_score = (
            semantic_score * 0.30 +
            term_score * 0.22 +
            phrase_score * 0.18 +
            quality_score * 0.08 +
            title_score * 0.10 +
            url_score * 0.05 +
            recency_score * 0.04 +
            format_score * 0.02 +
            image_score * 0.01
        )
        
        return min(1.0, final_score)

    async def _expand_query(self, original_query: str, key_terms: List[str]) -> str:
        """Expand query with related terms."""
        if not self.search_config["enable_query_expansion"]:
            return original_query
        
        try:
            expansion_prompt = (
                f"Given the search query '{original_query}', provide 3-5 related technical terms, "
                f"synonyms, or variations that would help find relevant information. "
                f"Return ONLY the terms, comma-separated, no explanations."
            )
            expanded_terms = await ai_service._call_chat_with_retries(
                expansion_prompt,
                max_tokens=60,
                temperature=0.3,
                system_message="You are a search query expansion expert. Provide only relevant terms.",
                timeout=30  # Increased from 15s to allow for slower API responses
            )
            if expanded_terms:
                additional_terms = [t.strip() for t in expanded_terms.strip().split(',') if t.strip()]
                original_words = set(original_query.lower().split())
                new_terms = [t for t in additional_terms if t.lower() not in original_words and len(t) > 2]
                if new_terms:
                    expanded_query = f"{original_query} {' '.join(new_terms[:4])}"
                    logger.info(f"Query expanded: '{original_query}' + {new_terms[:4]}")
                    return expanded_query
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
        
        return original_query

    def _apply_diversity_filtering(self, results: List[Dict[str, Any]], 
                                   diversity_factor: float = 0.2) -> List[Dict[str, Any]]:
        """Apply diversity filtering to avoid redundant results."""
        if len(results) <= 3:
            return results
        
        diverse_results = [results[0]]
        for candidate in results[1:]:
            candidate_content = candidate.get("content", "").lower()
            candidate_terms = set(re.findall(r'\b\w+\b', candidate_content))
            
            is_diverse = True
            for selected in diverse_results:
                selected_content = selected.get("content", "").lower()
                selected_terms = set(re.findall(r'\b\w+\b', selected_content))
                
                if selected_terms and candidate_terms:
                    overlap = len(candidate_terms & selected_terms)
                    union = len(candidate_terms | selected_terms)
                    similarity = overlap / union if union > 0 else 0
                    if similarity > (1 - diversity_factor):
                        is_diverse = False
                        break
            
            if is_diverse:
                diverse_results.append(candidate)
        
        logger.info(f"Diversity filtering: {len(results)} -> {len(diverse_results)} results")
        return diverse_results

    # ========== Search Documents ==========
    async def search_documents(self, query: str, n_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        PRODUCTION OPTIMIZED: Perform search with aggressive parameters for better recall.
        """
        if not query:
            return []

        if not self.collection and not self._connection_established:
            asyncio.create_task(self.initialize())
            logger.info("⏳ Milvus init scheduled (non-blocking)")
            return []
        if not self.collection:
            logger.warning("⚠️ Milvus unavailable")
            return []

        try:
            cleaned_query, key_terms = self._preprocess_query(query)
            if not cleaned_query:
                return []

            logger.info(f"🔍 Search: '{query[:60]}...' | Key terms: {key_terms[:5]}")

            # Query expansion
            expanded_query = cleaned_query
            if self.search_config["enable_query_expansion"]:
                expanded_query = await self._expand_query(cleaned_query, key_terms)

            # OPTIMIZED: Use HIGHER default limits
            default_k = int(getattr(settings, "MILVUS_QUERY_TOP_K", "100"))  # Increased from 50
            initial_k = min(self.search_config["max_initial_results"], int(n_results or default_k) * 2)
            final_k = int(n_results or default_k)

            logger.info(f"📊 Search limits: initial={initial_k}, final={final_k}")

            # Generate query embedding
            query_embeddings = await ai_service.generate_embeddings([expanded_query])
            if not query_embeddings:
                logger.warning("⚠️ Failed to generate embeddings")
                return []

            # Validate embedding dimension
            query_emb_dim = len(query_embeddings[0]) if isinstance(query_embeddings[0], (list, tuple)) else 0
            if query_emb_dim != self.embedding_dim:
                logger.error(f"❌ Query embedding dim mismatch: got {query_emb_dim}, expected {self.embedding_dim}")
                return []

            # OPTIMIZED: Build search params with HIGHER limits
            if self.index_type == "HNSW":
                search_params = {"metric_type": "L2", "params": {"ef": self.search_ef}}
            else:
                search_params = {"metric_type": "L2", "params": {"nprobe": self.search_nprobe}}

            logger.info(f"🔍 Search params: {search_params}")

            # Attempt search with retry
            search_attempts = 3
            backoff = float(os.getenv("MILVUS_CONNECT_BACKOFF", "2"))
            results = None
            last_exception = None

            for attempt in range(1, search_attempts + 1):
                try:
                    results = self.collection.search(
                        data=query_embeddings,
                        anns_field="embedding",
                        param=search_params,
                        limit=initial_k,
                        output_fields=[
                            "content", "url", "title", "format", "timestamp", "source",
                            "content_length", "word_count", "image_count", "has_images",
                            "domain", "content_hash", "images_json", "key_terms"
                        ]
                    )
                    last_exception = None
                    break
                except MilvusException as me:
                    last_exception = me
                    msg = str(me)
                    if "collection not loaded" in msg or getattr(me, "code", None) == 101:
                        logger.warning(f"⚠️ Collection not loaded, loading... (attempt {attempt})")
                        try:
                            self.collection.load()
                            logger.info("✅ Collection load triggered")
                            await asyncio.sleep(backoff * attempt)
                        except Exception as le:
                            logger.warning(f"⚠️ Failed to load collection: {le}")
                    else:
                        logger.exception(f"❌ Search error: {me}")
                        break
                except Exception as e:
                    last_exception = e
                    logger.exception(f"❌ Unexpected search error: {e}")
                    await asyncio.sleep(backoff * attempt)
                    continue

            if last_exception:
                logger.error(f"❌ Search aborted: {last_exception}")
                return []

            if not results or len(results) == 0 or len(results[0]) == 0:
                logger.info("ℹ️ No documents found")
                return []

            # Score and filter results with LOWER threshold
            scored_results = []
            for hit in results[0]:
                entity = hit.entity
                def get_field(name, default=None):
                    try:
                        return entity.get(name)
                    except Exception:
                        return default

                content = get_field("content", "")
                metadata = {
                    "url": get_field("url", ""),
                    "title": get_field("title", ""),
                    "format": get_field("format", ""),
                    "timestamp": get_field("timestamp", ""),
                    "source": get_field("source", ""),
                    "image_count": get_field("image_count", 0),
                    "has_images": get_field("has_images", False),
                    "domain": get_field("domain", ""),
                    "images_json": get_field("images_json", ""),
                    }
                distance = getattr(hit, "distance", 0.0)

                relevance_score = self._calculate_enhanced_relevance(
                    document=content,
                    metadata=metadata,
                    query=query,
                    key_terms=key_terms,
                    distance=distance
                )

                # OPTIMIZED: Use LOWER threshold for better recall
                if relevance_score >= self.search_config["min_relevance_threshold"]:
                    metadata["images"] = self._coerce_images_from_meta(metadata)
                    scored_results.append({
                        "content": content,
                        "metadata": metadata,
                        "relevance_score": relevance_score,
                    })

            scored_results.sort(key=lambda x: x["relevance_score"], reverse=True)
            logger.info(f"📊 Found {len(scored_results)} results above threshold {self.search_config['min_relevance_threshold']}")

            # Apply diversity filtering
            if self.search_config["enable_context_enrichment"] and len(scored_results) > final_k:
                scored_results = self._apply_diversity_filtering(
                    scored_results,
                    self.search_config["diversity_factor"]
                )

            # Apply semantic reranking
            if self.search_config["enable_semantic_rerank"] and len(scored_results) > final_k:
                rerank_k = min(len(scored_results), self.search_config["rerank_top_k"])
                scored_results = await self._semantic_rerank(scored_results[:rerank_k], query, rerank_k)

            final_results = []
            for r in scored_results[:final_k]:
                final_results.append({
                    "content": r["content"],
                    "relevance_score": float(r["relevance_score"]),
                    "metadata": r["metadata"]
                    })
            return final_results


        except Exception as e:
            logger.exception(f"❌ Search error: {e}")
            return []

    async def _semantic_rerank(self, results: List[Dict], query: str, top_k: int) -> List[Dict]:
        """Semantic reranking using embeddings."""
        try:
            contents = [result["content"][:3000] for result in results[:top_k]]
            all_texts = [query] + contents
            embeddings = await ai_service.generate_embeddings(all_texts)
            
            if not embeddings or len(embeddings) != len(all_texts):
                logger.warning("⚠️ Reranking failed: embedding generation error")
                return results[:top_k]
            
            query_embedding = embeddings[0]
            content_embeddings = embeddings[1:]

            def cosine_similarity(a, b):
                if not a or not b or len(a) != len(b):
                    return 0.0
                dot = sum(x * y for x, y in zip(a, b))
                norm_a = math.sqrt(sum(x * x for x in a))
                norm_b = math.sqrt(sum(x * x for x in b))
                return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

            for i, result in enumerate(results[:len(content_embeddings)]):
                semantic_sim = cosine_similarity(query_embedding, content_embeddings[i])
                original_score = result["relevance_score"]
                result["semantic_rerank_score"] = original_score * 0.65 + semantic_sim * 0.35
                result["semantic_similarity"] = semantic_sim

            results.sort(key=lambda x: x.get("semantic_rerank_score", x["relevance_score"]), reverse=True)
            logger.info("✅ Applied semantic reranking")
        except Exception as e:
            logger.warning(f"⚠️ Reranking failed: {e}")
        
        return results[:top_k]

    # ========== Add Documents ==========
    def _flatten_to_str(self, v: Any) -> str:
        """Flatten value to string."""
        if v is None:
            return ""
        if isinstance(v, (list, tuple)):
            try:
                return "-".join([str(x) for x in v])
            except Exception:
                return str(v)
        return str(v)

    def _ensure_embedding_list(self, embeddings: List[Any]) -> List[List[float]]:

        clean_embeds: List[List[float]] = []

        for idx, emb in enumerate(embeddings or []):
            try:
            # None -> zero vector
                if emb is None:
                    logger.warning(f"⚠️ Embedding {idx} is None, using zero vector")
                    clean_embeds.append([0.0] * self.embedding_dim)
                    continue

            # Convert to list of floats
                if isinstance(emb, (list, tuple)):
                    float_list = []
                    for i, v in enumerate(emb):
                        try:
                            float_list.append(float(v))
                        except Exception:
                            logger.warning(f"⚠️ Embedding {idx} element {i} not convertible to float, using 0.0")
                            float_list.append(0.0)
                else:
                    # Scalar value -> make a vector with scalar then zeros
                    try:
                        single_val = float(emb)
                        logger.warning(f"⚠️ Embedding {idx} is scalar, converting to vector")
                        float_list = [single_val]
                    except Exception:
                        logger.warning(f"⚠️ Embedding {idx} scalar conversion failed, using zero vector")
                        clean_embeds.append([0.0] * self.embedding_dim)
                        continue

            # Fix dimension: pad or truncate
                if len(float_list) < self.embedding_dim:
                    pad_len = self.embedding_dim - len(float_list)
                    float_list.extend([0.0] * pad_len)
                    logger.debug(f"⚠️ Padded embedding {idx} from {len(float_list) - pad_len} to {self.embedding_dim}")
                elif len(float_list) > self.embedding_dim:
                    float_list = float_list[:self.embedding_dim]
                    logger.debug(f"⚠️ Truncated embedding {idx} to {self.embedding_dim}")

            # Normalize (L2)
                norm = math.sqrt(sum(x * x for x in float_list))
                if norm > 0:
                    float_list = [x / norm for x in float_list]

                clean_embeds.append(float_list)

            except Exception as e:
                logger.error(f"❌ Embedding {idx} processing failed: {e}")
                clean_embeds.append([0.0] * self.embedding_dim)

        return clean_embeds


    def _validate_entities_data(self, entities: Dict[str, List[Any]]) -> Tuple[bool, str]:
        """Validate entities data structure."""
        if not isinstance(entities, dict):
            return False, "entities must be a dict"
        lengths = []
        for k, v in entities.items():
            if not isinstance(v, list):
                return False, f"field '{k}' must be a list, got {type(v)}"
            lengths.append(len(v))
        if not lengths:
            return False, "no columns present"
        if len(set(lengths)) != 1:
            return False, f"column length mismatch: {lengths}"
        return True, ""

    async def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Add documents to collection with full implementation."""
        if not self.collection and not self._connection_established:
            asyncio.create_task(self.initialize())
            logger.info("⏳ Milvus init scheduled (non-blocking)")
            return []
        if not self.collection:
            logger.warning("⚠️ Milvus unavailable, skipping document addition")
            return []
        
        try:
            ids: List[str] = []
            texts: List[str] = []
            entities_data: Dict[str, List[Any]] = {
                "id": [], "embedding": [], "content": [], "url": [], "title": [],
                "format": [], "timestamp": [], "source": [], "content_length": [],
                "word_count": [], "image_count": [], "has_images": [], "domain": [],
                "content_hash": [], "images_json": [], "key_terms": [],
            }

            for doc in documents or []:
                doc_id = str(uuid.uuid4())
                ids.append(doc_id)
                content = self._decode_to_text(doc.get("content", ""))
                content = self._clean_and_normalize_content(content)
                content = self._soft_truncate(content, limit=8000)
                texts.append(content)
                
                normalized_images = self._normalize_images(
                    images=doc.get("images", []),
                    page_url=doc.get("url", ""),
                    content_snippet=content[:400],
                    max_images=20
                )
                
                entities_data["id"].append(doc_id)
                entities_data["content"].append(content[:65535])
                entities_data["url"].append(str(doc.get("url", ""))[:2000])
                entities_data["title"].append(str(doc.get("title", ""))[:500])
                entities_data["format"].append(str(doc.get("format", "text"))[:100])
                entities_data["timestamp"].append(doc.get("timestamp", "") or datetime.now().isoformat())
                entities_data["source"].append(str(doc.get("source", "web_scraping"))[:100])
                entities_data["content_length"].append(len(content))
                entities_data["word_count"].append(len(content.split()))
                entities_data["image_count"].append(len(normalized_images))
                entities_data["has_images"].append(bool(normalized_images))
                entities_data["domain"].append(self._extract_domain(doc.get("url", ""))[:500])
                entities_data["content_hash"].append(abs(hash(content)) % (10**8))
                entities_data["images_json"].append(json.dumps(normalized_images, ensure_ascii=False)[:65535])
                entities_data["key_terms"].append(json.dumps(self._extract_key_terms(content)[:20], ensure_ascii=False)[:2000])

            # Generate embeddings with retries
            max_retries = 3
            embeddings = []
            for attempt in range(max_retries):
                try:
                    logger.info(f"🔄 Generating embeddings (attempt {attempt + 1}/{max_retries})...")
                    embeddings = await ai_service.generate_embeddings(texts)
                    if embeddings and len(embeddings) == len(texts):
                        first_emb_dim = len(embeddings[0]) if isinstance(embeddings[0], (list, tuple)) else 0
                        if first_emb_dim != self.embedding_dim:
                            logger.error(f"❌ Embedding dimension mismatch: AI={first_emb_dim}, expected={self.embedding_dim}")
                            if attempt < max_retries - 1:
                                logger.warning("⚠️ Retrying...")
                                await asyncio.sleep(2 ** attempt)
                                continue
                            else:
                                raise ValueError(f"Embedding dimension mismatch: AI={first_emb_dim}, Milvus={self.embedding_dim}")
                        logger.info(f"✅ Generated {len(embeddings)} embeddings (dim={first_emb_dim})")
                        break
                    else:
                        logger.warning(f"⚠️ Embedding attempt {attempt + 1} length mismatch")
                except Exception as e:
                    logger.warning(f"⚠️ Embedding attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                    else:
                        raise

            if not embeddings or len(embeddings) != len(texts):
                logger.error("❌ Failed to generate embeddings")
                return []

            embeddings = self._ensure_embedding_list(embeddings)
            entities_data["embedding"] = embeddings
            entities_data["id"] = [self._flatten_to_str(i) for i in entities_data["id"]]

            valid, msg = self._validate_entities_data(entities_data)
            if not valid:
                logger.error(f"❌ Entities validation failed: {msg}")
                return []

            # Build rows
            n_rows = len(entities_data["id"])
            rows: List[Dict[str, Any]] = []
            fields = list(entities_data.keys())
            
            for i in range(n_rows):
                row = {}
                for f in fields:
                    value = entities_data[f][i]
                    if f == "id":
                        row["id"] = self._flatten_to_str(value)
                    elif f == "embedding":
                        emb = entities_data["embedding"][i]
                        if isinstance(emb, (list, tuple)):
                            row["embedding"] = [float(x) for x in emb]
                        else:
                            row["embedding"] = [float(emb)] + [0.0] * (self.embedding_dim - 1)
                        if len(row["embedding"]) != self.embedding_dim:
                            logger.error(f"❌ Row {i} embedding wrong dim")
                            raise ValueError(f"Embedding dimension error at row {i}")
                    elif f in ("content", "url", "title", "format", "timestamp", "source", "domain", "images_json", "key_terms"):
                        row[f] = str(value) if value is not None else ""
                    elif f in ("content_length", "word_count", "image_count", "content_hash"):
                        try:
                            row[f] = int(value)
                        except Exception:
                            row[f] = 0
                    elif f == "has_images":
                        row[f] = bool(value)
                    else:
                        row[f] = value
                rows.append(row)

            # Insert with retries
            insert_attempts = 3
            for attempt in range(insert_attempts):
                try:
                    logger.info(f"📤 Inserting {len(rows)} rows (attempt {attempt+1}/{insert_attempts})")
                    self.collection.insert(rows)
                    self.collection.flush()
                    logger.info(f"✅ Inserted {len(rows)} rows")
                    
                    # Load collection after insert
                    try:
                        self.collection.load()
                        logger.debug("🔄 Collection loaded after insert")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to load after insert: {e}")
                    break
                except DataNotMatchException as dnme:
                    logger.exception(f"❌ DataNotMatch on insert: {dnme}")
                    if attempt >= insert_attempts - 1:
                        raise
                    await asyncio.sleep(2)
                except MilvusException as me:
                    logger.warning(f"⚠️ Insert attempt failed: {me}")
                    if attempt >= insert_attempts - 1:
                        raise
                    await asyncio.sleep(2)

            logger.info(f"✅ Successfully added {len(ids)} documents")
            return ids

        except Exception as e:
            logger.exception(f"❌ Error adding documents: {e}")
            return []

    # ========== Utility Methods ==========
    def _clean_and_normalize_content(self, content: str) -> str:
        """Clean and normalize content."""
        if not content:
            return ""
        content = re.sub(r'\s+', ' ', content)
        content = re.sub(r'[\r\n\t]+', ' ', content)
        html_entities = {
            '&nbsp;': ' ', '&amp;': '&', '&lt;': '<', '&gt;': '>',
            '&quot;': '"', '&#39;': "'", '&apos;': "'", '&mdash;': '—',
            '&ndash;': '–', '&hellip;': '...', '&copy;': '©'
        }
        for entity, char in html_entities.items():
            content = content.replace(entity, char)
        content = re.sub(r'([.!?])\1+', r'\1', content)
        content = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', content)
        return content.strip()

    def _extract_key_terms(self, content: str) -> List[str]:
        """Extract key terms from content."""
        if not content:
            return []
        words = re.findall(r'\b[a-zA-Z]{4,}\b', content.lower())
        word_freq = defaultdict(int)
        for word in words:
            if word not in self.stopwords and len(word) > 3:
                word_freq[word] += 1
        sorted_terms = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        filtered_terms = [term for term, freq in sorted_terms if 2 <= freq <= len(words) * 0.1]
        return filtered_terms[:20]

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            parsed = urllib.parse.urlparse(url)
            return parsed.netloc
        except Exception:
            return ""

    def _normalize_images(self, images: Any, page_url: str, content_snippet: str, max_images: int = 20) -> List[Dict[str, Any]]:
        """Normalize and filter images."""
        if not images or not isinstance(images, (list, tuple)):
            return []
        
        normalized = []
        seen_urls = set()
        
        for img in images[:max_images * 2]:
            if not isinstance(img, dict):
                continue
            
            url = self._safe_str(img.get("url"))
            if not url or url in seen_urls:
                continue
            
            url_lower = url.lower()
            noise_patterns = ['logo', 'icon', 'favicon', 'sprite', 'banner', 'avatar', 'badge', 'pixel', 'tracker', '1x1']
            if any(pattern in url_lower for pattern in noise_patterns):
                continue
            
            if page_url and not url.startswith(("http://", "https://", "data:")):
                try:
                    url = urllib.parse.urljoin(page_url, url)
                except Exception:
                    pass
            
            alt = self._safe_str(img.get("alt"))[:500]
            caption = self._safe_str(img.get("caption"))[:500]
            img_type = self._safe_str(img.get("type"))[:100]
            image_text = self._safe_str(img.get("text"))
            
            if not image_text:
                parts = []
                if alt:
                    parts.append(alt)
                if caption:
                    parts.append(caption)
                if content_snippet:
                    parts.append(content_snippet[:200])
                image_text = " | ".join(parts)
            
            normalized.append({
                "url": url[:1500],
                "alt": alt,
                "caption": caption,
                "type": img_type,
                "text": image_text[:1000]
            })
            seen_urls.add(url)
            
            if len(normalized) >= max_images:
                break
        
        return normalized

    def _decode_to_text(self, value: Any) -> str:
        """Decode bytes to text."""
        if value is None:
            return ""
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8")
            except UnicodeDecodeError:
                return value.decode("latin-1", errors="replace")
        return str(value)

    def _soft_truncate(self, text: str, limit: int = 8000) -> str:
        """Soft truncate text at sentence boundary."""
        if len(text) <= limit:
            return text
        truncated = text[:limit]
        last_period = truncated.rfind('. ')
        if last_period > limit * 0.8:
            return truncated[:last_period + 1] + " [truncated]"
        return truncated + "... [truncated]"

    def _coerce_images_from_meta(self, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract images from metadata."""
        s = meta.get("images_json")
        if isinstance(s, str) and s:
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
        
        images_field = meta.get("images")
        if isinstance(images_field, list):
            return images_field
        elif isinstance(images_field, str):
            try:
                parsed = json.loads(images_field)
                return parsed if isinstance(parsed, list) else []
            except Exception:
                pass
        return []

    def _safe_str(self, v: Any) -> str:
        """Safely convert to string."""
        if v is None:
            return ""
        return str(v).strip()

    # ========== Collection Management ==========
    async def delete_collection(self) -> None:
        """Delete the collection."""
        try:
            if utility.has_collection(self.collection_name, using=self.milvus_alias):
                try:
                    if self.collection:
                        try:
                            self.collection.release()
                            logger.info(f"Released collection: {self.collection_name}")
                        except Exception:
                            pass
                    utility.drop_collection(self.collection_name, using=self.milvus_alias)
                    logger.info(f"✅ Deleted collection: {self.collection_name}")
                except Exception as e:
                    logger.exception(f"❌ Error deleting collection: {e}")
                finally:
                    self.collection = None
        except Exception as e:
            logger.exception(f"❌ Error in delete_collection: {e}")

    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""

        if not self.collection and not self._connection_established:
            asyncio.create_task(self.initialize())
            logger.info("⏳ Milvus init scheduled (non-blocking)")
            return []

        if not self.collection:
            return {
                "document_count": 0,
                "collection_name": "unavailable",
                "status": "unavailable",
                "search_config": self.search_config,
                "ai_services": "grok+openrouter",
                "database": "milvus",
                "embedding_dimension": self.embedding_dim
            }
        
        try:
            try:
                self.collection.flush()
            except Exception:
                pass
            
            count = self.collection.num_entities
            indexes = getattr(self.collection, 'indexes', []) or []
            index_info = []
            
            for idx in indexes:
                try:
                    index_info.append({
                        "field": idx.field_name,
                        "type": idx.params.get("index_type", "unknown"),
                        "metric": idx.params.get("metric_type", "unknown")
                    })
                except Exception:
                    pass

            return {
                "document_count": int(count),
                "collection_name": self.collection_name,
                "status": "active",
                "search_config": self.search_config,
                "ai_services": "grok+openrouter",
                "database": "milvus",
                "connection": {
                    "host": self._env_host or "auto-detected",
                    "port": self._env_port,
                    "connected": self._connection_established
                },
                "indexes": index_info,
                "embedding_dimension": self.embedding_dim,
                "last_updated": datetime.now().isoformat()
            }
        except Exception as e:
            logger.exception(f"❌ Error getting stats: {e}")
            return {
                "document_count": 0,
                "collection_name": "error",
                "status": "error",
                "error": str(e),
                "embedding_dimension": self.embedding_dim
            }

    async def close(self) -> None:
        """Close Milvus connection."""
        try:
            if self.collection:
                try:
                    self.collection.release()
                    logger.info("Released Milvus collection")
                except Exception:
                    pass
            if self._connection_established:
                try:
                    connections.disconnect(self.milvus_alias)
                    self._connection_established = False
                    logger.info(f"✅ Disconnected from Milvus (alias={self.milvus_alias})")
                except Exception as e:
                    logger.warning(f"⚠️ Error disconnecting: {e}")
        except Exception as e:
            logger.exception(f"❌ Error closing Milvus connection: {e}")

    def __del__(self):
        """Cleanup on deletion."""
        try:
            if self.collection:
                try:
                    self.collection.release()
                except Exception:
                    pass
            if self._connection_established:
                try:
                    connections.disconnect(self.milvus_alias)
                except Exception:
                    pass
        except Exception:
            pass


# Global instance
milvus_service = MilvusService()
