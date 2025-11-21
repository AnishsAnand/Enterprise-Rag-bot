# milvus_service.py (production-ready, enhanced, fixed)
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
    Production-grade Milvus service with:
      - robust connection & load retrying
      - configurable index type (HNSW recommended for low-latency)
      - auto-load before search + retry-on-failure
      - embedding dimension validation / padding/truncation
      - immediate load after inserts
    """

    def __init__(self):
        self.collection: Optional[Collection] = None
        self.collection_name: str = getattr(settings, "MILVUS_COLLECTION", "enterprise_rag")

        # Connection params (prefer env vars / .env)
        self.milvus_uri: Optional[str] = os.getenv("MILVUS_URI")
        self._env_host = os.getenv("MILVUS_HOST", "")  # allow overriding
        self._env_port = os.getenv("MILVUS_PORT", "19530")
        self.milvus_user: str = os.getenv("MILVUS_USER", "")
        self.milvus_password: str = os.getenv("MILVUS_PASSWORD", "")
        self.milvus_alias: str = os.getenv("MILVUS_ALIAS", "default")

        self.debug_dump: bool = os.getenv("MILVUS_DEBUG_DUMP", "false").lower() == "true"

        # Index & search tuning (production)
        # Prefer HNSW (lower latency) for large embeddings; IVF_FLAT can be used for very large corpora with SSD-backed storage.
        self.index_type = os.getenv("MILVUS_INDEX_TYPE", "HNSW").upper()  # HNSW or IVF_FLAT
        # Default index params - can be overridden by environment var in JSON form
        self.default_index_params = {
            "HNSW": {"index_type": "HNSW", "params": {"M": int(os.getenv("MILVUS_HNSW_M", "16")),
                                                      "efConstruction": int(os.getenv("MILVUS_HNSW_EFC", "200"))}},
            "IVF_FLAT": {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": int(os.getenv("MILVUS_IVF_NLIST", "2048"))}},
        }
        # If user provided JSON for index params, honor it
        env_index_json = os.getenv("MILVUS_INDEX_PARAMS", "")
        if env_index_json:
            try:
                parsed = json.loads(env_index_json)
                if isinstance(parsed, dict):
                    self.default_index_params[self.index_type] = parsed
            except Exception:
                logger.warning("MILVUS_INDEX_PARAMS invalid JSON; ignoring")

        # Search-time tuning
        # For HNSW use 'ef' during search; for IVF use 'nprobe'
        self.search_ef = int(os.getenv("MILVUS_SEARCH_EF", "64"))
        self.search_nprobe = int(os.getenv("MILVUS_NPROBE", "32"))

        # Enhanced search configuration
        self.search_config = {
            "min_relevance_threshold": float(os.getenv("MILVUS_MIN_RELEVANCE", "0.12")),
            "max_initial_results": int(os.getenv("MILVUS_MAX_INITIAL_RESULTS", "120")),
            "rerank_top_k": int(os.getenv("MILVUS_RERANK_TOP_K", "60")),
            "enable_query_expansion": os.getenv("MILVUS_ENABLE_QUERY_EXPANSION", "true").lower() == "true",
            "enable_semantic_rerank": os.getenv("MILVUS_ENABLE_SEMANTIC_RERANK", "true").lower() == "true",
            "enable_context_enrichment": os.getenv("MILVUS_ENABLE_CONTEXT_ENRICHMENT", "true").lower() == "true",
            "diversity_factor": float(os.getenv("MILVUS_DIVERSITY_FACTOR", "0.3")),
        }

        # Embedding dimension must match AI service and Milvus schema
        self.embedding_dim: int = int(os.getenv("EMBEDDING_DIMENSION", "4096"))
        logger.info(f"🔧 MilvusService init: EMBEDDING_DIMENSION={self.embedding_dim}, INDEX_TYPE={self.index_type}")

        self._connection_established: bool = False

        # Enhanced stopwords for better key term extraction
        self.stopwords: Set[str] = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one',
            'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see',
            'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use', 'that',
            'with', 'have', 'this', 'will', 'your', 'from', 'they', 'know', 'want', 'been', 'good',
            'much', 'some', 'time', 'very', 'when', 'come', 'here', 'just', 'like', 'long', 'make',
            'many', 'over', 'such', 'take', 'than', 'them', 'well', 'were', 'what', 'about', 'into'
        }

    # ---------- Connection Helpers ----------
    def _try_socket(self, host: str, port: int, timeout: float = 1.0) -> bool:
        """Test socket connectivity (quick probe)."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            sock.connect((host, port))
            sock.close()
            return True
        except Exception:
            return False

    def _choose_host_candidate(self) -> Tuple[Optional[str], Optional[int]]:
        """Select host candidate intelligently (env, docker service name, loopback)."""
        if self.milvus_uri:
            return None, None

        candidates = []
        if self._env_host:
            candidates.append(self._env_host)
        # prefer docker service name if rag-app is containerized
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
                else:
                    logger.debug(f"Socket probe failed: {host}:{port_int}")
            except Exception:
                continue

        # fallback to env host or loopback
        if self._env_host:
            return self._env_host, port_int
        return "127.0.0.1", port_int

    # ---------- Lifecycle ----------
    async def initialize(self) -> None:
        """
        Connect to Milvus, ensure collection exists, create index if missing,
        and load collection into memory with retries.
        """
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

                logger.info(f"Attempting Milvus connect ({attempt}/{max_retries}) -> {connect_kwargs}")
                connections.connect(**connect_kwargs)
                self._connection_established = True
                logger.info(f"✅ Connected to Milvus (alias={self.milvus_alias}) via {connect_kwargs.get('host', 'uri')}")

                # Ensure collection exists (or create)
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
                            logger.info(f"🔍 Detected collection embedding dim={detected_dim}, expected={self.embedding_dim}")
                            if detected_dim != self.embedding_dim:
                                # If there are no docs we can rebuild automatically; otherwise require manual intervention
                                count = 0
                                try:
                                    count = self.collection.num_entities
                                except Exception:
                                    pass
                                if count == 0:
                                    logger.warning("Dimension mismatch but collection empty — rebuilding with correct dim.")
                                    utility.drop_collection(self.collection_name, using=self.milvus_alias)
                                    await self._create_collection()
                                    self.collection = Collection(self.collection_name, using=self.milvus_alias)
                                else:
                                    raise ValueError(
                                        f"Embedding dimension mismatch: collection={detected_dim}, env={self.embedding_dim} (count={count}). Manual migration required."
                                    )
                    except ValueError:
                        raise
                    except Exception as e:
                        logger.warning(f"Schema inspection warning: {e} — continuing")
                else:
                    logger.info(f"📦 Creating collection: {self.collection_name} (embedding_dim={self.embedding_dim})")
                    await self._create_collection()
                    self.collection = Collection(self.collection_name, using=self.milvus_alias)

                # Ensure index exists and is suitable
                try:
                    # If the collection has no indexes, create default tuned index
                    existing_indexes = getattr(self.collection, "indexes", []) or []
                    if not existing_indexes:
                        idx_conf = self.default_index_params.get(self.index_type, self.default_index_params["HNSW"])
                        # Always include metric_type for vector indexes
                        if self.index_type == "IVF_FLAT":
                            index_params = {
                                "metric_type": idx_conf.get("metric_type", "L2"),
                                "index_type": idx_conf.get("index_type", "IVF_FLAT"),
                                "params": idx_conf.get("params", {"nlist": 2048})
                            }
                        else:
                            # HNSW also requires metric_type
                            index_params = {
                                "metric_type": idx_conf.get("metric_type", "L2"),
                                "index_type": idx_conf.get("index_type", "HNSW"),
                                "params": idx_conf.get("params", {"M": 16, "efConstruction": 200})
                            }
                        try:
                            self.collection.create_index(field_name="embedding", index_params=index_params)
                            logger.info(f"✅ Created index for collection '{self.collection_name}': {index_params}")
                        except Exception as e:
                            logger.warning(f"⚠️ Index creation warning: {e}")
                    else:
                        logger.debug("Index(es) already present on collection")
                except Exception as e:
                    logger.warning(f"Index check/create failed: {e}")

                # Force-load collection into memory with retries (milvus can be slow during startup)
                for load_attempt in range(6):
                    try:
                        self.collection.load()
                        logger.info(f"✅ Collection loaded into memory (attempt {load_attempt + 1})")
                        break
                    except MilvusException as e:
                        logger.warning(f"⚠️ Collection load attempt {load_attempt+1}/6 failed: {e}")
                        await asyncio.sleep(2 ** load_attempt)
                    except Exception as e:
                        logger.warning(f"⚠️ Unexpected collection load error (attempt {load_attempt+1}/6): {e}")
                        await asyncio.sleep(2 ** load_attempt)
                else:
                    logger.error(f"❌ Failed to load collection '{self.collection_name}' after retries")

                # success -> break attempts loop
                break

            except Exception as e:
                logger.exception(f"❌ Milvus initialization attempt {attempt}/{max_retries} failed: {e}")
                self._connection_established = False
                self.collection = None
                if attempt >= max_retries:
                    logger.error("❌ Milvus initialization aborted after max retries")
                    return
                await asyncio.sleep(backoff * attempt)

    async def _create_collection(self) -> None:
        """Create collection with robust schema and index selection."""
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
            description=f"Enterprise RAG Knowledge Base - Embedding Dimension: {self.embedding_dim}",
            auto_id=False
        )

        # Drop if exists (safe-guard)
        try:
            if utility.has_collection(self.collection_name, using=self.milvus_alias):
                logger.warning(f"⚠️ Dropping existing collection: {self.collection_name}")
                utility.drop_collection(self.collection_name, using=self.milvus_alias)
        except Exception as e:
            logger.debug(f"No existing collection to drop: {e}")

        # Create collection
        self.collection = Collection(name=self.collection_name, schema=schema, using=self.milvus_alias)
        logger.info(f"✅ Collection '{self.collection_name}' created")

        # Create index according to selection (HNSW default for production low-latency)
        try:
            idx_conf = self.default_index_params.get(self.index_type, self.default_index_params["HNSW"])
            # Always include metric_type for all vector indexes
            if self.index_type == "IVF_FLAT":
                index_params = {
                    "metric_type": idx_conf.get("metric_type", "L2"),
                    "index_type": idx_conf.get("index_type", "IVF_FLAT"),
                    "params": idx_conf.get("params", {"nlist": 2048})
                }
            else:
                # HNSW also requires metric_type
                index_params = {
                    "metric_type": idx_conf.get("metric_type", "L2"),
                    "index_type": idx_conf.get("index_type", "HNSW"),
                    "params": idx_conf.get("params", {"M": 16, "efConstruction": 200})
                }
            self.collection.create_index(field_name="embedding", index_params=index_params)
            logger.info(f"✅ Created index: {index_params}")
        except Exception as e:
            logger.warning(f"⚠️ Index creation warning: {e}")

    # ---------- Enhanced Query Processing ----------
    def _preprocess_query(self, query: str) -> Tuple[str, List[str]]:
        if not query:
            return "", []
        original_query = query.strip()
        cleaned_query = re.sub(r'\s+', ' ', original_query.lower())
        key_terms: List[str] = []

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

        words = re.findall(r'\b\w+\b', cleaned_query)
        key_terms.extend([word for word in words if len(word) > 3 and word not in self.stopwords])

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
        if not document:
            return 0.0

        doc_lower = document.lower()
        query_lower = query.lower()

        semantic_score = max(0.0, 1.0 / (1.0 + distance))

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

        quality_score = 0.0
        doc_length = len(document)
        word_count = len(document.split())
        if 200 <= doc_length <= 5000:
            quality_score += 0.2
        elif 100 <= doc_length < 200:
            quality_score += 0.1
        elif 5000 < doc_length <= 8000:
            quality_score += 0.15
        unique_words = len(set(document.lower().split()))
        if word_count > 0:
            diversity = unique_words / word_count
            if 0.4 <= diversity <= 0.8:
                quality_score += 0.1

        title_score = 0.0
        if metadata.get("title"):
            title_lower = str(metadata["title"]).lower()
            title_matches = sum(1 for term in key_terms if term.lower() in title_lower)
            if title_matches > 0:
                title_score = min(0.25, title_matches * 0.15)
            if query_lower in title_lower:
                title_score += 0.15

        url_score = 0.0
        if metadata.get("url"):
            url_lower = str(metadata["url"]).lower()
            url_matches = sum(1 for term in key_terms if term.lower() in url_lower)
            if url_matches > 0:
                url_score = min(0.15, url_matches * 0.08)

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

        format_score = 0.0
        doc_format = metadata.get("format", "").lower()
        if doc_format in ["pdf", "docx", "html"]:
            format_score = 0.05

        final_score = (
            semantic_score * 0.30 +
            term_score * 0.22 +
            phrase_score * 0.18 +
            quality_score * 0.08 +
            title_score * 0.10 +
            url_score * 0.05 +
            recency_score * 0.05 +
            format_score * 0.02
        )
        return min(1.0, final_score)

    async def _expand_query(self, original_query: str, key_terms: List[str]) -> str:
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
                timeout=15
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

    def _apply_diversity_filtering(self, results: List[Dict[str, Any]], diversity_factor: float = 0.3) -> List[Dict[str, Any]]:
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

    # ---------- Add Documents ----------
    def _flatten_to_str(self, v: Any) -> str:
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
        for idx, emb in enumerate(embeddings):
            if emb is None:
                logger.warning(f"⚠️ Embedding {idx} is None, creating zero vector")
                clean_embeds.append([0.0] * self.embedding_dim)
                continue
            if isinstance(emb, (list, tuple)):
                try:
                    float_list = [float(x) for x in emb]
                    if len(float_list) != self.embedding_dim:
                        logger.error(f"❌ Embedding {idx} dimension mismatch: got {len(float_list)}, expected {self.embedding_dim}")
                        if len(float_list) < self.embedding_dim:
                            float_list.extend([0.0] * (self.embedding_dim - len(float_list)))
                            logger.warning(f"⚠️ Padded embedding {idx} with zeros")
                        else:
                            float_list = float_list[:self.embedding_dim]
                            logger.warning(f"⚠️ Truncated embedding {idx} to {self.embedding_dim}")
                    clean_embeds.append(float_list)
                except Exception as e:
                    logger.error(f"❌ Failed to convert embedding {idx}: {e}")
                    clean_embeds.append([0.0] * self.embedding_dim)
            else:
                try:
                    single_val = float(emb)
                    logger.warning(f"⚠️ Embedding {idx} is scalar, creating vector")
                    clean_embeds.append([single_val] + [0.0] * (self.embedding_dim - 1))
                except Exception:
                    clean_embeds.append([0.0] * self.embedding_dim)
        return clean_embeds

    def _validate_entities_data(self, entities: Dict[str, List[Any]]) -> Tuple[bool, str]:
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
        if not self.collection:
            await self.initialize()
        if not self.collection:
            logger.warning("⚠️ Milvus not available, skipping document addition")
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
                            logger.error(
                                f"❌ Embedding dimension mismatch: AI returned {first_emb_dim}, expected {self.embedding_dim}"
                            )
                            if attempt < max_retries - 1:
                                logger.warning("⚠️ Retrying embedding generation...")
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
                logger.error("❌ Failed to generate embeddings after retries")
                return []

            embeddings = self._ensure_embedding_list(embeddings)
            entities_data["embedding"] = embeddings
            entities_data["id"] = [self._flatten_to_str(i) for i in entities_data["id"]]

            valid, msg = self._validate_entities_data(entities_data)
            if not valid:
                logger.error(f"❌ Entities validation failed: {msg}")
                return []

            # Build rows and insert
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
                            logger.error(f"❌ Row {i} embedding wrong dim: {len(row['embedding'])}")
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

            insert_attempts = 3
            for attempt in range(insert_attempts):
                try:
                    logger.info(f"📤 Inserting {len(rows)} rows (attempt {attempt+1}/{insert_attempts})")
                    self.collection.insert(rows)
                    self.collection.flush()
                    logger.info(f"✅ Inserted {len(rows)} rows")
                    # Ensure inserted data is loaded for immediate search
                    try:
                        self.collection.load()
                        logger.debug("🔄 Collection loaded after insert")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to load collection after insert: {e}")
                    break
                except DataNotMatchException as dnme:
                    logger.exception(f"❌ DataNotMatch on insert: {dnme}")
                    if attempt >= insert_attempts - 1:
                        raise
                    await asyncio.sleep(2)
                except MilvusException as me:
                    logger.warning(f"⚠️ Milvus insert attempt failed: {me}")
                    if attempt >= insert_attempts - 1:
                        raise
                    await asyncio.sleep(2)

            logger.info(f"✅ Successfully added {len(ids)} documents to Milvus")
            return ids

        except Exception as e:
            logger.exception(f"❌ Error adding documents: {e}")
            return []

    # ---------- Search Documents ----------
    async def search_documents(self, query: str, n_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """Perform search with guaranteed collection load and retry on collection-not-loaded errors."""
        if not query:
            return []

        if not self.collection:
            await self.initialize()
        if not self.collection:
            logger.warning("⚠️ Milvus not available, returning empty results")
            return []

        try:
            cleaned_query, key_terms = self._preprocess_query(query)
            if not cleaned_query:
                return []

            logger.info(f"🔍 Search: '{query}' -> key_terms: {key_terms[:5]}")

            expanded_query = cleaned_query
            if self.search_config["enable_query_expansion"]:
                expanded_query = await self._expand_query(cleaned_query, key_terms)

            default_k = int(getattr(settings, "MILVUS_QUERY_TOP_K", 50))
            initial_k = min(self.search_config["max_initial_results"], int(n_results or default_k) * 3)
            final_k = int(n_results or default_k)

            # Generate query embedding
            query_embeddings = await ai_service.generate_embeddings([expanded_query])
            if not query_embeddings:
                logger.warning("⚠️ Failed to generate query embeddings")
                return []

            query_emb_dim = len(query_embeddings[0]) if isinstance(query_embeddings[0], (list, tuple)) else 0
            if query_emb_dim != self.embedding_dim:
                logger.error(f"❌ Query embedding dim mismatch: got {query_emb_dim}, expected {self.embedding_dim}")
                return []

            # Build search params depending on index type
            if self.index_type == "HNSW":
                search_params = {"metric_type": "L2", "params": {"ef": self.search_ef}}
            else:
                search_params = {"metric_type": "L2", "params": {"nprobe": self.search_nprobe}}

            # Attempt search with auto-load+retry on collection-not-loaded errors
            search_attempts = 3
            backoff = float(os.getenv("MILVUS_CONNECT_BACKOFF", "2"))
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
                    # If collection not loaded, try to load and retry
                    if "collection not loaded" in msg or getattr(me, "code", None) == 101:
                        logger.warning(f"⚠️ Milvus search failed (collection not loaded). Attempting to load collection (attempt {attempt}/{search_attempts})")
                        try:
                            self.collection.load()
                            logger.info("✅ Collection load triggered during search retry")
                        except Exception as le:
                            logger.warning(f"⚠️ Failed to load collection during search retry: {le}")
                        await asyncio.sleep(backoff * attempt)
                        continue
                    else:
                        logger.exception(f"❌ Milvus search error (non-retryable): {me}")
                        break
                except Exception as e:
                    last_exception = e
                    logger.exception(f"❌ Unexpected search error: {e}")
                    await asyncio.sleep(backoff * attempt)
                    continue

            if last_exception is not None and last_exception:
                logger.error(f"❌ Search aborted due to repeated errors: {last_exception}")
                return []

            if not results or len(results) == 0 or len(results[0]) == 0:
                logger.info("ℹ️ No documents found in Milvus")
                return []

            # Score and filter results
            scored_results = []
            for hit in results[0]:
                entity = hit.entity
                # Access fields safely (RowEntity.fields is a dict-like object)
                fields = getattr(entity, "fields", {}) or {}

                metadata = {
                    "url": fields.get("url", ""),
                    "title": fields.get("title", ""),
                    "format": fields.get("format", ""),
                    "timestamp": fields.get("timestamp", ""),
                    "source": fields.get("source", ""),
                    "content_length": fields.get("content_length", 0),
                    "word_count": fields.get("word_count", 0),
                    "image_count": fields.get("image_count", 0),
                    "has_images": fields.get("has_images", False),
                    "domain": fields.get("domain", ""),
                    "content_hash": fields.get("content_hash", 0),
                    "images_json": fields.get("images_json", ""),
                    "key_terms": fields.get("key_terms", "")
                }

                # Content handling
                content = fields.get("content", "")
                distance = getattr(hit, "distance", 0.0)

                relevance_score = self._calculate_enhanced_relevance(
                    document=content,
                    metadata=metadata,
                    query=query,
                    key_terms=key_terms,
                    distance=distance
                )

                if relevance_score >= self.search_config["min_relevance_threshold"]:
                    metadata["images"] = self._coerce_images_from_meta(metadata)
                    scored_results.append({
                        "content": content,
                        "metadata": metadata,
                        "distance": float(distance),
                        "relevance_score": relevance_score,
                        "semantic_similarity": max(0.0, 1.0 / (1.0 + float(distance)))
                    })

            scored_results.sort(key=lambda x: x["relevance_score"], reverse=True)
            logger.info(f"📊 Initial scoring: {len(scored_results)} results above threshold")


            # Diversity and reranking
            if self.search_config["enable_context_enrichment"] and len(scored_results) > final_k:
                scored_results = self._apply_diversity_filtering(scored_results, self.search_config["diversity_factor"])

            if self.search_config["enable_semantic_rerank"] and len(scored_results) > final_k:
                rerank_candidates = min(len(scored_results), self.search_config["rerank_top_k"])
                scored_results = await self._semantic_rerank(scored_results[:rerank_candidates], query, rerank_candidates)

            final_results = scored_results[:final_k]
            logger.info(f"✅ Returning {len(final_results)} relevant documents")
            return final_results

        except Exception as e:
            logger.exception(f"❌ Milvus search error: {e}")
            return []

    async def _semantic_rerank(self, results: List[Dict], query: str, top_k: int) -> List[Dict]:
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
                dot_product = sum(x * y for x, y in zip(a, b))
                norm_a = math.sqrt(sum(x * x for x in a))
                norm_b = math.sqrt(sum(x * x for x in b))
                if norm_a == 0 or norm_b == 0:
                    return 0.0
                return dot_product / (norm_a * norm_b)

            for i, result in enumerate(results[:len(content_embeddings)]):
                semantic_sim = cosine_similarity(query_embedding, content_embeddings[i])
                original_score = result["relevance_score"]
                result["semantic_rerank_score"] = original_score * 0.65 + semantic_sim * 0.35
                result["semantic_similarity"] = semantic_sim

            results.sort(key=lambda x: x.get("semantic_rerank_score", x["relevance_score"]), reverse=True)
            logger.info("✅ Applied semantic reranking")
        except Exception as e:
            logger.warning(f"⚠️ Semantic reranking failed: {e}")
        return results[:top_k]

    # ---------- Utility Methods ----------
    def _clean_and_normalize_content(self, content: str) -> str:
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
        try:
            parsed = urllib.parse.urlparse(url)
            return parsed.netloc
        except Exception:
            return ""

    def _normalize_images(self, images: Any, page_url: str, content_snippet: str, max_images: int = 20) -> List[Dict[str, Any]]:
        if not images or not isinstance(images, (list, tuple)):
            return []
        normalized = []
        seen_urls = set()
        for img in images[:max_images * 2]:
            if not isinstance(img, dict):
                continue
            url = self._safe_str(img.get("url"))
            if not url:
                continue
            if url in seen_urls:
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
            normalized.append({"url": url[:1500], "alt": alt, "caption": caption, "type": img_type, "text": image_text[:1000]})
            seen_urls.add(url)
            if len(normalized) >= max_images:
                break
        return normalized

    def _decode_to_text(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8")
            except UnicodeDecodeError:
                return value.decode("latin-1", errors="replace")
        return str(value)

    def _soft_truncate(self, text: str, limit: int = 8000) -> str:
        if len(text) <= limit:
            return text
        truncated = text[:limit]
        last_period = truncated.rfind('. ')
        if last_period > limit * 0.8:
            return truncated[:last_period + 1] + " [truncated]"
        return truncated + "... [truncated]"

    def _coerce_images_from_meta(self, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
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
        if v is None:
            return ""
        return str(v).strip()

    # ---------- Collection Management ----------
    async def delete_collection(self) -> None:
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
        if not self.collection:
            await self.initialize()
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
