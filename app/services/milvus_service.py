# app/services/milvus_service.py - Enhanced Production Version
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
    Production-grade Milvus service with enhanced search, context enrichment,
    and intelligent relevance scoring
    """

    def __init__(self):
        self.collection: Optional[Collection] = None
        self.collection_name: str = getattr(settings, "MILVUS_COLLECTION", "enterprise_rag")

        # Connection params
        self.milvus_uri: Optional[str] = os.getenv("MILVUS_URI")
        self._env_host = os.getenv("MILVUS_HOST", "")
        self._env_port = os.getenv("MILVUS_PORT", "19530")
        self.milvus_user: str = os.getenv("MILVUS_USER", "")
        self.milvus_password: str = os.getenv("MILVUS_PASSWORD", "")
        self.milvus_alias: str = os.getenv("MILVUS_ALIAS", "default")

        self.debug_dump: bool = os.getenv("MILVUS_DEBUG_DUMP", "false").lower() == "true"

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

        self.embedding_dim: int = int(os.getenv("EMBEDDING_DIMENSION", "1536"))
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
        """Test socket connectivity"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            sock.connect((host, port))
            sock.close()
            return True
        except Exception:
            return False

    def _choose_host_candidate(self) -> Tuple[Optional[str], Optional[int]]:
        """Intelligently select best Milvus host"""
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
                if self._try_socket(host, port_int, timeout=1.0):
                    logger.debug(f"Socket probe succeeded: {host}:{port_int}")
                    return host, port_int
                else:
                    logger.debug(f"Socket probe failed: {host}:{port_int}")
            except Exception:
                continue

        # Fallback
        if self._env_host:
            return self._env_host, port_int
        return "127.0.0.1", port_int

    # ---------- Lifecycle ----------
    
    async def initialize(self) -> None:
        """Initialize Milvus with robust retry logic"""
        if self.collection is not None:
            return

        max_retries = int(os.getenv("MILVUS_CONNECT_RETRIES", "3"))
        backoff = float(os.getenv("MILVUS_CONNECT_BACKOFF", "2"))

        for attempt in range(1, max_retries + 1):
            try:
                connect_kwargs: Dict[str, Any] = {"alias": self.milvus_alias}
                
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

                sanitized = {
                    k: (v if k != "password" else "***") 
                    for k, v in connect_kwargs.items()
                }
                logger.info(f"Milvus connection attempt {attempt}/{max_retries}: {sanitized}")

                connections.connect(**connect_kwargs)
                self._connection_established = True
                
                host_log = connect_kwargs.get("host") or self.milvus_uri or "uri"
                logger.info(f"✅ Connected to Milvus (alias={self.milvus_alias}) via {host_log}")

                # Collection handling
                if utility.has_collection(self.collection_name, using=self.milvus_alias):
                    self.collection = Collection(self.collection_name, using=self.milvus_alias)
                    logger.info(f"✅ Loaded existing collection: {self.collection_name}")
                    
                    # Detect embedding dimension
                    try:
                        schema = getattr(self.collection, "schema", None)
                        if schema:
                            for f in getattr(schema, "fields", []):
                                if getattr(f, "name", "") == "embedding":
                                    params = getattr(f, "params", None) or getattr(f, "type_params", None)
                                    if isinstance(params, dict):
                                        detected_dim = params.get("dim") or params.get("dimension")
                                    else:
                                        detected_dim = getattr(f, "dim", None)
                                    
                                    if detected_dim:
                                        try:
                                            self.embedding_dim = int(detected_dim)
                                            logger.info(f"Detected embedding_dim: {self.embedding_dim}")
                                        except Exception:
                                            logger.debug(f"Could not parse dimension: {detected_dim}")
                                    break
                    except Exception as e:
                        logger.debug(f"Schema inspection failed: {e}")
                else:
                    await self._create_collection()
                    logger.info(f"✅ Created new collection: {self.collection_name}")

                # Load collection into memory
                try:
                    if self.collection:
                        self.collection.load()
                        logger.info(f"✅ Collection loaded into memory: {self.collection_name}")
                except Exception as e:
                    logger.warning(f"Could not load collection: {e}")

                break  # Success

            except Exception as e:
                logger.exception(f"❌ Milvus initialization attempt {attempt}/{max_retries} failed: {e}")
                self._connection_established = False
                self.collection = None
                
                if attempt >= max_retries:
                    logger.error(f"Milvus initialization failed after {max_retries} attempts")
                    return
                
                await asyncio.sleep(backoff * attempt)

    async def _create_collection(self) -> None:
        """Create collection with optimized schema and index"""
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
            description="Enterprise RAG Knowledge Base with Enhanced Search",
            auto_id=False
        )

        # Drop if exists
        try:
            if utility.has_collection(self.collection_name, using=self.milvus_alias):
                utility.drop_collection(self.collection_name, using=self.milvus_alias)
        except Exception:
            pass

        self.collection = Collection(
            name=self.collection_name,
            schema=schema,
            using=self.milvus_alias
        )

        # Create optimized index
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 2048}  # Increased for better accuracy
        }
        
        try:
            self.collection.create_index(field_name="embedding", index_params=index_params)
            logger.info("✅ Created optimized vector index (IVF_FLAT, nlist=2048)")
        except Exception as e:
            logger.warning(f"Index creation failed: {e}")

    # ---------- Enhanced Query Processing ----------
    
    def _preprocess_query(self, query: str) -> Tuple[str, List[str]]:
        """Enhanced query preprocessing with better term extraction"""
        if not query:
            return "", []
        
        original_query = query.strip()
        cleaned_query = re.sub(r'\s+', ' ', original_query.lower())
        key_terms: List[str] = []

        # Technical patterns (improved)
        technical_patterns = [
            r'\b[A-Z]{2,}\b',  # Acronyms
            r'\b\w+(?:ing|ion|tion|sion|ness|ment|able|ible|ance|ence)\b',  # Technical suffixes
            r'\b\w+[-_]\w+(?:[-_]\w+)*\b',  # Hyphenated/underscored terms
            r'\b\d+(?:\.\d+)*\w*\b',  # Numbers with optional units
            r'\b(?:how|what|where|when|why|which)\s+\w+\b',  # Question patterns
            r'\b[a-z]+\.[a-z]+\b',  # Domain-like patterns
        ]

        for pattern in technical_patterns:
            matches = re.findall(pattern, original_query)
            key_terms.extend(matches)

        # Extract meaningful words
        words = re.findall(r'\b\w+\b', cleaned_query)
        key_terms.extend([
            word for word in words 
            if len(word) > 3 and word not in self.stopwords
        ])

        # Remove duplicates while preserving order
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
        """Enhanced multi-factor relevance scoring"""
        if not document:
            return 0.0

        doc_lower = document.lower()
        query_lower = query.lower()
        
        # 1. Semantic similarity (from vector distance)
        semantic_score = max(0.0, 1.0 / (1.0 + distance))

        # 2. Term matching with position weighting
        term_score = 0.0
        if key_terms:
            term_matches = 0
            early_matches = 0
            doc_len = len(doc_lower)
            
            for term in key_terms:
                term_lower = term.lower()
                if term_lower in doc_lower:
                    term_matches += 1
                    # Bonus for early occurrence
                    pos = doc_lower.find(term_lower)
                    if pos < doc_len * 0.3:  # First 30% of document
                        early_matches += 1
            
            term_score = min(1.0, (term_matches / len(key_terms)) + (early_matches * 0.1))

        # 3. Phrase and bigram matching
        phrase_score = 0.0
        query_words = query_lower.split()
        
        if len(query_words) > 1:
            # Exact phrase match
            if query_lower in doc_lower:
                phrase_score = 0.5
            else:
                # Bigram matching
                bigram_matches = 0
                for i in range(len(query_words) - 1):
                    bigram = f"{query_words[i]} {query_words[i+1]}"
                    if bigram in doc_lower:
                        bigram_matches += 1
                
                if bigram_matches > 0:
                    phrase_score = min(0.4, bigram_matches * 0.2)

        # 4. Document quality metrics
        quality_score = 0.0
        doc_length = len(document)
        word_count = len(document.split())
        
        # Optimal length range
        if 200 <= doc_length <= 5000:
            quality_score += 0.2
        elif 100 <= doc_length < 200:
            quality_score += 0.1
        elif 5000 < doc_length <= 8000:
            quality_score += 0.15
        
        # Word diversity
        unique_words = len(set(document.lower().split()))
        if word_count > 0:
            diversity = unique_words / word_count
            if 0.4 <= diversity <= 0.8:  # Good diversity
                quality_score += 0.1

        # 5. Title relevance
        title_score = 0.0
        if metadata.get("title"):
            title_lower = str(metadata["title"]).lower()
            title_matches = sum(1 for term in key_terms if term.lower() in title_lower)
            if title_matches > 0:
                title_score = min(0.25, title_matches * 0.15)
            
            # Exact query in title
            if query_lower in title_lower:
                title_score += 0.15

        # 6. URL relevance
        url_score = 0.0
        if metadata.get("url"):
            url_lower = str(metadata["url"]).lower()
            url_matches = sum(1 for term in key_terms if term.lower() in url_lower)
            if url_matches > 0:
                url_score = min(0.15, url_matches * 0.08)

        # 7. Recency bonus
        recency_score = 0.0
        if metadata.get("timestamp"):
            try:
                doc_date = datetime.fromisoformat(
                    str(metadata["timestamp"]).replace("Z", "+00:00")
                )
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

        # 8. Content type bonus
        format_score = 0.0
        doc_format = metadata.get("format", "").lower()
        if doc_format in ["pdf", "docx", "html"]:
            format_score = 0.05

        # Weighted combination
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
        """Expand query with synonyms and related terms"""
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
                additional_terms = [
                    t.strip() 
                    for t in expanded_terms.strip().split(',')
                    if t.strip()
                ]
                
                original_words = set(original_query.lower().split())
                new_terms = [
                    t for t in additional_terms
                    if t.lower() not in original_words and len(t) > 2
                ]
                
                if new_terms:
                    expanded_query = f"{original_query} {' '.join(new_terms[:4])}"
                    logger.info(f"Query expanded: '{original_query}' + {new_terms[:4]}")
                    return expanded_query
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
        
        return original_query

    def _apply_diversity_filtering(self, results: List[Dict[str, Any]], 
                                   diversity_factor: float = 0.3) -> List[Dict[str, Any]]:
        """Apply diversity filtering to avoid redundant results"""
        if len(results) <= 3:
            return results
        
        diverse_results = [results[0]]  # Always keep top result
        
        for candidate in results[1:]:
            candidate_content = candidate.get("content", "").lower()
            candidate_terms = set(re.findall(r'\b\w+\b', candidate_content))
            
            # Check similarity with already selected results
            is_diverse = True
            for selected in diverse_results:
                selected_content = selected.get("content", "").lower()
                selected_terms = set(re.findall(r'\b\w+\b', selected_content))
                
                # Calculate Jaccard similarity
                if selected_terms and candidate_terms:
                    overlap = len(candidate_terms & selected_terms)
                    union = len(candidate_terms | selected_terms)
                    similarity = overlap / union if union > 0 else 0
                    
                    # If too similar, skip this candidate
                    if similarity > (1 - diversity_factor):
                        is_diverse = False
                        break
            
            if is_diverse:
                diverse_results.append(candidate)
        
        logger.info(f"Diversity filtering: {len(results)} -> {len(diverse_results)} results")
        return diverse_results

    # ---------- Add Documents ----------
    
    def _flatten_to_str(self, v: Any) -> str:
        """Convert value to string safely"""
        if v is None:
            return ""
        if isinstance(v, (list, tuple)):
            try:
                return "-".join([str(x) for x in v])
            except Exception:
                return str(v)
        return str(v)

    def _ensure_embedding_list(self, embeddings: List[Any]) -> List[List[float]]:
        """Ensure embeddings are proper float lists"""
        clean_embeds: List[List[float]] = []
        for emb in embeddings:
            if emb is None:
                clean_embeds.append([])
                continue
            
            if isinstance(emb, (list, tuple)):
                try:
                    clean_embeds.append([float(x) for x in emb])
                except Exception:
                    converted = []
                    for x in emb:
                        try:
                            converted.append(float(x))
                        except Exception:
                            converted.append(0.0)
                    clean_embeds.append(converted)
            else:
                try:
                    clean_embeds.append([float(emb)])
                except Exception:
                    clean_embeds.append([0.0])
        
        return clean_embeds

    def _validate_entities_data(self, entities: Dict[str, List[Any]]) -> Tuple[bool, str]:
        """Validate entity data structure"""
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
        """Add documents with enhanced processing"""
        if not self.collection:
            await self.initialize()
        
        if not self.collection:
            logger.warning("Milvus not available, skipping document addition")
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
                
                # Process content
                content = self._decode_to_text(doc.get("content", ""))
                content = self._clean_and_normalize_content(content)
                content = self._soft_truncate(content, limit=8000)  # Increased limit
                texts.append(content)
                
                # Process images
                normalized_images = self._normalize_images(
                    images=doc.get("images", []),
                    page_url=doc.get("url", ""),
                    content_snippet=content[:400],  # More context
                    max_images=20  # Increased
                )

                # Build entity data
                entities_data["id"].append(doc_id)
                entities_data["content"].append(content[:65535])
                entities_data["url"].append(str(doc.get("url", ""))[:2000])
                entities_data["title"].append(str(doc.get("title", ""))[:500])
                entities_data["format"].append(str(doc.get("format", "text"))[:100])
                entities_data["timestamp"].append(
                    doc.get("timestamp", "") or datetime.now().isoformat()
                )
                entities_data["source"].append(str(doc.get("source", "web_scraping"))[:100])
                entities_data["content_length"].append(len(content))
                entities_data["word_count"].append(len(content.split()))
                entities_data["image_count"].append(len(normalized_images))
                entities_data["has_images"].append(bool(normalized_images))
                entities_data["domain"].append(self._extract_domain(doc.get("url", ""))[:500])
                entities_data["content_hash"].append(abs(hash(content)) % (10**8))
                entities_data["images_json"].append(
                    json.dumps(normalized_images, ensure_ascii=False)[:65535]
                )
                entities_data["key_terms"].append(
                    json.dumps(self._extract_key_terms(content)[:20], ensure_ascii=False)[:2000]
                )

            # Generate embeddings with retry
            max_retries = 3
            embeddings = []
            
            for attempt in range(max_retries):
                try:
                    embeddings = await ai_service.generate_embeddings(texts)
                    if embeddings and len(embeddings) == len(texts):
                        logger.info(f"✅ Generated {len(embeddings)} embeddings")
                        break
                    else:
                        logger.warning(f"Embedding attempt {attempt + 1} failed: length mismatch")
                except Exception as e:
                    logger.warning(f"Embedding attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)

            if not embeddings or len(embeddings) != len(texts):
                logger.error(f"Failed to generate embeddings after {max_retries} attempts")
                return []

            # Handle dimension mismatch
            new_dim = len(embeddings[0]) if embeddings and isinstance(embeddings[0], (list, tuple)) else None
            
            if new_dim and new_dim != self.embedding_dim:
                logger.warning(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {new_dim}")
                
                try:
                    current_count = self.collection.num_entities if self.collection else 0
                except Exception:
                    current_count = None

                if current_count in (None, 0):
                    logger.info(f"Recreating collection with new embedding_dim: {new_dim}")
                    try:
                        await self.delete_collection()
                    except Exception as e:
                        logger.warning(f"Failed to delete collection: {e}")

                    self.embedding_dim = new_dim
                    await self._create_collection()
                    self.collection = Collection(self.collection_name, using=self.milvus_alias)
                    logger.info(f"Recreated collection with embedding_dim: {self.embedding_dim}")
                else:
                    logger.error(f"Cannot insert: collection has data with dim={self.embedding_dim}")
                    return []

            # Final prep and insert
            embeddings = self._ensure_embedding_list(embeddings)
            entities_data["embedding"] = embeddings
            entities_data["id"] = [self._flatten_to_str(i) for i in entities_data["id"]]

            # Validate
            valid, msg = self._validate_entities_data(entities_data)
            if not valid:
                logger.error(f"Entities validation failed: {msg}")
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
                        row["embedding"] = [float(x) for x in emb] if isinstance(emb, (list, tuple)) else [float(emb)]
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

            # Insert with retry
            insert_attempts = 3
            for attempt in range(insert_attempts):
                try:
                    self.collection.insert(rows)
                    self.collection.flush()
                    logger.info(f"✅ Inserted {len(rows)} documents")
                    break
                except DataNotMatchException as dnme:
                    logger.exception(f"Data mismatch on insert attempt {attempt + 1}: {dnme}")
                    if attempt >= insert_attempts - 1:
                        raise
                    await asyncio.sleep(2)
                except MilvusException as me:
                    logger.warning(f"Milvus insert attempt {attempt + 1} failed: {me}")
                    if attempt >= insert_attempts - 1:
                        raise
                    await asyncio.sleep(2)

            logger.info(f"✅ Successfully added {len(ids)} documents to Milvus")
            return ids

        except Exception as e:
            logger.exception(f"❌ Error adding documents to Milvus: {e}")
            return []

    # ---------- Search Documents ----------
    
    async def search_documents(self, query: str, n_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """Enhanced search with multi-stage relevance scoring"""
        if not self.collection:
            await self.initialize()
        
        if not self.collection:
            logger.warning("Milvus not available, returning empty results")
            return []

        try:
            # Preprocess query
            cleaned_query, key_terms = self._preprocess_query(query)
            if not cleaned_query:
                return []

            logger.info(f"🔍 Search: '{query}' -> key terms: {key_terms[:5]}")

            # Query expansion
            expanded_query = cleaned_query
            if self.search_config["enable_query_expansion"]:
                expanded_query = await self._expand_query(cleaned_query, key_terms)

            # Determine search parameters
            default_k = int(getattr(settings, "MILVUS_QUERY_TOP_K", 50))
            initial_k = min(
                self.search_config["max_initial_results"],
                int(n_results or default_k) * 3
            )
            final_k = int(n_results or default_k)

            # Generate query embedding
            query_embeddings = await ai_service.generate_embeddings([expanded_query])
            if not query_embeddings:
                logger.warning("Failed to generate query embeddings")
                return []

            # Search Milvus
            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": int(os.getenv("MILVUS_NPROBE", "32"))}
            }

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

            if not results or len(results) == 0 or len(results[0]) == 0:
                logger.info("No documents found in Milvus")
                return []

            # Score and filter results
            scored_results = []
            for hit in results[0]:
                entity = hit.entity
                metadata = {
                    "url": entity.get("url", ""),
                    "title": entity.get("title", ""),
                    "format": entity.get("format", ""),
                    "timestamp": entity.get("timestamp", ""),
                    "source": entity.get("source", ""),
                    "content_length": entity.get("content_length", 0),
                    "word_count": entity.get("word_count", 0),
                    "image_count": entity.get("image_count", 0),
                    "has_images": entity.get("has_images", False),
                    "domain": entity.get("domain", ""),
                    "content_hash": entity.get("content_hash", 0),
                    "images_json": entity.get("images_json", ""),
                    "key_terms": entity.get("key_terms", ""),
                }
                
                content = entity.get("content", "")
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

            # Sort by relevance
            scored_results.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            logger.info(f"Initial scoring: {len(scored_results)} results above threshold")

            # Apply diversity filtering if enabled
            if self.search_config["enable_context_enrichment"] and len(scored_results) > final_k:
                scored_results = self._apply_diversity_filtering(
                    scored_results,
                    self.search_config["diversity_factor"]
                )

            # Semantic reranking
            if self.search_config["enable_semantic_rerank"] and len(scored_results) > final_k:
                rerank_candidates = min(len(scored_results), self.search_config["rerank_top_k"])
                scored_results = await self._semantic_rerank(
                    scored_results[:rerank_candidates],
                    query,
                    rerank_candidates
                )

            final_results = scored_results[:final_k]
            
            logger.info(f"✅ Returning {len(final_results)} relevant documents (threshold: {self.search_config['min_relevance_threshold']})")
            return final_results

        except Exception as e:
            logger.exception(f"❌ Milvus search error: {e}")
            return []

    async def _semantic_rerank(self, results: List[Dict], query: str, top_k: int) -> List[Dict]:
        """Enhanced semantic reranking with cosine similarity"""
        try:
            # Extract content for reranking
            contents = [result["content"][:3000] for result in results[:top_k]]
            all_texts = [query] + contents
            
            # Generate embeddings
            embeddings = await ai_service.generate_embeddings(all_texts)
            
            if not embeddings or len(embeddings) != len(all_texts):
                logger.warning("Reranking failed: embedding generation error")
                return results[:top_k]
            
            query_embedding = embeddings[0]
            content_embeddings = embeddings[1:]

            def cosine_similarity(a, b):
                """Compute cosine similarity"""
                if not a or not b or len(a) != len(b):
                    return 0.0
                
                dot_product = sum(x * y for x, y in zip(a, b))
                norm_a = math.sqrt(sum(x * x for x in a))
                norm_b = math.sqrt(sum(x * x for x in b))
                
                if norm_a == 0 or norm_b == 0:
                    return 0.0
                
                return dot_product / (norm_a * norm_b)

            # Compute rerank scores
            for i, result in enumerate(results[:len(content_embeddings)]):
                semantic_sim = cosine_similarity(query_embedding, content_embeddings[i])
                original_score = result["relevance_score"]
                
                # Weighted combination (favor original score slightly)
                result["semantic_rerank_score"] = (
                    original_score * 0.65 +
                    semantic_sim * 0.35
                )
                result["semantic_similarity"] = semantic_sim

            # Sort by rerank score
            results.sort(
                key=lambda x: x.get("semantic_rerank_score", x["relevance_score"]),
                reverse=True
            )
            
            logger.info("✅ Applied semantic reranking")
            
        except Exception as e:
            logger.warning(f"Semantic reranking failed: {e}")
        
        return results[:top_k]

    # ---------- Utility Methods ----------
    
    def _clean_and_normalize_content(self, content: str) -> str:
        """Enhanced content cleaning"""
        if not content:
            return ""
        
        # Normalize whitespace
        content = re.sub(r'\s+', ' ', content)
        content = re.sub(r'[\r\n\t]+', ' ', content)
        
        # HTML entities
        html_entities = {
            '&nbsp;': ' ', '&amp;': '&', '&lt;': '<', '&gt;': '>',
            '&quot;': '"', '&#39;': "'", '&apos;': "'", '&mdash;': '—',
            '&ndash;': '–', '&hellip;': '...', '&copy;': '©'
        }
        
        for entity, char in html_entities.items():
            content = content.replace(entity, char)
        
        # Remove excessive punctuation
        content = re.sub(r'([.!?])\1+', r'\1', content)
        
        # Remove control characters
        content = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', content)
        
        return content.strip()

    def _extract_key_terms(self, content: str) -> List[str]:
        """Enhanced key term extraction with frequency analysis"""
        if not content:
            return []
        
        # Extract words
        words = re.findall(r'\b[a-zA-Z]{4,}\b', content.lower())
        
        # Count frequencies
        word_freq = defaultdict(int)
        for word in words:
            if word not in self.stopwords and len(word) > 3:
                word_freq[word] += 1
        
        # Sort by frequency
        sorted_terms = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Filter: must appear at least twice and not too common
        filtered_terms = [
            term for term, freq in sorted_terms
            if 2 <= freq <= len(words) * 0.1  # Not too rare, not too common
        ]
        
        return filtered_terms[:20]

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            parsed = urllib.parse.urlparse(url)
            return parsed.netloc
        except Exception:
            return ""

    def _normalize_images(self, images: Any, page_url: str, 
                         content_snippet: str, max_images: int = 20) -> List[Dict[str, Any]]:
        """Enhanced image normalization"""
        if not images or not isinstance(images, (list, tuple)):
            return []
        
        normalized = []
        seen_urls = set()
        
        for img in images[:max_images * 2]:  # Process more, filter later
            if not isinstance(img, dict):
                continue
            
            url = self._safe_str(img.get("url"))
            if not url:
                continue
            
            # Skip duplicates
            if url in seen_urls:
                continue
            
            # Skip noise images
            url_lower = url.lower()
            noise_patterns = [
                'logo', 'icon', 'favicon', 'sprite', 'banner',
                'avatar', 'badge', 'pixel', 'tracker', '1x1'
            ]
            if any(pattern in url_lower for pattern in noise_patterns):
                continue
            
            # Make absolute URL
            if page_url and not url.startswith(("http://", "https://", "data:")):
                try:
                    url = urllib.parse.urljoin(page_url, url)
                except Exception:
                    pass
            
            # Extract metadata
            alt = self._safe_str(img.get("alt"))[:500]
            caption = self._safe_str(img.get("caption"))[:500]
            img_type = self._safe_str(img.get("type"))[:100]
            
            # Build image text
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
        """Decode bytes to text safely"""
        if value is None:
            return ""
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8")
            except UnicodeDecodeError:
                return value.decode("latin-1", errors="replace")
        return str(value)

    def _soft_truncate(self, text: str, limit: int = 8000) -> str:
        """Soft truncate with sentence boundary awareness"""
        if len(text) <= limit:
            return text
        
        # Try to truncate at sentence boundary
        truncated = text[:limit]
        last_period = truncated.rfind('. ')
        
        if last_period > limit * 0.8:  # If we can keep 80%+
            return truncated[:last_period + 1] + " [truncated]"
        
        return truncated + "... [truncated]"

    def _coerce_images_from_meta(self, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract images from metadata safely"""
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
        """Convert to string safely"""
        if v is None:
            return ""
        return str(v).strip()

    # ---------- Collection Management ----------
    
    async def delete_collection(self) -> None:
        """Delete collection safely"""
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
        """Get comprehensive collection statistics"""
        if not self.collection:
            await self.initialize()
        
        if not self.collection:
            return {
                "document_count": 0,
                "collection_name": "unavailable",
                "status": "unavailable",
                "search_config": self.search_config,
                "ai_services": "grok+openrouter",
                "database": "milvus"
            }

        try:
            # Flush to ensure count is accurate
            try:
                self.collection.flush()
            except Exception:
                pass

            count = self.collection.num_entities
            
            # Get index information
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
                "error": str(e)
            }

    async def close(self) -> None:
        """Close Milvus connection safely"""
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
                    logger.warning(f"Error disconnecting: {e}")
        except Exception as e:
            logger.exception(f"❌ Error closing Milvus connection: {e}")

    def __del__(self):
        """Cleanup on destruction"""
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