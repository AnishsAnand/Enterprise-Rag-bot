import json
import os
import urllib.parse
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import re
import math
import asyncio

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import settings
from app.services.ai_service import ai_service 

import logging
logger = logging.getLogger(__name__)


class ChromaService:
    """
    Streamlined ChromaDB service optimized for Grok+OpenRouter AI services.
    Enhanced search with better relevance scoring and query processing.
    """

    def __init__(self):
        self.client: Optional[Any] = None
        self.collection: Optional[Any] = None
        self._needs_explicit_persist: bool = False
        
        # Optimized search configuration for Grok+OpenRouter
        self.search_config = {
            "min_relevance_threshold": float(os.getenv("CHROMA_MIN_RELEVANCE", "0.15")),  # FIXED: Lowered from 0.3 to 0.15
            "max_initial_results": int(os.getenv("CHROMA_MAX_INITIAL_RESULTS", "80")),
            "rerank_top_k": int(os.getenv("CHROMA_RERANK_TOP_K", "40")),
            "enable_query_expansion": os.getenv("CHROMA_ENABLE_QUERY_EXPANSION", "false").lower() == "true",  # FIXED: Disabled by default
            "enable_semantic_rerank": os.getenv("CHROMA_ENABLE_SEMANTIC_RERANK", "true").lower() == "true"
        }

        try:
            os.environ.setdefault("CHROMA_TELEMETRY_DISABLED", "true")
            os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
        except Exception:
            pass

    async def initialize(self) -> None:
        """Initialize ChromaDB client and collection"""
        if self.collection:
            return

        collection_name = getattr(settings, "CHROMA_COLLECTION", "enterprise_rag_knowledge")
        persist_dir = getattr(settings, "CHROMA_PERSIST_DIRECTORY", "./chroma_db")
        persist_dir_abs = os.path.abspath(persist_dir)

        try:
            os.makedirs(persist_dir_abs, exist_ok=True)
            logger.info(f"Ensured ChromaDB persist directory: {persist_dir_abs}")
        except Exception as e:
            logger.exception(f"Failed to ensure persist dir {persist_dir_abs}: {e}")

        self.client = None
        self.collection = None
        self._needs_explicit_persist = False
        init_errors: List[str] = []

        # Try multiple initialization approaches
        try:
            try:
                self.client = chromadb.PersistentClient(
                    path=persist_dir_abs,
                    settings=ChromaSettings(
                        anonymized_telemetry=False,
                        allow_reset=False
                    )
                )
                logger.info("ChromaDB PersistentClient initialized successfully")
            except TypeError:
                self.client = chromadb.PersistentClient(
                    persist_directory=persist_dir_abs
                )
                logger.info("ChromaDB PersistentClient initialized (fallback method)")
        except Exception as e:
            init_errors.append(f"PersistentClient error: {e}")
            self.client = None

        if self.client is None:
            try:
                try:
                    self.client = chromadb.Client(
                        ChromaSettings(
                            anonymized_telemetry=False,
                            allow_reset=False,
                            persist_directory=persist_dir_abs,
                        )
                    )
                except Exception:
                    self.client = chromadb.Client(
                        persist_directory=persist_dir_abs
                    )

                self._needs_explicit_persist = True
                logger.info("ChromaDB Client initialized (legacy method)")
            except Exception as e:
                init_errors.append(f"Client error: {e}")
                self.client = None

        if self.client is None:
            logger.error(f"Failed to initialize ChromaDB: {' | '.join(init_errors)}")
            return

        # Initialize collection
        try:
            get_or_create = getattr(self.client, "get_or_create_collection", None)
            if callable(get_or_create):
                self.collection = get_or_create(
                    name=collection_name,
                    metadata={
                        "description": "Enterprise RAG Knowledge Base",
                        "version": "2.1",
                        "ai_services": "grok+openrouter",
                        "created_at": datetime.now().isoformat()
                    }
                )
            else:
                try:
                    self.collection = self.client.get_collection(collection_name)
                except Exception:
                    self.collection = self.client.create_collection(
                        name=collection_name,
                        metadata={
                            "description": "Enterprise RAG Knowledge Base",
                            "version": "2.1",
                            "ai_services": "grok+openrouter",
                            "created_at": datetime.now().isoformat()
                        }
                    )

            logger.info(f"Connected to ChromaDB collection: {collection_name}")
        except Exception as e:
            logger.exception(f"Failed to create/connect ChromaDB collection: {e}")

    def _preprocess_query(self, query: str) -> Tuple[str, List[str]]:
        """Enhanced query preprocessing optimized for Grok"""
        if not query:
            return "", []
        
        # Clean and normalize the query
        original_query = query.strip()
        cleaned_query = re.sub(r'\s+', ' ', original_query.lower())
        
        # Extract key terms with better filtering
        key_terms = []
        
        # Technical and domain-specific patterns
        technical_patterns = [
            r'\b[A-Z]{2,}\b',  # Acronyms (API, HTTP, SSL)
            r'\b\w+(?:ing|ion|tion|sion|ness|ment|able|ible)\b',  # Technical suffixes
            r'\b\w+[-_]\w+(?:[-_]\w+)*\b',  # Multi-part terms (web-server, snake_case)
            r'\b\d+(?:\.\d+)*\w*\b',  # Versions and measurements (v2.1, 3.5GB)
            r'\b(?:how|what|where|when|why|which)\s+\w+\b'  # Question patterns
        ]
        
        for pattern in technical_patterns:
            matches = re.findall(pattern, original_query)
            key_terms.extend(matches)
        
        # Extract meaningful words (length > 3, not common stopwords)
        stopwords = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one',
            'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see',
            'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use', 'that',
            'with', 'have', 'this', 'will', 'your', 'from', 'they', 'know', 'want', 'been', 'good',
            'much', 'some', 'time', 'very', 'when', 'come', 'here', 'just', 'like', 'long', 'make',
            'many', 'over', 'such', 'take', 'than', 'them', 'well', 'were'
        }
        
        words = re.findall(r'\b\w+\b', cleaned_query)
        key_terms.extend([word for word in words if len(word) > 3 and word not in stopwords])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in key_terms:
            if term.lower() not in seen:
                seen.add(term.lower())
                unique_terms.append(term)
        
        return cleaned_query, unique_terms[:12]  # Limit to top 12 terms

    def _calculate_enhanced_relevance(self, document: str, metadata: Dict, 
                                    query: str, key_terms: List[str], distance: float) -> float:
        """Enhanced relevance calculation using multiple signals optimized for Grok responses"""
        if not document:
            return 0.0
        
        doc_lower = document.lower()
        query_lower = query.lower()
        
        # Base semantic similarity (inverse of distance)
        semantic_score = max(0.0, 1.0 - distance)
        
        # Enhanced term frequency scoring
        term_score = 0.0
        if key_terms:
            term_matches = sum(1 for term in key_terms if term.lower() in doc_lower)
            term_score = min(1.0, term_matches / len(key_terms))
        
        # Exact phrase matching with higher weight
        phrase_score = 0.0
        query_words = query_lower.split()
        if len(query_words) > 1:
            if query_lower in doc_lower:
                phrase_score = 0.4  # Increased from 0.3
            else:
                # Enhanced partial phrase matching
                phrase_matches = 0
                for i in range(len(query_words) - 1):
                    bigram = f"{query_words[i]} {query_words[i+1]}"
                    if bigram in doc_lower:
                        phrase_matches += 1
                if phrase_matches > 0:
                    phrase_score = min(0.3, phrase_matches * 0.15)
        
        # Document quality signals
        quality_score = 0.0
        doc_length = len(document)
        if 150 <= doc_length <= 4000:  # Optimal length range
            quality_score += 0.15
        elif 50 <= doc_length < 150:
            quality_score += 0.08
        
        # Enhanced title matching
        title_score = 0.0
        if metadata.get("title"):
            title_lower = metadata["title"].lower()
            title_matches = sum(1 for term in key_terms if term.lower() in title_lower)
            if title_matches > 0:
                title_score = min(0.2, title_matches * 0.1)
        
        # URL path relevance
        url_score = 0.0
        if metadata.get("url"):
            url_lower = metadata["url"].lower()
            url_matches = sum(1 for term in key_terms if term.lower() in url_lower)
            if url_matches > 0:
                url_score = min(0.12, url_matches * 0.06)
        
        # Content recency bonus
        recency_score = 0.0
        if metadata.get("timestamp"):
            try:
                doc_date = datetime.fromisoformat(metadata["timestamp"].replace("Z", "+00:00"))
                days_old = (datetime.now().replace(tzinfo=doc_date.tzinfo) - doc_date).days
                if days_old < 7:
                    recency_score = 0.08
                elif days_old < 30:
                    recency_score = 0.05
                elif days_old < 90:
                    recency_score = 0.02
            except Exception:
                pass
        
        # Source quality bonus
        source_score = 0.0
        source = metadata.get("source", "")
        if source in ["manual_upload", "widget_scrape"]:
            source_score = 0.05  # Recent manual additions get slight boost
        
        # Combine all scores with optimized weights for Grok
        final_score = (
            semantic_score * 0.35 +      # Reduced semantic weight
            term_score * 0.25 +          # Maintained term matching
            phrase_score * 0.2 +         # Increased phrase matching
            quality_score * 0.06 +       # Document quality
            title_score * 0.08 +         # Title relevance
            url_score * 0.04 +           # URL relevance
            recency_score * 0.02         # Content freshness
        )
        
        return min(1.0, final_score)

    async def _expand_query(self, original_query: str, key_terms: List[str]) -> str:
        """Query expansion using Grok for better semantic understanding - DISABLED by default for performance"""
        if not self.search_config["enable_query_expansion"]:
            return original_query
        
        try:
            expansion_prompt = f"""Given this search query, provide 2-4 related terms or technical synonyms that would help find more relevant information. Focus on:
- Technical alternatives and abbreviations
- Domain-specific terminology
- Common variations in phrasing

Query: {original_query}
Key terms: {', '.join(key_terms[:6])}

Provide only the additional terms separated by spaces. No explanations.

Additional terms:"""

            expanded_terms = ai_service._llm_chat(
                expansion_prompt,
                max_tokens=50,
                temperature=0.3,
                system_message="You are a search expert. Provide precise technical terms and synonyms."
            )
            
            if expanded_terms:
                # Clean and validate the response
                additional_terms = expanded_terms.strip().split()
                # Filter out terms already in the original query
                original_words = set(original_query.lower().split())
                new_terms = [term for term in additional_terms 
                           if term.lower() not in original_words and len(term) > 2]
                
                if new_terms:
                    expanded_query = f"{original_query} {' '.join(new_terms[:3])}"
                    logger.info(f"Query expanded: '{original_query}' + {new_terms[:3]}")
                    return expanded_query
        
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
        
        return original_query

    async def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Enhanced document addition optimized for Grok+OpenRouter"""
        if not self.collection:
            await self.initialize()
        if not self.collection:
            logger.warning("ChromaDB not available, skipping document addition")
            return []

        try:
            ids: List[str] = []
            texts: List[str] = []
            metadatas: List[Dict[str, Any]] = []

            for doc in documents or []:
                doc_id = str(uuid.uuid4())
                ids.append(doc_id)

                # Enhanced content preprocessing
                content = self._decode_to_text(doc.get("content", ""))
                content = self._clean_and_normalize_content(content)
                content = self._soft_truncate(content, limit=7000)  # Slightly reduced for efficiency
                texts.append(content)

                # Enhanced image processing
                normalized_images = self._normalize_images(
                    images=doc.get("images", []),
                    page_url=doc.get("url", ""),
                    content_snippet=content[:300],
                    max_images=15  # Optimized count
                )

                # Optimized metadata
                metadata: Dict[str, Any] = {
                    "url": str(doc.get("url", ""))[:1500],
                    "format": str(doc.get("format", "text"))[:50],
                    "timestamp": doc.get("timestamp", "") or datetime.now().isoformat(),
                    "source": str(doc.get("source", "web_scraping"))[:50],
                    "title": str(doc.get("title", ""))[:300],
                    "content_length": len(content),
                    "images_json": json.dumps(normalized_images, ensure_ascii=False),
                    "image_count": len(normalized_images),
                    "has_images": bool(normalized_images),
                    "word_count": len(content.split()),
                    "key_terms": json.dumps(self._extract_key_terms(content)[:15]),  # Reduced count
                    "domain": self._extract_domain(doc.get("url", "")),
                    "content_hash": abs(hash(content)) % (10**8),
                }
                metadatas.append(metadata)

            # Generate embeddings with retry logic
            max_retries = 2
            embeddings = []
            
            for attempt in range(max_retries):
                try:
                    embeddings = await ai_service.generate_embeddings(texts)
                    if embeddings and len(embeddings) == len(texts):
                        break
                    else:
                        logger.warning(f"Embedding generation attempt {attempt + 1} failed: length mismatch")
                except Exception as e:
                    logger.warning(f"Embedding generation attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1)

            if not embeddings or len(embeddings) != len(texts):
                logger.error(f"Failed to generate embeddings after {max_retries} attempts")
                return []

            # Add to ChromaDB with retry
            for attempt in range(2):
                try:
                    self.collection.add(
                        documents=texts,
                        metadatas=metadatas,
                        ids=ids,
                        embeddings=embeddings
                    )
                    break
                except Exception as e:
                    if attempt == 1:
                        raise e
                    logger.warning(f"ChromaDB add attempt {attempt + 1} failed: {e}")
                    await asyncio.sleep(1)

            # Persist if needed
            if self._needs_explicit_persist and hasattr(self.client, "persist"):
                try:
                    self.client.persist()
                except Exception as e:
                    logger.warning(f"ChromaDB persist failed: {e}")

            logger.info(f"Successfully added {len(ids)} documents to ChromaDB")
            return ids

        except Exception as e:
            logger.exception(f"Error adding documents to ChromaDB: {e}")
            return []

    async def search_documents(self, query: str, n_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """Enhanced document search optimized for Grok+OpenRouter"""
        if not self.collection:
            await self.initialize()
        if not self.collection:
            logger.warning("ChromaDB not available, returning empty results")
            return []

        try:
            # Enhanced query preprocessing
            cleaned_query, key_terms = self._preprocess_query(query)
            if not cleaned_query:
                return []

            logger.info(f"Searching: '{query}' -> key terms: {key_terms[:5]}")

            # FIXED: Skip query expansion by default (controlled by config)
            expanded_query = cleaned_query
            if self.search_config["enable_query_expansion"]:
                expanded_query = await self._expand_query(cleaned_query, key_terms)

            # Optimized search parameters
            default_k = getattr(settings, "CHROMA_QUERY_TOP_K", 40)
            initial_k = min(self.search_config["max_initial_results"], int(n_results or default_k) * 2)
            final_k = int(n_results or default_k)

            # Generate embeddings
            query_embeddings = await ai_service.generate_embeddings([expanded_query])
            if not query_embeddings:
                logger.warning("Failed to generate query embeddings")
                return []

            # Execute search
            try:
                results = self.collection.query(
                    query_embeddings=query_embeddings,
                    n_results=initial_k,
                    include=["documents", "metadatas", "distances"]
                )
            except Exception as e:
                logger.error(f"ChromaDB query failed: {e}")
                return []

            # Parse results
            docs = results.get("documents", [[]])[0] if results.get("documents") else []
            metas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
            dists = results.get("distances", [[]])[0] if results.get("distances") else [1.0] * len(docs)

            if not docs:
                logger.info("No documents found in ChromaDB")
                return []

            # Enhanced scoring and filtering
            scored_results = []
            for doc, meta, dist in zip(docs, metas, dists):
                meta = dict(meta or {})
                
                # Calculate enhanced relevance
                relevance_score = self._calculate_enhanced_relevance(
                    document=doc,
                    metadata=meta,
                    query=query,
                    key_terms=key_terms,
                    distance=dist
                )
                
                # Apply threshold filter
                if relevance_score >= self.search_config["min_relevance_threshold"]:
                    meta["images"] = self._coerce_images_from_meta(meta)
                    
                    scored_results.append({
                        "content": doc,
                        "metadata": meta,
                        "distance": float(dist),
                        "relevance_score": relevance_score,
                        "semantic_similarity": max(0.0, 1.0 - float(dist))
                    })

            # Sort by relevance
            scored_results.sort(key=lambda x: x["relevance_score"], reverse=True)

            # Apply semantic re-ranking if enabled
            if (self.search_config["enable_semantic_rerank"] and 
                len(scored_results) > final_k):
                
                scored_results = await self._semantic_rerank(scored_results, query, final_k * 2)

            # Return final results
            final_results = scored_results[:final_k]
            
            logger.info(f"Found {len(final_results)} relevant documents (from {len(docs)} searched, threshold: {self.search_config['min_relevance_threshold']})")
            
            return final_results

        except Exception as e:
            logger.exception(f"ChromaDB search error: {e}")
            return []

    async def _semantic_rerank(self, results: List[Dict], query: str, top_k: int) -> List[Dict]:
        """Semantic re-ranking using Grok/OpenRouter embeddings"""
        try:
            # Extract content for re-ranking (limit for efficiency)
            contents = [result["content"][:2000] for result in results[:top_k]]
            all_texts = [query] + contents
            
            # Generate embeddings
            embeddings = await ai_service.generate_embeddings(all_texts)
            if not embeddings or len(embeddings) != len(all_texts):
                logger.warning("Re-ranking failed: embedding generation error")
                return results[:top_k]
            
            query_embedding = embeddings[0]
            content_embeddings = embeddings[1:]
            
            # Calculate cosine similarities
            def cosine_similarity(a, b):
                if not a or not b or len(a) != len(b):
                    return 0.0
                dot_product = sum(x * y for x, y in zip(a, b))
                norm_a = math.sqrt(sum(x * x for x in a))
                norm_b = math.sqrt(sum(x * x for x in b))
                if norm_a == 0 or norm_b == 0:
                    return 0.0
                return dot_product / (norm_a * norm_b)
            
            # Update scores
            for i, result in enumerate(results[:len(content_embeddings)]):
                semantic_sim = cosine_similarity(query_embedding, content_embeddings[i])
                # Weighted combination favoring original relevance
                result["semantic_rerank_score"] = (
                    result["relevance_score"] * 0.75 + semantic_sim * 0.25
                )
            
            # Re-sort
            results.sort(key=lambda x: x.get("semantic_rerank_score", x["relevance_score"]), reverse=True)
            logger.info("Applied semantic re-ranking")
            
        except Exception as e:
            logger.warning(f"Semantic re-ranking failed: {e}")
        
        return results[:top_k]

    # Utility methods (streamlined)
    def _clean_and_normalize_content(self, content: str) -> str:
        """Clean and normalize content for better indexing"""
        if not content:
            return ""
        
        # Basic cleaning
        content = re.sub(r'\s+', ' ', content)
        content = re.sub(r'[\r\n\t]+', ' ', content)
        
        # Remove HTML entities
        html_entities = {
            '&nbsp;': ' ', '&amp;': '&', '&lt;': '<', '&gt;': '>', 
            '&quot;': '"', '&#39;': "'", '&apos;': "'"
        }
        for entity, replacement in html_entities.items():
            content = content.replace(entity, replacement)
        
        return content.strip()

    def _extract_key_terms(self, content: str) -> List[str]:
        """Extract key terms optimized for search indexing"""
        if not content:
            return []
        
        words = re.findall(r'\b[a-zA-Z]{4,}\b', content.lower())
        
        # Enhanced stopwords list
        stopwords = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one',
            'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see',
            'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use', 'that',
            'with', 'have', 'this', 'will', 'your', 'from', 'they', 'know', 'want', 'been', 'good',
            'much', 'some', 'time', 'very', 'when', 'come', 'here', 'just', 'like', 'long', 'make',
            'many', 'over', 'such', 'take', 'than', 'them', 'well', 'were', 'into', 'what', 'where',
            'there', 'which', 'using', 'used', 'then', 'also', 'only', 'about', 'page', 'click'
        }
        
        # Count frequencies and filter
        word_freq = {}
        for word in words:
            if word not in stopwords and len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Return top terms by frequency
        sorted_terms = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [term for term, freq in sorted_terms[:15] if freq > 1]

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            parsed = urllib.parse.urlparse(url)
            return parsed.netloc
        except Exception:
            return ""

    def _normalize_images(self, images: Any, page_url: str, content_snippet: str, max_images: int = 15) -> List[Dict[str, Any]]:
        """Normalize images with enhanced context"""
        if not images or not isinstance(images, (list, tuple)):
            return []

        normalized: List[Dict[str, Any]] = []
        for img in images[:max_images]:
            if not isinstance(img, dict):
                continue

            url = self._safe_str(img.get("url"))
            if not url:
                continue

            if page_url and not url.startswith(("http://", "https://", "data:")):
                try:
                    url = urllib.parse.urljoin(page_url, url)
                except Exception:
                    pass

            alt = self._safe_str(img.get("alt"))[:500]
            caption = self._safe_str(img.get("caption"))[:500]
            itype = self._safe_str(img.get("type"))[:100]

            # Enhanced text extraction
            image_text = self._safe_str(img.get("text"))
            if not image_text:
                text_parts = []
                if alt:
                    text_parts.append(alt)
                if caption:
                    text_parts.append(caption)
                if content_snippet:
                    text_parts.append(content_snippet[:150])
                
                image_text = " | ".join(text_parts)

            normalized.append({
                "url": url[:1500],
                "alt": alt,
                "caption": caption,
                "type": itype,
                "text": image_text[:800],
            })

        return normalized

    # Standard utility methods
    def _decode_to_text(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8")
            except UnicodeDecodeError:
                return value.decode("latin-1", errors="replace")
        return str(value).encode("utf-8", errors="replace").decode("utf-8")

    def _soft_truncate(self, text: str, limit: int = 7000) -> str:
        if len(text) <= limit:
            return text
        return text[:limit] + "... [truncated]"

    def _coerce_images_from_meta(self, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract images from metadata"""
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

    # Collection management methods
    async def delete_collection(self) -> None:
        """Delete the collection"""
        try:
            if self.client and self.collection:
                collection_name = getattr(settings, "CHROMA_COLLECTION", "enterprise_rag_knowledge")
                try:
                    if hasattr(self.client, "delete_collection"):
                        self.client.delete_collection(collection_name)
                    elif hasattr(self.collection, "delete"):
                        self.collection.delete()
                    self.collection = None
                    logger.info("ChromaDB collection deleted")
                except Exception as e:
                    logger.exception(f"Error deleting ChromaDB collection: {e}")

                if self._needs_explicit_persist and hasattr(self.client, "persist"):
                    try:
                        self.client.persist()
                    except Exception:
                        pass
        except Exception as e:
            logger.exception(f"Error in delete_collection: {e}")

    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        if not self.collection:
            await self.initialize()

        if not self.collection:
            return {
                "document_count": 0,
                "collection_name": "unavailable",
                "status": "unavailable",
                "search_config": self.search_config,
                "ai_services": "grok+openrouter"
            }

        try:
            try:
                count = self.collection.count()
            except Exception:
                count = 0

            return {
                "document_count": int(count),
                "collection_name": getattr(settings, "CHROMA_COLLECTION", "enterprise_rag_knowledge"),
                "status": "active",
                "search_config": self.search_config,
                "ai_services": "grok+openrouter",
                "last_updated": datetime.now().isoformat()
            }
        except Exception as e:
            logger.exception(f"Error getting ChromaDB stats: {e}")
            return {
                "document_count": 0,
                "collection_name": "error",
                "status": "error",
                "search_config": self.search_config,
                "ai_services": "grok+openrouter"
            }


chroma_service = ChromaService()
