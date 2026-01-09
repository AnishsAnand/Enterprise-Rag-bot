"""
Production-grade RAG search service with accuracy improvements.
Prevents fallback to generic responses and ensures relevant results.
FIXED FOR PYDANTIC V2 COMPATIBILITY
"""

import logging
import asyncio
from typing import List, Dict, Optional, Tuple
import numpy as np
from datetime import datetime

from app.services.postgres_service import postgres_service
from app.services.ai_service import ai_service
from app.core.config import settings

logger = logging.getLogger(__name__)


class RAGSearchService:
    """
    Advanced RAG search with multi-stage retrieval and ranking.
    Ensures accurate, relevant responses based on trained knowledge base.
    
    PRODUCTION-READY: Fixed for Pydantic v2 Settings
    """
    
    def __init__(self):
        """
        Initialize RAG Search Service with production settings.
        FIXED: Changed from settings.get() to getattr() for Pydantic v2 compatibility
        """
        # ✅ FIXED: Use getattr() instead of settings.get()
        self.min_relevance_threshold = float(
            getattr(settings, "MIN_RELEVANCE_THRESHOLD", 0.5)
        )
        self.enable_query_expansion = getattr(
            settings, "ENABLE_QUERY_EXPANSION", True
        )
        self.enable_reranking = getattr(
            settings, "ENABLE_SEMANTIC_RERANK", True
        )
        self.max_chunks_return = int(
            getattr(settings, "MAX_CHUNKS_RETURN", 10)
        )
        self.chunk_overlap_penalty = 0.1
        
        logger.info("✅ RAGSearchService initialized (Production Mode)")
        logger.info(f"   - Min relevance threshold: {self.min_relevance_threshold}")
        logger.info(f"   - Query expansion: {self.enable_query_expansion}")
        logger.info(f"   - Semantic reranking: {self.enable_reranking}")
        logger.info(f"   - Max chunks return: {self.max_chunks_return}")
    
    async def search(
        self,
        query: str,
        user_id: Optional[int] = None,
        knowledge_base_id: Optional[int] = None,
        top_k: int = 10,
        include_confidence: bool = True
    ) -> Dict:
        """
        Comprehensive RAG search with multi-stage retrieval.
        
        Args:
            query: Search query string
            user_id: Optional user ID for personalization
            knowledge_base_id: Optional KB ID filter
            top_k: Number of top results to return
            include_confidence: Include confidence scores in results
        
        Returns:
            {
                "chunks": [
                    {
                        "id": str,
                        "text": str,
                        "document_id": int,
                        "document_title": str,
                        "confidence_score": float,
                        "relevance_reason": str
                    }
                ],
                "total_results": int,
                "search_quality": "high" | "medium" | "low",
                "query_expanded_terms": [str],
                "execution_time_ms": float,
                "metadata": {...}
            }
        """
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"[RAG Search] Starting search for query: '{query[:100]}'")
            
            # Stage 1: Query Expansion (optional)
            expanded_queries = [query]
            if self.enable_query_expansion:
                expanded_queries = await self._expand_query(query)
                logger.info(f"[RAG Search] Expanded queries: {expanded_queries}")
            
            # Stage 2: Multi-source Retrieval
            retrieved_chunks = await self._retrieve_chunks(
                queries=expanded_queries,
                user_id=user_id,
                knowledge_base_id=knowledge_base_id,
                top_k=top_k * 2  # Get more to rerank
            )
            
            if not retrieved_chunks:
                logger.warning(f"[RAG Search] No chunks retrieved for query: '{query}'")
                return self._create_no_results_response(
                    query, 
                    expanded_queries, 
                    start_time
                )
            
            logger.info(f"[RAG Search] Retrieved {len(retrieved_chunks)} chunks")
            
            # Stage 3: Deduplication
            deduplicated = self._deduplicate_chunks(retrieved_chunks)
            logger.info(f"[RAG Search] Deduplicated to {len(deduplicated)} chunks")
            
            # Stage 4: Semantic Reranking (crucial for accuracy)
            if self.enable_reranking:
                reranked = await self._rerank_chunks(query, deduplicated)
            else:
                reranked = deduplicated
            
            # Stage 5: Filtering by threshold
            filtered_chunks = [
                chunk for chunk in reranked 
                if chunk.get("confidence_score", 0) >= self.min_relevance_threshold
            ]
            
            if not filtered_chunks:
                logger.warning(
                    f"[RAG Search] No chunks passed confidence threshold "
                    f"({self.min_relevance_threshold})"
                )
                return self._create_no_results_response(
                    query, 
                    expanded_queries, 
                    start_time
                )
            
            # Stage 6: Context enrichment
            enriched_chunks = await self._enrich_chunks(
                filtered_chunks,
                query
            )
            
            # Final results
            final_chunks = enriched_chunks[:top_k]
            
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            logger.info(
                f"[RAG Search] Final results: {len(final_chunks)} chunks "
                f"in {execution_time:.1f}ms"
            )
            
            return {
                "chunks": final_chunks,
                "total_results": len(final_chunks),
                "search_quality": self._assess_search_quality(final_chunks),
                "query_expanded_terms": expanded_queries,
                "execution_time_ms": execution_time,
                "metadata": {
                    "retrieval_stage_results": len(retrieved_chunks),
                    "after_dedup": len(deduplicated),
                    "after_rerank": len(reranked),
                    "after_threshold": len(filtered_chunks),
                    "threshold_used": self.min_relevance_threshold,
                }
            }
        
        except Exception as e:
            logger.exception(f"[RAG Search] Error during search: {e}")
            raise
    
    async def _expand_query(self, query: str) -> List[str]:
        """
        Expand query with semantic variations and related terms.
        Uses AI to generate alternative phrasings.
        """
        try:
            prompt = f"""Generate 3-5 alternative phrasings and related search terms for this query:
Query: "{query}"

Return ONLY a JSON array of strings (variations and related terms), no other text.
Example: ["variation 1", "variation 2", "related term 1"]"""
            
            response = await ai_service.generate_response(prompt, [])
            
            # Handle different response formats
            if isinstance(response, dict):
                expanded = response.get("expanded_terms", [query])
            elif isinstance(response, list):
                expanded = response
            else:
                # Try parsing as JSON if string
                try:
                    import json
                    expanded = json.loads(response)
                    if not isinstance(expanded, list):
                        expanded = [query]
                except:
                    expanded = [query]
            
            # Ensure original query is included
            if query not in expanded:
                expanded.insert(0, query)
            
            return expanded if expanded else [query]
        
        except Exception as e:
            logger.warning(f"Query expansion failed, using original: {e}")
            return [query]
    
    async def _retrieve_chunks(
        self,
        queries: List[str],
        user_id: Optional[int],
        knowledge_base_id: Optional[int],
        top_k: int
    ) -> List[Dict]:
        """
        Retrieve chunks from multiple queries and combine results.
        Uses PostgreSQL vector store for semantic search.
        """
        all_chunks = {}
        
        for query in queries:
            try:
                # Use postgres_service.search_documents for retrieval
                # This is the production-ready method that's already working
                results = await postgres_service.search_documents(
                    query=query,
                    n_results=top_k
                )
                
                # Convert results to expected format
                for result in results:
                    chunk_id = result.get("metadata", {}).get("url", "") or str(hash(result.get("content", "")))
                    
                    if chunk_id not in all_chunks:
                        all_chunks[chunk_id] = {
                            "id": chunk_id,
                            "text": result.get("content", ""),
                            "score": result.get("relevance_score", 0),
                            "metadata": result.get("metadata", {}),
                            "document_title": result.get("metadata", {}).get("title", ""),
                            "document_id": chunk_id,
                        }
                    else:
                        # Update score if higher
                        if result.get("relevance_score", 0) > all_chunks[chunk_id].get("score", 0):
                            all_chunks[chunk_id]["score"] = result.get("relevance_score", 0)
            
            except Exception as e:
                logger.warning(f"Retrieval failed for query '{query}': {e}")
                continue
        
        return list(all_chunks.values())
    
    def _deduplicate_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Remove near-duplicate chunks based on content similarity.
        Uses simple substring matching for efficiency.
        """
        if not chunks:
            return []
        
        # Sort by score first to keep highest quality chunks
        sorted_chunks = sorted(
            chunks, 
            key=lambda x: x.get("score", 0), 
            reverse=True
        )
        
        deduplicated = [sorted_chunks[0]]
        
        for chunk in sorted_chunks[1:]:
            is_duplicate = False
            
            current_text = chunk.get("text", "").lower()
            
            for existing in deduplicated:
                existing_text = existing.get("text", "").lower()
                
                # Simple substring check
                if (len(current_text) > 50 and 
                    (current_text in existing_text or existing_text in current_text)):
                    is_duplicate = True
                    break
                
                # Jaccard similarity for additional duplicate detection
                current_words = set(current_text.split())
                existing_words = set(existing_text.split())
                if current_words and existing_words:
                    intersection = len(current_words & existing_words)
                    union = len(current_words | existing_words)
                    similarity = intersection / union if union > 0 else 0
                    if similarity > 0.8:  # 80% word overlap = duplicate
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                deduplicated.append(chunk)
        
        return deduplicated
    
    async def _rerank_chunks(
        self,
        query: str,
        chunks: List[Dict]
    ) -> List[Dict]:
        """
        Rerank chunks using semantic similarity to original query.
        Critical for ensuring relevant results and preventing generic responses.
        
        PRODUCTION NOTE: Uses cosine similarity between query and chunk embeddings.
        """
        if not chunks:
            return []
        
        try:
            # Get query embedding
            query_embeddings = await ai_service.generate_embeddings([query])
            if not query_embeddings:
                logger.warning("Failed to generate query embedding for reranking")
                # Fallback: use original scores
                for chunk in chunks:
                    chunk["confidence_score"] = chunk.get("score", 0.5)
                return chunks
            
            query_embedding = np.array(query_embeddings[0])
            
            # Score each chunk
            scored_chunks = []
            for chunk in chunks:
                # Get chunk embedding
                chunk_text = chunk.get("text", "")
                if not chunk_text:
                    continue
                
                chunk_embeddings = await ai_service.generate_embeddings([chunk_text[:1000]])  # Limit to first 1000 chars
                if not chunk_embeddings:
                    # Use original score if embedding fails
                    chunk_copy = chunk.copy()
                    chunk_copy["confidence_score"] = chunk.get("score", 0.5)
                    scored_chunks.append(chunk_copy)
                    continue
                
                chunk_embedding = np.array(chunk_embeddings[0])
                
                # Calculate cosine similarity
                try:
                    similarity = np.dot(query_embedding, chunk_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding) + 1e-10
                    )
                    
                    chunk_copy = chunk.copy()
                    # Combine with original score for robustness
                    chunk_copy["confidence_score"] = float(
                        0.7 * similarity + 0.3 * chunk.get("score", 0.5)
                    )
                    scored_chunks.append(chunk_copy)
                except Exception as e:
                    logger.debug(f"Similarity calculation failed: {e}")
                    chunk_copy = chunk.copy()
                    chunk_copy["confidence_score"] = chunk.get("score", 0.5)
                    scored_chunks.append(chunk_copy)
            
            # Sort by confidence score
            scored_chunks.sort(
                key=lambda x: x.get("confidence_score", 0), 
                reverse=True
            )
            
            logger.info(f"Reranked {len(scored_chunks)} chunks")
            
            return scored_chunks
        
        except Exception as e:
            logger.exception(f"Reranking failed: {e}")
            # Return original chunks with default scores
            for chunk in chunks:
                chunk["confidence_score"] = chunk.get("score", 0.5)
            return chunks
    
    async def _enrich_chunks(
        self,
        chunks: List[Dict],
        query: str
    ) -> List[Dict]:
        """
        Enrich chunks with context and reasoning.
        Explains why each chunk is relevant.
        
        PRODUCTION NOTE: Generates explanations asynchronously for efficiency.
        """
        enriched = []
        
        # Process chunks in batches for efficiency
        batch_size = 5
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            # Process batch concurrently
            tasks = [
                self._enrich_single_chunk(chunk, query)
                for chunk in batch
            ]
            
            try:
                batch_enriched = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in batch_enriched:
                    if isinstance(result, Exception):
                        logger.debug(f"Chunk enrichment failed: {result}")
                        continue
                    if result:
                        enriched.append(result)
            
            except Exception as e:
                logger.warning(f"Batch enrichment failed: {e}")
                # Add chunks without enrichment
                enriched.extend(batch)
        
        return enriched
    
    async def _enrich_single_chunk(
        self,
        chunk: Dict,
        query: str
    ) -> Dict:
        """Enrich a single chunk with relevance reasoning"""
        enriched_chunk = chunk.copy()
        
        try:
            # Add relevance reasoning
            relevance_prompt = f"""Explain briefly why this document chunk is relevant to the query:

Query: "{query}"
Chunk: "{chunk.get('text', '')[:200]}..."

Provide a concise 1-sentence explanation (under 20 words)."""
            
            reasoning = await ai_service.generate_response(
                relevance_prompt, 
                [],
                max_tokens=50,  # Keep it concise
                temperature=0.3  # More deterministic
            )
            
            if isinstance(reasoning, dict):
                enriched_chunk["relevance_reason"] = reasoning.get(
                    "response", 
                    "Relevant match found"
                )
            else:
                enriched_chunk["relevance_reason"] = str(reasoning)[:100]
        
        except Exception as e:
            logger.debug(f"Failed to generate relevance reasoning: {e}")
            enriched_chunk["relevance_reason"] = "Matched knowledge base query"
        
        return enriched_chunk
    
    def _assess_search_quality(self, chunks: List[Dict]) -> str:
        """
        Assess overall search quality based on confidence scores.
        
        Returns:
            "high": Average confidence >= 0.8
            "medium": Average confidence >= 0.6
            "low": Average confidence < 0.6
        """
        if not chunks:
            return "low"
        
        avg_score = np.mean([
            chunk.get("confidence_score", 0) for chunk in chunks
        ])
        
        if avg_score >= 0.8:
            return "high"
        elif avg_score >= 0.6:
            return "medium"
        else:
            return "low"
    
    def _create_no_results_response(
        self,
        query: str,
        expanded_queries: List[str],
        start_time: datetime
    ) -> Dict:
        """
        Create response when no relevant results found.
        
        PRODUCTION NOTE: Returns explicit "NO RESULTS" instead of falling back 
        to generic responses. This prevents hallucination and maintains accuracy.
        """
        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return {
            "chunks": [],
            "total_results": 0,
            "search_quality": "low",
            "query_expanded_terms": expanded_queries,
            "execution_time_ms": execution_time,
            "no_results": True,
            "no_results_message": (
                f"No relevant information found for '{query}' in knowledge base. "
                f"Please try a different search term or check the knowledge base training status."
            ),
            "metadata": {
                "retrieval_stage_results": 0,
                "after_dedup": 0,
                "after_rerank": 0,
                "after_threshold": 0,
                "threshold_used": self.min_relevance_threshold,
            }
        }
    
    def get_health(self) -> Dict:
        """Get service health status"""
        return {
            "service": "rag_search",
            "status": "healthy",
            "config": {
                "min_relevance_threshold": self.min_relevance_threshold,
                "enable_query_expansion": self.enable_query_expansion,
                "enable_reranking": self.enable_reranking,
                "max_chunks_return": self.max_chunks_return,
                "chunk_overlap_penalty": self.chunk_overlap_penalty,
            },
            "timestamp": datetime.utcnow().isoformat()
        }


# Singleton instance
rag_search_service = RAGSearchService()