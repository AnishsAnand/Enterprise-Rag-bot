# tests/test_rag_accuracy.py
import pytest
from app.services.postgres_service import postgres_service
from app.services.rag_search_service import rag_search_service

@pytest.mark.asyncio
async def test_search_accuracy():
    """Test search returns relevant results"""
    
    # Setup test data
    test_docs = [
        {
            "content": "Kubernetes cluster creation requires kubectl version 1.28 or higher",
            "url": "test://k8s-docs",
            "title": "K8s Setup Guide"
        }
    ]
    
    await postgres_service.add_documents(test_docs)
    
    # Test queries
    test_cases = [
        {
            "query": "how to create kubernetes cluster",
            "should_find": "Kubernetes cluster creation",
            "min_confidence": 0.7
        },
        {
            "query": "kubectl version requirement",
            "should_find": "kubectl version 1.28",
            "min_confidence": 0.6
        }
    ]
    
    for case in test_cases:
        results = await rag_search_service.search(
            query=case["query"],
            top_k=5
        )
        
        assert len(results["chunks"]) > 0, f"No results for: {case['query']}"
        
        top_result = results["chunks"][0]
        assert case["should_find"] in top_result["text"], \
            f"Expected text not found in top result"
        
        assert top_result["confidence_score"] >= case["min_confidence"], \
            f"Confidence too low: {top_result['confidence_score']}"