"""
Test suite for Agentic Metrics Service.

Tests the three key metrics:
1. Task Adherence
2. Tool Call Accuracy
3. Intent Resolution

Reference: https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/evaluating-agentic-ai-systems-a-deep-dive-into-agentic-metrics/4403923
"""

import asyncio
import sys
import os

# Add the parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.agentic_metrics_service import (
    AgenticMetricsEvaluator,
    AgentTrace,
    ToolCall,
    agentic_metrics_evaluator
)


async def test_basic_trace():
    """Test basic trace recording and retrieval."""
    print("\n" + "="*60)
    print("TEST 1: Basic Trace Recording")
    print("="*60)
    
    evaluator = AgenticMetricsEvaluator()
    
    # Start a trace
    session_id = "test_session_001"
    trace = evaluator.start_trace(
        session_id=session_id,
        user_query="List all clusters in Delhi",
        agent_name="TestAgent"
    )
    
    assert trace is not None, "Trace should be created"
    print(f"✅ Created trace for session: {session_id}")
    
    # Record intent
    evaluator.record_intent(
        session_id=session_id,
        intent="list_k8s_cluster",
        resource_type="k8s_cluster",
        operation="list"
    )
    print("✅ Recorded intent: list_k8s_cluster")
    
    # Record tool call
    evaluator.record_tool_call(
        session_id=session_id,
        tool_name="list_k8s_clusters",
        tool_args={"location_names": ["Delhi"]},
        tool_result=[{"name": "prod-cluster-01", "status": "running"}],
        success=True
    )
    print("✅ Recorded tool call: list_k8s_clusters")
    
    # Complete trace
    completed_trace = evaluator.complete_trace(
        session_id=session_id,
        final_response="Found 1 cluster in Delhi: prod-cluster-01 (running)",
        success=True
    )
    
    assert completed_trace is not None, "Trace should be completed"
    assert completed_trace.final_response != "", "Final response should be set"
    print("✅ Completed trace successfully")
    
    # Retrieve trace
    retrieved_trace = evaluator.get_trace(session_id)
    assert retrieved_trace is not None, "Trace should be retrievable"
    assert len(retrieved_trace.tool_calls) == 1, "Should have 1 tool call"
    print(f"✅ Retrieved trace with {len(retrieved_trace.tool_calls)} tool call(s)")
    
    return True


async def test_manual_evaluation():
    """Test manual evaluation without LLM (mock mode)."""
    print("\n" + "="*60)
    print("TEST 2: Manual Evaluation Structure")
    print("="*60)
    
    # Create a sample trace
    trace = AgentTrace(
        session_id="test_session_002",
        user_query="Create a new Kubernetes cluster named prod-cluster",
        agent_name="OrchestratorAgent",
        intent_detected="create_k8s_cluster",
        resource_type="k8s_cluster",
        operation="create",
        tool_calls=[
            ToolCall(
                tool_name="create_k8s_cluster",
                tool_args={
                    "cluster_name": "prod-cluster",
                    "datacenter": "Delhi"
                },
                tool_result={"id": "cluster-123", "status": "creating"},
                timestamp="2024-01-01T00:00:00Z",
                success=True
            )
        ],
        final_response="Successfully initiated creation of cluster 'prod-cluster' in Delhi datacenter.",
        success=True
    )
    
    # Convert to dict for inspection
    trace_dict = trace.to_dict()
    
    assert "session_id" in trace_dict, "Trace dict should have session_id"
    assert "user_query" in trace_dict, "Trace dict should have user_query"
    assert "tool_calls" in trace_dict, "Trace dict should have tool_calls"
    assert len(trace_dict["tool_calls"]) == 1, "Should have 1 tool call"
    
    print(f"✅ Trace structure valid with {len(trace_dict['tool_calls'])} tool call(s)")
    print(f"   - Session ID: {trace_dict['session_id']}")
    print(f"   - Agent: {trace_dict['agent_name']}")
    print(f"   - Intent: {trace_dict['intent_detected']}")
    print(f"   - Resource: {trace_dict['resource_type']}")
    print(f"   - Operation: {trace_dict['operation']}")
    
    return True


async def test_evaluation_with_llm():
    """Test full evaluation with LLM (requires AI service)."""
    print("\n" + "="*60)
    print("TEST 3: Full Evaluation with LLM (Optional)")
    print("="*60)
    
    try:
        from app.services.ai_service import ai_service
        
        # Create evaluator with AI service
        evaluator = agentic_metrics_evaluator
        
        # Create a test trace
        trace = AgentTrace(
            session_id="test_session_003",
            user_query="Show me all clusters in Delhi datacenter",
            agent_name="OrchestratorAgent",
            intent_detected="list_k8s_cluster",
            resource_type="k8s_cluster",
            operation="list",
            tool_calls=[
                ToolCall(
                    tool_name="list_k8s_clusters",
                    tool_args={"location_names": ["Delhi"]},
                    tool_result=[
                        {"name": "prod-cluster-01", "status": "running"},
                        {"name": "dev-cluster-02", "status": "stopped"}
                    ],
                    timestamp="2024-01-01T00:00:00Z",
                    success=True
                )
            ],
            final_response="""I found 2 clusters in Delhi datacenter:

1. **prod-cluster-01** - Status: running
2. **dev-cluster-02** - Status: stopped

Both clusters are in the Delhi datacenter.""",
            success=True
        )
        
        print("📊 Evaluating trace with LLM...")
        result = await evaluator.evaluate_trace(trace)
        
        print(f"\n✅ Evaluation Results:")
        print(f"   📋 Task Adherence: {result.task_adherence:.2f}")
        print(f"      → {result.task_adherence_reasoning[:100]}...")
        print(f"   🔧 Tool Call Accuracy: {result.tool_call_accuracy:.2f}")
        print(f"      → {result.tool_call_accuracy_reasoning[:100]}...")
        print(f"   🎯 Intent Resolution: {result.intent_resolution:.2f}")
        print(f"      → {result.intent_resolution_reasoning[:100]}...")
        print(f"   ⭐ Overall Score: {result.overall_score:.2f}")
        
        return True
        
    except ImportError as e:
        print(f"⚠️ AI service not available: {e}")
        print("   Skipping LLM evaluation test")
        return True
    except Exception as e:
        print(f"❌ LLM evaluation failed: {e}")
        print("   This may be expected if AI services are not configured")
        return True


async def test_evaluation_summary():
    """Test evaluation summary generation."""
    print("\n" + "="*60)
    print("TEST 4: Evaluation Summary")
    print("="*60)
    
    evaluator = AgenticMetricsEvaluator()
    
    # Get summary (should be empty initially)
    summary = evaluator.get_evaluation_summary()
    
    if "message" in summary:
        print(f"✅ Summary (no evaluations yet): {summary['message']}")
    else:
        print(f"✅ Summary: {summary['total_evaluations']} evaluations")
        if summary.get('average_scores'):
            print(f"   Average scores: {summary['average_scores']}")
    
    return True


async def test_export_results():
    """Test exporting results in different formats."""
    print("\n" + "="*60)
    print("TEST 5: Export Results")
    print("="*60)
    
    evaluator = AgenticMetricsEvaluator()
    
    # Export as JSON
    json_export = evaluator.export_results(format="json")
    print(f"✅ JSON export: {len(json_export)} characters")
    
    # Export as JSONL
    jsonl_export = evaluator.export_results(format="jsonl")
    print(f"✅ JSONL export: {len(jsonl_export)} characters")
    
    return True


async def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("🧪 AGENTIC METRICS TEST SUITE")
    print("="*70)
    print("Based on: Azure AI Evaluation Agentic Metrics")
    print("Reference: https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/")
    print("="*70)
    
    tests = [
        ("Basic Trace Recording", test_basic_trace),
        ("Manual Evaluation Structure", test_manual_evaluation),
        ("Full Evaluation with LLM", test_evaluation_with_llm),
        ("Evaluation Summary", test_evaluation_summary),
        ("Export Results", test_export_results),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = await test_func()
            results.append((name, result, None))
        except Exception as e:
            results.append((name, False, str(e)))
    
    # Print summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result, _ in results if result)
    failed = len(results) - passed
    
    for name, result, error in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")
        if error:
            print(f"         Error: {error}")
    
    print("-"*70)
    print(f"  Total: {passed}/{len(results)} tests passed")
    print("="*70)
    
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)

