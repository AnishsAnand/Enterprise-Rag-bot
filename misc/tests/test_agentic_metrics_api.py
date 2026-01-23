#!/usr/bin/env python3
"""
Agentic Metrics API Test Script

This script tests all the agentic metrics API endpoints by making actual HTTP requests.
It requires the backend server to be running on localhost:8001.

Usage:
    python tests/test_agentic_metrics_api.py

API Base URL: http://localhost:8001/api/v1/metrics
"""

import requests
import json
import sys
import time
from datetime import datetime
from typing import Dict, Any, Optional

# Configuration
BASE_URL = "http://localhost:8001"
API_PREFIX = "/api/v1/metrics"


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    END = "\033[0m"


def print_header(title: str):
    """Print a formatted header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{title}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")


def print_test(name: str):
    """Print test name."""
    print(f"\n{Colors.YELLOW}📍 {name}{Colors.END}")
    print("-" * 50)


def print_success(message: str):
    """Print success message."""
    print(f"{Colors.GREEN}✅ {message}{Colors.END}")


def print_error(message: str):
    """Print error message."""
    print(f"{Colors.RED}❌ {message}{Colors.END}")


def print_info(message: str):
    """Print info message."""
    print(f"{Colors.CYAN}ℹ️  {message}{Colors.END}")


def print_json(data: Any, indent: int = 2):
    """Print formatted JSON."""
    print(json.dumps(data, indent=indent, default=str))


def make_request(
    method: str,
    endpoint: str,
    data: Optional[Dict] = None,
    params: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Make an HTTP request to the API.
    
    Returns:
        Dict with 'success', 'status_code', 'data', and 'error' keys
    """
    url = f"{BASE_URL}{API_PREFIX}{endpoint}"
    
    try:
        if method.upper() == "GET":
            response = requests.get(url, params=params, timeout=30)
        elif method.upper() == "POST":
            response = requests.post(url, json=data, timeout=60)
        elif method.upper() == "DELETE":
            response = requests.delete(url, timeout=30)
        else:
            return {"success": False, "error": f"Unknown method: {method}"}
        
        try:
            response_data = response.json()
        except json.JSONDecodeError:
            response_data = {"raw": response.text}
        
        return {
            "success": response.status_code < 400,
            "status_code": response.status_code,
            "data": response_data,
            "error": None if response.status_code < 400 else response_data
        }
        
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "status_code": 0,
            "data": None,
            "error": f"Connection failed. Is the server running at {BASE_URL}?"
        }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "status_code": 0,
            "data": None,
            "error": "Request timed out"
        }
    except Exception as e:
        return {
            "success": False,
            "status_code": 0,
            "data": None,
            "error": str(e)
        }


def test_health_check() -> bool:
    """Test: GET /health - Check service health."""
    print_test("TEST 1: Health Check (GET /health)")
    
    result = make_request("GET", "/health")
    
    if result["success"]:
        print_success(f"Health check passed (HTTP {result['status_code']})")
        print_info("Response:")
        print_json(result["data"])
        return True
    else:
        print_error(f"Health check failed: {result['error']}")
        return False


def test_get_summary() -> bool:
    """Test: GET /summary - Get evaluation summary."""
    print_test("TEST 2: Get Evaluation Summary (GET /summary)")
    
    result = make_request("GET", "/summary")
    
    if result["success"]:
        print_success(f"Summary retrieved (HTTP {result['status_code']})")
        print_info("Response:")
        print_json(result["data"])
        return True
    else:
        print_error(f"Failed to get summary: {result['error']}")
        return False


def test_manual_evaluation() -> bool:
    """Test: POST /evaluate/manual - Evaluate an interaction manually."""
    print_test("TEST 3: Manual Evaluation (POST /evaluate/manual)")
    
    # Test data - a sample agent interaction
    test_data = {
        "user_query": "List all Kubernetes clusters in Delhi datacenter",
        "agent_response": "I found 2 Kubernetes clusters in Delhi datacenter:\n\n1. **prod-cluster-01**\n   - Status: Running\n   - Version: 1.28.0\n   - Nodes: 5\n\n2. **dev-cluster-02**\n   - Status: Running\n   - Version: 1.27.5\n   - Nodes: 3",
        "tool_calls": [
            {
                "tool_name": "list_k8s_clusters",
                "tool_args": {"datacenter": "Delhi"},
                "tool_result": [
                    {"name": "prod-cluster-01", "status": "running", "version": "1.28.0", "nodes": 5},
                    {"name": "dev-cluster-02", "status": "running", "version": "1.27.5", "nodes": 3}
                ],
                "success": True
            }
        ],
        "detected_intent": "list_k8s_cluster",
        "resource_type": "k8s_cluster",
        "operation": "list"
    }
    
    print_info("Request payload:")
    print_json(test_data)
    print()
    
    print_info("⏳ Sending request (this may take 10-30 seconds for LLM evaluation)...")
    start_time = time.time()
    
    result = make_request("POST", "/evaluate/manual", data=test_data)
    
    elapsed = time.time() - start_time
    
    if result["success"]:
        print_success(f"Manual evaluation completed in {elapsed:.2f}s (HTTP {result['status_code']})")
        print_info("Evaluation Results:")
        print_json(result["data"])
        
        # Extract and display scores
        data = result["data"]
        if "overall_score" in data:
            print()
            print_info("📊 Score Summary:")
            print(f"   Task Adherence:     {data.get('task_adherence', {}).get('score', 'N/A'):.2f}")
            print(f"   Tool Call Accuracy: {data.get('tool_call_accuracy', {}).get('score', 'N/A'):.2f}")
            print(f"   Intent Resolution:  {data.get('intent_resolution', {}).get('score', 'N/A'):.2f}")
            print(f"   Overall Score:      {data.get('overall_score', 'N/A'):.2f}")
        
        return True
    else:
        print_error(f"Manual evaluation failed: {result['error']}")
        if result["status_code"] == 500:
            print_info("This might be due to LLM service issues or timeout")
        return False


def test_get_trace_not_found() -> bool:
    """Test: GET /trace/{session_id} - Get trace for non-existent session."""
    print_test("TEST 4: Get Non-existent Trace (GET /trace/{session_id})")
    
    fake_session_id = "nonexistent_session_12345"
    result = make_request("GET", f"/trace/{fake_session_id}")
    
    if result["status_code"] == 404:
        print_success(f"Correctly returned 404 for non-existent session")
        print_info(f"Response: {result['data']}")
        return True
    elif result["success"]:
        print_error("Expected 404 but got success - trace should not exist")
        return False
    else:
        print_error(f"Unexpected error: {result['error']}")
        return False


def test_evaluate_session_not_found() -> bool:
    """Test: POST /evaluate/session - Evaluate non-existent session."""
    print_test("TEST 5: Evaluate Non-existent Session (POST /evaluate/session)")
    
    test_data = {"session_id": "nonexistent_session_67890"}
    result = make_request("POST", "/evaluate/session", data=test_data)
    
    if result["status_code"] == 404:
        print_success(f"Correctly returned 404 for non-existent session")
        print_info(f"Response: {result['data']}")
        return True
    elif result["success"]:
        print_error("Expected 404 but got success")
        return False
    else:
        print_info(f"Got error (expected): {result['error']}")
        return True


def test_batch_evaluate() -> bool:
    """Test: POST /evaluate/batch - Batch evaluate sessions."""
    print_test("TEST 6: Batch Evaluate Sessions (POST /evaluate/batch)")
    
    test_data = {
        "session_ids": [
            "session_001",
            "session_002",
            "nonexistent_session"
        ]
    }
    
    print_info("Request payload:")
    print_json(test_data)
    print()
    
    result = make_request("POST", "/evaluate/batch", data=test_data)
    
    if result["success"]:
        print_success(f"Batch evaluation completed (HTTP {result['status_code']})")
        print_info("Response:")
        print_json(result["data"])
        return True
    else:
        print_error(f"Batch evaluation failed: {result['error']}")
        return False


def test_export_json() -> bool:
    """Test: GET /export?format=json - Export results as JSON."""
    print_test("TEST 7: Export Results as JSON (GET /export?format=json)")
    
    result = make_request("GET", "/export", params={"format": "json"})
    
    if result["success"]:
        print_success(f"Export successful (HTTP {result['status_code']})")
        data = result["data"]
        print_info(f"Format: {data.get('format', 'unknown')}")
        print_info(f"Data length: {len(data.get('data', ''))} characters")
        return True
    else:
        print_error(f"Export failed: {result['error']}")
        return False


def test_export_jsonl() -> bool:
    """Test: GET /export?format=jsonl - Export results as JSONL."""
    print_test("TEST 8: Export Results as JSONL (GET /export?format=jsonl)")
    
    result = make_request("GET", "/export", params={"format": "jsonl"})
    
    if result["success"]:
        print_success(f"Export successful (HTTP {result['status_code']})")
        data = result["data"]
        print_info(f"Format: {data.get('format', 'unknown')}")
        print_info(f"Data length: {len(data.get('data', ''))} characters")
        return True
    else:
        print_error(f"Export failed: {result['error']}")
        return False


def test_get_history() -> bool:
    """Test: GET /history - Get evaluation history from database."""
    print_test("TEST 9: Get Evaluation History (GET /history)")
    
    result = make_request("GET", "/history", params={"limit": 10, "offset": 0})
    
    if result["success"]:
        print_success(f"History retrieved (HTTP {result['status_code']})")
        data = result["data"]
        print_info(f"Records returned: {data.get('count', 0)}")
        print_info(f"Limit: {data.get('limit', 'N/A')}, Offset: {data.get('offset', 'N/A')}")
        if data.get("evaluations"):
            print_info("Sample evaluation:")
            print_json(data["evaluations"][0] if data["evaluations"] else {})
        return True
    else:
        print_error(f"Failed to get history: {result['error']}")
        return False


def test_clear_results() -> bool:
    """Test: DELETE /clear - Clear all results (optional, commented out by default)."""
    print_test("TEST 10: Clear Results (DELETE /clear) - SKIPPED")
    
    print_info("⚠️  This test is skipped by default to preserve data")
    print_info("Uncomment the code below to test the clear endpoint")
    
    # Uncomment to test:
    # result = make_request("DELETE", "/clear")
    # if result["success"]:
    #     print_success(f"Results cleared (HTTP {result['status_code']})")
    #     print_json(result["data"])
    #     return True
    # else:
    #     print_error(f"Failed to clear results: {result['error']}")
    #     return False
    
    return True  # Skip


def run_all_tests():
    """Run all API tests."""
    print_header("🧪 AGENTIC METRICS API TEST SUITE")
    print(f"Base URL: {BASE_URL}{API_PREFIX}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Check if server is running
    print_test("PREREQUISITE: Server Connection Check")
    health_result = make_request("GET", "/health")
    if not health_result["success"] and health_result["status_code"] == 0:
        print_error(f"Cannot connect to server at {BASE_URL}")
        print_info("Please ensure the backend is running:")
        print_info("  uvicorn app.user_main:app --host 0.0.0.0 --port 8001 --reload")
        return False
    print_success("Server is reachable")
    
    # Run tests
    tests = [
        ("Health Check", test_health_check),
        ("Get Summary", test_get_summary),
        ("Manual Evaluation", test_manual_evaluation),
        ("Get Non-existent Trace", test_get_trace_not_found),
        ("Evaluate Non-existent Session", test_evaluate_session_not_found),
        ("Batch Evaluate", test_batch_evaluate),
        ("Export JSON", test_export_json),
        ("Export JSONL", test_export_jsonl),
        ("Get History", test_get_history),
        ("Clear Results", test_clear_results),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print_error(f"Test '{name}' raised exception: {e}")
            results.append((name, False))
    
    # Print summary
    print_header("📊 TEST RESULTS SUMMARY")
    
    passed = sum(1 for _, p in results if p)
    failed = len(results) - passed
    
    for name, result in results:
        status = f"{Colors.GREEN}PASS{Colors.END}" if result else f"{Colors.RED}FAIL{Colors.END}"
        print(f"  {status} - {name}")
    
    print()
    print(f"Total: {len(results)} | Passed: {Colors.GREEN}{passed}{Colors.END} | Failed: {Colors.RED}{failed}{Colors.END}")
    
    if failed == 0:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 All tests passed!{Colors.END}")
    else:
        print(f"\n{Colors.YELLOW}⚠️  Some tests failed. Check the output above for details.{Colors.END}")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

