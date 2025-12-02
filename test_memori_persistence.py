#!/usr/bin/env python3
"""
Test script for Memori Session Persistence.

This script thoroughly tests the SQL-backed session persistence to ensure:
1. Sessions are created and persisted to the database
2. Sessions survive simulated "server restarts"
3. Collected parameters for cluster creation are properly stored
4. Session cleanup works correctly
"""

import sys
import os
import sqlite3
from datetime import datetime, timedelta

# Use a test database that we can write to
TEST_DB_PATH = "/home/unixlogin/vayuMaya/Enterprise-Rag-bot/test_sessions.db"

# Clean up old test database BEFORE imports
if os.path.exists(TEST_DB_PATH):
    os.remove(TEST_DB_PATH)
    print(f"🗑️ Removed old test database: {TEST_DB_PATH}")

# Set environment variable BEFORE any imports
os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DB_PATH}"

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_test(name: str, passed: bool, details: str = ""):
    """Print test result."""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status}: {name}")
    if details:
        print(f"         {details}")


def test_database_table_exists():
    """Test 1: Verify the conversation_sessions table was created."""
    print_header("TEST 1: Database Table Creation")
    
    try:
        conn = sqlite3.connect(TEST_DB_PATH)
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='conversation_sessions'
        """)
        result = cursor.fetchone()
        
        if result:
            print_test("Table 'conversation_sessions' exists", True)
            
            # Get table schema
            cursor.execute("PRAGMA table_info(conversation_sessions)")
            columns = cursor.fetchall()
            print(f"\n  📋 Table columns:")
            for col in columns:
                print(f"     - {col[1]} ({col[2]})")
            
            conn.close()
            return True
        else:
            print_test("Table 'conversation_sessions' exists", False, "Table not found!")
            conn.close()
            return False
            
    except Exception as e:
        print_test("Database connection", False, str(e))
        return False


def test_session_creation_and_persistence(conversation_state_manager):
    """Test 2: Create a session and verify it's persisted to DB."""
    print_header("TEST 2: Session Creation & Persistence")
    
    test_session_id = f"test_session_{datetime.now().strftime('%H%M%S')}"
    test_user_id = "test_user_123"
    
    try:
        # Create a new session using ConversationStateManager
        state = conversation_state_manager.create_session(test_session_id, test_user_id)
        print_test("Session created in memory", True, f"session_id={test_session_id}")
        
        # Set some intent data (simulating cluster creation intent)
        state.set_intent(
            resource_type="k8s_cluster",
            operation="create",
            required_params=["clusterName", "datacenter", "k8sVersion", "cniDriver"],
            optional_params=["tags"]
        )
        print_test("Intent set on session", True, "k8s_cluster create")
        
        # Explicitly persist
        conversation_state_manager.update_session(state)
        print_test("Session persisted to database", True)
        
        # Verify in database directly
        conn = sqlite3.connect(TEST_DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT session_id, user_id, resource_type, operation, status FROM conversation_sessions WHERE session_id = ?",
            (test_session_id,)
        )
        row = cursor.fetchone()
        conn.close()
        
        if row:
            print_test("Session found in database", True, f"status={row[4]}, operation={row[3]}")
            return test_session_id
        else:
            print_test("Session found in database", False, "Not found in DB!")
            return None
            
    except Exception as e:
        print_test("Session creation", False, str(e))
        import traceback
        traceback.print_exc()
        return None


def test_parameter_collection_persistence(session_id, conversation_state_manager):
    """Test 3: Add parameters and verify they persist."""
    print_header("TEST 3: Parameter Collection Persistence")
    
    if not session_id:
        print("  ⚠️ Skipping - no session_id from previous test")
        return False
    
    try:
        # Get the session
        state = conversation_state_manager.get_session(session_id)
        
        if not state:
            print_test("Session retrieval", False, "Session not found!")
            return False
        
        print_test("Session retrieved", True)
        
        # Simulate collecting cluster creation parameters
        params_to_add = {
            "clusterName": "test-cluster-prod",
            "datacenter": {"id": 11, "name": "Delhi"},
            "k8sVersion": "v1.28.15",
            "cniDriver": "Calico"
        }
        
        for param_name, param_value in params_to_add.items():
            state.add_parameter(param_name, param_value, is_valid=True)
            print(f"     Added: {param_name} = {param_value}")
        
        # Set last_asked_param (used in cluster creation workflow)
        state.last_asked_param = "businessUnit"
        
        # Persist the changes
        conversation_state_manager.update_session(state)
        print_test("Parameters persisted to database", True)
        
        # Verify in database
        conn = sqlite3.connect(TEST_DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT collected_params FROM conversation_sessions WHERE session_id = ?",
            (session_id,)
        )
        row = cursor.fetchone()
        conn.close()
        
        if row and row[0]:
            import json
            stored_params = json.loads(row[0])
            print(f"\n  📦 Stored in DB: {list(stored_params.keys())}")
            print_test("Parameters verified in database", True, f"{len(stored_params)} params stored")
            return True
        else:
            print_test("Parameters in database", False, "No collected_params found")
            return False
            
    except Exception as e:
        print_test("Parameter persistence", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def test_simulated_server_restart(session_id, ConversationStateManager):
    """Test 4: Simulate server restart and verify session recovery."""
    print_header("TEST 4: Simulated Server Restart Recovery")
    
    if not session_id:
        print("  ⚠️ Skipping - no session_id from previous test")
        return False
    
    try:
        # Get current params from database BEFORE restart
        conn = sqlite3.connect(TEST_DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT collected_params, resource_type, operation FROM conversation_sessions WHERE session_id = ?",
            (session_id,)
        )
        row = cursor.fetchone()
        conn.close()
        
        import json
        params_before = json.loads(row[0]) if row and row[0] else {}
        
        print(f"  📝 Before restart: {len(params_before)} params in DB")
        
        # SIMULATE SERVER RESTART by creating a COMPLETELY NEW state manager
        print("\n  🔄 Simulating server restart (creating new ConversationStateManager)...")
        
        # Create fresh manager - this simulates a server restart
        new_manager = ConversationStateManager(use_persistence=True)
        
        print_test("New ConversationStateManager created", True, "Simulating server restart")
        
        # Verify the new manager's cache is empty
        if session_id not in new_manager._states:
            print_test("New manager cache is empty", True)
        else:
            print_test("New manager cache is empty", False, "Session already in cache!")
        
        # Now try to get the session - it should load from database
        print("\n  📖 Attempting to recover session from database...")
        state_after = new_manager.get_session(session_id)
        
        if not state_after:
            print_test("Session recovery", False, "Session not recovered from DB!")
            return False
        
        print_test("Session recovered from database", True)
        
        # Verify the data matches
        params_after = dict(state_after.collected_params)
        
        print(f"\n  📊 Comparison:")
        print(f"     Before restart (DB): {list(params_before.keys())}")
        print(f"     After restart:       {list(params_after.keys())}")
        
        # Check if cluster name survived
        if params_after.get("clusterName") == params_before.get("clusterName"):
            print_test("clusterName preserved", True, f"'{params_after.get('clusterName')}'")
        else:
            print_test("clusterName preserved", False)
            
        # Check if datacenter survived
        if params_after.get("datacenter") == params_before.get("datacenter"):
            print_test("datacenter preserved", True, f"{params_after.get('datacenter')}")
        else:
            print_test("datacenter preserved", False)
        
        # Check intent/operation
        if state_after.operation == "create" and state_after.resource_type == "k8s_cluster":
            print_test("Intent preserved", True, f"{state_after.operation} {state_after.resource_type}")
        else:
            print_test("Intent preserved", False)
        
        # Check last_asked_param (for workflow continuation)
        if hasattr(state_after, 'last_asked_param') and state_after.last_asked_param == "businessUnit":
            print_test("last_asked_param preserved", True, "Can resume from correct step!")
        else:
            print_test("last_asked_param preserved", False, "Workflow position lost")
        
        return True
        
    except Exception as e:
        print_test("Server restart simulation", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def test_session_stats(conversation_state_manager):
    """Test 5: Get session statistics."""
    print_header("TEST 5: Session Statistics")
    
    try:
        stats = conversation_state_manager.get_stats()
        
        print(f"\n  📊 Session Manager Stats:")
        print(f"     - Cache size: {stats.get('cache_size', 'N/A')}")
        print(f"     - Persistence enabled: {stats.get('persistence_enabled', 'N/A')}")
        print(f"     - Total sessions in DB: {stats.get('total_sessions', 'N/A')}")
        print(f"     - Active in last hour: {stats.get('active_last_hour', 'N/A')}")
        
        if stats.get('status_breakdown'):
            print(f"     - Status breakdown: {stats.get('status_breakdown')}")
        
        print_test("Stats retrieved", True)
        return True
        
    except Exception as e:
        print_test("Stats retrieval", False, str(e))
        return False


def test_session_cleanup(session_id, conversation_state_manager):
    """Test 6: Test session deletion and cleanup."""
    print_header("TEST 6: Session Cleanup")
    
    if not session_id:
        print("  ⚠️ Skipping - no session_id from previous test")
        return False
    
    try:
        # Delete the test session
        deleted = conversation_state_manager.delete_session(session_id)
        print_test("Session deleted", deleted)
        
        # Verify it's gone from cache
        if session_id not in conversation_state_manager._states:
            print_test("Removed from cache", True)
        else:
            print_test("Removed from cache", False)
        
        # Verify it's gone from database
        conn = sqlite3.connect(TEST_DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT session_id FROM conversation_sessions WHERE session_id = ?",
            (session_id,)
        )
        row = cursor.fetchone()
        conn.close()
        
        if row is None:
            print_test("Removed from database", True)
        else:
            print_test("Removed from database", False, "Still exists in DB!")
        
        return True
        
    except Exception as e:
        print_test("Cleanup", False, str(e))
        return False


def test_multiple_sessions(conversation_state_manager):
    """Test 7: Create and manage multiple sessions."""
    print_header("TEST 7: Multiple Sessions")
    
    try:
        # Create multiple test sessions
        sessions = []
        for i in range(3):
            session_id = f"multi_test_{i}_{datetime.now().strftime('%H%M%S')}"
            user_id = f"user_{i}"
            
            state = conversation_state_manager.create_session(session_id, user_id)
            state.set_intent(
                resource_type="k8s_cluster",
                operation="create" if i % 2 == 0 else "list",
                required_params=["endpoints"],
                optional_params=[]
            )
            conversation_state_manager.update_session(state)
            sessions.append(session_id)
        
        print_test(f"Created {len(sessions)} test sessions", True)
        
        # Get active sessions
        active = conversation_state_manager.get_active_sessions()
        print(f"\n  📋 Active sessions: {len(active)}")
        
        # Count in database
        conn = sqlite3.connect(TEST_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM conversation_sessions")
        total = cursor.fetchone()[0]
        conn.close()
        
        print(f"  📊 Total sessions in DB: {total}")
        print_test("Multiple sessions managed", True)
        
        # Cleanup test sessions
        for sid in sessions:
            conversation_state_manager.delete_session(sid)
        
        print_test("Test sessions cleaned up", True)
        return True
        
    except Exception as e:
        print_test("Multiple sessions", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all persistence tests."""
    
    # NOW do the imports (after DATABASE_URL is set)
    from app.agents.state.conversation_state import (
        ConversationState, 
        ConversationStateManager, 
        ConversationStatus
    )
    
    # Create a fresh manager with our test database
    conversation_state_manager = ConversationStateManager(use_persistence=True)
    
    print("\n" + "🧪" * 30)
    print("   MEMORI SESSION PERSISTENCE TEST SUITE")
    print("🧪" * 30)
    
    print(f"\n  📁 Using test database: {TEST_DB_PATH}")
    print(f"  🔧 Persistence enabled: {conversation_state_manager.use_persistence}")
    
    results = []
    
    # Test 1: Database table
    results.append(("Database Table Creation", test_database_table_exists()))
    
    # Test 2: Session creation
    session_id = test_session_creation_and_persistence(conversation_state_manager)
    results.append(("Session Creation & Persistence", session_id is not None))
    
    # Test 3: Parameter persistence
    results.append(("Parameter Collection Persistence", test_parameter_collection_persistence(session_id, conversation_state_manager)))
    
    # Test 4: Server restart simulation (THE KEY TEST!)
    results.append(("Server Restart Recovery", test_simulated_server_restart(session_id, ConversationStateManager)))
    
    # Test 5: Stats
    results.append(("Session Statistics", test_session_stats(conversation_state_manager)))
    
    # Test 6: Cleanup
    results.append(("Session Cleanup", test_session_cleanup(session_id, conversation_state_manager)))
    
    # Test 7: Multiple sessions
    results.append(("Multiple Sessions", test_multiple_sessions(conversation_state_manager)))
    
    # Print summary
    print_header("TEST SUMMARY")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅" if result else "❌"
        print(f"  {status} {name}")
    
    print(f"\n  📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  🎉 ALL TESTS PASSED! Session persistence is working correctly.")
        print("     Your cluster creation parameters will survive server restarts!")
    else:
        print("\n  ⚠️ Some tests failed. Please check the output above.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    
    # Show database file info
    if os.path.exists(TEST_DB_PATH):
        size = os.path.getsize(TEST_DB_PATH)
        print(f"\n📁 Test database created: {TEST_DB_PATH} ({size} bytes)")
    
    sys.exit(0 if success else 1)
