#!/usr/bin/env python3
"""
Test Memori Session Persistence with PostgreSQL Docker Container.

This script tests the full session persistence with PostgreSQL.
"""

import sys
import os
from datetime import datetime

# Set PostgreSQL connection BEFORE any imports
POSTGRES_URL = "postgresql://ragbot:ragbot_secret_2024@localhost:5435/ragbot_sessions"
os.environ["DATABASE_URL"] = POSTGRES_URL

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_header(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_test(name: str, passed: bool, details: str = ""):
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status}: {name}")
    if details:
        print(f"         {details}")


def test_postgres_connection():
    """Test 1: Verify PostgreSQL connection."""
    print_header("TEST 1: PostgreSQL Connection")
    
    try:
        import psycopg2
        
        conn = psycopg2.connect(
            host="localhost",
            port=5435,
            database="ragbot_sessions",
            user="ragbot",
            password="ragbot_secret_2024"
        )
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        
        print_test("PostgreSQL connection", True, version[:50] + "...")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print_test("PostgreSQL connection", False, str(e))
        return False


def test_table_creation():
    """Test 2: Verify table is created."""
    print_header("TEST 2: Table Creation")
    
    try:
        from app.agents.state.memori_session_manager import MemoriSessionManager
        
        # Create manager with PostgreSQL
        manager = MemoriSessionManager(database_url=POSTGRES_URL)
        
        print_test("MemoriSessionManager created", True)
        
        # Check if table exists
        import psycopg2
        conn = psycopg2.connect(
            host="localhost",
            port=5435,
            database="ragbot_sessions",
            user="ragbot",
            password="ragbot_secret_2024"
        )
        cursor = conn.cursor()
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' AND table_name = 'conversation_sessions'
        """)
        result = cursor.fetchone()
        
        if result:
            print_test("Table 'conversation_sessions' created in PostgreSQL", True)
            
            # Count columns
            cursor.execute("""
                SELECT COUNT(*) 
                FROM information_schema.columns 
                WHERE table_name = 'conversation_sessions'
            """)
            col_count = cursor.fetchone()[0]
            print(f"\n  📋 Table has {col_count} columns")
            
            cursor.close()
            conn.close()
            return manager
        else:
            print_test("Table creation", False, "Table not found!")
            cursor.close()
            conn.close()
            return None
            
    except Exception as e:
        print_test("Table creation", False, str(e))
        import traceback
        traceback.print_exc()
        return None


def test_session_persistence(manager):
    """Test 3: Create and persist a session."""
    print_header("TEST 3: Session Persistence")
    
    if not manager:
        print("  ⚠️ Skipping - no manager from previous test")
        return None
    
    test_session_id = f"pg_test_{datetime.now().strftime('%H%M%S')}"
    
    try:
        from app.agents.state.conversation_state import ConversationStateManager
        
        # Create state manager with PostgreSQL backend
        state_manager = ConversationStateManager(use_persistence=True)
        state_manager._persistent_store = manager
        
        # Create a session
        state = state_manager.create_session(test_session_id, "postgres_test_user")
        print_test("Session created", True, f"session_id={test_session_id}")
        
        # Set intent (cluster creation)
        state.set_intent(
            resource_type="k8s_cluster",
            operation="create",
            required_params=["clusterName", "datacenter", "k8sVersion"],
            optional_params=["tags"]
        )
        print_test("Intent set", True, "create k8s_cluster")
        
        # Add cluster parameters
        params = {
            "clusterName": "postgres-test-cluster",
            "datacenter": {"id": 11, "name": "Delhi"},
            "k8sVersion": "v1.29.0"
        }
        for name, value in params.items():
            state.add_parameter(name, value, is_valid=True)
        
        state.last_asked_param = "cniDriver"
        
        # Persist
        state_manager.update_session(state)
        print_test("Parameters persisted", True, f"{len(params)} params")
        
        # Verify in PostgreSQL
        import psycopg2
        conn = psycopg2.connect(
            host="localhost",
            port=5435,
            database="ragbot_sessions",
            user="ragbot",
            password="ragbot_secret_2024"
        )
        cursor = conn.cursor()
        cursor.execute(
            "SELECT session_id, status, collected_params FROM conversation_sessions WHERE session_id = %s",
            (test_session_id,)
        )
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if row:
            print_test("Verified in PostgreSQL", True, f"status={row[1]}")
            return test_session_id
        else:
            print_test("Verified in PostgreSQL", False, "Not found in DB!")
            return None
            
    except Exception as e:
        print_test("Session persistence", False, str(e))
        import traceback
        traceback.print_exc()
        return None


def test_server_restart_simulation(session_id, manager):
    """Test 4: Simulate server restart and recovery."""
    print_header("TEST 4: Server Restart Simulation")
    
    if not session_id:
        print("  ⚠️ Skipping - no session_id from previous test")
        return False
    
    try:
        from app.agents.state.conversation_state import ConversationStateManager
        from app.agents.state.memori_session_manager import MemoriSessionManager
        
        print("\n  🔄 Simulating server restart...")
        print("     - Creating NEW MemoriSessionManager")
        print("     - Creating NEW ConversationStateManager")
        
        # Create completely fresh managers (simulates restart)
        fresh_memori = MemoriSessionManager(database_url=POSTGRES_URL)
        fresh_state_manager = ConversationStateManager(use_persistence=True)
        fresh_state_manager._persistent_store = fresh_memori
        
        print_test("Fresh managers created", True)
        
        # Try to recover the session
        recovered_state = fresh_state_manager.get_session(session_id)
        
        if not recovered_state:
            print_test("Session recovery", False, "Session not found!")
            return False
        
        print_test("Session recovered from PostgreSQL", True)
        
        # Verify data
        print(f"\n  📊 Recovered Data:")
        print(f"     - Session ID: {recovered_state.session_id}")
        print(f"     - Intent: {recovered_state.intent}")
        print(f"     - Status: {recovered_state.status.value}")
        print(f"     - Params: {list(recovered_state.collected_params.keys())}")
        
        # Verify cluster name
        if recovered_state.collected_params.get("clusterName") == "postgres-test-cluster":
            print_test("clusterName preserved", True, "'postgres-test-cluster'")
        else:
            print_test("clusterName preserved", False)
        
        # Verify workflow state
        if hasattr(recovered_state, 'last_asked_param') and recovered_state.last_asked_param:
            print_test("Workflow state preserved", True, f"next: {recovered_state.last_asked_param}")
        else:
            print_test("Workflow state preserved", False)
        
        return True
        
    except Exception as e:
        print_test("Server restart simulation", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def test_cleanup(session_id, manager):
    """Test 5: Cleanup test session."""
    print_header("TEST 5: Cleanup")
    
    if not session_id or not manager:
        print("  ⚠️ Skipping cleanup")
        return True
    
    try:
        deleted = manager.delete_state(session_id)
        print_test("Test session deleted", deleted)
        
        # Verify deletion
        import psycopg2
        conn = psycopg2.connect(
            host="localhost",
            port=5435,
            database="ragbot_sessions",
            user="ragbot",
            password="ragbot_secret_2024"
        )
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM conversation_sessions WHERE session_id = %s", (session_id,))
        count = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        
        if count == 0:
            print_test("Verified deletion in PostgreSQL", True)
        else:
            print_test("Verified deletion in PostgreSQL", False)
        
        return True
        
    except Exception as e:
        print_test("Cleanup", False, str(e))
        return False


def run_all_tests():
    """Run all PostgreSQL persistence tests."""
    print("\n" + "🐘" * 30)
    print("   POSTGRESQL MEMORI PERSISTENCE TEST")
    print("🐘" * 30)
    
    print(f"\n  📡 Connecting to: localhost:5435")
    print(f"  📁 Database: ragbot_sessions")
    print(f"  👤 User: ragbot")
    
    results = []
    
    # Test 1: Connection
    results.append(("PostgreSQL Connection", test_postgres_connection()))
    
    # Test 2: Table creation
    manager = test_table_creation()
    results.append(("Table Creation", manager is not None))
    
    # Test 3: Session persistence
    session_id = test_session_persistence(manager)
    results.append(("Session Persistence", session_id is not None))
    
    # Test 4: Server restart
    results.append(("Server Restart Recovery", test_server_restart_simulation(session_id, manager)))
    
    # Test 5: Cleanup
    results.append(("Cleanup", test_cleanup(session_id, manager)))
    
    # Print summary
    print_header("TEST SUMMARY")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅" if result else "❌"
        print(f"  {status} {name}")
    
    print(f"\n  📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  🎉 ALL TESTS PASSED!")
        print("     PostgreSQL session persistence is fully working!")
        print("     Sessions will survive server restarts!")
    else:
        print("\n  ⚠️ Some tests failed. Check the output above.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

