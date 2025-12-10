#!/usr/bin/env python3
"""
Manual Interactive Test for Session Persistence.

Run this script in two steps to test persistence:

STEP 1: Run with --create to create a session with parameters
STEP 2: Run with --check to verify the session survived (simulating server restart)

Usage:
    python3 manual_test_persistence.py --create
    python3 manual_test_persistence.py --check
    python3 manual_test_persistence.py --cleanup
    python3 manual_test_persistence.py --show-db
"""

import sys
import os
import argparse
import sqlite3

# Set test database path
TEST_DB = "/home/unixlogin/vayuMaya/Enterprise-Rag-bot/test_sessions.db"
os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DB}"

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Fixed session ID for testing
TEST_SESSION_ID = "manual_test_session"
TEST_USER_ID = "manual_test_user"


def create_session():
    """Step 1: Create a session with cluster creation parameters."""
    print("\n" + "=" * 60)
    print("  STEP 1: Creating Session with Cluster Parameters")
    print("=" * 60)
    
    from app.agents.state.conversation_state import ConversationStateManager
    
    manager = ConversationStateManager(use_persistence=True)
    
    # Check if session already exists
    existing = manager.get_session(TEST_SESSION_ID)
    if existing:
        print(f"\n⚠️  Session '{TEST_SESSION_ID}' already exists!")
        print(f"   Current params: {list(existing.collected_params.keys())}")
        print("\n   Run with --cleanup first, or --check to verify persistence.")
        return
    
    # Create new session
    print(f"\n📝 Creating session: {TEST_SESSION_ID}")
    state = manager.create_session(TEST_SESSION_ID, TEST_USER_ID)
    
    # Set intent for cluster creation
    state.set_intent(
        resource_type="k8s_cluster",
        operation="create",
        required_params=["clusterName", "datacenter", "k8sVersion", "cniDriver", "businessUnit"],
        optional_params=["tags"]
    )
    print("✅ Intent set: create k8s_cluster")
    
    # Simulate collecting parameters (as if user provided them)
    params = {
        "clusterName": "my-production-cluster",
        "datacenter": {"id": 11, "name": "Delhi"},
        "k8sVersion": "v1.28.15",
        "cniDriver": "Calico"
    }
    
    print("\n📦 Adding cluster parameters:")
    for name, value in params.items():
        state.add_parameter(name, value, is_valid=True)
        print(f"   ✅ {name}: {value}")
    
    # Set workflow state (which parameter to ask next)
    state.last_asked_param = "businessUnit"
    
    # Persist to database
    manager.update_session(state)
    print("\n💾 Session persisted to database!")
    
    # Verify in database
    conn = sqlite3.connect(TEST_DB)
    cursor = conn.cursor()
    cursor.execute("SELECT session_id, status, operation FROM conversation_sessions WHERE session_id = ?", (TEST_SESSION_ID,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        print(f"\n✅ Verified in database: status={row[1]}, operation={row[2]}")
    
    print("\n" + "-" * 60)
    print("  NOW: Close this terminal, restart your server, or just wait...")
    print("  THEN: Run 'python3 manual_test_persistence.py --check'")
    print("-" * 60)


def check_session():
    """Step 2: Check if session survived (simulating server restart)."""
    print("\n" + "=" * 60)
    print("  STEP 2: Checking Session Recovery (Simulating Restart)")
    print("=" * 60)
    
    from app.agents.state.conversation_state import ConversationStateManager
    
    # Create a FRESH manager (simulates server restart)
    print("\n🔄 Creating fresh ConversationStateManager (simulating restart)...")
    manager = ConversationStateManager(use_persistence=True)
    
    # Try to get the session
    print(f"\n🔍 Looking for session: {TEST_SESSION_ID}")
    state = manager.get_session(TEST_SESSION_ID)
    
    if not state:
        print("\n❌ Session NOT found! Persistence may not be working.")
        print("   Make sure you ran --create first.")
        return
    
    print("\n🎉 SESSION RECOVERED FROM DATABASE!")
    print("\n📊 Session Details:")
    print(f"   - Session ID: {state.session_id}")
    print(f"   - User ID: {state.user_id}")
    print(f"   - Intent: {state.intent}")
    print(f"   - Operation: {state.operation}")
    print(f"   - Resource Type: {state.resource_type}")
    print(f"   - Status: {state.status.value}")
    
    print("\n📦 Recovered Parameters:")
    for name, value in state.collected_params.items():
        print(f"   ✅ {name}: {value}")
    
    if hasattr(state, 'last_asked_param') and state.last_asked_param:
        print(f"\n⏭️  Next step in workflow: Ask for '{state.last_asked_param}'")
    
    print("\n" + "-" * 60)
    print("  ✅ SUCCESS! Parameters survived the 'restart'!")
    print("  Run --cleanup when done testing.")
    print("-" * 60)


def cleanup_session():
    """Cleanup: Delete the test session."""
    print("\n" + "=" * 60)
    print("  CLEANUP: Removing Test Session")
    print("=" * 60)
    
    from app.agents.state.conversation_state import ConversationStateManager
    
    manager = ConversationStateManager(use_persistence=True)
    
    if manager.delete_session(TEST_SESSION_ID):
        print(f"\n✅ Session '{TEST_SESSION_ID}' deleted successfully!")
    else:
        print(f"\n⚠️  Session '{TEST_SESSION_ID}' not found (already deleted?).")


def show_database():
    """Show all sessions in the database."""
    print("\n" + "=" * 60)
    print("  DATABASE CONTENTS")
    print("=" * 60)
    
    if not os.path.exists(TEST_DB):
        print(f"\n❌ Database not found: {TEST_DB}")
        print("   Run --create first to initialize the database.")
        return
    
    conn = sqlite3.connect(TEST_DB)
    cursor = conn.cursor()
    
    # Count sessions
    cursor.execute("SELECT COUNT(*) FROM conversation_sessions")
    count = cursor.fetchone()[0]
    print(f"\n📊 Total sessions in database: {count}")
    
    if count > 0:
        # Show all sessions
        cursor.execute("""
            SELECT session_id, user_id, status, operation, resource_type, 
                   collected_params, created_at, updated_at
            FROM conversation_sessions
            ORDER BY updated_at DESC
        """)
        rows = cursor.fetchall()
        
        print("\n📋 Sessions:")
        for row in rows:
            print(f"\n   Session: {row[0]}")
            print(f"   User: {row[1]}")
            print(f"   Status: {row[2]}")
            print(f"   Operation: {row[3]} {row[4]}")
            print(f"   Created: {row[6]}")
            print(f"   Updated: {row[7]}")
            
            import json
            params = json.loads(row[5]) if row[5] else {}
            if params:
                print(f"   Params: {list(params.keys())}")
    
    conn.close()
    
    # Show file size
    size = os.path.getsize(TEST_DB)
    print(f"\n📁 Database file: {TEST_DB} ({size} bytes)")


def main():
    parser = argparse.ArgumentParser(description="Manual test for session persistence")
    parser.add_argument("--create", action="store_true", help="Create a test session with parameters")
    parser.add_argument("--check", action="store_true", help="Check if session survived (run after --create)")
    parser.add_argument("--cleanup", action="store_true", help="Delete the test session")
    parser.add_argument("--show-db", action="store_true", help="Show all sessions in database")
    
    args = parser.parse_args()
    
    if args.create:
        create_session()
    elif args.check:
        check_session()
    elif args.cleanup:
        cleanup_session()
    elif args.show_db:
        show_database()
    else:
        print(__doc__)
        print("\nQuick Test:")
        print("  1. python3 manual_test_persistence.py --create")
        print("  2. python3 manual_test_persistence.py --check")
        print("  3. python3 manual_test_persistence.py --cleanup")


if __name__ == "__main__":
    main()


