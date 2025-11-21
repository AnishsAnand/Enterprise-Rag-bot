#!/usr/bin/env python3
"""
Test script for dynamic token authentication.
Verifies that the token fetch mechanism works correctly.
"""

import asyncio
import os
from dotenv import load_dotenv

# IMPORTANT: Load environment variables BEFORE importing the service
# The service reads env vars during initialization
load_dotenv()

# Now import the service (it will read the loaded env vars)
from app.services.api_executor_service import api_executor_service

# Reinitialize credentials in case service was already instantiated
api_executor_service.auth_email = os.getenv("API_AUTH_EMAIL", "")
api_executor_service.auth_password = os.getenv("API_AUTH_PASSWORD", "")
api_executor_service.auth_url = os.getenv(
    "API_AUTH_URL",
    "https://ipcloud.tatacommunications.com/portalservice/api/v1/getAuthToken"
)

async def test_token_fetch():
    """Test fetching authentication token."""
    print("=" * 70)
    print("Testing Dynamic Token Authentication")
    print("=" * 70)
    
    # Check if credentials are configured
    if not os.getenv("API_AUTH_EMAIL") or not os.getenv("API_AUTH_PASSWORD"):
        print("\n❌ ERROR: API_AUTH_EMAIL and API_AUTH_PASSWORD not configured")
        print("\nPlease add to .env file:")
        print("API_AUTH_EMAIL=izo_cloud_admin@tatacommunications.onmicrosoft.com")
        print("API_AUTH_PASSWORD=Tata@1234")
        return False
    
    print(f"\n📧 Email: {os.getenv('API_AUTH_EMAIL')}")
    print(f"🔗 Auth URL: {api_executor_service.auth_url}")
    print(f"\n🔑 Fetching authentication token...")
    
    # Test token fetch
    token = await api_executor_service._fetch_auth_token()
    
    if token:
        print(f"\n✅ SUCCESS! Token fetched successfully")
        print(f"Token (first 50 chars): {token[:50]}...")
        print(f"Token length: {len(token)} characters")
        
        # Validate it's a JWT token
        if token.count('.') == 2:
            print(f"✅ Token format: Valid JWT (3 parts)")
            parts = token.split('.')
            print(f"   - Header length: {len(parts[0])}")
            print(f"   - Payload length: {len(parts[1])}")
            print(f"   - Signature length: {len(parts[2])}")
        else:
            print(f"⚠️ Token format: Not a standard JWT")
        
        # Test token validation
        print(f"\n🔄 Testing token validation...")
        is_valid = await api_executor_service._ensure_valid_token()
        
        if is_valid:
            print(f"✅ Token validation successful")
            print(f"Token cached until: {api_executor_service.token_expires_at}")
            from datetime import datetime
            time_remaining = (api_executor_service.token_expires_at - datetime.utcnow()).total_seconds() / 60
            print(f"Token will be refreshed in: ~{time_remaining:.1f} minutes")
            return True
        else:
            print(f"❌ Token validation failed")
            return False
    else:
        print(f"\n❌ FAILED to fetch token")
        print(f"\nPlease check:")
        print(f"1. Network connectivity to auth API")
        print(f"2. Credentials are correct")
        print(f"3. Auth URL is correct")
        return False

async def test_token_refresh():
    """Test token refresh mechanism."""
    print("\n" + "=" * 70)
    print("Testing Token Refresh Mechanism")
    print("=" * 70)
    
    # First fetch
    print("\n1️⃣ First token fetch...")
    is_valid1 = await api_executor_service._ensure_valid_token()
    token1 = api_executor_service.auth_token
    
    if not is_valid1:
        print("❌ First fetch failed")
        return False
    
    print(f"✅ First token: {token1[:20] if token1 else 'None'}...")
    
    # Second fetch (should use cached)
    print("\n2️⃣ Second token fetch (should use cached)...")
    is_valid2 = await api_executor_service._ensure_valid_token()
    token2 = api_executor_service.auth_token
    
    if not is_valid2:
        print("❌ Second fetch failed")
        return False
    
    print(f"✅ Second token: {token2[:20] if token2 else 'None'}...")
    
    # Verify same token
    if token1 == token2:
        print(f"✅ Token caching working (same token used)")
        return True
    else:
        print(f"⚠️ Different tokens (caching may not be working)")
        return False

async def main():
    """Run all tests."""
    print("\n🚀 Starting Token Authentication Tests\n")
    
    # Test 1: Token fetch
    test1_passed = await test_token_fetch()
    
    if not test1_passed:
        print("\n❌ Token fetch test failed. Cannot continue.")
        return
    
    # Test 2: Token refresh
    test2_passed = await test_token_refresh()
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"✅ Token Fetch: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"✅ Token Refresh: {'PASSED' if test2_passed else 'FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests PASSED! Token authentication is working correctly.")
    else:
        print("\n❌ Some tests FAILED. Please check configuration.")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️ Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

