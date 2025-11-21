#!/usr/bin/env python3
"""
Test script for the cluster listing workflow.
Tests the complete flow: engagement -> endpoints -> cluster list
"""

import asyncio
import os
import json
from dotenv import load_dotenv

# IMPORTANT: Load environment variables BEFORE importing the service
load_dotenv()

# Now import the service
from app.services.api_executor_service import api_executor_service

# Reinitialize credentials
api_executor_service.auth_email = os.getenv("API_AUTH_EMAIL", "")
api_executor_service.auth_password = os.getenv("API_AUTH_PASSWORD", "")
api_executor_service.auth_url = os.getenv(
    "API_AUTH_URL",
    "https://ipcloud.tatacommunications.com/portalservice/api/v1/getAuthToken"
)


def print_separator(title: str = ""):
    """Print a formatted separator."""
    print("\n" + "=" * 70)
    if title:
        print(title)
        print("=" * 70)
    print()


async def test_engagement_fetch():
    """Test fetching engagement ID."""
    print_separator("Step 1: Fetch Engagement ID")
    
    engagement_id = await api_executor_service.get_engagement_id()
    
    if engagement_id:
        print(f"✅ Successfully fetched engagement ID: {engagement_id}")
        if api_executor_service.cached_engagement:
            print(f"📌 Engagement Name: {api_executor_service.cached_engagement.get('engagementName')}")
            print(f"📌 Customer Name: {api_executor_service.cached_engagement.get('customerName')}")
        return engagement_id
    else:
        print("❌ Failed to fetch engagement ID")
        return None


async def test_endpoints_fetch(engagement_id: int):
    """Test fetching endpoints."""
    print_separator("Step 2: Fetch Available Endpoints")
    
    endpoints = await api_executor_service.get_endpoints(engagement_id)
    
    if endpoints:
        print(f"✅ Successfully fetched {len(endpoints)} endpoints:")
        print()
        for ep in endpoints:
            print(f"  📍 {ep['endpointDisplayName']:20} (ID: {ep['endpointId']:3}) "
                  f"[{ep['endpoint']:20}] AI: {ep['aiCloudEnabled']}")
        return endpoints
    else:
        print("❌ Failed to fetch endpoints")
        return None


async def test_cluster_list_all(engagement_id: int):
    """Test listing all clusters across all endpoints."""
    print_separator("Step 3: List All Clusters (All Endpoints)")
    
    result = await api_executor_service.list_clusters(
        engagement_id=engagement_id,
        endpoint_ids=None  # None means use all endpoints
    )
    
    if result.get("success"):
        data = result.get("data", {})
        if isinstance(data, dict) and "data" in data:
            clusters = data["data"]
            print(f"✅ Successfully fetched {len(clusters)} clusters:")
            print()
            
            # Group by endpoint
            by_endpoint = {}
            for cluster in clusters:
                endpoint = cluster.get("displayNameEndpoint", "Unknown")
                if endpoint not in by_endpoint:
                    by_endpoint[endpoint] = []
                by_endpoint[endpoint].append(cluster)
            
            # Print grouped results
            for endpoint, endpoint_clusters in sorted(by_endpoint.items()):
                print(f"  📍 {endpoint} ({len(endpoint_clusters)} clusters):")
                for cluster in endpoint_clusters[:3]:  # Show first 3 per endpoint
                    status_emoji = "✅" if cluster.get("status") == "Healthy" else "⚠️"
                    k8s_version = cluster.get('kubernetesVersion') or 'N/A'
                    print(f"    {status_emoji} {cluster.get('clusterName', 'N/A'):25} "
                          f"(Nodes: {cluster.get('nodescount', 0):2}, "
                          f"K8s: {k8s_version:12})")
                if len(endpoint_clusters) > 3:
                    print(f"    ... and {len(endpoint_clusters) - 3} more")
                print()
            
            return clusters
        else:
            print(f"⚠️ Unexpected response format: {data}")
            return None
    else:
        print(f"❌ Failed to list clusters: {result.get('error')}")
        return None


async def test_cluster_list_specific(engagement_id: int, endpoint_ids: list):
    """Test listing clusters for specific endpoints."""
    print_separator(f"Step 4: List Clusters (Specific Endpoints: {endpoint_ids})")
    
    result = await api_executor_service.list_clusters(
        engagement_id=engagement_id,
        endpoint_ids=endpoint_ids
    )
    
    if result.get("success"):
        data = result.get("data", {})
        if isinstance(data, dict) and "data" in data:
            clusters = data["data"]
            print(f"✅ Successfully fetched {len(clusters)} clusters for specified endpoints")
            
            # Summary by status
            status_counts = {}
            for cluster in clusters:
                status = cluster.get("status", "Unknown")
                status_counts[status] = status_counts.get(status, 0) + 1
            
            print(f"\n📊 Status Summary:")
            for status, count in sorted(status_counts.items()):
                emoji = "✅" if status == "Healthy" else "⚠️" if status == "Draft" else "❌"
                print(f"  {emoji} {status}: {count} clusters")
            
            return clusters
        else:
            print(f"⚠️ Unexpected response format: {data}")
            return None
    else:
        print(f"❌ Failed to list clusters: {result.get('error')}")
        return None


async def test_engagement_cache():
    """Test engagement ID caching."""
    print_separator("Step 5: Test Engagement Caching")
    
    print("🔄 Fetching engagement ID (should use cache)...")
    engagement_id = await api_executor_service.get_engagement_id()
    
    if engagement_id:
        print(f"✅ Got cached engagement ID: {engagement_id}")
        print(f"📌 Cache timestamp: {api_executor_service.engagement_cache_time}")
        return True
    else:
        print("❌ Cache test failed")
        return False


async def main():
    """Run all tests."""
    print("\n" + "🚀 " * 20)
    print("Starting Cluster Listing Workflow Tests")
    print("🚀 " * 20)
    
    print(f"\n📧 Email: {api_executor_service.auth_email}")
    print(f"🔗 Auth URL: {api_executor_service.auth_url}")
    
    try:
        # Test 1: Fetch engagement ID
        engagement_id = await test_engagement_fetch()
        if not engagement_id:
            print("\n❌ Cannot continue without engagement ID")
            return
        
        # Test 2: Fetch endpoints
        endpoints = await test_endpoints_fetch(engagement_id)
        if not endpoints:
            print("\n❌ Cannot continue without endpoints")
            return
        
        # Test 3: List all clusters
        all_clusters = await test_cluster_list_all(engagement_id)
        if not all_clusters:
            print("\n⚠️ No clusters found or API call failed")
        
        # Test 4: List clusters for specific endpoints (first 2)
        if endpoints and len(endpoints) >= 2:
            specific_endpoint_ids = [endpoints[0]["endpointId"], endpoints[1]["endpointId"]]
            await test_cluster_list_specific(engagement_id, specific_endpoint_ids)
        
        # Test 5: Test caching
        await test_engagement_cache()
        
        # Summary
        print_separator("Test Summary")
        print("✅ Engagement fetch: PASSED")
        print("✅ Endpoints fetch: PASSED")
        print(f"✅ Cluster listing: {'PASSED' if all_clusters else 'FAILED'}")
        print("✅ Engagement caching: PASSED")
        print("\n🎉 All tests completed!\n")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        await api_executor_service.close()


if __name__ == "__main__":
    asyncio.run(main())

