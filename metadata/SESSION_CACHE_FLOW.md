# Session Cache Flow Diagram

## Request Flow with Session Caching

### First Request (Cache Miss)

```
User Request: "List my VMs"
        ↓
┌───────────────────────────────────────────────────────────┐
│ 1. get_engagement_id()                                    │
│    ├─ Check session cache → MISS                          │
│    ├─ Call API: /paasservice/paas/engagements            │
│    └─ Cache result: paas_engagement_id = 1234            │
└───────────────────────────────────────────────────────────┘
        ↓
┌───────────────────────────────────────────────────────────┐
│ 2. get_ipc_engagement_id()                                │
│    ├─ Check session cache → MISS                          │
│    ├─ Call API: /paasservice/paas/getIpcEngFromPaasEng   │
│    └─ Cache result: ipc_engagement_id = 1602             │
└───────────────────────────────────────────────────────────┘
        ↓
┌───────────────────────────────────────────────────────────┐
│ 3. list_vms(ipc_engagement_id=1602)                      │
│    ├─ Call API: /portalservice/instances/vmlist/1602     │
│    └─ Return VM list                                      │
└───────────────────────────────────────────────────────────┘

Total API Calls: 3
Total Time: ~800-1200ms
```

### Second Request (Cache Hit)

```
User Request: "List my clusters"
        ↓
┌───────────────────────────────────────────────────────────┐
│ 1. get_engagement_id()                                    │
│    ├─ Check session cache → HIT ✅                        │
│    └─ Return cached: paas_engagement_id = 1234           │
└───────────────────────────────────────────────────────────┘
        ↓
┌───────────────────────────────────────────────────────────┐
│ 2. get_ipc_engagement_id()                                │
│    ├─ Check session cache → HIT ✅                        │
│    └─ Return cached: ipc_engagement_id = 1602            │
└───────────────────────────────────────────────────────────┘
        ↓
┌───────────────────────────────────────────────────────────┐
│ 3. list_clusters(engagement_id=1234)                     │
│    ├─ Call API: /paasservice/paas/clusterlist/stream     │
│    └─ Return cluster list                                 │
└───────────────────────────────────────────────────────────┘

Total API Calls: 1 (saved 2 calls!)
Total Time: ~300-500ms (60% faster!)
```

## Session Storage Structure

```
┌─────────────────────────────────────────────────────────────┐
│                    User Sessions Cache                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  user1@example.com:                                         │
│  ├─ cached_at: 2025-12-18 10:30:00                        │
│  ├─ paas_engagement_id: 1234                               │
│  ├─ ipc_engagement_id: 1602                                │
│  ├─ engagement_data: {...}                                 │
│  ├─ endpoints: [...]                                        │
│  ├─ business_units: {...}                                   │
│  └─ environments_list: [...]                                │
│                                                             │
│  user2@example.com:                                         │
│  ├─ cached_at: 2025-12-18 11:15:00                        │
│  ├─ paas_engagement_id: 5678                               │
│  ├─ ipc_engagement_id: 2001                                │
│  ├─ engagement_data: {...}                                 │
│  ├─ endpoints: [...]                                        │
│  ├─ business_units: {...}                                   │
│  └─ environments_list: [...]                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Cache Lifecycle

```
┌─────────────────────────────────────────────────────────────┐
│                    Cache Lifecycle                          │
└─────────────────────────────────────────────────────────────┘

Time: T+0 (First Request)
├─ Cache: EMPTY
├─ Action: Fetch from API
└─ Result: Data cached with timestamp

Time: T+1 hour (Second Request)
├─ Cache: VALID (< 24 hours old)
├─ Action: Return from cache
└─ Result: Fast response (no API call)

Time: T+12 hours (Multiple Requests)
├─ Cache: VALID (< 24 hours old)
├─ Action: Return from cache
└─ Result: Fast response (no API call)

Time: T+24 hours (Cache Expired)
├─ Cache: EXPIRED (≥ 24 hours old)
├─ Action: Fetch from API
└─ Result: Data re-cached with new timestamp

Time: T+24.5 hours (Force Refresh)
├─ Cache: VALID (< 24 hours since last refresh)
├─ Action: force_refresh=True → Fetch from API
└─ Result: Data re-cached (manual refresh)
```

## Multi-User Isolation

```
┌─────────────────────────────────────────────────────────────┐
│                    User Isolation                           │
└─────────────────────────────────────────────────────────────┘

User A Request                    User B Request
     ↓                                  ↓
┌─────────────┐                  ┌─────────────┐
│ Check Cache │                  │ Check Cache │
│  (User A)   │                  │  (User B)   │
└─────────────┘                  └─────────────┘
     ↓                                  ↓
┌─────────────┐                  ┌─────────────┐
│ User A Data │                  │ User B Data │
│ Engagement  │                  │ Engagement  │
│ ID: 1234    │                  │ ID: 5678    │
└─────────────┘                  └─────────────┘

No interference between users!
Each user has isolated session storage.
```

## Performance Comparison

### Without Session Caching

```
Request 1: List VMs
├─ get_engagement_id()      → 300ms (API call)
├─ get_ipc_engagement_id()  → 250ms (API call)
└─ list_vms()               → 400ms (API call)
Total: 950ms

Request 2: List Clusters
├─ get_engagement_id()      → 300ms (API call)
├─ get_ipc_engagement_id()  → 250ms (API call)
└─ list_clusters()          → 600ms (API call)
Total: 1150ms

Request 3: List Business Units
├─ get_engagement_id()      → 300ms (API call)
├─ get_ipc_engagement_id()  → 250ms (API call)
└─ get_business_units()     → 200ms (API call)
Total: 750ms

TOTAL TIME: 2850ms
TOTAL API CALLS: 9
```

### With Session Caching

```
Request 1: List VMs
├─ get_engagement_id()      → 300ms (API call + cache)
├─ get_ipc_engagement_id()  → 250ms (API call + cache)
└─ list_vms()               → 400ms (API call)
Total: 950ms

Request 2: List Clusters
├─ get_engagement_id()      → 2ms (from cache ✅)
├─ get_ipc_engagement_id()  → 2ms (from cache ✅)
└─ list_clusters()          → 600ms (API call)
Total: 604ms

Request 3: List Business Units
├─ get_engagement_id()      → 2ms (from cache ✅)
├─ get_ipc_engagement_id()  → 2ms (from cache ✅)
└─ get_business_units()     → 200ms (API call + cache)
Total: 204ms

TOTAL TIME: 1758ms (38% faster!)
TOTAL API CALLS: 5 (saved 4 calls!)
```

## Cache Hit Rate Over Time

```
┌─────────────────────────────────────────────────────────────┐
│                    Cache Hit Rate                           │
└─────────────────────────────────────────────────────────────┘

Request #1:  [MISS] [MISS] [----]  Hit Rate: 0%
Request #2:  [HIT✅] [HIT✅] [----]  Hit Rate: 50%
Request #3:  [HIT✅] [HIT✅] [MISS]  Hit Rate: 67%
Request #4:  [HIT✅] [HIT✅] [HIT✅]  Hit Rate: 75%
Request #5:  [HIT✅] [HIT✅] [HIT✅]  Hit Rate: 80%
Request #10: [HIT✅] [HIT✅] [HIT✅]  Hit Rate: 90%

After first request, most subsequent requests benefit from cache!
```

## API Call Reduction

```
┌─────────────────────────────────────────────────────────────┐
│            API Calls: Before vs After Caching               │
└─────────────────────────────────────────────────────────────┘

Operation               │ Before │ After │ Savings
────────────────────────┼────────┼───────┼─────────
List VMs                │   3    │   1   │  67%
List Clusters           │   3    │   1   │  67%
List Business Units     │   3    │   1   │  67%
List Environments       │   3    │   1   │  67%
List Endpoints          │   2    │   1   │  50%
────────────────────────┼────────┼───────┼─────────
5 Operations (session)  │  14    │   5   │  64%

* "Before" = first request, "After" = subsequent requests
```

## Memory Usage

```
┌─────────────────────────────────────────────────────────────┐
│                    Memory Footprint                         │
└─────────────────────────────────────────────────────────────┘

Per User Session:
├─ Engagement IDs:        ~100 bytes
├─ Engagement Data:       ~2 KB
├─ Endpoints:             ~5 KB (10 endpoints)
├─ Business Units:        ~10 KB (20 BUs)
├─ Environments:          ~8 KB (30 environments)
└─ Metadata:              ~200 bytes
    ─────────────────────────────────
    Total per user:       ~25 KB

For 100 concurrent users: ~2.5 MB
For 1000 concurrent users: ~25 MB

Very lightweight! Minimal memory overhead.
```

## Best Practices

### ✅ DO
- Let the system auto-cache (default behavior)
- Use `force_refresh=True` when you know data changed
- Clear sessions on user logout
- Monitor cache hit rates in production

### ❌ DON'T
- Don't bypass cache unnecessarily
- Don't cache user-specific resource data (VMs, clusters)
- Don't set cache duration too long (>24 hours)
- Don't forget to handle cache expiration gracefully

## Summary

Session caching provides:
- **38-67% faster response times** for subsequent requests
- **64% reduction in API calls** after first request
- **Per-user isolation** for multi-tenant scenarios
- **Automatic expiration** after 24 hours
- **Minimal memory overhead** (~25 KB per user)
- **Thread-safe** implementation with locks

