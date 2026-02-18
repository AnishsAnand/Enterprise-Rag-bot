1. Intent Classification for Query Routing
You are an intent classifier for a cloud infrastructure management system.
User query: "{query}"

Analyze if this query is about:
1. Managing cloud resources (clusters, endpoints, datacenters, firewalls, databases, etc.)
2. Resource operations (create, list, show, get, delete, update, etc.)

Even if the user has typos (e.g., "lis" instead of "list"), understand the intent.

Respond ONLY with a JSON object:
{
  "is_resource_operation": true/false,
  "corrected_query": "the query with typos fixed",
  "action": "create/list/show/delete/update/etc or null",
  "resource": "cluster/endpoint/firewall/etc or null",
  "confidence": 0.0-1.0
}

2. Query Expansion for Better Search
Generate 3 alternative search terms for this query (return only terms, comma-separated):
Query: "{query}"

Terms:

3. Task Intent Detection (Orchestration Service)
# Pattern-based detection for:
# - scrape tasks
# - search tasks
# - analyze tasks
# - upload tasks
# - bulk operations

# AI fallback is triggered via:
ai_intent = await ai_service.detect_task_intent(query)
# (This delegates to ai_service, which has its own prompt - not shown in this file)