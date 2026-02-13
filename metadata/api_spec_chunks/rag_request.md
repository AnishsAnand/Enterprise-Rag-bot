# Kubernetes Cluster API - Dynamic RAG Integration

Production-ready markdown files for the IPCloud Kubernetes cluster APIs with full RAG integration support.

## 📦 What's Included

### API Specification Files
1. **k8s_cluster_get_info.md** - Get detailed cluster information
   - URL: `/paasservice/paas/cluster/{cluster_id}/getclusterinfo`
   - Returns: Cluster config, nodes, status, resources

2. **k8s_cluster_get_metrics.md** - Get cluster performance metrics
   - URL: `/paasservice/paas/cluster/{cluster_id}/clustermetrics`
   - Returns: CPU, memory, network, storage metrics

### Deployment Tools
3. **deploy_cluster_apis.sh** - Automated deployment script
4. **validate_cluster_md.py** - Validation and testing tool
5. **CLUSTER_API_USAGE_GUIDE.md** - Complete usage documentation

## 🚀 Quick Start (3 Steps)

### Step 1: Copy Files

```bash
# Copy MD files to your metadata directory
cp k8s_cluster_get_info.md metadata/api_spec_chunks/
cp k8s_cluster_get_metrics.md metadata/api_spec_chunks/
```

### Step 2: Ingest into RAG

```bash
# Append to existing RAG (recommended)
python3 -m app.scripts.retrain_rag --no-clear

# OR full retrain (clears everything first)
python3 -m app.scripts.retrain_rag
```

### Step 3: Test

```bash
# Test RAG search
python3 -m app.scripts.test_rag_intent "show cluster information"
python3 -m app.scripts.test_rag_intent "get cluster metrics"
```

**Done!** Your system now understands cluster queries.

## 🎯 Automated Deployment (Recommended)

Use the included deployment script:

```bash
# Make executable
chmod +x deploy_cluster_apis.sh

# Run interactive deployment
./deploy_cluster_apis.sh
```

The script will:
1. ✅ Validate MD file structure
2. ✅ Copy files to metadata directory
3. ✅ Check database connection
4. ✅ Ingest into RAG (with options)
5. ✅ Run test queries

## 📝 Key Features

### ✅ Dynamic Parameters
- Uses `{cluster_id}` placeholder
- No hardcoded IDs that become stale
- Works with any cluster in your system

**Example:**
```
User: "Show info for cluster abc123"
→ URL becomes: .../cluster/abc123/getclusterinfo
```

### ✅ Comprehensive Documentation
- Real-world response examples
- Common use cases with natural language
- Error handling guide
- Performance baselines
- Health indicators

### ✅ RAG-Optimized
- Multiple aliases for flexible matching
- Clear parameter requirements
- Nested response field mappings
- Related operations linked

## 🔍 Testing & Validation

### Validate Files Before Deployment

```bash
# Validate structure
python3 validate_cluster_md.py

# Validate and ingest
python3 validate_cluster_md.py --ingest

# Validate, ingest, and test
python3 validate_cluster_md.py --ingest --test
```

### Manual Testing

```bash
# Test RAG search only (no LLM)
python3 -m app.scripts.test_rag_intent --rag-only

# Test full intent flow (RAG + LLM)
python3 -m app.scripts.test_rag_intent "show cluster abc123 info"
python3 -m app.scripts.test_rag_intent "what is CPU usage of cluster xyz"
```

### Check Database

```bash
# Check if files are ingested
python3 -m app.scripts.rag_monitor --db-stats | grep cluster

# Full health check
python3 -m app.scripts.rag_monitor --health-check
```

## 💬 Query Examples That Work

### Get Cluster Info Queries
```
✓ "Show me information about cluster abc123"
✓ "Get cluster details"
✓ "What is the status of cluster xyz?"
✓ "Show cluster configuration"
✓ "List nodes in the cluster"
✓ "Cluster abc123 info"
```

### Get Metrics Queries
```
✓ "Show cluster metrics"
✓ "What is the CPU usage of cluster abc?"
✓ "Get memory utilization"
✓ "Show cluster performance"
✓ "How much storage is used?"
✓ "Cluster health status"
```

## 🔧 When Base URL Changes

If your PaaS service URL changes:

### Option 1: Quick Update
```bash
# Update in both files
sed -i 's|ipcloud.tatacommunications.com/paasservice|new.domain.com/api|g' \
  metadata/api_spec_chunks/k8s_cluster_*.md

# Re-ingest
python3 -m app.scripts.retrain_rag --no-clear
```

### Option 2: Use Dynamic Generator
```bash
# Update config
nano api_config.json  # Update base URL

# Regenerate specs
python3 -m app.scripts.retrain_rag_enhanced \
  --generate-specs \
  --config api_config.json
```

## 📊 Response Structure

### Cluster Info Response
```json
{
  "data": {
    "id": "cluster-abc123",
    "name": "production-cluster-01",
    "status": "running",
    "nodeCount": 5,
    "nodes": [...],
    "version": "1.28.0"
  }
}
```

**RAG learns these mappings:**
- `data.id` → cluster_id
- `data.name` → cluster_name
- `data.status` → cluster_status
- `data.nodes[*]` → node array

### Metrics Response
```json
{
  "data": {
    "cpu": {
      "usage": 4.5,
      "percentage": 28.13
    },
    "memory": {
      "usage": 12.5,
      "percentage": 39.06
    },
    "pods": {
      "running": 42,
      "pending": 2
    }
  }
}
```

**RAG learns these mappings:**
- `data.cpu.usage` → CPU usage
- `data.cpu.percentage` → CPU %
- `data.memory.usage` → Memory GB
- `data.pods.running` → Running pods

## 🔄 Scheduled Updates (Optional)

Keep cluster APIs fresh with cron:

```bash
# Daily update at 2 AM
0 2 * * * /opt/rag/update_cluster_specs.sh >> /var/log/cluster_specs.log 2>&1
```

**update_cluster_specs.sh:**
```bash
#!/bin/bash
set -e

# Re-ingest cluster specs
cd /path/to/project
python3 -m app.scripts.retrain_rag --no-clear

echo "✅ Cluster specs updated: $(date)"
```

## 🐛 Troubleshooting

### Issue: RAG Not Finding Cluster Queries

**Check 1: Files ingested?**
```bash
python3 -c "
import asyncio
from app.services.postgres_service import postgres_service

async def check():
    await postgres_service.initialize()
    result = await postgres_service.pool.fetch(
        \"SELECT title FROM enterprise_rag WHERE title LIKE '%cluster%'\"
    )
    for r in result:
        print(r['title'])

asyncio.run(check())
"
```

**Check 2: RAG search working?**
```bash
python3 -m app.scripts.test_rag_intent --rag-only
# Type: cluster information
```

### Issue: Wrong Cluster ID

**Check intent extraction logs:**
```python
# Look for in logs:
"extracted params": {"cluster_id": "abc123"}
```

### Issue: Database Connection

```bash
# Verify env vars
env | grep POSTGRES

# Test connection
psql -h $POSTGRES_HOST -U $POSTGRES_USER -d $POSTGRES_DB -c "SELECT 1;"
```

## 📚 Integration Flow

```
1. User Query: "Show cluster abc123 info"
            ↓
2. Intent Agent searches RAG: "cluster info"
            ↓
3. RAG returns: k8s_cluster.get_info spec
            ↓
4. Agent extracts: cluster_id = "abc123"
            ↓
5. URL constructed: .../cluster/abc123/getclusterinfo
            ↓
6. API called with actual cluster ID
            ↓
7. Response returned to user
```

## ✅ Production Checklist

Before deploying to production:

- [ ] Files copied to `metadata/api_spec_chunks/`
- [ ] MD files validated (no errors)
- [ ] Files ingested into RAG successfully
- [ ] RAG search finds "cluster info" queries
- [ ] RAG search finds "cluster metrics" queries
- [ ] Intent agent extracts cluster_id correctly
- [ ] API calls construct correct URL
- [ ] Response mapping works for nested fields
- [ ] Tested with real cluster IDs
- [ ] Error handling working properly
- [ ] Monitoring set up (optional)
- [ ] Documentation updated for team

## 📖 Related Documentation

- **CLUSTER_API_USAGE_GUIDE.md** - Detailed usage guide
- **PRODUCTION_DEPLOYMENT_GUIDE.md** - Full production setup
- **MIGRATION_GUIDE.md** - Migrating from manual system

## 🎯 Success Metrics

After deployment, verify:
- ✅ Cluster queries return correct intent
- ✅ Dynamic cluster_id extraction works
- ✅ API calls succeed with real IDs
- ✅ Response parsing handles nested fields
- ✅ Users get accurate, helpful responses

## 🆘 Support

### Quick Help

1. **Validation issues?**
   ```bash
   python3 validate_cluster_md.py
   ```

2. **Ingestion failed?**
   ```bash
   python3 -m app.scripts.rag_monitor --health-check
   ```

3. **RAG not finding queries?**
   ```bash
   python3 -m app.scripts.test_rag_intent --rag-only
   ```

### Logs to Check
- `/var/log/rag_retrain.log` - Ingestion logs
- `/var/log/app.log` - Application logs
- Intent agent logs - RAG search results

## 📝 Quick Reference

### Essential Commands

```bash
# Deploy
./deploy_cluster_apis.sh

# Validate
python3 validate_cluster_md.py --ingest --test

# Test
python3 -m app.scripts.test_rag_intent "show cluster info"

# Monitor
python3 -m app.scripts.rag_monitor --db-stats

# Update URL
sed -i 's|old-url|new-url|g' metadata/api_spec_chunks/k8s_cluster_*.md
python3 -m app.scripts.retrain_rag --no-clear
```

---

**Version**: 1.0.0  
**Created**: 2025-02-13  
**For**: IPCloud Kubernetes Cluster APIs  
**Compatible with**: RAG Phase 1, 2, 3