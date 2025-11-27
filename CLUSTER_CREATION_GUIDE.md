# Kubernetes Cluster Creation Guide

## Overview

This document describes the multi-step workflow for creating Kubernetes clusters via the chatbot interface, including all required parameters, API payload structure, and validation rules.

---

## Quick Reference

### User Query Examples

**Implicit List Operations (NEW!)**:
- "cluster in delhi dc" ✅
- "show clusters in bengaluru" ✅
- "kubernetes in mumbai" ✅
- "clusters at chennai endpoint" ✅

**Explicit List Operations**:
- "list all clusters"
- "show me the clusters"
- "what clusters are available"

**Create Operations** (Coming Soon):
- "create a kubernetes cluster"
- "deploy a new cluster in delhi"
- "provision a k8s cluster"

---

## Endpoint Detection

### Supported Location Names

| User Input | Maps To | Endpoint ID |
|-----------|---------|-------------|
| delhi, delhi dc | Delhi | 11 |
| bengaluru, bangalore, blr, bengaluru dc | Bengaluru | 12 |
| mumbai, mumbai dc, bkc | Mumbai-BKC | 14 |
| chennai, chennai dc, amb | Chennai-AMB | 162 |
| cressex, uk, cressex dc, uk dc | Cressex | 204 |

### Implicit List Detection

The system now automatically detects implicit list operations when you mention a resource (cluster, k8s, kubernetes) and a location (delhi, bengaluru, etc.) **without** an explicit action verb.

**Examples**:
```
❌ Before: "cluster in delhi dc" → Fell back to RAG docs
✅ Now: "cluster in delhi dc" → Lists 13 Delhi clusters
```

---

## Cluster Creation

### API Endpoint

```
POST https://ipcloud.tatacommunications.com/paasservice/api/v1/iks
```

### Required Parameters

#### Basic Cluster Configuration

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `clusterName` | string | Cluster name (lowercase alphanumeric + hyphens) | `"my-prod-cluster"` |
| `k8sVersion` | string | Kubernetes version | `"v1.27.16"` |
| `clusterMode` | enum | Deployment mode | `"High availablity"` or `"Standard"` |
| `circuitId` | string | Circuit identifier | `"E-IPCTEAM-1602"` |
| `engagement_id` | integer | Customer engagement ID (auto-fetched) | `1923` |
| `endpoint_id` | integer | Data center endpoint ID | `11` (Delhi) |
| `zoneId` | integer | Availability zone ID | `16710` |

#### Resource Configuration

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `imageId` | integer | OS image ID | `43280` |
| `flavorId` | integer | Compute flavor ID | `3261` |
| `hypervisor` | enum | Hypervisor type | `"VCD_ESXI"` or `"KVM"` |

#### Master Node Configuration

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `master_node_config.vmFlavor` | string | Master node flavor | `"G.Gold.OL"` |
| `master_node_config.skuCode` | string | SKU code for billing | `"IKS.MGMT"` |
| `master_node_config.replicaCount` | integer | Number of master nodes (1-5) | `3` |
| `master_node_config.flavorDisk` | integer | Disk size in GB (50-1000) | `100` |

#### Worker Node Configuration

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `worker_node_config.vmHostName` | string | Worker node pool name | `"w1"` |
| `worker_node_config.vmFlavor` | string | Worker node flavor | `"C.Bronze.OL"` |
| `worker_node_config.skuCode` | string | SKU code for worker | `"C.Bronze.C.OLU"` |
| `worker_node_config.replicaCount` | integer | Number of worker nodes (1-100) | `3` |
| `worker_node_config.flavorDisk` | integer | Disk size in GB | `50` |

### Optional Parameters

#### Managed Services

```json
"managedServices": [
  {
    "name": "opensearch-v2.17.0"
  }
]
```

**Available Services**:
- opensearch-v2.17.0
- elasticsearch-v8.x
- kafka-v3.x
- prometheus-v2.x

#### Networking Driver

```json
"networkingDriver": [
  {
    "name": "calico-v3.25.1"
  }
]
```

**Available Drivers**:
- calico-v3.25.1
- flannel-v0.22
- weave-v2.8

#### Persistent Volumes

```json
"pvcsEnable": [
  {
    "size": 60,
    "iops": 1
  }
]
```

#### Backup Configuration

```json
"iksBackupDetails": {
  "labels": "l1=v1",
  "namespaces": [],
  "daily": {
    "startTime": "22:00",
    "retentionWindow": "7",
    "runsEvery": "1",
    "backupType": "FULL"
  },
  "weekly": {
    "startTime": "22:00",
    "retentionWindow": "30",
    "runsEvery": "1",
    "backupDay": "SUNDAY"
  },
  "monthly": {
    "startTime": "22:00",
    "retentionWindow": "90",
    "runsEvery": "1",
    "backupDate": "28"
  },
  "yearly": {
    "startTime": "22:00",
    "retentionWindow": "730",
    "backupDate": "31",
    "backupMonth": "DECEMBER"
  }
}
```

---

## Complete Payload Example

```json
{
  "name": "my-prod-cluster",
  "clusterName": "my-prod-cluster",
  "hypervisor": "VCD_ESXI",
  "purpose": "ipc",
  "vmPurpose": "APP",
  "imageId": 43280,
  "flavorId": 3261,
  "zoneId": 16710,
  "alertSuppression": true,
  "iops": 1,
  "isKdumpOrPageEnabled": "No",
  "managedServices": [
    {
      "name": "opensearch-v2.17.0"
    }
  ],
  "networkingDriver": [
    {
      "name": "calico-v3.25.1"
    }
  ],
  "pvcsEnable": [
    {
      "size": 60,
      "iops": 1
    }
  ],
  "logEnabled": true,
  "applicationType": "Container",
  "application": "Containers",
  "vmSpecificInput": [
    {
      "vmHostName": "",
      "vmFlavor": "G.Gold.OL",
      "skuCode": "IKS.MGMT",
      "nodeType": "Master",
      "replicaCount": 3,
      "maxReplicaCount": null,
      "additionalDisk": {},
      "flavorDisk": 100,
      "labelsNTaints": "no"
    },
    {
      "vmHostName": "w1",
      "vmFlavor": "C.Bronze.OL",
      "skuCode": "C.Bronze.C.OLU",
      "nodeType": "Worker",
      "replicaCount": 3,
      "maxReplicaCount": null,
      "additionalDisk": {},
      "flavorDisk": 50,
      "labelsNTaints": "no"
    }
  ],
  "clusterMode": "High availablity",
  "k8sVersion": "v1.27.16",
  "dedicatedDeployment": false,
  "circuitId": "E-IPCTEAM-1602",
  "vApp": "",
  "imageDetails": {
    "valueOSModel": "Ubuntu Linux",
    "valueOSMake": "Ubuntu",
    "valueOSVersion": "22.04 LTS",
    "valueOSServicePack": null
  },
  "flavorDisk": 50,
  "iksBackupDetails": {
    "labels": "l1=v1",
    "namespaces": [],
    "daily": {
      "startTime": "22:00",
      "retentionWindow": "7",
      "runsEvery": "1",
      "backupType": "FULL"
    },
    "weekly": {
      "startTime": "22:00",
      "retentionWindow": "30",
      "runsEvery": "1",
      "backupDay": "SUNDAY"
    },
    "monthly": {
      "startTime": "22:00",
      "retentionWindow": "90",
      "runsEvery": "1",
      "backupDate": "28"
    },
    "yearly": {
      "startTime": "22:00",
      "retentionWindow": "730",
      "backupDate": "31",
      "backupMonth": "DECEMBER"
    }
  }
}
```

---

## Cluster Creation Workflow

### Step-by-Step Process

The chatbot will guide you through these steps:

#### 1. Get Engagement Details
**Automatic** - Fetches customer engagement ID
- Cached for 1 hour
- Required for all subsequent API calls

#### 2. Select Data Center
**User Selection** - Choose target data center
```
Prompt: "Which data center would you like to deploy the cluster in?"
Options: Delhi, Bengaluru, Mumbai-BKC, Chennai-AMB, Cressex
```

#### 3. Collect Basic Configuration
**User Input** - Provide cluster details
```
- Cluster Name: (lowercase alphanumeric + hyphens)
- Kubernetes Version: (e.g., v1.27.16, v1.28.15, v1.29.12)
- Cluster Mode: (High availablity or Standard)
- Circuit ID: (e.g., E-IPCTEAM-1602)
```

#### 4. Select Zone and Resources
**User Selection** - Choose infrastructure resources
```
- Availability Zone
- OS Image (Ubuntu 22.04 LTS recommended)
- Compute Flavor
```

#### 5. Configure Node Pools
**User Input** - Define master and worker nodes
```
Master Nodes:
- Flavor: (e.g., G.Gold.OL)
- Count: (1-5, recommended: 3 for HA)
- Disk Size: (50-1000 GB, default: 100 GB)

Worker Nodes:
- Pool Name: (e.g., "w1", "default-workers")
- Flavor: (e.g., C.Bronze.OL)
- Count: (1-100)
- Disk Size: (50-1000 GB, default: 50 GB)
```

#### 6. Configure Optional Features
**Optional** - Add managed services, PVCs, backups
```
- Managed Services: (OpenSearch, Kafka, etc.)
- Persistent Volumes: (size and IOPS)
- Backup Schedules: (daily, weekly, monthly, yearly)
- Networking Driver: (Calico, Flannel, Weave)
```

#### 7. Validate Configuration
**Automatic** - Validates all parameters
- Checks naming conventions
- Validates resource availability
- Ensures required parameters are present

#### 8. Create Cluster
**Confirmation Required** - Submit creation request
```
Confirmation: "Ready to create cluster 'my-prod-cluster' in Delhi? This will take 15-30 minutes."
```

### Estimated Timeline

- **Configuration**: 5-10 minutes (with user input)
- **Cluster Provisioning**: 15-30 minutes (automated)
- **Total**: ~20-40 minutes

---

## Validation Rules

### Cluster Name
- Pattern: `^[a-z0-9-]+$`
- Length: 3-63 characters
- Must be lowercase
- Only alphanumeric and hyphens

### Kubernetes Version
- Pattern: `^v[0-9]+\.[0-9]+\.[0-9]+$`
- Example: `v1.27.16`
- Must start with "v"

### Node Counts
- Master: 1-5 nodes (recommended: 3 for HA)
- Worker: 1-100 nodes

### Disk Sizes
- Range: 50-1000 GB
- Master default: 100 GB
- Worker default: 50 GB

---

## Error Handling

### Common Errors

#### 1. Invalid Cluster Name
```
Error: "Cluster name must be lowercase alphanumeric with hyphens only"
Fix: Use only a-z, 0-9, and hyphens
```

#### 2. Zone/Resource Not Available
```
Error: "Selected zone or flavor not available in this data center"
Fix: Choose a different zone or flavor from available options
```

#### 3. Circuit ID Not Found
```
Error: "Circuit ID not found or not associated with your engagement"
Fix: Verify circuit ID with your account manager
```

#### 4. Insufficient Quota
```
Error: "Insufficient quota for requested resources"
Fix: Request quota increase or reduce resource requirements
```

---

## Monitoring Cluster Creation

### After Submission

1. **Immediate Response**: You'll receive a cluster ID
2. **Status Tracking**: Use "list clusters" to check status
3. **Completion Time**: Typically 15-30 minutes

### Cluster States

| State | Description | Duration |
|-------|-------------|----------|
| Pending | Cluster creation queued | 1-2 mins |
| Provisioning | Infrastructure being created | 10-15 mins |
| Configuring | K8s components being installed | 5-10 mins |
| Healthy | Cluster ready to use | - |
| Error | Creation failed (check logs) | - |

---

## Next Steps

Once your cluster is created:

1. **Download kubeconfig**: "get kubeconfig for my-prod-cluster"
2. **Verify access**: `kubectl get nodes`
3. **Deploy applications**: Use kubectl or Helm
4. **Monitor health**: Check cluster status regularly

---

## Resource Schema Location

Full schema definition: `app/config/resource_schema.json`

Updated sections:
- `k8s_cluster.parameters.create` - All create parameters and validation
- `k8s_cluster.workflow.create_cluster` - 8-step creation workflow

---

## Testing

### Test List Operations

```bash
# All clusters
curl -X POST http://localhost:8001/api/chat/query \
  -H "Content-Type: application/json" \
  -d '{"query": "list all clusters"}'

# Specific endpoint (implicit)
curl -X POST http://localhost:8001/api/chat/query \
  -H "Content-Type: application/json" \
  -d '{"query": "cluster in delhi dc"}'

# Specific endpoint (explicit)
curl -X POST http://localhost:8001/api/chat/query \
  -H "Content-Type: application/json" \
  -d '{"query": "show clusters in bengaluru"}'
```

### Test Create Operations (Coming Soon)

```bash
curl -X POST http://localhost:8001/api/chat/query \
  -H "Content-Type: application/json" \
  -d '{"query": "create a kubernetes cluster in delhi"}'
```

---

## Support

For issues or questions:
- Check logs: `/tmp/user_main.log`
- Review resource schema: `app/config/resource_schema.json`
- Contact: Platform team

---

**Last Updated**: November 24, 2025
**Status**: List operations ✅ | Create operations 🚧 (in progress)

