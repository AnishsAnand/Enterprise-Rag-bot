1. Detailed Load Balancer Formatting
You are formatting load balancer information for a network engineer. 
Format this data in a clear, professional manner suitable for production operations.

**Load Balancer:** {lb_name}
**LBCI:** {lbci}

**Virtual Services Data:**
{json.dumps(virtual_services, indent=2)}

**Configuration Details:**
{json.dumps(details, indent=2) if details else "Configuration details unavailable"}

**REQUIRED FORMAT:**

# Load Balancer: {lb_name}

**Circuit ID (LBCI):** `{lbci}`

## Virtual Services ({vs_count} configured)

For each virtual service, display:

### [Number]. [Virtual Server Name]

| Property | Value |
|----------|-------|
| **VIP Address** | [vipIp]:[virtualServerport] |
| **Protocol** | [protocol] |
| **Status** | [emoji] [status] |
| **Load Balancing** | [poolAlgorithm] |
| **Health Monitor** | [monitor array as comma-separated] |
| **Persistence** | [persistenceType] ([persistenceValue]) |
| **Pool Members** | [poolMembers count or details] |
| **Pool Path** | `[virtualServerPath]` |

**Status Icons:**
- UP = ✅
- DOWN = ⚠️
- DEGRADED = 🟡
- UNKNOWN = ❓

**Special Notes:**
- If certificate is configured, mention: **SSL Certificate:** [certificateName]
- If pool members exist, list them
- If persistence is null, show "None"
- Use proper formatting with tables for readability

**Example Output:**

### 1. TESTPUBLIC

| Property | Value |
|----------|-------|
| **VIP Address** | 100.94.45.12:9056 |
| **Protocol** | HTTP |
| **Status** | ⚠️ DOWN |
| **Load Balancing** | Round Robin |
| **Health Monitor** | System-TCP |
| **Persistence** | None |
| **Pool Path** | `IPC_VS_1602_DWZ_4762_TESTPUBLIC` |

---

## Configuration Summary

If configuration details are available, add a summary section with:
- Total virtual services
- Active vs. inactive services
- Most common protocol
- Health monitor types in use

**Important Rules:**
1. Use EXACT field names from the API response
2. Handle null values gracefully (show "N/A" or "None")
3. Format arrays as comma-separated strings
4. Use code blocks for technical paths
5. Include ALL virtual services
6. If no virtual services: show "ℹ️ No virtual services configured"

Return ONLY the formatted markdown. NO preamble or explanation.