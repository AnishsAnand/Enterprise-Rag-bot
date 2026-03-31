"""
ADK Agent Definitions for Enterprise RAG Bot.

Architecture (with native tool calling on gpt-oss-120b):
  root_agent (orchestrator)
    ├── rag_agent          → search_knowledge_base
    ├── resource_agent     → list_k8s_clusters, get_cluster_details, check_cluster_name,
    │                         list_vms, list_firewalls, list_load_balancers,
    │                         get_load_balancer_details, list_managed_services,
    │                         list_zones, list_environments, list_business_units,
    │                         get_cluster_report
    └── engagement_agent   → get_engagements, select_engagement

The root_agent decides which sub-agent to delegate to based on the user's query.
The LLM drives all routing and tool invocation natively (no regex needed).

NOTE: ENG/CUS engagement pre-check is handled BEFORE reaching ADK in runner.py.
Once engagement_id is set in state, the agents can call resource tools freely.
"""
import logging
import os

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model Configuration
# ---------------------------------------------------------------------------

def _build_model() -> LiteLlm:
    """Build the LiteLlm model connector for the configured LLM endpoint.

    Handles CHAT_MODEL values with or without the 'openai/' prefix.
    Example: 'openai/gpt-oss-120b' and 'gpt-oss-120b' both work correctly.
    """
    chat_model = os.getenv("CHAT_MODEL", "Qwen/Qwen2.5-Coder-14B-Instruct")
    api_key = os.getenv("GROK_API_KEY", "")
    api_base = os.getenv("GROK_BASE_URL", "https://models.cloudservices.tatacommunications.com/v1")

    # LiteLlm ALWAYS strips the 'openai/' provider prefix before sending the model
    # name to the API endpoint. The server registers models with their full ID
    # (e.g. "openai/gpt-oss-120b"), so we must always prepend "openai/" here
    # so that after LiteLlm strips it, the API receives the correct model ID.
    #
    # Examples:
    #   CHAT_MODEL=openai/gpt-oss-120b  → model_string=openai/openai/gpt-oss-120b
    #                                      LiteLlm strips → API receives openai/gpt-oss-120b ✅
    #   CHAT_MODEL=Qwen/Qwen2.5-Coder  → model_string=openai/Qwen/Qwen2.5-Coder
    #                                      LiteLlm strips → API receives Qwen/Qwen2.5-Coder ✅
    model_string = f"openai/{chat_model}"

    model = LiteLlm(
        model=model_string,
        api_key=api_key,
        api_base=api_base,
    )
    logger.info("ADK model configured: %s @ %s", model_string, api_base)
    return model


# ---------------------------------------------------------------------------
# Agent Instructions
# ---------------------------------------------------------------------------

ROOT_AGENT_INSTRUCTION = """You are **Vayu Maya**, the AI Cloud Assistant by Tata Communications.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 1 — GREETINGS (highest priority, no exceptions)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
If the message is a greeting or casual chat (hi, hello, hey, how are you,
what can you do, thanks, bye, okay, cool, namaste, kya hal hai, etc.):
  • Reply in 1–2 sentences only.
  • Do NOT call any sub-agent or tool.
  • Do NOT mention engagements, subscriptions, IDs, or resource status.
Example: "Hello! 👋 I'm Vayu Maya. Ask me to list clusters, VMs, firewalls, or anything about your cloud!"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 2 — ROUTING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• "what is X / explain / how do I / guide / best practice" → rag_agent
• "list / show / get clusters, VMs, firewalls, LBs, services..." → resource_agent
• "show my engagements / switch engagement / change project" → engagement_agent

NOTE: Engagement selection for ENG users is already handled by the system before
this agent runs. When you receive a resource query, the engagement_id is set in
session state — go directly to resource_agent without calling engagement_agent first.

STRICT RULES:
• If a sub-agent returns an error, report it clearly. Never retry or re-transfer.
• Only describe capabilities you actually have. No CSV, Terraform, billing, or cost reports.
• Be concise. Use Markdown.
"""

RAG_AGENT_INSTRUCTION = """You are the documentation specialist for the Vayu Maya cloud platform.

Your job: answer questions about the platform by searching the knowledge base.

ALWAYS call search_knowledge_base with the user's question.
Synthesise the results into a clear, well-structured Markdown answer.
If results are insufficient, say so honestly and suggest rephrasing.

FORMAT: Use headers, bullet points, and numbered lists where appropriate.
"""

RESOURCE_AGENT_INSTRUCTION = """You are the cloud resource specialist for the Vayu Maya platform.

Your job: list and inspect live cloud resources for the user's engagement.

AVAILABLE TOOLS:
- list_k8s_clusters: List all Kubernetes clusters
- get_cluster_details: Get details of a specific cluster (requires cluster_name)
- check_cluster_name: Check if a cluster name is available
- list_vms: List all virtual machines
- list_firewalls: List all firewalls
- list_load_balancers: List all load balancers
- get_load_balancer_details: Get details of a specific load balancer (requires lb_name)
- list_managed_services: List managed services — specify service_type: kafka, postgres, documentdb, gitlab, jenkins, container_registry
- list_zones: List available zones/regions
- list_environments: List available environments
- list_business_units: List business units/departments
- get_cluster_report: Generate cluster reports — specify report_type: common_cluster, cluster_inventory, cluster_compute, storage_inventory

STRICT RULES:
1. Call the appropriate tool based on what the user wants to see.
2. For cluster details, extract the cluster name from the user's message.
3. For managed services, identify the service type from the user's message.
4. Format responses with Markdown tables where appropriate.
5. If a tool returns an error, report the exact error message to the user cleanly. Do NOT retry the same tool. Do NOT transfer back to the root agent. Just explain what failed.
6. Never claim resources don't exist — say "unable to retrieve" and show the error.
"""

ENGAGEMENT_AGENT_INSTRUCTION = """You are the engagement management specialist for the Vayu Maya platform.

Your job: help users view and select their cloud engagements (subscriptions/projects).

AVAILABLE TOOLS:
- get_engagements: Lists all available engagements. Call this to get the list.
- select_engagement: Sets the active engagement by engagement_id (integer).

WORKFLOW:
1. Call get_engagements to fetch the list.
2. Present the engagements as a numbered Markdown table.
3. Ask the user to reply with a number (e.g. "Reply with 1, 2, 3...").
4. Do NOT auto-select or confirm any engagement — let the user choose explicitly.
5. Do NOT call select_engagement until the user replies with a number in the NEXT message.
"""


# ---------------------------------------------------------------------------
# Build Agent Hierarchy
# ---------------------------------------------------------------------------

def build_agent_hierarchy() -> LlmAgent:
    """Build and return the root_agent with sub-agents and tools attached.

    Architecture:
      root_agent (orchestrator, no direct tools)
        ├── rag_agent          (search_knowledge_base)
        ├── resource_agent     (all resource listing/inspection tools)
        └── engagement_agent   (get_engagements, select_engagement)
    """
    from app.adk.tools import (
        # RAG
        search_knowledge_base,
        # Engagement
        get_engagements,
        select_engagement,
        # K8s
        list_k8s_clusters,
        get_cluster_details,
        check_cluster_name,
        get_cluster_report,
        # VM
        list_vms,
        # Network
        list_firewalls,
        list_load_balancers,
        get_load_balancer_details,
        # Managed Services
        list_managed_services,
        # Infrastructure
        list_zones,
        list_environments,
        list_business_units,
    )

    model = _build_model()

    # --- RAG Sub-agent ---
    rag_agent = LlmAgent(
        model=model,
        name="rag_agent",
        description=(
            "Answers questions about the Vayu Maya platform using the knowledge base. "
            "Use for: documentation, how-to guides, concepts, best practices, explanations."
        ),
        instruction=RAG_AGENT_INSTRUCTION,
        tools=[search_knowledge_base],
    )
    logger.info("ADK rag_agent built with tools: [search_knowledge_base]")

    # --- Resource Sub-agent ---
    resource_tools = [
        list_k8s_clusters,
        get_cluster_details,
        check_cluster_name,
        get_cluster_report,
        list_vms,
        list_firewalls,
        list_load_balancers,
        get_load_balancer_details,
        list_managed_services,
        list_zones,
        list_environments,
        list_business_units,
    ]
    resource_agent = LlmAgent(
        model=model,
        name="resource_agent",
        description=(
            "Lists and inspects live cloud resources: Kubernetes clusters, VMs, firewalls, "
            "load balancers, managed services (Kafka, PostgreSQL, DocumentDB, GitLab, Jenkins, "
            "Container Registry), zones, environments, and business units."
        ),
        instruction=RESOURCE_AGENT_INSTRUCTION,
        tools=resource_tools,
    )
    logger.info(
        "ADK resource_agent built with %d tools: %s",
        len(resource_tools),
        [t.__name__ for t in resource_tools],
    )

    # --- Engagement Sub-agent ---
    engagement_agent = LlmAgent(
        model=model,
        name="engagement_agent",
        description=(
            "Manages cloud engagements (subscriptions/projects). "
            "Use for: listing engagements, selecting/switching engagement, "
            "checking which engagement is active."
        ),
        instruction=ENGAGEMENT_AGENT_INSTRUCTION,
        tools=[get_engagements, select_engagement],
    )
    logger.info("ADK engagement_agent built with tools: [get_engagements, select_engagement]")

    # --- Root Agent (Orchestrator) ---
    root_agent = LlmAgent(
        model=model,
        name="root_agent",
        description="Vayu Maya AI Cloud Assistant — routes requests to specialist sub-agents.",
        instruction=ROOT_AGENT_INSTRUCTION,
        sub_agents=[rag_agent, resource_agent, engagement_agent],
    )
    logger.info(
        "ADK root_agent built with sub_agents: [rag_agent, resource_agent, engagement_agent]"
    )

    return root_agent


def get_model() -> LiteLlm:
    """Get a LiteLlm model instance for ad-hoc LLM calls."""
    return _build_model()
