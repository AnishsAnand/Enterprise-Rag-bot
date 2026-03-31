"""
ADK Runner – the entry point that replaces AgentManager.

Provides ``process_request()`` with the exact same signature and return
format as the old ``AgentManager.process_request()`` so that the API layer
needs no changes.

ARCHITECTURE (with native tool calling on gpt-oss-120b):
  1. Engagement Pre-check (Python logic — fast, reliable)
     - ENG user: if no engagement selected yet, show engagement list
     - CUS user: auto-select single engagement from API
     - If user replies with a number: call select_engagement handler
  2. ADK Native Orchestration (LLM-driven — no regex)
     - Pass the query to ADK Runner.run_async()
     - root_agent decides: delegate to rag_agent, resource_agent, or engagement_agent
     - The sub-agent calls the right tool(s) automatically
     - ADK collects tool results and formats the final response

NOTE: The old regex-based routing and manual _dispatch_resource_tool are
commented out below for reference. They are no longer used.
"""
import logging
import os
import re  # kept only for commented-out legacy patterns below
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from app.adk.agents import build_agent_hierarchy

logger = logging.getLogger(__name__)

APP_NAME = "enterprise_rag"

# ---------------------------------------------------------------------------
# Pure-Python helpers — no LLM, no regex
# ---------------------------------------------------------------------------

def _is_engagement_number(text: str) -> Optional[int]:
    """Returns the integer if the user typed a selection number, else None."""
    cleaned = text.strip().lower()
    for prefix in ("select", "choose", "use", "pick", "go with", "switch to", "engagement", "#"):
        cleaned = cleaned.removeprefix(prefix).strip()
    return int(cleaned) if cleaned.isdigit() else None


# Single greeting-opener words — first word of message only.
_GREETING_FIRST_WORDS = {
    "hi", "hello", "hey", "namaste", "howdy", "greetings", "hiya", "ciao",
    "hola", "bonjour", "yo", "sup", "bye", "goodbye", "thanks", "thank",
    "okay", "ok", "cool", "great", "nice", "awesome", "welcome",
}

# Common typos / variants of "hello" and "hi" so ENG precheck + greeting path still match.
_GREETING_TYPOS = frozenset({
    "heloo", "helloo", "helo", "hallo", "hullo", "hulloo", "hii", "hiii", "heyy", "heyyy",
    "hlo", "ello",
})

# Two-word casual starters — checked against the first two words.
# Covers capability/conversational questions that don't need cloud data.
_CASUAL_TWO_WORD_STARTS = {
    "what can", "what do", "what are", "what is", "what what", "what would",
    "what could", "how can", "how do", "how are", "how is", "can you",
    "could you", "would you", "do you", "did you", "tell me", "who are",
    "who is", "i like", "i love", "i hate", "thank you", "good morning",
    "good afternoon", "good evening", "good night",
}


def _normalize_chat_input(text: str) -> str:
    """Strip whitespace and BOM so classification matches what the user sees."""
    if not text:
        return ""
    t = text.strip()
    if t.startswith("\ufeff"):
        t = t.lstrip("\ufeff").strip()
    return t


# If any of these tokens appear, the message is not "greeting-only" (avoid ADK tool loop skip).
_RESOURCE_HINT_WORDS = frozenset({
    "list", "show", "get", "fetch", "display", "give", "tell",
    "create", "delete", "update", "scale", "switch", "change",
    "cluster", "clusters", "kubernetes", "k8s",
    "vm", "vms", "instance", "instances",
    "firewall", "firewalls",
    "balancer", "lb", "lbs",
    "engagement", "engagements", "subscription", "project",
    "datacenter", "datacenters", "zone", "zones", "region", "regions",
    "namespace", "namespaces", "pod", "pods",
    "business", "environment", "environments", "report",
    "postgresql", "postgres", "kafka", "documentdb", "gitlab", "jenkins",
    "registry",
})


def _is_standalone_greeting_or_casual(text: str) -> bool:
    """
    True only for short social / capability chit-chat with no cloud action words.
    Used to skip the ADK tool loop — the chat model sometimes calls select_engagement
    on \"hello\" when tools are enabled, which produced bogus \"engagement switched\" replies.
    """
    if not _is_greeting(text):
        return False
    raw = text.strip().lower()
    for ch in ".,!?;:()[]{}\"'":
        raw = raw.replace(ch, " ")
    words = {w for w in raw.split() if w}
    if words & _RESOURCE_HINT_WORDS:
        return False
    return True


def _is_greeting(text: str) -> bool:
    """
    True if the message is a greeting or casual conversational question.
    Pure Python — no LLM, no regex.
      'hi'                         → True  (first word match)
      'hello there'                → True
      'what can you do for me'     → True  (two-word start match)
      'what is kubernetes'         → True  (casual question, no cloud data needed for ENG gate)
      'list clusters'              → False
      'show my VMs'                → False
    """
    words = text.strip().lower().split()
    if not words:
        return False
    first = words[0].rstrip("!?,.:")
    # Single greeting word at the start (including common "hello"/"hi" typos like "heloo")
    if first in _GREETING_FIRST_WORDS or first in _GREETING_TYPOS:
        return True
    # Two-word casual phrase at the start
    if len(words) >= 2:
        two = words[0] + " " + words[1]
        if two in _CASUAL_TWO_WORD_STARTS:
            return True
    return False


# ---------------------------------------------------------------------------
# ADKRunner
# ---------------------------------------------------------------------------


class ADKRunner:
    """
    Main orchestrator. Replaces AgentManager.

    Flow per request:
      1. Create/retrieve ADK session (stores auth_token, user_type, engagement_id)
      2. Engagement pre-check (Python, reliable):
         - ENG user without selection → show engagement list
         - User replies with number → select engagement
         - CUS user → auto-select from API
      3. Pass query to ADK Runner.run_async()
         → root_agent decides which sub-agent to delegate to
         → sub-agent calls the right tool(s) natively
         → ADK collects results and returns formatted response
    """

    def __init__(self):
        self._runner: Optional[Runner] = None
        self._session_service = None
        self._initialized = False
        self._total_requests = 0
        self._initialization_time: Optional[datetime] = None

    def initialize(self, session_service=None):
        if self._initialized:
            return

        root_agent = build_agent_hierarchy()
        self._session_service = session_service or InMemorySessionService()

        self._runner = Runner(
            agent=root_agent,
            app_name=APP_NAME,
            session_service=self._session_service,
        )

        self._initialized = True
        self._initialization_time = datetime.utcnow()
        logger.info("ADK Runner initialized (app=%s) with native tool calling", APP_NAME)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def process_request(
        self,
        user_input: str,
        session_id: str,
        user_id: str,
        user_roles: List[str] = None,
        auth_token: str = None,
        user_type: str = None,
        force_rag_only: bool = False,
    ) -> Dict[str, Any]:
        if not self._initialized:
            self.initialize()

        self._total_requests += 1
        start_time = datetime.utcnow()
        user_input = _normalize_chat_input(user_input)
        logger.info(
            "ADK request #%d | session=%s user=%s type=%s force_rag=%s query='%s'",
            self._total_requests, session_id, user_id, user_type, force_rag_only,
            user_input[:80],
        )

        try:
            if not user_input:
                duration = (datetime.utcnow() - start_time).total_seconds()
                return {
                    "success": True,
                    "response": "How can I help you today?",
                    "routing": "greeting",
                    "execution_result": None,
                    "sources": [],
                    "images": [],
                    "follow_ups": [],
                    "metadata": self._meta(duration, session_id, user_id),
                }

            # Step 1: Create/retrieve session with user context
            session = await self._get_or_create_session(
                session_id=session_id,
                user_id=user_id,
                auth_token=auth_token,
                user_type=user_type,
                user_roles=user_roles,
            )
            state = session.state

            # Step 2: Engagement pre-check (Python logic — fast, reliable)
            engagement_result = await self._engagement_precheck(
                user_input=user_input,
                state=state,
                session_id=session_id,
                user_id=user_id,
                start_time=start_time,
                force_rag_only=force_rag_only,
            )

            # Always persist state after pre-check: engagement_id, engagement_user_selected,
            # and engagement_list_shown may have been set/cleared in memory above.
            # Without this save, the NEXT request reloads the old DB state and loses the flags.
            try:
                await self._session_service.save_state(
                    app_name=APP_NAME,
                    user_id=user_id,
                    session_id=session_id,
                    state=state,
                )
            except Exception as _save_err:
                logger.warning("Could not persist session state: %s", _save_err)

            if engagement_result is not None:
                # Pre-check handled the request (showed engagement list, confirmed selection)
                logger.info("Engagement pre-check handled the request for user=%s", user_id)
                return engagement_result

            # Standalone greetings / casual chat: plain LLM text only (no ADK tools).
            # The tool-enabled root agent sometimes calls select_engagement on "hello",
            # which produced incorrect "engagement switched" replies.
            if not force_rag_only and _is_standalone_greeting_or_casual(user_input):
                logger.info(
                    "Standalone greeting/casual — text-only LLM (no ADK tools) session=%s",
                    session_id,
                )
                return await self._greeting_only_response(
                    user_input, session_id, user_id, start_time
                )

            # If an engagement is already chosen, clear "awaiting list row" so the LLM
            # does not think the user still owes a number (stale engagement_list_shown).
            if state.get("engagement_id") and state.get("engagement_user_selected"):
                if state.get("engagement_list_shown"):
                    state.pop("engagement_list_shown", None)
                    try:
                        await self._session_service.save_state(
                            app_name=APP_NAME,
                            user_id=user_id,
                            session_id=session_id,
                            state=state,
                        )
                    except Exception as _cle:
                        logger.warning("Could not persist engagement_list_shown clear: %s", _cle)

            # Step 3: Pass to ADK native orchestration
            # root_agent will decide which sub-agent and tool to use
            logger.info(
                "Passing to ADK native orchestration | engagement_id=%s user_type=%s",
                state.get("engagement_id"), state.get("user_type"),
            )
            return await self._run_adk(user_input, session_id, user_id, start_time)

        except Exception as e:
            logger.error("ADK request failed: %s", e, exc_info=True)
            duration = (datetime.utcnow() - start_time).total_seconds()
            return {
                "success": False,
                "error": str(e),
                "response": (
                    "I encountered an error while processing your request. "
                    "Please try again or contact support if the issue persists."
                ),
                "routing": "error",
                "execution_result": None,
                "follow_ups": [],
                "metadata": self._meta(duration, session_id, user_id),
            }

    # ------------------------------------------------------------------
    # Step 2: Engagement pre-check (Python, not LLM)
    # ------------------------------------------------------------------

    async def _engagement_precheck(
        self,
        user_input: str,
        state: dict,
        session_id: str,
        user_id: str,
        start_time: datetime,
        force_rag_only: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Python pre-check — fast, deterministic, no LLM.

        Responsibilities:
          1. Number input → process engagement selection (ENG + CUS).
          2. ENG user, no engagement selected yet, non-greeting message
             → show engagement list (reliable Python, not LLM).
          3. Everything else (greetings, CUS, already-selected) → pass to ADK.

        CUS engagement is resolved lazily inside resource tools (_ensure_engagement).
        Greetings are always handled by root_agent (no precheck involved).
        """
        if force_rag_only:
            return None

        user_type = state.get("user_type", "CUS")

        # ------------------------------------------------------------------
        # 1. Number selection: works for both ENG and CUS when a list is shown.
        # ------------------------------------------------------------------
        selection_num = _is_engagement_number(user_input)
        if selection_num is not None and (
            state.get("engagement_list_shown")
            or (user_type == "ENG" and not state.get("engagement_user_selected"))
        ):
            logger.info("Engagement number %d typed — handling selection", selection_num)
            return await self._handle_engagement_selection(
                user_input, state, session_id, user_id, start_time
            )

        # ------------------------------------------------------------------
        # 2. ENG user — must select engagement before any resource operation.
        #    Skip if engagement is already resolved (either via Python precheck
        #    OR via ADK's select_engagement tool) OR if this is a greeting.
        # ------------------------------------------------------------------
        if (
            user_type == "ENG"
            and not state.get("engagement_user_selected")
            and not state.get("engagement_id")  # also catches ADK-side selection
            and not _is_greeting(user_input)
        ):
            logger.info(
                "[ENG] No engagement selected and non-greeting query — showing engagement list"
            )
            state.pop("engagement_id", None)
            return await self._handle_engagement(
                user_input, state, session_id, user_id, start_time
            )

        # ------------------------------------------------------------------
        # 3. Everything else → ADK (greetings, CUS, already-selected ENG).
        # ------------------------------------------------------------------
        return None

    # ------------------------------------------------------------------
    # Engagement list + selection handlers (Python, not LLM)
    # ------------------------------------------------------------------

    async def _handle_engagement(
        self, query, state, session_id, user_id, start_time
    ) -> Dict[str, Any]:
        """
        Fetch engagements from API and show a numbered list for the user to pick from.
        Works for both ENG users (always shown) and CUS users with multiple engagements.
        """
        from app.services.api_executor_service import api_executor_service

        auth_token = state.get("auth_token")
        user_type = state.get("user_type", "CUS")

        engagements = await api_executor_service.get_engagements_list(
            auth_token=auth_token, user_id=user_id
        )

        if not engagements:
            logger.error("[%s] No engagements returned for user=%s", user_type, user_id)
            duration = (datetime.utcnow() - start_time).total_seconds()
            return {
                "success": False,
                "response": (
                    "No engagements found for your account. "
                    "Please contact your administrator."
                ),
                "routing": "engagement",
                "execution_result": None,
                "follow_ups": [],
                "metadata": self._meta(duration, session_id, user_id),
            }

        logger.info(
            "[%s] Presenting %d engagements to user=%s for selection",
            user_type, len(engagements), user_id,
        )

        lines = ["## Your Engagements\n"]
        lines.append("You have access to the following engagements. Please select one:\n")
        lines.append("| # | Engagement Name | ID |")
        lines.append("|---|----------------|-----|")
        for i, eng in enumerate(engagements, 1):
            name = eng.get("engagementName") or eng.get("name", "Unknown")
            eng_id = eng.get("id", "?")
            lines.append(f"| {i} | {name} | {eng_id} |")
        lines.append(
            "\nReply with the **number** (e.g. `1`) to select an engagement "
            "and I will fetch your resources in that engagement."
        )

        # Mark that we showed the list so we know to handle numeric replies as selections
        state["engagement_list_shown"] = True

        duration = (datetime.utcnow() - start_time).total_seconds()
        return {
            "success": True,
            "response": "\n".join(lines),
            "routing": "engagement",
            "execution_result": {
                "success": True,
                "data": {"engagements": engagements, "awaiting_selection": True},
            },
            "follow_ups": [f"Select engagement {i}" for i in range(1, min(4, len(engagements) + 1))],
            "metadata": {
                **self._meta(duration, session_id, user_id),
                "awaiting_engagement_selection": True,
                "engagement_count": len(engagements),
            },
        }

    async def _handle_engagement_selection(
        self, query, state, session_id, user_id, start_time
    ) -> Dict[str, Any]:
        """Process user's engagement number reply and store the selection. Works for ENG and CUS."""
        from app.services.api_executor_service import api_executor_service

        auth_token = state.get("auth_token")
        user_type = state.get("user_type", "CUS")

        # Use pure-Python helper — no regex needed for a simple number
        selection_num = _is_engagement_number(query)
        if selection_num is None:
            duration = (datetime.utcnow() - start_time).total_seconds()
            return {
                "success": False,
                "response": (
                    "I couldn't understand the engagement selection. "
                    "Please reply with a number (e.g., `1`) or say "
                    "`show my engagements` to see the list again."
                ),
                "routing": "engagement",
                "execution_result": None,
                "follow_ups": ["Show me my engagements"],
                "metadata": self._meta(duration, session_id, user_id),
            }

        engagements = await api_executor_service.get_engagements_list(
            auth_token=auth_token, user_id=user_id
        )
        if not engagements:
            duration = (datetime.utcnow() - start_time).total_seconds()
            return {
                "success": False,
                "response": "Could not retrieve engagements. Please try again.",
                "routing": "engagement",
                "execution_result": None,
                "follow_ups": ["Show me my engagements"],
                "metadata": self._meta(duration, session_id, user_id),
            }

        if 1 <= selection_num <= len(engagements):
            selected = engagements[selection_num - 1]
        else:
            duration = (datetime.utcnow() - start_time).total_seconds()
            return {
                "success": False,
                "response": (
                    f"Invalid selection: **{selection_num}**. "
                    f"Please pick a number between 1 and {len(engagements)}."
                ),
                "routing": "engagement",
                "execution_result": None,
                "follow_ups": ["Show me my engagements"],
                "metadata": self._meta(duration, session_id, user_id),
            }

        eng_id = selected.get("id")
        eng_name = selected.get("engagementName") or selected.get("name", "Unknown")

        # Persist in api_executor_service session cache
        await api_executor_service.set_engagement_id(
            user_id=user_id, engagement_id=eng_id, engagement_data=selected
        )
        # Persist in ADK session state
        state["engagement_id"] = eng_id
        state["engagement_user_selected"] = True
        state.pop("engagement_list_shown", None)  # clear the "waiting for selection" flag

        logger.info(
            "[%s] User selected engagement_id=%s name='%s' for user=%s",
            user_type, eng_id, eng_name, user_id,
        )

        duration = (datetime.utcnow() - start_time).total_seconds()
        return {
            "success": True,
            "response": (
                f"## Engagement Selected\n\n"
                f"You are now working with **{eng_name}** (ID: `{eng_id}`).\n\n"
                f"You can now ask me to list and manage resources in this engagement."
            ),
            "routing": "engagement",
            "execution_result": {
                "success": True,
                "data": {"engagement_id": eng_id, "engagement_name": eng_name},
            },
            "follow_ups": [
                "Show me all Kubernetes clusters",
                "List my virtual machines",
                "List all firewalls",
            ],
            "metadata": self._meta(duration, session_id, user_id),
        }

    # ------------------------------------------------------------------
    # Step 3: ADK native orchestration
    # ------------------------------------------------------------------

    async def _run_adk(
        self, user_input: str, session_id: str, user_id: str, start_time: datetime
    ) -> Dict[str, Any]:
        """
        Run the query through ADK's native agent loop.

        The root_agent will:
        1. Read the user query
        2. Decide which sub-agent to delegate to (rag_agent / resource_agent / engagement_agent)
        3. The sub-agent calls the appropriate tool(s)
        4. ADK collects tool results and generates the final response
        """
        content = types.Content(role="user", parts=[types.Part(text=user_input)])

        final_text = ""
        tool_calls_observed = []
        sources = []
        images = []
        routing = "adk"

        logger.info("Running ADK agent loop for session=%s", session_id)

        async for event in self._runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=content,
        ):
            # Log all tool calls for observability
            if hasattr(event, "content") and event.content:
                for part in event.content.parts or []:
                    if hasattr(part, "function_call") and part.function_call:
                        fn = part.function_call
                        tool_calls_observed.append(fn.name)
                        logger.info(
                            "ADK tool call observed: %s | args=%s",
                            fn.name,
                            str(getattr(fn, "args", {}))[:200],
                        )
                    if hasattr(part, "function_response") and part.function_response:
                        fn_resp = part.function_response
                        logger.info(
                            "ADK tool response: %s | status=%s",
                            fn_resp.name,
                            str(getattr(fn_resp, "response", {}).get("status", "?"))[:50],
                        )

            # Capture the final text response
            if event.is_final_response():
                if event.content and event.content.parts:
                    final_text = "".join(
                        p.text
                        for p in event.content.parts
                        if hasattr(p, "text") and p.text
                    )
                    routing = "adk_native"
                    logger.info(
                        "ADK final response received | tools_used=%s | length=%d chars",
                        tool_calls_observed,
                        len(final_text),
                    )

        if not final_text:
            logger.warning(
                "ADK returned no final text for session=%s — using fallback", session_id
            )
            final_text = (
                "I wasn't able to generate a response. "
                "Please rephrase your question or try again."
            )

        duration = (datetime.utcnow() - start_time).total_seconds()
        logger.info(
            "ADK request completed in %.2fs | tools=%s", duration, tool_calls_observed
        )

        return {
            "success": True,
            "response": final_text,
            "routing": routing,
            "execution_result": {
                "success": True,
                "data": {"tools_used": tool_calls_observed},
            } if tool_calls_observed else None,
            "sources": sources,
            "images": images,
            "follow_ups": self._generic_follow_ups(tool_calls_observed),
            "metadata": {
                **self._meta(duration, session_id, user_id),
                "tools_used": tool_calls_observed,
            },
        }

    async def _greeting_only_response(
        self,
        user_input: str,
        session_id: str,
        user_id: str,
        start_time: datetime,
    ) -> Dict[str, Any]:
        """Short reply via plain chat completion — no ADK agents/tools."""
        from app.services.ai_service import ai_service

        system = (
            "You are Vayu Maya, Tata Communications' AI cloud assistant.\n"
            "The user sent a short greeting or casual message (not a cloud command).\n"
            "Reply in 1–2 friendly sentences.\n"
            "Do NOT say you switched, selected, or activated an engagement or subscription.\n"
            "Do NOT mention any engagement ID, name, or specific cloud resource unless the user asked.\n"
            "You may briefly mention that you can help with clusters, VMs, firewalls, load balancers, and docs."
        )
        text = ""
        try:
            text = await ai_service._call_chat_with_retries(
                user_input,
                max_tokens=220,
                temperature=0.35,
                system_message=system,
                timeout=45,
            )
        except Exception as e:
            logger.warning("Greeting-only LLM call failed: %s", e, exc_info=True)
        if not (text or "").strip():
            text = (
                "Hello! 👋 I'm **Vayu Maya**, your cloud assistant. "
                "Ask me to list clusters, VMs, firewalls, or anything about your cloud."
            )
        duration = (datetime.utcnow() - start_time).total_seconds()
        logger.info(
            "Greeting-only response done in %.2fs | session=%s", duration, session_id
        )
        return {
            "success": True,
            "response": text.strip(),
            "routing": "greeting",
            "execution_result": None,
            "sources": [],
            "images": [],
            "follow_ups": [
                "List Kubernetes clusters",
                "What can you help me with?",
            ],
            "metadata": {
                **self._meta(duration, session_id, user_id),
                "tools_used": [],
            },
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _generic_follow_ups(tool_calls: List[str]) -> List[str]:
        """Generate context-aware follow-up suggestions based on which tools were called."""
        if not tool_calls:
            return [
                "Show me all Kubernetes clusters",
                "List my virtual machines",
                "Show my engagements",
            ]

        last_tool = tool_calls[-1] if tool_calls else ""

        if "k8s" in last_tool or "cluster" in last_tool:
            return [
                "Get details of a specific cluster",
                "List my virtual machines",
                "Show me cluster metrics",
            ]
        if "vm" in last_tool or "virtual_machine" in last_tool:
            return [
                "Show me all Kubernetes clusters",
                "List all firewalls",
                "Show my engagements",
            ]
        if "firewall" in last_tool:
            return [
                "List my load balancers",
                "Show me all Kubernetes clusters",
                "Show my engagements",
            ]
        if "knowledge_base" in last_tool or "search" in last_tool:
            return [
                "Tell me more about this topic",
                "How do I do this step by step?",
                "Show me related resources",
            ]
        if "engagement" in last_tool:
            return [
                "Show me all Kubernetes clusters",
                "List my virtual machines",
                "List all firewalls",
            ]
        return [
            "Show me more details",
            "List another type of resource",
            "Show my engagements",
        ]

    # ------------------------------------------------------------------
    # Session helpers
    # ------------------------------------------------------------------

    async def _get_or_create_session(
        self, session_id, user_id, auth_token, user_type, user_roles
    ):
        """Create or retrieve the ADK session. Always updates auth_token and user_type."""
        svc = self._session_service
        session = await svc.get_session(
            app_name=APP_NAME, user_id=user_id, session_id=session_id,
        )

        if session:
            state_changed = False

            # Always refresh auth_token (Keycloak tokens rotate)
            if auth_token and session.state.get("auth_token") != auth_token:
                session.state["auth_token"] = auth_token
                state_changed = True

            if user_type:
                old_type = session.state.get("user_type")
                if old_type != user_type:
                    session.state["user_type"] = user_type
                    state_changed = True

                # If user_type changed (e.g. CUS ↔ ENG), clear stale engagement data
                if old_type and old_type != user_type:
                    session.state.pop("engagement_id", None)
                    session.state.pop("engagement_user_selected", None)
                    session.state.pop("engagement_list_shown", None)
                    logger.info(
                        "user_type changed %s→%s for session=%s — cleared stale engagement",
                        old_type, user_type, session_id,
                    )
                    try:
                        from app.services.api_executor_service import api_executor_service

                        await api_executor_service._clear_user_session(user_id=user_id)
                        logger.info(
                            "Cleared api_executor user cache after user_type change for user=%s",
                            user_id,
                        )
                    except Exception as _ce:
                        logger.debug("Could not clear api_executor cache: %s", _ce)

            # Persist updated state so ADK tool context picks up the latest auth_token
            if state_changed:
                await svc.save_state(
                    app_name=APP_NAME,
                    user_id=user_id,
                    session_id=session_id,
                    state=session.state,
                )
                logger.debug("Persisted updated state for session=%s", session_id)

            logger.debug("Reusing existing session=%s user=%s", session_id, user_id)
            return session

        # New session — clear any stale engagement cache from api_executor_service
        # so a previous session's engagement never bleeds into this fresh chat.
        try:
            from app.services.api_executor_service import api_executor_service
            await api_executor_service._clear_user_session(user_id=user_id)
            logger.info("New session=%s — cleared stale engagement cache for user=%s", session_id, user_id)
        except Exception as _ce:
            logger.debug("Could not clear engagement cache: %s", _ce)

        new_session = await svc.create_session(
            app_name=APP_NAME,
            user_id=user_id,
            session_id=session_id,
            state={
                "auth_token": auth_token or "",
                "user_type": user_type or "CUS",
                "user_id": user_id,
                "user_roles": user_roles or ["admin", "developer", "viewer"],
            },
        )
        logger.info(
            "Created new ADK session=%s user=%s type=%s", session_id, user_id, user_type
        )
        return new_session

    def _meta(self, duration, session_id, user_id):
        return {
            "request_number": self._total_requests,
            "duration_seconds": round(duration, 3),
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": session_id,
            "user_id": user_id,
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            "initialized": self._initialized,
            "initialization_time": (
                self._initialization_time.isoformat()
                if self._initialization_time else None
            ),
            "total_requests": self._total_requests,
            "framework": "google-adk",
            "architecture": "native-tool-calling",
        }


# ---------------------------------------------------------------------------
# Global singleton & getter
# ---------------------------------------------------------------------------

_adk_runner: Optional[ADKRunner] = None


def get_adk_runner(session_service=None) -> ADKRunner:
    """Get or create the global ADK runner instance."""
    global _adk_runner
    if _adk_runner is None:
        _adk_runner = ADKRunner()
        _adk_runner.initialize(session_service=session_service)
    return _adk_runner
