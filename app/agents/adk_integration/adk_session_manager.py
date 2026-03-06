"""
ADK Session Manager
===================
Wraps Google ADK's InMemorySessionService to provide structured session state
while staying backward-compatible with the existing conversation_state_manager.

Design contract
---------------
- Existing LangChain agents call conversation_state_manager as before → NO changes.
- ADK layer sits ABOVE them: richer state, TTL, serialisable snapshots.
- State is kept in sync bidirectionally so both layers see the same data.

Swap InMemorySessionService → DatabaseSessionService in production:
    from google.adk.sessions import DatabaseSessionService
    service = DatabaseSessionService(db_url=os.getenv("DATABASE_URL"))
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

APP_NAME = "vayu_maya"          # ADK application namespace
_ADK_AVAILABLE = False          # flipped to True if google-adk is installed

# ---------------------------------------------------------------------------
# Optional import – graceful degradation when google-adk is not yet installed
# ---------------------------------------------------------------------------
try:
    from google.adk.sessions import InMemorySessionService, Session   # type: ignore
    _ADK_AVAILABLE = True
    logger.info("✅ google-adk found – ADK session management ENABLED")
except ImportError:
    logger.warning(
        "⚠️  google-adk not installed. "
        "ADKSessionManager will fall back to ConversationStateManager only. "
        "Install with:  pip install google-adk"
    )
    InMemorySessionService = None   # type: ignore
    Session = None                  # type: ignore


# ---------------------------------------------------------------------------
# Thin dict-based fallback session so the rest of the code never breaks
# ---------------------------------------------------------------------------
class _FallbackSession:
    """Minimal session shim used when google-adk is not installed."""
    def __init__(self, session_id: str, user_id: str, state: Dict[str, Any]) -> None:
        self.id = session_id
        self.user_id = user_id
        self.state = dict(state)


class ADKSessionManager:
    """
    Hybrid session manager.

    Responsibilities
    ----------------
    1. Create / resume ADK sessions (InMemorySessionService or fallback).
    2. Expose a clean async API: get_or_create, update, get_state, delete.
    3. Bidirectional sync with the existing ConversationStateManager so that
       all existing LangChain agents continue to work without modification.
    """

    def __init__(self) -> None:
        if _ADK_AVAILABLE:
            self._service: Any = InMemorySessionService()
        else:
            # Plain dict fallback – same interface
            self._service = None
            self._fallback_store: Dict[str, _FallbackSession] = {}

        self._user_prefix = "vayu"
        logger.info(
            "✅ ADKSessionManager ready  (adk_native=%s)", _ADK_AVAILABLE
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_or_create_session(
        self,
        session_id: str,
        user_id: str,
        initial_state: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Get an existing ADK session or create a new one.

        On first creation the session is bootstrapped from ConversationState
        (if one already exists) so the ADK layer immediately has full context.
        """
        adk_uid = f"{self._user_prefix}_{user_id}"
        merged_initial = {**(self._pull_conversation_state(session_id) or {}),
                          **(initial_state or {})}

        if _ADK_AVAILABLE:
            session = await self._service.get_session(
                app_name=APP_NAME, user_id=adk_uid, session_id=session_id
            )
            if session is None:
                session = await self._service.create_session(
                    app_name=APP_NAME,
                    user_id=adk_uid,
                    session_id=session_id,
                    state=merged_initial,
                )
                logger.info("📦 ADK session created: %s", session_id)
            else:
                logger.debug("🔄 ADK session resumed: %s", session_id)
            return session

        # Fallback
        if session_id not in self._fallback_store:
            self._fallback_store[session_id] = _FallbackSession(
                session_id, adk_uid, merged_initial
            )
            logger.info("📦 Fallback session created: %s", session_id)
        return self._fallback_store[session_id]

    async def update_session_state(
        self,
        session_id: str,
        user_id: str,
        updates: Dict[str, Any],
    ) -> None:
        """
        Merge `updates` into the ADK session state, then sync to ConversationState.
        Keeping both layers in sync is the key guarantee of this class.
        """
        session = await self.get_or_create_session(session_id, user_id)

        if _ADK_AVAILABLE:
            new_state = {**session.state, **updates}
            await self._service.update_session(
                app_name=APP_NAME,
                user_id=f"{self._user_prefix}_{user_id}",
                session_id=session_id,
                state=new_state,
            )
        else:
            session.state.update(updates)

        # Sync to ConversationState so existing LangChain agents see the change
        self._push_conversation_state(session_id, user_id, updates)
        logger.debug("💾 Session state updated: %s | keys=%s", session_id, list(updates))

    async def get_state(self, session_id: str, user_id: str) -> Dict[str, Any]:
        """Return the current session state as a plain dict."""
        session = await self.get_or_create_session(session_id, user_id)
        return dict(session.state)

    async def delete_session(self, session_id: str, user_id: str) -> bool:
        """Remove the ADK session AND the matching ConversationState."""
        try:
            if _ADK_AVAILABLE:
                await self._service.delete_session(
                    app_name=APP_NAME,
                    user_id=f"{self._user_prefix}_{user_id}",
                    session_id=session_id,
                )
            else:
                self._fallback_store.pop(session_id, None)
        except Exception as exc:
            logger.warning("ADK session delete warning: %s", exc)

        # Delegate legacy cleanup to ConversationStateManager
        try:
            from app.agents.state.conversation_state import conversation_state_manager
            return conversation_state_manager.delete_session(session_id)
        except Exception:
            return True

    def list_active_sessions(self) -> List[str]:
        """Return known active session IDs (from ConversationStateManager)."""
        try:
            from app.agents.state.conversation_state import conversation_state_manager
            return conversation_state_manager.get_active_sessions()
        except Exception:
            return list(self._fallback_store)

    # ------------------------------------------------------------------
    # Sync helpers – bridge ADK ↔ ConversationState
    # ------------------------------------------------------------------

    def _pull_conversation_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Read existing ConversationState into a flat dict for ADK bootstrapping."""
        try:
            from app.agents.state.conversation_state import conversation_state_manager
            state = conversation_state_manager.get_session(session_id)
            return state.to_dict() if state is not None else None
        except Exception:
            return None

    def _push_conversation_state(
        self,
        session_id: str,
        user_id: str,
        updates: Dict[str, Any],
    ) -> None:
        """
        Apply relevant ADK state keys back to ConversationState.

        Only well-known keys are mapped to avoid polluting ConversationState
        with ADK-internal bookkeeping.
        """
        try:
            from app.agents.state.conversation_state import conversation_state_manager
            state = conversation_state_manager.get_session(session_id)
            if state is None:
                state = conversation_state_manager.create_session(session_id, user_id)
            if state is None:
                return

            # Explicit key mapping  (adk_key → ConversationState attr)
            FIELD_MAP = {
                "last_intent":      "last_intent",
                "last_resource":    "current_resource",
                "last_operation":   "current_operation",
                "extracted_params": "extracted_params",
            }
            for adk_key, attr in FIELD_MAP.items():
                if adk_key in updates and hasattr(state, attr):
                    try:
                        setattr(state, attr, updates[adk_key])
                    except Exception:
                        pass
        except Exception as exc:
            logger.debug("_push_conversation_state skipped: %s", exc)


# Singleton – imported throughout the integration layer
adk_session_manager = ADKSessionManager()