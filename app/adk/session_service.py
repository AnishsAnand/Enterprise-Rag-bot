"""
Custom ADK SessionService backed by PostgreSQL.

Uses the existing asyncpg connection pool from postgres_service.
Stores sessions in a dedicated table so conversation state persists
across server restarts.
"""
import json
import logging
import time
from typing import Any, Dict, List, Optional

from google.adk.sessions import BaseSessionService, Session
from google.adk.sessions.base_session_service import ListSessionsResponse
from google.adk.events import Event

logger = logging.getLogger(__name__)

SESSION_TABLE = "adk_sessions"
EVENT_TABLE = "adk_session_events"


class PostgresSessionService(BaseSessionService):
    """ADK SessionService that persists sessions to PostgreSQL."""

    def __init__(self, pool_getter):
        """
        Args:
            pool_getter: Callable that returns the asyncpg Pool instance.
                         Using a getter avoids import-time initialization issues.
        """
        super().__init__()
        self._pool_getter = pool_getter
        self._tables_created = False

    @property
    def _pool(self):
        return self._pool_getter()

    async def _ensure_tables(self):
        if self._tables_created:
            return
        pool = self._pool
        if not pool:
            logger.warning("PostgreSQL pool not available; skipping table creation.")
            return
        try:
            async with pool.acquire() as conn:
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {SESSION_TABLE} (
                        app_name   TEXT NOT NULL,
                        user_id    TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        state      JSONB DEFAULT '{{}}'::jsonb,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        updated_at TIMESTAMPTZ DEFAULT NOW(),
                        PRIMARY KEY (app_name, user_id, session_id)
                    );
                """)
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {EVENT_TABLE} (
                        id         BIGSERIAL PRIMARY KEY,
                        app_name   TEXT NOT NULL,
                        user_id    TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        event_json JSONB NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                """)
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_adk_events_session
                    ON {EVENT_TABLE} (app_name, user_id, session_id, id);
                """)
            self._tables_created = True
            logger.info("ADK session tables ensured in PostgreSQL.")
        except Exception as e:
            logger.error(f"Failed to create ADK session tables: {e}", exc_info=True)

    # ------------------------------------------------------------------
    # SessionService interface
    # ------------------------------------------------------------------

    async def create_session(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: Optional[str] = None,
        state: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Session:
        await self._ensure_tables()
        import uuid
        sid = session_id or str(uuid.uuid4())
        init_state = state or {}

        pool = self._pool
        if pool:
            try:
                async with pool.acquire() as conn:
                    await conn.execute(
                        f"""
                        INSERT INTO {SESSION_TABLE} (app_name, user_id, session_id, state)
                        VALUES ($1, $2, $3, $4::jsonb)
                        ON CONFLICT (app_name, user_id, session_id) DO UPDATE
                        SET state = EXCLUDED.state, updated_at = NOW()
                        """,
                        app_name, user_id, sid, json.dumps(init_state),
                    )
            except Exception as e:
                logger.error(f"Failed to persist session {sid}: {e}", exc_info=True)

        session = Session(
            app_name=app_name,
            user_id=user_id,
            id=sid,
            state=init_state,
        )
        return session

    async def get_session(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        **kwargs,
    ) -> Optional[Session]:
        await self._ensure_tables()
        pool = self._pool
        if not pool:
            return None

        try:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    f"SELECT state FROM {SESSION_TABLE} WHERE app_name=$1 AND user_id=$2 AND session_id=$3",
                    app_name, user_id, session_id,
                )
            if not row:
                return None

            state = json.loads(row["state"]) if row["state"] else {}

            events = await self._load_events(app_name, user_id, session_id)

            session = Session(
                app_name=app_name,
                user_id=user_id,
                id=session_id,
                state=state,
                events=events,
            )
            return session
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}", exc_info=True)
            return None

    async def list_sessions(
        self,
        *,
        app_name: str,
        user_id: Optional[str] = None,
        **kwargs,
    ) -> ListSessionsResponse:
        await self._ensure_tables()
        pool = self._pool
        if not pool:
            return ListSessionsResponse(sessions=[])
        try:
            if user_id:
                query = f"SELECT session_id, state FROM {SESSION_TABLE} WHERE app_name=$1 AND user_id=$2 ORDER BY updated_at DESC"
                args = (app_name, user_id)
            else:
                query = f"SELECT session_id, state FROM {SESSION_TABLE} WHERE app_name=$1 ORDER BY updated_at DESC"
                args = (app_name,)
            async with pool.acquire() as conn:
                rows = await conn.fetch(query, *args)
            sessions = []
            for row in rows:
                state = json.loads(row["state"]) if row["state"] else {}
                sessions.append(Session(
                    app_name=app_name,
                    user_id=user_id or "",
                    id=row["session_id"],
                    state=state,
                ))
            return ListSessionsResponse(sessions=sessions)
        except Exception as e:
            logger.error(f"Failed to list sessions: {e}", exc_info=True)
            return ListSessionsResponse(sessions=[])

    async def save_state(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        state: Dict[str, Any],
    ) -> None:
        """Persist updated session state directly to Postgres (upsert)."""
        pool = self._pool
        if not pool:
            logger.warning("PostgreSQL pool not available; cannot save state.")
            return
        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    f"""
                    INSERT INTO {SESSION_TABLE} (app_name, user_id, session_id, state, updated_at)
                    VALUES ($1, $2, $3, $4::jsonb, NOW())
                    ON CONFLICT (app_name, user_id, session_id)
                    DO UPDATE SET state = EXCLUDED.state, updated_at = NOW()
                    """,
                    app_name, user_id, session_id, json.dumps(state),
                )
            logger.debug("Saved state for session=%s user=%s", session_id, user_id)
        except Exception as e:
            logger.error(f"Failed to save state for session {session_id}: {e}", exc_info=True)

    async def delete_session(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        **kwargs,
    ) -> None:
        pool = self._pool
        if not pool:
            return
        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    f"DELETE FROM {EVENT_TABLE} WHERE app_name=$1 AND user_id=$2 AND session_id=$3",
                    app_name, user_id, session_id,
                )
                await conn.execute(
                    f"DELETE FROM {SESSION_TABLE} WHERE app_name=$1 AND user_id=$2 AND session_id=$3",
                    app_name, user_id, session_id,
                )
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}", exc_info=True)

    async def append_event(self, session: Session, event: Event) -> Event:
        """Persist a new event to the session's event log.

        IMPORTANT: ADK records tool state mutations (e.g. engagement_user_selected=True)
        as state_delta on the event's actions. We MUST apply these deltas to session.state
        before saving to Postgres; otherwise the next request loads stale state from DB
        and Python precheck won't know the engagement was selected.
        """
        # Apply any state deltas from this event (tool calls write to state_delta)
        actions = getattr(event, "actions", None)
        if actions:
            delta = getattr(actions, "state_delta", None) or {}
            for key, value in delta.items():
                if value is None:
                    # Deletion — safe dict delete
                    try:
                        del session.state[key]
                    except KeyError:
                        pass
                else:
                    session.state[key] = value
            if delta:
                logger.debug("Applied state delta keys: %s for session=%s", list(delta.keys()), session.id)

        pool = self._pool
        if pool:
            try:
                event_dict = _event_to_dict(event)
                async with pool.acquire() as conn:
                    await conn.execute(
                        f"""
                        INSERT INTO {EVENT_TABLE} (app_name, user_id, session_id, event_json)
                        VALUES ($1, $2, $3, $4::jsonb)
                        """,
                        session.app_name, session.user_id, session.id,
                        json.dumps(event_dict),
                    )
                    if session.state:
                        await conn.execute(
                            f"""
                            UPDATE {SESSION_TABLE} SET state=$1::jsonb, updated_at=NOW()
                            WHERE app_name=$2 AND user_id=$3 AND session_id=$4
                            """,
                            json.dumps(dict(session.state)),
                            session.app_name, session.user_id, session.id,
                        )
            except Exception as e:
                logger.error(f"Failed to append event: {e}", exc_info=True)

        session.events.append(event)
        return event

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _load_events(self, app_name: str, user_id: str, session_id: str) -> List[Event]:
        pool = self._pool
        if not pool:
            return []
        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    f"""
                    SELECT event_json FROM {EVENT_TABLE}
                    WHERE app_name=$1 AND user_id=$2 AND session_id=$3
                    ORDER BY id ASC LIMIT 200
                    """,
                    app_name, user_id, session_id,
                )
            events = []
            for row in rows:
                event_data = json.loads(row["event_json"]) if row["event_json"] else {}
                event = _dict_to_event(event_data)
                if event:
                    events.append(event)
            return events
        except Exception as e:
            logger.error(f"Failed to load events: {e}", exc_info=True)
            return []


def _event_to_dict(event: Event) -> dict:
    """Serialize an ADK Event to a JSON-safe dict."""
    try:
        d = {
            "author": getattr(event, "author", ""),
            "invocation_id": getattr(event, "invocation_id", ""),
            "timestamp": time.time(),
        }
        if hasattr(event, "content") and event.content:
            parts_data = []
            for part in (event.content.parts or []):
                if hasattr(part, "text") and part.text:
                    parts_data.append({"text": part.text})
                elif hasattr(part, "function_call") and part.function_call:
                    parts_data.append({"function_call": str(part.function_call)})
                elif hasattr(part, "function_response") and part.function_response:
                    parts_data.append({"function_response": str(part.function_response)})
            d["content"] = {
                "role": getattr(event.content, "role", ""),
                "parts": parts_data,
            }
        return d
    except Exception:
        return {"author": "", "timestamp": time.time()}


def _dict_to_event(data: dict) -> Optional[Event]:
    """Deserialize a dict back to an ADK Event (best-effort)."""
    try:
        from google.genai import types

        content = None
        if "content" in data and data["content"]:
            parts = []
            for p in data["content"].get("parts", []):
                if "text" in p:
                    parts.append(types.Part(text=p["text"]))
            if parts:
                content = types.Content(
                    role=data["content"].get("role", "model"),
                    parts=parts,
                )

        return Event(
            author=data.get("author", ""),
            invocation_id=data.get("invocation_id", ""),
            content=content,
        )
    except Exception:
        return None
