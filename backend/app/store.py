from __future__ import annotations

from uuid import uuid4
from .schemas import SessionConfig, SessionState


class InMemoryStore:
    def __init__(self):
        self.sessions: dict[str, SessionState] = {}

    def create(self, capacity_k: int = 5, preserve_conflicts: bool = True) -> SessionState:
        session_id = uuid4().hex[:12]
        state = SessionState(
            session_id=session_id,
            config=SessionConfig(capacity_k=capacity_k, preserve_conflicts=preserve_conflicts),
        )
        self.sessions[session_id] = state
        return state

    def get(self, session_id: str) -> SessionState:
        if session_id not in self.sessions:
            raise KeyError(session_id)
        return self.sessions[session_id]

    def reset(self, session_id: str) -> SessionState:
        old = self.get(session_id)
        state = SessionState(session_id=session_id, config=old.config)
        self.sessions[session_id] = state
        return state


store = InMemoryStore()
