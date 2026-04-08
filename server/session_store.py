"""
server/session_store.py

Thread-safe in-memory session store for tracking active TriageEnv episodes.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from env.models import DifficultyLevel, EnvConfig, GeneratorConfig, RewardConfig
from env.triage_env import TriageEnv


@dataclass
class SessionState:
    session_id: str
    env: TriageEnv
    difficulty: DifficultyLevel
    created_at: float = field(default_factory=time.monotonic)
    last_accessed_at: float = field(default_factory=time.monotonic)
    step_count: int = 0
    done: bool = False

    def touch(self) -> None:
        self.last_accessed_at = time.monotonic()

    def age_seconds(self) -> float:
        return time.monotonic() - self.created_at

    def idle_seconds(self) -> float:
        return time.monotonic() - self.last_accessed_at


class SessionStore:
    def __init__(self, ttl_seconds: int = 1800, max_sessions: int = 100) -> None:
        self._ttl = ttl_seconds
        self._max_sessions = max_sessions
        self._sessions: Dict[str, SessionState] = {}
        self._lock = threading.Lock()

    def create_session(
        self,
        difficulty: DifficultyLevel = DifficultyLevel.MEDIUM,
        seed: Optional[int] = None,
        env_config: Optional[EnvConfig] = None,
        generator_config: Optional[GeneratorConfig] = None,
        reward_config: Optional[RewardConfig] = None,
    ) -> SessionState:
        self._evict_expired()
        with self._lock:
            if len(self._sessions) >= self._max_sessions:
                raise RuntimeError(
                    f"Session store at maximum capacity ({self._max_sessions}). "
                    "Try again later or destroy an existing session."
                )
            session_id = str(uuid.uuid4())
            env = TriageEnv(
                env_config=env_config or EnvConfig(),
                generator_config=generator_config or GeneratorConfig(),
                reward_config=reward_config or RewardConfig(),
                seed=seed,
            )
            state = SessionState(session_id=session_id, env=env, difficulty=difficulty)
            self._sessions[session_id] = state
            return state

    def get(self, session_id: str) -> Optional[SessionState]:
        self._evict_expired()
        with self._lock:
            state = self._sessions.get(session_id)
            if state is not None:
                state.touch()
            return state

    def destroy(self, session_id: str) -> bool:
        with self._lock:
            if session_id in self._sessions:
                try:
                    self._sessions[session_id].env.close()
                except Exception:
                    pass
                del self._sessions[session_id]
                return True
            return False

    def active_count(self) -> int:
        self._evict_expired()
        with self._lock:
            return len(self._sessions)

    def session_ids(self) -> List[str]:
        self._evict_expired()
        with self._lock:
            return list(self._sessions.keys())

    def evict_all(self) -> int:
        with self._lock:
            count = len(self._sessions)
            for state in self._sessions.values():
                try:
                    state.env.close()
                except Exception:
                    pass
            self._sessions.clear()
            return count

    def _evict_expired(self) -> None:
        with self._lock:
            expired = [
                sid for sid, state in self._sessions.items()
                if state.idle_seconds() > self._ttl
            ]
            for sid in expired:
                try:
                    self._sessions[sid].env.close()
                except Exception:
                    pass
                del self._sessions[sid]