"""
server/api_server.py

FastAPI application for the AI Medical Triage & Clinical Documentation
Environment. Exposes POST /env/reset, POST /env/step, GET /env/state,
GET /env/observation, POST /grade/trajectory, GET /grade/leaderboard,
GET /health.
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from env.models import (
    DifficultyLevel, EnvConfig, EpisodeTrajectory, GeneratorConfig,
    PatientObservation, RewardConfig, RewardSignal, StepResult,
    TriageAction, TriageLevel, ValidationResult,
)
from env.triage_env import TriageEnv
from graders.base_grader import GradeResult
from graders import GraderRegistry
from server.middleware import (
    ErrorNormalizationMiddleware, LoggingMiddleware, RequestIDMiddleware,
)
from server.rate_limiter import RateLimiter
from server.session_store import SessionState, SessionStore

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

_SESSION_STORE: Optional[SessionStore] = None
_RATE_LIMITER: Optional[RateLimiter] = None
_LEADERBOARD: List[Dict[str, Any]] = []
_SERVER_START_TIME: float = time.monotonic()

SESSION_TTL = int(os.environ.get("SESSION_TTL_SECONDS", "1800"))
MAX_SESSIONS = int(os.environ.get("MAX_SESSIONS", "100"))
RATE_LIMIT_RPM = int(os.environ.get("RATE_LIMIT_RPM", "60"))
RATE_LIMITING_ENABLED = os.environ.get("RATE_LIMITING_ENABLED", "true").lower() == "true"
ENV_NAME = os.environ.get("ENV", "production")
MAX_LEADERBOARD_ENTRIES = int(os.environ.get("MAX_LEADERBOARD_ENTRIES", "500"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _SESSION_STORE, _RATE_LIMITER, _SERVER_START_TIME
    logger.info("TriageEnv API Server starting up (env=%s)", ENV_NAME)
    _SESSION_STORE = SessionStore(ttl_seconds=SESSION_TTL, max_sessions=MAX_SESSIONS)
    _RATE_LIMITER = RateLimiter(requests_per_minute=RATE_LIMIT_RPM, enabled=RATE_LIMITING_ENABLED)
    _SERVER_START_TIME = time.monotonic()
    logger.info("Startup complete | max_sessions=%d | session_ttl=%ds | rpm=%d",
                MAX_SESSIONS, SESSION_TTL, RATE_LIMIT_RPM)
    yield
    if _SESSION_STORE:
        destroyed = _SESSION_STORE.evict_all()
        logger.info("Shutdown: destroyed %d active sessions.", destroyed)


def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Medical Triage & Clinical Documentation Environment",
        description="OpenEnv-compatible API for evaluating AI agents on emergency triage.",
        version=TriageEnv.VERSION,
        docs_url="/docs", redoc_url="/redoc", openapi_url="/openapi.json",
        lifespan=lifespan,
    )
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False,
                       allow_methods=["*"], allow_headers=["*"])
    app.add_middleware(ErrorNormalizationMiddleware)
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(RequestIDMiddleware)
    app.include_router(_env_router())
    app.include_router(_grade_router())
    app.include_router(_health_router())
    return app


def _get_session_store() -> SessionStore:
    if _SESSION_STORE is None:
        raise RuntimeError("SessionStore not initialised.")
    return _SESSION_STORE


def _get_rate_limiter() -> RateLimiter:
    if _RATE_LIMITER is None:
        raise RuntimeError("RateLimiter not initialised.")
    return _RATE_LIMITER


def _resolve_client_key(request: Request) -> str:
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return f"bearer:{auth[7:71]}"
    api_key = request.headers.get("X-API-Key", "")
    if api_key:
        return f"apikey:{api_key[:64]}"
    client_ip = (
        request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
        or (request.client.host if request.client else "unknown")
    )
    return f"ip:{client_ip}"


def _check_rate_limit(request: Request) -> None:
    limiter = _get_rate_limiter()
    allowed, retry_after = limiter.check(_resolve_client_key(request))
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "rate_limit_exceeded",
                "message": f"Rate limit exceeded. Maximum {RATE_LIMIT_RPM} requests/minute.",
                "retry_after_seconds": retry_after,
            },
            headers={"Retry-After": str(int(retry_after) + 1)},
        )


def _require_session(
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID"),
    store: SessionStore = Depends(_get_session_store),
) -> SessionState:
    if not x_session_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "missing_session_id", "message": "X-Session-ID header is required."},
        )
    session = store.get(x_session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "session_not_found",
                "message": (
                    f"Session '{x_session_id}' not found or expired. "
                    "Call POST /env/reset to start a new session."
                ),
            },
        )
    return session


class ResetRequest(BaseModel):
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM
    seed: Optional[int] = Field(None)
    esi_target: Optional[TriageLevel] = Field(None)
    env_config: Optional[EnvConfig] = None
    generator_config: Optional[GeneratorConfig] = None
    reward_config: Optional[RewardConfig] = None


class ResetResponse(BaseModel):
    session_id: str
    observation: PatientObservation
    difficulty: str
    env_id: str = TriageEnv.ENV_ID
    env_version: str = TriageEnv.VERSION


class StepRequest(BaseModel):
    action: TriageAction


class StepResponse(BaseModel):
    session_id: str
    observation: Optional[PatientObservation]
    reward: RewardSignal
    done: bool
    info: Dict[str, Any]
    validation: ValidationResult
    step: int


class StateResponse(BaseModel):
    session_id: str
    status: str
    step: int
    max_steps: int
    elapsed_seconds: int
    done: bool
    terminated_by: str
    difficulty: str
    esi_level: str
    case_id: str
    cumulative_reward: float
    n_actions: int
    penalty_flags_so_far: List[str]


class GradeRequest(BaseModel):
    session_id: str
    grader_tier: str = Field("medium")


class GradeResponse(BaseModel):
    session_id: str
    grade_result: GradeResult
    trajectory_id: str


class LeaderboardEntry(BaseModel):
    rank: int
    session_id: str
    trajectory_id: str
    case_id: str
    grader_tier: str
    total_score: float
    scaled_score: float
    pass_fail: bool
    difficulty: str
    graded_at: float


class HealthResponse(BaseModel):
    status: str
    env_id: str
    version: str
    uptime_seconds: float
    active_sessions: int
    max_sessions: int
    leaderboard_entries: int
    environment: str


def _env_router():
    from fastapi import APIRouter
    router = APIRouter(prefix="/env", tags=["Environment"])

    @router.post("/reset", response_model=ResetResponse, status_code=status.HTTP_201_CREATED,
                 summary="Reset environment and start a new episode")
    async def reset(
        body: ResetRequest, request: Request,
        store: SessionStore = Depends(_get_session_store),
        _rl: None = Depends(_check_rate_limit),
        x_session_id: Optional[str] = Header(None, alias="X-Session-ID"),
    ) -> ResetResponse:
        if x_session_id:
            store.destroy(x_session_id)
        try:
            session = store.create_session(
                difficulty=body.difficulty, seed=body.seed,
                env_config=body.env_config, generator_config=body.generator_config,
                reward_config=body.reward_config,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                                detail={"error": "capacity_exceeded", "message": str(exc)})
        try:
            observation = session.env.reset(
                difficulty=body.difficulty, seed=body.seed, esi_target=body.esi_target)
        except Exception as exc:
            store.destroy(session.session_id)
            logger.exception("TriageEnv.reset() failed: %s", exc)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail={"error": "env_reset_failed", "message": str(exc)})
        session.difficulty = body.difficulty
        logger.info("Session created: %s | difficulty=%s | seed=%s",
                    session.session_id, body.difficulty.value, body.seed)
        return ResetResponse(session_id=session.session_id, observation=observation,
                             difficulty=body.difficulty.value)

    @router.post("/step", response_model=StepResponse,
                 summary="Submit a triage action and advance the episode")
    async def step(
        body: StepRequest, request: Request,
        session: SessionState = Depends(_require_session),
        _rl: None = Depends(_check_rate_limit),
    ) -> StepResponse:
        if session.done:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                                detail={"error": "episode_done",
                                        "message": "Episode already ended. Call POST /env/reset."})
        try:
            result: StepResult = session.env.step(body.action)
        except RuntimeError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                                detail={"error": "step_error", "message": str(exc)})
        except Exception as exc:
            logger.exception("TriageEnv.step() failed: %s", exc)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail={"error": "step_failed", "message": str(exc)})
        session.step_count += 1
        session.done = result.done
        session.touch()
        current_step = session.env.state().get("step", session.step_count)
        return StepResponse(session_id=session.session_id, observation=result.observation,
                            reward=result.reward, done=result.done, info=result.info,
                            validation=result.validation, step=current_step)

    @router.get("/state", response_model=StateResponse,
                summary="Return current session metadata (no side effects)")
    async def state(
        request: Request,
        session: SessionState = Depends(_require_session),
        _rl: None = Depends(_check_rate_limit),
    ) -> StateResponse:
        env_state = session.env.state()
        return StateResponse(
            session_id=session.session_id,
            status=env_state.get("status", "unknown"),
            step=env_state.get("step", 0),
            max_steps=env_state.get("max_steps", 10),
            elapsed_seconds=env_state.get("elapsed_seconds", 0),
            done=env_state.get("done", False),
            terminated_by=env_state.get("terminated_by", "unknown"),
            difficulty=env_state.get("difficulty", "unknown"),
            esi_level=env_state.get("esi_level", "unknown"),
            case_id=env_state.get("case_id", ""),
            cumulative_reward=env_state.get("cumulative_reward", 0.0),
            n_actions=env_state.get("n_actions", 0),
            penalty_flags_so_far=env_state.get("penalty_flags_so_far", []),
        )

    @router.get("/observation", response_model=Optional[PatientObservation],
                summary="Return the current patient observation without advancing")
    async def observation(
        request: Request,
        session: SessionState = Depends(_require_session),
        _rl: None = Depends(_check_rate_limit),
    ) -> Optional[PatientObservation]:
        return session.env.current_observation()

    return router


def _grade_router():
    from fastapi import APIRouter
    router = APIRouter(prefix="/grade", tags=["Grading"])

    @router.post("/trajectory", response_model=GradeResponse,
                 summary="Grade a completed episode trajectory")
    async def grade_trajectory(
        body: GradeRequest, request: Request,
        store: SessionStore = Depends(_get_session_store),
        _rl: None = Depends(_check_rate_limit),
    ) -> GradeResponse:
        session = store.get(body.session_id)
        if session is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail={"error": "session_not_found",
                                        "message": f"Session '{body.session_id}' not found."})
        trajectory: EpisodeTrajectory = session.env.get_trajectory()
        if not trajectory.actions:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                                detail={"error": "no_actions",
                                        "message": "Cannot grade a trajectory with no actions."})
        try:
            grader = GraderRegistry.from_string(
                body.grader_tier,
                config={"api_key": os.environ.get("ANTHROPIC_API_KEY", "")},
            )
            grade_result: GradeResult = grader.grade(trajectory)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail={"error": "invalid_grader_tier", "message": str(exc)})
        except Exception as exc:
            logger.exception("Grader failed: %s", exc)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail={"error": "grader_failed", "message": str(exc)})
        _update_leaderboard(session_id=body.session_id, trajectory=trajectory,
                            grade_result=grade_result, difficulty=session.difficulty)
        return GradeResponse(session_id=body.session_id, grade_result=grade_result,
                             trajectory_id=trajectory.trajectory_id)

    @router.get("/leaderboard", response_model=List[LeaderboardEntry],
                summary="Return top scores")
    async def leaderboard(
        request: Request,
        tier: Optional[str] = None,
        limit: int = 20,
        _rl: None = Depends(_check_rate_limit),
    ) -> List[LeaderboardEntry]:
        entries = list(_LEADERBOARD)
        if tier:
            entries = [e for e in entries if e.get("grader_tier") == tier.lower()]
        entries.sort(key=lambda e: e["total_score"], reverse=True)
        entries = entries[:max(1, min(limit, 100))]
        return [
            LeaderboardEntry(
                rank=i + 1, session_id=e["session_id"],
                trajectory_id=e["trajectory_id"], case_id=e["case_id"],
                grader_tier=e["grader_tier"], total_score=e["total_score"],
                scaled_score=e["scaled_score"], pass_fail=e["pass_fail"],
                difficulty=e["difficulty"], graded_at=e["graded_at"],
            )
            for i, e in enumerate(entries)
        ]

    return router


def _health_router():
    from fastapi import APIRouter
    router = APIRouter(tags=["Observability"])

    @router.get("/health", response_model=HealthResponse, summary="Health check")
    async def health(store: SessionStore = Depends(_get_session_store)) -> HealthResponse:
        return HealthResponse(
            status="ok", env_id=TriageEnv.ENV_ID, version=TriageEnv.VERSION,
            uptime_seconds=round(time.monotonic() - _SERVER_START_TIME, 1),
            active_sessions=store.active_count(), max_sessions=MAX_SESSIONS,
            leaderboard_entries=len(_LEADERBOARD), environment=ENV_NAME,
        )

    @router.get("/", include_in_schema=False)
    async def root() -> Dict[str, str]:
        return {"service": "AI Medical Triage & Clinical Documentation Environment",
                "version": TriageEnv.VERSION, "docs": "/docs", "health": "/health"}

    return router


def _update_leaderboard(
    session_id: str, trajectory: EpisodeTrajectory,
    grade_result: GradeResult, difficulty: DifficultyLevel,
) -> None:
    entry: Dict[str, Any] = {
        "session_id": session_id,
        "trajectory_id": trajectory.trajectory_id,
        "case_id": trajectory.case_id,
        "grader_tier": grade_result.grader_tier.value,
        "total_score": grade_result.total_score,
        "scaled_score": grade_result.scaled_score(),
        "pass_fail": grade_result.pass_fail,
        "difficulty": difficulty.value,
        "graded_at": time.time(),
    }
    _LEADERBOARD.append(entry)
    if len(_LEADERBOARD) > MAX_LEADERBOARD_ENTRIES:
        _LEADERBOARD.sort(key=lambda e: e["total_score"], reverse=True)
        _LEADERBOARD.pop()


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    host = os.environ.get("HOST", "0.0.0.0")
    log_level = os.environ.get("LOG_LEVEL", "info").lower()
    uvicorn.run("server.api_server:app", host=host, port=port, workers=1,
                log_level=log_level, timeout_keep_alive=30, access_log=False)