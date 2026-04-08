from __future__ import annotations
import logging, time, uuid
from typing import Any, Dict, List, Optional, Tuple
from env.models import (
    DifficultyLevel, DispositionEnum, EnvConfig, EpisodeTrajectory,
    GeneratorConfig, PatientCase, PatientObservation, RewardConfig,
    RewardSignal, StepResult, TriageAction, TriageLevel, ValidationResult,
)
from env.patient_generator import PatientGenerator
from env.reward import RewardEngine

logger = logging.getLogger(__name__)


class TriageEnv:
    VERSION = "1.0.0"
    ENV_ID = "MedicalTriage-v1"

    def __init__(
        self,
        env_config: Optional[EnvConfig] = None,
        generator_config: Optional[GeneratorConfig] = None,
        reward_config: Optional[RewardConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.env_config = env_config or EnvConfig()
        self.generator_config = generator_config or GeneratorConfig()
        self.reward_config = reward_config or RewardConfig()
        self._master_seed = seed
        self._episode_counter: int = 0
        self._generator = PatientGenerator(config=self.generator_config, seed=seed)
        self._reward_engine = RewardEngine(config=self.reward_config)
        self._session_id: Optional[str] = None
        self._current_case: Optional[PatientCase] = None
        self._episode_step: int = 0
        self._episode_start_time: float = 0.
        self._done: bool = False
        self._terminated_by: str = "not_started"
        self._action_history: List[TriageAction] = []
        self._reward_history: List[RewardSignal] = []
        self._observation_history: List[PatientObservation] = []
        logging.basicConfig(level=self.env_config.log_level)

    def reset(
        self,
        difficulty: DifficultyLevel = DifficultyLevel.MEDIUM,
        seed: Optional[int] = None,
        esi_target: Optional[TriageLevel] = None,
    ) -> PatientObservation:
        self._episode_counter += 1
        effective_seed = self._resolve_seed(seed)
        self._generator.reseed(effective_seed)
        self._session_id = str(uuid.uuid4())
        self._episode_step = 0
        self._episode_start_time = time.monotonic()
        self._done = False
        self._terminated_by = "in_progress"
        self._action_history = []
        self._reward_history = []
        self._observation_history = []
        self._current_case = self._generator.generate(difficulty=difficulty, esi_target=esi_target)
        obs = self._current_case.to_observation(step=0, elapsed_seconds=0)
        self._observation_history.append(obs)
        return obs

    def step(self, action: TriageAction) -> StepResult:
        self._assert_episode_active()
        elapsed = self._elapsed_seconds()
        validation = self._validate_action(action)
        if not validation.valid and self.env_config.strict_action_validation:
            reward = self._zero_reward(self._episode_step)
            done = self._check_max_steps()
            if done:
                self._terminated_by = "max_steps"
                self._done = True
            return StepResult(
                observation=self._current_obs(), reward=reward,
                done=done, info={"validation_errors": validation.errors},
                validation=validation,
            )
        self._action_history.append(action)
        is_final = action.is_final or (self._episode_step + 1) >= self.env_config.max_steps
        ground_truth = self._current_case.get_ground_truth()
        reward = self._reward_engine.compute(
            action=action, ground_truth=ground_truth,
            elapsed_seconds=int(elapsed), step=self._episode_step,
            allergies=self._current_case.allergies,
            current_medication_names=[m.name for m in self._current_case.current_medications],
            is_final_step=is_final,
        )
        self._reward_history.append(reward)
        self._episode_step += 1
        done, terminated_by = self._check_terminal(action, reward)
        self._done = done
        self._terminated_by = terminated_by
        if not done and self.env_config.deterioration_enabled:
            self._apply_deterioration()
        if done:
            next_obs = None
        else:
            next_obs = self._current_case.to_observation(
                step=self._episode_step, elapsed_seconds=int(self._elapsed_seconds()))
            self._observation_history.append(next_obs)
        return StepResult(
            observation=next_obs, reward=reward, done=done,
            info=self._build_info(reward, done, terminated_by), validation=validation,
        )

    def state(self) -> Dict[str, Any]:
        if self._current_case is None:
            return {"status": "not_started", "session_id": None, "episode_counter": self._episode_counter}
        return {
            "status": "done" if self._done else "active",
            "session_id": self._session_id,
            "episode_counter": self._episode_counter,
            "step": self._episode_step,
            "max_steps": self.env_config.max_steps,
            "elapsed_seconds": int(self._elapsed_seconds()),
            "done": self._done,
            "terminated_by": self._terminated_by,
            "difficulty": self._current_case.difficulty.value,
            "esi_level": self._current_case.ground_truth.esi_level.value,
            "case_id": self._current_case.case_id,
            "cumulative_reward": self._cumulative_reward(),
            "n_actions": len(self._action_history),
            "penalty_flags_so_far": [f for rs in self._reward_history for f in rs.penalty_flags],
        }

    def render(self, mode: str = "text") -> str:
        if self._current_case is None:
            return "No active episode. Call reset() first."
        gt = self._current_case.get_ground_truth()
        lines = [
            f"{'='*60}", "  TriageEnv Episode Summary", f"{'='*60}",
            f"  Session ID:    {self._session_id}",
            f"  Case ID:       {self._current_case.case_id}",
            f"  Difficulty:    {self._current_case.difficulty.value}",
            f"  Status:        {'DONE' if self._done else 'ACTIVE'}",
            f"  Terminated by: {self._terminated_by}",
            f"  Steps taken:   {self._episode_step} / {self.env_config.max_steps}",
            f"  Elapsed:       {int(self._elapsed_seconds())}s", "",
            f"  Patient: {self._current_case.demographics.age}y "
            f"{self._current_case.demographics.sex.value} | "
            f"CC: {self._current_case.chief_complaint.text[:60]}",
            f"  Ground Truth ESI: {gt.esi_level.value}",
            f"  Ground Truth Dx:  {gt.primary_dx_name}", "",
            f"  Cumulative Reward:  {self._cumulative_reward():.4f}",
        ]
        for i, (act, rs) in enumerate(zip(self._action_history, self._reward_history)):
            lines.append(
                f"\n  Step {i}: ESI={act.assigned_triage_level.value} "
                f"| Disposition={act.disposition.value} "
                f"| Reward={rs.total_reward:.4f} | Flags={rs.penalty_flags or '[]'}"
            )
        lines.append(f"{'='*60}")
        if mode == "json":
            import json
            return json.dumps(self.state(), indent=2)
        return "\n".join(lines)

    def get_trajectory(self) -> EpisodeTrajectory:
        self._assert_case_exists()
        final_action = self._action_history[-1] if self._action_history else None
        return EpisodeTrajectory(
            case_id=self._current_case.case_id,
            patient_case=self._current_case,
            actions=list(self._action_history),
            reward_signals=list(self._reward_history),
            observations=list(self._observation_history),
            total_steps=self._episode_step,
            total_elapsed_seconds=int(self._elapsed_seconds()),
            final_disposition=final_action.disposition if final_action else None,
            terminated_by=self._terminated_by,
            cumulative_reward=self._cumulative_reward(),
        )

    def current_observation(self) -> Optional[PatientObservation]:
        if self._observation_history:
            return self._observation_history[-1]
        return None

    def close(self) -> None:
        logger.info("TriageEnv closing after %d episodes.", self._episode_counter)

    def _validate_action(self, action: TriageAction) -> ValidationResult:
        result = ValidationResult(valid=True)
        if self._current_case is None:
            result.add_error("No active patient case. Call reset() first.")
            return result
        available = {d.upper() for d in self._current_case.available_diagnostics}
        for order in action.diagnostic_orders:
            upper = order.test_name.upper().strip()
            if not any(upper in avail or avail in upper for avail in available):
                result.add_warning(f"Diagnostic '{order.test_name}' not in available_diagnostics.")
        if len(action.reasoning_chain.strip()) < 50:
            result.add_error(f"reasoning_chain too short ({len(action.reasoning_chain)} chars, min 50).")
        if action.is_final:
            if action.assigned_triage_level == TriageLevel.UNKNOWN:
                result.add_error("is_final=True requires assigned_triage_level other than UNKNOWN.")
            if action.disposition == DispositionEnum.PENDING:
                result.add_error("is_final=True requires disposition other than PENDING.")
        if not (0. <= action.confidence_score <= 1.):
            result.add_error(f"confidence_score must be in [0.0, 1.0], got {action.confidence_score}.")
        return result

    def _check_terminal(self, action: TriageAction, reward: RewardSignal) -> Tuple[bool, str]:
        if action.is_final:
            return True, "agent_final"
        if self._episode_step >= self.env_config.max_steps:
            return True, "max_steps"
        if self._elapsed_seconds() >= self.env_config.max_elapsed_seconds:
            return True, "timeout"
        if {"ALLERGY_VIOLATION", "UNSAFE_DISCHARGE"}.issubset(set(reward.penalty_flags)):
            return True, "unsafe_action"
        return False, "in_progress"

    def _check_max_steps(self) -> bool:
        return self._episode_step >= self.env_config.max_steps - 1

    def _apply_deterioration(self) -> None:
        if self._current_case is None:
            return
        esi = self._current_case.ground_truth.esi_level
        if esi not in (TriageLevel.ESI_1, TriageLevel.ESI_2):
            return
        if self._episode_step % self.env_config.deterioration_interval_steps != 0:
            return
        if self._critical_intervention_applied():
            return
        v = self._current_case.vitals
        from env.models import VitalSigns as VS
        new_vitals = VS(
            heart_rate=min(250, v.heart_rate + 8),
            bp_systolic=max(50, v.bp_systolic - 6),
            bp_diastolic=max(30, v.bp_diastolic - 4),
            respiratory_rate=min(50, v.respiratory_rate + 2),
            spo2=round(max(0.60, v.spo2 - 0.02), 3),
            temperature_celsius=round(min(42., v.temperature_celsius + 0.1), 1),
            gcs=max(3, v.gcs - 1),
            pain_score=min(10, v.pain_score + 1),
        )
        self._current_case = self._current_case.model_copy(update={"vitals": new_vitals})

    def _critical_intervention_applied(self) -> bool:
        if not self._action_history:
            return False
        gt = self._current_case.get_ground_truth()
        ci_upper = {c.upper() for c in gt.critical_interventions}
        if not ci_upper:
            return True
        for action in self._action_history:
            for tx in action.treatments:
                tx_upper = tx.intervention.upper()
                if any(ci in tx_upper or tx_upper in ci for ci in ci_upper):
                    return True
        return False

    def _current_obs(self) -> PatientObservation:
        self._assert_case_exists()
        return self._current_case.to_observation(
            step=self._episode_step, elapsed_seconds=int(self._elapsed_seconds()))

    def _cumulative_reward(self) -> float:
        if not self._reward_history:
            return 0.
        return self._reward_engine.compute_trajectory_reward(self._reward_history)

    def _elapsed_seconds(self) -> float:
        if self._episode_start_time == 0.:
            return 0.
        return time.monotonic() - self._episode_start_time

    def _resolve_seed(self, episode_seed: Optional[int]) -> int:
        if episode_seed is not None:
            return episode_seed
        if self._master_seed is not None:
            return self._master_seed + self._episode_counter
        return int(time.time() * 1000) % (2**31)

    def _zero_reward(self, step: int) -> RewardSignal:
        return RewardSignal(
            total_reward=0., scaled_score=50., triage_accuracy_score=0.,
            documentation_score=0., diagnostic_appropriateness_score=0.,
            treatment_safety_score=0., time_efficiency_score=0.,
            penalty_flags=["INVALID_ACTION"], penalty_total=0., step=step,
        )

    def _build_info(self, reward: RewardSignal, done: bool, terminated_by: str) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "step": self._episode_step - 1, "done": done,
            "terminated_by": terminated_by,
            "cumulative_reward": self._cumulative_reward(),
            "elapsed_seconds": int(self._elapsed_seconds()),
        }
        if done:
            info["final_scaled_score"] = reward.scaled_score
            info["penalty_summary"] = self._reward_engine.summarise_penalties(self._reward_history)
            info["ground_truth"] = {
                "esi_level": self._current_case.ground_truth.esi_level.value,
                "primary_dx": self._current_case.ground_truth.primary_dx_name,
                "correct_disposition": self._current_case.ground_truth.correct_disposition.value,
            }
        return info

    def _assert_episode_active(self) -> None:
        if self._current_case is None:
            raise RuntimeError("No active episode. Call reset() before step().")
        if self._done:
            raise RuntimeError(
                f"Episode is already done (terminated_by='{self._terminated_by}'). "
                "Call reset() to start a new episode."
            )

    def _assert_case_exists(self) -> None:
        if self._current_case is None:
            raise RuntimeError("No patient case exists. Call reset() to start an episode.")