"""
inference.py

Standalone evaluation script. Run all tasks and print evaluation scores.

Usage:
python inference.py                            # all difficulties, 10 eps each
python inference.py --difficulty easy --episodes 5
python inference.py --seed 42 --output results.json
ANTHROPIC_API_KEY=sk-... python inference.py --difficulty hard
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.models import DifficultyLevel, EnvConfig, TriageLevel
from env.triage_env import TriageEnv
from agents.baseline_agent import BaselineAgent
from graders import GraderRegistry

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class EpisodeResult:
    episode_index: int
    seed: int
    difficulty: str
    case_id: str
    esi_ground_truth: str
    esi_predicted: str
    steps_taken: int
    elapsed_seconds: float
    total_reward: float
    scaled_score: float
    triage_accuracy: float
    documentation: float
    diagnostic_appropriateness: float
    treatment_safety: float
    time_efficiency: float
    penalty_flags: List[str]
    grader_tier: str
    grader_score: float
    grader_pass: bool
    grader_feedback_summary: str
    terminated_by: str


@dataclass
class DifficultyReport:
    difficulty: str
    n_episodes: int
    n_pass: int
    pass_rate: float
    mean_total_reward: float
    mean_scaled_score: float
    mean_grader_score: float
    mean_triage_accuracy: float
    mean_documentation: float
    mean_diagnostic_appropriateness: float
    mean_treatment_safety: float
    mean_time_efficiency: float
    critical_miss_rate: float
    allergy_violation_rate: float
    unsafe_discharge_rate: float
    mean_steps: float
    mean_elapsed_seconds: float
    episodes: List[EpisodeResult] = field(default_factory=list)


@dataclass
class EvaluationReport:
    env_id: str
    env_version: str
    agent: str
    timestamp: float
    total_episodes: int
    overall_pass_rate: float
    overall_mean_reward: float
    overall_mean_grader_score: float
    difficulty_reports: Dict[str, DifficultyReport]
    runtime_seconds: float


def run_evaluation(
    n_episodes_per_difficulty: int = 10,
    difficulties: Optional[List[str]] = None,
    base_seed: int = 0,
    verbose: bool = False,
    api_key: str = "",
) -> EvaluationReport:
    if difficulties is None:
        difficulties = ["easy", "medium", "hard"]

    eval_start = time.monotonic()

    print(f"\n{'='*70}")
    print(f"  AI Medical Triage & Clinical Documentation Environment")
    print(f"  Inference & Evaluation Run")
    print(f"{'='*70}")
    print(f"  Agent:              BaselineAgent (deterministic ESI heuristic)")
    print(f"  Episodes per tier:  {n_episodes_per_difficulty}")
    print(f"  Difficulties:       {', '.join(difficulties)}")
    print(f"  Base seed:          {base_seed}")
    print(f"  LLM judge:          "
          f"{'Enabled (ANTHROPIC_API_KEY set)' if api_key else 'Disabled (automated fallback)'}")
    print(f"{'='*70}\n")

    difficulty_reports: Dict[str, DifficultyReport] = {}
    all_results: List[EpisodeResult] = []

    for difficulty_str in difficulties:
        difficulty = DifficultyLevel(difficulty_str)
        print(f"  ┌─ Difficulty: {difficulty_str.upper()} ({n_episodes_per_difficulty} episodes)")
        print(f"  │")
        episode_results: List[EpisodeResult] = []

        for ep_idx in range(n_episodes_per_difficulty):
            seed = base_seed + ep_idx
            ep_result = _run_single_episode(
                episode_index=ep_idx, seed=seed, difficulty=difficulty,
                grader_tier=difficulty_str, verbose=verbose, api_key=api_key,
            )
            episode_results.append(ep_result)
            all_results.append(ep_result)
            status = "✓ PASS" if ep_result.grader_pass else "✗ FAIL"
            print(f"  │  Ep {ep_idx + 1:02d}/{n_episodes_per_difficulty} | "
                  f"ESI_GT={ep_result.esi_ground_truth} ESI_pred={ep_result.esi_predicted} | "
                  f"Reward={ep_result.total_reward:+.3f} | Grader={ep_result.grader_score:.3f} | {status}")

        report = _compute_difficulty_report(difficulty_str, episode_results)
        difficulty_reports[difficulty_str] = report
        print(f"  │")
        print(f"  │  ── {difficulty_str.upper()} SUMMARY ──────────────────────────────")
        print(f"  │  Pass rate:         {report.pass_rate:.1%} ({report.n_pass}/{report.n_episodes})")
        print(f"  │  Mean reward:       {report.mean_total_reward:+.4f}")
        print(f"  │  Mean grader score: {report.mean_grader_score:.4f} ({report.mean_grader_score*100:.1f}/100)")
        print(f"  │  Triage accuracy:   {report.mean_triage_accuracy:.4f}")
        print(f"  │  Documentation:     {report.mean_documentation:.4f}")
        print(f"  │  Diag appropriate:  {report.mean_diagnostic_appropriateness:.4f}")
        print(f"  │  Treatment safety:  {report.mean_treatment_safety:.4f}")
        print(f"  │  Time efficiency:   {report.mean_time_efficiency:.4f}")
        print(f"  │  Critical miss:     {report.critical_miss_rate:.1%}")
        print(f"  │  Allergy violations:{report.allergy_violation_rate:.1%}")
        print(f"  └─ Mean steps: {report.mean_steps:.1f} | Mean elapsed: {report.mean_elapsed_seconds:.1f}s")
        print()

    runtime = time.monotonic() - eval_start
    all_rewards = [r.total_reward for r in all_results]
    all_grader = [r.grader_score for r in all_results]
    all_pass = [r.grader_pass for r in all_results]

    report_obj = EvaluationReport(
        env_id=TriageEnv.ENV_ID, env_version=TriageEnv.VERSION, agent="BaselineAgent",
        timestamp=time.time(), total_episodes=len(all_results),
        overall_pass_rate=sum(all_pass) / len(all_pass) if all_pass else 0.0,
        overall_mean_reward=sum(all_rewards) / len(all_rewards) if all_rewards else 0.0,
        overall_mean_grader_score=sum(all_grader) / len(all_grader) if all_grader else 0.0,
        difficulty_reports=difficulty_reports, runtime_seconds=round(runtime, 2),
    )
    _print_overall_summary(report_obj)
    return report_obj


def _run_single_episode(episode_index, seed, difficulty, grader_tier, verbose, api_key) -> EpisodeResult:
    new_env = TriageEnv(env_config=EnvConfig(max_steps=10, deterioration_enabled=True), seed=seed)
    agent = BaselineAgent(single_step=True, verbose=verbose)
    obs = new_env.reset(difficulty=difficulty, seed=seed)
    agent.reset()
    done = False
    last_reward = None
    ep_start = time.monotonic()

    while not done:
        action = agent.act(obs)
        result = new_env.step(action)
        last_reward = result.reward
        done = result.done
        if not done and result.observation is not None:
            obs = result.observation

    elapsed = time.monotonic() - ep_start
    trajectory = new_env.get_trajectory()
    grade_result = GraderRegistry.from_string(grader_tier, config={"api_key": api_key}).grade(trajectory)
    state = new_env.state()
    esi_gt = state.get("esi_level", "UNKNOWN")
    final_action = trajectory.actions[-1] if trajectory.actions else None
    esi_pred = final_action.assigned_triage_level.value if final_action else "UNKNOWN"
    all_flags = [f for rs in trajectory.reward_signals for f in rs.penalty_flags]
    feedback_summary = " | ".join(
        line.strip() for line in grade_result.feedback_text.split("\n")[:3] if line.strip()
    )[:200]

    return EpisodeResult(
        episode_index=episode_index, seed=seed, difficulty=difficulty.value,
        case_id=trajectory.case_id, esi_ground_truth=esi_gt, esi_predicted=esi_pred,
        steps_taken=trajectory.total_steps, elapsed_seconds=round(elapsed, 3),
        total_reward=round(last_reward.total_reward, 4) if last_reward else 0.0,
        scaled_score=round(last_reward.scaled_score, 2) if last_reward else 0.0,
        triage_accuracy=round(last_reward.triage_accuracy_score, 4) if last_reward else 0.0,
        documentation=round(last_reward.documentation_score, 4) if last_reward else 0.0,
        diagnostic_appropriateness=round(last_reward.diagnostic_appropriateness_score, 4) if last_reward else 0.0,
        treatment_safety=round(last_reward.treatment_safety_score, 4) if last_reward else 0.0,
        time_efficiency=round(last_reward.time_efficiency_score, 4) if last_reward else 0.0,
        penalty_flags=list(set(all_flags)), grader_tier=grader_tier,
        grader_score=round(grade_result.total_score, 4), grader_pass=grade_result.pass_fail,
        grader_feedback_summary=feedback_summary, terminated_by=state.get("terminated_by", "unknown"),
    )


def _compute_difficulty_report(difficulty_str, episodes) -> DifficultyReport:
    n = len(episodes)
    if n == 0:
        return DifficultyReport(
            difficulty=difficulty_str, n_episodes=0, n_pass=0,
            pass_rate=0.0, mean_total_reward=0.0, mean_scaled_score=0.0,
            mean_grader_score=0.0, mean_triage_accuracy=0.0, mean_documentation=0.0,
            mean_diagnostic_appropriateness=0.0, mean_treatment_safety=0.0,
            mean_time_efficiency=0.0, critical_miss_rate=0.0,
            allergy_violation_rate=0.0, unsafe_discharge_rate=0.0,
            mean_steps=0.0, mean_elapsed_seconds=0.0,
        )

    def mean(vals): return sum(vals) / len(vals)
    def flag_rate(flag): return sum(1 for ep in episodes if flag in ep.penalty_flags) / n

    return DifficultyReport(
        difficulty=difficulty_str, n_episodes=n,
        n_pass=sum(1 for ep in episodes if ep.grader_pass),
        pass_rate=mean([1.0 if ep.grader_pass else 0.0 for ep in episodes]),
        mean_total_reward=mean([ep.total_reward for ep in episodes]),
        mean_scaled_score=mean([ep.scaled_score for ep in episodes]),
        mean_grader_score=mean([ep.grader_score for ep in episodes]),
        mean_triage_accuracy=mean([ep.triage_accuracy for ep in episodes]),
        mean_documentation=mean([ep.documentation for ep in episodes]),
        mean_diagnostic_appropriateness=mean([ep.diagnostic_appropriateness for ep in episodes]),
        mean_treatment_safety=mean([ep.treatment_safety for ep in episodes]),
        mean_time_efficiency=mean([ep.time_efficiency for ep in episodes]),
        critical_miss_rate=flag_rate("CRITICAL_MISS"),
        allergy_violation_rate=flag_rate("ALLERGY_VIOLATION"),
        unsafe_discharge_rate=flag_rate("UNSAFE_DISCHARGE"),
        mean_steps=mean([float(ep.steps_taken) for ep in episodes]),
        mean_elapsed_seconds=mean([ep.elapsed_seconds for ep in episodes]),
        episodes=episodes,
    )


def _print_overall_summary(report: EvaluationReport) -> None:
    print(f"\n{'='*70}")
    print(f"  OVERALL EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"  Env:              {report.env_id} v{report.env_version}")
    print(f"  Agent:            {report.agent}")
    print(f"  Total episodes:   {report.total_episodes}")
    print(f"  Runtime:          {report.runtime_seconds:.1f}s "
          f"({report.runtime_seconds/max(report.total_episodes,1):.2f}s/episode)")
    print()
    col_w = 14
    print(f"  {'Difficulty':<12}" + f"{'Pass Rate':>{col_w}}" + f"{'Mean Reward':>{col_w}}" +
          f"{'Grader Score':>{col_w}}" + f"{'Triage Acc':>{col_w}}" +
          f"{'Doc Score':>{col_w}}" + f"{'Safety':>{col_w}}")
    print(f"  {'-'*12}" + f"{'-'*col_w}" * 6)
    for diff_str, dr in report.difficulty_reports.items():
        print(f"  {diff_str.capitalize():<12}" + f"{dr.pass_rate:>{col_w}.1%}" +
              f"{dr.mean_total_reward:>{col_w}.4f}" + f"{dr.mean_grader_score:>{col_w}.4f}" +
              f"{dr.mean_triage_accuracy:>{col_w}.4f}" + f"{dr.mean_documentation:>{col_w}.4f}" +
              f"{dr.mean_treatment_safety:>{col_w}.4f}")
    print(f"  {'-'*12}" + f"{'-'*col_w}" * 6)
    print(f"  {'OVERALL':<12}" + f"{report.overall_pass_rate:>{col_w}.1%}" +
          f"{report.overall_mean_reward:>{col_w}.4f}" + f"{report.overall_mean_grader_score:>{col_w}.4f}")
    print(f"{'='*70}\n")
    print("  Safety Metrics by Difficulty:")
    for diff_str, dr in report.difficulty_reports.items():
        print(f"    {diff_str.capitalize():<8} | Critical Miss: {dr.critical_miss_rate:.1%} | "
              f"Allergy Violations: {dr.allergy_violation_rate:.1%} | "
              f"Unsafe Discharge: {dr.unsafe_discharge_rate:.1%}")
    print(f"\n{'='*70}")
    print(f"  EVALUATION COMPLETE")
    print(f"  Overall pass rate: {report.overall_pass_rate:.1%} | "
          f"Mean grader score: {report.overall_mean_grader_score * 100:.1f}/100")
    print(f"{'='*70}\n")


def _report_to_dict(report: EvaluationReport) -> dict:
    difficulty_reports_out: Dict[str, dict] = {}
    for diff_str, diff_report in report.difficulty_reports.items():
        dr_dict = asdict(diff_report)
        dr_dict["episodes"] = [asdict(ep) for ep in diff_report.episodes]
        difficulty_reports_out[diff_str] = dr_dict
    return {
        "env_id": report.env_id, "env_version": report.env_version, "agent": report.agent,
        "timestamp": report.timestamp, "total_episodes": report.total_episodes,
        "overall_pass_rate": report.overall_pass_rate, "overall_mean_reward": report.overall_mean_reward,
        "overall_mean_grader_score": report.overall_mean_grader_score,
        "runtime_seconds": report.runtime_seconds, "difficulty_reports": difficulty_reports_out,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AI Medical Triage evaluation.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--difficulty", type=str, choices=["easy", "medium", "hard", "all"], default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=os.environ.get("ANTHROPIC_API_KEY", ""))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    difficulties = ["easy", "medium", "hard"] if args.difficulty == "all" else [args.difficulty]
    report = run_evaluation(
        n_episodes_per_difficulty=args.episodes, difficulties=difficulties,
        base_seed=args.seed, verbose=args.verbose, api_key=args.api_key,
    )
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(_report_to_dict(report), fh, indent=2, ensure_ascii=False)
        print(f"  JSON report written to: {args.output}\n")
    sys.exit(0 if report.overall_pass_rate > 0.0 else 1)


if __name__ == "__main__":
    main()