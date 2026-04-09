"""
benchmark.py

Automatic benchmark: runs N episodes and prints average reward and success rate.

Usage:
    python benchmark.py                              # baseline, 20 episodes, all difficulties
    python benchmark.py --agent llm --provider openai
    python benchmark.py --agent llm --provider claude --model claude-opus-4-6
    python benchmark.py --episodes 50 --difficulty hard
    python benchmark.py --output report.json

Environment variables (for LLM agent):
    OPENAI_API_KEY
    ANTHROPIC_API_KEY
    GEMINI_API_KEY
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EpisodeRecord:
    episode: int
    seed: int
    difficulty: str
    agent: str
    total_reward: float
    scaled_score: float
    triage_accuracy: float
    documentation: float
    treatment_safety: float
    time_efficiency: float
    penalty_flags: List[str]
    success: bool
    elapsed_seconds: float


@dataclass
class BenchmarkReport:
    agent: str
    provider: Optional[str]
    model: Optional[str]
    difficulties: List[str]
    n_episodes: int
    n_success: int
    success_rate: float
    mean_total_reward: float
    mean_scaled_score: float
    mean_triage_accuracy: float
    mean_documentation: float
    mean_treatment_safety: float
    mean_time_efficiency: float
    critical_miss_rate: float
    allergy_violation_rate: float
    unsafe_discharge_rate: float
    runtime_seconds: float
    episodes: List[EpisodeRecord] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _flag_rate(episodes: List[EpisodeRecord], flag: str) -> float:
    if not episodes:
        return 0.0
    return sum(1 for ep in episodes if flag in ep.penalty_flags) / len(episodes)


def _build_agent(agent_type: str, provider: Optional[str], model: Optional[str]):
    if agent_type == "baseline":
        from agents.baseline_agent import BaselineAgent
        return BaselineAgent(single_step=True, verbose=False)
    elif agent_type == "llm":
        from agents.llm_agent import LLMAgent
        return LLMAgent(provider=provider or "openai", model=model or None, verbose=False)
    else:
        raise ValueError(f"Unknown agent type: {agent_type!r}. Choose 'baseline' or 'llm'.")


# ---------------------------------------------------------------------------
# Single episode runner
# ---------------------------------------------------------------------------

def _run_episode(
    episode_idx: int,
    seed: int,
    difficulty: str,
    agent,
    agent_label: str,
) -> EpisodeRecord:
    from env.models import DifficultyLevel, EnvConfig
    from env.triage_env import TriageEnv

    env = TriageEnv(env_config=EnvConfig(max_steps=10, deterioration_enabled=True), seed=seed)
    obs = env.reset(difficulty=DifficultyLevel(difficulty), seed=seed)
    if hasattr(agent, "reset"):
        agent.reset()

    done = False
    last_reward = None
    ep_start = time.monotonic()

    while not done:
        action = agent.act(obs)
        result = env.step(action)
        last_reward = result.reward
        done = result.done
        if not done and result.observation is not None:
            obs = result.observation

    elapsed = time.monotonic() - ep_start
    trajectory = env.get_trajectory()
    all_flags = list({f for rs in trajectory.reward_signals for f in rs.penalty_flags})

    success = False
    scaled = 0.0
    total_r = 0.0
    triage_acc = 0.0
    doc_score = 0.0
    safety = 0.0
    time_eff = 0.0

    if last_reward is not None:
        total_r = round(last_reward.total_reward, 4)
        scaled = round(last_reward.scaled_score, 4)
        triage_acc = round(last_reward.triage_accuracy_score, 4)
        doc_score = round(last_reward.documentation_score, 4)
        safety = round(last_reward.treatment_safety_score, 4)
        time_eff = round(last_reward.time_efficiency_score, 4)
        success = scaled >= 0.6

    return EpisodeRecord(
        episode=episode_idx,
        seed=seed,
        difficulty=difficulty,
        agent=agent_label,
        total_reward=total_r,
        scaled_score=scaled,
        triage_accuracy=triage_acc,
        documentation=doc_score,
        treatment_safety=safety,
        time_efficiency=time_eff,
        penalty_flags=all_flags,
        success=success,
        elapsed_seconds=round(elapsed, 3),
    )


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    n_episodes: int = 20,
    difficulties: Optional[List[str]] = None,
    agent_type: str = "baseline",
    provider: Optional[str] = None,
    model: Optional[str] = None,
    base_seed: int = 0,
    verbose: bool = False,
) -> BenchmarkReport:
    if difficulties is None:
        difficulties = ["easy", "medium", "hard"]

    agent_label = (
        f"LLMAgent({provider or 'openai'})"
        if agent_type == "llm"
        else "BaselineAgent"
    )
    agent = _build_agent(agent_type, provider, model)

    print(f"\n{'='*65}")
    print(f"  Medical Triage Benchmark")
    print(f"{'='*65}")
    print(f"  Agent:      {agent_label}")
    print(f"  Episodes:   {n_episodes}")
    print(f"  Difficulties: {', '.join(difficulties)}")
    print(f"  Base seed:  {base_seed}")
    print(f"{'='*65}\n")

    episodes: List[EpisodeRecord] = []
    start_time = time.monotonic()

    diff_cycle = difficulties * ((n_episodes // len(difficulties)) + 1)
    diff_cycle = diff_cycle[:n_episodes]

    for i in range(n_episodes):
        seed = base_seed + i
        diff = diff_cycle[i]
        ep = _run_episode(i + 1, seed, diff, agent, agent_label)
        episodes.append(ep)

        status = "✓" if ep.success else "✗"
        flags_str = ", ".join(ep.penalty_flags) if ep.penalty_flags else "none"
        print(
            f"  {status} Ep {i+1:02d}/{n_episodes} | {diff:<6} | "
            f"seed={seed} | reward={ep.total_reward:+.4f} | "
            f"scaled={ep.scaled_score:.4f} | flags=[{flags_str}]"
        )
        if verbose:
            print(
                f"       triage={ep.triage_accuracy:.4f} doc={ep.documentation:.4f} "
                f"safety={ep.treatment_safety:.4f} time={ep.time_efficiency:.4f} "
                f"elapsed={ep.elapsed_seconds:.2f}s"
            )

    runtime = time.monotonic() - start_time
    n_success = sum(1 for ep in episodes if ep.success)

    report = BenchmarkReport(
        agent=agent_label,
        provider=provider,
        model=model,
        difficulties=difficulties,
        n_episodes=n_episodes,
        n_success=n_success,
        success_rate=_mean([1.0 if ep.success else 0.0 for ep in episodes]),
        mean_total_reward=_mean([ep.total_reward for ep in episodes]),
        mean_scaled_score=_mean([ep.scaled_score for ep in episodes]),
        mean_triage_accuracy=_mean([ep.triage_accuracy for ep in episodes]),
        mean_documentation=_mean([ep.documentation for ep in episodes]),
        mean_treatment_safety=_mean([ep.treatment_safety for ep in episodes]),
        mean_time_efficiency=_mean([ep.time_efficiency for ep in episodes]),
        critical_miss_rate=_flag_rate(episodes, "CRITICAL_MISS"),
        allergy_violation_rate=_flag_rate(episodes, "ALLERGY_VIOLATION"),
        unsafe_discharge_rate=_flag_rate(episodes, "UNSAFE_DISCHARGE"),
        runtime_seconds=round(runtime, 2),
        episodes=episodes,
    )

    _print_summary(report)
    return report


def _print_summary(r: BenchmarkReport) -> None:
    print(f"\n{'='*65}")
    print(f"  BENCHMARK RESULTS — {r.agent}")
    print(f"{'='*65}")
    print(f"  Episodes:          {r.n_episodes}")
    print(f"  Success rate:      {r.success_rate:.1%}  ({r.n_success}/{r.n_episodes})")
    print(f"  Mean total reward: {r.mean_total_reward:+.4f}")
    print(f"  Mean scaled score: {r.mean_scaled_score:.4f}  ({r.mean_scaled_score*100:.1f}/100)")
    print()
    print(f"  Triage accuracy:   {r.mean_triage_accuracy:.4f}")
    print(f"  Documentation:     {r.mean_documentation:.4f}")
    print(f"  Treatment safety:  {r.mean_treatment_safety:.4f}")
    print(f"  Time efficiency:   {r.mean_time_efficiency:.4f}")
    print()
    print(f"  Critical miss:     {r.critical_miss_rate:.1%}")
    print(f"  Allergy violations:{r.allergy_violation_rate:.1%}")
    print(f"  Unsafe discharge:  {r.unsafe_discharge_rate:.1%}")
    print()
    print(f"  Runtime:           {r.runtime_seconds:.1f}s  "
          f"({r.runtime_seconds/max(r.n_episodes,1):.2f}s/episode)")
    print(f"{'='*65}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Medical Triage benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--agent",
        choices=["baseline", "llm"],
        default="baseline",
        help="Agent type to benchmark.",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "claude", "gemini"],
        default="openai",
        help="LLM provider (only used when --agent=llm).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model override for the LLM provider (leave unset for default).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Total number of episodes to run.",
    )
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard", "all"],
        default="all",
        help="Difficulty tier(s) to cycle through.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-episode sub-metric breakdown.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write a JSON report.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    difficulties = (
        ["easy", "medium", "hard"] if args.difficulty == "all" else [args.difficulty]
    )

    report = run_benchmark(
        n_episodes=args.episodes,
        difficulties=difficulties,
        agent_type=args.agent,
        provider=args.provider if args.agent == "llm" else None,
        model=args.model,
        base_seed=args.seed,
        verbose=args.verbose,
    )

    if args.output:
        report_dict = asdict(report)
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(report_dict, fh, indent=2, ensure_ascii=False)
        print(f"  JSON report written to: {args.output}\n")

    sys.exit(0 if report.success_rate > 0.0 else 1)


if __name__ == "__main__":
    main()