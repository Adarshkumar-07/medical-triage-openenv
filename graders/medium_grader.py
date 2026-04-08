from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from env.models import DispositionEnum, EpisodeTrajectory, TriageLevel
from graders.base_grader import BaseGrader, GradeResult, GraderTier


class MediumGrader(BaseGrader):
    """
    Weighted rubric grader for MEDIUM difficulty episodes.

    Dimensions and weights:
      triage_accuracy        25%
      documentation_quality  25%
      diagnostic_workup      20%
      treatment_safety       20%
      time_to_disposition    10%

    Pass threshold: 0.70 (70/100).
    Safety violations hard-cap at 0.60.
    """

    pass_threshold: float = 0.70

    WEIGHTS: Dict[str, float] = {
        "triage_accuracy": 0.25,
        "documentation_quality": 0.25,
        "diagnostic_workup": 0.20,
        "treatment_safety": 0.20,
        "time_to_disposition": 0.10,
    }

    SAFETY_VIOLATION_CAP: float = 0.60

    TIME_THRESHOLDS: Dict[TriageLevel, int] = {
        TriageLevel.ESI_1: 60,
        TriageLevel.ESI_2: 60,
        TriageLevel.ESI_3: 120,
        TriageLevel.ESI_4: 300,
        TriageLevel.ESI_5: 300,
    }

    TIME_FLOOR: float = 0.20

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)

    def grade(self, trajectory: EpisodeTrajectory) -> GradeResult:
        actions = trajectory.actions
        ground_truth = trajectory.patient_case.get_ground_truth()
        allergies = trajectory.patient_case.allergies
        reward_signals = trajectory.reward_signals

        if not actions:
            return self._empty_trajectory_result(trajectory)

        final_action = self._best_action(actions)
        all_treatments = [tx for a in actions for tx in a.treatments]
        all_diagnostics = [d for a in actions for d in a.diagnostic_orders]
        observed_flags = self._collect_all_flags(reward_signals)

        triage_score = self._score_triage_accuracy(final_action, ground_truth)
        doc_score = self._score_documentation_quality(final_action)
        diag_score = self._compute_diagnostic_coverage(
            all_diagnostics, ground_truth.required_diagnostics, penalise_overordering=True
        )
        tx_score = self._score_treatment_safety(
            all_treatments=all_treatments, allergies=allergies,
            critical_interventions=ground_truth.critical_interventions,
            esi_level=ground_truth.esi_level,
        )
        time_score = self._score_time_efficiency(
            elapsed_seconds=trajectory.total_elapsed_seconds,
            esi_level=ground_truth.esi_level,
        )

        rubric_scores = {
            "triage_accuracy": round(triage_score, 4),
            "documentation_quality": round(doc_score, 4),
            "diagnostic_workup": round(diag_score, 4),
            "treatment_safety": round(tx_score, 4),
            "time_to_disposition": round(time_score, 4),
        }

        weighted_total = sum(self.WEIGHTS[dim] * score for dim, score in rubric_scores.items())

        safety_violated, safety_reason = self._detect_safety_violations(
            final_action, all_treatments, allergies, ground_truth
        )
        if safety_violated:
            weighted_total = min(weighted_total, self.SAFETY_VIOLATION_CAP)

        total_score = max(0.0, min(1.0, round(weighted_total, 4)))
        passed = total_score >= self.pass_threshold

        improvement_notes = self._build_improvement_notes(
            rubric_scores=rubric_scores, final_action=final_action,
            ground_truth=ground_truth, safety_violated=safety_violated,
            safety_reason=safety_reason,
        )
        feedback = self._build_feedback_text(
            rubric_scores=rubric_scores, total_score=total_score, passed=passed,
            observed_flags=observed_flags, improvement_notes=improvement_notes,
            safety_violated=safety_violated,
        )

        return GradeResult(
            grader_tier=GraderTier.MEDIUM, pass_fail=passed, total_score=total_score,
            rubric_scores=rubric_scores, feedback_text=feedback, llm_critique=None,
            penalty_flags_observed=observed_flags,
            case_id=trajectory.case_id, trajectory_id=trajectory.trajectory_id,
        )

    def _score_triage_accuracy(self, final_action, ground_truth) -> float:
        esi_score = self._compute_esi_accuracy(
            final_action.assigned_triage_level, ground_truth.esi_level
        )
        if self._is_under_triaged(final_action.assigned_triage_level, ground_truth.esi_level, threshold=2):
            esi_score = max(0.0, esi_score - 0.20)
        disposition_score = self._compute_disposition_score(
            final_action.disposition, ground_truth.correct_disposition
        )
        return round(0.70 * esi_score + 0.30 * disposition_score, 4)

    def _score_documentation_quality(self, final_action) -> float:
        note_score = self._compute_note_completeness(final_action.clinical_note)
        reasoning_score = self._compute_reasoning_quality(final_action.reasoning_chain)
        return round(0.65 * note_score + 0.35 * reasoning_score, 4)

    def _score_treatment_safety(
        self, all_treatments, allergies: List[str],
        critical_interventions: List[str], esi_level: TriageLevel,
    ) -> float:
        coverage = self._compute_treatment_coverage(all_treatments, critical_interventions)
        allergy_safe = not self._check_allergy_violation(all_treatments, allergies)
        allergy_score = 1.0 if allergy_safe else 0.0
        if critical_interventions and esi_level in (TriageLevel.ESI_1, TriageLevel.ESI_2):
            priority_score = self._score_treatment_priority(all_treatments, critical_interventions)
        else:
            priority_score = 1.0
        combined = 0.60 * coverage + 0.25 * allergy_score + 0.15 * priority_score
        return round(min(1.0, max(0.0, combined)), 4)

    def _score_treatment_priority(self, all_treatments, critical_interventions: List[str]) -> float:
        from env.models import PriorityEnum
        ci_upper = [self._normalise_test_name(c) for c in critical_interventions]
        matched_priorities = []
        for ci in ci_upper:
            for tx in all_treatments:
                tx_upper = self._normalise_test_name(tx.intervention)
                if ci in tx_upper or tx_upper in ci:
                    matched_priorities.append(1.0 if tx.priority == PriorityEnum.IMMEDIATE else 0.5)
                    break
        if not matched_priorities:
            return 0.0
        return round(sum(matched_priorities) / len(matched_priorities), 4)

    def _score_time_efficiency(self, elapsed_seconds: int, esi_level: TriageLevel) -> float:
        threshold = self.TIME_THRESHOLDS.get(esi_level, 120)
        if elapsed_seconds <= threshold:
            return 1.0
        overshoot = elapsed_seconds - threshold
        decay = (overshoot / threshold) * 0.5
        return max(self.TIME_FLOOR, round(1.0 - decay, 4))

    def _detect_safety_violations(
        self, final_action, all_treatments, allergies: List[str], ground_truth
    ) -> Tuple[bool, str]:
        if self._check_allergy_violation(all_treatments, allergies):
            return True, "Allergy violation: treatment given that contradicts a known allergy."
        unsafe = {DispositionEnum.DISCHARGE_HOME, DispositionEnum.DISCHARGE_PCP_FOLLOWUP}
        if (ground_truth.esi_level in (TriageLevel.ESI_1, TriageLevel.ESI_2)
                and final_action.disposition in unsafe):
            return True, (
                f"Unsafe discharge: {ground_truth.esi_level.value} patient "
                f"assigned {final_action.disposition.value}."
            )
        if self._is_critical_miss(final_action.assigned_triage_level, ground_truth.esi_level):
            return True, (
                f"Critical miss: life-threatening patient ({ground_truth.esi_level.value}) "
                f"assigned {final_action.assigned_triage_level.value}."
            )
        return False, ""

    def _build_improvement_notes(
        self, rubric_scores: Dict[str, float], final_action, ground_truth,
        safety_violated: bool, safety_reason: str,
    ) -> List[str]:
        notes: List[str] = []
        for dim, score in rubric_scores.items():
            if score >= 0.85:
                continue
            if dim == "triage_accuracy":
                notes.append(
                    f"• Triage accuracy ({score:.2f}): Predicted "
                    f"{final_action.assigned_triage_level.value}, correct is "
                    f"{ground_truth.esi_level.value}."
                )
            elif dim == "documentation_quality":
                notes.append(
                    f"• Documentation ({score:.2f}): Improve HPI length and detail. "
                    f"Include 3+ differential diagnoses."
                )
            elif dim == "diagnostic_workup":
                notes.append(
                    f"• Diagnostic workup ({score:.2f}): Required tests include: "
                    f"{ground_truth.required_diagnostics}."
                )
            elif dim == "treatment_safety":
                notes.append(
                    f"• Treatment safety ({score:.2f}): Required interventions: "
                    f"{ground_truth.critical_interventions}."
                )
            elif dim == "time_to_disposition":
                notes.append(
                    f"• Time efficiency ({score:.2f}): Episode took too long for "
                    f"ESI {ground_truth.esi_level.value} acuity."
                )
        if safety_violated:
            notes.insert(0, f"⚠ SAFETY VIOLATION (capped at {self.SAFETY_VIOLATION_CAP}): {safety_reason}")
        return notes

    def _build_feedback_text(
        self, rubric_scores: Dict[str, float], total_score: float, passed: bool,
        observed_flags: List[str], improvement_notes: List[str], safety_violated: bool,
    ) -> str:
        status = "✓ PASS" if passed else "✗ FAIL"
        lines = [
            f"[MEDIUM GRADER] {status}",
            f"Total Score: {total_score * 100:.1f} / 100  "
            f"(Pass threshold: {self.pass_threshold * 100:.0f})",
            "", "Rubric Breakdown:",
        ]
        for dim, score in rubric_scores.items():
            weight = self.WEIGHTS[dim]
            pts = score * weight * 100
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            icon = "✓" if score >= 0.75 else ("△" if score >= 0.50 else "✗")
            lines.append(
                f"  {icon} {dim:<32} [{bar}] {score:.3f}  ({pts:.1f}/{weight * 100:.0f} pts)"
            )
        if safety_violated:
            lines += ["", f"⚠  Score capped at {self.SAFETY_VIOLATION_CAP * 100:.0f} due to safety violation."]
        if observed_flags:
            lines += ["", "Penalty Flags Observed:"]
            seen: set = set()
            for flag in observed_flags:
                if flag not in seen:
                    lines.append(f"  ⚠  {flag}")
                    seen.add(flag)
        if improvement_notes:
            lines += ["", "Improvement Guidance:"]
            lines.extend(improvement_notes)
        return "\n".join(lines)

    def _empty_trajectory_result(self, trajectory: EpisodeTrajectory) -> GradeResult:
        return GradeResult(
            grader_tier=GraderTier.MEDIUM, pass_fail=False, total_score=0.0,
            rubric_scores={dim: 0.0 for dim in self.WEIGHTS},
            feedback_text="[MEDIUM GRADER] ✗ FAIL\n\nTrajectory contains no actions.",
            llm_critique=None, penalty_flags_observed=[],
            case_id=trajectory.case_id, trajectory_id=trajectory.trajectory_id,
        )