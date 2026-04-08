from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from env.models import DispositionEnum, EpisodeTrajectory, TriageLevel
from graders.base_grader import BaseGrader, GradeResult, GraderTier


class EasyGrader(BaseGrader):
    """
    Binary pass/fail grader for EASY difficulty episodes.

    All six conditions must be met for PASS:
      1. ESI level exact match
      2. Clinical note meets minimum field lengths
      3. At least one required diagnostic ordered
      4. No allergy violations
      5. No unsafe discharge (ESI 1-2 patient sent home)
      6. Committed disposition (not PENDING) on final action

    total_score = 1.0 (PASS) or 0.0 (FAIL)
    """

    pass_threshold: float = 1.0
    MIN_DIAGNOSTIC_HITS: int = 1
    MIN_FIELD_LENGTHS: Dict[str, int] = {
        "chief_complaint_summary": 20,
        "history_of_present_illness": 100,
        "physical_exam_findings": 50,
        "assessment": 50,
        "plan": 50,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)

    def grade(self, trajectory: EpisodeTrajectory) -> GradeResult:
        actions = trajectory.actions
        ground_truth = trajectory.patient_case.get_ground_truth()
        allergies = trajectory.patient_case.allergies
        reward_signals = trajectory.reward_signals

        final_action = self._best_action(actions)
        if final_action is None:
            return self._fail_result(
                trajectory=trajectory,
                reason="No actions found in trajectory. Agent produced no output.",
                rubric_scores={},
            )

        all_treatments = [tx for a in actions for tx in a.treatments]
        all_diagnostics = [d for a in actions for d in a.diagnostic_orders]

        esi_score = self._compute_esi_accuracy(
            final_action.assigned_triage_level, ground_truth.esi_level
        )
        esi_exact = esi_score == 1.0

        note_complete, note_score, note_fail_reason = self._check_note_completeness(
            final_action.clinical_note
        )

        diag_hit, diag_score = self._check_diagnostic_minimum(
            all_diagnostics, ground_truth.required_diagnostics
        )

        allergy_safe = not self._check_allergy_violation(all_treatments, allergies)
        allergy_score = 1.0 if allergy_safe else 0.0

        safe_discharge, discharge_score = self._check_discharge_safety(
            final_action.disposition, ground_truth.esi_level
        )

        disposition_committed = final_action.disposition != DispositionEnum.PENDING
        disposition_score = 1.0 if disposition_committed else 0.0

        rubric_scores = {
            "esi_exact_match": round(esi_score, 4),
            "note_completeness": round(note_score, 4),
            "diagnostic_minimum_coverage": round(diag_score, 4),
            "allergy_safety": allergy_score,
            "discharge_safety": discharge_score,
            "disposition_committed": disposition_score,
        }

        failures: List[str] = []
        if not esi_exact:
            failures.append(
                f"ESI level mismatch: predicted "
                f"{final_action.assigned_triage_level.value}, "
                f"ground truth {ground_truth.esi_level.value}."
            )
        if not note_complete:
            failures.append(f"Clinical note incomplete: {note_fail_reason}")
        if not diag_hit:
            failures.append(
                f"No required diagnostic ordered. Required at least one of: "
                f"{ground_truth.required_diagnostics}."
            )
        if not allergy_safe:
            failures.append("Allergy violation detected.")
        if not safe_discharge:
            failures.append(
                f"Unsafe discharge: ESI {ground_truth.esi_level.value} patient "
                f"assigned disposition {final_action.disposition.value}."
            )
        if not disposition_committed:
            failures.append("Final action has disposition=PENDING.")

        passed = len(failures) == 0
        total_score = 1.0 if passed else 0.0
        observed_flags = self._collect_all_flags(reward_signals)

        if passed:
            feedback = self._build_pass_feedback(rubric_scores)
        else:
            feedback = self._build_fail_feedback(rubric_scores, failures, observed_flags)

        return GradeResult(
            grader_tier=GraderTier.EASY,
            pass_fail=passed,
            total_score=total_score,
            rubric_scores=rubric_scores,
            feedback_text=feedback,
            llm_critique=None,
            penalty_flags_observed=observed_flags,
            case_id=trajectory.case_id,
            trajectory_id=trajectory.trajectory_id,
        )

    def _check_note_completeness(self, note) -> Tuple[bool, float, str]:
        field_scores: List[float] = []
        fail_reason = ""
        for field_name, min_len in self.MIN_FIELD_LENGTHS.items():
            text = getattr(note, field_name, "") or ""
            if len(text.strip()) < min_len:
                if not fail_reason:
                    fail_reason = (
                        f"Field '{field_name}' is too short "
                        f"({len(text.strip())} chars, minimum {min_len})."
                    )
                field_scores.append(0.0)
            else:
                field_scores.append(1.0)
        if len(note.differential_diagnoses) < 1:
            fail_reason = fail_reason or "No differential diagnoses provided."
            field_scores.append(0.0)
        else:
            field_scores.append(1.0)
        complete = all(s == 1.0 for s in field_scores)
        rich_score = self._compute_note_completeness(note)
        return complete, rich_score, fail_reason

    def _check_diagnostic_minimum(
        self, ordered, required: List[str]
    ) -> Tuple[bool, float]:
        if not required:
            return True, 1.0
        required_upper = [self._normalise_test_name(r) for r in required]
        ordered_upper = [self._normalise_test_name(o.test_name) for o in ordered]
        hits = sum(
            1 for req in required_upper
            if any(req in ord_name or ord_name in req for ord_name in ordered_upper)
        )
        met = hits >= self.MIN_DIAGNOSTIC_HITS
        score = round(hits / len(required_upper), 4)
        return met, score

    @staticmethod
    def _check_discharge_safety(
        disposition: DispositionEnum, esi_level: TriageLevel
    ) -> Tuple[bool, float]:
        unsafe = {DispositionEnum.DISCHARGE_HOME, DispositionEnum.DISCHARGE_PCP_FOLLOWUP}
        critical = {TriageLevel.ESI_1, TriageLevel.ESI_2}
        if esi_level in critical and disposition in unsafe:
            return False, 0.0
        return True, 1.0

    def _build_pass_feedback(self, rubric_scores: Dict[str, float]) -> str:
        lines = ["[EASY GRADER] ✓ PASS", "", "All six pass conditions satisfied.", "", "Rubric Breakdown:"]
        for dim, score in rubric_scores.items():
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            lines.append(f"  {dim:<40} [{bar}] {score:.3f}")
        return "\n".join(lines)

    def _build_fail_feedback(
        self, rubric_scores: Dict[str, float], failures: List[str], flags: List[str]
    ) -> str:
        lines = ["[EASY GRADER] ✗ FAIL", "", f"Failed {len(failures)} of 6 pass condition(s):"]
        for i, reason in enumerate(failures, 1):
            lines.append(f"  {i}. {reason}")
        lines += ["", "Rubric Breakdown (for diagnostics):"]
        for dim, score in rubric_scores.items():
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            status = "✓" if score >= 0.9 else "✗"
            lines.append(f"  {status} {dim:<38} [{bar}] {score:.3f}")
        if flags:
            lines += ["", "Penalty Flags Observed:"]
            seen: set = set()
            for f in flags:
                if f not in seen:
                    lines.append(f"  ⚠  {f}")
                    seen.add(f)
        return "\n".join(lines)

    def _fail_result(
        self, trajectory: EpisodeTrajectory, reason: str, rubric_scores: Dict[str, float]
    ) -> GradeResult:
        return GradeResult(
            grader_tier=GraderTier.EASY, pass_fail=False, total_score=0.0,
            rubric_scores=rubric_scores,
            feedback_text=f"[EASY GRADER] ✗ FAIL\n\n{reason}",
            llm_critique=None, penalty_flags_observed=[],
            case_id=trajectory.case_id, trajectory_id=trajectory.trajectory_id,
        )