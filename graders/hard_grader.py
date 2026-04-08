from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from env.models import DispositionEnum, EpisodeTrajectory, GroundTruth, TriageLevel
from graders.base_grader import BaseGrader, GradeResult, GraderTier
from graders.medium_grader import MediumGrader

logger = logging.getLogger(__name__)


class HardGrader(BaseGrader):
    """
    LLM-assisted two-stage grader for HARD difficulty episodes.

    Stage 1 — Automated rubric (60% weight by default)
    Stage 2 — LLM judge via Anthropic Messages API (40% weight)

    Falls back to automated-only scoring when ANTHROPIC_API_KEY is absent
    or the API call fails for any reason.

    Pass threshold: 0.80 (80/100).
    """

    pass_threshold: float = 0.80

    DEFAULT_AUTOMATED_WEIGHT: float = 0.60
    DEFAULT_LLM_WEIGHT: float = 0.40

    SAFETY_CONCERN_CAP: float = 0.60
    CRITICAL_MISS_CAP: float = 0.40
    HIDDEN_FINDING_BONUS: float = 0.10

    AUTOMATED_WEIGHTS: Dict[str, float] = {
        "triage_accuracy": 0.22,
        "documentation_quality": 0.20,
        "diagnostic_workup": 0.18,
        "treatment_safety": 0.18,
        "hidden_finding_awareness": 0.12,
        "drug_interaction_awareness": 0.10,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self._medium_grader = MediumGrader(config)
        cfg = self.config
        self.judge_model = cfg.get("judge_model", "claude-sonnet-4-20250514")
        self.judge_max_tokens = int(cfg.get("judge_max_tokens", 1000))
        self.judge_timeout = int(cfg.get("judge_timeout", 30))
        self.llm_weight = float(cfg.get("llm_weight", self.DEFAULT_LLM_WEIGHT))
        self.automated_weight = float(cfg.get("automated_weight", self.DEFAULT_AUTOMATED_WEIGHT))
        self._api_key = cfg.get("api_key") or os.environ.get("ANTHROPIC_API_KEY", "")

    def grade(self, trajectory: EpisodeTrajectory) -> GradeResult:
        if not trajectory.actions:
            return self._empty_trajectory_result(trajectory)

        ground_truth = trajectory.patient_case.get_ground_truth()
        allergies = trajectory.patient_case.allergies
        final_action = self._best_action(trajectory.actions)
        all_treatments = [tx for a in trajectory.actions for tx in a.treatments]
        all_diagnostics = [d for a in trajectory.actions for d in a.diagnostic_orders]
        observed_flags = self._collect_all_flags(trajectory.reward_signals)

        automated_scores, automated_total = self._run_automated_rubric(
            trajectory=trajectory, final_action=final_action,
            all_treatments=all_treatments, all_diagnostics=all_diagnostics,
            ground_truth=ground_truth, allergies=allergies,
        )

        llm_result, llm_critique_text, llm_used = self._run_llm_judge(
            trajectory=trajectory, ground_truth=ground_truth
        )

        if llm_used and llm_result is not None:
            llm_normalised: Optional[float] = self._normalise_llm_result(llm_result)
            fused = (self.automated_weight * automated_total
                     + self.llm_weight * llm_normalised)
        else:
            fused = automated_total
            llm_normalised = None

        fused, override_notes = self._apply_overrides(
            score=fused, final_action=final_action, ground_truth=ground_truth,
            llm_result=llm_result, observed_flags=observed_flags,
        )

        total_score = max(0.0, min(1.0, round(fused, 4)))
        passed = total_score >= self.pass_threshold

        rubric_scores = {**automated_scores}
        if llm_result:
            rubric_scores["llm_clinical_reasoning"] = round(
                llm_result.get("clinical_reasoning_score", 0) / 10.0, 4
            )
            rubric_scores["llm_differential_quality"] = round(
                llm_result.get("differential_quality_score", 0) / 10.0, 4
            )

        feedback = self._build_feedback_text(
            rubric_scores=rubric_scores, automated_total=automated_total,
            llm_normalised=llm_normalised, total_score=total_score, passed=passed,
            llm_used=llm_used, llm_result=llm_result, override_notes=override_notes,
            observed_flags=observed_flags,
        )

        return GradeResult(
            grader_tier=GraderTier.HARD, pass_fail=passed, total_score=total_score,
            rubric_scores=rubric_scores, feedback_text=feedback,
            llm_critique=llm_critique_text, penalty_flags_observed=observed_flags,
            case_id=trajectory.case_id, trajectory_id=trajectory.trajectory_id,
        )

    def _run_automated_rubric(
        self, trajectory, final_action, all_treatments, all_diagnostics,
        ground_truth: GroundTruth, allergies: List[str],
    ) -> Tuple[Dict[str, float], float]:
        triage_score = self._medium_grader._score_triage_accuracy(final_action, ground_truth)
        doc_score = self._medium_grader._score_documentation_quality(final_action)
        diag_score = self._compute_diagnostic_coverage(
            all_diagnostics, ground_truth.required_diagnostics, penalise_overordering=True
        )
        tx_score = self._medium_grader._score_treatment_safety(
            all_treatments=all_treatments, allergies=allergies,
            critical_interventions=ground_truth.critical_interventions,
            esi_level=ground_truth.esi_level,
        )
        hidden_score = self._score_hidden_finding_awareness(trajectory, ground_truth)
        drug_score = self._score_drug_interaction_awareness(trajectory, ground_truth)

        scores = {
            "triage_accuracy": round(triage_score, 4),
            "documentation_quality": round(doc_score, 4),
            "diagnostic_workup": round(diag_score, 4),
            "treatment_safety": round(tx_score, 4),
            "hidden_finding_awareness": round(hidden_score, 4),
            "drug_interaction_awareness": round(drug_score, 4),
        }
        weighted_total = sum(self.AUTOMATED_WEIGHTS[dim] * score for dim, score in scores.items())
        return scores, round(weighted_total, 4)

    def _score_hidden_finding_awareness(
        self, trajectory: EpisodeTrajectory, ground_truth: GroundTruth
    ) -> float:
        if not ground_truth.hidden_findings:
            return 1.0
        agent_text = self._collect_agent_text(trajectory)
        hits = sum(
            1 for finding in ground_truth.hidden_findings
            if any(kw in agent_text for kw in self._extract_keywords_from_finding(finding))
        )
        return round(hits / len(ground_truth.hidden_findings), 4)

    def _score_drug_interaction_awareness(
        self, trajectory: EpisodeTrajectory, ground_truth: GroundTruth
    ) -> float:
        interaction_findings = [
            f for f in ground_truth.hidden_findings if "INTERACTION" in f.upper()
        ]
        if not interaction_findings:
            return 1.0
        agent_text = self._collect_agent_text(trajectory)
        interaction_keywords = [
            "interaction", "interacts", "contraindicated", "avoid",
            "drug-drug", "DDI", "caution", "warfarin", "anticoagul",
        ]
        awareness_hits = sum(1 for kw in interaction_keywords if kw.lower() in agent_text)
        awareness_score = min(1.0, awareness_hits / 2.0)
        all_tx = [tx for a in trajectory.actions for tx in a.treatments]
        for finding in interaction_findings:
            match = re.search(r"(\w+)\s*\+\s*(\w+)", finding)
            if match:
                drug_a = match.group(1).upper()
                drug_b = match.group(2).upper()
                for tx in all_tx:
                    tx_upper = tx.intervention.upper()
                    if (drug_a in tx_upper or drug_b in tx_upper) and awareness_hits < 1:
                        awareness_score = max(0.0, awareness_score - 0.30)
        return round(min(1.0, max(0.0, awareness_score)), 4)

    def _run_llm_judge(
        self, trajectory: EpisodeTrajectory, ground_truth: GroundTruth
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str], bool]:
        if not self._api_key:
            logger.info("HardGrader: ANTHROPIC_API_KEY not set. Using automated fallback.")
            return None, None, False

        prompt = self._build_judge_prompt(trajectory, ground_truth)

        try:
            import urllib.request

            payload = json.dumps({
                "model": self.judge_model,
                "max_tokens": self.judge_max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }).encode("utf-8")

            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": self._api_key,
                    "anthropic-version": "2023-06-01",
                },
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=self.judge_timeout) as resp:
                raw = resp.read().decode("utf-8")

            response_data = json.loads(raw)
            content_blocks = response_data.get("content", [])
            text_response = " ".join(
                block.get("text", "")
                for block in content_blocks
                if block.get("type") == "text"
            ).strip()

            parsed = self._parse_llm_response(text_response)
            return parsed, text_response, True

        except Exception as exc:
            logger.warning(
                "HardGrader: LLM judge call failed (%s: %s). Automated fallback.",
                type(exc).__name__, exc,
            )
            return None, None, False

    def _build_judge_prompt(
        self, trajectory: EpisodeTrajectory, ground_truth: GroundTruth
    ) -> str:
        case = trajectory.patient_case
        patient_summary = (
            f"Patient: {case.demographics.age}y {case.demographics.sex.value}, "
            f"presenting with: {case.chief_complaint.text}\n"
            f"Vitals: HR={case.vitals.heart_rate}, "
            f"BP={case.vitals.bp_systolic}/{case.vitals.bp_diastolic}, "
            f"SpO2={case.vitals.spo2:.1%}, GCS={case.vitals.gcs}\n"
            f"Comorbidities: {[c.name for c in case.comorbidities]}\n"
            f"Medications: {[m.name for m in case.current_medications]}\n"
            f"Allergies: {case.allergies}"
        )
        gt_summary = (
            f"GROUND TRUTH (revealed to judge):\n"
            f"  ESI: {ground_truth.esi_level.value}\n"
            f"  Primary Dx: {ground_truth.primary_dx_name}\n"
            f"  Required Diagnostics: {ground_truth.required_diagnostics}\n"
            f"  Critical Interventions: {ground_truth.critical_interventions}\n"
            f"  Correct Disposition: {ground_truth.correct_disposition.value}\n"
            f"  HIDDEN FINDINGS: {ground_truth.hidden_findings}"
        )
        action_summaries = []
        for i, action in enumerate(trajectory.actions):
            action_summaries.append(
                f"Step {i}: ESI={action.assigned_triage_level.value}, "
                f"Disposition={action.disposition.value}, "
                f"Diagnostics={[d.test_name for d in action.diagnostic_orders]}, "
                f"Differentials={action.clinical_note.differential_diagnoses}, "
                f"Reasoning={action.reasoning_chain[:300]}"
            )

        return f"""You are an expert emergency medicine physician evaluating an AI triage agent.

=== PATIENT CASE ===
{patient_summary}

=== {gt_summary} ===

=== AGENT ACTIONS ===
{chr(10).join(action_summaries)}

Score the agent. Return ONLY valid JSON with no other text:
{{
  "clinical_reasoning_score": <integer 0-10>,
  "differential_quality_score": <integer 0-10>,
  "hidden_finding_detection": <true or false>,
  "safety_concerns": [<string>, ...],
  "clinical_narrative_critique": "<string max 200 words>"
}}"""

    def _parse_llm_response(self, text: str) -> Dict[str, Any]:
        defaults: Dict[str, Any] = {
            "clinical_reasoning_score": 5,
            "differential_quality_score": 5,
            "hidden_finding_detection": False,
            "safety_concerns": [],
            "clinical_narrative_critique": "Unable to parse LLM judge response.",
        }
        if not text:
            return defaults
        cleaned = re.sub(r"```(?:json)?", "", text).strip()
        cleaned = re.sub(r"```", "", cleaned).strip()
        json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not json_match:
            return defaults
        try:
            parsed = json.loads(json_match.group())
        except json.JSONDecodeError:
            return defaults
        result: Dict[str, Any] = dict(defaults)
        result["clinical_reasoning_score"] = max(0, min(10, int(parsed.get("clinical_reasoning_score", 5))))
        result["differential_quality_score"] = max(0, min(10, int(parsed.get("differential_quality_score", 5))))
        result["hidden_finding_detection"] = bool(parsed.get("hidden_finding_detection", False))
        sc = parsed.get("safety_concerns", [])
        result["safety_concerns"] = [str(s) for s in sc] if isinstance(sc, list) else []
        result["clinical_narrative_critique"] = str(parsed.get("clinical_narrative_critique", ""))[:1000]
        return result

    def _normalise_llm_result(self, llm_result: Optional[Dict[str, Any]]) -> float:
        if not llm_result:
            return 0.5
        crs = llm_result.get("clinical_reasoning_score", 5) / 10.0
        dqs = llm_result.get("differential_quality_score", 5) / 10.0
        return round(0.50 * crs + 0.50 * dqs, 4)

    def _apply_overrides(
        self, score: float, final_action, ground_truth: GroundTruth,
        llm_result: Optional[Dict[str, Any]], observed_flags: List[str],
    ) -> Tuple[float, List[str]]:
        override_notes: List[str] = []

        if llm_result and llm_result.get("hidden_finding_detection", False):
            old = score
            score = min(1.0, score + self.HIDDEN_FINDING_BONUS)
            override_notes.append(
                f"Hidden finding detected: +{self.HIDDEN_FINDING_BONUS:.2f} bonus ({old:.3f} → {score:.3f})."
            )

        if llm_result and llm_result.get("safety_concerns"):
            if score > self.SAFETY_CONCERN_CAP:
                override_notes.append(
                    f"Score capped at {self.SAFETY_CONCERN_CAP} due to "
                    f"{len(llm_result['safety_concerns'])} safety concern(s)."
                )
                score = self.SAFETY_CONCERN_CAP

        if self._is_critical_miss(final_action.assigned_triage_level, ground_truth.esi_level):
            if score > self.CRITICAL_MISS_CAP:
                override_notes.append(
                    f"Score capped at {self.CRITICAL_MISS_CAP} due to CRITICAL_MISS."
                )
                score = self.CRITICAL_MISS_CAP

        if "ALLERGY_VIOLATION" in observed_flags and "UNSAFE_DISCHARGE" in observed_flags:
            combined_cap = min(self.SAFETY_CONCERN_CAP, self.CRITICAL_MISS_CAP + 0.20)
            if score > combined_cap:
                override_notes.append(
                    f"Score capped at {combined_cap:.2f}: both ALLERGY_VIOLATION and UNSAFE_DISCHARGE."
                )
                score = combined_cap

        return score, override_notes

    @staticmethod
    def _collect_agent_text(trajectory: EpisodeTrajectory) -> str:
        parts = []
        for action in trajectory.actions:
            parts.append(action.reasoning_chain)
            parts.append(action.clinical_note.assessment)
            parts.append(action.clinical_note.plan)
            parts.append(action.clinical_note.history_of_present_illness)
            parts.extend(action.clinical_note.differential_diagnoses)
            parts.extend([d.rationale for d in action.diagnostic_orders])
            parts.extend([t.rationale for t in action.treatments])
        return " ".join(parts).lower()

    @staticmethod
    def _extract_keywords_from_finding(finding: str) -> List[str]:
        cleaned = re.sub(r"\[.*?\]", "", finding).strip()
        tokens = re.split(r"[^a-zA-Z]+", cleaned)
        stopwords = {
            "with", "from", "that", "this", "have", "been", "were",
            "after", "before", "their", "which", "while", "about",
            "initial", "found", "standard", "patient", "known",
        }
        return [t.lower() for t in tokens if len(t) >= 4 and t.lower() not in stopwords]

    def _build_feedback_text(
        self, rubric_scores: Dict[str, float], automated_total: float,
        llm_normalised: Optional[float], total_score: float, passed: bool,
        llm_used: bool, llm_result: Optional[Dict[str, Any]],
        override_notes: List[str], observed_flags: List[str],
    ) -> str:
        status = "✓ PASS" if passed else "✗ FAIL"
        lines = [
            f"[HARD GRADER] {status}",
            f"Total Score: {total_score * 100:.1f} / 100  "
            f"(Pass threshold: {self.pass_threshold * 100:.0f})",
            "",
        ]
        if llm_used and llm_normalised is not None:
            lines += [
                "Score Fusion:",
                f"  Automated ({self.automated_weight:.0%}): {automated_total * 100:.1f} pts",
                f"  LLM Judge ({self.llm_weight:.0%}):   {llm_normalised * 100:.1f} pts",
                "",
            ]
        else:
            lines += [
                "Score Fusion: Automated only (LLM judge unavailable).",
                f"  Automated (100%): {automated_total * 100:.1f} pts",
                "",
            ]

        lines.append("Automated Rubric Breakdown:")
        for dim in list(self.AUTOMATED_WEIGHTS.keys()):
            if dim in rubric_scores:
                score = rubric_scores[dim]
                weight = self.AUTOMATED_WEIGHTS[dim]
                pts = score * weight * 100
                bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
                icon = "✓" if score >= 0.80 else ("△" if score >= 0.55 else "✗")
                lines.append(
                    f"  {icon} {dim:<32} [{bar}] {score:.3f}  ({pts:.1f}/{weight * 100:.0f} pts)"
                )

        if llm_used and llm_result:
            lines += ["", "LLM Judge Scores:"]
            lines.append(f"  Clinical Reasoning:    {llm_result.get('clinical_reasoning_score', 'N/A')}/10")
            lines.append(f"  Differential Quality:  {llm_result.get('differential_quality_score', 'N/A')}/10")
            hfd = llm_result.get("hidden_finding_detection", False)
            lines.append(f"  Hidden Finding Found:  {'Yes ✓' if hfd else 'No ✗'}")
            sc = llm_result.get("safety_concerns", [])
            if sc:
                lines += ["", "  LLM Safety Concerns:"]
                for concern in sc:
                    lines.append(f"    ⚠  {concern}")

        if override_notes:
            lines += ["", "Override Conditions Applied:"]
            for note in override_notes:
                lines.append(f"  → {note}")

        if observed_flags:
            lines += ["", "Trajectory Penalty Flags:"]
            seen: set = set()
            for flag in observed_flags:
                if flag not in seen:
                    lines.append(f"  ⚠  {flag}")
                    seen.add(flag)

        if llm_used and llm_result:
            critique = llm_result.get("clinical_narrative_critique", "")
            if critique:
                lines += ["", "LLM Clinical Narrative Critique:", "", f"  {critique}"]

        if not llm_used:
            lines += [
                "",
                "Note: Set ANTHROPIC_API_KEY to enable the LLM judge for HARD grading.",
            ]
        return "\n".join(lines)

    def _empty_trajectory_result(self, trajectory: EpisodeTrajectory) -> GradeResult:
        return GradeResult(
            grader_tier=GraderTier.HARD, pass_fail=False, total_score=0.0,
            rubric_scores={dim: 0.0 for dim in self.AUTOMATED_WEIGHTS},
            feedback_text="[HARD GRADER] ✗ FAIL\n\nTrajectory contains no actions.",
            llm_critique=None, penalty_flags_observed=[],
            case_id=trajectory.case_id, trajectory_id=trajectory.trajectory_id,
        )