from __future__ import annotations

import re
import unicodedata
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

from env.models import (
    ClinicalNote,
    DiagnosticOrder,
    DispositionEnum,
    EpisodeTrajectory,
    GroundTruth,
    RewardSignal,
    TriageAction,
    TriageLevel,
    TreatmentAction,
)


class GraderTier(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class GradeResult(BaseModel):
    grader_tier: GraderTier
    pass_fail: bool
    total_score: float = Field(..., ge=0.0, le=1.0)
    rubric_scores: Dict[str, float] = Field(default_factory=dict)
    feedback_text: str = ""
    llm_critique: Optional[str] = None
    penalty_flags_observed: List[str] = Field(default_factory=list)
    case_id: str = ""
    trajectory_id: str = ""

    def scaled_score(self) -> float:
        return round(self.total_score * 100.0, 2)


class BaseGrader(ABC):
    pass_threshold: float = 0.70

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}

    @abstractmethod
    def grade(self, trajectory: EpisodeTrajectory) -> GradeResult: ...

    @staticmethod
    def _compute_esi_accuracy(predicted: TriageLevel, ground_truth: TriageLevel) -> float:
        if predicted == TriageLevel.UNKNOWN:
            return 0.0
        level_map = {
            TriageLevel.ESI_1: 1, TriageLevel.ESI_2: 2, TriageLevel.ESI_3: 3,
            TriageLevel.ESI_4: 4, TriageLevel.ESI_5: 5,
        }
        p = level_map.get(predicted, 0)
        t = level_map.get(ground_truth, 0)
        if p == 0 or t == 0:
            return 0.0
        return {0: 1.0, 1: 0.6, 2: 0.2}.get(abs(p - t), 0.0)

    @staticmethod
    def _is_under_triaged(
        predicted: TriageLevel, ground_truth: TriageLevel, threshold: int = 2
    ) -> bool:
        level_map = {
            TriageLevel.ESI_1: 1, TriageLevel.ESI_2: 2, TriageLevel.ESI_3: 3,
            TriageLevel.ESI_4: 4, TriageLevel.ESI_5: 5,
        }
        return (level_map.get(predicted, 0) - level_map.get(ground_truth, 0)) >= threshold

    @staticmethod
    def _is_critical_miss(predicted: TriageLevel, ground_truth: TriageLevel) -> bool:
        critical = {TriageLevel.ESI_1, TriageLevel.ESI_2}
        non_urgent = {TriageLevel.ESI_4, TriageLevel.ESI_5, TriageLevel.UNKNOWN}
        return ground_truth in critical and predicted in non_urgent

    @staticmethod
    def _compute_note_completeness(note: ClinicalNote) -> float:
        fields_config = [
            ("cc",  note.chief_complaint_summary, 20, [], 0.10),
            ("hpi", note.history_of_present_illness, 100,
             ["onset", "duration", "severity", "associated", "history", "prior", "allerg", "medic"], 0.30),
            ("exam", note.physical_exam_findings, 50,
             ["alert", "heart", "lung", "abdomen", "neuro", "vital", "general", "appear", "skin", "extremit"], 0.20),
            ("assessment", note.assessment, 50,
             ["consistent", "likely", "differential", "impression", "diagnosis", "presents", "consider"], 0.20),
            ("plan", note.plan, 50,
             ["order", "monitor", "consult", "admit", "discharge", "IV", "labs", "imaging", "follow", "antibiotic"], 0.20),
        ]
        total = sum(w * BaseGrader._score_field(text, ml, kws)
                    for _, text, ml, kws, w in fields_config)
        diff_bonus = min(0.05, len(note.differential_diagnoses) * 0.015)
        return min(1.0, total + diff_bonus)

    @staticmethod
    def _score_field(text: str, min_len: int, keywords: List[str]) -> float:
        if not text or len(text.strip()) < min_len:
            return 0.0
        if not keywords:
            return 1.0
        hits = sum(1 for kw in keywords if kw.lower() in text.lower())
        return round(0.5 + 0.5 * (hits / len(keywords)), 4)

    @staticmethod
    def _compute_diagnostic_coverage(
        ordered: List[DiagnosticOrder],
        required: List[str],
        penalise_overordering: bool = True,
    ) -> float:
        if not required:
            if penalise_overordering and len(ordered) > 5:
                return max(0.0, 1.0 - (len(ordered) - 5) * 0.05)
            return 1.0
        required_upper = [BaseGrader._normalise_test_name(r) for r in required]
        ordered_upper = [BaseGrader._normalise_test_name(o.test_name) for o in ordered]
        hits = sum(
            1 for req in required_upper
            if any(req in o or o in req for o in ordered_upper)
        )
        coverage = hits / len(required_upper)
        over = max(0, len(ordered) - len(required) - 3)
        penalty = over * 0.04 if penalise_overordering else 0.0
        return max(0.0, min(1.0, coverage - penalty))

    @staticmethod
    def _normalise_test_name(name: str) -> str:
        name = unicodedata.normalize("NFKD", name).upper().strip()
        return re.sub(r"[\s\-_/]+", "_", name)

    @staticmethod
    def _compute_treatment_coverage(
        treatments: List[TreatmentAction],
        critical_interventions: List[str],
    ) -> float:
        if not critical_interventions:
            return 1.0
        ci_upper = [BaseGrader._normalise_test_name(c) for c in critical_interventions]
        tx_upper = [BaseGrader._normalise_test_name(t.intervention) for t in treatments]
        hits = sum(1 for ci in ci_upper if any(ci in tx or tx in ci for tx in tx_upper))
        return round(hits / len(ci_upper), 4)

    @staticmethod
    def _check_allergy_violation(
        treatments: List[TreatmentAction], allergies: List[str]
    ) -> bool:
        nkda = {"NKDA", "NO KNOWN DRUG ALLERGIES", "NO KNOWN ALLERGIES"}
        filtered = [a.upper().strip() for a in allergies if a.upper().strip() not in nkda]
        if not filtered:
            return False
        for tx in treatments:
            tx_upper = tx.intervention.upper()
            for allergy in filtered:
                if allergy in tx_upper or tx_upper in allergy:
                    return True
        return False

    @staticmethod
    def _compute_disposition_score(
        assigned: DispositionEnum, correct: DispositionEnum
    ) -> float:
        if assigned == correct:
            return 1.0
        adjacency: Dict[DispositionEnum, Set[DispositionEnum]] = {
            DispositionEnum.ADMIT_ICU: {DispositionEnum.ADMIT_FLOOR},
            DispositionEnum.ADMIT_FLOOR: {DispositionEnum.ADMIT_ICU, DispositionEnum.OBSERVATION},
            DispositionEnum.OBSERVATION: {DispositionEnum.ADMIT_FLOOR},
            DispositionEnum.DISCHARGE_HOME: {DispositionEnum.DISCHARGE_PCP_FOLLOWUP},
            DispositionEnum.DISCHARGE_PCP_FOLLOWUP: {DispositionEnum.DISCHARGE_HOME},
        }
        return 0.5 if assigned in adjacency.get(correct, set()) else 0.0

    @staticmethod
    def _compute_confidence_calibration(confidence: float, esi_accuracy: float) -> float:
        if confidence >= 0.85 and esi_accuracy >= 0.8:
            return 1.0
        elif confidence >= 0.85 and esi_accuracy < 0.5:
            return max(0.0, esi_accuracy - 0.2)
        elif confidence < 0.4:
            return 0.6
        return round(0.5 + 0.5 * esi_accuracy, 4)

    @staticmethod
    def _compute_reasoning_quality(reasoning_chain: str) -> float:
        text = reasoning_chain.strip()
        if len(text) < 50:
            return 0.0
        score = 0.0
        if len(text) >= 500:
            score += 0.40
        elif len(text) >= 200:
            score += 0.30
        elif len(text) >= 100:
            score += 0.20
        else:
            score += 0.10
        markers = [
            "because", "therefore", "given", "due to", "consistent with",
            "differential", "ruled out", "likely", "unlikely", "concern",
            "risk", "vital", "history", "examination", "assessment",
        ]
        hits = sum(1 for m in markers if m in text.lower())
        score += min(0.40, hits * 0.04)
        if re.search(r"(\d+[\.\)]\s|\-\s|\*\s|•\s)", text):
            score += 0.10
        if re.search(r"\b\d+[\.,]?\d*\s*(mg|mmHg|bpm|%|°|g/dL|mmol)\b", text):
            score += 0.10
        return min(1.0, round(score, 4))

    @staticmethod
    def _normalize_score(raw: float, min_val: float, max_val: float) -> float:
        if max_val <= min_val:
            return 0.0
        return max(0.0, min(1.0, (raw - min_val) / (max_val - min_val)))

    @staticmethod
    def _collect_all_flags(reward_signals: List[RewardSignal]) -> List[str]:
        return [flag for rs in reward_signals for flag in rs.penalty_flags]

    @staticmethod
    def _best_action(actions: List[TriageAction]) -> Optional[TriageAction]:
        if not actions:
            return None
        for action in reversed(actions):
            if action.is_final:
                return action
        return actions[-1]

    @staticmethod
    def _build_feedback(
        rubric_scores: Dict[str, float],
        flags: List[str],
        pass_fail: bool,
        tier: GraderTier,
    ) -> str:
        lines = [
            f"[{tier.value.upper()} GRADER] {'PASS' if pass_fail else 'FAIL'}",
            "", "Rubric Breakdown:",
        ]
        for dim, score in rubric_scores.items():
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            lines.append(f"  {dim:<35} [{bar}] {score:.3f}")
        if flags:
            lines.append("")
            lines.append("Penalty Flags Observed:")
            seen: set = set()
            for flag in flags:
                if flag not in seen:
                    lines.append(f"  ⚠  {flag}")
                    seen.add(flag)
        return "\n".join(lines)