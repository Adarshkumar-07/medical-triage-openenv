from __future__ import annotations
import re
from typing import List, Optional, Tuple
from env.models import (
    ClinicalNote, DiagnosticOrder, DispositionEnum, GroundTruth,
    RewardConfig, RewardSignal, TreatmentAction, TriageAction, TriageLevel,
)

FLAG_CRITICAL_MISS = "CRITICAL_MISS"
FLAG_UNSAFE_DISCHARGE = "UNSAFE_DISCHARGE"
FLAG_ALLERGY_VIOLATION = "ALLERGY_VIOLATION"
FLAG_EMPTY_DOCUMENTATION = "EMPTY_DOCUMENTATION"
FLAG_OVERCONFIDENCE = "OVERCONFIDENCE"
FLAG_UNDER_TRIAGE = "UNDER_TRIAGE"
FLAG_MISSING_CRITICAL_INTERVENTION = "MISSING_CRITICAL_INTERVENTION"


class RewardEngine:
    def __init__(self, config: Optional[RewardConfig] = None) -> None:
        self.config = config or RewardConfig()

    def compute(
        self,
        action: TriageAction,
        ground_truth: GroundTruth,
        elapsed_seconds: int,
        step: int,
        allergies: Optional[List[str]] = None,
        current_medication_names: Optional[List[str]] = None,
        is_final_step: bool = False,
    ) -> RewardSignal:
        cfg = self.config
        allergies = allergies or []
        triage_score = self._score_triage_level(action.assigned_triage_level, ground_truth.esi_level)
        doc_score = self._score_documentation(action.clinical_note, cfg)
        diag_score = self._score_diagnostics(action.diagnostic_orders, ground_truth.required_diagnostics)
        tx_score = self._score_treatment_safety(
            action.treatments, allergies, current_medication_names or [],
            ground_truth.critical_interventions, ground_truth.esi_level,
        )
        time_score = self._score_time_efficiency(elapsed_seconds, ground_truth.esi_level, cfg)
        flags, penalty_total = self._apply_penalties(action, ground_truth, allergies, is_final_step)
        weighted = (
            cfg.weight_triage_accuracy * triage_score
            + cfg.weight_documentation * doc_score
            + cfg.weight_diagnostic_appropriateness * diag_score
            + cfg.weight_treatment_safety * tx_score
            + cfg.weight_time_efficiency * time_score
        )
        if not is_final_step:
            weighted = weighted * 0.40
        total_reward = max(-1., min(1., weighted + penalty_total))
        scaled = round((total_reward + 1.) / 2. * 100., 2)
        return RewardSignal(
            total_reward=round(total_reward, 4), scaled_score=scaled,
            triage_accuracy_score=round(triage_score, 4),
            documentation_score=round(doc_score, 4),
            diagnostic_appropriateness_score=round(diag_score, 4),
            treatment_safety_score=round(tx_score, 4),
            time_efficiency_score=round(time_score, 4),
            penalty_flags=flags, penalty_total=round(penalty_total, 4), step=step,
        )

    def _score_triage_level(self, assigned: TriageLevel, ground_truth: TriageLevel) -> float:
        if assigned == TriageLevel.UNKNOWN:
            return 0.
        level_map = {TriageLevel.ESI_1:1, TriageLevel.ESI_2:2, TriageLevel.ESI_3:3,
                     TriageLevel.ESI_4:4, TriageLevel.ESI_5:5}
        a = level_map.get(assigned, 0)
        t = level_map.get(ground_truth, 0)
        if a == 0 or t == 0:
            return 0.
        delta = abs(a - t)
        return {0: 1., 1: 0.6, 2: 0.2}.get(delta, 0.)

    def _score_documentation(self, note: ClinicalNote, cfg: RewardConfig) -> float:
        fields_config = [
            ("chief_complaint_summary", note.chief_complaint_summary, 20, [], 0.10),
            ("history_of_present_illness", note.history_of_present_illness, cfg.min_hpi_length,
             ["onset","duration","severity","associated","history","prior","allerg","medic"], 0.30),
            ("physical_exam_findings", note.physical_exam_findings, cfg.min_exam_length,
             ["alert","heart","lung","abdomen","neuro","vital","general","appear","skin","extremit"], 0.20),
            ("assessment", note.assessment, cfg.min_assessment_length,
             ["consistent","likely","differential","impression","diagnosis","presents","consider"], 0.20),
            ("plan", note.plan, cfg.min_plan_length,
             ["order","monitor","consult","admit","discharge","IV","labs","imaging","follow","antibiotic"], 0.20),
        ]
        total = 0.
        for _name, text, min_len, keywords, weight in fields_config:
            total += weight * self._score_note_field(text, min_len, keywords)
        diff_bonus = min(0.05, len(note.differential_diagnoses) * 0.015)
        return min(1., total + diff_bonus)

    def _score_diagnostics(self, ordered: List[DiagnosticOrder], required: List[str]) -> float:
        if not required:
            over_penalty = max(0, len(ordered) - 3) * 0.05
            return max(0., 1. - over_penalty)
        ordered_names = {o.test_name.upper().strip() for o in ordered}
        required_upper = [r.upper().strip() for r in required]
        hits = sum(
            1 for req in required_upper
            if any(req in ord_name or ord_name in req for ord_name in ordered_names)
        )
        coverage = min(1., hits / len(required_upper))
        over_penalty = max(0, len(ordered) - len(required) - 2) * 0.05
        turnaround_bonus = min(0.05, sum(
            0.005 for order in ordered
            if (exp := self._expected_turnaround(order.test_name)) is not None
            and abs(order.expected_turnaround_minutes - exp) / exp <= 0.20
        ))
        return max(0., min(1., coverage - over_penalty + turnaround_bonus))

    def _score_treatment_safety(
        self, treatments: List[TreatmentAction], allergies: List[str],
        current_medications: List[str], critical_interventions: List[str],
        esi_level: TriageLevel,
    ) -> float:
        from env.models import PriorityEnum
        if not critical_interventions:
            return 0. if self._check_allergy_violations(treatments, allergies) else 1.
        treatment_names = [t.intervention.upper() for t in treatments]
        ci_upper = [c.upper() for c in critical_interventions]
        score = 0.
        for ci in ci_upper:
            for t in treatments:
                if ci in t.intervention.upper() or t.intervention.upper() in ci:
                    if esi_level in (TriageLevel.ESI_1, TriageLevel.ESI_2):
                        score += 0.40 if t.priority == PriorityEnum.IMMEDIATE else 0.20
                    else:
                        score += 0.30
                    break
        max_score = 0.40 * len(ci_upper)
        normalised = min(1., score / max_score) if max_score > 0 else 1.
        if self._check_allergy_violations(treatments, allergies):
            normalised = max(0., normalised - 0.50)
        return round(normalised, 4)

    def _score_time_efficiency(self, elapsed_seconds: int, esi_level: TriageLevel, cfg: RewardConfig) -> float:
        threshold = cfg.time_threshold_for_esi(esi_level)
        if elapsed_seconds <= threshold:
            return 1.
        overshoot = elapsed_seconds - threshold
        decay = (overshoot / threshold) * 0.5
        return max(cfg.time_efficiency_floor, round(1. - decay, 4))

    def _apply_penalties(
        self, action: TriageAction, ground_truth: GroundTruth,
        allergies: List[str], is_final_step: bool,
    ) -> Tuple[List[str], float]:
        flags: List[str] = []
        total = 0.
        cfg = self.config
        if self._is_under_triaged(action.assigned_triage_level, ground_truth.esi_level):
            flags.append(FLAG_UNDER_TRIAGE); total += cfg.penalty_under_triage
        if self._is_unsafe_discharge(action.disposition, ground_truth.esi_level, action.is_final):
            flags.append(FLAG_UNSAFE_DISCHARGE); total += cfg.penalty_unsafe_discharge
        if self._check_allergy_violations(action.treatments, allergies):
            flags.append(FLAG_ALLERGY_VIOLATION); total += cfg.penalty_allergy_violation
        if self._has_empty_documentation(action.clinical_note):
            flags.append(FLAG_EMPTY_DOCUMENTATION); total += cfg.penalty_empty_documentation
        if self._is_overconfident(action.confidence_score, action.assigned_triage_level, ground_truth.esi_level):
            flags.append(FLAG_OVERCONFIDENCE); total += cfg.penalty_overconfidence
        if is_final_step and self._is_critical_miss(action, ground_truth):
            flags.append(FLAG_CRITICAL_MISS); total += cfg.penalty_critical_miss
        if is_final_step and ground_truth.esi_level in (TriageLevel.ESI_1, TriageLevel.ESI_2):
            if self._missing_critical_intervention(action.treatments, ground_truth.critical_interventions):
                flags.append(FLAG_MISSING_CRITICAL_INTERVENTION); total += -0.30
        return flags, round(max(-1., total), 4)

    @staticmethod
    def _is_under_triaged(assigned: TriageLevel, truth: TriageLevel) -> bool:
        level_map = {TriageLevel.ESI_1:1, TriageLevel.ESI_2:2, TriageLevel.ESI_3:3,
                     TriageLevel.ESI_4:4, TriageLevel.ESI_5:5}
        return (level_map.get(assigned, 99) - level_map.get(truth, 99)) >= 2

    @staticmethod
    def _is_unsafe_discharge(disposition: DispositionEnum, esi_level: TriageLevel, is_final: bool) -> bool:
        if not is_final:
            return False
        return (esi_level in (TriageLevel.ESI_1, TriageLevel.ESI_2) and
                disposition in (DispositionEnum.DISCHARGE_HOME, DispositionEnum.DISCHARGE_PCP_FOLLOWUP))

    @staticmethod
    def _check_allergy_violations(treatments: List[TreatmentAction], allergies: List[str]) -> bool:
        nkda = {"NKDA", "NO KNOWN DRUG ALLERGIES"}
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
    def _has_empty_documentation(note: ClinicalNote) -> bool:
        return any(len(getattr(note, f, "") or "") < ml for f, ml in [
            ("chief_complaint_summary", 20), ("history_of_present_illness", 100),
            ("physical_exam_findings", 50), ("assessment", 50), ("plan", 50),
        ])

    @staticmethod
    def _is_overconfident(confidence: float, assigned: TriageLevel, truth: TriageLevel) -> bool:
        return confidence >= 1.0 and assigned != truth

    @staticmethod
    def _is_critical_miss(action: TriageAction, ground_truth: GroundTruth) -> bool:
        return (ground_truth.esi_level in (TriageLevel.ESI_1, TriageLevel.ESI_2) and
                action.assigned_triage_level in (TriageLevel.ESI_4, TriageLevel.ESI_5, TriageLevel.UNKNOWN))

    @staticmethod
    def _missing_critical_intervention(treatments: List[TreatmentAction], critical_interventions: List[str]) -> bool:
        if not critical_interventions:
            return False
        tx_upper = [t.intervention.upper() for t in treatments]
        for ci in [c.upper() for c in critical_interventions]:
            for t in tx_upper:
                if ci in t or t in ci:
                    return False
        return True

    @staticmethod
    def _score_note_field(text: str, min_len: int, keywords: List[str]) -> float:
        if not text or len(text) < min_len:
            return 0.
        base = 0.5
        if not keywords:
            return 1.
        hits = sum(1 for kw in keywords if kw.lower() in text.lower())
        return round(base + 0.5 * (hits / len(keywords)), 4)

    @staticmethod
    def _expected_turnaround(test_name: str) -> Optional[int]:
        return {
            "CBC":60,"BMP":60,"LFT":90,"TROPONIN":60,"D_DIMER":60,"ECG":5,
            "CHEST_XRAY":30,"CT_HEAD":45,"CT_ABDOMEN_PELVIS":60,"CTPA":60,
            "MRI_BRAIN":90,"URINALYSIS":30,"ABG":15,"LACTATE":30,
            "BLOOD_GLUCOSE":5,"TOXICOLOGY":120,
        }.get(test_name.upper().replace(" ","_").replace("-","_"))

    def compute_trajectory_reward(self, reward_signals: List[RewardSignal]) -> float:
        if not reward_signals:
            return 0.
        gamma = 0.85
        n = len(reward_signals)
        cumulative = sum(gamma**(n-1-i) * rs.total_reward for i, rs in enumerate(reward_signals))
        weight_sum = sum(gamma**k for k in range(n))
        return round(cumulative / weight_sum, 4) if weight_sum > 0 else 0.

    def summarise_penalties(self, reward_signals: List[RewardSignal]) -> dict:
        from collections import Counter
        all_flags = [f for rs in reward_signals for f in rs.penalty_flags]
        counts = Counter(all_flags)
        return {
            "total_penalty_events": len(all_flags),
            "unique_flag_types": len(counts),
            "flag_counts": dict(counts),
            "total_penalty_value": round(sum(rs.penalty_total for rs in reward_signals), 4),
        }