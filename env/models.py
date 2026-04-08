from __future__ import annotations
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator

class TriageLevel(str, Enum):
    ESI_1 = "ESI_1"; ESI_2 = "ESI_2"; ESI_3 = "ESI_3"
    ESI_4 = "ESI_4"; ESI_5 = "ESI_5"; UNKNOWN = "UNKNOWN"

class DispositionEnum(str, Enum):
    ADMIT_ICU = "ADMIT_ICU"; ADMIT_FLOOR = "ADMIT_FLOOR"
    OBSERVATION = "OBSERVATION"; DISCHARGE_HOME = "DISCHARGE_HOME"
    DISCHARGE_PCP_FOLLOWUP = "DISCHARGE_PCP_FOLLOWUP"
    TRANSFER = "TRANSFER"; PENDING = "PENDING"

class PriorityEnum(str, Enum):
    IMMEDIATE = "IMMEDIATE"; URGENT = "URGENT"; ROUTINE = "ROUTINE"

class DifficultyLevel(str, Enum):
    EASY = "easy"; MEDIUM = "medium"; HARD = "hard"

class SexEnum(str, Enum):
    M = "M"; F = "F"; X = "X"

class PresentationCategory(str, Enum):
    CARDIAC = "cardiac"; NEUROLOGICAL = "neurological"
    RESPIRATORY = "respiratory"; GASTROINTESTINAL = "gastrointestinal"
    TRAUMA = "trauma"; PSYCHIATRIC = "psychiatric"
    OBSTETRIC = "obstetric"; PEDIATRIC = "pediatric"
    TOXICOLOGICAL = "toxicological"; INFECTIOUS = "infectious"
    ENDOCRINE = "endocrine"; MUSCULOSKELETAL = "musculoskeletal"
    UROLOGICAL = "urological"; DERMATOLOGICAL = "dermatological"
    OPHTHALMOLOGICAL = "ophthalmological"

class VitalSigns(BaseModel):
    heart_rate: int = Field(..., ge=0, le=300)
    bp_systolic: int = Field(..., ge=0, le=300)
    bp_diastolic: int = Field(..., ge=0, le=200)
    respiratory_rate: int = Field(..., ge=0, le=60)
    spo2: float = Field(..., ge=0.0, le=1.0)
    temperature_celsius: float = Field(..., ge=25.0, le=45.0)
    gcs: int = Field(..., ge=3, le=15)
    pain_score: int = Field(..., ge=0, le=10)

    def is_critical(self) -> bool:
        return any([
            self.heart_rate < 40 or self.heart_rate > 150,
            self.bp_systolic < 80 or self.bp_systolic > 220,
            self.respiratory_rate < 8 or self.respiratory_rate > 30,
            self.spo2 < 0.88,
            self.temperature_celsius < 35.0 or self.temperature_celsius > 40.5,
            self.gcs <= 8,
        ])

    def mean_arterial_pressure(self) -> float:
        return self.bp_diastolic + (self.bp_systolic - self.bp_diastolic) / 3.0

    def to_normalized_vector(self) -> List[float]:
        return [
            self.heart_rate / 300.0, self.bp_systolic / 300.0,
            self.bp_diastolic / 200.0, self.respiratory_rate / 60.0,
            self.spo2, (self.temperature_celsius - 25.0) / 20.0,
            (self.gcs - 3) / 12.0, self.pain_score / 10.0,
        ]

    def shock_index(self) -> float:
        if self.bp_systolic == 0:
            return float("inf")
        return self.heart_rate / self.bp_systolic

class Demographics(BaseModel):
    age: int = Field(..., ge=0, le=120)
    sex: SexEnum
    weight_kg: float = Field(..., ge=0.5, le=500.0)
    height_cm: float = Field(..., ge=30.0, le=250.0)
    pregnant: Optional[bool] = None

    def bmi(self) -> float:
        if self.height_cm == 0:
            return 0.0
        return self.weight_kg / ((self.height_cm / 100.0) ** 2)

    def is_paediatric(self) -> bool:
        return self.age < 18

    def is_geriatric(self) -> bool:
        return self.age >= 65

class Comorbidity(BaseModel):
    icd10_code: str = Field(..., min_length=3, max_length=10)
    name: str = Field(..., min_length=2, max_length=200)
    active: bool = True
    on_treatment: bool = False

    @field_validator("icd10_code")
    @classmethod
    def validate_icd10_format(cls, v: str) -> str:
        v = v.upper().strip()
        if not (v[0].isalpha() and v[1].isdigit()):
            raise ValueError(f"ICD-10 code '{v}' does not match expected format")
        return v

class Medication(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    dose: str = Field(..., min_length=1, max_length=100)
    route: str = Field(..., min_length=1, max_length=50)
    frequency: str = Field(..., min_length=1, max_length=50)
    last_given: Optional[str] = None

class ChiefComplaint(BaseModel):
    text: str = Field(..., min_length=3, max_length=500)
    onset_hours: float = Field(..., ge=0.0)
    severity_0_10: int = Field(..., ge=0, le=10)
    associated_symptoms: List[str] = Field(default_factory=list)

    @field_validator("associated_symptoms")
    @classmethod
    def deduplicate_symptoms(cls, v: List[str]) -> List[str]:
        seen: set = set()
        result = []
        for sym in v:
            sym_lower = sym.lower().strip()
            if sym_lower not in seen:
                seen.add(sym_lower)
                result.append(sym.strip())
        return result

class GroundTruth(BaseModel):
    esi_level: TriageLevel
    primary_dx_icd10: str
    primary_dx_name: str
    differential_diagnoses: List[str] = Field(default_factory=list)
    required_diagnostics: List[str] = Field(default_factory=list)
    critical_interventions: List[str] = Field(default_factory=list)
    correct_disposition: DispositionEnum
    hidden_findings: List[str] = Field(default_factory=list)
    time_to_critical_intervention_minutes: Optional[int] = None

class PatientCase(BaseModel):
    case_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    demographics: Demographics
    vitals: VitalSigns
    chief_complaint: ChiefComplaint
    comorbidities: List[Comorbidity] = Field(default_factory=list)
    current_medications: List[Medication] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    available_diagnostics: List[str] = Field(default_factory=list)
    ground_truth: GroundTruth
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM
    presentation_category: PresentationCategory = PresentationCategory.CARDIAC

    def to_observation(self, step: int = 0, elapsed_seconds: int = 0) -> "PatientObservation":
        return PatientObservation(
            case_id=self.case_id, step=step, elapsed_time_seconds=elapsed_seconds,
            demographics=self.demographics, vitals=self.vitals,
            chief_complaint=self.chief_complaint, comorbidities=self.comorbidities,
            current_medications=self.current_medications, allergies=self.allergies,
            available_diagnostics=self.available_diagnostics,
        )

    def get_ground_truth(self) -> GroundTruth:
        return self.ground_truth

class PatientObservation(BaseModel):
    case_id: str
    step: int = Field(..., ge=0)
    elapsed_time_seconds: int = Field(..., ge=0)
    demographics: Demographics
    vitals: VitalSigns
    chief_complaint: ChiefComplaint
    comorbidities: List[Comorbidity] = Field(default_factory=list)
    current_medications: List[Medication] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    available_diagnostics: List[str] = Field(default_factory=list)

class DiagnosticOrder(BaseModel):
    test_name: str = Field(..., min_length=1, max_length=200)
    rationale: str = Field(..., min_length=5, max_length=500)
    expected_turnaround_minutes: int = Field(..., ge=1, le=1440)

class TreatmentAction(BaseModel):
    intervention: str = Field(..., min_length=2, max_length=300)
    dose_if_applicable: Optional[str] = Field(None, max_length=100)
    rationale: str = Field(..., min_length=5, max_length=500)
    priority: PriorityEnum = PriorityEnum.ROUTINE

class ClinicalNote(BaseModel):
    chief_complaint_summary: str = Field(..., min_length=20, max_length=500)
    history_of_present_illness: str = Field(..., min_length=100, max_length=3000)
    physical_exam_findings: str = Field(..., min_length=50, max_length=2000)
    assessment: str = Field(..., min_length=50, max_length=2000)
    plan: str = Field(..., min_length=50, max_length=2000)
    differential_diagnoses: List[str] = Field(..., description="1-5 differential diagnoses")

    @field_validator("differential_diagnoses")
    @classmethod
    def validate_differentials_count(cls, v: List[str]) -> List[str]:
        cleaned = [d.strip() for d in v if d.strip()]
        if not (1 <= len(cleaned) <= 5):
            raise ValueError("differential_diagnoses must contain between 1 and 5 entries")
        return cleaned

class TriageAction(BaseModel):
    assigned_triage_level: TriageLevel = TriageLevel.UNKNOWN
    diagnostic_orders: List[DiagnosticOrder] = Field(default_factory=list)
    treatments: List[TreatmentAction] = Field(default_factory=list)
    clinical_note: ClinicalNote
    disposition: DispositionEnum = DispositionEnum.PENDING
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    reasoning_chain: str = Field(..., min_length=50, max_length=5000)
    is_final: bool = False

    @model_validator(mode="after")
    def validate_final_action_completeness(self) -> "TriageAction":
        if self.is_final:
            if self.assigned_triage_level == TriageLevel.UNKNOWN:
                raise ValueError("is_final=True requires assigned_triage_level other than UNKNOWN")
            if self.disposition == DispositionEnum.PENDING:
                raise ValueError("is_final=True requires disposition other than PENDING")
        return self

class RewardSignal(BaseModel):
    total_reward: float = Field(..., ge=-1.0, le=1.0)
    scaled_score: float = Field(..., ge=0.0, le=100.0)
    triage_accuracy_score: float = Field(..., ge=0.0, le=1.0)
    documentation_score: float = Field(..., ge=0.0, le=1.0)
    diagnostic_appropriateness_score: float = Field(..., ge=0.0, le=1.0)
    treatment_safety_score: float = Field(..., ge=0.0, le=1.0)
    time_efficiency_score: float = Field(..., ge=0.0, le=1.0)
    penalty_flags: List[str] = Field(default_factory=list)
    penalty_total: float = Field(0.0, le=0.0)
    step: int = Field(..., ge=0)

    def is_passing(self, threshold: float = 0.7) -> bool:
        return self.total_reward >= threshold

class EpisodeTrajectory(BaseModel):
    trajectory_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    case_id: str
    patient_case: PatientCase
    actions: List[TriageAction] = Field(default_factory=list)
    reward_signals: List[RewardSignal] = Field(default_factory=list)
    observations: List[PatientObservation] = Field(default_factory=list)
    total_steps: int = 0
    total_elapsed_seconds: int = 0
    final_disposition: Optional[DispositionEnum] = None
    terminated_by: str = "unknown"
    cumulative_reward: float = 0.0

    def summary(self) -> Dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id, "case_id": self.case_id,
            "difficulty": self.patient_case.difficulty,
            "total_steps": self.total_steps,
            "total_elapsed_seconds": self.total_elapsed_seconds,
            "cumulative_reward": self.cumulative_reward,
            "final_disposition": self.final_disposition,
            "terminated_by": self.terminated_by,
            "penalty_flags": [f for rs in self.reward_signals for f in rs.penalty_flags],
        }

class ValidationResult(BaseModel):
    valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.valid = False
        self.errors.append(msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

class StepResult(BaseModel):
    observation: Optional[PatientObservation]
    reward: RewardSignal
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)
    validation: ValidationResult

class EnvConfig(BaseModel):
    max_steps: int = Field(10, ge=1, le=50)
    max_elapsed_seconds: int = Field(1800, ge=60)
    deterioration_enabled: bool = True
    deterioration_interval_steps: int = Field(3, ge=1)
    reward_shaping_enabled: bool = True
    allow_multi_step: bool = True
    strict_action_validation: bool = True
    log_level: str = "INFO"

class GeneratorConfig(BaseModel):
    seed: Optional[int] = None
    esi_distribution: Dict[str, float] = Field(default_factory=lambda: {
        "ESI_1": 0.05, "ESI_2": 0.15, "ESI_3": 0.40, "ESI_4": 0.25, "ESI_5": 0.15,
    })
    max_comorbidities: int = Field(5, ge=0, le=10)
    max_medications: int = Field(10, ge=0, le=20)
    inject_drug_interactions: bool = True
    data_dir: str = "data"

    @field_validator("esi_distribution")
    @classmethod
    def validate_distribution_sums_to_one(cls, v: Dict[str, float]) -> Dict[str, float]:
        total = sum(v.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"ESI distribution must sum to 1.0, got {total:.4f}")
        return v

class RewardConfig(BaseModel):
    weight_triage_accuracy: float = Field(0.30, ge=0.0, le=1.0)
    weight_documentation: float = Field(0.25, ge=0.0, le=1.0)
    weight_diagnostic_appropriateness: float = Field(0.20, ge=0.0, le=1.0)
    weight_treatment_safety: float = Field(0.15, ge=0.0, le=1.0)
    weight_time_efficiency: float = Field(0.10, ge=0.0, le=1.0)
    penalty_critical_miss: float = Field(-0.40, le=0.0)
    penalty_unsafe_discharge: float = Field(-0.50, le=0.0)
    penalty_allergy_violation: float = Field(-0.50, le=0.0)
    penalty_empty_documentation: float = Field(-0.20, le=0.0)
    penalty_overconfidence: float = Field(-0.05, le=0.0)
    penalty_under_triage: float = Field(-0.30, le=0.0)
    time_threshold_esi1: int = Field(60, ge=10)
    time_threshold_esi2: int = Field(60, ge=10)
    time_threshold_esi3: int = Field(120, ge=30)
    time_threshold_esi4: int = Field(300, ge=60)
    time_threshold_esi5: int = Field(300, ge=60)
    min_hpi_length: int = Field(100, ge=20)
    min_exam_length: int = Field(50, ge=10)
    min_assessment_length: int = Field(50, ge=10)
    min_plan_length: int = Field(50, ge=10)
    time_efficiency_floor: float = Field(0.2, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_weights_sum(self) -> "RewardConfig":
        total = (self.weight_triage_accuracy + self.weight_documentation +
                 self.weight_diagnostic_appropriateness + self.weight_treatment_safety +
                 self.weight_time_efficiency)
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Reward weights must sum to 1.0, got {total:.4f}")
        return self

    def time_threshold_for_esi(self, esi: TriageLevel) -> int:
        mapping = {
            TriageLevel.ESI_1: self.time_threshold_esi1,
            TriageLevel.ESI_2: self.time_threshold_esi2,
            TriageLevel.ESI_3: self.time_threshold_esi3,
            TriageLevel.ESI_4: self.time_threshold_esi4,
            TriageLevel.ESI_5: self.time_threshold_esi5,
        }
        return mapping.get(esi, self.time_threshold_esi3)