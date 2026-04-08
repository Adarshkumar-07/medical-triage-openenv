"""
tests/test_env.py

Integration and unit tests for the TriageEnv environment.
86 tests covering lifecycle, observation schema, action validation,
reward signals, patient generator, reward engine, baseline agent,
grader integration, and multi-patient episodes.
"""
from __future__ import annotations

import time
from typing import List, Optional

import pytest

from env.models import (
    ChiefComplaint, ClinicalNote, Demographics, DifficultyLevel,
    DispositionEnum, EnvConfig, GeneratorConfig, PatientObservation,
    PriorityEnum, RewardConfig, RewardSignal, TriageAction, TriageLevel,
    TreatmentAction,
)
from env.patient_generator import PatientGenerator
from env.reward import RewardEngine
from env.triage_env import TriageEnv
from agents.baseline_agent import BaselineAgent
from graders import GraderRegistry, GraderTier
from graders.easy_grader import EasyGrader
from graders.medium_grader import MediumGrader
from graders.hard_grader import HardGrader


def _make_note(
    cc: str = "Patient presents with severe chest pain radiating to the left arm.",
    hpi: str = (
        "A 55-year-old male presents with sudden onset crushing chest pain radiating "
        "to the left arm, onset 2 hours ago, associated with diaphoresis and nausea. "
        "Severity 9/10. No prior cardiac history. On aspirin and atorvastatin. "
        "No known drug allergies. Pain is constant and not relieved by rest."
    ),
    exam: str = (
        "Alert and oriented, appears diaphoretic and pale. Heart rate regular. "
        "Breath sounds clear bilaterally. Abdomen soft and non-tender. "
        "No peripheral oedema. GCS 15. Vital signs documented above."
    ),
    assessment: str = (
        "Presentation is consistent with acute coronary syndrome. Differential "
        "includes aortic dissection, pulmonary embolism, and unstable angina. "
        "Requires emergent cardiac evaluation and immediate workup."
    ),
    plan: str = (
        "ECG and troponin stat. Aspirin 300mg PO if no contraindication. "
        "IV access established. Continuous cardiac monitoring. Cardiology "
        "consult activated. Admit to monitored bed for further evaluation."
    ),
    differentials: Optional[List[str]] = None,
) -> ClinicalNote:
    return ClinicalNote(
        chief_complaint_summary=cc,
        history_of_present_illness=hpi,
        physical_exam_findings=exam,
        assessment=assessment,
        plan=plan,
        differential_diagnoses=differentials or ["Acute MI", "Unstable angina", "Pulmonary embolism"],
    )


def _make_action(
    esi: TriageLevel = TriageLevel.ESI_3,
    disposition: DispositionEnum = DispositionEnum.OBSERVATION,
    is_final: bool = False,
    confidence: float = 0.7,
    treatments: Optional[List[TreatmentAction]] = None,
) -> TriageAction:
    return TriageAction(
        assigned_triage_level=esi,
        diagnostic_orders=[],
        treatments=treatments or [],
        clinical_note=_make_note(),
        disposition=disposition,
        confidence_score=confidence,
        reasoning_chain=(
            "VITAL SIGN ANALYSIS: Vital signs reviewed and interpreted in context. "
            "ESI CLASSIFICATION: Level assigned based on acuity and resource needs. "
            "DIAGNOSTIC REASONING: Tests ordered to evaluate primary and differentials. "
            "TREATMENT REASONING: Interventions selected based on presentation. "
            "ALLERGY REVIEW: No known drug allergies. "
            "DISPOSITION: Disposition based on ESI level and clinical trajectory."
        ),
        is_final=is_final,
    )


@pytest.fixture
def env() -> TriageEnv:
    return TriageEnv(seed=42)


@pytest.fixture
def fast_env() -> TriageEnv:
    return TriageEnv(env_config=EnvConfig(max_steps=3, deterioration_enabled=False), seed=0)


@pytest.fixture
def agent() -> BaselineAgent:
    return BaselineAgent(single_step=True, verbose=False)


@pytest.fixture
def final_action() -> TriageAction:
    return _make_action(esi=TriageLevel.ESI_3, disposition=DispositionEnum.ADMIT_FLOOR,
                        is_final=True, confidence=0.75)


class TestEpisodeLifecycle:

    def test_reset_returns_observation(self, env): assert isinstance(env.reset(), PatientObservation)

    def test_reset_observation_has_required_fields(self, env):
        obs = env.reset()
        assert obs.case_id != "" and obs.step == 0
        assert isinstance(obs.demographics, Demographics)
        assert isinstance(obs.vitals.heart_rate, int)

    def test_observation_does_not_expose_ground_truth(self, env):
        obs = env.reset()
        assert not hasattr(obs, "ground_truth") and not hasattr(obs, "hidden_findings")

    def test_step_returns_step_result(self, env, final_action):
        env.reset()
        result = env.step(final_action)
        assert result.done is not None and isinstance(result.reward, RewardSignal)

    def test_step_after_done_raises(self, env, final_action):
        env.reset(); env.step(final_action)
        with pytest.raises(RuntimeError, match="already done"):
            env.step(final_action)

    def test_step_before_reset_raises(self):
        with pytest.raises(RuntimeError, match="No active episode"):
            TriageEnv(seed=1).step(_make_action(is_final=True))

    def test_episode_terminates_on_is_final(self, env, final_action):
        env.reset(); result = env.step(final_action)
        assert result.done is True and env.state()["terminated_by"] == "agent_final"

    def test_episode_terminates_on_max_steps(self, fast_env):
        fast_env.reset(); result = None
        for _ in range(3):
            result = fast_env.step(_make_action(is_final=False))
        assert result.done is True and fast_env.state()["terminated_by"] == "max_steps"

    def test_state_before_reset(self):
        assert TriageEnv(seed=10).state()["status"] == "not_started"

    def test_state_during_active_episode(self, env):
        env.reset(); state = env.state()
        assert state["status"] == "active" and state["step"] == 0

    def test_state_after_episode_done(self, env, final_action):
        env.reset(); env.step(final_action)
        assert env.state()["status"] == "done"

    def test_sequential_resets_different_cases(self, env):
        assert env.reset(seed=100).case_id != env.reset(seed=200).case_id

    def test_same_seed_same_vitals(self):
        e1, e2 = TriageEnv(seed=99), TriageEnv(seed=99)
        o1, o2 = e1.reset(seed=99), e2.reset(seed=99)
        assert o1.vitals.heart_rate == o2.vitals.heart_rate

    def test_current_observation_after_reset(self, env):
        obs = env.reset()
        assert env.current_observation().case_id == obs.case_id

    def test_current_observation_before_reset_is_none(self):
        assert TriageEnv().current_observation() is None

    def test_get_trajectory_after_episode(self, env, final_action):
        env.reset(); env.step(final_action)
        traj = env.get_trajectory()
        assert len(traj.actions) == 1 and traj.terminated_by == "agent_final"

    def test_render_text_mode(self, env):
        env.reset(); assert "TriageEnv" in env.render(mode="text")

    def test_render_json_mode(self, env):
        import json; env.reset()
        assert "session_id" in json.loads(env.render(mode="json"))

    def test_render_before_reset(self):
        assert "reset" in TriageEnv().render().lower()

    def test_close_does_not_raise(self, env):
        env.reset(); env.close()


class TestObservationSchema:

    @pytest.mark.parametrize("difficulty", [DifficultyLevel.EASY, DifficultyLevel.MEDIUM, DifficultyLevel.HARD])
    def test_observation_valid_for_all_difficulties(self, difficulty):
        obs = TriageEnv(seed=7).reset(difficulty=difficulty)
        assert 0 <= obs.demographics.age <= 120 and 3 <= obs.vitals.gcs <= 15

    def test_vitals_physiologically_plausible(self, env):
        for seed in range(20):
            obs = TriageEnv(seed=seed).reset()
            assert obs.vitals.bp_diastolic < obs.vitals.bp_systolic
            assert 0.0 <= obs.vitals.spo2 <= 1.0

    def test_available_diagnostics_is_list_of_strings(self, env):
        assert all(isinstance(d, str) for d in env.reset().available_diagnostics)

    def test_comorbidities_icd10_format(self):
        for seed in range(20):
            obs = TriageEnv(seed=seed).reset(difficulty=DifficultyLevel.HARD)
            for c in obs.comorbidities:
                assert c.icd10_code[0].isalpha() and c.icd10_code[1].isdigit()

    def test_elapsed_time_zero_on_reset(self, env):
        assert env.reset().elapsed_time_seconds == 0


class TestActionValidation:

    def test_valid_action_passes_validation(self, env, final_action):
        env.reset(); result = env.step(final_action)
        assert result.validation.valid is True

    def test_short_reasoning_fails(self):
        with pytest.raises(Exception):
            TriageAction(assigned_triage_level=TriageLevel.ESI_3, clinical_note=_make_note(),
                         disposition=DispositionEnum.OBSERVATION, confidence_score=0.5,
                         reasoning_chain="Too short.", is_final=False)

    def test_final_with_unknown_esi_rejected(self):
        with pytest.raises(Exception):
            TriageAction(assigned_triage_level=TriageLevel.UNKNOWN, clinical_note=_make_note(),
                         disposition=DispositionEnum.PENDING, confidence_score=0.5,
                         reasoning_chain="A"*60, is_final=True)

    def test_invalid_confidence_rejected(self):
        with pytest.raises(Exception):
            TriageAction(assigned_triage_level=TriageLevel.ESI_3, clinical_note=_make_note(),
                         disposition=DispositionEnum.OBSERVATION, confidence_score=1.5,
                         reasoning_chain="A"*60, is_final=False)

    def test_empty_differentials_rejected(self):
        with pytest.raises(Exception):
            ClinicalNote(chief_complaint_summary="Patient presents with chest pain.",
                         history_of_present_illness="A"*110, physical_exam_findings="B"*55,
                         assessment="C"*55, plan="D"*55, differential_diagnoses=[])

    def test_too_many_differentials_rejected(self):
        with pytest.raises(Exception):
            ClinicalNote(chief_complaint_summary="Patient presents with chest pain.",
                         history_of_present_illness="A"*110, physical_exam_findings="B"*55,
                         assessment="C"*55, plan="D"*55,
                         differential_diagnoses=["A","B","C","D","E","F"])


class TestRewardSignal:

    def test_reward_signal_in_bounds(self, env, final_action):
        env.reset(); rs = env.step(final_action).reward
        assert -1.0 <= rs.total_reward <= 1.0 and 0.0 <= rs.scaled_score <= 100.0

    def test_allergy_violation_flag(self):
        for seed in range(50):
            new_env = TriageEnv(seed=seed)
            obs = new_env.reset()
            if not obs.allergies or obs.allergies == ["NKDA"]:
                continue
            bad_tx = TreatmentAction(intervention=f"{obs.allergies[0]} 500mg IV",
                                     dose_if_applicable="500mg",
                                     rationale="Test allergy violation trigger.",
                                     priority=PriorityEnum.URGENT)
            result = new_env.step(_make_action(esi=TriageLevel.ESI_3,
                                               disposition=DispositionEnum.OBSERVATION,
                                               is_final=True, treatments=[bad_tx]))
            assert "ALLERGY_VIOLATION" in result.reward.penalty_flags
            return
        pytest.skip("No allergic patient found")

    def test_unsafe_discharge_esi1_flagged(self):
        new_env = TriageEnv(seed=42)
        new_env.reset(esi_target=TriageLevel.ESI_1)
        result = new_env.step(_make_action(esi=TriageLevel.ESI_4,
                                           disposition=DispositionEnum.DISCHARGE_HOME, is_final=True))
        assert "UNSAFE_DISCHARGE" in result.reward.penalty_flags or "UNDER_TRIAGE" in result.reward.penalty_flags

    def test_reward_step_index_correct(self, env):
        env.reset()
        assert env.step(_make_action(is_final=True)).reward.step == 0

    def test_cumulative_reward_in_state(self, fast_env):
        fast_env.reset(); fast_env.step(_make_action(is_final=False))
        assert isinstance(fast_env.state()["cumulative_reward"], float)


class TestPatientGenerator:

    def test_generates_valid_case(self):
        case = PatientGenerator(seed=1).generate()
        assert case.case_id != "" and case.ground_truth is not None

    @pytest.mark.parametrize("difficulty", [DifficultyLevel.EASY, DifficultyLevel.MEDIUM, DifficultyLevel.HARD])
    def test_difficulty_controls_hidden_findings(self, difficulty):
        case = PatientGenerator(seed=5).generate(difficulty=difficulty)
        n = len(case.ground_truth.hidden_findings)
        if difficulty == DifficultyLevel.EASY: assert n == 0
        elif difficulty == DifficultyLevel.MEDIUM: assert n >= 1
        else: assert n >= 2

    def test_generate_batch_unique_ids(self):
        cases = PatientGenerator(seed=10).generate_batch(5)
        assert len({c.case_id for c in cases}) == 5

    def test_reseed_reproducibility(self):
        gen = PatientGenerator(seed=None)
        gen.reseed(42); c1 = gen.generate()
        gen.reseed(42); c2 = gen.generate()
        assert c1.vitals.heart_rate == c2.vitals.heart_rate

    @pytest.mark.parametrize("esi", [TriageLevel.ESI_1, TriageLevel.ESI_2, TriageLevel.ESI_3,
                                      TriageLevel.ESI_4, TriageLevel.ESI_5])
    def test_forced_esi_level(self, esi):
        assert PatientGenerator(seed=20).generate(esi_target=esi).ground_truth.esi_level == esi

    def test_vitals_always_consistent(self):
        gen = PatientGenerator(seed=0)
        for _ in range(50):
            v = gen.generate().vitals
            assert v.bp_diastolic < v.bp_systolic and 3 <= v.gcs <= 15

    def test_allergies_list_non_empty(self):
        gen = PatientGenerator(seed=4)
        for _ in range(10):
            assert len(gen.generate().allergies) >= 1


class TestRewardEngine:

    def test_exact_esi_match(self):
        assert RewardEngine()._score_triage_level(TriageLevel.ESI_2, TriageLevel.ESI_2) == 1.0

    def test_esi_off_by_one(self):
        assert RewardEngine()._score_triage_level(TriageLevel.ESI_2, TriageLevel.ESI_3) == 0.6

    def test_esi_off_by_two(self):
        assert RewardEngine()._score_triage_level(TriageLevel.ESI_1, TriageLevel.ESI_3) == 0.2

    def test_esi_off_by_three(self):
        assert RewardEngine()._score_triage_level(TriageLevel.ESI_1, TriageLevel.ESI_5) == 0.0

    def test_unknown_esi_zero(self):
        assert RewardEngine()._score_triage_level(TriageLevel.UNKNOWN, TriageLevel.ESI_3) == 0.0

    def test_documentation_score_adequate_note(self):
        eng = RewardEngine()
        assert eng._score_documentation(_make_note(), eng.config) > 0.5

    def test_allergy_violation_detected(self):
        tx = TreatmentAction(intervention="Penicillin 500mg IV", dose_if_applicable="500mg",
                             rationale="Antibiotic therapy.", priority=PriorityEnum.URGENT)
        assert RewardEngine()._check_allergy_violations([tx], ["Penicillin"]) is True

    def test_no_allergy_violation_nkda(self):
        tx = TreatmentAction(intervention="Penicillin 500mg IV", dose_if_applicable="500mg",
                             rationale="Antibiotic therapy.", priority=PriorityEnum.URGENT)
        assert RewardEngine()._check_allergy_violations([tx], ["NKDA"]) is False

    def test_time_efficiency_full_within_threshold(self):
        assert RewardEngine()._score_time_efficiency(30, TriageLevel.ESI_1, RewardEngine().config) == 1.0

    def test_time_efficiency_decays_over_threshold(self):
        eng = RewardEngine()
        assert eng._score_time_efficiency(600, TriageLevel.ESI_1, eng.config) < eng._score_time_efficiency(60, TriageLevel.ESI_1, eng.config)

    def test_time_efficiency_floor_respected(self):
        eng = RewardEngine()
        assert eng._score_time_efficiency(100_000, TriageLevel.ESI_1, eng.config) >= eng.config.time_efficiency_floor

    def test_trajectory_reward_single_signal(self):
        rs = RewardSignal(total_reward=0.8, scaled_score=90.0, triage_accuracy_score=1.0,
                          documentation_score=0.8, diagnostic_appropriateness_score=0.7,
                          treatment_safety_score=0.9, time_efficiency_score=0.8,
                          penalty_flags=[], penalty_total=0.0, step=0)
        assert RewardEngine().compute_trajectory_reward([rs]) == pytest.approx(0.8, abs=1e-3)

    def test_trajectory_reward_discounts_early_steps(self):
        def _rs(r, s): return RewardSignal(total_reward=r, scaled_score=(r+1)/2*100,
                                           triage_accuracy_score=0.5, documentation_score=0.5,
                                           diagnostic_appropriateness_score=0.5, treatment_safety_score=0.5,
                                           time_efficiency_score=0.5, penalty_flags=[], penalty_total=0.0, step=s)
        assert RewardEngine().compute_trajectory_reward([_rs(0.2,0), _rs(0.8,1)]) > 0.5


class TestBaselineAgent:

    @pytest.mark.parametrize("difficulty", [DifficultyLevel.EASY, DifficultyLevel.MEDIUM, DifficultyLevel.HARD])
    def test_agent_completes_episode(self, difficulty):
        env = TriageEnv(seed=42); agent = BaselineAgent(single_step=True)
        obs = env.reset(difficulty=difficulty); agent.reset()
        assert env.step(agent.act(obs)).done is True

    def test_agent_action_schema_valid(self):
        env = TriageEnv(seed=5); agent = BaselineAgent(single_step=True)
        obs = env.reset(); agent.reset(); action = agent.act(obs)
        assert action.assigned_triage_level != TriageLevel.UNKNOWN
        assert len(action.reasoning_chain) >= 50
        assert len(action.clinical_note.differential_diagnoses) >= 1

    def test_agent_produces_valid_esi(self):
        valid = {TriageLevel.ESI_1, TriageLevel.ESI_2, TriageLevel.ESI_3, TriageLevel.ESI_4, TriageLevel.ESI_5}
        agent = BaselineAgent(single_step=True)
        for seed in range(20):
            env = TriageEnv(seed=seed); obs = env.reset(); agent.reset()
            assert agent.act(obs).assigned_triage_level in valid

    def test_agent_note_meets_min_lengths(self):
        agent = BaselineAgent(single_step=True)
        for seed in range(10):
            obs = TriageEnv(seed=seed).reset(); agent.reset(); note = agent.act(obs).clinical_note
            assert (len(note.chief_complaint_summary) >= 20 and
                    len(note.history_of_present_illness) >= 100 and
                    len(note.physical_exam_findings) >= 50 and
                    len(note.assessment) >= 50 and len(note.plan) >= 50)

    def test_esi1_gets_critical_treatments(self):
        agent = BaselineAgent(single_step=True)
        for seed in range(10):
            obs = TriageEnv(seed=seed).reset(esi_target=TriageLevel.ESI_1); agent.reset()
            tx_text = " ".join(t.intervention.upper() for t in agent.act(obs).treatments)
            assert "IV" in tx_text or "OXYGEN" in tx_text or "O2" in tx_text

    def test_agent_reset_clears_counter(self):
        agent = BaselineAgent(single_step=False)
        obs = TriageEnv(seed=1).reset(); agent.reset(); agent.act(obs)
        assert agent._step_within_episode == 1
        agent.reset(); assert agent._step_within_episode == 0


class TestGraderIntegration:

    def _run_episode(self, seed=42, difficulty=DifficultyLevel.EASY, esi_target=None):
        env = TriageEnv(seed=seed); agent = BaselineAgent(single_step=True)
        obs = env.reset(difficulty=difficulty, esi_target=esi_target); agent.reset()
        env.step(agent.act(obs)); return env.get_trajectory()

    def test_easy_grader_grade_result(self):
        result = EasyGrader().grade(self._run_episode())
        assert 0.0 <= result.total_score <= 1.0 and isinstance(result.pass_fail, bool)

    def test_medium_grader_rubric_scores(self):
        result = MediumGrader().grade(self._run_episode(difficulty=DifficultyLevel.MEDIUM))
        assert {"triage_accuracy","documentation_quality","diagnostic_workup",
                "treatment_safety","time_to_disposition"}.issubset(result.rubric_scores.keys())

    def test_hard_grader_automated_fallback(self):
        result = HardGrader(config={"api_key":""}).grade(self._run_episode(difficulty=DifficultyLevel.HARD))
        assert 0.0 <= result.total_score <= 1.0 and result.llm_critique is None

    def test_grader_registry_easy(self): assert isinstance(GraderRegistry.from_string("easy"), EasyGrader)
    def test_grader_registry_medium(self): assert isinstance(GraderRegistry.from_string("medium"), MediumGrader)
    def test_grader_registry_hard(self): assert isinstance(GraderRegistry.from_string("hard"), HardGrader)

    def test_grader_registry_invalid(self):
        with pytest.raises(ValueError, match="Unknown grader tier"):
            GraderRegistry.from_string("extreme")

    def test_empty_trajectory_zero_score(self):
        env = TriageEnv(seed=1); env.reset(); traj = env.get_trajectory()
        for tier in ["easy","medium","hard"]:
            result = GraderRegistry.from_string(tier, config={"api_key":""}).grade(traj)
            assert result.total_score == 0.0 and result.pass_fail is False

    def test_all_rubric_scores_in_unit_interval(self):
        for seed in range(5):
            result = MediumGrader().grade(self._run_episode(seed=seed, difficulty=DifficultyLevel.MEDIUM))
            for dim, score in result.rubric_scores.items():
                assert 0.0 <= score <= 1.0, f"'{dim}' = {score} outside [0,1]"

    def test_baseline_passes_some_easy_cases(self):
        grader = EasyGrader(); passes = 0
        for seed in range(15):
            try:
                if grader.grade(self._run_episode(seed=seed, difficulty=DifficultyLevel.EASY)).pass_fail:
                    passes += 1
            except Exception:
                pass
        assert passes >= 3, f"BaselineAgent passed only {passes}/15 easy cases"

    def test_critical_miss_caps_hard_grader(self):
        env = TriageEnv(seed=42); env.reset(esi_target=TriageLevel.ESI_1)
        env.step(_make_action(esi=TriageLevel.ESI_5, disposition=DispositionEnum.DISCHARGE_HOME, is_final=True))
        result = HardGrader(config={"api_key":""}).grade(env.get_trajectory())
        assert result.total_score <= HardGrader.CRITICAL_MISS_CAP + 0.05


class TestMultiPatientEpisodes:

    def test_ten_sequential_episodes(self):
        env = TriageEnv(seed=0); agent = BaselineAgent(single_step=True); scores = []
        for ep in range(10):
            obs = env.reset(seed=ep); agent.reset()
            result = env.step(agent.act(obs))
            assert result.done; scores.append(result.reward.total_reward)
        assert all(-1.0 <= s <= 1.0 for s in scores)

    def test_episode_case_ids_unique(self):
        env = TriageEnv(seed=0)
        assert len({env.reset(seed=i).case_id for i in range(20)}) == 20

    def test_all_difficulties_different_hidden_finding_counts(self):
        gen = PatientGenerator(seed=99); counts = {}
        for d in [DifficultyLevel.EASY, DifficultyLevel.MEDIUM, DifficultyLevel.HARD]:
            cases = gen.generate_batch(5, difficulty=d)
            counts[d] = sum(len(c.ground_truth.hidden_findings) for c in cases) / len(cases)
        assert counts[DifficultyLevel.EASY] == 0.0
        assert counts[DifficultyLevel.MEDIUM] >= 0.5
        assert counts[DifficultyLevel.HARD] >= counts[DifficultyLevel.MEDIUM]