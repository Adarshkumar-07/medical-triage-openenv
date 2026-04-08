"""
agents/baseline_agent.py

Deterministic rule-based baseline agent implementing ESI triage heuristics.
Requires no model weights, no API calls, and no randomness.
"""
from __future__ import annotations

import textwrap
from typing import Dict, List, Optional, Tuple

from env.models import (
    ChiefComplaint, ClinicalNote, Demographics, DiagnosticOrder,
    DispositionEnum, PatientObservation, PriorityEnum, TriageAction,
    TriageLevel, TreatmentAction, VitalSigns,
)

_COMPLAINT_TO_DIAGNOSTICS: Dict[str, List[Tuple[str, int]]] = {
    "chest":       [("ECG",5),("Troponin",60),("Chest_Xray",30),("BMP",60),("CBC",60),("D_Dimer",60)],
    "cardiac":     [("ECG",5),("Troponin",60),("BMP",60),("CBC",60),("Chest_Xray",30)],
    "palpitat":    [("ECG",5),("BMP",60),("CBC",60),("Troponin",60),("TSH",90)],
    "headache":    [("CT_Head",45),("BMP",60),("CBC",60)],
    "head":        [("CT_Head",45),("BMP",60),("CBC",60)],
    "stroke":      [("CT_Head",45),("MRI_Brain",90),("CBC",60),("BMP",60),("ECG",5)],
    "weakness":    [("CT_Head",45),("BMP",60),("CBC",60),("Blood_glucose",5)],
    "seizure":     [("CT_Head",45),("BMP",60),("CBC",60),("Blood_glucose",5),("Toxicology",120)],
    "breath":      [("Chest_Xray",30),("ABG",15),("BMP",60),("CBC",60),("ECG",5),("D_Dimer",60)],
    "dyspnoea":    [("Chest_Xray",30),("ABG",15),("BMP",60),("CBC",60),("ECG",5)],
    "respiratory": [("Chest_Xray",30),("ABG",15),("BMP",60),("CBC",60)],
    "abdom":       [("CT_Abdomen_Pelvis",60),("CBC",60),("BMP",60),("Lipase",60),("Urinalysis",30)],
    "belly":       [("CT_Abdomen_Pelvis",60),("CBC",60),("BMP",60),("Lipase",60)],
    "nausea":      [("BMP",60),("CBC",60),("Lipase",60),("Urinalysis",30)],
    "vomit":       [("BMP",60),("CBC",60),("Lipase",60)],
    "urinar":      [("Urinalysis",30),("Urine_Culture",1440),("CBC",60),("BMP",60)],
    "dysuria":     [("Urinalysis",30),("Urine_Culture",1440),("CBC",60)],
    "fever":       [("CBC",60),("BMP",60),("Blood_Cultures",1440),("Urinalysis",30),("Chest_Xray",30),("Lactate",30)],
    "sepsis":      [("CBC",60),("BMP",60),("Blood_Cultures",1440),("Lactate",30),("Urinalysis",30)],
    "unconscious": [("Blood_glucose",5),("CT_Head",45),("BMP",60),("CBC",60),("Toxicology",120),("ABG",15)],
    "unresponsive":[("Blood_glucose",5),("CT_Head",45),("BMP",60),("CBC",60),("Toxicology",120)],
    "overdose":    [("Toxicology",120),("BMP",60),("CBC",60),("ECG",5),("Blood_glucose",5)],
    "ankle":       [("XRAY_Ankle",30)],
    "wrist":       [("XRAY_Wrist",30)],
    "throat":      [("Strep_rapid_test",15)],
    "pharyngi":    [("Strep_rapid_test",15)],
    "pulmonary":   [("CTPA",60),("D_Dimer",60),("ECG",5),("BNP",60),("Troponin",60)],
    "embolism":    [("CTPA",60),("D_Dimer",60),("ECG",5)],
    "lacerat":     [],
    "rash":        [],
}

_ESI_DEFAULT_DIAGNOSTICS: Dict[TriageLevel, List[Tuple[str, int]]] = {
    TriageLevel.ESI_1: [("CBC",60),("BMP",60),("Lactate",30),("Blood_glucose",5),("ABG",15),("ECG",5)],
    TriageLevel.ESI_2: [("CBC",60),("BMP",60),("ECG",5),("Troponin",60),("Chest_Xray",30)],
    TriageLevel.ESI_3: [("CBC",60),("BMP",60),("Urinalysis",30)],
    TriageLevel.ESI_4: [("Urinalysis",30)],
    TriageLevel.ESI_5: [],
}

_COMPLAINT_TO_DIFFERENTIALS: Dict[str, List[str]] = {
    "chest":       ["Acute coronary syndrome","Pulmonary embolism","Aortic dissection","Musculoskeletal chest pain","Oesophageal spasm"],
    "cardiac":     ["Acute myocardial infarction","Unstable angina","Pulmonary embolism","Pericarditis"],
    "palpitat":    ["Atrial fibrillation","Supraventricular tachycardia","Ventricular tachycardia","Hyperthyroidism","Anxiety"],
    "headache":    ["Subarachnoid haemorrhage","Migraine","Tension headache","Hypertensive emergency","Meningitis"],
    "stroke":      ["Ischaemic stroke","Haemorrhagic stroke","Todd's paralysis","Hypertensive emergency"],
    "breath":      ["Pulmonary embolism","Acute heart failure","Pneumonia","Asthma exacerbation","Pneumothorax"],
    "abdom":       ["Appendicitis","Bowel obstruction","Mesenteric ischaemia","Ovarian pathology","Renal colic"],
    "fever":       ["Bacterial sepsis","Pneumonia","Urinary tract infection","Cellulitis","Viral syndrome"],
    "unconscious": ["Hypoglycaemia","Opioid overdose","Haemorrhagic stroke","Status epilepticus","Severe sepsis"],
}

_ESI_TO_DISPOSITION: Dict[TriageLevel, DispositionEnum] = {
    TriageLevel.ESI_1: DispositionEnum.ADMIT_ICU,
    TriageLevel.ESI_2: DispositionEnum.ADMIT_FLOOR,
    TriageLevel.ESI_3: DispositionEnum.OBSERVATION,
    TriageLevel.ESI_4: DispositionEnum.DISCHARGE_HOME,
    TriageLevel.ESI_5: DispositionEnum.DISCHARGE_HOME,
}


class BaselineAgent:
    """Deterministic rule-based triage agent."""

    def __init__(self, single_step: bool = True, verbose: bool = False) -> None:
        self.single_step = single_step
        self.verbose = verbose
        self._step_within_episode: int = 0

    def act(self, observation: PatientObservation) -> TriageAction:
        self._step_within_episode += 1
        esi_level    = self._esi_heuristic(observation.vitals, observation.chief_complaint)
        diagnostics  = self._select_diagnostics(observation.chief_complaint, esi_level, observation.available_diagnostics)
        treatments   = self._select_treatments(observation.vitals, esi_level, observation.allergies)
        disposition  = _ESI_TO_DISPOSITION.get(esi_level, DispositionEnum.OBSERVATION)
        differentials = self._select_differentials(observation.chief_complaint)
        note         = self._generate_note(observation, esi_level, differentials)
        reasoning    = self._generate_reasoning(observation, esi_level, diagnostics, treatments)
        confidence   = self._estimate_confidence(observation.vitals, esi_level)
        is_final     = self.single_step or (self._step_within_episode >= 3)
        action = TriageAction(
            assigned_triage_level=esi_level, diagnostic_orders=diagnostics,
            treatments=treatments, clinical_note=note, disposition=disposition,
            confidence_score=confidence, reasoning_chain=reasoning, is_final=is_final,
        )
        if self.verbose:
            print(f"[BaselineAgent] Step {self._step_within_episode} | ESI={esi_level.value} | "
                  f"Disposition={disposition.value} | Confidence={confidence:.2f} | "
                  f"Diagnostics={[d.test_name for d in diagnostics]} | "
                  f"Treatments={[t.intervention for t in treatments]}")
        return action

    def reset(self) -> None:
        self._step_within_episode = 0

    def _esi_heuristic(self, vitals: VitalSigns, complaint: ChiefComplaint) -> TriageLevel:
        if self._is_esi1(vitals): return TriageLevel.ESI_1
        if self._is_esi2(vitals, complaint): return TriageLevel.ESI_2
        if self._is_esi3(vitals, complaint): return TriageLevel.ESI_3
        if self._is_esi4(vitals, complaint): return TriageLevel.ESI_4
        return TriageLevel.ESI_5

    @staticmethod
    def _is_esi1(v: VitalSigns) -> bool:
        return any([v.gcs<=8, v.spo2<0.88, v.heart_rate<40, v.heart_rate>150,
                    v.bp_systolic<80, v.respiratory_rate<8, v.respiratory_rate>30,
                    v.temperature_celsius>40.5, v.temperature_celsius<35.0])

    @staticmethod
    def _is_esi2(v: VitalSigns, c: ChiefComplaint) -> bool:
        borderline = any([8<v.gcs<=12, 0.88<=v.spo2<0.92, 80<=v.bp_systolic<90,
                          130<=v.heart_rate<=150, v.respiratory_rate>24, v.temperature_celsius>=39.5])
        severe_pain = c.severity_0_10 >= 8
        high_risk = any(kw in c.text.lower() for kw in [
            "arrest","fibrillation","pulseless","unresponsive","chest pain",
            "crushing","dissection","stroke","subarachnoid","worst headache",
            "haemoptysis","overdose","anaphylaxis"])
        return borderline or severe_pain or high_risk

    @staticmethod
    def _is_esi3(v: VitalSigns, c: ChiefComplaint) -> bool:
        moderate_pain = 4 <= c.severity_0_10 <= 7
        borderline = any([90<=v.heart_rate<=110, 0.92<=v.spo2<0.95, 90<=v.bp_systolic<100,
                          20<=v.respiratory_rate<=24, v.temperature_celsius>=38.5])
        multi_resource = any(kw in c.text.lower() for kw in [
            "abdom","fever","breath","palpitat","dizzy","back","flank",
            "urinar","vomit","nausea","syncope","confusion","weakness","swelling"])
        return moderate_pain or borderline or multi_resource

    @staticmethod
    def _is_esi4(v: VitalSigns, c: ChiefComplaint) -> bool:
        mild_pain = 1 <= c.severity_0_10 <= 3
        normal_vitals = (v.gcs==15 and v.spo2>=0.95 and 55<=v.heart_rate<=100
                         and v.bp_systolic>=100 and 12<=v.respiratory_rate<=20)
        one_resource = any(kw in c.text.lower() for kw in [
            "ankle","wrist","knee","throat","sore","rash","cut","lacerat","splinter","stitch","earache"])
        return (mild_pain and normal_vitals) or one_resource

    def _select_diagnostics(self, complaint: ChiefComplaint, esi_level: TriageLevel,
                             available: List[str]) -> List[DiagnosticOrder]:
        complaint_lower = complaint.text.lower()
        selected_tests: List[Tuple[str, int]] = []
        for keyword, tests in _COMPLAINT_TO_DIAGNOSTICS.items():
            if keyword in complaint_lower:
                selected_tests = tests
                break
        if not selected_tests:
            for symptom in complaint.associated_symptoms:
                for keyword, tests in _COMPLAINT_TO_DIAGNOSTICS.items():
                    if keyword in symptom.lower():
                        selected_tests = tests
                        break
                if selected_tests:
                    break
        if not selected_tests:
            selected_tests = _ESI_DEFAULT_DIAGNOSTICS.get(esi_level, [])
        max_tests = {TriageLevel.ESI_1:6, TriageLevel.ESI_2:5, TriageLevel.ESI_3:4,
                     TriageLevel.ESI_4:2, TriageLevel.ESI_5:1}.get(esi_level, 3)
        orders: List[DiagnosticOrder] = []
        for test_name, turnaround in selected_tests[:max_tests]:
            orders.append(DiagnosticOrder(
                test_name=test_name,
                rationale=self._diagnostic_rationale(test_name, complaint, esi_level),
                expected_turnaround_minutes=turnaround,
            ))
        return orders

    @staticmethod
    def _diagnostic_rationale(test_name: str, complaint: ChiefComplaint, esi_level: TriageLevel) -> str:
        rationale_map = {
            "ECG": "Evaluate for cardiac arrhythmia, ischaemia, and conduction abnormalities.",
            "Troponin": "Rule out acute myocardial injury given the clinical presentation.",
            "Chest_Xray": "Assess cardiac silhouette, pulmonary vasculature, exclude pneumothorax.",
            "CBC": "Evaluate for anaemia, leukocytosis suggesting infection, or thrombocytopenia.",
            "BMP": "Assess electrolytes, renal function, glucose, and acid-base status.",
            "D_Dimer": "Screen for pulmonary embolism or DVT given the clinical picture.",
            "CTPA": "Definitive imaging for pulmonary embolism given high pre-test probability.",
            "CT_Head": "Exclude intracranial haemorrhage, mass lesion, or acute stroke.",
            "MRI_Brain": "High-sensitivity imaging for acute ischaemic stroke.",
            "CT_Abdomen_Pelvis": "Evaluate for acute intra-abdominal pathology.",
            "ABG": "Assess oxygenation, ventilation, and acid-base status.",
            "Lactate": "Evaluate for tissue hypoperfusion and sepsis severity.",
            "Blood_Cultures": "Identify bacteraemia before initiating antibiotic therapy.",
            "Urinalysis": "Evaluate for urinary tract infection, haematuria, or renal pathology.",
            "Urine_Culture": "Identify causative organism and antibiotic sensitivities.",
            "Blood_glucose": "Rapid glucose check to exclude hypoglycaemia as reversible cause.",
            "Toxicology": "Broad toxicological screen given altered mental status.",
            "Lipase": "Evaluate for acute pancreatitis given abdominal pain.",
            "TSH": "Thyroid function for possible thyrotoxicosis or hypothyroidism.",
            "BNP": "Evaluate for acute heart failure as contributing aetiology.",
            "XRAY_Ankle": "Ottawa rules met; exclude fracture before mobilising patient.",
            "XRAY_Wrist": "Evaluate for distal radius or scaphoid fracture post-fall.",
            "Strep_rapid_test": "Rapid antigen detection to guide antibiotic prescribing.",
        }
        return rationale_map.get(
            test_name,
            f"Clinically indicated for evaluation of {complaint.text.lower()[:60]} "
            f"in this {esi_level.value} patient.",
        )

    def _select_treatments(self, vitals: VitalSigns, esi_level: TriageLevel,
                           allergies: List[str]) -> List[TreatmentAction]:
        allergies_upper = {a.upper().strip() for a in allergies if a.upper().strip() != "NKDA"}
        if esi_level == TriageLevel.ESI_1:
            return self._critical_resuscitation_bundle(vitals, allergies_upper)
        if esi_level == TriageLevel.ESI_2:
            return self._high_acuity_bundle(vitals, allergies_upper)
        if esi_level == TriageLevel.ESI_3:
            return self._moderate_acuity_bundle(vitals, allergies_upper)
        if vitals.pain_score > 2:
            if "IBUPROFEN" not in allergies_upper and "NSAID" not in allergies_upper:
                return [TreatmentAction(intervention="Ibuprofen 400mg PO", dose_if_applicable="400mg",
                                        rationale="Oral NSAID for mild-to-moderate pain in stable patient.",
                                        priority=PriorityEnum.ROUTINE)]
            return [TreatmentAction(intervention="Paracetamol 1g PO", dose_if_applicable="1g",
                                    rationale="Oral analgesia; NSAID avoided due to allergy.",
                                    priority=PriorityEnum.ROUTINE)]
        return []

    @staticmethod
    def _critical_resuscitation_bundle(vitals: VitalSigns, allergies_upper: set) -> List[TreatmentAction]:
        bundle = [
            TreatmentAction(intervention="IV Access — large-bore bilateral peripheral IV",
                            rationale="Immediate vascular access for resuscitation and medication delivery.",
                            priority=PriorityEnum.IMMEDIATE),
            TreatmentAction(intervention="High-flow oxygen via non-rebreather mask 15L/min",
                            dose_if_applicable="15L/min",
                            rationale="Maximise oxygen delivery in critically ill patient. Target SpO2 >= 94%.",
                            priority=PriorityEnum.IMMEDIATE),
            TreatmentAction(intervention="Continuous cardiac monitoring and pulse oximetry",
                            rationale="Continuous haemodynamic surveillance in ESI 1 patient.",
                            priority=PriorityEnum.IMMEDIATE),
        ]
        if vitals.bp_systolic < 90:
            bundle.append(TreatmentAction(intervention="IV Normal Saline 500ml bolus",
                                          dose_if_applicable="500ml",
                                          rationale=f"Fluid resuscitation for haemodynamic instability (SBP {vitals.bp_systolic} mmHg).",
                                          priority=PriorityEnum.IMMEDIATE))
        if vitals.gcs <= 12:
            bundle.append(TreatmentAction(intervention="Point-of-care blood glucose check",
                                          rationale=f"Exclude hypoglycaemia as reversible cause (GCS {vitals.gcs}).",
                                          priority=PriorityEnum.IMMEDIATE))
        return bundle

    @staticmethod
    def _high_acuity_bundle(vitals: VitalSigns, allergies_upper: set) -> List[TreatmentAction]:
        bundle = [
            TreatmentAction(intervention="IV Access — peripheral IV cannula",
                            rationale="Vascular access for medication delivery in ESI 2 patient.",
                            priority=PriorityEnum.IMMEDIATE),
            TreatmentAction(intervention="Supplemental oxygen 4L/min via nasal cannula",
                            dose_if_applicable="4L/min",
                            rationale=f"Supplemental oxygen for SpO2 {vitals.spo2:.1%}. Titrate to >= 94%.",
                            priority=PriorityEnum.URGENT),
            TreatmentAction(intervention="Continuous cardiac monitoring, SpO2, and BP monitoring",
                            rationale="Haemodynamic monitoring for high-acuity presentation.",
                            priority=PriorityEnum.URGENT),
        ]
        if vitals.pain_score >= 7:
            if "MORPHINE" not in allergies_upper and "OPIOID" not in allergies_upper:
                bundle.append(TreatmentAction(intervention="IV Morphine 2-4mg titrated to pain",
                                              dose_if_applicable="2-4mg IV",
                                              rationale=f"IV opioid analgesia for severe pain (score {vitals.pain_score}/10).",
                                              priority=PriorityEnum.URGENT))
            else:
                bundle.append(TreatmentAction(intervention="IV Ketorolac 15mg",
                                              dose_if_applicable="15mg IV",
                                              rationale=f"IV NSAID analgesia; opioid allergy documented.",
                                              priority=PriorityEnum.URGENT))
        return bundle

    @staticmethod
    def _moderate_acuity_bundle(vitals: VitalSigns, allergies_upper: set) -> List[TreatmentAction]:
        bundle: List[TreatmentAction] = []
        if vitals.heart_rate > 100 or vitals.bp_systolic < 100 or vitals.spo2 < 0.95:
            bundle.append(TreatmentAction(intervention="IV Access — peripheral IV cannula",
                                          rationale="Vascular access given borderline haemodynamic parameters.",
                                          priority=PriorityEnum.URGENT))
        if vitals.pain_score >= 4:
            if "IBUPROFEN" not in allergies_upper and "NSAID" not in allergies_upper:
                bundle.append(TreatmentAction(intervention="Ibuprofen 600mg PO with food",
                                              dose_if_applicable="600mg PO",
                                              rationale=f"Oral NSAID for moderate pain (score {vitals.pain_score}/10).",
                                              priority=PriorityEnum.ROUTINE))
            else:
                bundle.append(TreatmentAction(intervention="Paracetamol 1g PO",
                                              dose_if_applicable="1g PO",
                                              rationale=f"Oral analgesia; NSAID avoided.",
                                              priority=PriorityEnum.ROUTINE))
        return bundle

    def _select_differentials(self, complaint: ChiefComplaint) -> List[str]:
        complaint_lower = complaint.text.lower()
        for keyword, diffs in _COMPLAINT_TO_DIFFERENTIALS.items():
            if keyword in complaint_lower:
                return diffs[:5]
        for symptom in complaint.associated_symptoms:
            for keyword, diffs in _COMPLAINT_TO_DIFFERENTIALS.items():
                if keyword in symptom.lower():
                    return diffs[:5]
        return ["Primary presenting condition","Infectious aetiology",
                "Cardiovascular cause","Metabolic derangement"]

    def _generate_note(self, obs: PatientObservation, esi_level: TriageLevel,
                       differentials: List[str]) -> ClinicalNote:
        v = obs.vitals; d = obs.demographics; cc = obs.chief_complaint
        comorbidities_text = (", ".join(c.name for c in obs.comorbidities)
                               if obs.comorbidities else "None documented")
        medications_text = (", ".join(f"{m.name} {m.dose} {m.route} {m.frequency}"
                                       for m in obs.current_medications)
                             if obs.current_medications else "None")
        allergies_text = ", ".join(obs.allergies) if obs.allergies else "NKDA"
        associated_text = ", ".join(cc.associated_symptoms) if cc.associated_symptoms else "none reported"
        pregnancy_text = "The patient is pregnant." if d.pregnant else ""

        cc_summary = (f"{d.age}-year-old {d.sex.value} presenting with {cc.text.lower()} "
                      f"for approximately {cc.onset_hours:.1f} hours.")

        hpi = textwrap.dedent(f"""
            {d.age}-year-old {d.sex.value} with a background of {comorbidities_text}
            presents to the emergency department with a chief complaint of {cc.text.lower()}.
            Symptom onset approximately {cc.onset_hours:.1f} hours prior to arrival.
            Severity rated {cc.severity_0_10}/10 on the numeric pain scale.
            Associated symptoms include: {associated_text}.
            Current medications: {medications_text}.
            Documented allergies: {allergies_text}. {pregnancy_text}
            The patient denies recent travel or sick contacts unless noted above.
        """).strip()

        gcs_desc = "Alert and oriented" if v.gcs == 15 else f"GCS {v.gcs}"
        distress_desc = ("Appears in acute distress." if cc.severity_0_10 >= 7
                         else "Appears in mild-to-moderate discomfort." if cc.severity_0_10 >= 4
                         else "Appears comfortable at rest.")
        diaphoretic = "and diaphoretic" if v.temperature_celsius > 38.5 else "and dry"
        breath_sounds = ("Breath sounds diminished — further assessment required."
                         if v.spo2 < 0.92 else "Breath sounds clear to auscultation bilaterally.")

        exam = textwrap.dedent(f"""
            General: Patient {gcs_desc}. {distress_desc}
            Vital Signs: HR {v.heart_rate} bpm, BP {v.bp_systolic}/{v.bp_diastolic} mmHg,
            RR {v.respiratory_rate} breaths/min, SpO2 {v.spo2 * 100:.1f}% on room air,
            Temperature {v.temperature_celsius:.1f}C, GCS {v.gcs}/15, Pain {v.pain_score}/10.
            Cardiovascular: Heart sounds present. No peripheral oedema on initial assessment.
            Respiratory: {breath_sounds}
            Abdomen: Soft to palpation on initial inspection.
            Neurological: {gcs_desc}, pupils not yet formally assessed.
            Skin: Warm {diaphoretic} to touch.
        """).strip()

        acuity_desc = {
            TriageLevel.ESI_1: "critically ill requiring immediate resuscitation",
            TriageLevel.ESI_2: "high acuity presentation requiring emergent evaluation",
            TriageLevel.ESI_3: "moderate acuity presentation requiring urgent workup",
            TriageLevel.ESI_4: "low acuity presentation requiring limited resources",
            TriageLevel.ESI_5: "minor presentation likely requiring no diagnostic resources",
        }.get(esi_level, "undifferentiated presentation")

        vitals_concern = ("Vital signs are critically abnormal and require immediate intervention."
                          if esi_level == TriageLevel.ESI_1
                          else "Vital signs are borderline and warrant close monitoring."
                          if esi_level == TriageLevel.ESI_2
                          else "Vital signs are relatively stable at this time.")

        assessment = textwrap.dedent(f"""
            This is a {d.age}-year-old {d.sex.value} presenting as a {acuity_desc}.
            {vitals_concern}
            The presentation is most consistent with the chief complaint of {cc.text.lower()}.
            Relevant background conditions include: {comorbidities_text}.
            ESI triage level assigned: {esi_level.value}.
            Differential diagnoses under consideration: {'; '.join(differentials)}.
        """).strip()

        disposition_target = _ESI_TO_DISPOSITION.get(esi_level, DispositionEnum.OBSERVATION)
        reassess_interval = ("15 minutes" if esi_level in (TriageLevel.ESI_1, TriageLevel.ESI_2)
                             else "30 minutes" if esi_level == TriageLevel.ESI_3
                             else "60 minutes")
        specialist_line = ("Emergent specialist consult to be activated immediately."
                           if esi_level in (TriageLevel.ESI_1, TriageLevel.ESI_2)
                           else "Specialist consult to be requested if workup supports indication.")

        plan = textwrap.dedent(f"""
            1. Initiate monitoring per ESI {esi_level.value} protocol.
            2. Order indicated diagnostic investigations as per clinical assessment above.
            3. Administer analgesia and symptomatic relief as clinically indicated.
            4. {specialist_line}
            5. Reassess vital signs every {reassess_interval}.
            6. Target disposition: {disposition_target.value.replace('_', ' ')}.
            7. Patient and family education regarding diagnosis, treatment plan,
               and return precautions to be provided prior to discharge or transfer.
            8. Medication reconciliation completed; allergies reviewed and confirmed.
        """).strip()

        return ClinicalNote(
            chief_complaint_summary=cc_summary, history_of_present_illness=hpi,
            physical_exam_findings=exam, assessment=assessment, plan=plan,
            differential_diagnoses=differentials[:5],
        )

    def _generate_reasoning(self, obs: PatientObservation, esi_level: TriageLevel,
                            diagnostics: List[DiagnosticOrder],
                            treatments: List[TreatmentAction]) -> str:
        v = obs.vitals
        hr_label = "tachycardic" if v.heart_rate > 100 else "bradycardic" if v.heart_rate < 60 else "normal rate"
        bp_label = "hypotensive" if v.bp_systolic < 90 else "hypertensive" if v.bp_systolic > 140 else "normotensive"
        rr_label = "tachypnoeic" if v.respiratory_rate > 20 else "bradypnoeic" if v.respiratory_rate < 12 else "normal"
        spo2_label = "critically low" if v.spo2 < 0.88 else "low" if v.spo2 < 0.94 else "acceptable"
        temp_label = "febrile" if v.temperature_celsius > 38.0 else "hypothermic" if v.temperature_celsius < 36.0 else "afebrile"
        gcs_label = "critically impaired" if v.gcs <= 8 else "impaired" if v.gcs <= 12 else "normal"
        pain_label = "severe" if v.pain_score >= 7 else "moderate" if v.pain_score >= 4 else "mild"

        vital_analysis = (
            f"HR {v.heart_rate} bpm ({hr_label}), BP {v.bp_systolic}/{v.bp_diastolic} mmHg ({bp_label}), "
            f"RR {v.respiratory_rate} ({rr_label}), SpO2 {v.spo2 * 100:.1f}% ({spo2_label}), "
            f"Temp {v.temperature_celsius:.1f}C ({temp_label}), GCS {v.gcs} ({gcs_label}), "
            f"Pain {v.pain_score}/10 ({pain_label})."
        )
        esi_rationale = {
            TriageLevel.ESI_1: (f"ESI 1 assigned: patient requires immediate life-saving intervention. "
                                f"Triggered by: {self._esi1_trigger_description(v)}."),
            TriageLevel.ESI_2: (f"ESI 2 assigned: high-risk presentation with potential for rapid deterioration. "
                                f"Severe pain ({v.pain_score}/10) and/or borderline vital signs indicate emergent evaluation."),
            TriageLevel.ESI_3: "ESI 3 assigned: stable but multiple ED resources anticipated.",
            TriageLevel.ESI_4: "ESI 4 assigned: stable, one resource anticipated.",
            TriageLevel.ESI_5: "ESI 5 assigned: stable, no resources anticipated.",
        }.get(esi_level, f"ESI {esi_level.value} assigned based on clinical assessment.")

        diag_text = (f"Diagnostic plan: ordering {len(diagnostics)} test(s) — "
                     + ", ".join(d.test_name for d in diagnostics)
                     + " — to evaluate the presenting complaint and exclude serious pathology."
                     if diagnostics else "No diagnostics ordered.")

        tx_text = (f"Treatment plan: {len(treatments)} intervention(s) initiated — "
                   + "; ".join(t.intervention for t in treatments) + "."
                   if treatments else "Supportive care only; no immediate interventions required.")

        disposition_val = _ESI_TO_DISPOSITION.get(esi_level, DispositionEnum.OBSERVATION).value
        allergies_note = (f"Allergy review: {', '.join(obs.allergies)} — accounted for in treatment selection."
                          if obs.allergies and obs.allergies != ["NKDA"]
                          else "No known drug allergies documented.")

        return "\n\n".join([
            f"VITAL SIGN ANALYSIS:\n{vital_analysis}",
            f"ESI CLASSIFICATION:\n{esi_rationale}",
            f"DIAGNOSTIC REASONING:\n{diag_text}",
            f"TREATMENT REASONING:\n{tx_text}",
            f"ALLERGY REVIEW:\n{allergies_note}",
            f"DISPOSITION:\nPlanned disposition: {disposition_val}. Will revise based on results.",
        ])

    @staticmethod
    def _esi1_trigger_description(v: VitalSigns) -> str:
        triggers = []
        if v.gcs <= 8:              triggers.append(f"GCS {v.gcs} <= 8")
        if v.spo2 < 0.88:           triggers.append(f"SpO2 {v.spo2:.1%} < 88%")
        if v.heart_rate < 40:       triggers.append(f"HR {v.heart_rate} < 40 bpm")
        if v.heart_rate > 150:      triggers.append(f"HR {v.heart_rate} > 150 bpm")
        if v.bp_systolic < 80:      triggers.append(f"SBP {v.bp_systolic} < 80 mmHg")
        if v.respiratory_rate < 8:  triggers.append(f"RR {v.respiratory_rate} < 8")
        if v.respiratory_rate > 30: triggers.append(f"RR {v.respiratory_rate} > 30")
        if v.temperature_celsius > 40.5: triggers.append(f"Temp {v.temperature_celsius}C > 40.5C")
        if v.temperature_celsius < 35.0: triggers.append(f"Temp {v.temperature_celsius}C < 35.0C")
        return "; ".join(triggers) if triggers else "multiple critical vital sign abnormalities"

    @staticmethod
    def _estimate_confidence(vitals: VitalSigns, esi_level: TriageLevel) -> float:
        if esi_level == TriageLevel.ESI_1:
            n_criteria = sum([vitals.gcs<=8, vitals.spo2<0.88,
                              vitals.heart_rate<40 or vitals.heart_rate>150,
                              vitals.bp_systolic<80,
                              vitals.respiratory_rate<8 or vitals.respiratory_rate>30])
            return min(0.95, 0.60 + n_criteria * 0.10)
        if esi_level == TriageLevel.ESI_5:
            all_normal = (60<=vitals.heart_rate<=90 and vitals.bp_systolic>=110
                          and vitals.spo2>=0.97 and vitals.gcs==15 and vitals.pain_score<=2)
            return 0.85 if all_normal else 0.65
        return {TriageLevel.ESI_2: 0.75, TriageLevel.ESI_3: 0.70, TriageLevel.ESI_4: 0.72}.get(esi_level, 0.70)