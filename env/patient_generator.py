from __future__ import annotations
import json, math, random, uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from env.models import (
    ChiefComplaint, Comorbidity, Demographics, DifficultyLevel,
    GeneratorConfig, GroundTruth, Medication, PatientCase,
    PresentationCategory, DispositionEnum, SexEnum, TriageLevel, VitalSigns,
)

_SYMPTOM_PROFILES: Dict[str, Any] = {
    "ESI_1": {
        "cardiac": {
            "complaints": ["Cardiac arrest, unresponsive","Ventricular fibrillation found by EMS","Pulseless electrical activity"],
            "icd10": "I46.9","dx_name": "Cardiac arrest, unspecified",
            "required_diagnostics": ["ECG","Troponin","BMP","CBC","Chest_Xray"],
            "critical_interventions": ["CPR","Defibrillation","IV_Access","Intubation"],
            "disposition": DispositionEnum.ADMIT_ICU,
            "differentials": ["Massive pulmonary embolism","Tension pneumothorax","Hypovolaemic shock","Aortic dissection"],
        },
        "respiratory": {
            "complaints": ["Severe respiratory distress, SpO2 75%","Acute respiratory failure","Status asthmaticus unresponsive to nebulisers"],
            "icd10": "J96.00","dx_name": "Acute respiratory failure, unspecified",
            "required_diagnostics": ["ABG","Chest_Xray","BMP","CBC","ECG"],
            "critical_interventions": ["High_Flow_O2","Intubation","Bronchodilators","IV_Access"],
            "disposition": DispositionEnum.ADMIT_ICU,
            "differentials": ["Tension pneumothorax","Pulmonary embolism","Severe pneumonia","ARDS"],
        },
        "neurological": {
            "complaints": ["Unresponsive, GCS 3","Active generalised seizure","Sudden onset coma"],
            "icd10": "R55","dx_name": "Altered level of consciousness, severe",
            "required_diagnostics": ["CT_Head","BMP","CBC","Blood_glucose","Toxicology"],
            "critical_interventions": ["Airway_protection","IV_Access","Glucose_check","Benzodiazepine_if_seizing"],
            "disposition": DispositionEnum.ADMIT_ICU,
            "differentials": ["Haemorrhagic stroke","Status epilepticus","Hypoglycaemia","Opioid overdose"],
        },
    },
    "ESI_2": {
        "cardiac": {
            "complaints": ["Severe crushing chest pain radiating to left arm","Diaphoresis with chest tightness and nausea","Chest pain with ST elevation on prehospital ECG"],
            "icd10": "I21.9","dx_name": "Acute myocardial infarction, unspecified",
            "required_diagnostics": ["ECG","Troponin","BMP","CBC","Chest_Xray"],
            "critical_interventions": ["Aspirin","Heparin","IV_Access","Cardiology_consult"],
            "disposition": DispositionEnum.ADMIT_ICU,
            "differentials": ["Unstable angina","Aortic dissection","Pulmonary embolism","Oesophageal spasm"],
        },
        "neurological": {
            "complaints": ["Sudden onset worst headache of life","Right-sided weakness and slurred speech","Acute facial droop and arm drift"],
            "icd10": "I63.9","dx_name": "Cerebral infarction, unspecified",
            "required_diagnostics": ["CT_Head","MRI_Brain","CBC","BMP","ECG"],
            "critical_interventions": ["tPA_evaluation","IV_Access","Neurology_consult","BP_management"],
            "disposition": DispositionEnum.ADMIT_FLOOR,
            "differentials": ["Haemorrhagic stroke","Todd's paralysis","Hypertensive emergency","Brain tumour"],
        },
        "respiratory": {
            "complaints": ["Sudden onset pleuritic chest pain and shortness of breath","Haemoptysis with dyspnoea after long flight","Dyspnoea, tachycardia, and calf swelling"],
            "icd10": "I26.99","dx_name": "Pulmonary embolism without acute cor pulmonale",
            "required_diagnostics": ["CTPA","D_Dimer","BNP","Troponin","ECG"],
            "critical_interventions": ["Anticoagulation","IV_Access","O2_supplementation"],
            "disposition": DispositionEnum.ADMIT_FLOOR,
            "differentials": ["Pneumothorax","Pneumonia","Pericarditis","Musculoskeletal chest pain"],
        },
    },
    "ESI_3": {
        "gastrointestinal": {
            "complaints": ["Severe abdominal pain, right lower quadrant, 18 hours","Nausea, vomiting, and periumbilical pain migrating to RLQ","Anorexia with progressive right-sided abdominal pain"],
            "icd10": "K37","dx_name": "Unspecified appendicitis",
            "required_diagnostics": ["CT_Abdomen_Pelvis","CBC","BMP","Urinalysis","Lipase"],
            "critical_interventions": ["IV_Access","Surgical_consult","NPO"],
            "disposition": DispositionEnum.ADMIT_FLOOR,
            "differentials": ["Ovarian cyst rupture","Pelvic inflammatory disease","Renal colic","Mesenteric lymphadenitis"],
        },
        "infectious": {
            "complaints": ["Fever, chills, and dysuria for 3 days","Flank pain with nausea and fever","Burning urination with back pain and rigors"],
            "icd10": "N10","dx_name": "Acute pyelonephritis",
            "required_diagnostics": ["Urinalysis","Urine_Culture","CBC","BMP","Blood_Cultures"],
            "critical_interventions": ["IV_Antibiotics","IV_Fluids","IV_Access"],
            "disposition": DispositionEnum.ADMIT_FLOOR,
            "differentials": ["Renal abscess","Nephrolithiasis","Ovarian torsion","Appendicitis"],
        },
        "cardiac": {
            "complaints": ["Palpitations and light-headedness for 2 hours","Rapid irregular heartbeat with mild dyspnoea","Heart racing with near-syncope"],
            "icd10": "I48.91","dx_name": "Unspecified atrial fibrillation",
            "required_diagnostics": ["ECG","BMP","CBC","Troponin","TSH"],
            "critical_interventions": ["IV_Access","Rate_control","Anticoagulation_evaluation"],
            "disposition": DispositionEnum.OBSERVATION,
            "differentials": ["SVT","Ventricular tachycardia","Hyperthyroidism","Pulmonary embolism"],
        },
    },
    "ESI_4": {
        "musculoskeletal": {
            "complaints": ["Twisted ankle after sport, can partially weight-bear","Wrist pain after fall on outstretched hand","Knee pain after twisting injury playing football"],
            "icd10": "S93.409A","dx_name": "Sprain of unspecified ligament of ankle",
            "required_diagnostics": ["XRAY_Ankle"],
            "critical_interventions": ["Analgesia","RICE"],
            "disposition": DispositionEnum.DISCHARGE_HOME,
            "differentials": ["Fracture","Tendon rupture","Contusion"],
        },
        "infectious": {
            "complaints": ["Sore throat, fever, and white tonsillar exudate","Painful swallowing with cervical lymphadenopathy","Throat pain and fever for 2 days"],
            "icd10": "J02.9","dx_name": "Acute pharyngitis, unspecified",
            "required_diagnostics": ["Strep_rapid_test"],
            "critical_interventions": ["Antibiotics_if_strep_positive","Analgesia"],
            "disposition": DispositionEnum.DISCHARGE_HOME,
            "differentials": ["Peritonsillar abscess","Infectious mononucleosis","Viral pharyngitis"],
        },
    },
    "ESI_5": {
        "dermatological": {
            "complaints": ["Small laceration on finger, controlled bleeding","Minor rash on arm, no systemic symptoms","Insect bite with local swelling, no anaphylaxis"],
            "icd10": "S61.209A","dx_name": "Unspecified open wound of finger",
            "required_diagnostics": [],
            "critical_interventions": ["Wound_irrigation","Closure","Tetanus_if_due"],
            "disposition": DispositionEnum.DISCHARGE_HOME,
            "differentials": ["Abrasion","Contusion"],
        },
        "musculoskeletal": {
            "complaints": ["Chronic low back pain flare, ambulatory, stable","Mild neck stiffness after sleeping awkwardly","Sore muscles after exercise yesterday"],
            "icd10": "M54.5","dx_name": "Low back pain",
            "required_diagnostics": [],
            "critical_interventions": ["NSAIDs","Muscle_relaxant_if_indicated"],
            "disposition": DispositionEnum.DISCHARGE_PCP_FOLLOWUP,
            "differentials": ["Herniated disc","Muscle strain","Sciatica"],
        },
    },
}

_COMORBIDITIES_POOL: List[Dict[str, str]] = [
    {"icd10_code": "I10", "name": "Essential hypertension"},
    {"icd10_code": "E11.9", "name": "Type 2 diabetes mellitus without complications"},
    {"icd10_code": "J44.1", "name": "Chronic obstructive pulmonary disease with exacerbation"},
    {"icd10_code": "I25.10", "name": "Atherosclerotic heart disease of native coronary artery"},
    {"icd10_code": "N18.3", "name": "Chronic kidney disease, stage 3"},
    {"icd10_code": "F32.9", "name": "Major depressive disorder, single episode"},
    {"icd10_code": "M81.0", "name": "Age-related osteoporosis"},
    {"icd10_code": "I48.91", "name": "Unspecified atrial fibrillation"},
    {"icd10_code": "E78.5", "name": "Hyperlipidaemia, unspecified"},
    {"icd10_code": "K21.0", "name": "Gastro-oesophageal reflux disease with oesophagitis"},
    {"icd10_code": "J45.909", "name": "Unspecified asthma, uncomplicated"},
    {"icd10_code": "G40.909", "name": "Epilepsy, unspecified, not intractable"},
    {"icd10_code": "Z87.891", "name": "History of nicotine dependence"},
]

_MEDICATIONS_POOL: List[Dict[str, str]] = [
    {"name": "Metformin", "dose": "500mg", "route": "PO", "frequency": "BD"},
    {"name": "Lisinopril", "dose": "10mg", "route": "PO", "frequency": "OD"},
    {"name": "Atorvastatin", "dose": "40mg", "route": "PO", "frequency": "Nocte"},
    {"name": "Aspirin", "dose": "81mg", "route": "PO", "frequency": "OD"},
    {"name": "Metoprolol succinate", "dose": "50mg", "route": "PO", "frequency": "OD"},
    {"name": "Warfarin", "dose": "5mg", "route": "PO", "frequency": "OD"},
    {"name": "Apixaban", "dose": "5mg", "route": "PO", "frequency": "BD"},
    {"name": "Salbutamol inhaler", "dose": "100mcg", "route": "INH", "frequency": "PRN"},
    {"name": "Omeprazole", "dose": "20mg", "route": "PO", "frequency": "OD"},
    {"name": "Amlodipine", "dose": "5mg", "route": "PO", "frequency": "OD"},
    {"name": "Furosemide", "dose": "40mg", "route": "PO", "frequency": "OD"},
    {"name": "Sertraline", "dose": "50mg", "route": "PO", "frequency": "OD"},
    {"name": "Levothyroxine", "dose": "50mcg", "route": "PO", "frequency": "OD"},
    {"name": "Levetiracetam", "dose": "500mg", "route": "PO", "frequency": "BD"},
]

_ALLERGIES_POOL: List[str] = [
    "Penicillin","Sulfonamides","Aspirin","Ibuprofen","Codeine",
    "Morphine","Latex","Iodine contrast","Cephalosporins","Quinolones",
    "NKDA","NKDA","NKDA",
]

_AVAILABLE_DIAGNOSTICS_POOL: List[str] = [
    "CBC","BMP","LFT","Lipase","Troponin","BNP","TSH","D_Dimer","ECG",
    "Chest_Xray","CT_Head","CT_Abdomen_Pelvis","CTPA","MRI_Brain",
    "XRAY_Ankle","XRAY_Wrist","XRAY_Chest","Urinalysis","Urine_Culture",
    "Blood_Cultures","ABG","Lactate","Coagulation_panel","Toxicology",
    "Blood_glucose","Strep_rapid_test","Pregnancy_test",
]

_HIDDEN_FINDINGS_POOL: List[str] = [
    "Subtle ST depression in leads V4-V6 on initial ECG",
    "Occult drug-drug interaction between warfarin and new antibiotic",
    "Silent atrial fibrillation with rapid ventricular response on rhythm strip",
    "Troponin trending upward on repeat testing",
    "Hypokalaemia exacerbating QT prolongation on medications",
    "Unrecognised PE presenting as anxiety and mild dyspnoea",
    "Early appendicitis with atypical presentation in elderly patient",
    "Carbon monoxide poisoning in winter presentation of headache",
    "Ectopic pregnancy in reproductive-age female with abdominal pain",
    "Posterior STEMI missed on standard 12-lead ECG",
    "Diabetic ketoacidosis underlying apparent gastroenteritis",
    "Epidural abscess in IV drug user with back pain",
    "Intracranial haemorrhage in anticoagulated patient after fall",
    "Occult sepsis with near-normal temperature in elderly",
    "Anaphylaxis evolving from initial mild allergic reaction",
]

# (mean, std, lo, hi)
_VITAL_PARAMS: Dict[str, Dict[str, Tuple[float,float,float,float]]] = {
    "ESI_1": {
        "heart_rate": (130.,20.,0.,250.),"bp_systolic": (75.,15.,50.,140.),
        "bp_diastolic": (45.,10.,30.,90.),"respiratory_rate": (28.,6.,4.,50.),
        "spo2": (.82,.07,.60,.94),"temperature_celsius": (38.8,.8,35.,41.5),
        "gcs": (4.,2.,3.,9.),"pain_score": (9.,1.,6.,10.),
    },
    "ESI_2": {
        "heart_rate": (115.,15.,60.,180.),"bp_systolic": (160.,30.,80.,220.),
        "bp_diastolic": (95.,15.,50.,130.),"respiratory_rate": (22.,4.,12.,35.),
        "spo2": (.93,.04,.82,.98),"temperature_celsius": (38.3,.7,36.,40.5),
        "gcs": (12.,2.,8.,15.),"pain_score": (8.,1.5,5.,10.),
    },
    "ESI_3": {
        "heart_rate": (95.,15.,55.,140.),"bp_systolic": (140.,20.,100.,190.),
        "bp_diastolic": (85.,12.,55.,115.),"respiratory_rate": (18.,3.,12.,26.),
        "spo2": (.96,.02,.90,.99),"temperature_celsius": (38.,.6,36.5,40.),
        "gcs": (14.5,.8,12.,15.),"pain_score": (6.,2.,3.,9.),
    },
    "ESI_4": {
        "heart_rate": (82.,12.,55.,110.),"bp_systolic": (125.,15.,100.,160.),
        "bp_diastolic": (78.,10.,55.,100.),"respiratory_rate": (15.,2.,12.,20.),
        "spo2": (.98,.01,.95,1.),"temperature_celsius": (37.3,.5,36.2,38.5),
        "gcs": (15.,0.,15.,15.),"pain_score": (4.,2.,1.,7.),
    },
    "ESI_5": {
        "heart_rate": (76.,10.,55.,100.),"bp_systolic": (120.,10.,100.,145.),
        "bp_diastolic": (75.,8.,55.,92.),"respiratory_rate": (14.,1.5,12.,18.),
        "spo2": (.99,.005,.97,1.),"temperature_celsius": (37.,.3,36.2,37.8),
        "gcs": (15.,0.,15.,15.),"pain_score": (2.,1.5,0.,5.),
    },
}


class PatientGenerator:
    def __init__(
        self,
        config: Optional[GeneratorConfig] = None,
        seed: Optional[int] = None,
        data_dir: Optional[Path] = None,
    ) -> None:
        self.config = config or GeneratorConfig()
        effective_seed = seed if seed is not None else self.config.seed
        self._rng = random.Random(effective_seed)
        self._data_dir = data_dir or Path(self.config.data_dir)
        self._symptom_profiles = _SYMPTOM_PROFILES

    def generate(
        self,
        difficulty: DifficultyLevel = DifficultyLevel.MEDIUM,
        esi_target: Optional[TriageLevel] = None,
    ) -> PatientCase:
        if esi_target is None:
            esi_target = self._sample_esi_level()
        demographics = self._sample_demographics(esi_target)
        presentation_key = esi_target.value
        category, profile = self._sample_presentation_profile(presentation_key)
        vitals = self._sample_vitals(esi_target, demographics)
        chief_complaint = self._sample_chief_complaint(profile, esi_target)
        comorbidities = self._sample_comorbidities(demographics.age)
        medications = self._sample_medications(comorbidities)
        allergies = self._sample_allergies()
        available_diagnostics = self._build_available_diagnostics(profile)
        ground_truth = GroundTruth(
            esi_level=esi_target,
            primary_dx_icd10=profile["icd10"],
            primary_dx_name=profile["dx_name"],
            differential_diagnoses=list(profile.get("differentials", [])),
            required_diagnostics=list(profile.get("required_diagnostics", [])),
            critical_interventions=list(profile.get("critical_interventions", [])),
            correct_disposition=profile["disposition"],
            hidden_findings=[],
            time_to_critical_intervention_minutes=self._time_to_critical(esi_target),
        )
        case = PatientCase(
            case_id=str(uuid.uuid4()),
            demographics=demographics, vitals=vitals,
            chief_complaint=chief_complaint, comorbidities=comorbidities,
            current_medications=medications, allergies=allergies,
            available_diagnostics=available_diagnostics,
            ground_truth=ground_truth, difficulty=difficulty,
            presentation_category=PresentationCategory(category),
        )
        case = self._inject_hidden_findings(case, difficulty)
        if self.config.inject_drug_interactions and difficulty in (
            DifficultyLevel.MEDIUM, DifficultyLevel.HARD
        ):
            case = self._inject_drug_interaction_opportunity(case, difficulty)
        return case

    def generate_batch(self, n: int, difficulty: DifficultyLevel = DifficultyLevel.MEDIUM) -> List[PatientCase]:
        return [self.generate(difficulty=difficulty) for _ in range(n)]

    def reseed(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def _sample_esi_level(self) -> TriageLevel:
        dist = self.config.esi_distribution
        keys = list(dist.keys())
        weights = [dist[k] for k in keys]
        chosen = self._rng.choices(keys, weights=weights, k=1)[0]
        return TriageLevel(chosen)

    def _sample_demographics(self, esi: TriageLevel) -> Demographics:
        paediatric = self._rng.random() < 0.15
        if paediatric:
            age = self._rng.randint(1, 17)
        else:
            esi_age_bias = {
                TriageLevel.ESI_1: (65,15), TriageLevel.ESI_2: (58,18),
                TriageLevel.ESI_3: (48,20), TriageLevel.ESI_4: (38,18),
                TriageLevel.ESI_5: (32,15),
            }
            mean_age, sd_age = esi_age_bias.get(esi, (45,20))
            age = max(18, min(99, int(self._rng_gauss(mean_age, sd_age))))
        sex = self._rng.choice([SexEnum.M, SexEnum.F, SexEnum.F])
        pregnant: Optional[bool] = None
        if sex == SexEnum.F and 15 <= age <= 50:
            pregnant = self._rng.random() < 0.08
        if sex == SexEnum.M:
            weight = max(40., min(180., self._rng_gauss(82., 15.)))
            height = max(150., min(210., self._rng_gauss(176., 8.)))
        else:
            weight = max(35., min(150., self._rng_gauss(68., 13.)))
            height = max(140., min(195., self._rng_gauss(163., 7.)))
        if paediatric:
            weight = max(5., float(age * 3 + self._rng_gauss(0, 3)))
            height = max(50., float(age * 6 + 50 + self._rng_gauss(0, 5)))
        return Demographics(age=age, sex=sex, weight_kg=round(weight,1), height_cm=round(height,1), pregnant=pregnant)

    def _sample_vitals(self, esi: TriageLevel, demographics: Demographics) -> VitalSigns:
        params = _VITAL_PARAMS[esi.value]
        for _ in range(5):
            hr = self._sample_from_params(params["heart_rate"])
            sbp = self._sample_from_params(params["bp_systolic"])
            dbp = self._sample_from_params(params["bp_diastolic"])
            rr = self._sample_from_params(params["respiratory_rate"])
            spo2 = self._sample_from_params(params["spo2"])
            temp = self._sample_from_params(params["temperature_celsius"])
            gcs_raw = self._sample_from_params(params["gcs"])
            pain = self._sample_from_params(params["pain_score"])
            if demographics.is_paediatric():
                hr = min(220, hr + int(30 - demographics.age))
            vitals = VitalSigns(
                heart_rate=int(round(hr)), bp_systolic=int(round(sbp)),
                bp_diastolic=int(round(dbp)), respiratory_rate=int(round(rr)),
                spo2=round(min(1., max(0., spo2)), 3),
                temperature_celsius=round(temp, 1),
                gcs=int(round(max(3, min(15, gcs_raw)))),
                pain_score=int(round(max(0, min(10, pain)))),
            )
            if self._vitals_are_consistent(vitals, esi):
                return vitals
        return vitals

    def _vitals_are_consistent(self, v: VitalSigns, esi: TriageLevel) -> bool:
        if v.bp_diastolic >= v.bp_systolic:
            return False
        pp = v.bp_systolic - v.bp_diastolic
        if not (10 <= pp <= 100):
            return False
        if v.gcs == 3 and esi not in (TriageLevel.ESI_1,):
            return False
        if esi == TriageLevel.ESI_1 and v.spo2 > 0.95:
            return False
        return True

    def _sample_presentation_profile(self, esi_key: str) -> Tuple[str, Dict[str, Any]]:
        esi_profiles = self._symptom_profiles.get(esi_key, {})
        categories = list(esi_profiles.keys())
        chosen_category = self._rng.choice(categories)
        return chosen_category, esi_profiles[chosen_category]

    def _sample_chief_complaint(self, profile: Dict[str, Any], esi: TriageLevel) -> ChiefComplaint:
        complaint_text = self._rng.choice(profile["complaints"])
        onset_mean = {
            TriageLevel.ESI_1: 0.5, TriageLevel.ESI_2: 2., TriageLevel.ESI_3: 12.,
            TriageLevel.ESI_4: 48., TriageLevel.ESI_5: 120.,
        }.get(esi, 24.)
        onset_hours = max(0.1, self._rng_gauss(onset_mean, onset_mean * 0.5))
        severity = int(round(self._sample_from_params(_VITAL_PARAMS[esi.value]["pain_score"])))
        associated = self._rng.sample(
            ["nausea","vomiting","diaphoresis","dyspnoea","dizziness","palpitations",
             "fever","chills","fatigue","anorexia","syncope","headache","back pain"],
            k=self._rng.randint(0, 4),
        )
        return ChiefComplaint(
            text=complaint_text, onset_hours=round(onset_hours, 2),
            severity_0_10=max(0, min(10, severity)), associated_symptoms=associated,
        )

    def _sample_comorbidities(self, age: int) -> List[Comorbidity]:
        lam = min(age / 25., 4.)
        n = min(self.config.max_comorbidities, self._rng_poisson(lam))
        if n == 0:
            return []
        pool = list(_COMORBIDITIES_POOL)
        self._rng.shuffle(pool)
        selected = pool[:n]
        return [Comorbidity(
            icd10_code=c["icd10_code"], name=c["name"],
            active=self._rng.random() > 0.1, on_treatment=self._rng.random() > 0.3,
        ) for c in selected]

    def _sample_medications(self, comorbidities: List[Comorbidity]) -> List[Medication]:
        n_meds = min(self.config.max_medications, len(comorbidities) + self._rng.randint(0, 2))
        if n_meds == 0:
            return []
        pool = list(_MEDICATIONS_POOL)
        self._rng.shuffle(pool)
        selected = pool[:n_meds]
        return [Medication(name=m["name"], dose=m["dose"], route=m["route"], frequency=m["frequency"]) for m in selected]

    def _sample_allergies(self) -> List[str]:
        n = self._rng.choices([0,1,2,3], weights=[0.60,0.25,0.10,0.05])[0]
        non_nkda = [a for a in _ALLERGIES_POOL if a != "NKDA"]
        if n == 0:
            return ["NKDA"]
        return self._rng.sample(non_nkda, k=min(n, len(non_nkda)))

    def _build_available_diagnostics(self, profile: Dict[str, Any]) -> List[str]:
        required = set(profile.get("required_diagnostics", []))
        extra_n = min(self._rng.randint(3, 8), len(_AVAILABLE_DIAGNOSTICS_POOL))
        extra = set(self._rng.sample(_AVAILABLE_DIAGNOSTICS_POOL, k=extra_n))
        combined = list(required | extra)
        self._rng.shuffle(combined)
        return combined

    def _inject_hidden_findings(self, case: PatientCase, difficulty: DifficultyLevel) -> PatientCase:
        n_findings = {
            DifficultyLevel.EASY: 0,
            DifficultyLevel.MEDIUM: 1,
            DifficultyLevel.HARD: self._rng.randint(2, 3),
        }[difficulty]
        if n_findings == 0:
            return case
        pool = list(_HIDDEN_FINDINGS_POOL)
        self._rng.shuffle(pool)
        findings = pool[:n_findings]
        updated_gt = case.ground_truth.model_copy(update={"hidden_findings": findings})
        return case.model_copy(update={"ground_truth": updated_gt})

    def _inject_drug_interaction_opportunity(self, case: PatientCase, difficulty: DifficultyLevel) -> PatientCase:
        if len(case.current_medications) < 2:
            return case
        severity_label = "severe" if difficulty == DifficultyLevel.HARD else "mild"
        interaction_note = (
            f"[{severity_label.upper()} INTERACTION] "
            f"{case.current_medications[0].name} + {case.current_medications[1].name}"
        )
        existing = list(case.ground_truth.hidden_findings)
        existing.append(interaction_note)
        updated_gt = case.ground_truth.model_copy(update={"hidden_findings": existing})
        return case.model_copy(update={"ground_truth": updated_gt})

    @staticmethod
    def _time_to_critical(esi: TriageLevel) -> Optional[int]:
        return {TriageLevel.ESI_1: 0, TriageLevel.ESI_2: 10, TriageLevel.ESI_3: 60,
                TriageLevel.ESI_4: None, TriageLevel.ESI_5: None}.get(esi)

    def _rng_gauss(self, mu: float, sigma: float) -> float:
        return self._rng.gauss(mu, sigma)

    def _rng_poisson(self, lam: float) -> int:
        if lam <= 0:
            return 0
        l_val = math.exp(-lam)
        k, p = 0, 1.0
        while p > l_val:
            k += 1
            p *= self._rng.random()
        return k - 1

    def _sample_from_params(self, params: Tuple[float,float,float,float]) -> float:
        mean, std, lo, hi = params
        return max(lo, min(hi, self._rng_gauss(mean, std)))