````markdown
---
title: AI Medical Triage & Clinical Documentation Environment
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: apache-2.0
---

# AI Medical Triage & Clinical Documentation Environment

A production-grade [OpenEnv](https://openenv.ai)-compatible environment that
challenges AI agents to perform emergency department triage and clinical
documentation. Agents receive partial patient observations and must produce
structured triage decisions evaluated by a three-tier grading system.

---

## Quick Start

### Docker (Hugging Face Spaces / production)

```bash
docker build -t medical-triage-openenv .
docker run -p 7860:7860 -e ANTHROPIC_API_KEY=sk-ant-... medical-triage-openenv
```

### Local development

```bash
pip install -r requirements.txt

# Run evaluation (all difficulties, 10 episodes each)
python inference.py

# With options
python inference.py --episodes 5 --difficulty easy --seed 42 --output results.json

# Start API server
python -m server.api_server
```

### Run tests

```bash
pip install -r requirements.txt pytest
pytest tests/ -v --tb=short
```

---

## Repository Structure

```
medical-triage-openenv/
├── env/
│   ├── models.py               # All Pydantic data models
│   ├── patient_generator.py    # Synthetic patient case generator
│   ├── reward.py               # Stateless reward computation engine
│   └── triage_env.py           # OpenEnv-compliant TriageEnv class
├── graders/
│   ├── base_grader.py          # Abstract base + shared scoring utilities
│   ├── easy_grader.py          # Binary pass/fail grader
│   ├── medium_grader.py        # Weighted rubric grader
│   └── hard_grader.py          # LLM-assisted two-stage grader
├── server/
│   ├── api_server.py           # FastAPI application
│   ├── middleware.py           # Logging, request ID, error normalisation
│   ├── rate_limiter.py         # Token-bucket rate limiting
│   └── session_store.py        # In-memory session management
├── agents/
│   └── baseline_agent.py       # Deterministic ESI heuristic baseline
├── tests/
│   └── test_env.py             # 86 integration and unit tests
├── inference.py                # Standalone evaluation script
├── openenv.yaml                # OpenEnv manifest
├── Dockerfile                  # Multi-stage container build
├── requirements.txt            # Production dependencies
└── README.md
```

---

## API Reference

All endpoints are documented at `/docs` (Swagger UI) when the server is running.

### POST `/env/reset`

Start a new episode.

```bash
curl -X POST http://localhost:7860/env/reset \
  -H "Content-Type: application/json" \
  -d '{"difficulty": "medium", "seed": 42}'
```

Response includes `session_id` and the first `PatientObservation`.

### POST `/env/step`

Submit a `TriageAction`. Requires `X-Session-ID` header.

```bash
curl -X POST http://localhost:7860/env/step \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: <session_id>" \
  -d '{
    "action": {
      "assigned_triage_level": "ESI_2",
      "diagnostic_orders": [{"test_name": "ECG", "rationale": "Rule out STEMI", "expected_turnaround_minutes": 5}],
      "treatments": [{"intervention": "IV Access", "rationale": "Vascular access", "priority": "IMMEDIATE"}],
      "clinical_note": {
        "chief_complaint_summary": "58-year-old male with severe crushing chest pain.",
        "history_of_present_illness": "58-year-old male presenting with sudden onset crushing chest pain radiating to the left arm, onset 2 hours ago, associated with diaphoresis and nausea. Severity 9/10. No prior cardiac history. On aspirin and atorvastatin. No known drug allergies.",
        "physical_exam_findings": "Alert and oriented. Appears diaphoretic and pale. Heart sounds regular. Breath sounds clear bilaterally. Abdomen soft. No peripheral oedema. GCS 15.",
        "assessment": "Presentation consistent with acute coronary syndrome. Differential includes aortic dissection, pulmonary embolism, and unstable angina.",
        "plan": "ECG stat. Aspirin 300mg PO. IV access. Continuous cardiac monitoring. Cardiology consult activated. Admit to ICU.",
        "differential_diagnoses": ["Acute myocardial infarction", "Unstable angina", "Aortic dissection"]
      },
      "disposition": "ADMIT_ICU",
      "confidence_score": 0.85,
      "reasoning_chain": "VITAL SIGN ANALYSIS: HR 115 tachycardic, BP 160/95 hypertensive, SpO2 93% borderline, GCS 15, pain 9/10 severe. ESI CLASSIFICATION: ESI 2 — high risk cardiac. DIAGNOSTIC REASONING: ECG to detect STEMI. TREATMENT REASONING: IV access, aspirin per ACS protocol. ALLERGY REVIEW: No known drug allergies. DISPOSITION: ICU admission.",
      "is_final": true
    }
  }'
```

### GET `/env/state`

Return episode metadata without advancing.

```bash
curl http://localhost:7860/env/state -H "X-Session-ID: <session_id>"
```

### POST `/grade/trajectory`

Grade a completed episode.

```bash
curl -X POST http://localhost:7860/grade/trajectory \
  -H "Content-Type: application/json" \
  -d '{"session_id": "<session_id>", "grader_tier": "medium"}'
```

### GET `/health`

```bash
curl http://localhost:7860/health
```

---

## Grading System

### Easy Grader

Binary pass/fail. **All six conditions must be met:**

1. ESI level exact match with ground truth
2. Clinical note meets minimum field lengths
3. At least one required diagnostic ordered
4. No allergy violations
5. No unsafe discharge (ESI 1-2 patient sent home)
6. Committed disposition (not `PENDING`) on final action

**Pass threshold:** 1.0

### Medium Grader

Weighted rubric with partial credit.

| Dimension | Weight |
|---|---|
| Triage accuracy (ESI + disposition) | 25% |
| Documentation quality (note + reasoning) | 25% |
| Diagnostic workup coverage | 20% |
| Treatment safety | 20% |
| Time to disposition | 10% |

**Pass threshold:** 70/100. Safety violations hard-cap at 60/100.

### Hard Grader

Two-stage evaluation for complex cases.

| Stage | Weight |
|---|---|
| Automated rubric (6 dimensions + hidden finding + drug interaction) | 60% |
| LLM judge (clinical reasoning, differential quality, hidden finding detection) | 40% |

**Pass threshold:** 80/100.

- Hidden finding detected: +10 bonus
- LLM safety concerns: cap at 60
- Critical miss (ESI 1-2 → ESI 4-5): cap at 40

*Falls back to automated-only scoring without `ANTHROPIC_API_KEY`.*

---

## Patient Generator

| Parameter | Easy | Medium | Hard |
|---|---|---|---|
| Hidden findings | 0 | 1 | 2-3 |
| Comorbidities | 0-1 | 1-3 | 3-5 |
| Drug interactions | None | Mild | Severe |
| Vital ambiguity | Clear | Borderline | Near-normal masking severity |

---

## Action Schema

```json
{
  "assigned_triage_level": "ESI_1 | ESI_2 | ESI_3 | ESI_4 | ESI_5",
  "diagnostic_orders": [{"test_name": "ECG", "rationale": "...", "expected_turnaround_minutes": 5}],
  "treatments": [{"intervention": "IV Access", "rationale": "...", "priority": "IMMEDIATE"}],
  "clinical_note": {
    "chief_complaint_summary": "string (min 20 chars)",
    "history_of_present_illness": "string (min 100 chars)",
    "physical_exam_findings": "string (min 50 chars)",
    "assessment": "string (min 50 chars)",
    "plan": "string (min 50 chars)",
    "differential_diagnoses": ["1-5 entries"]
  },
  "disposition": "ADMIT_ICU | ADMIT_FLOOR | OBSERVATION | DISCHARGE_HOME | DISCHARGE_PCP_FOLLOWUP | TRANSFER | PENDING",
  "confidence_score": 0.0,
  "reasoning_chain": "string (min 50 chars)",
  "is_final": false
}
```

---

## Python API

```python
from env import TriageEnv, DifficultyLevel
from agents.baseline_agent import BaselineAgent
from graders import GraderRegistry

env = TriageEnv(seed=42)
agent = BaselineAgent()

obs = env.reset(difficulty=DifficultyLevel.MEDIUM)
agent.reset()

done = False
while not done:
    action = agent.act(obs)
    result = env.step(action)
    done = result.done
    if not done:
        obs = result.observation

trajectory = env.get_trajectory()
grader = GraderRegistry.from_string("medium")
grade = grader.grade(trajectory)

print(f"Score: {grade.total_score * 100:.1f}/100 | Pass: {grade.pass_fail}")
print(grade.feedback_text)
```

---

## Deployment

### Hugging Face Spaces

1. Fork this repository
2. Set `ANTHROPIC_API_KEY` secret in Space settings (optional)
3. Space builds and deploys automatically using the `Dockerfile`
4. API available at `https://<username>-medical-triage-openenv.hf.space`

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | `""` | For HardGrader LLM judge |
| `PORT` | `7860` | Must be 7860 for Spaces |
| `WORKERS` | `1` | Keep at 1 for in-memory session store |
| `SESSION_TTL_SECONDS` | `1800` | Session expiry |
| `MAX_SESSIONS` | `100` | Max concurrent sessions |
| `RATE_LIMIT_RPM` | `60` | Requests/minute per client |
| `LOG_LEVEL` | `info` | Logging verbosity |

---

## License

Apache 2.0
````