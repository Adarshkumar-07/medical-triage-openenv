"""
app.py

Gradio web interface for judges to evaluate AI medical triage agents.

Usage:
    python app.py

Environment variables (optional — only needed for LLM agent):
    OPENAI_API_KEY
    ANTHROPIC_API_KEY
    GEMINI_API_KEY

The FastAPI backend must be running at SERVER_URL (default: http://localhost:8000).
Set SERVER_URL env var to override.
"""
from __future__ import annotations

import json
import os
from typing import Optional, Tuple

import requests
import gradio as gr

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SERVER_URL: str = os.environ.get("SERVER_URL", "http://localhost:8000")
_TIMEOUT: int = 30


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _post(endpoint: str, payload: dict) -> dict:
    url = f"{SERVER_URL}{endpoint}"
    try:
        r = requests.post(url, json=payload, timeout=_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return {"error": f"Cannot connect to backend at {SERVER_URL}. Is the server running?"}
    except requests.exceptions.HTTPError as exc:
        return {"error": f"HTTP {exc.response.status_code}: {exc.response.text[:300]}"}
    except Exception as exc:
        return {"error": str(exc)}


def _get(endpoint: str) -> dict:
    url = f"{SERVER_URL}{endpoint}"
    try:
        r = requests.get(url, timeout=_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return {"error": f"Cannot connect to backend at {SERVER_URL}."}
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Agent helpers
# ---------------------------------------------------------------------------

def _get_llm_action(observation: dict, provider: str, model: Optional[str]) -> dict:
    """Call LLMAgent locally to generate an action dict."""
    try:
        from agents.llm_agent import LLMAgent
        agent = LLMAgent(provider=provider, model=model or None, verbose=False)
        return agent.act(observation)
    except Exception as exc:
        return {"error": str(exc)}


def _get_baseline_action(observation: dict) -> dict:
    """Call BaselineAgent locally to generate an action dict."""
    try:
        from agents.baseline_agent import BaselineAgent
        agent = BaselineAgent(single_step=True, verbose=False)
        agent.reset()
        return agent.act(observation)
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# UI logic
# ---------------------------------------------------------------------------

def _fmt(obj) -> str:
    if isinstance(obj, dict):
        return json.dumps(obj, indent=2, ensure_ascii=False)
    return str(obj)


def generate_patient(difficulty: str) -> Tuple[str, str, str, str, str, str]:
    """Reset the environment and return patient observation details."""
    resp = _post("/env/reset", {"difficulty": difficulty})
    if "error" in resp:
        return (resp["error"], "", "", "", "", "")

    observation = resp.get("observation", resp)

    vitals_keys = ["heart_rate", "blood_pressure", "respiratory_rate",
                   "temperature", "spo2", "pain_score"]
    vitals = {k: observation.get(k) for k in vitals_keys if k in observation}

    demographics = {
        k: observation.get(k)
        for k in ["age", "sex", "weight_kg", "chief_complaint", "allergies",
                  "medical_history", "medications"]
        if k in observation
    }

    return (
        observation.get("chief_complaint", "—"),
        _fmt(demographics),
        _fmt(vitals),
        _fmt(observation.get("additional_info", {})),
        _fmt(observation),   # raw full observation stored in state
        "",                  # clear previous action panel
    )


def run_triage(
    agent_type: str,
    llm_provider: str,
    llm_model: str,
    raw_observation_json: str,
) -> Tuple[str, str, str, str]:
    """Run chosen agent on the current observation, step the env, display results."""

    if not raw_observation_json.strip():
        return ("No patient loaded. Click 'Generate Patient' first.", "", "", "")

    try:
        observation = json.loads(raw_observation_json)
    except json.JSONDecodeError:
        return ("Could not parse observation. Please generate a patient first.", "", "", "")

    # Generate action
    if agent_type == "Baseline (deterministic)":
        action = _get_baseline_action(observation)
    else:
        action = _get_llm_action(observation, llm_provider, llm_model or None)

    if "error" in action:
        return (f"Agent error: {action['error']}", "", "", "")

    # Step the environment
    step_resp = _post("/env/step", {"action": action})
    if "error" in step_resp:
        return (_fmt(action), f"Step error: {step_resp['error']}", "", "")

    reward_obj = step_resp.get("reward", {})
    penalties = step_resp.get("penalty_flags", [])
    state = _get("/env/state")

    reward_lines = []
    for k, v in reward_obj.items() if isinstance(reward_obj, dict) else []:
        reward_lines.append(f"  {k}: {v}")
    reward_str = "\n".join(reward_lines) if reward_lines else _fmt(reward_obj)

    penalty_str = (
        "\n".join(f"  ⚠  {p}" for p in penalties)
        if penalties
        else "  None"
    )

    return (
        _fmt(action),
        reward_str,
        penalty_str,
        _fmt(state),
    )


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Medical Triage Judge Panel", theme=gr.themes.Soft()) as demo:

        gr.Markdown(
            """
# AI Medical Triage — Judge Evaluation Panel
Use the controls below to generate patient cases and evaluate AI triage decisions.
"""
        )

        # ── Hidden state ──────────────────────────────────────────────────
        raw_obs_state = gr.State("")

        with gr.Row():
            # ── Left column: patient generation ───────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### 1 — Generate patient case")
                difficulty = gr.Radio(
                    choices=["easy", "medium", "hard"],
                    value="medium",
                    label="Difficulty",
                )
                gen_btn = gr.Button("Generate patient", variant="primary")

                gr.Markdown("#### Chief complaint")
                chief_complaint_box = gr.Textbox(
                    label="", interactive=False, lines=2
                )
                gr.Markdown("#### Demographics & history")
                demographics_box = gr.Code(
                    label="", language="json", interactive=False, lines=8
                )
                gr.Markdown("#### Vital signs")
                vitals_box = gr.Code(
                    label="", language="json", interactive=False, lines=6
                )
                gr.Markdown("#### Additional info")
                extra_box = gr.Code(
                    label="", language="json", interactive=False, lines=4
                )

            # ── Right column: agent & results ─────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### 2 — Configure and run agent")

                agent_type = gr.Radio(
                    choices=["Baseline (deterministic)", "LLM agent"],
                    value="Baseline (deterministic)",
                    label="Agent type",
                )
                with gr.Group(visible=False) as llm_group:
                    llm_provider = gr.Dropdown(
                        choices=["openai", "claude", "gemini"],
                        value="openai",
                        label="LLM provider",
                    )
                    llm_model = gr.Textbox(
                        label="Model override (leave blank for default)",
                        placeholder="e.g. gpt-4o / claude-opus-4-6 / gemini-1.5-pro",
                    )

                agent_type.change(
                    fn=lambda t: gr.update(visible=(t == "LLM agent")),
                    inputs=agent_type,
                    outputs=llm_group,
                )

                run_btn = gr.Button("Run AI triage", variant="primary")

                gr.Markdown("#### Triage action")
                action_box = gr.Code(
                    label="", language="json", interactive=False, lines=10
                )
                gr.Markdown("#### Reward scores")
                reward_box = gr.Textbox(
                    label="", interactive=False, lines=8
                )
                gr.Markdown("#### Penalty flags")
                penalty_box = gr.Textbox(
                    label="", interactive=False, lines=4
                )
                gr.Markdown("#### Environment state")
                state_box = gr.Code(
                    label="", language="json", interactive=False, lines=6
                )

        # ── Wire buttons ──────────────────────────────────────────────────

        gen_btn.click(
            fn=generate_patient,
            inputs=[difficulty],
            outputs=[
                chief_complaint_box,
                demographics_box,
                vitals_box,
                extra_box,
                raw_obs_state,
                action_box,
            ],
        )

        run_btn.click(
            fn=run_triage,
            inputs=[agent_type, llm_provider, llm_model, raw_obs_state],
            outputs=[action_box, reward_box, penalty_box, state_box],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("GRADIO_PORT", "7860")),
        share=False,
        show_error=True,
    )