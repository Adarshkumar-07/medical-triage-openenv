"""
agents/llm_agent.py

Multi-LLM decision agent supporting OpenAI, Anthropic Claude, and Google Gemini.
Reads API keys from environment variables:
    OPENAI_API_KEY
    ANTHROPIC_API_KEY
    GEMINI_API_KEY

Usage:
    from agents.llm_agent import LLMAgent
    agent = LLMAgent(provider="openai")   # or "claude" / "gemini"
    action = agent.act(observation)
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert emergency-department triage nurse.
Given a patient observation you must output a single JSON object — no markdown,
no explanation — that represents the triage action.

The JSON must conform exactly to this schema:
{
  "assigned_triage_level": "<ESI-1|ESI-2|ESI-3|ESI-4|ESI-5>",
  "chief_complaint": "<one-sentence summary>",
  "ordered_diagnostics": ["<test1>", "<test2>"],
  "treatments_initiated": ["<treatment1>"],
  "disposition": "<admit|discharge|observation|transfer>",
  "documentation_notes": "<clinical notes string>"
}

ESI levels:
  ESI-1 = immediate life threat
  ESI-2 = high-risk / should not wait
  ESI-3 = urgent, multiple resources needed
  ESI-4 = less urgent, one resource
  ESI-5 = non-urgent, no resources

Return ONLY the JSON object. Do not include any other text."""


def _build_user_prompt(observation: Dict[str, Any]) -> str:
    lines = ["PATIENT OBSERVATION:"]
    for key, value in observation.items():
        if value is not None and value != "" and value != []:
            lines.append(f"  {key}: {value}")
    lines.append("\nProvide your triage action JSON:")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    # Strip markdown fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Attempt to find first {...} block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"Could not parse JSON from LLM response: {text[:300]}")


# ---------------------------------------------------------------------------
# Default fallback action
# ---------------------------------------------------------------------------

def _default_action(observation: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "assigned_triage_level": "ESI-3",
        "chief_complaint": observation.get("chief_complaint", "Unknown complaint"),
        "ordered_diagnostics": ["CBC", "BMP"],
        "treatments_initiated": ["IV access"],
        "disposition": "observation",
        "documentation_notes": "Default action generated due to LLM parse failure.",
    }


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------

class _OpenAIProvider:
    def __init__(self, model: str = "gpt-4o-mini"):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Install openai: pip install openai") from exc
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY environment variable not set.")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def complete(self, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=512,
        )
        return response.choices[0].message.content or ""


class _ClaudeProvider:
    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError("Install anthropic: pip install anthropic") from exc
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY environment variable not set.")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def complete(self, user_prompt: str) -> str:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return message.content[0].text if message.content else ""


class _GeminiProvider:
    def __init__(self, model: str = "gemini-1.5-flash"):
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise ImportError(
                "Install google-generativeai: pip install google-generativeai"
            ) from exc
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY environment variable not set.")
        genai.configure(api_key=api_key)
        self.model_client = genai.GenerativeModel(
            model_name=model,
            system_instruction=SYSTEM_PROMPT,
        )

    def complete(self, user_prompt: str) -> str:
        response = self.model_client.generate_content(
            user_prompt,
            generation_config={"temperature": 0.0, "max_output_tokens": 512},
        )
        return response.text or ""


# ---------------------------------------------------------------------------
# Public agent class
# ---------------------------------------------------------------------------

_PROVIDER_MAP = {
    "openai": _OpenAIProvider,
    "claude": _ClaudeProvider,
    "gemini": _GeminiProvider,
}

_DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "claude": "claude-haiku-4-5-20251001",
    "gemini": "gemini-1.5-flash",
}


class LLMAgent:
    """
    Multi-provider LLM triage agent.

    Parameters
    ----------
    provider : str
        One of "openai", "claude", "gemini".
    model : str, optional
        Override the default model for the chosen provider.
    verbose : bool
        Log prompt / response details.
    """

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        if provider not in _PROVIDER_MAP:
            raise ValueError(
                f"Unknown provider '{provider}'. Choose from: {list(_PROVIDER_MAP)}"
            )
        self.provider_name = provider
        self.verbose = verbose
        chosen_model = model or _DEFAULT_MODELS[provider]
        self._provider = _PROVIDER_MAP[provider](model=chosen_model)
        logger.info("LLMAgent initialised: provider=%s model=%s", provider, chosen_model)

    def reset(self) -> None:
        """No internal state to reset; kept for API compatibility with BaselineAgent."""
        pass

    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a triage action for the given observation.

        Parameters
        ----------
        observation : dict
            The env observation dict returned by env.reset() or env.step().

        Returns
        -------
        dict
            Action dict ready to be passed to env.step().
        """
        user_prompt = _build_user_prompt(observation)
        if self.verbose:
            logger.debug("LLMAgent prompt:\n%s", user_prompt)

        try:
            raw = self._provider.complete(user_prompt)
            if self.verbose:
                logger.debug("LLMAgent raw response:\n%s", raw)
            action = _extract_json(raw)
        except Exception as exc:
            logger.warning(
                "LLMAgent (%s) failed to produce valid action: %s. Using fallback.",
                self.provider_name,
                exc,
            )
            action = _default_action(observation)

        return action

    def __repr__(self) -> str:
        return f"LLMAgent(provider={self.provider_name!r})"