"""
agents/storyteller.py — Storyteller_Agent: Gemini LlmAgent producing the narrative.

Model:     Gemini 2.0 Flash (native ADK, no LiteLLM needed)
Tools:     None — Storyteller MUST NOT call ds_toolkit
Input:     session.state["analysis_result_json"]  (injected into prompt)
Output:    Narrative string → session.state["narrative"]
"""

from __future__ import annotations

from pathlib import Path

from google.adk.agents import LlmAgent
from google.genai import types


def _load_skill(skill_name: str) -> str:
    skill_path = Path(__file__).parent.parent / "skills" / skill_name / "SKILL.md"
    if not skill_path.exists():
        raise FileNotFoundError(f"Skill file not found: {skill_path}")
    text = skill_path.read_text(encoding="utf-8")
    if text.startswith("---"):
        parts = text.split("---", 2)
        return parts[2].strip() if len(parts) >= 3 else text
    return text


_BASE_INSTRUCTION = _load_skill("storyteller_translation")

# Storyteller receives the AnalysisResult JSON via a dynamic prompt prefix.
# The prefix is constructed at runtime by the pipeline (analysis.py) and
# prepended to the message content — not baked into the instruction string.
# This keeps the skill instruction clean and reusable.

storyteller_agent = LlmAgent(
    model="gemini-2.0-flash",
    name="storyteller",
    description=(
        "Translates structured AnalysisResult findings into a plain spoken-narrative "
        "for final_presentation.md. Never calls analysis tools."
    ),
    instruction=_BASE_INSTRUCTION,
    tools=[],  # Enforced: no tool access for Storyteller
    output_key="narrative",  # ADK stores the response in session.state["narrative"]
)
