"""
agents/schema.py — Shared data contract between DS_Agent and Storyteller_Agent.

Uses Pydantic for JSON serialisation (required for ADK session state storage
and MCP tool return values).
"""

from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, field_validator


# ── Certainty field ────────────────────────────────────────────────────────────
# Replaces the previous is_arithmetic + confidence pair.
# One field, one decision, no invalid combinations.
#
#   Factual      — arithmetic computation; state flatly regardless of n
#   High         — p < 0.05 and practically meaningful effect size
#   Medium       — directional but not statistically robust
#   Low          — weak signal, high variance, or very small n
#   Inconclusive — data cannot answer this question as posed

CertaintyLevel = Literal["Factual", "High", "Medium", "Low", "Inconclusive"]

# Storyteller language ceiling per certainty × sample size
CERTAINTY_SPOKEN = {
    "Factual":       "state flatly — no hedge required, n-ceiling does not apply",
    "High":          "state plainly; if n < 20, still cap at 'looks like'",
    "Medium":        "looks like / seems like / probably",
    "Low":           "there's a hint of / worth a look but / might be",
    "Inconclusive":  "we genuinely couldn't find an answer to that one — here's why",
}

SMALL_SAMPLE_THRESHOLD = 20   # n below this: cap at "looks like/seems like" for non-Factual


class Finding(BaseModel):
    """One measured claim from the DS_Agent, fully self-describing."""

    metric_name: str
    """Short identifier for what was measured, e.g. 'return_rate_paid_ads'."""

    measured_value: Any
    """The computed result — float, dict, list of dicts, string, etc."""

    certainty: CertaintyLevel
    """Certainty level. 'Factual' = arithmetic; others = statistical inference."""

    sample_size: int
    """Number of observations this finding is based on."""

    comparison_group: Optional[str] = None
    """What the measured value is being compared against, if applicable."""

    statistical_basis: Optional[dict] = None
    """Only populated when certainty != 'Factual'.
    Keys: test, p_value, odds_ratio, ci_95, effect_size, etc."""

    supporting_data: dict = {}
    """Raw numbers the Storyteller needs to narrate this finding.
    All values must be JSON-serialisable Python primitives."""

    @field_validator("statistical_basis", mode="before")
    @classmethod
    def require_basis_for_inferences(cls, v, info):
        certainty = info.data.get("certainty")
        if certainty and certainty != "Factual" and v is None:
            # Warn but don't error — DS_Agent may omit for descriptive findings
            pass
        return v

    @property
    def requires_hedge(self) -> bool:
        """True if Storyteller must use hedging language for this finding."""
        return self.certainty != "Factual" and self.sample_size < SMALL_SAMPLE_THRESHOLD

    @property
    def spoken_guidance(self) -> str:
        """Guidance string for Storyteller's language calibration."""
        base = CERTAINTY_SPOKEN[self.certainty]
        if self.certainty != "Factual" and self.sample_size < SMALL_SAMPLE_THRESHOLD:
            base += f" [n={self.sample_size} < {SMALL_SAMPLE_THRESHOLD}: cap at 'looks like/seems like']"
        return base


class DataGap(BaseModel):
    """A missing data source that would have enabled stronger analysis."""
    missing_data: str
    why_it_matters: str
    analysis_enabled: str


class AnalysisResult(BaseModel):
    """Complete structured output from DS_Agent — the agent-to-agent contract."""

    business_problem: str
    """Verbatim problem statement from the user input."""

    dataset_info: dict
    """Keys: filename, n_rows, n_cols, columns, analysis_timestamp."""

    domain_context: str
    """Web searches performed and key takeaways, or 'None'."""

    findings: list[Finding]
    """All findings in order of impact (highest certainty / largest effect first)."""

    data_gaps: list[DataGap]
    """What additional data would have enabled stronger conclusions."""

    next_steps: list[str]
    """Concrete, data-grounded actions. Each must trace to a specific finding."""

    def to_storyteller_prompt(self) -> str:
        """Serialise for injection into Storyteller's prompt context."""
        import json
        return json.dumps(self.model_dump(), indent=2, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "AnalysisResult":
        """Parse from DS_Agent's JSON output."""
        import json
        return cls.model_validate(json.loads(json_str))
