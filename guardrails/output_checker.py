"""
guardrails/output_checker.py — Deterministic post-generation checks on Storyteller output.

Three checks run before final_presentation.md is written:
    1. claim_grounding   — every numeric claim traces to a Finding
    2. confidence_calibration — language matches certainty × sample size
    3. narrative_overreach   — no causal/motive claims without supporting data

All checks are regex-based and deterministic — no LLM involved.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from agents.schema import AnalysisResult, Finding, SMALL_SAMPLE_THRESHOLD


@dataclass
class GuardrailViolation:
    check: str
    detail: str
    severity: Literal["error", "warning"]

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.check}: {self.detail}"


# ── Banned phrase lists ────────────────────────────────────────────────────────

_CAUSAL_PHRASES = [
    r"the (?:core |main |primary |real )?reason\b",
    r"this (?:is why|caused|explains why)",
    r"that(?:'s| is) why",
    r"\bcaused by\b",
    r"\bdirectly caused\b",
    r"\bthe reason (?:profits|revenue|sales)\b",
]

_MOTIVE_PHRASES = [
    r"\bhe (?:wanted|was trying|never meant|planned|intended)\b",
    r"\bshe (?:wanted|was trying|never meant|planned|intended)\b",
    r"\bthey (?:were targeting|planned|intended|wanted to)\b",
    r"\bthe customer (?:wanted|intended|decided|chose to)\b",
    r"\b(?:never going to|was always going to)\b",
]

_CERTAINTY_IMPLYING = [
    r"\bwe (?:know|confirmed|proved|verified|are certain)\b",
    r"\bclearly\b",
    r"\bdefinitely\b",
    r"\bobviously\b",
    r"\bwe(?:'re| are) (?:sure|certain|confident)\b",
    r"\bproven\b",
    r"\bconfirmed\b",
]

# Variables whose absence blocks profit-causal claims
_PROFIT_CAUSAL_VARIABLES = {
    "ad_spend", "cogs", "cost_of_goods", "margin", "advertising_cost",
    "marketing_spend", "cost_per_acquisition",
}

# Regex for numeric claims in prose
_DOLLAR_RE   = re.compile(r"\$[\d,]+(?:\.\d+)?")
_FRACTION_RE = re.compile(r"\b(\d+)\s+of\s+(?:the\s+)?(\d+)\b")
_PCT_RE      = re.compile(r"\b(\d+(?:\.\d+)?)\s*%")


# ── Helper: extract all numbers mentioned in the narrative ─────────────────────

def _extract_numeric_claims(narrative: str) -> list[str]:
    claims = []
    claims += _DOLLAR_RE.findall(narrative)
    claims += [f"{a}/{b}" for a, b in _FRACTION_RE.findall(narrative)]
    claims += [f"{p}%" for p in _PCT_RE.findall(narrative)]
    return claims


def _finding_numeric_surface(finding: Finding) -> set[str]:
    """Return string representations of all numbers in a Finding."""
    import json
    raw = json.dumps(finding.model_dump(), default=str)
    dollars  = set(_DOLLAR_RE.findall(raw))
    pcts     = {f"{p}%" for p in _PCT_RE.findall(raw)}
    # Also catch plain floats that might appear as dollar or percentage in narrative
    numbers  = set(re.findall(r"\b\d+(?:\.\d+)?\b", raw))
    return dollars | pcts | numbers


# ── Check 1: Claim Grounding ───────────────────────────────────────────────────

def check_claim_grounding(
    narrative: str, result: AnalysisResult
) -> list[GuardrailViolation]:
    """
    Every specific numeric claim in the narrative must be traceable to at least
    one Finding's measured_value or supporting_data.
    Orphan claims are warnings (not errors) — formatting differences cause false positives.
    """
    violations: list[GuardrailViolation] = []
    all_finding_numbers: set[str] = set()
    for f in result.findings:
        all_finding_numbers |= _finding_numeric_surface(f)

    for claim in _extract_numeric_claims(narrative):
        # Strip formatting for comparison
        bare = re.sub(r"[$,%,]", "", claim)
        if not any(bare in num for num in all_finding_numbers):
            violations.append(GuardrailViolation(
                check="claim_grounding",
                detail=f"Numeric claim '{claim}' does not trace to any Finding.",
                severity="warning",
            ))
    return violations


# ── Check 2: Confidence Calibration ───────────────────────────────────────────

def check_confidence_calibration(
    narrative: str, result: AnalysisResult
) -> list[GuardrailViolation]:
    """
    For any Finding with certainty != 'Factual' and sample_size < threshold,
    the narrative must not contain certainty-implying language.
    """
    violations: list[GuardrailViolation] = []
    small_sample_findings = [
        f for f in result.findings
        if f.certainty != "Factual" and f.sample_size < SMALL_SAMPLE_THRESHOLD
    ]

    if not small_sample_findings:
        return violations

    for pattern in _CERTAINTY_IMPLYING:
        matches = re.findall(pattern, narrative, re.IGNORECASE)
        if matches:
            violations.append(GuardrailViolation(
                check="confidence_calibration",
                detail=(
                    f"Certainty-implying phrase found ('{matches[0]}') "
                    f"but {len(small_sample_findings)} finding(s) have n < {SMALL_SAMPLE_THRESHOLD}. "
                    "Use 'looks like / seems like / probably' instead."
                ),
                severity="error",
            ))
    return violations


# ── Check 3: Narrative Overreach ──────────────────────────────────────────────

def check_narrative_overreach(
    narrative: str, result: AnalysisResult
) -> list[GuardrailViolation]:
    """
    Flag causal profit claims when ad_spend/COGS are absent from findings.
    Flag intent/motive language regardless of data.
    """
    violations: list[GuardrailViolation] = []

    # Collect all metric names and supporting_data keys from findings
    available_variables: set[str] = set()
    for f in result.findings:
        available_variables.add(f.metric_name.lower())
        available_variables |= {k.lower() for k in f.supporting_data}

    # Causal profit claims — only blocked when required variables are absent
    if not (_PROFIT_CAUSAL_VARIABLES & available_variables):
        for pattern in _CAUSAL_PHRASES:
            matches = re.findall(pattern, narrative, re.IGNORECASE)
            if matches:
                violations.append(GuardrailViolation(
                    check="narrative_overreach",
                    detail=(
                        f"Causal phrase '{matches[0]}' found, but ad_spend/COGS are absent "
                        "from the data. Use 'probably a piece of it' instead."
                    ),
                    severity="error",
                ))

    # Motive/intent language — always flagged
    for pattern in _MOTIVE_PHRASES:
        matches = re.findall(pattern, narrative, re.IGNORECASE)
        if matches:
            violations.append(GuardrailViolation(
                check="narrative_overreach",
                detail=(
                    f"Intent/motive phrase found: '{matches[0]}'. "
                    "Describe observable behaviour only — do not imply motive."
                ),
                severity="error",
            ))
    return violations


# ── Main entry point ──────────────────────────────────────────────────────────

def run_all(
    narrative: str, result: AnalysisResult
) -> list[GuardrailViolation]:
    """Run all three checks. Returns combined list of violations."""
    return (
        check_claim_grounding(narrative, result)
        + check_confidence_calibration(narrative, result)
        + check_narrative_overreach(narrative, result)
    )


def format_violations(violations: list[GuardrailViolation]) -> str:
    if not violations:
        return "All guardrail checks passed."
    lines = [str(v) for v in violations]
    return "\n".join(lines)
