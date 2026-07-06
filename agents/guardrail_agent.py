"""
agents/guardrail_agent.py — Deterministic guardrail check inside the LoopAgent.

Reads:   session.state["narrative"]  + session.state["analysis_result"]
Writes:  session.state["guardrail_passed"]    bool
         session.state["violations"]          list[dict]
         session.state["guardrail_exit"]      bool  ← informational only, see below

IMPORTANT — how the loop actually exits early:
    ADK's LoopAgent (this version) only breaks out of its loop when a
    sub-agent yields an Event with actions.escalate=True, or when
    max_iterations is reached. It does NOT read arbitrary session.state
    keys. An earlier version of this file only set
    session.state["guardrail_exit"] and relied on a comment saying
    "sub-agents check this flag" — nothing ever did, so the loop always
    ran both iterations regardless of whether the first pass passed,
    and a good first draft could get silently overwritten by a worse
    second draft. We checked SequentialAgent's source directly: it does
    NOT inspect event.actions.escalate at all, only LoopAgent does — so
    escalating here is safe and does not affect output_writer running
    afterward.

    We now set actions.escalate=True on every genuine exit condition
    (pass, or unrecoverable failure) and leave it False only when we
    want the Storyteller to get a real retry attempt.

On first-pass failure (recoverable): violations stay in state, escalate
stays False, so Storyteller sees them on retry.
On second-pass failure: max_iterations stops the loop → OutputWriter
writes a refusal.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import AsyncGenerator

from google.adk.agents import BaseAgent, InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types

from agents.schema import AnalysisResult
from guardrails import output_checker


class GuardrailAgent(BaseAgent):
    """Agent that deterministically checks storyteller narrative against schema & logic constraints.

    This agent reads storyteller narrative and raw data science findings from
    session state, executes checks, logs violations, and flags loop exits.
    """

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """Asynchronously runs the guardrail agent checks on session narrative.

        Args:
            ctx: The execution context containing session state and event history.

        Yields:
            Event containing verification status and violation logs.
        """

        # ── Read inputs from session state ─────────────────────────────────
        narrative = ctx.session.state.get("narrative", "")
        raw_result = ctx.session.state.get("analysis_result")

        if not narrative:
            ctx.session.state["guardrail_passed"] = False
            ctx.session.state["violations"] = [
                {"check": "pipeline", "detail": "No narrative found in session state.", "severity": "error"}
            ]
            ctx.session.state["guardrail_exit"] = True  # unrecoverable
            yield self._event(
                "Guardrail: no narrative to check — aborting loop.", escalate=True
            )
            return

        if not raw_result:
            ctx.session.state["guardrail_passed"] = False
            ctx.session.state["violations"] = [
                {"check": "pipeline", "detail": "No analysis_result found in session state.", "severity": "error"}
            ]
            ctx.session.state["guardrail_exit"] = True  # unrecoverable
            yield self._event(
                "Guardrail: no analysis_result to check against — aborting loop.",
                escalate=True,
            )
            return

        # ── Run checks ─────────────────────────────────────────────────────
        try:
            result = AnalysisResult.model_validate(raw_result)
        except Exception as e:
            ctx.session.state["guardrail_passed"] = False
            ctx.session.state["violations"] = [
                {"check": "schema", "detail": f"Could not parse AnalysisResult: {e}", "severity": "error"}
            ]
            ctx.session.state["guardrail_exit"] = True
            yield self._event(
                f"Guardrail: schema validation failed — {e}", escalate=True
            )
            return

        violations = output_checker.run_all(narrative, result)
        errors = [v for v in violations if v.severity == "error"]
        passed = len(errors) == 0

        # Store violation dicts for Storyteller's retry context
        ctx.session.state["violations"] = [asdict(v) for v in violations]
        ctx.session.state["guardrail_passed"] = passed

        # Signal loop exit only when PASSED — on failure, loop continues
        # (up to max_iterations) so Storyteller gets a real retry attempt.
        ctx.session.state["guardrail_exit"] = passed

        summary = output_checker.format_violations(violations)
        yield self._event(
            f"Guardrail: {'PASSED' if passed else 'FAILED'}\n{summary}",
            escalate=passed,
        )

    def _event(self, text: str, escalate: bool = False) -> Event:
        """Helper method to construct a standard ADK Event with actions.

        Args:
            text: Content message for the event.
            escalate: Boolean flag indicating if pipeline should escalate/exit.

        Returns:
            The constructed Event object.
        """
        return Event(
            author=self.name,
            content=types.Content(
                parts=[types.Part(text=text)],
                role="model",
                ),
            actions=EventActions(escalate=escalate),
        )


guardrail_agent = GuardrailAgent(name="guardrail")
