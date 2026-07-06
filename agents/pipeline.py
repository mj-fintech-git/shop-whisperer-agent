"""
agents/pipeline.py — Assembles the full SequentialAgent + LoopAgent pipeline.

Execution order:
    1. ds_agent         — analysis via MCP tools → JSON output
    2. state_bridge     — parses JSON → session.state["analysis_result"]
    3. LoopAgent (max 2 iterations):
        a. storyteller  — narrative → session.state["narrative"]
        b. guardrail    — deterministic checks → session.state["guardrail_passed"]
                          yields an Event with actions.escalate=True on pass
                          (or on unrecoverable failure) to exit the loop early

Note on LoopAgent exit:
    guardrail_agent yields actions.escalate=True whenever the checks pass,
    or whenever the failure is unrecoverable (e.g. no narrative/analysis
    found at all). LoopAgent stops looping as soon as it sees escalate=True
    on any sub-agent event — this is the ADK mechanism for early exit, not
    a session-state flag. We verified SequentialAgent's source directly:
    it does not inspect event.actions.escalate at all, so escalating out
    of the inner LoopAgent does not prevent output_writer (the next
    SequentialAgent step) from running afterward.

    The LoopAgent's max_iterations=2 remains a hard stop as a fallback:
    on a genuine guardrail failure (recoverable), escalate stays False so
    Storyteller gets one retry with the specific violations in state.
"""

from __future__ import annotations

from google.adk.agents import LoopAgent, SequentialAgent

from agents.ds_agent import ds_agent
from agents.state_bridge import state_bridge
from agents.storyteller import storyteller_agent
from agents.guardrail_agent import guardrail_agent
from agents.output_writer import output_writer

# ── Storyteller + Guardrail loop ──────────────────────────────────────────────
# max_iterations=2: one shot + one retry. A clean first pass exits immediately
# because guardrail_agent yields escalate=True when checks pass.

storyteller_loop = LoopAgent(
    name="storyteller_loop",
    sub_agents=[storyteller_agent, guardrail_agent],
    max_iterations=2,
)

# ── Full pipeline ─────────────────────────────────────────────────────────────

pipeline = SequentialAgent(
    name="shop_whisperer_pipeline",
    sub_agents=[
        ds_agent,
        state_bridge,
        storyteller_loop,
        output_writer,
    ],
)
