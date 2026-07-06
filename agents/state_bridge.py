"""
agents/state_bridge.py — Parses DS_Agent's JSON output and writes it to session state.

DS_Agent produces a ```json``` block in its final response.
This agent extracts and validates it into an AnalysisResult, then stores it
at session.state["analysis_result"] for Storyteller to consume.

This is a thin BaseAgent — no LLM call, purely deterministic.
"""

from __future__ import annotations

import json
import re
from typing import AsyncGenerator

from google.adk.agents import BaseAgent, InvocationContext
from google.adk.events import Event
from google.genai import types

from agents.schema import AnalysisResult

_JSON_BLOCK_RE = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)


class StateBridgeAgent(BaseAgent):
    """Bridge agent that extracts raw JSON from history and writes to session state.

    This agent acts as a parser to parse the structured markdown JSON output of the
    data scientist agent and validate it against Pydantic models.
    """

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """Asynchronously parses and validates raw JSON output into AnalysisResult.

        Args:
            ctx: The execution context containing session state and events history.

        Yields:
            Event detailing whether parsing and validation succeeded.
        """
        # Find the last model message in the event history
        raw_json: str | None = None
        for event in reversed(ctx.session.events):
            if event.author != self.name and event.content:
                for part in event.content.parts or []:
                    if part.text:
                        match = _JSON_BLOCK_RE.search(part.text)
                        if match:
                            raw_json = match.group(1)
                            break
            if raw_json:
                break

        if not raw_json:
            error_msg = (
                "StateBridge: Could not find a ```json``` block in DS_Agent output. "
                "Ensure DS_Agent produces a single JSON block as its final response."
            )
            ctx.session.state["state_bridge_error"] = error_msg
            yield Event(
                author=self.name,
                content=types.Content(
                    parts=[types.Part(text=error_msg)],
                    role="model",
                ),
            )
            return

        try:
            result = AnalysisResult.model_validate(json.loads(raw_json))
            ctx.session.state["analysis_result"] = result.model_dump()
            ctx.session.state["analysis_result_json"] = result.to_storyteller_prompt()
            status = f"StateBridge: AnalysisResult stored — {len(result.findings)} findings."
        except Exception as e:
            status = f"StateBridge: Failed to parse AnalysisResult — {e}"
            ctx.session.state["state_bridge_error"] = status

        yield Event(
            author=self.name,
            content=types.Content(
                parts=[types.Part(text=status)],
                role="model",
            ),
        )


state_bridge = StateBridgeAgent(name="state_bridge")
