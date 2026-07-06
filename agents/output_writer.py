"""
agents/output_writer.py — Final step: writes the narrative or a refusal to disk.

Reads:   session.state["guardrail_passed"]
         session.state["narrative"]
         session.state["violations"]
Writes:  final_presentation.md (in the workspace root)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import AsyncGenerator

from google.adk.agents import BaseAgent, InvocationContext
from google.adk.events import Event
from google.genai import types

_OUTPUT_PATH = Path(__file__).parent.parent / "final_presentation.md"
_JSON_OUTPUT_PATH = Path(__file__).parent.parent / "analysis_result.json"

_REFUSAL_TEMPLATE = """\
[Pipeline refusal — narrative did not pass guardrail checks after {n_attempts} attempt(s)]

The Storyteller output failed the following checks:

{violations}

To diagnose: review the violations above, check the AnalysisResult in session state,
and re-run the pipeline with a corrected Storyteller skill or updated Finding data.
"""


class OutputWriter(BaseAgent):
    """Agent that writes the final storyteller narrative or refusal template to disk.

    Reads the verification status and storyteller text from session state,
    then writes the clean prose or formatted error log.
    """

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """Asynchronously runs the output writer to dump narrative to a markdown file.

        Args:
            ctx: The execution context containing session state.

        Yields:
            Event detailing the status of the file write operation.
        """

        passed = ctx.session.state.get("guardrail_passed", False)
        narrative = ctx.session.state.get("narrative", "")
        violations = ctx.session.state.get("violations", [])
        raw_json = ctx.session.state.get("analysis_result_json")

        if passed and narrative:
            output = narrative
            status = f"[done] Written to {_OUTPUT_PATH}"
        else:
            output = _REFUSAL_TEMPLATE.format(
                n_attempts=2,
                violations=json.dumps(violations, indent=2),
            )
            status = f"[refusal] Guardrail failed after max iterations. Refusal written to {_OUTPUT_PATH}"

        _OUTPUT_PATH.write_text(output, encoding="utf-8")

        if raw_json:
            _JSON_OUTPUT_PATH.write_text(raw_json, encoding="utf-8")
            status += f" (raw JSON written to {_JSON_OUTPUT_PATH.name})"

        yield Event(
            author=self.name,
            content=types.Content(
                parts=[types.Part(text=status)],
                role="model",
            ),
        )


output_writer = OutputWriter(name="output_writer")
