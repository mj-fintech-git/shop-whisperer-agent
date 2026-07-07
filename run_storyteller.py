"""run_storyteller.py — Step 2 of 2: Run Storyteller + Guardrail + OutputWriter.

Reads outputs/analysis_result_prompt.txt produced by run_analyst.py
and writes outputs/final_presentation.md.

Usage:
    py run_storyteller.py [--problem "Your original question here"]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from agents.pipeline import storyteller_loop
from agents.output_writer import output_writer
from google.adk.agents import SequentialAgent

_storyteller_pipeline = SequentialAgent(
    name="storyteller_pipeline",
    sub_agents=[storyteller_loop, output_writer],
)

_PROMPT_PATH = Path("outputs") / "analysis_result_prompt.txt"


async def run(analysis_json: str, problem: str) -> None:
    session_service = InMemorySessionService()
    runner = Runner(
        agent=_storyteller_pipeline,
        app_name="shop_whisperer_storyteller",
        session_service=session_service,
    )
    session = await session_service.create_session(
        app_name="shop_whisperer_storyteller",
        user_id="analyst",
        # Pre-load the analysis result so Storyteller can read it from state
        state={
            "analysis_result_json": analysis_json,
            "business_problem": problem,
        },
    )

    # The initial message mirrors what the real pipeline passes to storyteller
    prompt = (
        f"Business problem: {problem}\n\n"
        f"Here is the structured analysis JSON for you to translate into a narrative:\n\n"
        f"{analysis_json}"
    )

    print(f"[storyteller] Starting narrative translation...")
    print(f"[storyteller] Problem: {problem[:80]}...")

    async for event in runner.run_async(
        session_id=session.id,
        user_id="analyst",
        new_message=types.Content(
            role="user",
            parts=[types.Part(text=prompt)],
        ),
    ):
        if event.author and event.content:
            for part in event.content.parts or []:
                if part.text and part.text.strip():
                    preview = part.text.strip()[:120].replace("\n", " ")
                    print(f"  [{event.author}] {preview}")

    final = await session_service.get_session(
        app_name="shop_whisperer_storyteller",
        user_id="analyst",
        session_id=session.id,
    )

    output_path = Path("outputs") / "final_presentation.md"
    passed = final.state.get("guardrail_passed", False)

    if passed and output_path.exists():
        print(f"\n[done] outputs/final_presentation.md written ({output_path.stat().st_size} bytes)")
    else:
        violations = final.state.get("violations", [])
        print(f"\n[refusal] Guardrail failed. Violations: {len(violations)}")
        print(f"          See: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Shop Whisperer — Step 2: Storyteller")
    parser.add_argument(
        "--problem",
        default=os.getenv("BUSINESS_PROBLEM", "We spent a large portion of our marketing budget on Paid Ads last month to boost sales. While revenue looks high, overall profits seem flat."),
        help="Free-text business problem (should match what you gave run_analyst.py)",
    )
    args = parser.parse_args()

    if not _PROMPT_PATH.exists():
        print(
            f"Error: {_PROMPT_PATH} not found.\n"
            "Please run Step 1 first:  py run_analyst.py --csv your_file.csv --problem \"...\"",
            file=sys.stderr,
        )
        sys.exit(1)

    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    analysis_json = _PROMPT_PATH.read_text(encoding="utf-8")
    asyncio.run(run(analysis_json, args.problem))


if __name__ == "__main__":
    main()
