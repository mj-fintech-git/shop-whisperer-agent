"""run_analyst.py — Step 1 of 2: Run only DS_Agent + StateBridge.

Saves outputs/analysis_result.json for use by run_storyteller.py.

Usage:
    py run_analyst.py --csv orders.csv --problem "Your question here"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from google.adk.agents import SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from agents.ds_agent import ds_agent
from agents.state_bridge import state_bridge

_analyst_pipeline = SequentialAgent(
    name="analyst_pipeline",
    sub_agents=[ds_agent, state_bridge],
)


async def run(csv_path: str, problem: str) -> None:
    session_service = InMemorySessionService()
    runner = Runner(
        agent=_analyst_pipeline,
        app_name="shop_whisperer_analyst",
        session_service=session_service,
    )
    session = await session_service.create_session(
        app_name="shop_whisperer_analyst",
        user_id="analyst",
        state={"csv_path": csv_path, "business_problem": problem},
    )

    prompt = (
        f"Dataset path: {Path(csv_path).resolve()}\n"
        f"Business problem: {problem}\n\n"
        "Analyse the dataset and produce a structured AnalysisResult JSON."
    )

    print(f"[analyst] Starting — dataset: {csv_path}")
    print(f"[analyst] Problem: {problem[:80]}...")

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
        app_name="shop_whisperer_analyst",
        user_id="analyst",
        session_id=session.id,
    )

    analysis_json = final.state.get("analysis_result_json")
    analysis_dict = final.state.get("analysis_result")

    if not analysis_json:
        print("\n[error] StateBridge did not produce analysis_result_json. Check DS_Agent output.", file=sys.stderr)
        sys.exit(1)

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Save the Pydantic dict for reference
    if analysis_dict:
        raw_path = outputs_dir / "analysis_result.json"
        raw_path.write_text(json.dumps(analysis_dict, indent=2), encoding="utf-8")
        print(f"\n[analyst] outputs/analysis_result.json written ({raw_path.stat().st_size} bytes)")

    # Save the storyteller prompt string — this is what run_storyteller.py will load
    prompt_path = outputs_dir / "analysis_result_prompt.txt"
    prompt_path.write_text(analysis_json, encoding="utf-8")
    print(f"[analyst] outputs/analysis_result_prompt.txt written ({prompt_path.stat().st_size} bytes)")
    print("\n[analyst] Done. Now run: py run_storyteller.py")


def main() -> None:
    parser = argparse.ArgumentParser(description="Shop Whisperer — Step 1: Analyst")
    parser.add_argument("--csv", default=os.getenv("DATASET_PATH"), help="Path to the input CSV file")
    parser.add_argument(
        "--problem",
        default=os.getenv("BUSINESS_PROBLEM", "We spent a large portion of our marketing budget on Paid Ads last month to boost sales. While revenue looks high, overall profits seem flat."),
        help="Free-text business problem statement",
    )
    args = parser.parse_args()

    if not args.csv:
        print("Error: No dataset CSV provided. Use --csv or set DATASET_PATH.", file=sys.stderr)
        sys.exit(1)
    if not Path(args.csv).exists():
        print(f"Error: Dataset file not found: {args.csv}", file=sys.stderr)
        sys.exit(1)
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("[analyst] ANTHROPIC_API_KEY not set. Falling back to gemini-2.5-flash for DS_Agent.")
        ds_agent.model = "gemini-2.5-flash"

    asyncio.run(run(args.csv, args.problem))


if __name__ == "__main__":
    main()
