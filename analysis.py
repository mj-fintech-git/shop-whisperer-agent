"""analysis.py — Entrypoint for the shop_whisperer 2-agent pipeline.

Usage:
    py -3 analysis.py [--csv orders.csv] [--problem "..."]

    Or set environment variables:
        DATASET_PATH=orders.csv
        BUSINESS_PROBLEM="..."
        ANTHROPIC_API_KEY=sk-...
        GOOGLE_API_KEY=...

The pipeline:
    DS_Agent (Claude + MCP tools) → StateBridge → LoopAgent[Storyteller + Guardrail] → OutputWriter
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

from agents.pipeline import pipeline


def _build_prompt(csv_path: str, problem: str) -> str:
    """Construct the initial prompt context for the DS_Agent.

    Args:
        csv_path: Path to the CSV dataset.
        problem: Free-text business problem statement.

    Returns:
        The formatted prompt text (str).
    """
    return (
        f"Dataset path: {Path(csv_path).resolve()}\n"
        f"Business problem: {problem}\n\n"
        "Analyse the dataset and produce a structured AnalysisResult JSON."
    )


async def run(csv_path: str, problem: str) -> None:
    """Asynchronously runs the sequential multi-agent analysis pipeline.

    Args:
        csv_path: Path to the input dataset CSV.
        problem: Business problem statement.
    """
    session_service = InMemorySessionService()

    runner = Runner(
        agent=pipeline,
        app_name="shop_whisperer",
        session_service=session_service,
    )

    session = await session_service.create_session(
        app_name="shop_whisperer",
        user_id="analyst",
        state={
            "csv_path": csv_path,
            "business_problem": problem,
        },
    )

    prompt = _build_prompt(csv_path, problem)
    print(f"[pipeline] Starting — dataset: {csv_path}")
    print(f"[pipeline] Problem: {problem[:80]}...")

    async for event in runner.run_async(
        session_id=session.id,
        user_id="analyst",
        new_message=types.Content(
            role="user",
            parts=[types.Part(text=prompt)],
        ),
    ):
        # Log agent activity
        if event.author and event.content:
            for part in event.content.parts or []:
                if part.text and part.text.strip():
                    preview = part.text.strip()[:120].replace("\n", " ")
                    print(f"  [{event.author}] {preview}")

    # Final state check
    final_session = await session_service.get_session(
        app_name="shop_whisperer",
        user_id="analyst",
        session_id=session.id,
    )
    passed = final_session.state.get("guardrail_passed", False)
    output_path = Path(__file__).parent / "outputs" / "final_presentation.md"

    if passed:
        print(f"\n[done] outputs/final_presentation.md written ({output_path.stat().st_size} bytes)")
    else:
        violations = final_session.state.get("violations", [])
        print(f"\n[refusal] Guardrail failed. Violations: {len(violations)}")
        print(f"          Refusal written to {output_path}")


def main() -> None:
    """Main CLI entrypoint to parse arguments, validate environment, and trigger pipeline execution."""
    parser = argparse.ArgumentParser(description="Shop Whisperer — 2-agent DS pipeline")
    parser.add_argument(
        "--csv",
        default=os.getenv("DATASET_PATH"),
        help="Path to the input CSV file",
    )
    parser.add_argument(
        "--problem",
        default=os.getenv(
            "BUSINESS_PROBLEM",
            "We spent a large portion of our marketing budget on Paid Ads last month "
            "to boost e-commerce sales. While revenue looks high, our overall profits seem flat.",
        ),
        help="Free-text business problem statement",
    )
    args = parser.parse_args()

    # Validate inputs
    if not args.csv:
        print("Error: No dataset CSV file provided. Please specify one with --csv or the DATASET_PATH environment variable.", file=sys.stderr)
        sys.exit(1)

    if not Path(args.csv).exists():
        print(f"Error: Dataset file not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise EnvironmentError("ANTHROPIC_API_KEY is not set (required for DS_Agent)")
    if not os.getenv("GOOGLE_API_KEY"):
        raise EnvironmentError("GOOGLE_API_KEY is not set (required for Storyteller)")

    asyncio.run(run(args.csv, args.problem))


if __name__ == "__main__":
    main()
