"""
agents/ds_agent.py — DS_Agent: LlmAgent wrapping ds_toolkit via MCP.

Model:     Claude Sonnet (via LiteLLM) with reasoning_effort="medium"
Tools:     All ds_toolkit functions via MCPToolset (stdio subprocess)
Output:    JSON block matching AnalysisResult schema, stored to session state
           by state_bridge.py downstream.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool import MCPToolset, StdioConnectionParams
from mcp import StdioServerParameters

# ── Skill loader ──────────────────────────────────────────────────────────────

def _load_skill(skill_name: str) -> str:
    skill_path = Path(__file__).parent.parent / "skills" / skill_name / "SKILL.md"
    if not skill_path.exists():
        raise FileNotFoundError(f"Skill file not found: {skill_path}")
    # Strip YAML frontmatter
    text = skill_path.read_text(encoding="utf-8")
    if text.startswith("---"):
        parts = text.split("---", 2)
        return parts[2].strip() if len(parts) >= 3 else text
    return text


# ── MCP server subprocess config ──────────────────────────────────────────────

_SERVER_SCRIPT = str(Path(__file__).parent.parent / "mcp_server" / "toolkit_server.py")
_PYTHON = sys.executable   # same interpreter as the runner

_mcp_toolset = MCPToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command=_PYTHON,
            args=[_SERVER_SCRIPT],
            env={**os.environ, "MCP_TRANSPORT": "stdio"},
        )
    )
)


# ── Agent definition ──────────────────────────────────────────────────────────

ds_agent = LlmAgent(
    # LiteLlm wrapper routes through LiteLLM → Anthropic.
    # reasoning_effort is a LiteLlm constructor kwarg, not a
    # generate_content_config field — passing it via extra_body raises a
    # pydantic ValidationError since GenerateContentConfig has no such field.
    # Set ANTHROPIC_API_KEY in environment before running.
    model=LiteLlm(model="anthropic/claude-sonnet-4-5", reasoning_effort="medium"),
    name="ds_agent",
    description=(
        "Expert data scientist. Calls ds_toolkit MCP tools to analyse the dataset "
        "and produces a structured AnalysisResult JSON — no prose narrative."
    ),
    instruction=_load_skill("analyst"),
    tools=[_mcp_toolset],
    generate_content_config={
        "temperature": 1.0,          # required when reasoning is active
    },
)
