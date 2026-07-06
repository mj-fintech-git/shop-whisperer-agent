# Shop Whisperer

<p align="center">
  <img src="assets/shop_whisperer_thumbnail.png" alt="Shop Whisperer Thumbnail" width="600"/>
</p>

**Shop Whisperer** is a production-ready, multi-agent data analysis pipeline built on the **Google Agent Development Kit (ADK)** and the **Model Context Protocol (MCP)**. It bridges the gap between rigorous statistical analysis and intuitive, jargon-free business explanation.

By orchestrating a pipeline of specialized agents backed by a session-scoped analysis tool server and deterministic validation guardrails, Shop Whisperer ensures that data insights are statistically sound, logically grounded, and communicated in an accessible narrative format.

---

## Architecture Overview

```
                       [ Input Dataset & Problem ]
                                    │
                                    ▼
        ┌─────────────────────────────────────────────────────────┐
        │             ADK Sequential Pipeline Execution           │
        │                                                         │
        │  Step 1: DS_Agent (Claude Sonnet + MCP Tools)           │
        │           │                                             │
        │           ├─▶ Calls toolkit_server.py (FastMCP Server)  │
        │           └─▶ Executes ds_toolkit calculations          │
        │                                                         │
        │  Step 2: State Bridge (Deterministic Pydantic Parser)   │
        │                                                         │
        │  Step 3: Refinement Loop (LoopAgent - Max 2 Retries)    │
        │           ├──▶ storyteller (Gemini 2.0 Flash)           │
        │           └──▶ guardrail (Deterministic Python checks)   │
        │                                                         │
        │  Step 4: Output Writer (Write narrative vs. refusal)    │
        └─────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                          [ final_presentation.md ]
```

1. **The Analyst Agent (`ds_agent.py`)**: Uses a high-reasoning LLM (Claude Sonnet via LiteLLM) to perform mathematical and statistical analysis. It interacts with the dataset solely through MCP tools and outputs a structured `AnalysisResult` JSON object.
2. **The State Bridge (`state_bridge.py`)**: deterministically parses the Analyst's JSON output, validates it against a shared Pydantic schema, and saves it to the session state.
3. **The Storyteller Agent (`storyteller.py`)**: Translates structured JSON findings into a continuous, kitchen-table narrative. It has no tool access, preventing it from performing direct calculations or fabricating data.
4. **The Guardrail Agent (`guardrail_agent.py`)**: Evaluates the Storyteller's narrative using regex-based logic to verify claim grounding, confidence calibration (hedging for sample sizes $n < 20$), and narrative overreach.
5. **The Output Writer (`output_writer.py`)**: Writes the finalized narrative or a detailed refusal log to `final_presentation.md`.

---

## Key Features

- **Decoupled Computation & Storytelling**: Calculation and narrative synthesis are handled by separate agents to eliminate hallucination of statistical findings.
- **Session-Scoped MCP Caching**: The FastMCP tool server (`toolkit_server.py`) caches datasets using `moka-py` with a 30-minute idle-time expiration (TTI) for safe concurrent use.
- **Deterministic Calibration**: Programmatic rule enforcement prevents overreaching causal claims (e.g. profit claims when ad spend is absent) and motive attribution.
- **Robust Iterative Loops**: The storyteller and guardrail run inside an ADK `LoopAgent`, allowing the storyteller to self-correct based on precise feedback if a check fails.

---

## Installation Instructions

### Prerequisites
- Python 3.10 or higher
- Windows OS (runner uses `py` command launcher)

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd shop_whisperer
```

### Step 2: Install Dependencies
Install all package requirements listed in `requirements.txt`:
```bash
py -m pip install -r requirements.txt
```

### Step 3: Configure Environment Variables
Set the required API keys for the pipeline agents:
```cmd
:: Windows CMD
set ANTHROPIC_API_KEY=your-anthropic-key-here
set GOOGLE_API_KEY=your-google-gemini-key-here

:: PowerShell
$env:ANTHROPIC_API_KEY="your-anthropic-key-here"
$env:GOOGLE_API_KEY="your-google-gemini-key-here"
```

---

## Quick-Start Guide

To run the end-to-end analysis pipeline:

```bash
py analysis.py --csv orders.csv --problem "We spent a large portion of our marketing budget on Paid Ads last month to boost sales. While revenue looks high, overall profits seem flat."
```

### Command Line Arguments
- `--csv`: Path to the input CSV dataset (defaults to `orders.csv` or `DATASET_PATH` environment variable).
- `--problem`: Free-text business problem statement (defaults to the Paid Ads query or `BUSINESS_PROBLEM` environment variable).

### Output
- If the narrative passes the guardrails, the result is saved to `final_presentation.md` in the root folder.
- If the narrative fails to calibrate/ground after 2 attempts, a structured refusal report detailing the violations is written to `final_presentation.md`.

---

## API & Module Examples

### 1. Using `ds_toolkit` Directly

You can import and use the analytical primitives from `ds_toolkit` in custom scripts:

```python
import pandas as pd
from ds_toolkit import audit, metrics, stats, segments

# Load a dataset
df = audit.load_csv("orders.csv")

# 1. Audit - Check distributions
dist_df = audit.distributions(df, columns=["OrderValue_USD"])
print("Distributions:\n", dist_df)

# 2. Metrics - Calculate channel revenue summary
channel_rev = metrics.channel_revenue_summary(
    df, 
    channel_col="TrafficSource", 
    value_col="OrderValue_USD"
)
print("Channel Revenue:\n", channel_rev)

# 3. Stats - run Welch's t-test comparing group means
group_a = df.loc[df["TrafficSource"] == "Paid Ads", "OrderValue_USD"]
group_b = df.loc[df["TrafficSource"] == "Organic Search", "OrderValue_USD"]
t_test_results = stats.welch_t_test(group_a, group_b)
print("t-test results: p =", t_test_results["p_value"])

# 4. Segments - Check for potential category confounders
confound_df = segments.confound_check(
    df, 
    primary_col="TrafficSource", 
    outcome_col="OrderStatus", 
    potential_confound="Category"
)
```

### 2. Calling the MCP Server Primitives

The FastMCP server (`toolkit_server.py`) can be launched directly or queried as an MCP tool repository:

```bash
# Start the MCP server on stdio transport
py mcp_server/toolkit_server.py
```

It exposes the following tools to MCP-enabled clients:
- `load_dataset(filepath)`: Loads and caches the CSV.
- `get_schema()`: Returns schema metrics.
- `get_duplicates()`: Detects duplicate records.
- `get_outliers(method, z_threshold)`: Finds outliers in numeric variables.
- `get_distributions()`: Describes numeric columns.
- `get_channel_revenue_summary()`: Provides channel revenue breakdowns.
- `run_fisher_exact(a, b, c, d)`: Runs Fisher's Exact test.
- `run_welch_t_test(value_col, group_col, group1, group2)`: Runs Welch's t-test on cached data.
- `check_confound(primary_col, outcome_col, potential_confound)`: Identifies confounding variables.

---

## Running Tests

Verify tool parity between direct `ds_toolkit` calls and MCP tool wrapping:

```bash
py -m pytest
```
