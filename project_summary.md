# Project Write-Up: Shop Whisperer

**Shop Whisperer** is a localized, production-ready multi-agent system designed to solve a classic corporate friction: the gap between rigorous, expert-level data science and the intuitive, jargon-free explanations needed by business leaders, clients, or non-technical stakeholders.

The project is structured around the **Google Agent Development Kit (ADK)** and the **Model Context Protocol (MCP)**. It orchestrates a linear pipeline of specialized agents backed by a stateful, session-scoped analysis tool server and deterministic validation guardrails.

---

## 1. Core Architecture & Component Separation

```
                       [ Input Dataset & Problem ]
                                    │
                                    ▼
       ┌─────────────────────────────────────────────────────────┐
       │             ADK Sequential Pipeline Execution           │
       │                                                         │
       │  Step 1: DS_Agent (Claude Sonnet + reasoning_effort)   │
       │           │                                             │
       │           ├─▶ MCP Tool Calls                            │
       │           └─▶ toolkit_server.py (FastMCP + moka TTI)    │
       │                                                         │
       │  Step 2: State Bridge (Deterministic JSON Parser)      │
       │                                                         │
       │  Step 3: Refinement Loop (ADK LoopAgent — 2 Iterations) │
       │           │                                             │
       │           ├──▶ storyteller_agent (Gemini 2.0 Flash)     │
       │           │                                             │
       │           └──▶ guardrail_agent (Post-check validations)  │
       │                                                         │
       │  Step 4: Output Writer (Refusal vs. Clean Presentation) │
       └─────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                         [ final_presentation.md ]
```

### The Analyst Agent (`ds_agent.py`)
*   **Model**: `claude-sonnet-4-5` (via LiteLLM) with `reasoning_effort` enabled.
*   **Responsibility**: Performs pure, structured mathematical and statistical analysis. It is explicitly prohibited from generating prose, narratives, or hand-waving explanations.
*   **Tools**: Connects as an MCP client to a subprocess tool server containing data science primitives.
*   **Output**: A strict `AnalysisResult` JSON object mapping findings, data gaps, next steps, and metadata.

### The Storyteller Agent (`storyteller.py`)
*   **Model**: `gemini-2.0-flash` (native ADK).
*   **Responsibility**: Takes the structured JSON output from the Analyst and translates it into a kitchen-table explanation. It is completely isolated from the data tools to prevent it from performing calculations or fabricating findings.
*   **Persona constraints**: Enforces a natural, spoken rhythm (varying sentence structures, uneven cadences) and bans common "composed corporate prose" indicators.

### The State Bridge (`state_bridge.py`)
*   A deterministic intermediate step that parses the Analyst's JSON output block from raw history, validates it against Pydantic models, and writes it directly to the ADK session state. This decouples the inter-agent communications protocol from conversational text.

---

## 2. Shared Data Contract (`schema.py`)

A unified Pydantic schema dictates the interface between the Analyst and the Storyteller.

*   **Collapsed Certainty Tiers**: Rather than passing separate boolean variables for arithmetic facts and confidence metrics, findings use a single `CertaintyLevel` classification:
    *   `Factual`: Pure arithmetic calculations (e.g., channel return rates) requiring no statistical inference. Hedges are never required.
    *   `High`: Statistically significant tests ($p < 0.05$) with a meaningful effect size.
    *   `Medium`: Directional indicators that are not statistically robust (e.g., $p > 0.05$ with large effect, or $p < 0.05$ with small samples).
    *   `Low`: Weak signals or descriptive measurements on tiny subgroups ($n < 5$).
    *   `Inconclusive`: Explicit declarations where missing variables or design parameters prevent an answer.
*   **Calibration Rules**: Any finding marked as non-factual with a sample size ($n$) under 20 is programmatically capped, forcing the Storyteller to use hedging vocabulary ("looks like," "seems like," "probably").

---

## 3. Session-Scoped MCP Tool Server (`toolkit_server.py`)

The system converts a modular library of analytics scripts (`ds_toolkit`) into an **MCP Server** using the `FastMCP` framework.

*   **Stateful Cache**: Exposes a `load_dataset` tool that caches the dataset inside a `moka-py` Moka cache. All subsequent analysis tools (outliers, distributions, regressions, Fisher's exact tests, confound checks) read from this cache.
*   **TTI Eviction (Time-To-Idle)**: Keying the cache by `ctx.client_id` with a 30-minute idle-time expiration ensures that the server behaves statelessly over `stdio` (single client), but remains fully compatible with multi-user HTTP/SSE environments (preventing memory leaks or data cross-contamination).
*   **Stdout Redirect Wrapper**: Since MCP utilizes `stdout` for JSON-RPC messages, any diagnostic `print()` statements inside legacy toolkit code would corrupt the stream. The server wraps all tool calls in a context manager that redirects standard output to `sys.stderr`.

---

## 4. Deterministic Guardrails (`output_checker.py`)

Rather than relying on LLM self-policing, the storyteller's narrative is validated by a deterministic Python module before being written to disk:

1.  **Claim Grounding**: Extracts all percentages, monetary values, and fractions from the narrative, verifying that every number exists verbatim in the Analyst's structured findings.
2.  **Confidence Calibration**: Checks that certainty-implying phrases ("confirmed," "proven," "we know") are absent if any non-factual finding relies on a sample size $n < 20$.
3.  **Narrative Overreach**: Block profit-causation statements if required cost variables (ad spend, cost of goods sold) are missing. Flags any motive-attributing language (e.g., "customer decided to return it because they changed their mind") to ensure only observable behavior is narrated.

---

## 5. Iterative Refinement Loop

The storyteller and guardrail checker run inside an ADK `LoopAgent` restricted to a maximum of 2 iterations:

*   **Iteration 1**: Storyteller generates a draft. Guardrail agent runs checks. If checks pass, `guardrail_exit` is set to `True` in the session state, ending the loop immediately.
*   **Iteration 2 (Retry)**: If the draft fails, the guardrail agent writes the specific failures back to the session state. The storyteller reads this context and attempts a corrected draft.
*   **Refusal Path**: If the checks still fail on the second pass, the pipeline halts and writes a structured refusal document outlining the violations instead of outputting unverified narrative content.
