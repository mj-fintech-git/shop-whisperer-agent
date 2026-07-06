"""mcp_server/toolkit_server.py — FastMCP server wrapping all ds_toolkit functions.

Key design decisions:
  - Session-scoped DataFrame storage via moka-py TTI cache (idle-time eviction, not session-age)
  - stdout redirected to stderr during all toolkit calls (MCP uses stdout for JSON-RPC)
  - All outputs are JSON-serialisable (DataFrames → list[dict], numpy → Python primitives)
  - load_dataset() must be called first — all analysis tools read from cache

Transport: stdio (default). SSE-ready by changing mcp.run(transport=...) — no logic changes.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
from pathlib import Path
from typing import Any

# ── Ensure ds_toolkit is importable ───────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from mcp.server.fastmcp import FastMCP, Context

from ds_toolkit import audit, metrics, segments, stats

# ── Session-scoped TTI cache ──────────────────────────────────────────────────
try:
    from moka_py import Moka
    _session_data: Any = Moka(
        int(os.getenv("MCP_MAX_SESSIONS", "200")),          # capacity (positional)
        tti=float(os.getenv("SESSION_TTI_SECONDS", "1800")), # idle-time TTL in seconds
    )
    _MOKA_AVAILABLE = True
except ImportError:
    # Fallback: plain dict (stdio only — no concurrent sessions)
    _session_data = {}
    _MOKA_AVAILABLE = False
    print("[mcp_server] moka-py not available — using plain dict cache (stdio only)", file=sys.stderr)


def _session_id(ctx: Context) -> str:
    """Retrieve the session identifier from the execution context.

    Args:
        ctx: FastMCP Context object.

    Returns:
        The session ID (str). Defaults to "default" if none is present.
    """
    return getattr(ctx, "client_id", None) or "default"


def _get_df(ctx: Context) -> pd.DataFrame:
    """Retrieve the loaded dataset for the current session.

    Args:
        ctx: FastMCP Context object.

    Returns:
        The session-scoped pandas DataFrame.

    Raises:
        ValueError: If no dataset has been loaded for the current session.
    """
    sid = _session_id(ctx)
    df = _session_data.get(sid) if _MOKA_AVAILABLE else _session_data.get(sid)
    if df is None:
        raise ValueError(
            f"No dataset loaded for session '{sid}'. Call load_dataset(filepath) first."
        )
    return df


def _set_df(ctx: Context, df: pd.DataFrame) -> None:
    """Cache the loaded dataset for the current session.

    Args:
        ctx: FastMCP Context object.
        df: The pandas DataFrame to cache.
    """
    sid = _session_id(ctx)
    if _MOKA_AVAILABLE:
        _session_data.set(sid, df)   # moka-py API: .set(key, value)
    else:
        _session_data[sid] = df


# ── Serialisation helpers ──────────────────────────────────────────────────────

def _to_python(obj: Any) -> Any:
    """Recursively convert numpy/pandas types to JSON-serialisable Python primitives.

    Args:
        obj: The object to convert.

    Returns:
        The converted JSON-serialisable object.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return round(float(obj), 6)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return [_to_python(r) for r in obj.to_dict("records")]
    if isinstance(obj, pd.Series):
        return _to_python(obj.to_dict())
    if isinstance(obj, dict):
        return {str(k): _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(v) for v in obj]
    return obj


# ── stdout redirect context manager ──────────────────────────────────────────

@contextlib.contextmanager
def _toolkit_call():
    """Context manager redirecting stdout to stderr during execution.

    MCP transport uses stdout for JSON-RPC communications, so printing
    to stdout inside tool execution will corrupt the communication channel.
    """
    old_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = old_stdout


# ── FastMCP server ────────────────────────────────────────────────────────────

mcp = FastMCP("ds-toolkit")


# ── Dataset loading ───────────────────────────────────────────────────────────

@mcp.tool()
def load_dataset(filepath: str, ctx: Context) -> dict:
    """Load a CSV file into the session cache. Must be called before any analysis tool.

    Args:
        filepath: Absolute path to the CSV file to load.
        ctx: FastMCP execution context.

    Returns:
        A dictionary with loaded dataset metadata (session_id, rows, cols, columns, dtypes).
    """
    with _toolkit_call():
        df = audit.load_csv(filepath)
    _set_df(ctx, df)
    return _to_python({
        "session_id": _session_id(ctx),
        "rows": len(df),
        "cols": len(df.columns),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    })


# ── Audit tools ───────────────────────────────────────────────────────────────

@mcp.tool()
def get_schema(ctx: Context) -> list[dict]:
    """Return column schema: name, dtype, non-null count, null count, null %.

    Args:
        ctx: FastMCP execution context.

    Returns:
        List of dictionaries detailing column schemas.
    """
    with _toolkit_call():
        result = audit.schema(_get_df(ctx))
    return _to_python(result)


@mcp.tool()
def get_duplicates(ctx: Context) -> dict:
    """Return duplicate row count, percentage, and action taken.

    Args:
        ctx: FastMCP execution context.

    Returns:
        Dictionary with duplicate metrics.
    """
    with _toolkit_call():
        result = audit.duplicates(_get_df(ctx))
    return _to_python(result)


@mcp.tool()
def get_outliers(
    ctx: Context,
    method: str = "iqr",
    z_threshold: float = 3.0,
) -> dict:
    """Detect outliers in numeric columns using IQR or Z-score method.

    Args:
        ctx: FastMCP execution context.
        method: Method name, one of "iqr" or "zscore". Defaults to "iqr".
        z_threshold: Z-score cutoff (used when method='zscore'). Defaults to 3.0.

    Returns:
        Dictionary detailing outliers found.
    """
    with _toolkit_call():
        result = audit.outliers(_get_df(ctx), method=method, z_threshold=z_threshold)
    return _to_python(result)


@mcp.tool()
def get_distributions(ctx: Context) -> list[dict]:
    """Return mean, median, std, min, max, skewness for all numeric columns.

    Args:
        ctx: FastMCP execution context.

    Returns:
        List of dictionaries with column distributions.
    """
    with _toolkit_call():
        result = audit.distributions(_get_df(ctx))
    return _to_python(result)


@mcp.tool()
def get_categorical_frequencies(ctx: Context) -> dict:
    """Return value counts for all categorical (object/string) columns.

    Args:
        ctx: FastMCP execution context.

    Returns:
        Dictionary with column frequency counts.
    """
    with _toolkit_call():
        result = audit.categorical_frequencies(_get_df(ctx))
    return _to_python(result)


# ── Metrics tools ─────────────────────────────────────────────────────────────

@mcp.tool()
def get_effective_revenue(
    ctx: Context,
    value_col: str = "OrderValue_USD",
    status_col: str = "OrderStatus",
    returned_value: str = "Returned",
) -> dict:
    """Overall gross revenue, return value, effective revenue, and rates.

    Args:
        ctx: FastMCP execution context.
        value_col: Numeric column for monetary order value. Defaults to "OrderValue_USD".
        status_col: Column containing order status. Defaults to "OrderStatus".
        returned_value: Value representing returned orders. Defaults to "Returned".

    Returns:
        Dictionary with effective revenue metrics.
    """
    with _toolkit_call():
        result = metrics.effective_revenue(
            _get_df(ctx),
            value_col=value_col,
            status_col=status_col,
            returned_value=returned_value,
        )
    return _to_python(result)


@mcp.tool()
def get_return_rate_by_group(
    ctx: Context,
    group_col: str,
    status_col: str = "OrderStatus",
    returned_value: str = "Returned",
) -> list[dict]:
    """Return rate (count and value) broken down by a categorical group column.

    Args:
        ctx: FastMCP execution context.
        group_col: Column to group by.
        status_col: Column containing order status. Defaults to "OrderStatus".
        returned_value: Value representing returned orders. Defaults to "Returned".

    Returns:
        List of dictionaries representing return rates per group.
    """
    with _toolkit_call():
        result = metrics.return_rate_by_group(
            _get_df(ctx),
            group_col=group_col,
            status_col=status_col,
            returned_value=returned_value,
        )
    return _to_python(result)


@mcp.tool()
def get_channel_revenue_summary(
    ctx: Context,
    channel_col: str = "TrafficSource",
    value_col: str = "OrderValue_USD",
    status_col: str = "OrderStatus",
    returned_value: str = "Returned",
) -> list[dict]:
    """Full revenue breakdown per acquisition channel: gross, effective, returns, AOV.

    Args:
        ctx: FastMCP execution context.
        channel_col: Channel column name. Defaults to "TrafficSource".
        value_col: Monetary value column. Defaults to "OrderValue_USD".
        status_col: Order status column. Defaults to "OrderStatus".
        returned_value: Value representing returns. Defaults to "Returned".

    Returns:
        List of dictionaries summarizing channel revenues.
    """
    with _toolkit_call():
        result = metrics.channel_revenue_summary(
            _get_df(ctx),
            channel_col=channel_col,
            value_col=value_col,
            status_col=status_col,
            returned_value=returned_value,
        )
    return _to_python(result)


@mcp.tool()
def get_aov_by_group(
    ctx: Context,
    group_col: str,
    value_col: str = "OrderValue_USD",
    status_col: str | None = None,
    completed_only: bool = False,
    returned_value: str = "Returned",
) -> list[dict]:
    """Average order value per group, optionally restricted to completed orders.

    Args:
        ctx: FastMCP execution context.
        group_col: Column to group by.
        value_col: Monetary value column. Defaults to "OrderValue_USD".
        status_col: Order status column. Defaults to None.
        completed_only: Boolean flag to check only completed orders. Defaults to False.
        returned_value: Value representing returns. Defaults to "Returned".

    Returns:
        List of dictionaries with group AOVs.
    """
    with _toolkit_call():
        result = metrics.aov_by_group(
            _get_df(ctx),
            group_col=group_col,
            value_col=value_col,
            status_col=status_col,
            completed_only=completed_only,
            returned_value=returned_value,
        )
    return _to_python(result)


@mcp.tool()
def get_revenue_concentration(
    ctx: Context,
    group_col: str,
    value_col: str = "OrderValue_USD",
    top_n: int = 3,
) -> dict:
    """Compute what % of total revenue comes from the top N groups.

    Args:
        ctx: FastMCP execution context.
        group_col: Column to check concentration for.
        value_col: Monetary value column. Defaults to "OrderValue_USD".
        top_n: Rank threshold. Defaults to 3.

    Returns:
        Dictionary detailing concentration metrics.
    """
    with _toolkit_call():
        result = metrics.revenue_concentration(
            _get_df(ctx),
            group_col=group_col,
            value_col=value_col,
            top_n=top_n,
        )
    return _to_python(result)


@mcp.tool()
def get_repeat_customer_profile(
    ctx: Context,
    customer_col: str = "CustomerID",
    value_col: str = "OrderValue_USD",
    status_col: str = "OrderStatus",
    channel_col: str = "TrafficSource",
    returned_value: str = "Returned",
) -> list[dict]:
    """Profile of customers with more than one order: counts, values, return rates.

    Args:
        ctx: FastMCP execution context.
        customer_col: Customer identifier column. Defaults to "CustomerID".
        value_col: Monetary value column. Defaults to "OrderValue_USD".
        status_col: Order status column. Defaults to "OrderStatus".
        channel_col: Traffic source channel column. Defaults to "TrafficSource".
        returned_value: Value representing returns. Defaults to "Returned".

    Returns:
        List of dictionaries representing repeat customer profiles.
    """
    with _toolkit_call():
        result = metrics.repeat_customer_profile(
            _get_df(ctx),
            customer_col=customer_col,
            value_col=value_col,
            status_col=status_col,
            channel_col=channel_col,
            returned_value=returned_value,
        )
    return _to_python(result)


# ── Statistics tools ──────────────────────────────────────────────────────────

@mcp.tool()
def run_fisher_exact(
    a: int, b: int, c: int, d: int,
    alternative: str = "two-sided",
    ctx: Context = None,
) -> dict:
    """Fisher's exact test on a 2x2 contingency table.

    Table layout: [[a, b], [c, d]] where a=group1_event, b=group1_no_event, etc.

    Args:
        a: Cell A frequency.
        b: Cell B frequency.
        c: Cell C frequency.
        d: Cell D frequency.
        alternative: Test hypothesis type ("two-sided", "greater", "less"). Defaults to "two-sided".
        ctx: FastMCP execution context. Defaults to None.

    Returns:
        Dictionary with Fisher test statistics.
    """
    with _toolkit_call():
        result = stats.fisher_exact_2x2(a, b, c, d, alternative=alternative)
    return _to_python(result)


@mcp.tool()
def run_chi_squared(
    ctx: Context,
    col1: str,
    col2: str,
) -> dict:
    """Chi-squared test of independence between two categorical columns.

    Args:
        ctx: FastMCP execution context.
        col1: First categorical column name.
        col2: Second categorical column name.

    Returns:
        Dictionary with Chi-Squared test statistics.
    """
    df = _get_df(ctx)
    with _toolkit_call():
        contingency = pd.crosstab(df[col1], df[col2])
        result = stats.chi_squared(contingency.values)
    return _to_python(result)


@mcp.tool()
def run_pearson_correlation(
    ctx: Context,
    col1: str,
    col2: str,
) -> dict:
    """Pearson correlation coefficient between two numeric columns.

    Args:
        ctx: FastMCP execution context.
        col1: First numeric column name.
        col2: Second numeric column name.

    Returns:
        Dictionary with Pearson correlation statistics.
    """
    df = _get_df(ctx)
    with _toolkit_call():
        result = stats.pearson_correlation(df[col1], df[col2])
    return _to_python(result)


@mcp.tool()
def run_spearman_correlation(
    ctx: Context,
    col1: str,
    col2: str,
) -> dict:
    """Spearman rank correlation between two numeric columns (robust to outliers).

    Args:
        ctx: FastMCP execution context.
        col1: First numeric column name.
        col2: Second numeric column name.

    Returns:
        Dictionary with Spearman correlation statistics.
    """
    df = _get_df(ctx)
    with _toolkit_call():
        result = stats.spearman_correlation(df[col1], df[col2])
    return _to_python(result)


@mcp.tool()
def run_mannwhitney(
    ctx: Context,
    value_col: str,
    group_col: str,
    group1: str,
    group2: str,
    alternative: str = "two-sided",
) -> dict:
    """Mann-Whitney U test comparing distributions of two groups.

    Args:
        ctx: FastMCP execution context.
        value_col: Numeric value column name.
        group_col: Categorical group column name.
        group1: Label of the first group.
        group2: Label of the second group.
        alternative: Test hypothesis type ("two-sided", "greater", "less"). Defaults to "two-sided".

    Returns:
        Dictionary with Mann-Whitney U test statistics.
    """
    df = _get_df(ctx)
    with _toolkit_call():
        group_a = df.loc[df[group_col] == group1, value_col]
        group_b = df.loc[df[group_col] == group2, value_col]
        result = stats.mannwhitney_u(group_a, group_b, alternative=alternative)
    return _to_python(result)


@mcp.tool()
def run_welch_t_test(
    ctx: Context,
    value_col: str,
    group_col: str,
    group1: str,
    group2: str,
    alternative: str = "two-sided",
) -> dict:
    """Welch's t-test comparing means of two groups (unequal variances assumed).

    Args:
        ctx: FastMCP execution context.
        value_col: Numeric value column name.
        group_col: Categorical group column name.
        group1: Label of the first group.
        group2: Label of the second group.
        alternative: Test hypothesis type ("two-sided", "greater", "less"). Defaults to "two-sided".

    Returns:
        Dictionary with Welch t-test statistics.
    """
    df = _get_df(ctx)
    with _toolkit_call():
        group_a = df.loc[df[group_col] == group1, value_col]
        group_b = df.loc[df[group_col] == group2, value_col]
        result = stats.welch_t_test(group_a, group_b, alternative=alternative)
    return _to_python(result)


# ── Segments tools ────────────────────────────────────────────────────────────

@mcp.tool()
def get_cross_tab(
    ctx: Context,
    row_col: str,
    col_col: str,
    normalize: bool = False,
) -> dict:
    """Cross-tabulation between two categorical columns.

    Args:
        ctx: FastMCP execution context.
        row_col: Row categorical column name.
        col_col: Column categorical column name.
        normalize: Boolean flag to return percentages. Defaults to False.

    Returns:
        Dictionary crosstab output.
    """
    with _toolkit_call():
        result = segments.cross_tab(_get_df(ctx), row_col=row_col, col_col=col_col, normalize=normalize)
    return _to_python(result)


@mcp.tool()
def check_confound(
    ctx: Context,
    primary_col: str,
    outcome_col: str,
    potential_confound: str,
) -> dict:
    """Check whether potential_confound explains the relationship between primary_col and outcome_col.

    Returns cross-tab of potential_confound x primary_col.

    Args:
        ctx: FastMCP execution context.
        primary_col: Column representing independent variable.
        outcome_col: Column representing outcome variable.
        potential_confound: Column representing potential confounder variable.

    Returns:
        Dictionary confound check output.
    """
    with _toolkit_call():
        result = segments.confound_check(
            _get_df(ctx),
            primary_col=primary_col,
            outcome_col=outcome_col,
            potential_confound=potential_confound,
        )
    return _to_python(result)


@mcp.tool()
def get_cohort_return_analysis(
    ctx: Context,
    cohort_col: str,
    status_col: str = "OrderStatus",
    value_col: str = "OrderValue_USD",
    returned_value: str = "Returned",
) -> list[dict]:
    """Return profile by cohort group.

    Args:
        ctx: FastMCP execution context.
        cohort_col: Column defining cohorts.
        status_col: Order status column. Defaults to "OrderStatus".
        value_col: Monetary value column. Defaults to "OrderValue_USD".
        returned_value: Value representing returns. Defaults to "Returned".

    Returns:
        List of dictionaries with cohort return details.
    """
    with _toolkit_call():
        result = segments.cohort_return_analysis(
            _get_df(ctx),
            cohort_col=cohort_col,
            status_col=status_col,
            value_col=value_col,
            returned_value=returned_value,
        )
    return _to_python(result)


@mcp.tool()
def get_segment_profile(
    ctx: Context,
    segment_col: str,
    value_col: str = "OrderValue_USD",
    status_col: str = "OrderStatus",
    returned_value: str = "Returned",
) -> dict:
    """Full segment profile: order counts, revenue stats, return rates per segment.

    Args:
        ctx: FastMCP execution context.
        segment_col: Column defining segments.
        value_col: Monetary value column. Defaults to "OrderValue_USD".
        status_col: Order status column. Defaults to "OrderStatus".
        returned_value: Value representing returns. Defaults to "Returned".

    Returns:
        Dictionary detailing segment profile stats.
    """
    with _toolkit_call():
        result = segments.segment_profile(
            _get_df(ctx),
            segment_col=segment_col,
            numeric_cols=[value_col],
            categorical_cols=[status_col],
        )
    return _to_python(result)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "stdio")
    print(f"[ds-toolkit MCP server] Starting ({transport})", file=sys.stderr)
    mcp.run(transport=transport)
