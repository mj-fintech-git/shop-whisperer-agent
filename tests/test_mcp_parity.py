"""tests/test_mcp_parity.py — Verify MCP-wrapped tools produce identical output to direct calls.

Each test:
  1. Calls the original ds_toolkit function directly
  2. Calls the equivalent MCP tool via the FastMCP test client
  3. Asserts outputs are identical (within float precision)

This is a refactor test — computed values must not shift.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from mcp.server.fastmcp import FastMCP

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from ds_toolkit import audit, metrics, segments, stats

# ── Load test fixture ─────────────────────────────────────────────────────────
ORDERS_CSV = str(ROOT / "orders.csv")
_df: pd.DataFrame = audit.load_csv(ORDERS_CSV)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _roundtrip(obj: Any) -> object:
    """Simulate JSON roundtrip (what MCP does over the wire).

    Args:
        obj: The python object to serialise and deserialise.

    Returns:
        The roundtripped Python object.
    """
    return json.loads(json.dumps(obj, default=str))


def _df_to_records(df: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame to a list of record dictionaries.

    Args:
        df: The pandas DataFrame.

    Returns:
        A list of dictionaries.
    """
    return _roundtrip(df.to_dict("records"))


def _to_python(obj: Any) -> Any:
    """Mirror the MCP server's _to_python helper.

    Converts pandas/numpy types recursively.

    Args:
        obj: Object to convert.

    Returns:
        The converted serialisable object.
    """
    import numpy as np
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


# ── Import MCP server tools for direct invocation ─────────────────────────────
# We import the server module and invoke the underlying functions directly
# (bypassing the FastMCP protocol layer) to verify the wrapping logic is correct.
# This tests the tool *implementation*, not the JSON-RPC transport.

from mcp_server.toolkit_server import (
    _to_python as server_to_python,
    _toolkit_call,
)

# Simulate a session with a pre-loaded DataFrame
class _FakeCtx:
    """Fake Context class mimicking FastMCP context for tests."""
    client_id = "test_session"

_fake_ctx = _FakeCtx()

# Seed the server's session cache directly
from mcp_server import toolkit_server as _srv
_srv._set_df(_fake_ctx, _df.copy())


# ── Test: Schema ──────────────────────────────────────────────────────────────

def test_schema_parity():
    direct = _to_python(audit.schema(_df))
    via_mcp = _srv.get_schema(_fake_ctx)
    assert direct == via_mcp, f"Schema mismatch:\n{direct}\nvs\n{via_mcp}"


# ── Test: Duplicates ──────────────────────────────────────────────────────────

def test_duplicates_parity():
    direct = _to_python(audit.duplicates(_df))
    via_mcp = _srv.get_duplicates(_fake_ctx)
    assert direct == via_mcp


# ── Test: Outliers ────────────────────────────────────────────────────────────

def test_outliers_parity():
    direct = _to_python(audit.outliers(_df, method="iqr", z_threshold=3.0))
    via_mcp = _srv.get_outliers(_fake_ctx, method="iqr", z_threshold=3.0)
    assert direct == via_mcp


# ── Test: Distributions ───────────────────────────────────────────────────────

def test_distributions_parity():
    direct = _to_python(audit.distributions(_df))
    via_mcp = _srv.get_distributions(_fake_ctx)
    assert direct == via_mcp


# ── Test: Effective revenue ───────────────────────────────────────────────────

def test_effective_revenue_parity():
    direct = _to_python(metrics.effective_revenue(_df))
    via_mcp = _srv.get_effective_revenue(_fake_ctx)
    assert direct == via_mcp


# ── Test: Return rate by group ────────────────────────────────────────────────

def test_return_rate_by_group_parity():
    direct = _to_python(metrics.return_rate_by_group(_df, group_col="TrafficSource"))
    via_mcp = _srv.get_return_rate_by_group(_fake_ctx, group_col="TrafficSource")
    assert direct == via_mcp


# ── Test: Channel revenue summary ─────────────────────────────────────────────

def test_channel_revenue_summary_parity():
    direct = _to_python(metrics.channel_revenue_summary(_df))
    via_mcp = _srv.get_channel_revenue_summary(_fake_ctx)
    assert direct == via_mcp


# ── Test: Repeat customer profile ─────────────────────────────────────────────

def test_repeat_customer_profile_parity():
    direct = _to_python(metrics.repeat_customer_profile(_df))
    via_mcp = _srv.get_repeat_customer_profile(_fake_ctx)
    assert direct == via_mcp


# ── Test: Fisher's Exact ──────────────────────────────────────────────────────

def test_fisher_exact_parity():
    # Paid Ads: 4 returned, 3 completed vs Others: 1 returned, 7 completed
    direct = _to_python(stats.fisher_exact_2x2(4, 3, 1, 7, alternative="greater"))
    via_mcp = _srv.run_fisher_exact(4, 3, 1, 7, alternative="greater", ctx=_fake_ctx)
    # Compare key fields (OR, p-value)
    assert abs(direct["odds_ratio"] - via_mcp["odds_ratio"]) < 1e-4
    assert abs(direct["p_value"] - via_mcp["p_value"]) < 1e-4


# ── Test: Chi-squared (via crosstab) ──────────────────────────────────────────

def test_chi_squared_parity():
    contingency = pd.crosstab(_df["TrafficSource"], _df["OrderStatus"])
    direct = _to_python(stats.chi_squared(contingency.values))
    via_mcp = _srv.run_chi_squared(_fake_ctx, col1="TrafficSource", col2="OrderStatus")
    assert direct == via_mcp


# ── Test: Pearson correlation ─────────────────────────────────────────────────

def test_pearson_correlation_parity():
    direct = _to_python(stats.pearson_correlation(_df["OrderValue_USD"], _df["OrderValue_USD"]))
    via_mcp = _srv.run_pearson_correlation(_fake_ctx, col1="OrderValue_USD", col2="OrderValue_USD")
    assert direct == via_mcp


# ── Test: Spearman correlation ─────────────────────────────────────────────────

def test_spearman_correlation_parity():
    direct = _to_python(stats.spearman_correlation(_df["OrderValue_USD"], _df["OrderValue_USD"]))
    via_mcp = _srv.run_spearman_correlation(_fake_ctx, col1="OrderValue_USD", col2="OrderValue_USD")
    assert direct == via_mcp


# ── Test: Mann-Whitney U ───────────────────────────────────────────────────────

def test_mannwhitney_parity():
    a = _df.loc[_df["TrafficSource"] == "Paid Ads", "OrderValue_USD"]
    b = _df.loc[_df["TrafficSource"] == "Organic Search", "OrderValue_USD"]
    direct = _to_python(stats.mannwhitney_u(a, b, alternative="two-sided"))
    via_mcp = _srv.run_mannwhitney(
        _fake_ctx, value_col="OrderValue_USD", group_col="TrafficSource",
        group1="Paid Ads", group2="Organic Search", alternative="two-sided",
    )
    assert direct == via_mcp


# ── Test: Welch's t-test ───────────────────────────────────────────────────────

def test_welch_t_test_parity():
    a = _df.loc[_df["TrafficSource"] == "Paid Ads", "OrderValue_USD"]
    b = _df.loc[_df["TrafficSource"] == "Organic Search", "OrderValue_USD"]
    direct = _to_python(stats.welch_t_test(a, b, alternative="two-sided"))
    via_mcp = _srv.run_welch_t_test(
        _fake_ctx, value_col="OrderValue_USD", group_col="TrafficSource",
        group1="Paid Ads", group2="Organic Search", alternative="two-sided",
    )
    assert direct == via_mcp

def test_cross_tab_parity():
    direct = _to_python(segments.cross_tab(_df, row_col="TrafficSource", col_col="Category"))
    via_mcp = _srv.get_cross_tab(_fake_ctx, row_col="TrafficSource", col_col="Category")
    assert direct == via_mcp


# ── Test: Confound check ──────────────────────────────────────────────────────

def test_confound_check_parity():
    direct = _to_python(segments.confound_check(
        _df, primary_col="TrafficSource", outcome_col="OrderStatus", potential_confound="Category"
    ))
    via_mcp = _srv.check_confound(
        _fake_ctx, primary_col="TrafficSource", outcome_col="OrderStatus", potential_confound="Category"
    )
    assert direct == via_mcp
