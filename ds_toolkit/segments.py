"""segments.py — Segmentation and cross-tabulation helpers.

Use these for any groupby analysis, customer/product/channel segmentation,
and cross-tabs that check for confounding variables.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cross_tab(
    df: pd.DataFrame,
    row_col: str,
    col_col: str,
    value_col: str | None = None,
    aggfunc: str = "count",
    normalize: bool = False,
) -> pd.DataFrame:
    """Create a cross-tabulation between two categorical columns.

    Args:
        df: The input pandas DataFrame.
        row_col: Column name to use for the cross-tab rows.
        col_col: Column name to use for the cross-tab columns.
        value_col: Optional numeric column name to aggregate. If not specified,
            frequencies/counts are returned. Defaults to None.
        aggfunc: Aggregation function to apply to value_col if specified
            ("count", "sum", "mean"). Defaults to "count".
        normalize: If True, return percentages of total instead of counts.
            Defaults to False.

    Returns:
        The resulting cross-tabulation DataFrame.
    """
    if value_col:
        ct = pd.crosstab(df[row_col], df[col_col], values=df[value_col], aggfunc=aggfunc)
    else:
        ct = pd.crosstab(df[row_col], df[col_col])

    if normalize:
        ct = ct.div(ct.sum().sum()) * 100
        ct = ct.round(1)

    return ct


def segment_profile(
    df: pd.DataFrame,
    segment_col: str,
    numeric_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
) -> dict:
    """Profile a segmentation variable with distributions and frequency tables.

    For each segment, compute summary statistics for numeric variables and
    crosstabs/frequencies for categorical variables.

    Args:
        df: The input pandas DataFrame.
        segment_col: The column defining segments (e.g., "TrafficSource").
        numeric_cols: Numeric columns to describe per segment. Defaults to all
            numeric columns in the DataFrame.
        categorical_cols: Categorical columns to frequency-count per segment.
            Defaults to all object/categorical columns in the DataFrame except
            segment_col itself.

    Returns:
        A dictionary with keys:
            numeric: Dictionary mapping each numeric column name to its summary stats DataFrame.
            categorical: Dictionary mapping each categorical column name to its crosstab DataFrame.
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if categorical_cols is None:
        categorical_cols = [
            c for c in df.select_dtypes(include=["object", "category"]).columns
            if c != segment_col
        ]

    numeric_result = {}
    for col in numeric_cols:
        numeric_result[col] = (
            df.groupby(segment_col)[col]
            .agg(["count", "mean", "median", "std", "min", "max"])
            .round(2)
        )

    cat_result = {}
    for col in categorical_cols:
        cat_result[col] = pd.crosstab(df[segment_col], df[col])

    return {"numeric": numeric_result, "categorical": cat_result}


def top_n_by_metric(
    df: pd.DataFrame,
    group_col: str,
    metric_col: str,
    n: int = 5,
    aggfunc: str = "sum",
    ascending: bool = False,
) -> pd.DataFrame:
    """Return top N groups ranked by an aggregated metric.

    Args:
        df: The input pandas DataFrame.
        group_col: Column name to group by.
        metric_col: Column name to rank on.
        n: Number of top groups to return. Defaults to 5.
        aggfunc: Aggregation function ("sum", "mean", "count", "max").
            Defaults to "sum".
        ascending: If True, return the bottom N groups instead of top N.
            Defaults to False.

    Returns:
        DataFrame containing the grouped column and the aggregated metric,
        sorted accordingly.
    """
    grp = df.groupby(group_col)[metric_col].agg(aggfunc).reset_index()
    grp.columns = [group_col, f"{aggfunc}_{metric_col}"]
    return grp.sort_values(f"{aggfunc}_{metric_col}", ascending=ascending).head(n).reset_index(drop=True)


def cohort_return_analysis(
    df: pd.DataFrame,
    cohort_col: str,
    status_col: str = "OrderStatus",
    value_col: str = "OrderValue_USD",
    returned_value: str = "Returned",
) -> pd.DataFrame:
    """Compare cohorts on volume, return rates, and effective revenue.

    For each cohort, calculate total orders, gross revenue, completed orders,
    effective revenue, returned value, and return rates.

    Args:
        df: The input pandas DataFrame.
        cohort_col: Column name specifying the cohorts.
        status_col: Column name containing order status. Defaults to "OrderStatus".
        value_col: Column name containing monetary order value. Defaults to "OrderValue_USD".
        returned_value: Value indicating that an order was returned. Defaults to "Returned".

    Returns:
        DataFrame containing cohort metrics, sorted by return rate descending.
    """
    completed_mask = df[status_col] != returned_value
    returned_mask = df[status_col] == returned_value

    base = df.groupby(cohort_col).agg(
        total_orders=(status_col, "count"),
        gross_revenue=(value_col, "sum"),
    ).reset_index()

    comp = df[completed_mask].groupby(cohort_col).agg(
        effective_revenue=(value_col, "sum"),
        completed_orders=(status_col, "count"),
    ).reset_index()

    ret = df[returned_mask].groupby(cohort_col).agg(
        returned_orders=(status_col, "count"),
        returned_value=(value_col, "sum"),
    ).reset_index()

    result = base.merge(comp, on=cohort_col, how="left").merge(ret, on=cohort_col, how="left").fillna(0)
    result["return_rate"] = (result["returned_orders"] / result["total_orders"]).round(4)
    result["effective_pct"] = (result["effective_revenue"] / result["gross_revenue"]).round(4)
    result["gross_revenue"] = result["gross_revenue"].round(2)
    result["effective_revenue"] = result["effective_revenue"].round(2)
    result["returned_value"] = result["returned_value"].round(2)

    return result.sort_values("return_rate", ascending=False).reset_index(drop=True)


def confound_check(
    df: pd.DataFrame,
    primary_col: str,
    outcome_col: str,
    potential_confound: str,
) -> pd.DataFrame:
    """Check for confounders by crosstabulating primary vs potential confounder.

    Args:
        df: The input pandas DataFrame.
        primary_col: The primary independent variable column.
        outcome_col: The dependent outcome variable column.
        potential_confound: The potential confounding variable column.

    Returns:
        DataFrame containing the crosstabulation of primary_col vs potential_confound.
    """
    print(f"[segments] Confound check: {primary_col} × {potential_confound}")
    ct = pd.crosstab(df[primary_col], df[potential_confound])
    print(ct)
    return ct
