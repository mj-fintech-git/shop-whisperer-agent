"""metrics.py — E-commerce and general business metric computations.

All functions are dataset-agnostic — they take column name strings as args
so DS_Agent can call them with whatever columns exist in the CSV.
"""

from __future__ import annotations

import pandas as pd


def return_rate_by_group(
    df: pd.DataFrame,
    group_col: str,
    status_col: str = "OrderStatus",
    returned_value: str = "Returned",
) -> pd.DataFrame:
    """Compute return rate (% of orders returned) per group.

    Args:
        df: The input pandas DataFrame.
        group_col: Column name to group by (e.g., "TrafficSource", "Category").
        status_col: Column name containing order status. Defaults to "OrderStatus".
        returned_value: Value indicating a returned order. Defaults to "Returned".

    Returns:
        DataFrame with columns: [group_col, total_orders, returned, completed,
        return_rate, return_rate_pct] sorted by return_rate descending.
    """
    grp = df.groupby(group_col).agg(
        total_orders=(status_col, "count"),
        returned=(status_col, lambda x: (x == returned_value).sum()),
        completed=(status_col, lambda x: (x != returned_value).sum()),
    ).reset_index()
    grp["return_rate"] = (grp["returned"] / grp["total_orders"]).round(4)
    grp["return_rate_pct"] = (grp["return_rate"] * 100).round(1)
    return grp.sort_values("return_rate", ascending=False).reset_index(drop=True)


def effective_revenue(
    df: pd.DataFrame,
    value_col: str = "OrderValue_USD",
    status_col: str = "OrderStatus",
    returned_value: str = "Returned",
) -> dict:
    """Compute gross revenue, returned value, and effective (net) revenue.

    Args:
        df: The input pandas DataFrame.
        value_col: Column name containing monetary order value. Defaults to "OrderValue_USD".
        status_col: Column name containing order status. Defaults to "OrderStatus".
        returned_value: Value indicating a returned order. Defaults to "Returned".

    Returns:
        A dictionary with keys:
            gross_revenue: Gross total revenue.
            returned_value: Total value of returned orders.
            effective_revenue: Revenue after subtracting returns.
            retention_rate: Effective revenue / Gross revenue ratio.
            loss_rate: Returned value / Gross revenue ratio.
            loss_pct: Percentage loss due to returns.
    """
    gross = df[value_col].sum()
    ret = df.loc[df[status_col] == returned_value, value_col].sum()
    eff = gross - ret

    result = {
        "gross_revenue": round(gross, 2),
        "returned_value": round(ret, 2),
        "effective_revenue": round(eff, 2),
        "retention_rate": round(eff / gross, 4) if gross > 0 else 0.0,
        "loss_rate": round(ret / gross, 4) if gross > 0 else 0.0,
        "loss_pct": round(ret / gross * 100, 1) if gross > 0 else 0.0,
    }
    print(
        f"[metrics] Revenue — Gross: ${result['gross_revenue']}, "
        f"Returns: ${result['returned_value']} ({result['loss_pct']}%), "
        f"Effective: ${result['effective_revenue']}"
    )
    return result


def channel_revenue_summary(
    df: pd.DataFrame,
    channel_col: str = "TrafficSource",
    value_col: str = "OrderValue_USD",
    status_col: str = "OrderStatus",
    returned_value: str = "Returned",
) -> pd.DataFrame:
    """Full revenue breakdown by acquisition/traffic channel.

    Args:
        df: The input pandas DataFrame.
        channel_col: Column name representing traffic/acquisition source. Defaults to "TrafficSource".
        value_col: Column name containing monetary order value. Defaults to "OrderValue_USD".
        status_col: Column name containing order status. Defaults to "OrderStatus".
        returned_value: Value indicating a returned order. Defaults to "Returned".

    Returns:
        DataFrame with columns: channel_col, order_count, gross_revenue, mean_aov,
        effective_revenue, completed_orders, returned_value, returned_orders,
        return_rate, return_rate_pct, effective_pct, gross_revenue_share.
    """
    completed_mask = df[status_col] != returned_value
    returned_mask = df[status_col] == returned_value

    grp = df.groupby(channel_col).agg(
        order_count=(value_col, "count"),
        gross_revenue=(value_col, "sum"),
        mean_aov=(value_col, "mean"),
    ).reset_index()

    comp = df[completed_mask].groupby(channel_col).agg(
        effective_revenue=(value_col, "sum"),
        completed_orders=(value_col, "count"),
    ).reset_index()

    ret = df[returned_mask].groupby(channel_col).agg(
        returned_value=(value_col, "sum"),
        returned_orders=(value_col, "count"),
    ).reset_index()

    result = grp.merge(comp, on=channel_col, how="left").merge(ret, on=channel_col, how="left").fillna(0)
    result["return_rate"] = (result["returned_orders"] / result["order_count"]).round(4)
    result["return_rate_pct"] = (result["return_rate"] * 100).round(1)
    result["effective_pct"] = (result["effective_revenue"] / result["gross_revenue"]).round(4)
    total_gross = result["gross_revenue"].sum()
    result["gross_revenue_share"] = (result["gross_revenue"] / total_gross).round(4)

    result["mean_aov"] = result["mean_aov"].round(2)
    result["gross_revenue"] = result["gross_revenue"].round(2)
    result["effective_revenue"] = result["effective_revenue"].round(2)
    result["returned_value"] = result["returned_value"].round(2)

    return result.sort_values("gross_revenue", ascending=False).reset_index(drop=True)


def aov_by_group(
    df: pd.DataFrame,
    group_col: str,
    value_col: str = "OrderValue_USD",
    status_col: str | None = None,
    completed_only: bool = False,
    returned_value: str = "Returned",
) -> pd.DataFrame:
    """Average Order Value (AOV) by group, optionally restricted to completed orders.

    Args:
        df: The input pandas DataFrame.
        group_col: Column name to group by.
        value_col: Column name containing monetary order value. Defaults to "OrderValue_USD".
        status_col: Column name containing order status. Defaults to None.
        completed_only: If True, filters out returned orders. Only applied if
            status_col is also provided. Defaults to False.
        returned_value: Value indicating a returned order. Defaults to "Returned".

    Returns:
        DataFrame with columns: [group_col, order_count, mean_aov, median_aov, std_aov]
        sorted by mean_aov descending.
    """
    data = df.copy()
    if completed_only and status_col:
        data = data[data[status_col] != returned_value]

    grp = data.groupby(group_col)[value_col].agg(
        order_count="count",
        mean_aov="mean",
        median_aov="median",
        std_aov="std",
    ).reset_index()
    grp[["mean_aov", "median_aov", "std_aov"]] = grp[["mean_aov", "median_aov", "std_aov"]].round(2)
    return grp.sort_values("mean_aov", ascending=False).reset_index(drop=True)


def revenue_concentration(
    df: pd.DataFrame,
    group_col: str,
    value_col: str = "OrderValue_USD",
    top_n: int = 3,
) -> dict:
    """Compute what % of total revenue comes from the top N groups.

    Useful for identifying customer, product, or SKU concentration risks.

    Args:
        df: The input pandas DataFrame.
        group_col: Column name representing the entities (e.g., "CustomerID", "SKU").
        value_col: Column name containing monetary order value. Defaults to "OrderValue_USD".
        top_n: Number of top groups to calculate concentration for. Defaults to 3.

    Returns:
        A dictionary with keys:
            group_col: Group column name.
            top_n: Evaluated top N value.
            top_groups: DataFrame detailing the revenue of the top N groups.
            top_n_revenue: Combined revenue from the top N groups.
            total_revenue: Overall dataset revenue.
            concentration_pct: Percentage share of total revenue from top N groups.
    """
    total = df[value_col].sum()
    grp = df.groupby(group_col)[value_col].sum().sort_values(ascending=False)
    top = grp.head(top_n)
    share = (top.sum() / total * 100).round(1)
    return {
        "group_col": group_col,
        "top_n": top_n,
        "top_groups": top.reset_index().rename(columns={value_col: "revenue"}),
        "top_n_revenue": round(top.sum(), 2),
        "total_revenue": round(total, 2),
        "concentration_pct": share,
    }


def repeat_customer_profile(
    df: pd.DataFrame,
    customer_col: str = "CustomerID",
    order_col: str = "OrderID",
    value_col: str = "OrderValue_USD",
    status_col: str = "OrderStatus",
    channel_col: str | None = "TrafficSource",
    returned_value: str = "Returned",
) -> pd.DataFrame:
    """Profile repeat customers having more than one order.

    Args:
        df: The input pandas DataFrame.
        customer_col: Column name containing customer identifiers. Defaults to "CustomerID".
        order_col: Column name containing order identifiers. Defaults to "OrderID".
        value_col: Column name containing monetary order value. Defaults to "OrderValue_USD".
        status_col: Column name containing order status. Defaults to "OrderStatus".
        channel_col: Optional column name for acquisition/traffic sources. Defaults to "TrafficSource".
        returned_value: Value indicating a returned order. Defaults to "Returned".

    Returns:
        DataFrame detailing repeat customer profiles, sorted by order count descending.
    """
    agg_dict = {
        "order_count": (order_col, "count"),
        "total_value": (value_col, "sum"),
        "returned_orders": (status_col, lambda x: (x == returned_value).sum()),
    }
    if channel_col:
        agg_dict["channels_used"] = (channel_col, lambda x: list(x.unique()))

    grp = df.groupby(customer_col).agg(**agg_dict).reset_index()
    grp["return_rate"] = (grp["returned_orders"] / grp["order_count"]).round(4)
    grp["total_value"] = grp["total_value"].round(2)

    repeat = grp[grp["order_count"] > 1].sort_values("order_count", ascending=False).reset_index(drop=True)
    print(f"[metrics] Repeat customers: {len(repeat)} of {grp[customer_col].nunique()} unique customers")
    return repeat
