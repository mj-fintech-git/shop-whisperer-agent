"""audit.py — Data quality audit functions.

Always run these at the top of any analysis.py. Output feeds directly into
Section 1 (Data Quality Audit) of technical_raw.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def load_csv(filepath: str) -> pd.DataFrame:
    """Load a CSV and return a DataFrame. Prints basic confirmation.

    Args:
        filepath: Path to the input CSV file.

    Returns:
        The loaded pandas DataFrame.
    """
    df = pd.read_csv(filepath)
    print(f"[audit] Loaded '{filepath}': {len(df)} rows × {len(df.columns)} cols")
    return df


def schema(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise column dtypes, non-null counts, null counts, and null percentages.

    Maps directly to Section 1.1 of technical_raw.md.

    Args:
        df: The pandas DataFrame to analyse.

    Returns:
        DataFrame with columns: Column, Dtype, Non-Null, Null Count, Null %.
    """
    total = len(df)
    report = pd.DataFrame({
        "Column": df.columns,
        "Dtype": df.dtypes.values,
        "Non-Null": df.notnull().sum().values,
        "Null Count": df.isnull().sum().values,
        "Null %": (df.isnull().sum().values / total * 100).round(1),
    })
    return report


def duplicates(df: pd.DataFrame, action: str = "report") -> dict:
    """Detect and handle duplicate rows in a DataFrame.

    Args:
        df: The pandas DataFrame to check.
        action: Mode of operation. If "drop", duplicate rows are removed from
            the returned DataFrame; if "report", they are kept. Defaults to "report".

    Returns:
        A dictionary with keys:
            count: Number of duplicate rows found.
            pct: Percentage of rows that are duplicates.
            result_df: The cleaned or original DataFrame based on action.
    """
    count = df.duplicated().sum()
    pct = round(count / len(df) * 100, 1)
    result = df.drop_duplicates() if action == "drop" else df
    print(f"[audit] Duplicates: {count} ({pct}%)" + (" — dropped." if action == "drop" else " — kept."))
    return {"count": count, "pct": pct, "result_df": result}


def outliers(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    method: str = "iqr",
    z_threshold: float = 3.0,
    action: str = "flag",
) -> dict:
    """Detect outliers in numeric columns using IQR or Z-score methods.

    Args:
        df: The pandas DataFrame to check.
        columns: Specific numeric columns to check. Defaults to all numeric columns.
        method: Method for outlier detection. One of "iqr" (Interquartile Range)
            or "zscore" (Z-score). Defaults to "iqr".
        z_threshold: Z-score threshold to flag outliers. Only used when method="zscore".
            Defaults to 3.0.
        action: Handing action. If "drop", rows containing outliers are removed.
            If "flag", helper boolean columns ending in "_outlier" are added.
            Defaults to "flag".

    Returns:
        A dictionary with keys:
            summary_df: DataFrame listing flagged counts and thresholds per column.
            flagged_indices: Sorted list of indices containing flagged outliers.
            result_df: The modified DataFrame.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    result_df = df.copy()
    summary_rows = []
    all_flagged = set()

    for col in columns:
        s = df[col].dropna()
        if method == "iqr":
            Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = Q3 - Q1
            lo, hi = Q1 - 1.5 * iqr, Q3 + 1.5 * iqr
            mask = (df[col] < lo) | (df[col] > hi)
            threshold_str = f"[{lo:.2f}, {hi:.2f}]"
        else:
            mu, sigma = s.mean(), s.std()
            mask = (df[col] - mu).abs() / sigma > z_threshold
            threshold_str = f"z > {z_threshold}"

        flagged_idx = df.index[mask].tolist()
        all_flagged.update(flagged_idx)
        result_df[f"{col}_outlier"] = mask

        summary_rows.append({
            "Column": col,
            "Method": method.upper(),
            "Threshold": threshold_str,
            "Flagged Rows": len(flagged_idx),
            "Action": action,
        })
        print(f"[audit] Outliers in '{col}': {len(flagged_idx)} flagged ({method})")

    if action == "drop":
        result_df = result_df[~result_df.index.isin(all_flagged)].copy()
        outlier_cols = [c for c in result_df.columns if c.endswith("_outlier")]
        result_df = result_df.drop(columns=outlier_cols)

    return {
        "summary_df": pd.DataFrame(summary_rows),
        "flagged_indices": sorted(all_flagged),
        "result_df": result_df,
    }


def distributions(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Compute descriptive statistics for numeric columns.

    Maps directly to Section 1.4 of technical_raw.md.

    Args:
        df: The pandas DataFrame to profile.
        columns: Specific columns to describe. Defaults to all numeric columns.

    Returns:
        DataFrame with mean, median, std, min, max, skewness per column.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    rows = []
    for col in columns:
        s = df[col].dropna()
        rows.append({
            "Column": col,
            "Mean": round(s.mean(), 2),
            "Median": round(s.median(), 2),
            "Std": round(s.std(), 2),
            "Min": round(s.min(), 2),
            "Max": round(s.max(), 2),
            "Skewness": round(s.skew(), 4),
        })
    return pd.DataFrame(rows)


def categorical_frequencies(df: pd.DataFrame, columns: list[str] | None = None) -> dict:
    """Return value counts for categorical columns.

    Args:
        df: The pandas DataFrame to analyse.
        columns: Specific columns to evaluate. Defaults to all object/categorical columns.

    Returns:
        A dictionary mapping column name to its pd.Series of value counts.
    """
    if columns is None:
        columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
    return {col: df[col].value_counts() for col in columns}


def full_audit(df: pd.DataFrame) -> dict:
    """Run all audit steps at once and return aggregated outputs.

    Use this for a quick, complete Section 1 of technical_raw.md.

    Args:
        df: The pandas DataFrame to audit.

    Returns:
        A dictionary with keys schema, duplicates, outliers, distributions,
            and categorical_frequencies.
    """
    return {
        "schema": schema(df),
        "duplicates": duplicates(df),
        "outliers": outliers(df),
        "distributions": distributions(df),
        "categorical_frequencies": categorical_frequencies(df),
    }
