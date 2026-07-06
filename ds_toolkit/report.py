"""report.py — Markdown formatters for technical_raw.md schema compliance.

This module provides helper functions to format data science metrics, schemas, and
findings as markdown sections and GitHub-style tables.
"""

from __future__ import annotations

from datetime import datetime, timezone
import pandas as pd


def df_to_markdown(df: pd.DataFrame, index: bool = False) -> str:
    """Convert a DataFrame to a GitHub-style markdown table string.

    Args:
        df: The pandas DataFrame to convert.
        index: Whether to include the DataFrame index in the table.

    Returns:
        The markdown table representation of the DataFrame.
    """
    return df.to_markdown(index=index)


def schema_section(schema_df: pd.DataFrame) -> str:
    """Render the Schema section (Section 1.1) of the report.

    Args:
        schema_df: Summary DataFrame containing column names and dtypes.

    Returns:
        A formatted markdown string for the schema section.
    """
    return "### 1.1 Schema\n\n" + df_to_markdown(schema_df) + "\n"


def duplicates_section(dup_dict: dict) -> str:
    """Render the Duplicates section (Section 1.2) of the report.

    Args:
        dup_dict: Dictionary containing duplicate count and actions.

    Returns:
        A formatted markdown string for the duplicates section.
    """
    count = dup_dict["count"]
    action = "Dropped." if dup_dict.get("action") == "drop" else "No action taken."
    return f"### 1.2 Duplicates\n\n{count} duplicate row(s) found. {action}\n"


def outliers_section(outlier_dict: dict) -> str:
    """Render the Outliers section (Section 1.3) of the report.

    Args:
        outlier_dict: Dictionary containing outlier summary DataFrames.

    Returns:
        A formatted markdown string for the outliers section.
    """
    summary_df = outlier_dict["summary_df"]
    return "### 1.3 Outliers\n\n" + df_to_markdown(summary_df) + "\n"


def distributions_section(dist_df: pd.DataFrame) -> str:
    """Render the Distributions section (Section 1.4) of the report.

    Args:
        dist_df: DataFrame summarizing numeric column distributions.

    Returns:
        A formatted markdown string for the distributions section.
    """
    return "### 1.4 Distributions\n\n" + df_to_markdown(dist_df) + "\n"


def header(
    business_problem: str,
    dataset_name: str,
    n_rows: int,
    n_cols: int,
    domain_context: str = "None",
) -> str:
    """Render the technical_raw.md header block with metadata.

    Args:
        business_problem: The original business problem statement.
        dataset_name: Name or path of the analysed dataset.
        n_rows: Number of rows in the dataset.
        n_cols: Number of columns in the dataset.
        domain_context: Summary of domain searches or external context.

    Returns:
        A formatted markdown header string.
    """
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return (
        f"# Technical Analysis Report\n\n"
        f"**Business Problem**: {business_problem}\n"
        f"**Dataset**: {dataset_name} | {n_rows} rows | {n_cols} columns\n"
        f"**Analysis Timestamp**: {ts}\n"
        f"**Domain Context Used**: {domain_context}\n"
    )


def finding(
    n: int,
    title: str,
    confidence: str,
    evidence: str,
    statistical_basis: str,
    business_implication: str,
) -> str:
    """Render a single structured Finding block.

    Args:
        n: The sequence index of the finding.
        title: Short title summarizing the finding.
        confidence: Certainty classification (High, Medium, Low, Inconclusive).
        evidence: Textual summary of numbers and evidence.
        statistical_basis: Statistical test details and parameters.
        business_implication: What this finding means for business stakeholders.

    Returns:
        A formatted markdown string for a single finding.

    Raises:
        ValueError: If confidence is not one of the allowed categories.
    """
    valid = {"High", "Medium", "Low", "Inconclusive"}
    if confidence not in valid:
        raise ValueError(f"confidence must be one of {valid}, got '{confidence}'")

    return (
        f"### Finding {n}: {title}\n"
        f"- **Confidence**: {confidence}\n"
        f"- **Evidence**: {evidence}\n"
        f"- **Statistical Basis**: {statistical_basis}\n"
        f"- **Business Implication**: {business_implication}\n"
    )


def data_gaps_section(gaps: list[dict]) -> str:
    """Render Section 5 (Data Gaps) of the report.

    Args:
        gaps: List of dictionaries with keys missing_data, why_it_matters,
            and analysis_enabled.

    Returns:
        A formatted markdown string detailing data gaps.
    """
    rows = []
    for g in gaps:
        rows.append({
            "Missing Data": g.get("missing_data", ""),
            "Why It Matters": g.get("why_it_matters", ""),
            "Analysis It Would Have Enabled": g.get("analysis_enabled", ""),
        })
    df = pd.DataFrame(rows)
    return "## 5. Data Gaps — What Would Have Helped\n\n" + df_to_markdown(df) + "\n"


def cannot_tell_section(items: list[str]) -> str:
    """Render Section 4 (What the Data Cannot Tell Us) of the report.

    Args:
        items: Bullet points describing unanswered questions.

    Returns:
        A formatted markdown string.
    """
    lines = "\n".join(f"- {item}" for item in items)
    return f"## 4. What the Data Cannot Tell Us\n\n{lines}\n"


def next_steps_section(steps: list[str]) -> str:
    """Render Section 6 (Recommended Next Steps) of the report.

    Args:
        steps: Recommended business/analytic actions.

    Returns:
        A formatted markdown string listing recommendations.
    """
    lines = "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))
    return f"## 6. Recommended Next Steps\n\n{lines}\n"


def methodology_section(techniques: list[dict]) -> str:
    """Render Section 2 (Methodology) of the report.

    Args:
        techniques: List of dictionaries with keys technique, justification,
            assumptions, and limitations.

    Returns:
        A formatted markdown string including the methodology table.
    """
    df = pd.DataFrame(techniques)
    df.columns = ["Technique", "Justification", "Assumptions", "Limitations"]
    return "## 2. Methodology\n\n" + df_to_markdown(df) + "\n"


def save_report(sections: list[str], filepath: str) -> None:
    """Join all sections and write the technical report to a file.

    Args:
        sections: List of individual markdown sections.
        filepath: Absolute destination path for the report file.
    """
    content = "\n\n---\n\n".join(sections)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[report] Saved: {filepath}")
