"""
ds_toolkit — Reusable Data Science toolkit for DS_Agent.

Modules:
    audit    — Data quality checks (schema, nulls, duplicates, outliers, distributions)
    stats    — Statistical tests (Fisher's exact, chi-squared, correlation, t-test)
    metrics  — Business metrics (return rates, effective revenue, AOV, channel comparison)
    segments — Segmentation helpers (groupby, repeat customers, cross-tabs)
    report   — Markdown formatters for technical_raw.md schema compliance
"""

from . import audit, stats, metrics, segments, report
