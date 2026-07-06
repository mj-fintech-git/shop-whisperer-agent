# Finding and AnalysisResult field definitions
# Reference for DS_Agent and SKILL.md authors

## CertaintyLevel

| Value | When to use | Storyteller language |
|-------|-------------|---------------------|
| `Factual` | Pure arithmetic: ratio, sum, count, percentage. No test, no inference. | State flatly. n-ceiling does NOT apply. |
| `High` | p < 0.05 AND effect size is practically meaningful. n >= 20 required. | State plainly (still subject to n < 20 override if applicable). |
| `Medium` | Directional but not statistically robust. Applies when: p > 0.05 but large effect, OR p < 0.05 but n < 20. | "looks like," "seems like," "probably" |
| `Low` | Weak signal, high variance, or n < 5 for the group in question. | "there's a hint of," "worth a look but," "might be" |
| `Inconclusive` | Data cannot answer this question: missing variable, observational design limit, insufficient n. | "we genuinely couldn't find an answer — here's why" |

## Finding fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `metric_name` | str | Yes | Short identifier, e.g. `return_rate_paid_ads` |
| `measured_value` | Any | Yes | Computed result — float, dict, list[dict], str |
| `certainty` | CertaintyLevel | Yes | One of the 5 values above |
| `sample_size` | int | Yes | n observations this finding is based on |
| `comparison_group` | str or null | No | What the value is compared against |
| `statistical_basis` | dict or null | No | Only when certainty != Factual. Keys: test, p_value, odds_ratio, ci_95, effect_size |
| `supporting_data` | dict | No | Raw numbers Storyteller needs to narrate. All values must be JSON primitives. |

## AnalysisResult fields

| Field | Type | Description |
|-------|------|-------------|
| `business_problem` | str | Verbatim user-provided problem statement |
| `dataset_info` | dict | filename, n_rows, n_cols, columns, analysis_timestamp |
| `domain_context` | str | Web searches performed and key takeaways, or "None" |
| `findings` | list[Finding] | All findings, ordered by impact (highest certainty / largest effect first) |
| `data_gaps` | list[DataGap] | Missing data that would have enabled stronger conclusions |
| `next_steps` | list[str] | Concrete actions, each tracing to a specific finding |

## DataGap fields

| Field | Type | Description |
|-------|------|-------------|
| `missing_data` | str | What data is absent |
| `why_it_matters` | str | What analysis it blocked |
| `analysis_enabled` | str | What could have been concluded with it |
