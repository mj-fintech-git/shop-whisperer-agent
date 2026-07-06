"""stats.py — Statistical test wrappers for DS_Agent.

All functions return a structured dict with the exact values needed
for the Statistical Basis field in technical_raw.md findings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def fisher_exact_2x2(
    a: int, b: int, c: int, d: int, alternative: str = "two-sided"
) -> dict:
    """Fisher's Exact Test for a 2×2 contingency table.

    Table layout:
        [[a, b],   <- group 1: [event, no_event]
         [c, d]]   <- group 2: [event, no_event]

    Args:
        a: Group 1 count with event.
        b: Group 1 count without event.
        c: Group 2 count with event.
        d: Group 2 count without event.
        alternative: One of "two-sided", "greater", or "less". Defaults to "two-sided".

    Returns:
        A dictionary with keys:
            test: Name of the test ("Fisher's Exact").
            contingency_table: The raw 2x2 table list of lists.
            odds_ratio: The computed odds ratio.
            p_value: The test p-value.
            ci_95_or: The 95% confidence interval tuple for the odds ratio.
            alternative: The chosen alternative hypothesis.
            significant_at_05: Boolean indicating if p < 0.05.
            interpretation: Spoken interpretation summary.
    """
    table = [[a, b], [c, d]]
    odds_ratio, p_value = scipy_stats.fisher_exact(table, alternative=alternative)

    # Confidence interval for OR (Woolf's method, approx for moderate n)
    try:
        log_or = np.log(odds_ratio)
        se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d)
        ci_lo = np.exp(log_or - 1.96 * se_log_or)
        ci_hi = np.exp(log_or + 1.96 * se_log_or)
        ci = (round(ci_lo, 4), round(ci_hi, 4))
    except (ZeroDivisionError, ValueError):
        ci = (None, None)

    result = {
        "test": "Fisher's Exact",
        "contingency_table": table,
        "odds_ratio": round(odds_ratio, 4),
        "p_value": round(p_value, 4),
        "ci_95_or": ci,
        "alternative": alternative,
        "significant_at_05": p_value < 0.05,
        "interpretation": _fisher_interpret(odds_ratio, p_value, alternative),
    }
    print(f"[stats] Fisher's Exact: OR={result['odds_ratio']}, p={result['p_value']}, sig={result['significant_at_05']}")
    return result


def chi_squared(
    observed: np.ndarray | list,
    correction: bool = True,
) -> dict:
    """Chi-squared test of independence on a contingency table.

    Args:
        observed: 2D array or list of lists representing observed counts.
        correction: Yates' correction for continuity (for 2x2 tables).
            Defaults to True.

    Returns:
        A dictionary with keys:
            test: Name of the test ("Chi-Squared").
            chi2: The test chi-squared statistic.
            p_value: The test p-value.
            degrees_of_freedom: Degrees of freedom.
            expected_frequencies: Calculated expected frequencies.
            cramers_v: Cramér's V effect size coefficient.
            effect_size: Qualitative label for Cramér's V.
            significant_at_05: Boolean indicating if p < 0.05.
    """
    obs = np.array(observed)
    chi2, p_value, dof, expected = scipy_stats.chi2_contingency(obs, correction=correction)
    cv = cramers_v(obs)

    result = {
        "test": "Chi-Squared",
        "chi2": round(chi2, 4),
        "p_value": round(p_value, 4),
        "degrees_of_freedom": dof,
        "expected_frequencies": expected.round(2).tolist(),
        "cramers_v": round(cv, 4),
        "effect_size": _effect_size_label(cv, "cramers_v"),
        "significant_at_05": p_value < 0.05,
    }
    print(f"[stats] Chi-Squared: χ²={result['chi2']}, p={result['p_value']}, V={result['cramers_v']}")
    return result


def cramers_v(contingency_table: np.ndarray) -> float:
    """Compute Cramér's V effect size for a contingency table.

    Args:
        contingency_table: 2D numpy array of observed frequencies.

    Returns:
        Cramér's V effect size (float).
    """
    obs = np.array(contingency_table)
    chi2 = scipy_stats.chi2_contingency(obs, correction=False)[0]
    n = obs.sum()
    k = min(obs.shape) - 1
    return float(np.sqrt(chi2 / (n * k))) if k > 0 and n > 0 else 0.0


def pearson_correlation(x: pd.Series, y: pd.Series) -> dict:
    """Pearson correlation coefficient between two numeric series.

    Args:
        x: First numeric pandas Series.
        y: Second numeric pandas Series.

    Returns:
        A dictionary with keys:
            test: Name of the test ("Pearson Correlation").
            r: Pearson correlation coefficient.
            r_squared: R-squared value.
            p_value: Correlation p-value.
            n: Number of valid matched observations.
            significant_at_05: Boolean indicating if p < 0.05.
            effect_size: Qualitative label for the effect size.
    """
    mask = x.notna() & y.notna()
    r, p = scipy_stats.pearsonr(x[mask], y[mask])
    n = mask.sum()

    result = {
        "test": "Pearson Correlation",
        "r": round(r, 4),
        "r_squared": round(r ** 2, 4),
        "p_value": round(p, 4),
        "n": int(n),
        "significant_at_05": p < 0.05,
        "effect_size": _effect_size_label(abs(r), "r"),
    }
    print(f"[stats] Pearson r={result['r']}, p={result['p_value']}, R²={result['r_squared']}")
    return result


def spearman_correlation(x: pd.Series, y: pd.Series) -> dict:
    """Spearman rank correlation coefficient between two numeric series.

    Args:
        x: First numeric pandas Series.
        y: Second numeric pandas Series.

    Returns:
        A dictionary with keys:
            test: Name of the test ("Spearman Correlation").
            rho: Spearman rank correlation coefficient.
            p_value: Correlation p-value.
            n: Number of valid matched observations.
            significant_at_05: Boolean indicating if p < 0.05.
            effect_size: Qualitative label for the effect size.
    """
    mask = x.notna() & y.notna()
    rho, p = scipy_stats.spearmanr(x[mask], y[mask])
    n = mask.sum()

    result = {
        "test": "Spearman Correlation",
        "rho": round(rho, 4),
        "p_value": round(p, 4),
        "n": int(n),
        "significant_at_05": p < 0.05,
        "effect_size": _effect_size_label(abs(rho), "r"),
    }
    print(f"[stats] Spearman ρ={result['rho']}, p={result['p_value']}")
    return result


def mannwhitney_u(
    group_a: pd.Series, group_b: pd.Series, alternative: str = "two-sided"
) -> dict:
    """Mann-Whitney U test comparing distributions of two groups.

    Args:
        group_a: Numeric pandas Series for Group A.
        group_b: Numeric pandas Series for Group B.
        alternative: One of "two-sided", "greater", or "less". Defaults to "two-sided".

    Returns:
        A dictionary with keys:
            test: Name of the test ("Mann-Whitney U").
            U_statistic: The Mann-Whitney U test statistic.
            p_value: Test p-value.
            n_a: Sample size for Group A.
            n_b: Sample size for Group B.
            effect_size_r: Calculated effect size r.
            effect_size_label: Qualitative label for effect size.
            significant_at_05: Boolean indicating if p < 0.05.
    """
    a = group_a.dropna()
    b = group_b.dropna()
    u_stat, p_value = scipy_stats.mannwhitneyu(a, b, alternative=alternative)
    n1, n2 = len(a), len(b)
    # Effect size r = Z / sqrt(N)
    z = scipy_stats.norm.ppf(1 - p_value / (2 if alternative == "two-sided" else 1))
    r = abs(z) / np.sqrt(n1 + n2)

    result = {
        "test": "Mann-Whitney U",
        "U_statistic": round(u_stat, 4),
        "p_value": round(p_value, 4),
        "n_a": n1,
        "n_b": n2,
        "effect_size_r": round(r, 4),
        "effect_size_label": _effect_size_label(r, "r"),
        "significant_at_05": p_value < 0.05,
    }
    print(f"[stats] Mann-Whitney U={result['U_statistic']}, p={result['p_value']}, r={result['effect_size_r']}")
    return result


def welch_t_test(
    group_a: pd.Series, group_b: pd.Series, alternative: str = "two-sided"
) -> dict:
    """Welch's t-test comparing means of two groups.

    Assumes unequal variances.

    Args:
        group_a: Numeric pandas Series for Group A.
        group_b: Numeric pandas Series for Group B.
        alternative: One of "two-sided", "greater", or "less". Defaults to "two-sided".

    Returns:
        A dictionary with keys:
            test: Name of the test ("Welch's t-test").
            t_statistic: Welch t-statistic.
            p_value: Test p-value.
            cohen_d: Cohen's d effect size.
            effect_size_label: Qualitative label for Cohen's d.
            mean_a: Mean of Group A.
            mean_b: Mean of Group B.
            n_a: Sample size for Group A.
            n_b: Sample size for Group B.
            significant_at_05: Boolean indicating if p < 0.05.
    """
    a = group_a.dropna()
    b = group_b.dropna()
    t_stat, p_value = scipy_stats.ttest_ind(a, b, equal_var=False, alternative=alternative)

    # Cohen's d
    pooled_std = np.sqrt((a.std() ** 2 + b.std() ** 2) / 2)
    d = (a.mean() - b.mean()) / pooled_std if pooled_std > 0 else 0.0

    result = {
        "test": "Welch's t-test",
        "t_statistic": round(t_stat, 4),
        "p_value": round(p_value, 4),
        "cohen_d": round(d, 4),
        "effect_size_label": _effect_size_label(abs(d), "d"),
        "mean_a": round(a.mean(), 4),
        "mean_b": round(b.mean(), 4),
        "n_a": len(a),
        "n_b": len(b),
        "significant_at_05": p_value < 0.05,
    }
    print(f"[stats] Welch t={result['t_statistic']}, p={result['p_value']}, d={result['cohen_d']}")
    return result


def _effect_size_label(value: float, kind: str) -> str:
    """Return a qualitative label for common effect size metrics.

    Args:
        value: The calculated numeric effect size.
        kind: The metric key, one of "r", "d", or "cramers_v".

    Returns:
        Qualitative categorization label ("Large", "Medium", "Small", "Negligible", etc.).
    """
    if kind == "r":
        if value >= 0.5: return "Large"
        if value >= 0.3: return "Medium"
        if value >= 0.1: return "Small"
        return "Negligible"
    if kind == "d":
        if value >= 0.8: return "Large"
        if value >= 0.5: return "Medium"
        if value >= 0.2: return "Small"
        return "Negligible"
    if kind == "cramers_v":
        if value >= 0.35: return "Large"
        if value >= 0.15: return "Medium"
        if value >= 0.05: return "Small"
        return "Negligible"
    return "Unknown"


def _fisher_interpret(odds_ratio: float, p_value: float, alternative: str) -> str:
    """Interpret the odds ratio and statistical significance for Fisher's test.

    Args:
        odds_ratio: The odds ratio from the Fisher's Exact test.
        p_value: The test p-value.
        alternative: The alternative hypothesis type.

    Returns:
        A formatted string explanation of the result.
    """
    sig = "statistically significant" if p_value < 0.05 else "not statistically significant"
    direction = "higher" if odds_ratio > 1 else "lower"
    return (
        f"OR={odds_ratio:.2f} — group 1 has {direction} odds of the event. "
        f"Result is {sig} (p={p_value:.4f}, {alternative})."
    )
