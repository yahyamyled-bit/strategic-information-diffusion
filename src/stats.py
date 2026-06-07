"""
Statistical helpers shared across stages (SciPy/NumPy based).

Stage 2 uses bootstrap_ci for baseline reference means; Stages 4 and 6 reuse
this module for KS tests, ANOVA, Bonferroni t-tests, Spearman/Kendall, etc.
"""

from __future__ import annotations

import numpy as np


def bootstrap_ci(data, n_resamples: int = 10_000, ci: float = 0.95,
                 statistic=np.mean, seed: int = 0) -> dict:
    """Nonparametric bootstrap confidence interval for a statistic.

    Returns a dict with the point estimate and the lower/upper percentile CI
    bounds over `n_resamples` resamples with replacement (methodology default
    n_resamples = 10,000).
    """
    data = np.asarray(data, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = data.size
    if n == 0:
        return {"estimate": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "n": 0, "n_resamples": n_resamples}
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot = statistic(data[idx], axis=1)
    alpha = (1.0 - ci) / 2.0
    low, high = np.quantile(boot, [alpha, 1.0 - alpha])
    return {
        "estimate": float(statistic(data)),
        "ci_low": float(low),
        "ci_high": float(high),
        "n": int(n),
        "n_resamples": int(n_resamples),
        "ci_level": ci,
    }
