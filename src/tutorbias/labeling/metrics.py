"""Agreement metrics for judge calibration (E1), with bootstrap confidence intervals."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, confusion_matrix


def kappa(a: Sequence, b: Sequence, weights: str | None = None) -> float:
    """Cohen's kappa. weights=None for nominal agreement, 'quadratic' to credit near-misses
    (scaffolding is ordinal: -1 < 0 < +1, so confusing -1 with +1 is worse than -1 with 0)."""
    return float(cohen_kappa_score(a, b, weights=weights, labels=sorted(set(a) | set(b))))


def spearman(a: Sequence, b: Sequence) -> float:
    rho = spearmanr(a, b).statistic
    return float(rho) if rho == rho else float("nan")  # nan if a column is constant


def bootstrap_ci(fn: Callable[..., float], *cols: Sequence, n_boot: int = 2000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float, float]:
    """Point estimate and percentile CI, resampling items with replacement."""
    arrs = [np.asarray(c) for c in cols]
    n = len(arrs[0])
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            v = fn(*[a[idx] for a in arrs])
        except ValueError:
            continue
        if v == v:
            stats.append(v)
    lo, hi = (np.quantile(stats, [alpha / 2, 1 - alpha / 2]) if stats else (float("nan"), float("nan")))
    return float(fn(*arrs)), float(lo), float(hi)


def confusion(a: Sequence, b: Sequence, labels: Sequence) -> list[list[int]]:
    return confusion_matrix(a, b, labels=list(labels)).tolist()


def agreement_report(human_s, judge_s, human_h, judge_h_expected, seed: int = 0, n_boot: int = 2000) -> dict:
    """All E1 numbers for one set of items."""
    k, klo, khi = bootstrap_ci(kappa, human_s, judge_s, n_boot=n_boot, seed=seed)
    kq, kqlo, kqhi = bootstrap_ci(lambda a, b: kappa(a, b, "quadratic"), human_s, judge_s, n_boot=n_boot, seed=seed)
    r, rlo, rhi = bootstrap_ci(spearman, human_h, judge_h_expected, n_boot=n_boot, seed=seed)
    return {
        "n": len(human_s),
        "scaffolding_kappa": k, "scaffolding_kappa_ci": [klo, khi],
        "scaffolding_kappa_quadratic": kq, "scaffolding_kappa_quadratic_ci": [kqlo, kqhi],
        "scaffolding_accuracy": float(np.mean(np.asarray(human_s) == np.asarray(judge_s))),
        "helpfulness_spearman": r, "helpfulness_spearman_ci": [rlo, rhi],
        "confusion_scaffolding": confusion(human_s, judge_s, [-1, 0, 1]),
    }
