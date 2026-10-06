from __future__ import annotations

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr, wasserstein_distance


def activity_count_metrics(
    empirical: np.ndarray, modeled: np.ndarray, dimension: int
) -> dict[str, float]:
    empirical = np.asarray(empirical, dtype=int)
    modeled = np.asarray(modeled, dtype=int)
    support = np.arange(dimension + 1)
    p = np.bincount(empirical, minlength=dimension + 1).astype(float)
    q = np.bincount(modeled, minlength=dimension + 1).astype(float)
    p /= p.sum()
    q /= q.sum()
    return {
        "js_divergence": float(jensenshannon(p, q, base=2.0) ** 2),
        "wasserstein_distance": float(wasserstein_distance(support, support, p, q)),
        "empirical_mean": float(empirical.mean()),
        "model_mean": float(modeled.mean()),
        "empirical_sd": float(empirical.std(ddof=1)) if len(empirical) > 1 else 0.0,
        "model_sd": float(modeled.std(ddof=1)) if len(modeled) > 1 else 0.0,
    }


def marginal_agreement_metrics(
    empirical: np.ndarray,
    modeled: np.ndarray,
    n_heldout: int,
) -> dict[str, float]:
    empirical = np.asarray(empirical, dtype=float)
    modeled = np.asarray(modeled, dtype=float)
    if empirical.shape != modeled.shape or empirical.ndim != 1:
        raise ValueError("Marginal arrays must be aligned vectors")
    error = modeled - empirical
    se = np.sqrt(np.maximum(empirical * (1.0 - empirical), 0.0) / max(n_heldout, 1))
    se = np.maximum(se, 0.5 / max(n_heldout, 1))
    standardized = error / se
    variable = len(empirical) > 1 and np.ptp(empirical) > 0 and np.ptp(modeled) > 0
    pearson = np.corrcoef(empirical, modeled)[0, 1] if variable else np.nan
    spearman = spearmanr(empirical, modeled).statistic if variable else np.nan
    total = np.sum((empirical - empirical.mean()) ** 2)
    return {
        "n_marginals": int(len(empirical)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "max_abs_error": float(np.max(np.abs(error))),
        "pearson_r": float(pearson),
        "spearman_rho": float(spearman),
        "r2": float(1.0 - np.sum(error**2) / total) if total > 0 else np.nan,
        "standardized_rms_error": float(np.sqrt(np.mean(standardized**2))),
        "fraction_within_1se": float(np.mean(np.abs(standardized) <= 1.0)),
        "fraction_within_2se": float(np.mean(np.abs(standardized) <= 2.0)),
    }
