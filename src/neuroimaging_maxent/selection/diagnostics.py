from __future__ import annotations

import numpy as np
import pandas as pd


def jeffreys_standard_error(probability: np.ndarray, n: int) -> np.ndarray:
    probability = np.asarray(probability, dtype=np.float64)
    counts = np.clip(probability * n, 0.0, float(n))
    alpha, beta = counts + 0.5, n - counts + 0.5
    total = alpha + beta
    return np.sqrt(alpha * beta / (total * total * (total + 1.0)))


def moment_fit_summary(
    values: np.ndarray,
    model_first: np.ndarray,
    model_pairwise: np.ndarray,
    *,
    edges: np.ndarray | None = None,
    model_first_se: np.ndarray | None = None,
    model_pairwise_se: np.ndarray | None = None,
) -> list[dict[str, float | int | str]]:
    """Held-out unary/pairwise residuals, standardized by sampling uncertainty."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not len(values):
        raise ValueError("Held-out moment diagnostics need a nonempty matrix")
    first = values.mean(axis=0)
    if edges is None:
        upper = np.triu_indices(values.shape[1], 1)
        pair = (values.T @ values / len(values))[upper]
        model_pairwise = (
            np.asarray(model_pairwise)[upper]
            if np.asarray(model_pairwise).ndim == 2
            else np.asarray(model_pairwise)
        )
    else:
        edges = np.asarray(edges, dtype=np.int64)
        pair = np.mean(values[:, edges[:, 0]] * values[:, edges[:, 1]], axis=0)
    rows: list[dict[str, float | int | str]] = []
    for kind, empirical, model, model_se in (
        ("unary", first, np.asarray(model_first), model_first_se),
        ("pairwise", pair, np.asarray(model_pairwise), model_pairwise_se),
    ):
        empirical = np.asarray(empirical, dtype=np.float64).ravel()
        model = np.asarray(model, dtype=np.float64).ravel()
        if empirical.shape != model.shape:
            raise ValueError(
                f"{kind} empirical/model moment shapes differ: {empirical.shape}/{model.shape}"
            )
        empirical_se = jeffreys_standard_error(empirical, len(values))
        extra = (
            np.zeros_like(empirical_se)
            if model_se is None
            else np.nan_to_num(np.asarray(model_se, dtype=np.float64).ravel(), nan=0.0)
        )
        combined = np.sqrt(empirical_se**2 + extra**2)
        residual = np.divide(
            model - empirical,
            combined,
            out=np.full_like(model, np.nan),
            where=combined > 0,
        )
        rows.append(
            {
                "moment_type": kind,
                "n_observations": int(len(values)),
                "n_moments": int(len(residual)),
                "raw_rmse": float(np.sqrt(np.mean((model - empirical) ** 2))),
                "standardized_rms": float(np.sqrt(np.nanmean(residual**2))),
                "fraction_within_2se": float(np.nanmean(np.abs(residual) <= 2.0)),
            }
        )
    return rows


def aggregate_fit_score(rows: pd.DataFrame) -> float:
    """Equal weight outcome, moment type, class, and fold by averaging group RMS."""
    if rows.empty or not np.all(np.isfinite(rows["standardized_rms"])):
        return float("nan")
    keys = ["outcome", "moment_type", "class_name", "fold"]
    equally_weighted = rows.groupby(keys, observed=True)["standardized_rms"].mean()
    return -float(equally_weighted.mean())
