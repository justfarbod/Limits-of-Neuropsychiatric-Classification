from __future__ import annotations

import numpy as np


def exact_weighted_auc(
    positive_scores: np.ndarray,
    negative_scores: np.ndarray,
    positive_weights: np.ndarray,
    negative_weights: np.ndarray,
    tie_decimals: int | None = 12,
) -> float:
    """Weighted Mann-Whitney AUC with half credit for ties."""
    positive_scores = np.asarray(positive_scores, dtype=np.float64)
    negative_scores = np.asarray(negative_scores, dtype=np.float64)
    positive_weights = np.asarray(positive_weights, dtype=np.float64)
    negative_weights = np.asarray(negative_weights, dtype=np.float64)
    positive_weights = positive_weights / positive_weights.sum()
    negative_weights = negative_weights / negative_weights.sum()
    if tie_decimals is not None:
        positive_scores = np.round(positive_scores, tie_decimals)
        negative_scores = np.round(negative_scores, tie_decimals)

    negative_order = np.argsort(negative_scores)
    negative_sorted = negative_scores[negative_order]
    negative_weights_sorted = negative_weights[negative_order]
    unique_negative, negative_starts = np.unique(negative_sorted, return_index=True)
    negative_mass = np.add.reduceat(negative_weights_sorted, negative_starts)
    cumulative_negative = np.concatenate(([0.0], np.cumsum(negative_mass)))

    positive_order = np.argsort(positive_scores)
    positive_sorted = positive_scores[positive_order]
    positive_weights_sorted = positive_weights[positive_order]
    unique_positive, positive_starts = np.unique(positive_sorted, return_index=True)
    positive_mass = np.add.reduceat(positive_weights_sorted, positive_starts)
    left = np.searchsorted(unique_negative, unique_positive, side="left")
    right = np.searchsorted(unique_negative, unique_positive, side="right")
    less = cumulative_negative[left]
    equal = cumulative_negative[right] - cumulative_negative[left]
    return float(np.sum(positive_mass * (less + 0.5 * equal)))
