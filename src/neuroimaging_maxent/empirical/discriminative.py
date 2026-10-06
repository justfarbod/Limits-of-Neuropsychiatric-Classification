from __future__ import annotations

from typing import Any
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC


def make_fixed_pipeline(n_features: int, c: float) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer()),
            ("scaler", RobustScaler()),
            ("svc", SVC(C=float(c), kernel="rbf", gamma=1.0 / n_features, max_iter=-1)),
        ]
    )


def _fit_fold(
    values: np.ndarray,
    labels: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    seed: int,
    fold: int,
    c: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    estimator = make_fixed_pipeline(values.shape[1], c)
    estimator.fit(values[train], labels[train])
    scores = np.asarray(estimator.decision_function(values[test]), dtype=np.float64)
    row = {
        "outer_seed": int(seed),
        "fold": int(fold),
        "auc": float(roc_auc_score(labels[test], scores)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_train_negative": int(np.sum(labels[train] == 0)),
        "n_train_positive": int(np.sum(labels[train] == 1)),
        "n_test_negative": int(np.sum(labels[test] == 0)),
        "n_test_positive": int(np.sum(labels[test] == 1)),
        "gamma": float(1.0 / values.shape[1]),
        "C": float(c),
    }
    return row, test, scores


def bootstrap_repeated_fold_auc(
    labels: np.ndarray,
    score_matrix: np.ndarray,
    fold_matrix: np.ndarray,
    n_repeats: int,
    seed: int,
) -> np.ndarray:
    """Bootstrap subjects once per draw and average AUC across held-out folds.

    Counts are sampled from the class-specific multinomial distribution, which
    is equivalent to resampling subject indices with replacement. Fixed score
    orderings let weighted fold AUCs be evaluated in vectorized batches.
    """
    labels = np.asarray(labels, dtype=np.int8)
    negative = np.flatnonzero(labels == 0)
    positive = np.flatnonzero(labels == 1)
    rng = np.random.default_rng(seed)
    output = np.empty(n_repeats, dtype=np.float64)
    fold_specs: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for seed_index in range(score_matrix.shape[0]):
        for fold in np.unique(fold_matrix[seed_index]):
            positions = np.flatnonzero(fold_matrix[seed_index] == fold)
            order = np.argsort(score_matrix[seed_index, positions], kind="mergesort")
            ordered_positions = positions[order]
            fold_specs.append(
                (
                    ordered_positions,
                    labels[ordered_positions],
                    score_matrix[seed_index, ordered_positions],
                )
            )

    batch_size = min(256, n_repeats)
    probabilities_negative = np.full(len(negative), 1.0 / len(negative))
    probabilities_positive = np.full(len(positive), 1.0 / len(positive))
    for start in range(0, n_repeats, batch_size):
        stop = min(start + batch_size, n_repeats)
        size = stop - start
        counts = np.zeros((size, len(labels)), dtype=np.int32)
        counts[:, negative] = rng.multinomial(
            len(negative), probabilities_negative, size=size
        )
        counts[:, positive] = rng.multinomial(
            len(positive), probabilities_positive, size=size
        )
        fold_aucs = np.empty((size, len(fold_specs)), dtype=np.float64)
        for spec_index, (positions, fold_labels, fold_scores) in enumerate(fold_specs):
            weights = counts[:, positions]
            negative_weights = weights * (fold_labels == 0)
            positive_weights = weights * (fold_labels == 1)
            negative_total = negative_weights.sum(axis=1)
            positive_total = positive_weights.sum(axis=1)

            if np.all(np.diff(fold_scores) != 0):
                negative_at_or_below = np.cumsum(negative_weights, axis=1)
                numerator = np.sum(positive_weights * negative_at_or_below, axis=1)
            else:
                numerator = np.zeros(size, dtype=np.float64)
                negative_before = np.zeros(size, dtype=np.float64)
                boundaries = np.r_[
                    0, np.flatnonzero(np.diff(fold_scores) != 0) + 1, len(fold_scores)
                ]
                for left, right in zip(boundaries[:-1], boundaries[1:]):
                    group_negative = negative_weights[:, left:right].sum(axis=1)
                    group_positive = positive_weights[:, left:right].sum(axis=1)
                    numerator += group_positive * (
                        negative_before + 0.5 * group_negative
                    )
                    negative_before += group_negative
            denominator = negative_total * positive_total
            fold_aucs[:, spec_index] = np.divide(
                numerator, denominator, out=np.full(size, np.nan), where=denominator > 0
            )
        valid_counts = np.sum(np.isfinite(fold_aucs), axis=1)
        if np.any(valid_counts == 0):
            raise RuntimeError(
                "Bootstrap sample contained no valid two-class test fold"
            )
        output[start:stop] = np.nansum(fold_aucs, axis=1) / valid_counts
    return output
