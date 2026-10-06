from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


from ..utils.seeds import stable_seed


@dataclass(frozen=True)
class SelectionResult:
    principle: str
    method: str
    status: str
    winner_key: str | None
    winner: dict[str, Any] | None
    ranking: pd.DataFrame
    reason: str


def _candidate_summary(
    metrics: pd.DataFrame, outcomes: tuple[str, ...]
) -> pd.DataFrame:
    required = {
        "candidate_key",
        "outcome",
        "auc",
        "eligible",
        "latent_dim",
        "parameter_count",
    }
    missing = required - set(metrics)
    if missing:
        raise ValueError(f"Candidate metrics missing columns: {sorted(missing)}")
    rows = []
    for key, frame in metrics.groupby("candidate_key", sort=False):
        auc = frame.set_index("outcome")["auc"]
        row = frame.iloc[0]
        rows.append(
            {
                "candidate_key": key,
                "method": row["method"],
                "latent_dim": int(row["latent_dim"]),
                "requested_treewidth": int(row["requested_treewidth"])
                if pd.notna(row.get("requested_treewidth"))
                else np.nan,
                "actual_treewidth": int(frame["actual_treewidth"].max())
                if "actual_treewidth" in frame
                and frame["actual_treewidth"].notna().any()
                else np.nan,
                "edge_count": int(frame["edge_count"].max())
                if "edge_count" in frame and frame["edge_count"].notna().any()
                else np.nan,
                "parameter_count": int(frame["parameter_count"].max()),
                "eligible": bool(frame["eligible"].all())
                and set(auc.index) == set(outcomes),
                "macro_auc": float(auc.reindex(outcomes).mean()),
                "fit_score": float(frame["fit_score"].iloc[0])
                if "fit_score" in frame
                else float("nan"),
                "fit_se": float(frame["fit_se"].iloc[0])
                if "fit_se" in frame
                else float("nan"),
                **{
                    f"auc_{outcome}": float(auc.get(outcome, np.nan))
                    for outcome in outcomes
                },
            }
        )
    return pd.DataFrame(rows)


def _paired_task_se(
    predictions: pd.DataFrame,
    candidate: str,
    best: str,
    outcome: str,
    repeats: int,
    seed: int,
    fallback: float,
) -> float:
    if predictions.empty or not {"outcome", "candidate_key", "score"}.issubset(
        predictions.columns
    ):
        return fallback
    selected = predictions[
        (predictions["outcome"] == outcome)
        & predictions["candidate_key"].isin((candidate, best))
    ]
    if selected.empty:
        return fallback
    index = [column for column in ("row_id", "fold", "label") if column in selected]
    wide = selected.pivot_table(
        index=index, columns="candidate_key", values="score", aggfunc="first"
    ).dropna()
    if candidate not in wide or best not in wide or "label" not in wide.index.names:
        return fallback
    labels = wide.index.get_level_values("label").to_numpy()
    if np.unique(labels).size != 2:
        return fallback
    rng = np.random.default_rng(seed)
    differences = np.empty(repeats, dtype=np.float64)
    negative, positive = np.flatnonzero(labels == 0), np.flatnonzero(labels == 1)
    for repeat in range(repeats):
        draw = np.concatenate(
            (
                rng.choice(negative, len(negative), replace=True),
                rng.choice(positive, len(positive), replace=True),
            )
        )
        try:
            differences[repeat] = roc_auc_score(
                labels[draw], wide[best].to_numpy()[draw]
            ) - roc_auc_score(labels[draw], wide[candidate].to_numpy()[draw])
        except ValueError:
            differences[repeat] = np.nan
    value = float(np.nanstd(differences, ddof=1))
    return value if np.isfinite(value) and value > 0 else fallback


def _complexity_sort(
    frame: pd.DataFrame, primary: list[str], ascending: list[bool]
) -> pd.DataFrame:
    local = frame.copy()
    local["_treewidth"] = local["actual_treewidth"].fillna(np.inf)
    return local.sort_values(
        [*primary, "_treewidth", "parameter_count", "latent_dim", "candidate_key"],
        ascending=[*ascending, True, True, True, True],
        kind="mergesort",
    ).drop(columns="_treewidth")


def select_candidates(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame | None,
    principle: str,
    *,
    method: str,
    bootstrap_repeats: int = 2000,
    seed: int = 42,
    outcomes: tuple[str, ...] | None = None,
) -> SelectionResult:
    """Apply one prespecified rule to diagnostics-eligible candidates of one method."""
    outcomes = tuple(outcomes or sorted(metrics["outcome"].unique()))
    if principle not in {"balanced-one-se", "macro-auc", "fit-first"}:
        raise ValueError(f"Unknown principle: {principle}")
    summary = _candidate_summary(metrics[metrics["method"] == method], outcomes)
    eligible = summary[summary["eligible"]].copy()
    if eligible.empty:
        summary["rank"] = np.nan
        return SelectionResult(
            principle,
            method,
            "no-winner",
            None,
            None,
            summary,
            "Every candidate failed at least one hard diagnostic; thresholds were not weakened.",
        )

    predictions = (
        pd.DataFrame()
        if predictions is None
        else predictions[predictions["method"] == method]
    )
    for outcome in outcomes:
        column = f"auc_{outcome}"
        best_index = eligible[column].idxmax()
        best_key = str(eligible.loc[best_index, "candidate_key"])
        best_auc = float(eligible.loc[best_index, column])
        regrets, standard_errors, standardized = [], [], []
        for _, row in eligible.iterrows():
            fallback_row = metrics[
                (metrics["candidate_key"] == row.candidate_key)
                & (metrics["outcome"] == outcome)
            ]
            candidate_se = (
                float(fallback_row["auc_se"].iloc[0])
                if "auc_se" in fallback_row and len(fallback_row)
                else 0.0
            )
            best_row = metrics[
                (metrics["candidate_key"] == best_key) & (metrics["outcome"] == outcome)
            ]
            best_se = (
                float(best_row["auc_se"].iloc[0])
                if "auc_se" in best_row and len(best_row)
                else 0.0
            )
            fallback = max(float(np.hypot(candidate_se, best_se)), 1e-12)
            paired = _paired_task_se(
                predictions,
                row.candidate_key,
                best_key,
                outcome,
                bootstrap_repeats,
                stable_seed(method, principle, outcome, row.candidate_key, base=seed),
                fallback,
            )
            regret = best_auc - float(row[column])
            regrets.append(regret)
            standard_errors.append(paired)
            standardized.append(regret / max(paired, 1e-12))
        eligible[f"regret_{outcome}"] = regrets
        eligible[f"paired_se_{outcome}"] = standard_errors
        eligible[f"within_one_se_{outcome}"] = (
            eligible[f"regret_{outcome}"] <= eligible[f"paired_se_{outcome}"] + 1e-15
        )
        eligible[f"standardized_regret_{outcome}"] = standardized
    eligible["worst_standardized_task_regret"] = eligible[
        [f"standardized_regret_{x}" for x in outcomes]
    ].max(axis=1)

    if principle == "macro-auc":
        ranked = _complexity_sort(eligible, ["macro_auc"], [False])
        rationale = "highest equally weighted mean AUC, followed by prespecified complexity tie-breakers"
    elif principle == "balanced-one-se":
        common = eligible[[f"within_one_se_{x}" for x in outcomes]].all(axis=1)
        if common.any():
            eligible["balanced_common_set"] = common
            pool = eligible[common]
            ranked_pool = _complexity_sort(pool, ["fit_score"], [False])
            remainder = _complexity_sort(
                eligible[~common],
                ["worst_standardized_task_regret", "fit_score"],
                [True, False],
            )
            ranked = pd.concat((ranked_pool, remainder), ignore_index=True)
            rationale = (
                "common all-task paired one-SE set; best held-out fit then complexity"
            )
        else:
            eligible["balanced_common_set"] = False
            ranked = _complexity_sort(
                eligible, ["worst_standardized_task_regret", "fit_score"], [True, False]
            )
            rationale = "no common one-SE candidate; minimized worst standardized task regret, then fit and complexity"
    else:
        if not eligible["fit_score"].notna().any():
            ranked = _complexity_sort(eligible, ["macro_auc"], [False])
            rationale = (
                "held-out fit unavailable; explicit deterministic macro-AUC fallback"
            )
        else:
            best_fit = float(eligible["fit_score"].max())
            best_fit_se = float(eligible.loc[eligible["fit_score"].idxmax(), "fit_se"])
            eligible["within_one_se_best_fit"] = eligible[
                "fit_score"
            ] >= best_fit - max(best_fit_se, 0.0)
            pool = eligible[eligible["within_one_se_best_fit"]]
            remainder = eligible[~eligible["within_one_se_best_fit"]]
            ranked = pd.concat(
                (
                    _complexity_sort(pool, ["macro_auc"], [False]),
                    _complexity_sort(
                        remainder, ["fit_score", "macro_auc"], [False, False]
                    ),
                ),
                ignore_index=True,
            )
            rationale = (
                "within one SE of best held-out moment fit; macro AUC then complexity"
            )
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    winner_row = ranked.iloc[0]
    winner = {
        key: (
            None
            if pd.isna(value)
            else value.item()
            if isinstance(value, np.generic)
            else value
        )
        for key, value in winner_row.items()
    }
    return SelectionResult(
        principle,
        method,
        "selected",
        str(winner_row["candidate_key"]),
        winner,
        ranked,
        rationale,
    )
