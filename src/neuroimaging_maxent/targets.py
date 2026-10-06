from dataclasses import dataclass, field
from itertools import combinations, product
import re

import numpy as np
import pandas as pd


@dataclass
class Target:
    key: str
    indices: np.ndarray
    labels: np.ndarray
    metadata: dict = field(default_factory=dict)


def _target(frame, key, mask, positive, metadata=None):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", key):
        raise ValueError("Target keys must be filename-safe identifiers.")
    positions = np.flatnonzero(np.asarray(mask, dtype=bool))
    labels = np.asarray(positive, dtype=bool)[positions].astype(np.int8)
    if len(np.unique(labels)) != 2:
        raise ValueError("Each target must contain both classes.")
    return Target(key, positions, labels, metadata or {})


def build_targets(frame: pd.DataFrame, definitions: list[dict]) -> list[Target]:
    targets = []
    for definition in definitions:
        kind, key, column = definition["kind"], definition["key"], definition["column"]
        if column not in frame:
            raise ValueError("A configured target column is missing.")
        values = frame[column]
        if kind in {"binary", "groups"}:
            negative, positive = definition["negative"], definition["positive"]
            if negative == positive:
                raise ValueError("Class definitions must differ.")
            targets.append(
                _target(
                    frame, key, values.isin([negative, positive]), values.eq(positive)
                )
            )
        elif kind == "threshold":
            numeric = pd.to_numeric(values, errors="coerce")
            if (
                (values.notna() & numeric.isna()).any()
                or not np.isfinite(numeric.dropna()).all()
                or not np.isfinite(definition["threshold"])
            ):
                raise ValueError("Threshold covariates must be numeric.")
            targets.append(
                _target(
                    frame, key, numeric.notna(), numeric.ge(definition["threshold"])
                )
            )
        elif kind == "pairwise_groups":
            groups = definition["groups"]
            if len(set(groups)) != len(groups) or len(groups) < 2:
                raise ValueError(
                    "Pairwise groups must contain at least two distinct values."
                )
            for i, j in combinations(range(len(groups)), 2):
                targets.append(
                    _target(
                        frame,
                        f"{key}_{i}_{j}",
                        values.isin([groups[i], groups[j]]),
                        values.eq(groups[j]),
                    )
                )
        elif kind == "score_strata":
            group_column = definition["group_column"]
            if group_column not in frame:
                raise ValueError("The configured stratum group column is missing.")
            numeric = pd.to_numeric(values, errors="coerce")
            if (values.notna() & numeric.isna()).any() or not np.isfinite(
                numeric.dropna()
            ).all():
                raise ValueError("Stratification scores must be numeric.")
            groups = frame[group_column]
            negative, positive = definition["negative"], definition["positive"]
            if negative == positive:
                raise ValueError("Stratum groups must differ.")
            medians = {
                group: float(numeric[groups.eq(group)].median())
                for group in [negative, positive]
            }
            if not all(np.isfinite(v) for v in medians.values()):
                raise ValueError("Both groups require observed scores.")
            high = pd.Series(False, index=frame.index)
            for group in [negative, positive]:
                high.loc[groups.eq(group)] = numeric[groups.eq(group)] > medians[group]
            for p_high, n_high in product([True, False], [False, True]):
                mask = numeric.notna() & (
                    (groups.eq(negative) & high.eq(n_high))
                    | (groups.eq(positive) & high.eq(p_high))
                )
                targets.append(
                    _target(
                        frame,
                        f"{key}_{int(p_high)}_{int(n_high)}",
                        mask,
                        groups.eq(positive),
                        {
                            "median_scope": "analysis subset before cross-validation",
                            "ties": "low",
                        },
                    )
                )
        else:
            raise ValueError("Unknown target definition.")
    if not targets or len({t.key for t in targets}) != len(targets):
        raise ValueError("Target keys must be nonempty and unique.")
    return targets
