import json

import numpy as np
import pandas as pd
import pytest

from neuroimaging_maxent.selection.diagnostics import (
    jeffreys_standard_error,
    moment_fit_summary,
)
from neuroimaging_maxent.selection.rules import select_candidates
from neuroimaging_maxent.selection.workflow import (
    joint_cells,
    partitions,
    run_selection,
)


def candidate_metrics():
    rows = []
    for key, dimension, auc, fit in [("small", 3, 0.65, -1.0), ("wide", 6, 0.68, -1.2)]:
        for outcome in ["task_one", "task_two"]:
            rows.append(
                {
                    "candidate_key": key,
                    "outcome": outcome,
                    "auc": auc,
                    "auc_se": 0.02,
                    "eligible": True,
                    "latent_dim": dimension,
                    "parameter_count": 2 * dimension - 1,
                    "method": "sparse",
                    "requested_treewidth": 1,
                    "actual_treewidth": 1,
                    "edge_count": dimension - 1,
                    "fit_score": fit,
                    "fit_se": 0.05,
                }
            )
    return pd.DataFrame(rows)


def test_selection_rules_and_no_winner():
    metrics = candidate_metrics()
    assert (
        select_candidates(metrics, None, "macro-auc", method="sparse").winner_key
        == "wide"
    )
    assert (
        select_candidates(metrics, None, "fit-first", method="sparse").winner_key
        == "small"
    )
    assert (
        select_candidates(metrics, None, "balanced-one-se", method="sparse").status
        == "selected"
    )
    metrics.loc[:, "eligible"] = False
    assert select_candidates(metrics, None, "fit-first", method="sparse").winner is None


def test_complexity_ties_and_dynamic_outcome_names():
    metrics = candidate_metrics()
    metrics.loc[:, "auc"] = 0.6
    metrics.loc[:, "fit_score"] = -1.0
    metrics["outcome"] = metrics.outcome.str.replace("_", "-")
    result = select_candidates(metrics, None, "macro-auc", method="sparse")
    assert result.winner_key == "small"


def test_joint_stratification_and_scope():
    labels = [np.tile([0, 1], 40), np.tile(np.repeat([0, 1], 2), 20)]
    cells = joint_cells(labels)
    pool = np.arange(len(cells))
    splits = partitions(pool, cells, 5, 42)
    assert len(np.unique(cells)) == 4
    for train, test in splits:
        assert not np.intersect1d(train, test).size
        assert set(cells[test]) == set(cells)
        for inner_train, inner_test in partitions(train, cells, 2, 43):
            assert not np.intersect1d(test, inner_train).size
            assert not np.intersect1d(test, inner_test).size
    with pytest.raises(ValueError):
        partitions(np.arange(3), cells, 5, 42)


def test_selected_edge_moment_diagnostic():
    rng = np.random.default_rng(7)
    states = rng.integers(0, 2, size=(40, 4))
    edges = np.array([(0, 1), (1, 2)])
    first = states.mean(axis=0)
    second = np.mean(states[:, edges[:, 0]] * states[:, edges[:, 1]], axis=0)
    rows = moment_fit_summary(states, first, second, edges=edges)
    assert [r["n_moments"] for r in rows] == [4, 2]
    assert all(r["standardized_rms"] == 0 for r in rows)
    assert np.all(jeffreys_standard_error(np.linspace(0, 1, 5), 40) > 0)


def test_nested_workflow_audit(toy_config, private_dir):
    config = {
        **toy_config,
        "latent_dims": [3],
        "treewidths": [1],
        "inner_folds": 2,
        "selection_bootstrap_repeats": 20,
    }
    run_selection(config)
    folds = list((private_dir / "output").glob("outer_*/training.json"))
    assert len(folds) == 2
    for file in folds:
        inner = json.loads(file.read_text())
        for row in inner:
            assert set(row["training_positions"]).isdisjoint(
                row["validation_positions"]
            )
            assert set(row["training_positions"]) <= set(row["selection_positions"])
    assert (private_dir / "output" / "full_candidate_metrics.csv").exists()
    assert (private_dir / "output" / "outer_predictions.csv").exists()
