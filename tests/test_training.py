import numpy as np
import pandas as pd
import pytest

from neuroimaging_maxent.autoencoders.representation import (
    fit_representation,
    load_representation,
)
from neuroimaging_maxent.empirical.discriminative import (
    bootstrap_repeated_fold_auc,
    make_fixed_pipeline,
)
from neuroimaging_maxent.empirical.workflow import fold_representation, run_empirical
from neuroimaging_maxent.inputs import load_inputs


def test_training_excludes_validation_rows(toy_config, private_dir):
    features = load_inputs(toy_config).features
    train = np.arange(24)
    altered = features.copy()
    altered[24:] += 500
    first = fit_representation(features, train, toy_config, 33)
    second = fit_representation(altered, train, toy_config, 33)
    np.testing.assert_array_equal(first.mean, second.mean)
    np.testing.assert_array_equal(first.std, second.std)
    for key in first.model.state_dict():
        np.testing.assert_array_equal(
            first.model.state_dict()[key], second.model.state_dict()[key]
        )
    first.save(private_dir / "checkpoint.pt")
    loaded = load_representation(private_dir / "checkpoint.pt", "cpu")
    np.testing.assert_array_equal(first.encode(features)[1], loaded.encode(features)[1])
    np.testing.assert_allclose(
        first.decode(first.encode(features)[1]),
        loaded.decode(loaded.encode(features)[1]),
    )


def test_staged_autoencoder_is_distinct(toy_config):
    config = {
        **toy_config,
        "staged_autoencoder": {
            "hidden_1": 16,
            "hidden_2": 8,
            "pretrain_epochs": 1,
            "finetune_epochs": 1,
        },
    }
    features = load_inputs(config).features
    staged = fit_representation(features, np.arange(24), config, 10, "staged")
    compact = fit_representation(features, np.arange(24), config, 10)
    assert type(staged.model) is not type(compact.model)
    continuous, binary = staged.encode(features)
    assert binary.shape == (len(features), config["latent_dim"])
    assert set(np.unique(binary)) <= {0, 1}
    assert staged.decode(binary).shape == features.shape


def test_cache_changes_with_training_data(toy_config, private_dir):
    features = load_inputs(toy_config).features
    train = np.arange(24)
    cache = private_dir / "cache"
    first, _ = fold_representation(features, train, toy_config, 3, cache)
    repeat, _ = fold_representation(features, train, toy_config, 3, cache)
    assert len(list(cache.iterdir())) == 1
    np.testing.assert_array_equal(first.encode(features)[1], repeat.encode(features)[1])
    altered = features.copy()
    altered[0, 0] += 10
    fold_representation(altered, train, toy_config, 3, cache)
    assert len(list(cache.iterdir())) == 2


def test_bootstrap_grouping_matches_direct_resampling():
    from sklearn.metrics import roc_auc_score

    labels = np.tile([0, 1], 12)
    rng = np.random.default_rng(9)
    scores = rng.normal(size=(2, len(labels)))
    folds = np.tile(np.repeat(np.arange(3), 8), (2, 1))
    draws = bootstrap_repeated_fold_auc(labels, scores, folds, 20, 17)
    rng = np.random.default_rng(17)
    negative, positive = np.flatnonzero(labels == 0), np.flatnonzero(labels == 1)
    counts = np.zeros((20, len(labels)), dtype=int)
    counts[:, negative] = rng.multinomial(
        len(negative), np.full(len(negative), 1 / len(negative)), size=20
    )
    counts[:, positive] = rng.multinomial(
        len(positive), np.full(len(positive), 1 / len(positive)), size=20
    )
    expected = []
    for weights in counts:
        aucs = []
        for repeat in range(2):
            for fold in range(3):
                subset = np.flatnonzero(folds[repeat] == fold)
                if all(weights[subset[labels[subset] == c]].sum() > 0 for c in [0, 1]):
                    aucs.append(
                        roc_auc_score(
                            labels[subset],
                            scores[repeat, subset],
                            sample_weight=weights[subset],
                        )
                    )
        expected.append(np.mean(aucs))
    np.testing.assert_allclose(draws, expected, atol=1e-12)


def test_svc_settings():
    estimator = make_fixed_pipeline(12, 1)
    assert estimator.named_steps["svc"].gamma == pytest.approx(1 / 12)
    assert estimator.named_steps["svc"].C == 1
    assert estimator.named_steps["imputer"].strategy == "mean"


def test_empirical_workflow_and_common_baseline(toy_config, private_dir):
    first = run_empirical(toy_config)
    second_config = {
        **toy_config,
        "latent_dim": 4,
        "output_dir": str(private_dir / "other"),
    }
    second = run_empirical(second_config)
    baseline1 = first[first.method == "original_fc"].auc.to_numpy()
    baseline2 = second[second.method == "original_fc"].auc.to_numpy()
    np.testing.assert_array_equal(baseline1, baseline2)
    assert len(first) == 5
    folds = pd.read_csv(private_dir / "output" / "folds.csv")
    assert folds[folds.method == "maxent"].fit_valid.all()
    predictions = pd.read_csv(private_dir / "output" / "predictions.csv")
    assert (
        predictions.groupby(["method", "row_id"])
        .size()
        .eq(len(toy_config["outer_seeds"]))
        .all()
    )


def test_invalid_retry_blocks_complete_summary(monkeypatch, toy_config, private_dir):
    monkeypatch.setattr(
        "neuroimaging_maxent.empirical.workflow.fit_valid", lambda *_: False
    )
    with pytest.raises(RuntimeError):
        run_empirical(toy_config)
    assert not (private_dir / "output" / "summary.csv").exists()
    assert (private_dir / "output" / "failed_folds.csv").exists()
