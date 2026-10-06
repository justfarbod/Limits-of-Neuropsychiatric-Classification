import json

import numpy as np
import pandas as pd
import pytest

from neuroimaging_maxent.config import load_config
from neuroimaging_maxent.inputs import build_features, connectivity_vector, load_inputs
from neuroimaging_maxent.targets import build_targets
from neuroimaging_maxent.utils.paths import CODE_ROOT, code_root, external_path


def test_alignment_and_string_identifiers(toy_config, private_dir):
    data = load_inputs(toy_config)
    ids = np.array([f"00{i}-arbitrary" for i in range(len(data.features))])
    np.savez(private_dir / "features.npz", features=data.features, row_ids=ids)
    metadata = pd.DataFrame(
        {"row_id": ids, "position": np.arange(len(ids)), "group": data.metadata.group}
    )
    metadata.sample(frac=1, random_state=3).to_csv(
        private_dir / "metadata.csv", index=False
    )
    aligned = load_inputs(toy_config)
    np.testing.assert_array_equal(aligned.row_ids, ids)
    np.testing.assert_array_equal(aligned.metadata.position, np.arange(len(ids)))


@pytest.mark.parametrize("failure", ["duplicate", "missing", "nonfinite"])
def test_bad_inputs_fail(toy_config, private_dir, failure):
    data = load_inputs(toy_config)
    metadata = data.metadata
    if failure == "duplicate":
        metadata = pd.concat([metadata, metadata.iloc[:1]])
    elif failure == "missing":
        metadata = metadata.iloc[1:]
    else:
        data.features[0, 0] = np.nan
        np.savez(
            private_dir / "features.npz", features=data.features, row_ids=data.row_ids
        )
    metadata.to_csv(private_dir / "metadata.csv", index=False)
    with pytest.raises(ValueError):
        load_inputs(toy_config)


def test_target_scope_ties_and_threshold():
    score = np.tile(np.arange(5), 2).astype(float)
    frame = pd.DataFrame({"group": np.repeat(["A", "B"], 5), "score": score})
    strata = build_targets(
        frame,
        [
            {
                "kind": "score_strata",
                "key": "strata",
                "column": "score",
                "group_column": "group",
                "negative": "A",
                "positive": "B",
            }
        ],
    )
    assert len(strata) == 4
    for target in strata:
        p_high, n_high = map(int, target.key.split("_")[-2:])
        for class_value, is_high in [(0, n_high), (1, p_high)]:
            selected = score[target.indices[target.labels == class_value]]
            assert np.all(selected > 2) if is_high else np.all(selected <= 2)
        assert (
            target.metadata["median_scope"] == "analysis subset before cross-validation"
        )
    threshold = build_targets(
        frame,
        [{"kind": "threshold", "key": "threshold", "column": "score", "threshold": 2}],
    )[0]
    np.testing.assert_array_equal(threshold.labels, score >= 2)


def test_pairwise_and_subset(toy_config):
    data = load_inputs({**toy_config, "subset": {"column": "group", "values": [1]}})
    assert data.metadata.group.eq(1).all()
    frame = pd.DataFrame({"group": np.tile(["A", "B", "C"], 4)})
    targets = build_targets(
        frame,
        [
            {
                "kind": "pairwise_groups",
                "key": "pair",
                "column": "group",
                "groups": ["A", "B", "C"],
            }
        ],
    )
    assert len(targets) == 3
    assert all(len(t.labels) == 8 for t in targets)


def test_external_boundary_and_placeholder(private_dir):
    for path in ["<FEATURES_FILE>", CODE_ROOT / "private" / "input.npz", CODE_ROOT]:
        with pytest.raises(ValueError):
            external_path(path)
    link = private_dir / "link"
    link.symlink_to(CODE_ROOT, target_is_directory=True)
    with pytest.raises(ValueError):
        external_path(link / "input.npz")
    with pytest.raises(ValueError):
        load_config(CODE_ROOT / "configs" / "empirical_tree.json")
    other_checkout = private_dir / "checkout"
    other_checkout.mkdir()
    (other_checkout / "pyproject.toml").write_text(
        '[project]\nname="neuroimaging-maxent-analysis"\n'
    )
    with pytest.raises(ValueError):
        external_path(other_checkout / "output")
    config = private_dir / "config.json"
    config.write_text(json.dumps({"features_file": "<FEATURES_FILE>"}))
    with pytest.raises(ValueError):
        load_config(config)


def test_installed_path_boundary_does_not_block_external_temp_directory(private_dir):
    package = private_dir / "installed" / "neuroimaging_maxent"
    assert code_root(package / "utils" / "paths.py") == package
    checkout = private_dir / "checkout"
    assert (
        code_root(checkout / "src" / "neuroimaging_maxent" / "utils" / "paths.py")
        == checkout
    )


def test_connectivity_order_and_manifest(private_dir):
    signals = np.random.default_rng(5).normal(size=(20, 4))
    matrix = np.corrcoef(signals, rowvar=False)
    lower = connectivity_vector(signals)
    np.testing.assert_allclose(lower, matrix[np.tril_indices(4, -1)], atol=1e-7)
    np.testing.assert_allclose(
        connectivity_vector(signals.T, orientation="region_by_time"), lower
    )
    np.testing.assert_allclose(
        connectivity_vector(signals, triangle="upper"),
        matrix[np.triu_indices(4, 1)],
        atol=1e-7,
    )
    np.save(private_dir / "signal.npy", signals)
    pd.DataFrame({"row_id": ["toy-signal"], "file": ["signal.npy"]}).to_csv(
        private_dir / "manifest.csv", index=False
    )
    file = build_features(
        {
            "time_series_manifest": str(private_dir / "manifest.csv"),
            "output_dir": str(private_dir / "output"),
        }
    )
    with np.load(file) as data:
        np.testing.assert_array_equal(data["features"][0], lower)
    signals[:, 1] = 0
    assert np.isfinite(connectivity_vector(signals)).all()
    with pytest.raises(ValueError):
        connectivity_vector(signals, constant_policy="error")
