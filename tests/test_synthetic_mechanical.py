from itertools import product

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from neuroimaging_maxent.mechanical.quadrature import triple_well
from neuroimaging_maxent.models.graph import LearnedTopology, TreeDecomposition
from neuroimaging_maxent.models.pair import from_parameters
from neuroimaging_maxent.synthetic.source import fit_source
from neuroimaging_maxent.synthetic.workflow import oracle_auc, run_synthetic


@pytest.mark.parametrize("method", ["sparse", "exact"])
def test_oracle_negative_control_and_interpolation(method):
    dimension = 4
    graph = nx.path_graph(dimension)
    topology = LearnedTopology(
        graph,
        TreeDecomposition.from_graph(graph),
        np.empty((0, 0)),
        pd.DataFrame(),
        np.empty((0, 0)),
        pd.DataFrame(),
    )
    h0 = np.zeros(dimension)
    h1 = np.full(dimension, 0.8)
    j = np.zeros((dimension, dimension))
    pair = from_parameters(method, h0, j, h1, j, topology)
    null = pair.interpolated(0)
    assert oracle_auc(null)[0] == [0.5]
    states = np.array(list(product([0, 1], repeat=dimension)), dtype=np.int8)
    np.testing.assert_array_equal(null.score(states), 0)
    np.testing.assert_allclose(pair.interpolated(0.5).parameters(1)[0], 0.5 * h1)
    aucs, _ = oracle_auc(pair, repeats=3, samples_per_class=5000)
    assert 0.65 < np.mean(aucs) < 0.9
    sampled = pair.sample(1, 12000, 6)
    np.testing.assert_allclose(sampled.mean(axis=0), 1 / (1 + np.exp(-h1)), atol=0.02)


@pytest.mark.parametrize("method", ["sparse", "exact"])
def test_source_and_roundtrip_workflow(toy_config, private_dir, method):
    source_config = {
        **toy_config,
        "method": method,
        "source_autoencoder": "compact",
        "exact_steps": 50,
        "output_dir": str(private_dir / "source"),
    }
    fidelity = fit_source(source_config)
    assert fidelity["scope"] == "descriptive full input subset"
    config = {
        **source_config,
        "source_model": str(private_dir / "source" / "source_model.npz"),
        "autoencoder_checkpoint": str(private_dir / "source" / "autoencoder.pt"),
        "output_dir": str(private_dir / "simulation"),
        "samples_per_class": 24,
        "n_replicates": 2,
        "alpha_values": [0.0, 1.0],
        "oracle_mc_repeats": 2,
        "oracle_mc_samples_per_class": 100,
    }
    summary = run_synthetic(config)
    null = summary[(summary.alpha == 0) & (summary.method == "oracle_population")]
    assert null.auc.iloc[0] == 0.5
    assert {
        "oracle_finite",
        "binary_latent_svc",
        "decoded_fc_svc",
        "roundtrip_maxent",
    } <= set(summary.method)
    fractions = pd.read_csv(private_dir / "simulation" / "roundtrip.csv")
    assert fractions.bit_fraction.between(0, 1).all()
    assert fractions.exact_fraction.between(0, 1).all()


@pytest.mark.parametrize("alpha", [0.0, 0.39, 1.26, -0.39])
def test_triple_well_identity_and_symmetry(alpha):
    row = triple_well(alpha, points=10001)
    assert abs(row["work_identity_error"]) < 1e-12
    assert row["forward_work"] >= -1e-12
    assert row["reverse_work"] >= -1e-12
    assert row["auc"] >= 0.5
    if alpha == 0:
        assert row["auc"] == 0.5
        assert row["reciprocal_work"] == 0
    else:
        reflected = triple_well(-alpha, points=10001)
        assert row["auc"] == pytest.approx(reflected["auc"], abs=1e-10)
        assert row["right_occupancy"] == pytest.approx(
            reflected["left_occupancy"], abs=1e-10
        )


def test_quadrature_resolution():
    coarse = triple_well(0.6, points=3001)
    fine = triple_well(0.6, points=12001)
    assert coarse["auc"] == pytest.approx(fine["auc"], abs=1e-6)
    assert coarse["reciprocal_work"] == pytest.approx(fine["reciprocal_work"], abs=1e-8)
