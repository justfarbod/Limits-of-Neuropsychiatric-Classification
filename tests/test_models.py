from itertools import product

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp

from neuroimaging_maxent.models.graph import LearnedTopology, TreeDecomposition
from neuroimaging_maxent.models.inference import JunctionTreeInference, NUMBA_AVAILABLE
from neuroimaging_maxent.models.pair import fit_pair, from_parameters, load_pair
from neuroimaging_maxent.utils.metrics import exact_weighted_auc


def topology(graph):
    return LearnedTopology(
        graph,
        TreeDecomposition.from_graph(graph),
        np.empty((0, 0)),
        pd.DataFrame(),
        np.empty((0, 0)),
        pd.DataFrame(),
    )


@pytest.mark.parametrize("width", [1, 2, 3])
@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_inference_matches_enumeration(width, backend):
    if backend == "numba" and not NUMBA_AVAILABLE:
        pytest.skip("Optional compiled backend is unavailable.")
    dimension = 6
    graph = nx.path_graph(dimension)
    for i in range(dimension):
        for j in range(i + 2, min(dimension, i + width + 1)):
            graph.add_edge(i, j)
    topo = topology(graph)
    edges = sorted(graph.edges)
    plan = JunctionTreeInference(dimension, edges, topo.decomposition, backend=backend)
    rng = np.random.default_rng(width)
    h, couplings = (
        rng.normal(scale=0.3, size=dimension),
        rng.normal(scale=0.2, size=len(edges)),
    )
    result = plan.calibrate(h, couplings)
    states = np.array(list(product([0, 1], repeat=dimension)), dtype=float)
    logweight = states @ h + sum(
        c * states[:, i] * states[:, j] for (i, j), c in zip(edges, couplings)
    )
    probabilities = np.exp(logweight - logsumexp(logweight))
    first, second = plan.expectations(result)
    np.testing.assert_allclose(result.log_z, logsumexp(logweight), atol=1e-12)
    np.testing.assert_allclose(first, probabilities @ states, atol=1e-12)
    np.testing.assert_allclose(
        second,
        [probabilities @ (states[:, i] * states[:, j]) for i, j in edges],
        atol=1e-12,
    )
    sampled = plan.sample(result, 20000, 4)
    np.testing.assert_allclose(sampled.mean(axis=0), first, atol=0.015)


def test_sparse_and_exact_energy_likelihood_parity(private_dir):
    dimension = 5
    rng = np.random.default_rng(4)
    topo = topology(nx.path_graph(dimension))
    h = rng.normal(scale=0.2, size=dimension)
    j = np.zeros((dimension, dimension))
    for a, b in topo.graph.edges:
        j[a, b] = j[b, a] = rng.normal(scale=0.1)
    sparse = from_parameters("sparse", h, j, h + 0.1, j, topo)
    exact = from_parameters("exact", h, j, h + 0.1, j)
    states = np.array(list(product([0, 1], repeat=dimension)), dtype=np.int8)
    for class_value in [0, 1]:
        np.testing.assert_allclose(
            sparse.log_prob(states, class_value),
            exact.log_prob(states, class_value),
            atol=1e-6,
        )
        assert np.exp(sparse.log_prob(states, class_value)).sum() == pytest.approx(1)
    expected_energy = -states @ h - 0.5 * np.einsum("bi,ij,bj->b", states, j, states)
    np.testing.assert_allclose(
        sparse.models[0].energy(states), expected_energy, atol=1e-12
    )
    sparse.save(private_dir / "model.npz")
    reloaded = load_pair(private_dir / "model.npz")
    np.testing.assert_allclose(reloaded.score(states), sparse.score(states), atol=1e-12)


def test_single_retry_retains_topology(monkeypatch, toy_config):
    from neuroimaging_maxent.models.model import SharedGraphIsingClassifier

    original = SharedGraphIsingClassifier.fit
    calls = []

    def record(self, *args, **kwargs):
        output = original(self, *args, **kwargs)
        calls.append((self.config.fit.max_iterations, output.topology))
        for model in output._models():
            model.diagnostics["success"] = False
        return output

    monkeypatch.setattr(SharedGraphIsingClassifier, "fit", record)
    rng = np.random.default_rng(4)
    states = rng.integers(0, 2, size=(80, 4))
    labels = np.tile([0, 1], 40)
    pair = fit_pair(states, labels, toy_config, 7)
    assert len(calls) == 2
    assert calls[0][0] == toy_config["max_iterations"]
    assert calls[1][0] == toy_config["retry_max_iterations"]
    assert calls[0][1] is calls[1][1]
    assert len(pair.attempts) == 2


def test_weighted_auc_ties_and_orientation():
    rng = np.random.default_rng(11)
    scores = rng.integers(0, 3, size=7).astype(float)
    positive, negative = rng.random(7), rng.random(7)
    positive /= positive.sum()
    negative /= negative.sum()
    expected = sum(
        positive[i]
        * negative[j]
        * ((scores[i] > scores[j]) + 0.5 * (scores[i] == scores[j]))
        for i in range(7)
        for j in range(7)
    )
    assert exact_weighted_auc(scores, scores, positive, negative) == pytest.approx(
        expected
    )
    assert exact_weighted_auc(scores, scores, positive, positive) == pytest.approx(0.5)
