from dataclasses import replace
import json

import networkx as nx
import numpy as np
import pandas as pd
import torch

from .config import SparseTreewidthConfig, _merge
from .enumerated import MaxEnt
from .graph import LearnedTopology, TreeDecomposition
from .model import SharedGraphIsingClassifier, SparseTreewidthIsingModel
from ..autoencoders.representation import choose_device
from ..utils.paths import external_path


def sparse_config(config, seed):
    result = SparseTreewidthConfig()
    _merge(result, config.get("sparse", {}))
    result.graph.treewidth = int(config["treewidth"])
    result.fit.max_iterations = int(config["max_iterations"])
    result.inference.backend = config["inference_backend"]
    result.random_seed = int(seed)
    return result.validate()


def fit_valid(diagnostics, limit):
    return all(
        d.get("success", False)
        and np.isfinite(d.get("gradient_max_abs", np.inf))
        and d["gradient_max_abs"] <= limit
        and np.isfinite(d.get("log_partition_function", np.nan))
        for d in diagnostics
    )


class ModelPair:
    def __init__(self, models, method, diagnostics=None, attempts=None):
        self.models = tuple(models)
        self.method = method
        self.dimension = models[0].dimension if method == "sparse" else models[0].n
        self.diagnostics = diagnostics or []
        self.attempts = attempts or []
        if method == "exact":
            with torch.no_grad():
                self.log_partitions = [
                    torch.logsumexp(-m._energy(), dim=0).detach() for m in self.models
                ]

    @property
    def edges(self):
        return (
            self.models[0].edges
            if self.method == "sparse"
            else list(zip(*np.triu_indices(self.dimension, 1)))
        )

    def parameters(self, class_value):
        model = self.models[class_value]
        if self.method == "sparse":
            return model.h.copy(), model.j_matrix.copy()
        return (
            model.h.detach().cpu().numpy().copy(),
            model._symmetrize_J().detach().cpu().numpy().copy(),
        )

    def log_prob(self, values, class_value):
        model = self.models[class_value]
        if self.method == "sparse":
            return np.asarray(model.log_prob(values), dtype=np.float64)
        with torch.no_grad():
            states = torch.as_tensor(values, dtype=torch.float32, device=model.device)
            if states.ndim == 1:
                states = states[None]
            return (
                (-model._energy(states) - self.log_partitions[class_value])
                .cpu()
                .numpy()
                .astype(np.float64)
            )

    def score(self, values):
        return self.log_prob(values, 1) - self.log_prob(values, 0)

    def sample(self, class_value, n, seed):
        if self.method == "sparse":
            return self.models[class_value].sample(n, seed)
        model = self.models[class_value]
        with torch.no_grad():
            probabilities = (
                torch.softmax(-model._energy(), dim=0).cpu().numpy().astype(np.float64)
            )
            states = model.states.cpu().numpy().astype(np.int8)
        indices = np.random.default_rng(seed).choice(
            len(states), size=n, p=probabilities / probabilities.sum()
        )
        return states[indices]

    def moments(self, class_value):
        model = self.models[class_value]
        if self.method == "sparse":
            return model.inference.expectations(model._require_fit())
        with torch.no_grad():
            first, second = model.get_model_marginals()
        return first.cpu().numpy(), second.cpu().numpy()

    def save(self, path):
        path = external_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        h0, j0 = self.parameters(0)
        h1, j1 = self.parameters(1)
        topology = (
            self.models[0].topology.decomposition.to_dict()
            if self.method == "sparse"
            else {}
        )
        np.savez_compressed(
            path,
            format_version=1,
            method=self.method,
            h0=h0,
            j0=j0,
            h1=h1,
            j1=j1,
            edges=np.asarray(self.edges, dtype=int).reshape(-1, 2),
            decomposition=json.dumps(topology),
        )

    def interpolated(self, alpha):
        h0, j0 = self.parameters(0)
        h1, j1 = self.parameters(1)
        return from_parameters(
            self.method,
            h0,
            j0,
            h0 + alpha * (h1 - h0),
            j0 + alpha * (j1 - j0),
            self.models[0].topology if self.method == "sparse" else None,
        )


def from_parameters(method, h0, j0, h1, j1, topology=None, device="cpu"):
    if np.shape(h0) != np.shape(h1):
        raise ValueError("Class source dimensions must agree.")
    models = []
    for h, j in [(h0, j0), (h1, j1)]:
        h, j = np.asarray(h), np.asarray(j)
        if (
            h.ndim != 1
            or len(h) < 1
            or j.shape != (len(h), len(h))
            or not np.isfinite(h).all()
            or not np.isfinite(j).all()
        ):
            raise ValueError("Invalid source parameters.")
        if not np.allclose(j, j.T, atol=1e-12) or not np.allclose(
            np.diag(j), 0, atol=1e-12
        ):
            raise ValueError(
                "Source interactions must be symmetric with a zero diagonal."
            )
        if method == "sparse":
            if topology is None:
                raise ValueError("A sparse source requires a certified topology.")
            if len(topology.graph) != len(h):
                raise ValueError(
                    "Source parameters and topology dimensions must agree."
                )
            mask = np.zeros(j.shape, dtype=bool)
            for left, right in topology.graph.edges:
                mask[left, right] = mask[right, left] = True
            if np.any(np.abs(j[~mask]) > 1e-12):
                raise ValueError(
                    "Sparse source interactions must lie on its selected edges."
                )
            model = SparseTreewidthIsingModel(
                topology, SparseTreewidthConfig().fit, inference_backend="numpy"
            )
            model.h = h.astype(np.float64)
            model.couplings = np.array(
                [j[a, b] for a, b in model.edges], dtype=np.float64
            )
            model.result = model.inference.calibrate(model.h, model.couplings)
        elif method == "exact" and len(h) <= 22:
            model = MaxEnt(len(h), device=device)
            model.h.data = torch.as_tensor(h, dtype=torch.float32, device=device)
            model.J.data = torch.as_tensor(j, dtype=torch.float32, device=device)
        else:
            raise ValueError("Unsupported source method or dimension.")
        models.append(model)
    return ModelPair(models, method)


def load_pair(path, device="cpu"):
    with np.load(external_path(path, must_exist=True), allow_pickle=False) as payload:
        if int(payload["format_version"]) != 1:
            raise ValueError("Unsupported source model version.")
        method = str(payload["method"])
        topology = None
        if method == "sparse":
            graph = nx.Graph()
            graph.add_nodes_from(range(len(payload["h0"])))
            graph.add_edges_from(payload["edges"].tolist())
            decomposition = TreeDecomposition.from_dict(
                json.loads(str(payload["decomposition"]))
            ).validate(graph)
            topology = LearnedTopology(
                graph,
                decomposition,
                np.empty((0, 0)),
                pd.DataFrame(),
                np.empty((0, 0)),
                pd.DataFrame(),
            )
        return from_parameters(
            method,
            payload["h0"],
            payload["j0"],
            payload["h1"],
            payload["j1"],
            topology,
            device,
        )


def fit_pair(values, labels, config, seed, topology=None):
    values, labels = np.asarray(values), np.asarray(labels)
    if (
        values.ndim != 2
        or not np.all(np.isin(values, [0, 1]))
        or labels.shape != (len(values),)
        or set(np.unique(labels)) != {0, 1}
    ):
        raise ValueError("Fitting requires aligned binary states and both classes.")
    if config["method"] == "sparse":
        settings = sparse_config(config, seed)
        classifier = SharedGraphIsingClassifier(settings).fit(
            values, labels, learned_topology=topology
        )
        models = classifier._models()
        initial = [dict(m.diagnostics) for m in models]
        attempts = [
            {"max_iterations": settings.fit.max_iterations, "diagnostics": initial}
        ]
        if (
            not fit_valid(initial, config["gradient_max_abs"])
            and config["retry_max_iterations"] > settings.fit.max_iterations
        ):
            settings.fit = replace(
                settings.fit, max_iterations=int(config["retry_max_iterations"])
            )
            classifier = SharedGraphIsingClassifier(settings).fit(
                values, labels, learned_topology=classifier.topology
            )
            models = classifier._models()
            attempts.append(
                {
                    "max_iterations": settings.fit.max_iterations,
                    "diagnostics": [dict(m.diagnostics) for m in models],
                }
            )
        return ModelPair(
            models, "sparse", [dict(m.diagnostics) for m in models], attempts
        )
    if values.shape[1] > 22:
        raise ValueError("Full enumeration is limited to 22 variables.")
    models, diagnostics = [], []
    device = choose_device(config["device"])
    for class_value in [0, 1]:
        torch.manual_seed(seed + class_value)
        model = MaxEnt(values.shape[1], device=device)
        model.fit(
            values[labels == class_value],
            lr=config["exact_lr"],
            steps=config["exact_steps"],
            patience=config["exact_patience"],
            lambda_=config["exact_l1"],
            verbose=False,
        )
        with torch.no_grad():
            first, second = model.get_model_marginals()
            empirical = values[labels == class_value].astype(float)
            errors = np.r_[
                first.cpu().numpy() - empirical.mean(axis=0),
                second.cpu().numpy()
                - (empirical.T @ empirical / len(empirical))[
                    np.triu_indices(values.shape[1], 1)
                ],
            ]
            log_z = float(torch.logsumexp(-model._energy(), dim=0).cpu())
        diagnostics.append(
            {
                "success": bool(np.isfinite(errors).all() and np.isfinite(log_z)),
                "gradient_max_abs": float(np.max(np.abs(errors))),
                "log_partition_function": log_z,
            }
        )
        models.append(model)
    return ModelPair(models, "exact", diagnostics)
