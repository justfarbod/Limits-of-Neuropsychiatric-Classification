"""Label-blind screening and constructive bounded-width decompositions."""

from __future__ import annotations

import json
import hashlib
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from ..utils.paths import external_path
from time import perf_counter
from typing import Any, Iterable

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression

from .config import GraphConfig, ScreeningConfig


SCREENING_COLUMNS = [
    "source",
    "target",
    "selection_frequency",
    "coefficient_strength",
    "coef_source_on_target",
    "coef_target_on_source",
]
DECISION_COLUMNS = [
    "rank",
    "source",
    "target",
    "selection_frequency",
    "coefficient_strength",
    "accepted",
    "reason",
]


def validate_binary_matrix(
    values: np.ndarray, dimension: int | None = None
) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError(f"Binary states must be two-dimensional, got {values.shape}")
    if len(values) < 1 or values.shape[1] < 1:
        raise ValueError("Binary states must have at least one row and one column")
    if dimension is not None and values.shape[1] != dimension:
        raise ValueError(f"Expected dimension {dimension}, got {values.shape[1]}")
    if not np.all(np.isfinite(values)) or not np.all(np.isin(values, (0, 1))):
        raise ValueError("Latent states must contain only finite 0/1 values")
    return values.astype(np.int8, copy=False)


def binary_mutual_information(
    values: np.ndarray, pseudocount: float = 0.5
) -> np.ndarray:
    """Vectorized smoothed pairwise mutual information for binary columns."""
    values = validate_binary_matrix(values)
    if pseudocount < 0:
        raise ValueError("pseudocount cannot be negative")
    x = values.astype(np.float64, copy=False)
    n = float(len(x))
    ones = x.sum(axis=0)
    n11 = x.T @ x
    n10 = ones[:, None] - n11
    n01 = ones[None, :] - n11
    n00 = n - ones[:, None] - ones[None, :] + n11
    counts = np.stack((n00, n01, n10, n11), axis=0) + pseudocount
    joint = counts / counts.sum(axis=0, keepdims=True)
    p_left_0 = joint[0] + joint[1]
    p_left_1 = joint[2] + joint[3]
    p_right_0 = joint[0] + joint[2]
    p_right_1 = joint[1] + joint[3]
    products = np.stack(
        (
            p_left_0 * p_right_0,
            p_left_0 * p_right_1,
            p_left_1 * p_right_0,
            p_left_1 * p_right_1,
        ),
        axis=0,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(joint > 0, joint * np.log(joint / products), 0.0)
    result = terms.sum(axis=0)
    result = 0.5 * (result + result.T)
    np.fill_diagonal(result, 0.0)
    return result


def chow_liu_tree(mutual_information: np.ndarray) -> nx.Graph:
    mutual_information = np.asarray(mutual_information, dtype=np.float64)
    if (
        mutual_information.ndim != 2
        or mutual_information.shape[0] != mutual_information.shape[1]
    ):
        raise ValueError("Mutual information must be a square matrix")
    dimension = len(mutual_information)
    graph = nx.Graph()
    graph.add_nodes_from(range(dimension))
    for left in range(dimension):
        for right in range(left + 1, dimension):
            graph.add_edge(left, right, weight=float(mutual_information[left, right]))
    tree = nx.maximum_spanning_tree(graph, algorithm="kruskal", weight="weight")
    if dimension > 1 and not nx.is_tree(tree):
        raise RuntimeError("Chow-Liu construction did not produce a spanning tree")
    return tree


def _fit_node(
    values: np.ndarray, target: int, config: ScreeningConfig, seed: int
) -> np.ndarray:
    dimension = values.shape[1]
    output = np.zeros(dimension, dtype=np.float64)
    if dimension == 1:
        return output
    response = values[:, target]
    if np.unique(response).size < 2:
        return output
    predictors = np.delete(values, target, axis=1)
    estimator = LogisticRegression(
        penalty="l1",
        C=config.logistic_c,
        solver=config.solver,
        tol=config.tolerance,
        max_iter=config.max_iter,
        random_state=seed,
    )
    estimator.fit(predictors, response)
    indices = np.delete(np.arange(dimension), target)
    output[indices] = estimator.coef_[0]
    return output


def nodewise_coefficients(
    values: np.ndarray, config: ScreeningConfig, seed: int
) -> np.ndarray:
    values = validate_binary_matrix(values)
    rows = Parallel(n_jobs=config.n_jobs, prefer="threads")(
        delayed(_fit_node)(values, target, config, seed + target)
        for target in range(values.shape[1])
    )
    output = np.vstack(rows)
    np.fill_diagonal(output, 0.0)
    return output


def _edge_selected(coefficients: np.ndarray, config: ScreeningConfig) -> np.ndarray:
    directed = np.abs(coefficients) > config.coefficient_tolerance
    selected = (
        directed | directed.T
        if config.symmetrization == "or"
        else directed & directed.T
    )
    np.fill_diagonal(selected, False)
    return selected


def screen_edges(
    values: np.ndarray,
    config: ScreeningConfig,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return stable undirected candidates and the full-data directed coefficients."""
    values = validate_binary_matrix(values)
    full = nodewise_coefficients(values, config, seed)
    full_selected = _edge_selected(full, config)
    dimension = values.shape[1]
    if config.stability_selection:
        selected_count = np.zeros((dimension, dimension), dtype=np.int32)
        rng = np.random.default_rng(seed)
        sample_size = max(
            2, int(np.ceil(config.stability_subsample_fraction * len(values)))
        )
        sample_size = min(sample_size, len(values))
        for repeat in range(config.stability_repetitions):
            indices = rng.choice(len(values), size=sample_size, replace=False)
            coefficients = nodewise_coefficients(
                values[indices], config, seed + 10_000 * (repeat + 1)
            )
            selected_count += _edge_selected(coefficients, config)
        frequency = selected_count / float(config.stability_repetitions)
        selected = frequency >= config.stability_threshold
    else:
        frequency = full_selected.astype(np.float64)
        selected = full_selected
    strength = 0.5 * (np.abs(full) + np.abs(full.T))
    rows: list[dict[str, Any]] = []
    for left in range(dimension):
        candidates: list[tuple[int, float, float]] = []
        for right in range(left + 1, dimension):
            if selected[left, right]:
                candidates.append(
                    (right, float(frequency[left, right]), float(strength[left, right]))
                )
        for right, stable, score in candidates:
            rows.append(
                {
                    "source": left,
                    "target": right,
                    "selection_frequency": stable,
                    "coefficient_strength": score,
                    "coef_source_on_target": float(full[right, left]),
                    "coef_target_on_source": float(full[left, right]),
                }
            )
    table = pd.DataFrame(rows, columns=SCREENING_COLUMNS)
    if not table.empty and config.max_candidates_per_node is not None:
        ranked = table.sort_values(
            ["selection_frequency", "coefficient_strength", "source", "target"],
            ascending=[False, False, True, True],
        )
        counts = np.zeros(dimension, dtype=np.int64)
        keep: list[int] = []
        for index, row in ranked.iterrows():
            left, right = int(row["source"]), int(row["target"])
            if (
                counts[left] < config.max_candidates_per_node
                and counts[right] < config.max_candidates_per_node
            ):
                keep.append(index)
                counts[left] += 1
                counts[right] += 1
        table = table.loc[keep]
    if not table.empty:
        table = table.sort_values(
            ["selection_frequency", "coefficient_strength", "source", "target"],
            ascending=[False, False, True, True],
        ).reset_index(drop=True)
    return table, full


@dataclass
class TreeDecomposition:
    bags: dict[int, frozenset[int]]
    tree: nx.Graph

    @classmethod
    def from_graph(cls, graph: nx.Graph) -> "TreeDecomposition":
        if len(graph) == 1:
            node = int(next(iter(graph.nodes)))
            tree = nx.Graph()
            tree.add_node(0)
            return cls({0: frozenset({node})}, tree)
        if nx.is_tree(graph):
            root = int(min(graph.nodes))
            parent = {root: None}
            for left, right in nx.bfs_edges(graph, root):
                parent[int(right)] = int(left)
            non_root = sorted(node for node in graph.nodes if node != root)
            bag_for_node = {int(node): index for index, node in enumerate(non_root)}
            bags = {
                bag_for_node[int(node)]: frozenset({int(node), int(parent[int(node)])})
                for node in non_root
            }
            tree = nx.Graph()
            tree.add_nodes_from(bags)
            root_children = [
                int(node) for node in non_root if parent[int(node)] == root
            ]
            anchor = bag_for_node[root_children[0]]
            for child in root_children[1:]:
                tree.add_edge(anchor, bag_for_node[child])
            for node in non_root:
                node = int(node)
                if parent[node] != root:
                    tree.add_edge(bag_for_node[node], bag_for_node[int(parent[node])])
            return cls(bags, tree).validate(graph)
        width, decomposition = nx.approximation.treewidth_min_fill_in(graph)
        if width > 1 and nx.is_tree(graph):
            raise RuntimeError(
                "Tree decomposition of a tree has width greater than one"
            )
        bag_nodes = sorted(
            (frozenset(int(value) for value in bag) for bag in decomposition.nodes),
            key=lambda bag: (min(bag), len(bag), tuple(sorted(bag))),
        )
        ids = {bag: index for index, bag in enumerate(bag_nodes)}
        tree = nx.Graph()
        tree.add_nodes_from(ids.values())
        tree.add_edges_from(
            (ids[left], ids[right]) for left, right in decomposition.edges
        )
        return cls({index: bag for bag, index in ids.items()}, tree)

    @property
    def width(self) -> int:
        return max(len(bag) for bag in self.bags.values()) - 1

    def validate(self, graph: nx.Graph | None = None) -> "TreeDecomposition":
        if len(self.tree) > 1 and not nx.is_tree(self.tree):
            raise ValueError("Decomposition structure must be a tree")
        for variable in set().union(*self.bags.values()):
            nodes = [bag_id for bag_id, bag in self.bags.items() if variable in bag]
            if nodes and not nx.is_connected(self.tree.subgraph(nodes)):
                raise ValueError(
                    f"Running-intersection property failed for variable {variable}"
                )
        if graph is not None:
            for left, right in graph.edges:
                if not any({left, right}.issubset(bag) for bag in self.bags.values()):
                    raise ValueError(
                        f"Graph edge {(left, right)} is absent from every bag"
                    )
        return self

    def try_add_edge(self, left: int, right: int, max_width: int) -> tuple[bool, str]:
        left_bags = [index for index, bag in self.bags.items() if left in bag]
        right_bags = {index for index, bag in self.bags.items() if right in bag}
        if not left_bags or not right_bags:
            raise ValueError("Edge endpoint is missing from the decomposition")
        common = set(left_bags) & right_bags
        if common:
            return True, "already_covered"
        if not hasattr(self, "_path_cache"):
            self._path_cache = {
                int(source): {int(target): route for target, route in routes.items()}
                for source, routes in nx.all_pairs_shortest_path(self.tree)
            }
        paths: list[list[int]] = []
        for start in sorted(left_bags):
            destination = min(
                right_bags,
                key=lambda item: (len(self._path_cache[start][item]), item),
            )
            paths.append(self._path_cache[start][destination])
        path = min(paths, key=lambda item: (len(item), tuple(item)))
        options: list[tuple[int, int, int, list[frozenset[int]], str]] = []
        for variable, name in ((left, "extend_source"), (right, "extend_target")):
            changed = [self.bags[index] | {variable} for index in path]
            maximum = max(
                [len(bag) for index, bag in self.bags.items() if index not in path]
                + [len(bag) for bag in changed]
            )
            total = sum(len(bag) for bag in changed)
            options.append((maximum, total, variable, changed, name))
        maximum, _, _, changed, name = min(
            options, key=lambda item: (item[0], item[1], item[2])
        )
        if maximum - 1 > max_width:
            return False, "treewidth_limit"
        for index, bag in zip(path, changed):
            self.bags[index] = frozenset(bag)
        return True, name

    def find_bag(self, variables: Iterable[int]) -> int:
        variables = set(variables)
        matches = [
            (len(bag), index)
            for index, bag in self.bags.items()
            if variables.issubset(bag)
        ]
        if not matches:
            raise KeyError(f"No decomposition bag contains {sorted(variables)}")
        return min(matches)[1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "bags": [
                {"bag_id": index, "variables": sorted(bag)}
                for index, bag in sorted(self.bags.items())
            ],
            "tree_edges": [
                list(map(int, edge))
                for edge in sorted(self.tree.edges())
                if edge[0] != edge[1]
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TreeDecomposition":
        bags = {
            int(row["bag_id"]): frozenset(map(int, row["variables"]))
            for row in payload["bags"]
        }
        tree = nx.Graph()
        tree.add_nodes_from(bags)
        tree.add_edges_from(
            (int(left), int(right)) for left, right in payload["tree_edges"]
        )
        return cls(bags, tree).validate()


@dataclass
class LearnedTopology:
    graph: nx.Graph
    decomposition: TreeDecomposition
    mutual_information: np.ndarray
    screening_table: pd.DataFrame
    directed_coefficients: np.ndarray
    edge_decisions: pd.DataFrame
    screening_cache_key: str | None = None
    screening_cache_hit: bool = False
    screening_seconds: float = 0.0
    topology_build_seconds: float = 0.0


@dataclass
class TopologyScreening:
    mutual_information: np.ndarray
    candidates: pd.DataFrame
    directed_coefficients: np.ndarray
    cache_key: str
    cache_hit: bool = False
    screening_seconds: float = 0.0


def _screening_key(
    values: np.ndarray,
    screening: ScreeningConfig,
    graph_config: GraphConfig,
    seed: int,
) -> str:
    digest = hashlib.sha256()
    contiguous = np.ascontiguousarray(values, dtype=np.int8)
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())
    settings = {
        "screening": asdict(screening),
        "mutual_information_pseudocount": graph_config.mutual_information_pseudocount,
        "seed": int(seed) if screening.stability_selection else None,
        "cache_format": 1,
    }
    digest.update(json.dumps(settings, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def screen_training_data(
    values: np.ndarray,
    screening: ScreeningConfig,
    graph_config: GraphConfig,
    seed: int,
    cache_dir: str | Path | None = None,
) -> TopologyScreening:
    """Compute or load every width-independent, label-blind topology statistic."""
    started = perf_counter()
    values = validate_binary_matrix(values)
    key = _screening_key(values, screening, graph_config, seed)
    cache_path = (
        external_path(cache_dir) / f"{key}.npz" if cache_dir is not None else None
    )
    if cache_path is not None and cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as payload:
            matrix = payload["screening_table"]
            candidates = pd.DataFrame(matrix, columns=SCREENING_COLUMNS)
            if len(candidates):
                candidates[["source", "target"]] = candidates[
                    ["source", "target"]
                ].astype(int)
            return TopologyScreening(
                mutual_information=payload["mutual_information"],
                candidates=candidates,
                directed_coefficients=payload["directed_coefficients"],
                cache_key=key,
                cache_hit=True,
                screening_seconds=perf_counter() - started,
            )

    mutual_information = binary_mutual_information(
        values, graph_config.mutual_information_pseudocount
    )
    candidates, coefficients = screen_edges(values, screening, seed)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                mutual_information=mutual_information,
                directed_coefficients=coefficients,
                screening_table=candidates.to_numpy(dtype=np.float64),
            )

        os.replace(temporary, cache_path)
    return TopologyScreening(
        mutual_information=mutual_information,
        candidates=candidates,
        directed_coefficients=coefficients,
        cache_key=key,
        cache_hit=False,
        screening_seconds=perf_counter() - started,
    )


def build_bounded_width_topology(
    screened: TopologyScreening,
    graph_config: GraphConfig,
) -> LearnedTopology:
    """Build one width-specific graph from reusable screening output."""
    started = perf_counter()
    mutual_information = screened.mutual_information
    candidates = screened.candidates
    coefficients = screened.directed_coefficients
    graph = chow_liu_tree(mutual_information)
    for left, right in graph.edges:
        graph.edges[left, right]["edge_type"] = "chow_liu"
    decomposition = TreeDecomposition.from_graph(graph)
    decisions: list[dict[str, Any]] = []
    accepted_extra = 0
    tree_edges = {tuple(sorted(edge)) for edge in graph.edges}
    for rank, row in candidates.iterrows():
        left, right = int(row["source"]), int(row["target"])
        edge = tuple(sorted((left, right)))
        if edge in tree_edges:
            accepted, reason = True, "chow_liu_edge"
        elif (
            graph_config.max_extra_edges is not None
            and accepted_extra >= graph_config.max_extra_edges
        ):
            accepted, reason = False, "edge_budget"
        else:
            accepted, reason = decomposition.try_add_edge(
                left, right, graph_config.treewidth
            )
            if accepted:
                graph.add_edge(
                    left,
                    right,
                    edge_type="screened",
                    weight=float(row["coefficient_strength"]),
                )
                accepted_extra += 1
        decisions.append(
            {
                "rank": int(rank),
                "source": left,
                "target": right,
                "selection_frequency": float(row["selection_frequency"]),
                "coefficient_strength": float(row["coefficient_strength"]),
                "accepted": bool(accepted),
                "reason": reason,
            }
        )
    decomposition.validate(graph)
    if decomposition.width > graph_config.treewidth:
        raise RuntimeError("Constructed decomposition exceeded configured treewidth")
    return LearnedTopology(
        graph=graph,
        decomposition=decomposition,
        mutual_information=mutual_information,
        screening_table=candidates,
        directed_coefficients=coefficients,
        edge_decisions=pd.DataFrame(decisions, columns=DECISION_COLUMNS),
        screening_cache_key=screened.cache_key,
        screening_cache_hit=screened.cache_hit,
        screening_seconds=screened.screening_seconds,
        topology_build_seconds=perf_counter() - started,
    )


def learn_topology(
    values: np.ndarray,
    screening: ScreeningConfig,
    graph_config: GraphConfig,
    seed: int,
    cache_dir: str | Path | None = None,
) -> LearnedTopology:
    values = validate_binary_matrix(values)
    screened = screen_training_data(
        values, screening, graph_config, seed, cache_dir=cache_dir
    )
    return build_bounded_width_topology(screened, graph_config)
