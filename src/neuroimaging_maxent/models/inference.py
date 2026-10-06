"""Exact log-space inference on a bounded-width tree decomposition."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable

import networkx as nx
import numpy as np
from scipy.special import logsumexp

from .graph import TreeDecomposition

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:
    njit = None
    NUMBA_AVAILABLE = False


def _states(variables: tuple[int, ...]) -> np.ndarray:
    if not variables:
        return np.zeros((1, 0), dtype=np.int8)
    return (
        np.indices((2,) * len(variables), dtype=np.int8).reshape(len(variables), -1).T
    )


def _align(
    values: np.ndarray, scope: tuple[int, ...], target: tuple[int, ...]
) -> np.ndarray:
    """Broadcast a factor with ordered scope axes into ordered target axes."""
    if not scope:
        return np.asarray(values).reshape((1,) * len(target))
    order = [scope.index(variable) for variable in target if variable in scope]
    aligned = (
        np.transpose(values, order) if order != list(range(len(scope))) else values
    )
    shape = [2 if variable in scope else 1 for variable in target]
    return aligned.reshape(shape)


def _marginalize_log(
    log_values: np.ndarray,
    variables: tuple[int, ...],
    keep: tuple[int, ...],
) -> np.ndarray:
    axes = tuple(
        index for index, variable in enumerate(variables) if variable not in keep
    )
    result = logsumexp(log_values, axis=axes) if axes else log_values.copy()
    remaining = tuple(variable for variable in variables if variable in keep)
    if remaining != keep and keep:
        result = np.transpose(result, [remaining.index(variable) for variable in keep])
    return np.asarray(result)


def _group_logsumexp_numpy(
    values: np.ndarray, groups: np.ndarray, n_groups: int
) -> np.ndarray:
    maximum = np.full(n_groups, -np.inf, dtype=np.float64)
    np.maximum.at(maximum, groups, values)
    totals = np.zeros(n_groups, dtype=np.float64)
    np.add.at(totals, groups, np.exp(values - maximum[groups]))
    return maximum + np.log(totals)


if NUMBA_AVAILABLE:

    @njit(cache=False, nogil=True)
    def _group_logsumexp_numba(
        values: np.ndarray, groups: np.ndarray, n_groups: int
    ) -> np.ndarray:
        maximum = np.full(n_groups, -np.inf, dtype=np.float64)
        for index in range(values.size):
            group = groups[index]
            if values[index] > maximum[group]:
                maximum[group] = values[index]
        totals = np.zeros(n_groups, dtype=np.float64)
        for index in range(values.size):
            group = groups[index]
            totals[group] += np.exp(values[index] - maximum[group])
        return maximum + np.log(totals)


@dataclass
class InferenceResult:
    log_z: float
    clique_log_beliefs: dict[int, np.ndarray]
    clique_variables: dict[int, tuple[int, ...]]

    def marginal(self, variables: Iterable[int]) -> np.ndarray:
        scope = tuple(variables)
        candidates = [
            (len(clique), bag_id)
            for bag_id, clique in self.clique_variables.items()
            if set(scope).issubset(clique)
        ]
        if not candidates:
            raise KeyError(f"No clique contains variables {scope}")
        return self.marginal_from_bag(scope, min(candidates)[1])

    def marginal_from_bag(self, variables: Iterable[int], bag_id: int) -> np.ndarray:
        scope = tuple(variables)
        clique = self.clique_variables[bag_id]
        if not set(scope).issubset(clique):
            raise KeyError(f"Bag {bag_id} does not contain variables {scope}")
        log_marginal = _marginalize_log(self.clique_log_beliefs[bag_id], clique, scope)
        return np.exp(log_marginal - logsumexp(log_marginal))


@dataclass(frozen=True)
class _DirectedMessagePlan:
    source_groups: np.ndarray
    target_groups: np.ndarray
    n_groups: int


@dataclass(frozen=True)
class _SamplingPlan:
    separator: tuple[int, ...]
    exclusive: tuple[int, ...]
    state_rows_by_separator: tuple[np.ndarray, ...]
    exclusive_states: np.ndarray
    separator_weights: np.ndarray


class JunctionTreeInference:
    """Calibrate binary unary/pairwise factors on an existing decomposition."""

    def __init__(
        self,
        dimension: int,
        edges: list[tuple[int, int]],
        decomposition: TreeDecomposition,
        backend: str = "auto",
    ) -> None:
        if backend not in {"auto", "numba", "numpy"}:
            raise ValueError("backend must be auto, numba, or numpy")
        if backend == "numba" and not NUMBA_AVAILABLE:
            raise ImportError("Numba backend requested but numba is not installed")
        self.requested_backend = backend
        self.backend = (
            "numba"
            if backend == "numba" or (backend == "auto" and NUMBA_AVAILABLE)
            else "numpy"
        )
        self.dimension = int(dimension)
        self.edges = [tuple(sorted(map(int, edge))) for edge in edges]
        self.decomposition = decomposition
        self.cliques = {
            bag_id: tuple(sorted(bag)) for bag_id, bag in decomposition.bags.items()
        }
        self.unary_bags = {
            variable: decomposition.find_bag([variable])
            for variable in range(dimension)
        }
        self.edge_bags = {edge: decomposition.find_bag(edge) for edge in self.edges}
        self.root = min(self.cliques)
        self.parent: dict[int, int | None] = {self.root: None}
        self.order = [self.root]
        for parent, child in nx.bfs_edges(decomposition.tree, self.root):
            self.parent[int(child)] = int(parent)
            self.order.append(int(child))
        self.children = {
            node: tuple(child for child in self.order if self.parent.get(child) == node)
            for node in self.order
        }
        self._state_tables = {
            bag: _states(clique) for bag, clique in self.cliques.items()
        }
        self._parameter_indices: dict[int, np.ndarray] = {}
        self._design: dict[int, np.ndarray] = {}
        self._message_plans: dict[tuple[int, int], _DirectedMessagePlan] = {}
        self._sampling_plans: dict[int, _SamplingPlan] = {}
        started = perf_counter()
        self._compile_execution_plan()
        self.plan_build_seconds = perf_counter() - started
        self.numba_initialization_seconds = 0.0
        if self.backend == "numba":
            started = perf_counter()
            try:
                _group_logsumexp_numba(
                    np.asarray([0.0, 1.0]), np.asarray([0, 0], dtype=np.int64), 1
                )
            except Exception:
                if self.requested_backend == "numba":
                    raise
                self.backend = "numpy"
            self.numba_initialization_seconds = perf_counter() - started

    @staticmethod
    def _codes(states: np.ndarray, positions: list[int]) -> np.ndarray:
        if not positions:
            return np.zeros(len(states), dtype=np.int64)
        weights = 1 << np.arange(len(positions) - 1, -1, -1, dtype=np.int64)
        return states[:, positions].astype(np.int64) @ weights

    def _compile_execution_plan(self) -> None:
        assigned: dict[int, list[tuple[int, np.ndarray]]] = {
            bag: [] for bag in self.cliques
        }
        for variable in range(self.dimension):
            bag = self.unary_bags[variable]
            position = self.cliques[bag].index(variable)
            feature = self._state_tables[bag][:, position].astype(np.float64)
            assigned[bag].append((variable, feature))
        for edge_index, edge in enumerate(self.edges):
            bag = self.edge_bags[edge]
            left = self.cliques[bag].index(edge[0])
            right = self.cliques[bag].index(edge[1])
            feature = (
                self._state_tables[bag][:, left] * self._state_tables[bag][:, right]
            )
            assigned[bag].append(
                (self.dimension + edge_index, feature.astype(np.float64))
            )
        for bag, rows in assigned.items():
            self._parameter_indices[bag] = np.asarray(
                [item[0] for item in rows], dtype=np.int64
            )
            self._design[bag] = (
                np.column_stack([item[1] for item in rows])
                if rows
                else np.empty((len(self._state_tables[bag]), 0), dtype=np.float64)
            )

        for left, right in self.decomposition.tree.edges:
            for source, target in ((int(left), int(right)), (int(right), int(left))):
                separator = tuple(
                    sorted(set(self.cliques[source]) & set(self.cliques[target]))
                )
                source_positions = [
                    self.cliques[source].index(variable) for variable in separator
                ]
                target_positions = [
                    self.cliques[target].index(variable) for variable in separator
                ]
                self._message_plans[(source, target)] = _DirectedMessagePlan(
                    source_groups=self._codes(
                        self._state_tables[source], source_positions
                    ),
                    target_groups=self._codes(
                        self._state_tables[target], target_positions
                    ),
                    n_groups=1 << len(separator),
                )

        for child in self.order[1:]:
            parent = self.parent[child]
            assert parent is not None
            clique = self.cliques[child]
            separator = tuple(sorted(set(clique) & set(self.cliques[parent])))
            exclusive = tuple(
                variable for variable in clique if variable not in separator
            )
            positions = [clique.index(variable) for variable in separator]
            groups = self._codes(self._state_tables[child], positions)
            rows = tuple(
                np.flatnonzero(groups == code) for code in range(1 << len(separator))
            )
            self._sampling_plans[child] = _SamplingPlan(
                separator=separator,
                exclusive=exclusive,
                state_rows_by_separator=rows,
                exclusive_states=_states(exclusive),
                separator_weights=(
                    1 << np.arange(len(separator) - 1, -1, -1, dtype=np.int64)
                ),
            )

    def _reduce(self, values: np.ndarray, plan: _DirectedMessagePlan) -> np.ndarray:
        if self.backend == "numba":
            return _group_logsumexp_numba(values, plan.source_groups, plan.n_groups)
        return _group_logsumexp_numpy(values, plan.source_groups, plan.n_groups)

    def _local_potentials(self, parameters: np.ndarray) -> dict[int, np.ndarray]:
        return {
            bag: self._design[bag] @ parameters[self._parameter_indices[bag]]
            for bag in self.cliques
        }

    def calibrate(self, h: np.ndarray, couplings: np.ndarray) -> InferenceResult:
        h = np.asarray(h, dtype=np.float64)
        couplings = np.asarray(couplings, dtype=np.float64)
        if h.shape != (self.dimension,) or couplings.shape != (len(self.edges),):
            raise ValueError("Parameter shapes do not match the inference graph")
        local = self._local_potentials(np.concatenate((h, couplings)))
        messages: dict[tuple[int, int], np.ndarray] = {}

        for node in reversed(self.order[1:]):
            parent = self.parent[node]
            assert parent is not None
            total = local[node].copy()
            for child in self.children[node]:
                plan = self._message_plans[(child, node)]
                total += messages[(child, node)][plan.target_groups]
            plan = self._message_plans[(node, parent)]
            messages[(node, parent)] = self._reduce(total, plan)

        beliefs: dict[int, np.ndarray] = {}
        for node in self.order:
            total = local[node].copy()
            parent = self.parent[node]
            if parent is not None:
                plan = self._message_plans[(parent, node)]
                total += messages[(parent, node)][plan.target_groups]
            for child in self.children[node]:
                plan = self._message_plans[(child, node)]
                total += messages[(child, node)][plan.target_groups]
            beliefs[node] = total
            for child in self.children[node]:
                incoming = self._message_plans[(child, node)]
                excluding_child = (
                    total - messages[(child, node)][incoming.target_groups]
                )
                outgoing = self._message_plans[(node, child)]
                messages[(node, child)] = self._reduce(excluding_child, outgoing)

        log_z = float(logsumexp(beliefs[self.root]))
        shaped = {
            bag: belief.reshape((2,) * len(self.cliques[bag]))
            for bag, belief in beliefs.items()
        }
        return InferenceResult(log_z, shaped, self.cliques)

    def expectations(self, result: InferenceResult) -> tuple[np.ndarray, np.ndarray]:
        expected = np.empty(self.dimension + len(self.edges), dtype=np.float64)
        for bag in self.cliques:
            indices = self._parameter_indices[bag]
            if not len(indices):
                continue
            log_belief = result.clique_log_beliefs[bag].reshape(-1)
            probability = np.exp(log_belief - logsumexp(log_belief))
            expected[indices] = self._design[bag].T @ probability
        return expected[: self.dimension], expected[self.dimension :]

    def sample(self, result: InferenceResult, n_samples: int, seed: int) -> np.ndarray:
        if n_samples < 1:
            raise ValueError("n_samples must be positive")
        rng = np.random.default_rng(seed)
        samples = np.full((n_samples, self.dimension), -1, dtype=np.int8)
        root_variables = self.cliques[self.root]
        root_states = self._state_tables[self.root]
        root_log = result.clique_log_beliefs[self.root].reshape(-1)
        probability = np.exp(root_log - logsumexp(root_log))
        choices = rng.choice(len(root_states), size=n_samples, p=probability)
        samples[:, root_variables] = root_states[choices]

        for child in self.order[1:]:
            plan = self._sampling_plans[child]
            if not plan.exclusive:
                continue
            clique_log = result.clique_log_beliefs[child].reshape(-1)
            separator_codes = (
                samples[:, plan.separator].astype(np.int64) @ plan.separator_weights
                if plan.separator
                else np.zeros(n_samples, dtype=np.int64)
            )
            for code, state_rows in enumerate(plan.state_rows_by_separator):
                row_indices = np.flatnonzero(separator_codes == code)
                if not len(row_indices):
                    continue
                probability_log = clique_log[state_rows]
                probability = np.exp(probability_log - logsumexp(probability_log))
                draws = rng.choice(
                    len(plan.exclusive_states), size=len(row_indices), p=probability
                )
                samples[np.ix_(row_indices, plan.exclusive)] = plan.exclusive_states[
                    draws
                ]
        if np.any(samples < 0):
            raise RuntimeError("Junction-tree sampling left variables unassigned")
        return samples
