"""Exact-likelihood sparse bounded-treewidth Ising estimators."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .config import FitConfig, SparseTreewidthConfig
from .graph import LearnedTopology, learn_topology, validate_binary_matrix
from .inference import InferenceResult, JunctionTreeInference


class SparseTreewidthIsingModel:
    """Pairwise 0/1 MaxEnt model fitted on a supplied fixed graph."""

    def __init__(
        self,
        topology: LearnedTopology,
        fit_config: FitConfig,
        inference_backend: str = "auto",
        score_batch_size: int = 10_000,
        inference_plan: JunctionTreeInference | None = None,
    ) -> None:
        self.topology = topology
        self.fit_config = fit_config
        self.dimension = len(topology.graph)
        self.edges = sorted(
            tuple(sorted(map(int, edge))) for edge in topology.graph.edges
        )
        self.inference = inference_plan or JunctionTreeInference(
            self.dimension,
            self.edges,
            topology.decomposition,
            backend=inference_backend,
        )
        if (
            self.inference.dimension != self.dimension
            or self.inference.edges != self.edges
        ):
            raise ValueError("Shared inference plan does not match the topology")
        self.score_batch_size = int(score_batch_size)
        self._edge_left = np.asarray([edge[0] for edge in self.edges], dtype=np.int64)
        self._edge_right = np.asarray([edge[1] for edge in self.edges], dtype=np.int64)
        self.h = np.zeros(self.dimension, dtype=np.float64)
        self.couplings = np.zeros(len(self.edges), dtype=np.float64)
        self.result: InferenceResult | None = None
        self.optimization_history = pd.DataFrame()
        self.diagnostics: dict[str, Any] = {}
        self.empirical_first = np.empty(0)
        self.empirical_second = np.empty(0)
        self.model_first = np.empty(0)
        self.model_second = np.empty(0)
        self._objective_calls = 0
        self._objective_wall_seconds = 0.0

    def _split(self, parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return parameters[: self.dimension], parameters[self.dimension :]

    def _objective(
        self,
        parameters: np.ndarray,
        empirical: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        started = perf_counter()
        h, couplings = self._split(parameters)
        result = self.inference.calibrate(h, couplings)
        first, second = self.inference.expectations(result)
        model = np.concatenate((first, second))
        raw_nll = result.log_z - float(parameters @ empirical)
        penalty = 0.5 * self.fit_config.l2_penalty * float(parameters @ parameters)
        gradient = model - empirical + self.fit_config.l2_penalty * parameters
        self._last_parameters = parameters.copy()
        self._last_evaluation = (raw_nll, penalty, result, model, gradient)
        self._objective_calls += 1
        self._objective_wall_seconds += perf_counter() - started
        return raw_nll + penalty, gradient

    def fit(
        self,
        values: np.ndarray,
        initial_parameters: np.ndarray | None = None,
        warm_start_source: str | None = None,
    ) -> "SparseTreewidthIsingModel":
        fit_started = perf_counter()
        self._objective_calls = 0
        self._objective_wall_seconds = 0.0
        values = validate_binary_matrix(values, self.dimension)
        if len(values) < 1:
            raise ValueError("At least one training state is required")
        self.empirical_first = values.mean(axis=0, dtype=np.float64)
        self.empirical_second = np.array(
            [np.mean(values[:, left] * values[:, right]) for left, right in self.edges],
            dtype=np.float64,
        )
        empirical = np.concatenate((self.empirical_first, self.empirical_second))
        initial = np.zeros(len(empirical), dtype=np.float64)
        if initial_parameters is not None:
            initial_parameters = np.asarray(initial_parameters, dtype=np.float64)
            if initial_parameters.shape != initial.shape:
                raise ValueError("Warm-start parameter shape does not match the graph")
            initial[:] = initial_parameters
        elif self.fit_config.initialize_from_independent:
            clipped = np.clip(self.empirical_first, 1e-4, 1 - 1e-4)
            initial[: self.dimension] = np.log(clipped / (1 - clipped))
        history: list[dict[str, Any]] = []

        def callback(parameters: np.ndarray) -> None:
            if not hasattr(self, "_last_parameters") or not np.array_equal(
                parameters, self._last_parameters
            ):
                self._objective(parameters, empirical)
            raw, penalty, _, _, gradient = self._last_evaluation
            objective = raw + penalty
            history.append(
                {
                    "iteration": len(history) + 1,
                    "objective_per_sample": float(objective),
                    "negative_log_likelihood_per_sample": float(raw),
                    "l2_penalty": float(penalty),
                    "gradient_max_abs": float(np.max(np.abs(gradient))),
                }
            )

        optimized = minimize(
            fun=lambda parameters: self._objective(parameters, empirical),
            x0=initial,
            method="L-BFGS-B",
            jac=True,
            callback=callback,
            options={
                "maxiter": self.fit_config.max_iterations,
                "gtol": self.fit_config.gradient_tolerance,
                "ftol": self.fit_config.function_tolerance,
                "maxls": self.fit_config.max_line_search_steps,
            },
        )
        self.h, self.couplings = (part.copy() for part in self._split(optimized.x))
        self.result = self.inference.calibrate(self.h, self.couplings)
        self.model_first, self.model_second = self.inference.expectations(self.result)
        final_nll = self.result.log_z - float(optimized.x @ empirical)
        final_penalty = (
            0.5 * self.fit_config.l2_penalty * float(optimized.x @ optimized.x)
        )
        final_gradient = (
            np.concatenate((self.model_first, self.model_second))
            - empirical
            + self.fit_config.l2_penalty * optimized.x
        )
        self.optimization_history = pd.DataFrame(
            history,
            columns=[
                "iteration",
                "objective_per_sample",
                "negative_log_likelihood_per_sample",
                "l2_penalty",
                "gradient_max_abs",
            ],
        )
        self.diagnostics = {
            "success": bool(optimized.success),
            "status": int(optimized.status),
            "message": str(optimized.message),
            "n_iterations": int(optimized.nit),
            "n_function_evaluations": int(optimized.nfev),
            "negative_log_likelihood_per_sample": float(final_nll),
            "negative_log_likelihood_total": float(len(values) * final_nll),
            "l2_penalty": float(final_penalty),
            "objective_per_sample": float(final_nll + final_penalty),
            "log_partition_function": self.result.log_z,
            "gradient_max_abs": float(np.max(np.abs(final_gradient))),
            "first_moment_max_abs_error": float(
                np.max(np.abs(self.model_first - self.empirical_first))
            ),
            "edge_moment_max_abs_error": float(
                np.max(np.abs(self.model_second - self.empirical_second))
            )
            if len(self.edges)
            else 0.0,
            "n_samples": int(len(values)),
            "dimension": self.dimension,
            "n_edges": len(self.edges),
            "treewidth": self.topology.decomposition.width,
            "inference_backend": self.inference.backend,
            "inference_plan_build_seconds": self.inference.plan_build_seconds,
            "numba_initialization_seconds": self.inference.numba_initialization_seconds,
            "fit_wall_seconds": perf_counter() - fit_started,
            "warm_started": initial_parameters is not None,
            "warm_start_source": warm_start_source,
            "screening_cache_hit": self.topology.screening_cache_hit,
            "screening_cache_key": self.topology.screening_cache_key,
            "screening_seconds": self.topology.screening_seconds,
            "topology_build_seconds": self.topology.topology_build_seconds,
            "objective_calls": self._objective_calls,
            "objective_wall_seconds": self._objective_wall_seconds,
        }
        return self

    def _require_fit(self) -> InferenceResult:
        if self.result is None:
            raise RuntimeError("Model has not been fitted")
        return self.result

    @property
    def j_matrix(self) -> np.ndarray:
        output = np.zeros((self.dimension, self.dimension), dtype=np.float64)
        for (left, right), value in zip(self.edges, self.couplings):
            output[left, right] = output[right, left] = value
        return output

    def _prepare_scoring_values(self, values: np.ndarray) -> tuple[np.ndarray, bool]:
        values = np.asarray(values)
        single = values.ndim == 1
        if single:
            values = values[None, :]
        return validate_binary_matrix(values, self.dimension).astype(np.float64), single

    def log_unnormalized(self, values: np.ndarray) -> np.ndarray | float:
        values, single = self._prepare_scoring_values(values)
        output = np.empty(len(values), dtype=np.float64)
        for start in range(0, len(values), self.score_batch_size):
            stop = min(start + self.score_batch_size, len(values))
            batch = values[start:stop]
            score = batch @ self.h
            if len(self.edges):
                score += (
                    batch[:, self._edge_left] * batch[:, self._edge_right]
                ) @ self.couplings
            output[start:stop] = score
        return float(output[0]) if single else output

    def energy(self, values: np.ndarray) -> np.ndarray | float:
        return -self.log_unnormalized(values)

    def log_prob(self, values: np.ndarray) -> np.ndarray | float:
        return self.log_unnormalized(values) - self._require_fit().log_z

    def sample(self, n_samples: int, seed: int) -> np.ndarray:
        return self.inference.sample(self._require_fit(), n_samples, seed)


class SharedGraphIsingClassifier:
    """Learn one pooled topology and fit two class-specific exact models."""

    def __init__(
        self,
        config: SparseTreewidthConfig | None = None,
        negative_name: str = "negative",
        positive_name: str = "positive",
    ) -> None:
        self.config = (config or SparseTreewidthConfig()).validate()
        self.negative_name = str(negative_name)
        self.positive_name = str(positive_name)
        if not self.negative_name or not self.positive_name:
            raise ValueError("Class names cannot be empty")
        self.topology: LearnedTopology | None = None
        self.negative_model: SparseTreewidthIsingModel | None = None
        self.positive_model: SparseTreewidthIsingModel | None = None

    def fit(
        self,
        values: np.ndarray,
        labels: np.ndarray,
        screening_cache_dir: str | Path | None = None,
        initial_negative: np.ndarray | None = None,
        initial_positive: np.ndarray | None = None,
        warm_start_source: str | None = None,
        learned_topology: LearnedTopology | None = None,
    ) -> "SharedGraphIsingClassifier":
        values = validate_binary_matrix(values)
        labels = np.asarray(labels)
        if labels.shape != (len(values),) or not np.all(np.isin(labels, (0, 1))):
            raise ValueError("Labels must be a binary vector aligned with values")
        if np.any(np.bincount(labels.astype(int), minlength=2) == 0):
            raise ValueError(
                "Both negative-class (0) and positive-class (1) samples are required"
            )
        if learned_topology is None:
            self.topology = learn_topology(
                values,
                self.config.screening,
                self.config.graph,
                self.config.random_seed,
                cache_dir=screening_cache_dir,
            )
        else:
            if len(learned_topology.graph) != values.shape[1]:
                raise ValueError(
                    "Supplied topology dimension does not match the values"
                )
            if learned_topology.decomposition.width > self.config.graph.treewidth:
                raise ValueError("Supplied topology exceeds the configured treewidth")
            self.topology = learned_topology
        model_kwargs = {
            "inference_backend": self.config.inference.backend,
            "score_batch_size": self.config.inference.score_batch_size,
        }

        shared_plan = JunctionTreeInference(
            len(self.topology.graph),
            sorted(tuple(sorted(map(int, edge))) for edge in self.topology.graph.edges),
            self.topology.decomposition,
            backend=self.config.inference.backend,
        )
        model_kwargs["inference_plan"] = shared_plan
        self.negative_model = SparseTreewidthIsingModel(
            self.topology, self.config.fit, **model_kwargs
        ).fit(values[labels == 0], initial_negative, warm_start_source)
        self.positive_model = SparseTreewidthIsingModel(
            self.topology, self.config.fit, **model_kwargs
        ).fit(values[labels == 1], initial_positive, warm_start_source)
        return self

    def _models(self) -> tuple[SparseTreewidthIsingModel, SparseTreewidthIsingModel]:
        if self.negative_model is None or self.positive_model is None:
            raise RuntimeError("Classifier has not been fitted")
        return self.negative_model, self.positive_model

    def log_prob(self, values: np.ndarray, class_value: int) -> np.ndarray | float:
        negative, positive = self._models()
        if class_value not in (0, 1):
            raise ValueError("class_value must be 0 (negative) or 1 (positive)")
        return (negative if class_value == 0 else positive).log_prob(values)

    def log_likelihood_ratio(self, values: np.ndarray) -> np.ndarray | float:
        negative, positive = self._models()
        return positive.log_prob(values) - negative.log_prob(values)

    def sample(self, class_value: int, n_samples: int, seed: int) -> np.ndarray:
        negative, positive = self._models()
        if class_value not in (0, 1):
            raise ValueError("class_value must be 0 (negative) or 1 (positive)")
        return (negative if class_value == 0 else positive).sample(n_samples, seed)
