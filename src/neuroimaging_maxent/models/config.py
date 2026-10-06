from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any


@dataclass
class ScreeningConfig:
    logistic_c: float = 0.1
    solver: str = "liblinear"
    max_iter: int = 5_000
    tolerance: float = 1e-4
    coefficient_tolerance: float = 1e-8
    symmetrization: str = "or"
    stability_selection: bool = False
    stability_repetitions: int = 50
    stability_subsample_fraction: float = 0.5
    stability_threshold: float = 0.8
    max_candidates_per_node: int | None = None
    n_jobs: int = 1


@dataclass
class GraphConfig:
    treewidth: int = 1
    mutual_information_pseudocount: float = 0.5
    max_extra_edges: int | None = None


@dataclass
class InferenceConfig:
    backend: str = "auto"
    score_batch_size: int = 10_000


@dataclass
class FitConfig:
    l2_penalty: float = 1e-6
    max_iterations: int = 500
    gradient_tolerance: float = 1e-7
    function_tolerance: float = 1e-10
    max_line_search_steps: int = 30
    initialize_from_independent: bool = True
    warm_start_widths: bool = True


@dataclass
class EvaluationConfig:
    n_splits: int = 5
    heldout_fit: bool = True
    auc_samples_per_class: int = 20_000
    auc_repeats: int = 5
    save_auc_samples: bool = True


@dataclass
class SparseTreewidthConfig:
    screening: ScreeningConfig = field(default_factory=ScreeningConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    fit: FitConfig = field(default_factory=FitConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    random_seed: int = 42

    def validate(self) -> "SparseTreewidthConfig":
        screen = self.screening
        graph = self.graph
        inference = self.inference
        fit = self.fit
        evaluation = self.evaluation
        if screen.logistic_c <= 0:
            raise ValueError("screening.logistic_c must be positive")
        if screen.solver not in {"liblinear", "saga"}:
            raise ValueError("screening.solver must be liblinear or saga")
        if screen.max_iter < 1 or screen.tolerance <= 0:
            raise ValueError("screening iterations and tolerance must be positive")
        if screen.coefficient_tolerance < 0:
            raise ValueError("screening.coefficient_tolerance cannot be negative")
        if screen.symmetrization not in {"or", "and"}:
            raise ValueError("screening.symmetrization must be or or and")
        if screen.stability_repetitions < 1:
            raise ValueError("screening.stability_repetitions must be positive")
        if not 0 < screen.stability_subsample_fraction <= 1:
            raise ValueError("stability_subsample_fraction must lie in (0, 1]")
        if not 0 <= screen.stability_threshold <= 1:
            raise ValueError("stability_threshold must lie in [0, 1]")
        if (
            screen.max_candidates_per_node is not None
            and screen.max_candidates_per_node < 1
        ):
            raise ValueError("max_candidates_per_node must be positive or null")
        if screen.n_jobs == 0:
            raise ValueError("screening.n_jobs cannot be zero")
        if graph.treewidth not in {1, 2, 3, 4, 5}:
            raise ValueError("graph.treewidth must be one of 1, 2, 3, 4, 5")
        if graph.mutual_information_pseudocount < 0:
            raise ValueError("mutual_information_pseudocount cannot be negative")
        if graph.max_extra_edges is not None and graph.max_extra_edges < 0:
            raise ValueError("max_extra_edges must be nonnegative or null")
        if inference.backend not in {"auto", "numba", "numpy"}:
            raise ValueError("inference.backend must be auto, numba, or numpy")
        if inference.score_batch_size < 1:
            raise ValueError("inference.score_batch_size must be positive")
        if fit.l2_penalty < 0 or fit.max_iterations < 1:
            raise ValueError("fit penalty must be nonnegative and iterations positive")
        if fit.gradient_tolerance <= 0 or fit.function_tolerance <= 0:
            raise ValueError("fit tolerances must be positive")
        if evaluation.n_splits < 2:
            raise ValueError("evaluation.n_splits must be at least two")
        if evaluation.auc_samples_per_class < 2 or evaluation.auc_repeats < 1:
            raise ValueError("AUC sample count and repeats must be positive")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge(instance: Any, values: dict[str, Any], prefix: str = "") -> None:
    allowed = {item.name for item in fields(instance)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(
            f"Unknown keys in {prefix or type(instance).__name__}: {sorted(unknown)}"
        )
    for key, value in values.items():
        current = getattr(instance, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise TypeError(f"{prefix}{key} must be a JSON object")
            _merge(current, value, f"{prefix}{key}.")
        else:
            setattr(instance, key, value)
