import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .diagnostics import moment_fit_summary
from .rules import select_candidates
from ..empirical.workflow import fold_representation
from ..inputs import load_inputs
from ..models.graph import learn_topology
from ..models.pair import fit_pair, fit_valid, sparse_config
from ..targets import build_targets
from ..utils.io import write_csv, write_json
from ..utils.paths import external_path
from ..utils.seeds import stable_seed


def joint_cells(labels):
    matrix = np.column_stack(labels)
    if not np.all(np.isin(matrix, [0, 1])):
        raise ValueError("Joint stratification requires aligned binary outcomes.")
    return np.unique(matrix, axis=0, return_inverse=True)[1]


def partitions(pool, cells, folds, seed):
    _, counts = np.unique(cells[pool], return_counts=True)
    if counts.min() < folds:
        raise ValueError(
            "Every populated joint cell must have at least one observation per fold."
        )
    splitter = StratifiedKFold(folds, shuffle=True, random_state=seed)
    return [
        (pool[train], pool[test]) for train, test in splitter.split(pool, cells[pool])
    ]


def evaluate_candidate(features, targets, train, test, config, seed, binary=None):
    if binary is None:
        _, representations = fold_representation(features, train, config, seed)
        binary = representations["binary_latent"]
    topology = None
    if config["method"] == "sparse":
        settings = sparse_config(config, seed)
        topology = learn_topology(
            binary[train], settings.screening, settings.graph, seed
        )
    predictions, identities, fit_scores, eligible = [], [], [], []
    for target in targets:
        pair = fit_pair(binary[train], target.labels[train], config, seed, topology)
        score = pair.score(binary[test])
        passed = fit_valid(pair.diagnostics, config["gradient_max_abs"]) and bool(
            np.isfinite(score).all()
        )
        predictions.append(
            pd.DataFrame(
                {
                    "outcome": target.key,
                    "position": test,
                    "label": target.labels[test],
                    "score": score,
                }
            )
        )
        moment_rows = []
        for class_value in [0, 1]:
            first, second = pair.moments(class_value)
            edges = np.asarray(pair.edges, dtype=int).reshape(-1, 2)
            moment_rows.extend(
                moment_fit_summary(
                    binary[test[target.labels[test] == class_value]],
                    first,
                    second,
                    edges=edges,
                )
            )
        fit_scores.append(-float(np.mean([r["standardized_rms"] for r in moment_rows])))
        eligible.append(passed)
        actual_width = (
            topology.decomposition.width if topology else config["latent_dim"] - 1
        )
        identities.append(
            {
                "actual_treewidth": actual_width,
                "edge_count": len(pair.edges),
                "parameter_count": config["latent_dim"] + len(pair.edges),
            }
        )
    return pd.concat(predictions, ignore_index=True), identities, fit_scores, eligible


def candidate_key(dim, width, method):
    return f"{method}_d{dim}_w{width}"


def candidate_grid(config):
    dimensions = [
        int(d) for d in config.get("latent_dims", [20, 40, *range(80, 481, 40)])
    ]
    widths = [int(w) for w in config.get("treewidths", [1, 2, 3, 4, 5])]
    if (
        not dimensions
        or not widths
        or len(set(dimensions)) != len(dimensions)
        or len(set(widths)) != len(widths)
        or min(dimensions) < 1
        or set(widths) - {1, 2, 3, 4, 5}
    ):
        raise ValueError(
            "Candidate dimensions and widths must be nonempty, valid, and distinct."
        )
    return [(d, w) for d in dimensions for w in widths]


def score_candidates(data, targets, pool, cells, config, scope, output):
    splits = partitions(
        pool,
        cells,
        config.get("inner_folds", 5),
        stable_seed(scope, "inner", base=config["seed"]),
    )
    metrics, all_predictions, audit = [], [], []
    for dim in dict.fromkeys(d for d, _ in candidate_grid(config)):
        representations = []
        local = {
            **config,
            "latent_dim": dim,
            "retry_max_iterations": config["max_iterations"],
        }
        for fold, (train, test) in enumerate(splits, 1):
            seed = stable_seed(scope, fold, dim, "representation", base=config["seed"])
            _, values = fold_representation(
                data.features, train, local, seed, output / "cache"
            )
            representations.append(values["binary_latent"])
            audit.append(
                {
                    "scope": scope,
                    "fold": fold,
                    "latent_dim": dim,
                    "training_positions": train,
                    "validation_positions": test,
                    "selection_positions": pool,
                }
            )
        for _, width in [item for item in candidate_grid(config) if item[0] == dim]:
            key = candidate_key(dim, width, config["method"])
            local["treewidth"] = width
            candidate_predictions, identities, fit_scores, eligible = [], [], [], []
            for fold, ((train, test), binary) in enumerate(
                zip(splits, representations), 1
            ):
                seed = stable_seed(
                    scope, fold, key, "label-blind-topology", base=config["seed"]
                )
                frame, identity, fit, passed = evaluate_candidate(
                    data.features, targets, train, test, local, seed, binary
                )
                frame["fold"] = fold
                frame["row_id"] = data.row_ids[frame.position.to_numpy()]
                candidate_predictions.append(frame)
                identities.extend(identity)
                fit_scores.extend(fit)
                eligible.extend(passed)
            frame = pd.concat(candidate_predictions, ignore_index=True)
            frame["candidate_key"], frame["method"] = key, config["method"]
            all_predictions.append(frame)
            for target in targets:
                outcome = frame[frame.outcome == target.key]
                fold_aucs = [
                    roc_auc_score(group.label, group.score)
                    for _, group in outcome.groupby("fold")
                ]
                metrics.append(
                    {
                        "candidate_key": key,
                        "method": config["method"],
                        "latent_dim": dim,
                        "requested_treewidth": width,
                        "actual_treewidth": max(
                            i["actual_treewidth"] for i in identities
                        ),
                        "edge_count": max(i["edge_count"] for i in identities),
                        "parameter_count": max(
                            i["parameter_count"] for i in identities
                        ),
                        "outcome": target.key,
                        "auc": roc_auc_score(outcome.label, outcome.score),
                        "auc_se": float(
                            np.std(fold_aucs, ddof=1) / np.sqrt(len(fold_aucs))
                        ),
                        "fit_score": float(np.mean(fit_scores)),
                        "fit_se": float(
                            np.std(fit_scores, ddof=1) / np.sqrt(len(fit_scores))
                        ),
                        "eligible": all(eligible),
                    }
                )
    write_json(output / scope / "training.json", audit)
    return pd.DataFrame(metrics), pd.concat(all_predictions, ignore_index=True)


def choose(metrics, predictions, config, outcomes):
    return [
        select_candidates(
            metrics,
            predictions,
            principle,
            method=config["method"],
            outcomes=tuple(outcomes),
            bootstrap_repeats=config.get("selection_bootstrap_repeats", 2000),
            seed=config["seed"],
        )
        for principle in config.get(
            "principles", ["macro-auc", "balanced-one-se", "fit-first"]
        )
    ]


def run_selection(config):
    data = load_inputs(config)
    targets = build_targets(data.metadata, config["targets"])
    if any(
        not np.array_equal(t.indices, np.arange(len(data.features))) for t in targets
    ):
        raise ValueError(
            "Selection outcomes must cover the same complete input subset."
        )
    cells = joint_cells([t.labels for t in targets])
    output = external_path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    outer_predictions, winners = [], []
    pool = np.arange(len(data.features))
    for outer_fold, (train, test) in enumerate(
        partitions(pool, cells, config["outer_folds"], config["seed"]), 1
    ):
        scope = f"outer_{outer_fold}"
        metrics, predictions = score_candidates(
            data, targets, train, cells, config, scope, output
        )
        write_csv(output / scope / "candidate_metrics.csv", metrics)
        write_csv(output / scope / "inner_predictions.csv", predictions)
        for result in choose(metrics, predictions, config, [t.key for t in targets]):
            write_csv(
                output / scope / f"ranking_{result.principle}.csv", result.ranking
            )
            winners.append(
                {
                    "scope": scope,
                    "principle": result.principle,
                    "status": result.status,
                    "winner": result.winner,
                    "reason": result.reason,
                }
            )
            if result.winner is None:
                continue
            local = {
                **config,
                "latent_dim": result.winner["latent_dim"],
                "treewidth": result.winner["requested_treewidth"],
                "retry_max_iterations": config["max_iterations"],
            }
            frame, identities, fit, valid = evaluate_candidate(
                data.features,
                targets,
                train,
                test,
                local,
                stable_seed(scope, result.winner_key, "refit", base=config["seed"]),
            )
            frame["principle"], frame["outer_fold"] = result.principle, outer_fold
            frame["fit_valid"] = all(valid)
            frame["row_id"] = data.row_ids[frame.position.to_numpy()]
            outer_predictions.append(frame)
            write_json(
                output / scope / f"refit_{result.principle}.json",
                {
                    "training_positions": train,
                    "test_positions": test,
                    "identities": identities,
                    "fit_scores": fit,
                    "fit_valid": valid,
                },
            )
    metrics, predictions = score_candidates(
        data, targets, pool, cells, config, "full_selection", output
    )
    write_csv(output / "full_candidate_metrics.csv", metrics)
    write_csv(output / "full_inner_predictions.csv", predictions)
    for result in choose(metrics, predictions, config, [t.key for t in targets]):
        write_csv(output / f"full_ranking_{result.principle}.csv", result.ranking)
        winners.append(
            {
                "scope": "full_selection",
                "principle": result.principle,
                "status": result.status,
                "winner": result.winner,
                "reason": result.reason,
            }
        )
    frame = (
        pd.concat(outer_predictions, ignore_index=True)
        if outer_predictions
        else pd.DataFrame(columns=["principle", "outcome", "label", "score"])
    )
    write_csv(output / "outer_predictions.csv", frame)
    summary = pd.DataFrame(
        [
            {
                "principle": principle,
                "outcome": outcome,
                "auc": roc_auc_score(group.label, group.score)
                if group.fit_valid.all()
                else np.nan,
                "fit_valid": bool(group.fit_valid.all()),
                "scope": "pooled outer predictions",
            }
            for (principle, outcome), group in frame.groupby(["principle", "outcome"])
        ]
    )
    write_csv(output / "summary.csv", summary)
    write_json(output / "selection.json", winners)
    return summary
