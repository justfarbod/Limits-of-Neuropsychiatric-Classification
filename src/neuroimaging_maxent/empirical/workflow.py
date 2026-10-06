import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .diagnostics import activity_count_metrics, marginal_agreement_metrics
from .discriminative import _fit_fold, bootstrap_repeated_fold_auc
from ..autoencoders.metrics import reconstruction_metrics
from ..autoencoders.representation import fit_representation, load_representation
from ..inputs import load_inputs
from ..models.pair import fit_pair, fit_valid
from ..targets import build_targets
from ..utils.io import fingerprint, write_csv, write_json
from ..utils.paths import external_path
from ..utils.seeds import stable_seed


def fold_representation(features, train, config, seed, cache_dir=None):
    settings = {
        key: config[key]
        for key in [
            "latent_dim",
            "autoencoder_epochs",
            "autoencoder_batch_size",
            "device",
        ]
    }
    identity = fingerprint(features, train, settings={**settings, "seed": seed})
    if cache_dir is not None:
        directory = external_path(cache_dir) / identity
        directory.mkdir(parents=True, exist_ok=True)
        checkpoint = directory / "autoencoder.pt"
        if checkpoint.is_file():
            representation = load_representation(checkpoint, config["device"])
            representation.training_indices = np.asarray(train).copy()
        else:
            representation = fit_representation(features, train, config, seed)
            representation.save(checkpoint)
            write_json(
                directory / "training.json",
                {
                    "identity": identity,
                    "training_positions": train,
                    "settings": settings,
                },
            )
    else:
        representation = fit_representation(features, train, config, seed)
    continuous, binary = representation.encode(features)
    decoded = representation.decode(binary)
    return representation, {
        "original_fc": features,
        "continuous_latent": continuous,
        "binary_latent": binary,
        "decoded_fc": decoded,
    }


def evaluate_diagnostics(pair, binary, labels, config, seed):
    rows, arrays = [], []
    upper = np.triu_indices(binary.shape[1], 1)
    for class_value in [0, 1]:
        empirical = binary[labels == class_value].astype(float)
        sampled = pair.sample(
            class_value, config["ppc_samples_per_class"], stable_seed(seed, class_value)
        )
        first = empirical.mean(axis=0)
        second = (empirical.T @ empirical / len(empirical))[upper]
        model_first = sampled.mean(axis=0)
        model_second = (sampled.astype(float).T @ sampled / len(sampled))[upper]
        for kind, observed, modeled in [
            ("unary", first, model_first),
            ("pairwise", second, model_second),
        ]:
            rows.append(
                {
                    "class_value": class_value,
                    "moment_type": kind,
                    **marginal_agreement_metrics(observed, modeled, len(empirical)),
                }
            )
            arrays.extend(
                {
                    "class_value": class_value,
                    "moment_type": kind,
                    "moment_index": i,
                    "observed": x,
                    "modeled": y,
                }
                for i, (x, y) in enumerate(zip(observed, modeled))
            )
        unique, counts = np.unique(empirical, axis=0, return_counts=True)
        frequencies = counts / counts.sum()
        nll = -float(np.mean(pair.log_prob(empirical, class_value)))
        divergence = float(
            np.sum(
                frequencies * (np.log(frequencies) - pair.log_prob(unique, class_value))
            )
        )
        rows.append(
            {
                "class_value": class_value,
                "moment_type": "activity",
                "heldout_nll": nll,
                "empirical_joint_kl": divergence,
                **activity_count_metrics(
                    empirical.sum(axis=1), sampled.sum(axis=1), binary.shape[1]
                ),
            }
        )
    return rows, arrays


def run_empirical(config):
    data = load_inputs(config)
    targets = build_targets(data.metadata, config["targets"])
    output = external_path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    folds, predictions, summary, diagnostics, marginals, reconstructions = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for target in targets:
        features = data.features[target.indices]
        labels = target.labels
        if np.bincount(labels, minlength=2).min() < config["outer_folds"]:
            raise ValueError(
                "An outcome has too few observations per class for the requested folds."
            )
        methods = ["maxent", *config["representations"]]
        scores = {
            method: np.empty((len(config["outer_seeds"]), len(labels)))
            for method in methods
        }
        fold_ids = np.empty_like(scores["maxent"], dtype=int)
        for seed_index, seed in enumerate(config["outer_seeds"]):
            splitter = StratifiedKFold(
                config["outer_folds"], shuffle=True, random_state=seed
            )
            for fold, (train, test) in enumerate(splitter.split(features, labels), 1):
                local_seed = stable_seed(
                    config["base_seed"], "fold_autoencoder", target.key, seed, fold
                )
                representation, representations = fold_representation(
                    features, train, config, local_seed, output / "cache"
                )
                for kind, decoded in [
                    ("binary", representations["decoded_fc"]),
                    (
                        "continuous",
                        representation.decode_continuous(
                            representations["continuous_latent"]
                        ),
                    ),
                ]:
                    reconstructions.append(
                        {
                            "outcome": target.key,
                            "outer_seed": seed,
                            "fold": fold,
                            "representation": kind,
                            **reconstruction_metrics(features[test], decoded[test]),
                        }
                    )
                binary = representations["binary_latent"]
                pair = fit_pair(
                    binary[train],
                    labels[train],
                    config,
                    stable_seed(config["base_seed"], "maxent", target.key, seed, fold),
                )
                valid = (
                    fit_valid(pair.diagnostics, config["gradient_max_abs"])
                    if pair.method == "sparse"
                    else all(d["success"] for d in pair.diagnostics)
                )
                score = pair.score(binary[test])
                valid = valid and bool(np.isfinite(score).all())
                prior = float(labels[train].mean())
                probabilities = expit(score + np.log(prior / (1 - prior)))
                row = {
                    "outcome": target.key,
                    "method": "maxent",
                    "outer_seed": seed,
                    "fold": fold,
                    "auc": roc_auc_score(labels[test], score),
                    "fit_valid": valid,
                    "brier_score": brier_score_loss(labels[test], probabilities),
                    "log_loss": log_loss(labels[test], probabilities, labels=[0, 1]),
                    "n_train": len(train),
                    "n_test": len(test),
                    "retry_count": max(0, len(pair.attempts) - 1),
                }
                folds.append(row)
                scores["maxent"][seed_index, test] = score
                fold_ids[seed_index, test] = fold
                write_json(
                    output
                    / "fits"
                    / target.key
                    / f"repeat_{seed_index}_fold_{fold}.json",
                    {
                        "training_positions": train,
                        "test_positions": test,
                        "diagnostics": pair.diagnostics,
                        "attempts": pair.attempts,
                    },
                )
                for method in config["representations"]:
                    row, _, score = _fit_fold(
                        representations[method],
                        labels,
                        train,
                        test,
                        seed,
                        fold,
                        config["svc_c"],
                    )
                    folds.append({"outcome": target.key, "method": method, **row})
                    scores[method][seed_index, test] = score
                diagnostic_rows, marginal_rows = evaluate_diagnostics(
                    pair, binary[test], labels[test], config, local_seed
                )
                for row in diagnostic_rows:
                    diagnostics.append(
                        {"outcome": target.key, "outer_seed": seed, "fold": fold, **row}
                    )
                for row in marginal_rows:
                    marginals.append(
                        {"outcome": target.key, "outer_seed": seed, "fold": fold, **row}
                    )
        for method in methods:
            draws = bootstrap_repeated_fold_auc(
                labels,
                scores[method],
                fold_ids,
                config["bootstrap_repeats"],
                stable_seed(config["bootstrap_seed"], method, target.key),
            )
            lower, upper = np.quantile(draws, [0.025, 0.975])
            mean = np.mean(
                [
                    r["auc"]
                    for r in folds
                    if r["outcome"] == target.key and r["method"] == method
                ]
            )
            summary.append(
                {
                    "outcome": target.key,
                    "method": method,
                    "auc": mean,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "n": len(labels),
                    "interval": "class-stratified participant bootstrap of mean repeated fold AUC",
                }
            )
            np.savez_compressed(
                external_path(output / f"bootstrap_{target.key}_{method}.npz"),
                auc=draws,
            )
            for seed_index, seed in enumerate(config["outer_seeds"]):
                predictions.extend(
                    {
                        "outcome": target.key,
                        "method": method,
                        "outer_seed": seed,
                        "fold": int(fold_ids[seed_index, i]),
                        "row_id": data.row_ids[target.indices[i]],
                        "label": int(labels[i]),
                        "score": float(scores[method][seed_index, i]),
                    }
                    for i in range(len(labels))
                )
    if any(not row["fit_valid"] for row in folds if row["method"] == "maxent"):
        write_csv(output / "failed_folds.csv", pd.DataFrame(folds))
        write_json(
            output / "status.json",
            {
                "status": "failed",
                "reason": "Final numerical fits did not all meet the unchanged diagnostic criteria.",
            },
        )
        raise RuntimeError(
            "Final numerical fits did not all pass; no complete empirical summary was exported."
        )
    tables = {
        "summary": summary,
        "folds": folds,
        "predictions": predictions,
        "diagnostics": diagnostics,
        "marginals": marginals,
        "reconstruction": reconstructions,
    }
    for name, rows in tables.items():
        write_csv(output / f"{name}.csv", pd.DataFrame(rows))
    profile_rows = []
    frame = pd.DataFrame(summary).pivot(index="outcome", columns="method", values="auc")
    for method in config["representations"]:
        x, y = frame["maxent"], frame[method]
        profile_rows.append(
            {
                "method": method,
                "pearson_r": float(x.corr(y)) if len(x) > 1 else np.nan,
                "spearman_rho": float(x.corr(y, method="spearman"))
                if len(x) > 1
                else np.nan,
                "mean_auc_difference": float((y - x).mean()),
                "mean_above_chance_attainment": float(
                    ((y - 0.5) / (x - 0.5)).replace([np.inf, -np.inf], np.nan).mean()
                ),
            }
        )
    write_csv(output / "profiles.csv", pd.DataFrame(profile_rows))
    write_json(
        output / "analysis.json",
        {
            "configuration": config,
            "target_definitions": {t.key: t.metadata for t in targets},
            "scope": "fixed pipelines, repeated held-out folds",
        },
    )
    return pd.DataFrame(summary)
