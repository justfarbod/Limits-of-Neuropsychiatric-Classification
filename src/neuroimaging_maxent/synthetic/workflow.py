import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from ..autoencoders.representation import load_representation
from ..empirical.discriminative import _fit_fold
from ..models.pair import fit_pair, fit_valid, load_pair
from ..utils.io import write_csv, write_json
from ..utils.metrics import exact_weighted_auc
from ..utils.paths import external_path
from ..utils.seeds import stable_seed


def oracle_auc(pair, *, repeats=10, samples_per_class=20000, seed=42):
    h0, j0 = pair.parameters(0)
    h1, j1 = pair.parameters(1)
    if np.array_equal(h0, h1) and np.array_equal(j0, j1):
        return [0.5], "identical distributions"
    if pair.method == "exact":
        states = pair.models[0].states.detach().cpu().numpy().astype(np.int8)
        log0, log1 = pair.log_prob(states, 0), pair.log_prob(states, 1)
        score = log1 - log0
        return [
            exact_weighted_auc(score, score, np.exp(log1), np.exp(log0))
        ], "exhaustive weighted population AUC"
    aucs = []
    for repeat in range(repeats):
        negative = pair.sample(0, samples_per_class, stable_seed(seed, repeat, 0))
        positive = pair.sample(1, samples_per_class, stable_seed(seed, repeat, 1))
        scores = np.r_[pair.score(negative), pair.score(positive)]
        aucs.append(
            float(
                roc_auc_score(
                    np.r_[np.zeros(len(negative)), np.ones(len(positive))], scores
                )
            )
        )
    return aucs, "population Monte Carlo from independent exact junction-tree samples"


def t_interval(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, mean, mean
    half = float(
        t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    )
    return mean, mean - half, mean + half


def run_synthetic(config):
    pair = load_pair(config["source_model"])
    representation = load_representation(
        config["autoencoder_checkpoint"], config["device"]
    )
    if (
        pair.method != config["method"]
        or pair.dimension != config["latent_dim"]
        or representation.settings["latent_dim"] != pair.dimension
    ):
        raise ValueError(
            "Source model, checkpoint, and configured method/dimension must agree."
        )
    if (
        pair.method == "sparse"
        and pair.models[0].topology.decomposition.width > config["treewidth"]
    ):
        raise ValueError("Source topology exceeds configured treewidth.")
    output = external_path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    seed = config["base_seed"]
    n = int(config.get("samples_per_class", 1000))
    replicates = int(config.get("n_replicates", 5))
    alphas = config.get(
        "alpha_values",
        [round(x, 1) for x in np.arange(0, 2.01, 0.1)]
        + list(np.arange(2.5, 7.01, 0.5)),
    )
    if (
        n < config["outer_folds"]
        or replicates < 1
        or not alphas
        or not np.isfinite(alphas).all()
    ):
        raise ValueError("Invalid simulation sizes or interpolation values.")
    reference = {
        rep: pair.sample(0, n, stable_seed(seed, "reference", rep))
        for rep in range(replicates)
    }
    rows, fold_rows, roundtrip_rows, populations = [], [], [], []
    for alpha in alphas:
        source = pair.interpolated(float(alpha))
        values, estimate = oracle_auc(
            source,
            repeats=config.get("oracle_mc_repeats", 10),
            samples_per_class=config.get("oracle_mc_samples_per_class", 20000),
            seed=stable_seed(seed, alpha, "population"),
        )
        mean, lower, upper = t_interval(values)
        populations.append(
            {
                "alpha": alpha,
                "method": "oracle_population",
                "auc": mean,
                "ci_lower": lower,
                "ci_upper": upper,
                "estimate": estimate,
                "scope": "population draws",
            }
        )
        for replicate in range(replicates):
            negative = reference[replicate]
            positive = source.sample(
                1, n, stable_seed(seed, "target", alpha, replicate)
            )
            binary = np.vstack([negative, positive])
            labels = np.r_[np.zeros(n, dtype=int), np.ones(n, dtype=int)]
            decoded = representation.decode(binary)
            _, reencoded = representation.encode(decoded)
            roundtrip_rows.append(
                {
                    "alpha": alpha,
                    "replicate": replicate,
                    "exact_fraction": float(np.all(binary == reencoded, axis=1).mean()),
                    "bit_fraction": float((binary == reencoded).mean()),
                }
            )
            rows.append(
                {
                    "alpha": alpha,
                    "replicate": replicate,
                    "method": "oracle_finite",
                    "auc": roc_auc_score(labels, source.score(binary)),
                    "fit_valid": True,
                }
            )
            splitter = StratifiedKFold(
                config["outer_folds"], shuffle=True, random_state=seed + replicate
            )
            local_aucs = {
                "binary_latent_svc": [],
                "decoded_fc_svc": [],
                "roundtrip_maxent": [],
            }
            validity = []
            for fold, (train, test) in enumerate(splitter.split(binary, labels), 1):
                for method, inputs in [
                    ("binary_latent_svc", binary),
                    ("decoded_fc_svc", decoded),
                ]:
                    row, _, _ = _fit_fold(
                        inputs,
                        labels,
                        train,
                        test,
                        seed + replicate,
                        fold,
                        config["svc_c"],
                    )
                    local_aucs[method].append(row["auc"])
                    fold_rows.append(
                        {
                            "alpha": alpha,
                            "replicate": replicate,
                            "method": method,
                            **row,
                        }
                    )
                if config.get("roundtrip", True):
                    refitted = fit_pair(
                        reencoded[train],
                        labels[train],
                        config,
                        stable_seed(seed, alpha, replicate, fold, "refit"),
                    )
                    valid = (
                        fit_valid(refitted.diagnostics, config["gradient_max_abs"])
                        if refitted.method == "sparse"
                        else all(d["success"] for d in refitted.diagnostics)
                    )
                    score = refitted.score(reencoded[test])
                    valid = valid and bool(np.isfinite(score).all())
                    auc = roc_auc_score(labels[test], score)
                    local_aucs["roundtrip_maxent"].append(auc)
                    validity.append(valid)
                    fold_rows.append(
                        {
                            "alpha": alpha,
                            "replicate": replicate,
                            "fold": fold,
                            "method": "roundtrip_maxent",
                            "auc": auc,
                            "fit_valid": valid,
                        }
                    )
                    write_json(
                        output
                        / "refits"
                        / f"alpha_{alpha}_rep_{replicate}_fold_{fold}.json",
                        {
                            "training_positions": train,
                            "test_positions": test,
                            "diagnostics": refitted.diagnostics,
                            "attempts": refitted.attempts,
                        },
                    )
            for method, aucs in local_aucs.items():
                if aucs:
                    rows.append(
                        {
                            "alpha": alpha,
                            "replicate": replicate,
                            "method": method,
                            "auc": float(np.mean(aucs)),
                            "fit_valid": all(validity)
                            if method == "roundtrip_maxent"
                            else True,
                        }
                    )
    table = pd.DataFrame(rows)
    summary = list(populations)
    for (alpha, method), frame in table.groupby(["alpha", "method"]):
        mean, lower, upper = (
            t_interval(frame.auc) if frame.fit_valid.all() else (np.nan, np.nan, np.nan)
        )
        summary.append(
            {
                "alpha": alpha,
                "method": method,
                "auc": mean,
                "ci_lower": lower,
                "ci_upper": upper,
                "scope": "independent simulation replicates",
                "fit_valid": bool(frame.fit_valid.all()),
            }
        )
    for name, frame in [
        ("replicates", table),
        ("folds", pd.DataFrame(fold_rows)),
        ("roundtrip", pd.DataFrame(roundtrip_rows)),
        ("summary", pd.DataFrame(summary)),
    ]:
        write_csv(output / f"{name}.csv", frame)
    write_json(
        output / "simulation.json",
        {
            "method": pair.method,
            "latent_dim": pair.dimension,
            "alphas": alphas,
            "replicates": replicates,
            "samples_per_class": n,
            "interpretation": "performance relative to a supplied fitted source; not an empirical Bayes ceiling",
        },
    )
    return pd.DataFrame(summary)
