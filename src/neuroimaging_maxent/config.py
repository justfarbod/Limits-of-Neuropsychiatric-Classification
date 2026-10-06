from copy import deepcopy
import json
from pathlib import Path

from .utils.paths import external_path


DEFAULTS = {
    "method": "sparse",
    "latent_dim": 120,
    "treewidth": 1,
    "device": "auto",
    "autoencoder_epochs": 100,
    "autoencoder_batch_size": 128,
    "outer_seeds": [42, 43, 44, 45, 46],
    "outer_folds": 5,
    "base_seed": 20260825,
    "bootstrap_seed": 20260825,
    "bootstrap_repeats": 10000,
    "svc_c": 1.0,
    "representations": [
        "original_fc",
        "continuous_latent",
        "binary_latent",
        "decoded_fc",
    ],
    "max_iterations": 500,
    "retry_max_iterations": 2000,
    "gradient_max_abs": 0.001,
    "ppc_samples_per_class": 5000,
    "exact_steps": 10000,
    "exact_patience": 300,
    "exact_lr": 0.01,
    "exact_l1": 0.0,
    "inference_backend": "auto",
    "source_autoencoder": "staged",
    "seed": 42,
}


def resolve_config(config: dict) -> dict:
    result = deepcopy(DEFAULTS)
    result.update(deepcopy(config))
    if result["method"] not in {"sparse", "exact"}:
        raise ValueError("method must be sparse or exact")
    if not 1 <= int(result["latent_dim"]):
        raise ValueError("latent_dim must be positive")
    if result["method"] == "exact" and int(result["latent_dim"]) > 22:
        raise ValueError("Full enumeration is limited to 22 variables.")
    if result["treewidth"] not in {1, 2, 3, 4, 5}:
        raise ValueError("treewidth must be between 1 and 5")
    if result["outer_folds"] < 2 or not result["outer_seeds"]:
        raise ValueError("Cross-validation needs at least two folds and one seed.")
    for name in [
        "autoencoder_epochs",
        "autoencoder_batch_size",
        "bootstrap_repeats",
        "max_iterations",
        "ppc_samples_per_class",
        "exact_steps",
    ]:
        if result[name] < 1:
            raise ValueError(f"{name} must be positive")
    return result


def load_config(path: str | Path) -> dict:
    path = external_path(path, must_exist=True)
    config = json.loads(path.read_text())
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a JSON object.")
    result = resolve_config(config)
    for key in [
        "features_file",
        "metadata_file",
        "time_series_manifest",
        "source_model",
        "autoencoder_checkpoint",
        "output_dir",
        "results_file",
    ]:
        if key in result and result[key] is not None:
            result[key] = str(external_path(result[key]))
    return result
