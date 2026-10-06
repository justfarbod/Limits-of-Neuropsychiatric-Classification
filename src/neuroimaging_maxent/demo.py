import tempfile

import numpy as np
import pandas as pd

from .config import resolve_config
from .empirical.workflow import run_empirical
from .mechanical.quadrature import run_mechanical
from .selection.workflow import run_selection
from .synthetic.source import fit_source
from .synthetic.workflow import run_synthetic
from .utils.paths import external_path


def run_demo(output_dir=None):
    output = external_path(output_dir or tempfile.mkdtemp(prefix="maxent-demo-"))
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    labels = np.tile([0, 1], 32)
    features = rng.normal(size=(64, 12)).astype(np.float32)
    features[:, :3] += labels[:, None] * 0.6
    identifiers = np.array([f"toy-{i:03d}" for i in range(len(labels))])
    np.savez_compressed(output / "features.npz", features=features, row_ids=identifiers)
    pd.DataFrame({"row_id": identifiers, "group": labels}).to_csv(
        output / "metadata.csv", index=False
    )
    config = resolve_config(
        {
            "features_file": str(output / "features.npz"),
            "metadata_file": str(output / "metadata.csv"),
            "targets": [
                {
                    "key": "group_comparison",
                    "kind": "groups",
                    "column": "group",
                    "negative": 0,
                    "positive": 1,
                }
            ],
            "device": "cpu",
            "latent_dim": 3,
            "autoencoder_epochs": 2,
            "outer_folds": 2,
            "outer_seeds": [42],
            "bootstrap_repeats": 30,
            "ppc_samples_per_class": 100,
            "inference_backend": "numpy",
            "max_iterations": 100,
            "retry_max_iterations": 300,
        }
    )
    run_empirical({**config, "output_dir": str(output / "empirical")})
    run_selection(
        {
            **config,
            "output_dir": str(output / "selection"),
            "inner_folds": 2,
            "latent_dims": [3],
            "treewidths": [1],
            "selection_bootstrap_repeats": 20,
        }
    )
    fit_source(
        {
            **config,
            "output_dir": str(output / "source"),
            "source_autoencoder": "compact",
        }
    )
    run_synthetic(
        {
            **config,
            "output_dir": str(output / "synthetic"),
            "source_model": str(output / "source" / "source_model.npz"),
            "autoencoder_checkpoint": str(output / "source" / "autoencoder.pt"),
            "samples_per_class": 32,
            "n_replicates": 2,
            "alpha_values": [0.0, 1.0],
            "oracle_mc_repeats": 2,
            "oracle_mc_samples_per_class": 100,
        }
    )
    run_mechanical(
        {
            "output_dir": str(output / "mechanical"),
            "alpha_values": [0.0, 0.5, 1.0],
            "points": 1001,
        }
    )
    return output
