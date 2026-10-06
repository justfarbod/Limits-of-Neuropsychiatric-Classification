import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neuroimaging_maxent.config import resolve_config


@pytest.fixture(scope="session", autouse=True)
def limit_threads():
    import torch

    torch.set_num_threads(1)


@pytest.fixture
def private_dir():
    with tempfile.TemporaryDirectory(prefix="maxent-test-") as directory:
        yield Path(directory)


@pytest.fixture
def toy_config(private_dir):
    rng = np.random.default_rng(19)
    labels = np.tile([0, 1], 24)
    features = rng.normal(size=(len(labels), 10)).astype(np.float32)
    features[:, :2] += labels[:, None]
    ids = np.array([f"test-{i:04d}" for i in range(len(labels))])
    np.savez(private_dir / "features.npz", features=features, row_ids=ids)
    pd.DataFrame({"row_id": ids, "group": labels}).to_csv(
        private_dir / "metadata.csv", index=False
    )
    return resolve_config(
        {
            "features_file": str(private_dir / "features.npz"),
            "metadata_file": str(private_dir / "metadata.csv"),
            "output_dir": str(private_dir / "output"),
            "targets": [
                {
                    "key": "groups",
                    "kind": "groups",
                    "column": "group",
                    "negative": 0,
                    "positive": 1,
                }
            ],
            "latent_dim": 3,
            "device": "cpu",
            "autoencoder_epochs": 2,
            "autoencoder_batch_size": 16,
            "outer_folds": 2,
            "outer_seeds": [42, 43],
            "bootstrap_repeats": 30,
            "ppc_samples_per_class": 100,
            "max_iterations": 100,
            "retry_max_iterations": 300,
            "inference_backend": "numpy",
        }
    )
