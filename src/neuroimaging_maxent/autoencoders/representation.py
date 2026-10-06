from dataclasses import asdict, dataclass
from types import SimpleNamespace

import numpy as np
import torch

from .compact import BinaryAutoencoder, train_compact
from .config import AutoencoderConfig, PreprocessingConfig
from .staged import IsingAutoencoder, train_autoencoder
from ..utils.paths import external_path


def choose_device(value):
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if str(value).startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("Requested accelerator is unavailable.")
    return torch.device(value)


@dataclass
class Representation:
    model: torch.nn.Module
    mean: np.ndarray
    std: np.ndarray
    architecture: str
    settings: dict
    training_indices: np.ndarray

    @property
    def device(self):
        return next(self.model.parameters()).device

    def encode(self, features):
        scaled = (np.asarray(features, dtype=np.float32) - self.mean) / self.std
        self.model.eval()
        continuous = []
        with torch.no_grad():
            for start in range(0, len(scaled), self.settings["batch_size"]):
                batch = torch.from_numpy(
                    scaled[start : start + self.settings["batch_size"]]
                ).to(self.device)
                if self.architecture == "compact":
                    _, z = self.model(batch)
                else:
                    _, z, _ = self.model(batch)
                continuous.append(z.cpu().numpy())
        z = np.vstack(continuous).astype(np.float64)
        return z, (z >= 0).astype(np.int8)

    def decode(self, binary):
        binary = np.asarray(binary, dtype=np.float32)
        if (
            binary.ndim != 2
            or binary.shape[1] != self.settings["latent_dim"]
            or not np.all(np.isin(binary, [0, 1]))
        ):
            raise ValueError(
                "Decoder needs a binary state matrix of the checkpoint dimension."
            )
        return self.decode_continuous(2 * binary - 1)

    def decode_continuous(self, continuous):
        continuous = np.asarray(continuous, dtype=np.float32)
        self.model.eval()
        decoded = []
        with torch.no_grad():
            for start in range(0, len(continuous), self.settings["batch_size"]):
                batch = torch.from_numpy(
                    continuous[start : start + self.settings["batch_size"]]
                ).to(self.device)
                decoded.append(self.model.decoder(batch).cpu().numpy())
        return np.vstack(decoded) * self.std + self.mean

    def save(self, path):
        path = external_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        settings = dict(self.settings)
        if self.architecture == "staged":
            settings["tanh_scale"] = self.model.tanh_scale
        torch.save(
            {
                "format_version": 1,
                "architecture": self.architecture,
                "settings": settings,
                "state_dict": {
                    k: v.detach().cpu() for k, v in self.model.state_dict().items()
                },
                "mean": torch.from_numpy(self.mean),
                "std": torch.from_numpy(self.std),
            },
            path,
        )


def make_model(architecture, settings):
    if architecture == "compact":
        return BinaryAutoencoder(settings["input_dim"], settings["latent_dim"])
    if architecture == "staged":
        return IsingAutoencoder(
            settings["input_dim"],
            settings["latent_dim"],
            settings["hidden_1"],
            settings["hidden_2"],
            settings["dropout"],
            settings.get("tanh_scale", 0.5),
        )
    raise ValueError("Unknown autoencoder architecture.")


def load_representation(path, device="auto"):
    payload = torch.load(
        external_path(path, must_exist=True), map_location="cpu", weights_only=True
    )
    if payload["format_version"] != 1:
        raise ValueError("Unsupported checkpoint version.")
    model = make_model(payload["architecture"], payload["settings"])
    model.load_state_dict(payload["state_dict"])
    model.to(choose_device(device)).eval()
    return Representation(
        model,
        payload["mean"].numpy(),
        payload["std"].numpy(),
        payload["architecture"],
        payload["settings"],
        np.empty(0, dtype=int),
    )


def fit_representation(features, train, config, seed, architecture="compact"):
    features = np.asarray(features, dtype=np.float32)
    train = np.asarray(train, dtype=int)
    if (
        not len(train)
        or len(np.unique(train)) != len(train)
        or np.any(train < 0)
        or np.any(train >= len(features))
    ):
        raise ValueError("Training positions must be nonempty, unique, and in range.")
    device = choose_device(config["device"])
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
    if architecture == "staged":
        mean = features[train].mean(axis=0).astype(np.float32)
        std = features[train].std(axis=0).astype(np.float32) + 1e-6
    else:
        mean = features[train].mean(axis=0, dtype=np.float64).astype(np.float32)
        std = features[train].std(axis=0, dtype=np.float64).astype(np.float32)
        std[std < 1e-6] = 1.0
    scaled = (features - mean) / std
    settings = {
        "input_dim": features.shape[1],
        "latent_dim": int(config["latent_dim"]),
        "batch_size": int(config["autoencoder_batch_size"]),
    }
    if architecture == "compact":
        model = make_model(architecture, settings).to(device)
        train_compact(
            model,
            scaled,
            train,
            seed=seed,
            epochs=int(config["autoencoder_epochs"]),
            batch_size=settings["batch_size"],
            device=device,
        )
    elif architecture == "staged":
        ae = AutoencoderConfig(**config.get("staged_autoencoder", {}))
        ae.latent_dim = settings["latent_dim"]
        settings.update(
            {
                k: v
                for k, v in asdict(ae).items()
                if k in ["hidden_1", "hidden_2", "dropout"]
            }
        )
        model = make_model(architecture, settings).to(device)
        generator = torch.Generator().manual_seed(seed)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(scaled[train])),
            batch_size=ae.train_batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=device.type == "cuda",
            generator=generator,
        )
        model, _ = train_autoencoder(
            model,
            loader,
            SimpleNamespace(autoencoder=ae, preprocessing=PreprocessingConfig()),
            device,
        )
    else:
        raise ValueError("Unknown autoencoder architecture.")
    return Representation(model.eval(), mean, std, architecture, settings, train.copy())
