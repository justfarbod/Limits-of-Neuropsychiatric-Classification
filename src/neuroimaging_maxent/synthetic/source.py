import numpy as np

from ..autoencoders.metrics import reconstruction_metrics
from ..autoencoders.representation import fit_representation, load_representation
from ..inputs import load_inputs
from ..models.pair import fit_pair, fit_valid
from ..targets import build_targets
from ..utils.io import write_json
from ..utils.paths import external_path


def fit_source(config):
    data = load_inputs(config)
    targets = build_targets(data.metadata, config["targets"])
    if len(targets) != 1:
        raise ValueError("Source fitting requires one binary comparison.")
    target = targets[0]
    features = data.features[target.indices]
    if config.get("autoencoder_checkpoint"):
        representation = load_representation(
            config["autoencoder_checkpoint"], config["device"]
        )
        if representation.settings["latent_dim"] != config["latent_dim"]:
            raise ValueError(
                "Source checkpoint dimension differs from the configuration."
            )
        representation_scope = (
            "externally supplied checkpoint; training scope supplied by the user"
        )
    else:
        representation = fit_representation(
            features,
            np.arange(len(features)),
            config,
            config["seed"],
            config["source_autoencoder"],
        )
        representation_scope = "descriptive full input subset"
    continuous, binary = representation.encode(features)
    pair = fit_pair(binary, target.labels, config, config["seed"])
    valid = (
        fit_valid(pair.diagnostics, config["gradient_max_abs"])
        if pair.method == "sparse"
        else all(d["success"] for d in pair.diagnostics)
    )
    if not valid:
        raise RuntimeError(
            "Source fit failed its numerical diagnostics; no source was exported."
        )
    output = external_path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    pair.save(output / "source_model.npz")
    representation.save(output / "autoencoder.pt")
    decoded_continuous = representation.decode_continuous(continuous)
    decoded_binary = representation.decode(binary)
    fidelity = {
        "continuous": reconstruction_metrics(features, decoded_continuous),
        "binary": reconstruction_metrics(features, decoded_binary),
        "scope": representation_scope,
    }
    write_json(
        output / "source_fit.json",
        {
            "method": pair.method,
            "latent_dim": pair.dimension,
            "diagnostics": pair.diagnostics,
            "attempts": pair.attempts,
            "fidelity": fidelity,
        },
    )
    return fidelity
