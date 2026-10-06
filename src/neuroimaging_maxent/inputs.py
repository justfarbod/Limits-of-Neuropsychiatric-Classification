from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .utils.paths import external_path


@dataclass
class InputData:
    features: np.ndarray
    row_ids: np.ndarray
    metadata: pd.DataFrame


def load_inputs(config: dict) -> InputData:
    features_path = external_path(config["features_file"], must_exist=True)
    metadata_path = external_path(config["metadata_file"], must_exist=True)
    with np.load(features_path, allow_pickle=False) as saved:
        if not {"features", "row_ids"}.issubset(saved.files):
            raise ValueError("Feature archive requires features and row_ids arrays.")
        features = np.asarray(saved["features"], dtype=np.float32)
        raw_ids = saved["row_ids"]
        if raw_ids.dtype.kind not in {"U", "S", "i", "u"}:
            raise ValueError(
                "row_ids must be strings or integers, without pickled objects."
            )
        ids = raw_ids.astype(str)
    if features.ndim != 2 or features.shape[0] != len(ids) or ids.ndim != 1:
        raise ValueError("Features and row IDs must be aligned.")
    if not features.size or not np.isfinite(features).all():
        raise ValueError("Features must be nonempty and finite.")
    if len(np.unique(ids)) != len(ids) or np.any(ids == ""):
        raise ValueError("Feature row IDs must be nonempty and unique.")
    metadata = pd.read_csv(metadata_path, dtype={"row_id": str})
    if (
        "row_id" not in metadata
        or metadata.row_id.isna().any()
        or metadata.row_id.duplicated().any()
    ):
        raise ValueError("Metadata requires a unique, nonmissing row_id column.")
    aligned = pd.DataFrame({"row_id": ids}).merge(
        metadata, how="left", on="row_id", validate="one_to_one", indicator=True
    )
    if not aligned["_merge"].eq("both").all():
        raise ValueError("Some feature rows have no matching metadata.")
    aligned = aligned.drop(columns="_merge")
    subset = config.get("subset")
    if subset:
        if subset["column"] not in aligned:
            raise ValueError("The configured subset column is missing.")
        keep = aligned[subset["column"]].isin(subset["values"]).to_numpy()
        if not keep.any():
            raise ValueError("The configured subset is empty.")
        features, ids = features[keep], ids[keep]
        aligned = aligned.loc[keep].reset_index(drop=True)
    return InputData(features, ids, aligned)


def connectivity_vector(
    signals, *, orientation="time_by_region", triangle="lower", constant_policy="zero"
):
    values = np.asarray(signals, dtype=float)
    if orientation == "region_by_time":
        values = values.T
    elif orientation != "time_by_region":
        raise ValueError("Unknown signal orientation.")
    if values.ndim != 2 or min(values.shape) < 2 or not np.isfinite(values).all():
        raise ValueError("Signals must be a finite time-by-region matrix.")
    constant = values.std(axis=0) == 0
    if constant.any() and constant_policy == "error":
        raise ValueError("A signal has zero temporal variance.")
    if constant_policy not in {"zero", "error"}:
        raise ValueError("Unknown constant-signal policy.")
    with np.errstate(invalid="ignore", divide="ignore"):
        matrix = np.corrcoef(values, rowvar=False)
    matrix[constant, :] = 0
    matrix[:, constant] = 0
    if triangle == "lower":
        index = np.tril_indices(values.shape[1], -1)
    elif triangle == "upper":
        index = np.triu_indices(values.shape[1], 1)
    else:
        raise ValueError("triangle must be lower or upper")
    return matrix[index].astype(np.float32)


def build_features(config: dict) -> Path:
    manifest = external_path(config["time_series_manifest"], must_exist=True)
    table = pd.read_csv(manifest, dtype={"row_id": str})
    if (
        not {"row_id", "file"}.issubset(table)
        or table.row_id.isna().any()
        or table.row_id.duplicated().any()
        or table.empty
    ):
        raise ValueError("Signal manifest requires unique row_id and file columns.")
    vectors = []
    for name in table.file:
        file = Path(name)
        if not file.is_absolute():
            file = manifest.parent / file
        signals = np.load(external_path(file, must_exist=True), allow_pickle=False)
        vectors.append(
            connectivity_vector(
                signals,
                orientation=config.get("orientation", "time_by_region"),
                triangle=config.get("triangle", "lower"),
                constant_policy=config.get("constant_policy", "zero"),
            )
        )
    if len({len(v) for v in vectors}) != 1:
        raise ValueError("Signals must have the same number of regions.")
    output = external_path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    path = output / "features.npz"
    np.savez_compressed(
        path, features=np.stack(vectors), row_ids=table.row_id.to_numpy(dtype=str)
    )
    return path
