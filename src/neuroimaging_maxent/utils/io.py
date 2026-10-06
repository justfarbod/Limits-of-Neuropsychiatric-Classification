import hashlib
import json
from pathlib import Path

import numpy as np

from .paths import external_path


def fingerprint(*arrays, settings=None) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        values = np.asarray(array)
        digest.update(str(values.shape).encode())
        digest.update(str(values.dtype).encode())
        digest.update(values.tobytes())
    digest.update(json.dumps(settings, sort_keys=True, default=str).encode())
    return digest.hexdigest()


def write_json(path: str | Path, payload) -> None:
    path = external_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=_json_value) + "\n")
    temporary.replace(path)


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError("Unsupported JSON value")


def read_json(path: str | Path):
    return json.loads(external_path(path, must_exist=True).read_text())


def write_csv(path, table) -> None:
    path = external_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
