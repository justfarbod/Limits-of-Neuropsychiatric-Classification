from pathlib import Path
import re
import tomllib


def code_root(module_file):
    package = Path(module_file).resolve().parents[1]
    return package.parent.parent if package.parent.name == "src" else package


CODE_ROOT = code_root(__file__)


def is_analysis_checkout(path):
    for directory in (path, *path.parents):
        marker = directory / "pyproject.toml"
        if marker.is_file():
            try:
                if (
                    tomllib.loads(marker.read_text()).get("project", {}).get("name")
                    == "neuroimaging-maxent-analysis"
                ):
                    return True
            except (OSError, ValueError):
                continue
    return False


def external_path(value: str | Path, *, must_exist: bool = False) -> Path:
    text = str(value)
    if not text.strip() or re.search(r"<[^>]*>", text):
        raise ValueError(
            "Replace the input/output placeholder in a private configuration."
        )
    path = Path(value).expanduser().resolve()
    if path == CODE_ROOT or CODE_ROOT in path.parents or is_analysis_checkout(path):
        raise ValueError(
            "Private inputs and generated outputs must be outside the code repository."
        )
    if must_exist and not path.exists():
        raise ValueError("A configured external input does not exist.")
    return path
