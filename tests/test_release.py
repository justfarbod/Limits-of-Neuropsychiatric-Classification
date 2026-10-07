import ast
import importlib
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from neuroimaging_maxent.cli import main
from neuroimaging_maxent.reporting.plots import plot_results


@pytest.mark.parametrize(
    "command",
    [
        "validate-inputs",
        "build-fc",
        "empirical",
        "select",
        "fit-source",
        "synthetic",
        "mechanical",
        "plot",
        "demo",
    ],
)
def test_command_help(command):
    result = subprocess.run(
        [sys.executable, "-m", "neuroimaging_maxent", command, "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "usage:" in result.stdout


def test_all_modules_import():
    import neuroimaging_maxent

    root = Path(neuroimaging_maxent.__file__).parent
    for file in root.rglob("*.py"):
        if file.name == "__main__.py":
            continue
        name = ".".join(
            ("neuroimaging_maxent", *file.relative_to(root).with_suffix("").parts)
        )
        importlib.import_module(name)


def test_cli_validation_and_private_error(toy_config, private_dir, capsys):
    path = private_dir / "config.json"
    path.write_text(json.dumps(toy_config))
    main(["validate-inputs", "--config", str(path)])
    assert '"rows": 48' in capsys.readouterr().out
    path.write_text(
        json.dumps(
            {
                "features_file": str(private_dir / "missing"),
                "metadata_file": str(private_dir / "missing"),
            }
        )
    )
    with pytest.raises(SystemExit):
        main(["validate-inputs", "--config", str(path)])
    error = capsys.readouterr().err
    assert str(private_dir) not in error


def test_plots_use_supplied_tables(private_dir):
    frame = pd.DataFrame(
        {"alpha": [0, 1], "method": ["supplied", "supplied"], "auc": [0.5, 0.7]}
    )
    file = private_dir / "results.csv"
    frame.to_csv(file, index=False)
    plot = plot_results(
        {"results_file": str(file), "output_dir": str(private_dir / "figures")}
    )
    assert plot.is_file()
    pd.DataFrame({"observed": [0.2, 0.8], "modeled": [0.3, 0.7]}).to_csv(
        file, index=False
    )
    assert plot_results(
        {
            "results_file": str(file),
            "output_dir": str(private_dir / "figures"),
            "plot_kind": "marginals",
        }
    ).is_file()


def test_release_has_only_source_and_templates():
    root = Path(__file__).resolve().parents[1]
    allowed = {".py", ".md", ".json", ".toml", ".yml"}
    for file in root.rglob("*"):
        if (
            not file.is_file()
            or any(
                part in {".git", "__pycache__", "build", "dist", ".venv"}
                or part.endswith(".egg-info")
                for part in file.parts
            )
            or file.suffix == ".pyc"
        ):
            continue
        assert not file.is_symlink()
        assert file.suffix in allowed or file.name == ".gitignore"
        if file.suffix == ".py":
            tree = ast.parse(file.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    assert not (
                        Path(node.value).is_absolute()
                        and len(Path(node.value).parts) > 1
                    )
    for config in (root / "configs").glob("*.json"):
        data = json.loads(config.read_text())
        for key in [
            "features_file",
            "metadata_file",
            "source_model",
            "autoencoder_checkpoint",
            "output_dir",
            "landmarks_file",
        ]:
            if key in data:
                assert data[key].startswith("<") and data[key].endswith(">")
