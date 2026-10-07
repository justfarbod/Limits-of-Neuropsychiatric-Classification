import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy.integrate import cumulative_trapezoid, trapezoid

from neuroimaging_maxent.cli import main
from neuroimaging_maxent.config import load_config
from neuroimaging_maxent.mechanical.figure import calculate_figure, render_figure
from neuroimaging_maxent.mechanical.quadrature import (
    TripleWellQuadrature,
    run_mechanical,
    triple_well,
)
from neuroimaging_maxent.utils.paths import CODE_ROOT


def metadata():
    return {
        "landmarks": [
            {
                "label": "first matched",
                "auc": 0.60,
                "color": "rose",
                "marker": "^",
                "show_distribution": True,
            },
            {"label": "intermediate", "auc": 0.68, "color": "purple"},
            {
                "label": "second matched",
                "auc": 0.78,
                "color": "blue",
                "show_distribution": True,
            },
        ],
        "bands": [
            {
                "label": "supplied comparisons",
                "lower": 0.59,
                "upper": 0.62,
                "color": "rose",
            }
        ],
    }


def test_default_agrees_with_independent_historical_quadrature():
    alpha = 0.4
    x = np.linspace(-2.25, 2.25, 120001)
    potential = 6 * x**2 * (x**2 - 1) ** 2
    p0 = np.exp(-1.5 * potential)
    p1 = np.exp(-1.5 * (potential - alpha * x))
    z0, z1 = trapezoid(p0, x), trapezoid(p1, x)
    p0, p1 = p0 / z0, p1 / z1
    cdf = cumulative_trapezoid(p0, x, initial=0)
    expected_auc = trapezoid(p1 * cdf, x)
    expected_forward = -1.5 * alpha * trapezoid(p0 * x, x) + np.log(z1 / z0)
    expected_reverse = 1.5 * alpha * trapezoid(p1 * x, x) - np.log(z1 / z0)
    result = triple_well(alpha)
    assert result["auc"] == pytest.approx(expected_auc, abs=1e-12)
    assert result["forward_work"] == pytest.approx(expected_forward, abs=1e-12)
    assert result["reverse_work"] == pytest.approx(expected_reverse, abs=1e-12)


def test_field_scale_changes_tilt_only():
    first = triple_well(0.3, h=2, points=10001)
    second = triple_well(0.6, h=1, points=10001)
    for name in first:
        if name != "alpha":
            assert first[name] == pytest.approx(second[name], abs=1e-12)
    reference = TripleWellQuadrature(h=2, points=1001).reference
    baseline = TripleWellQuadrature(h=1, points=1001).reference
    np.testing.assert_array_equal(reference["potential"], baseline["potential"])


def test_landmark_roots_occupancies_and_independent_identity():
    quadrature = TripleWellQuadrature(points=10001)
    data = calculate_figure(quadrature, np.linspace(0, 1.5, 21), metadata())
    np.testing.assert_allclose(
        data["landmarks"].auc, data["landmarks"].target_auc, atol=1e-12
    )
    assert data["report"]["three_wells_retained"]
    assert data["report"]["chance_auc"] == 0.5
    for name, value in data["report"].items():
        if name.endswith(("error", "excess")):
            assert value < 1e-12
    row = quadrature.evaluate(0.7)
    target = quadrature.equilibrium(0.7)
    cdf = cumulative_trapezoid(target["density"], quadrature.x, initial=0)
    roots = np.sort(np.roots([36, 0, -48, 0, 12, -0.7]).real)
    assert row["left_occupancy"] == pytest.approx(
        np.interp(roots[1], quadrature.x, cdf)
    )
    assert row["right_occupancy"] == pytest.approx(
        1 - np.interp(roots[3], quadrature.x, cdf)
    )
    data["landmarks"].loc[1, "label"] = "custom marker"
    fig = render_figure(data)
    try:
        assert any(text.get_text() == "custom marker" for text in fig.axes[3].texts)
        assert any(
            "supplied comparisons\n(outcome range)" == text.get_text()
            for text in fig.axes[3].texts
        )
        assert fig.axes[1].get_legend() is None
        assert fig.axes[2].lines[2].get_color() == "#087E8B"
        assert len(fig.axes[3].patches) == 1
        assert all(
            spine.get_visible() for axis in fig.axes for spine in axis.spines.values()
        )
        assert not any(
            line.get_visible() for axis in fig.axes for line in axis.get_xgridlines()
        )
        assert data["landmarks"].marker.iloc[0] == "^"
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "change",
    [
        "unreachable",
        "nan",
        "negative",
        "duplicate",
        "bad_range",
        "bad_marker",
        "no_distribution",
    ],
)
def test_invalid_landmarks(change):
    supplied = metadata()
    if change == "unreachable":
        supplied["landmarks"][0]["auc"] = 0.99
    elif change == "nan":
        supplied["landmarks"][0]["auc"] = float("nan")
    elif change == "negative":
        supplied["landmarks"][0]["auc"] = 0.2
    elif change == "duplicate":
        supplied["landmarks"][1]["label"] = supplied["landmarks"][0]["label"]
    elif change == "bad_range":
        supplied["bands"][0]["lower"] = 0.9
    elif change == "bad_marker":
        supplied["landmarks"][0]["marker"] = "invalid-marker"
    else:
        for item in supplied["landmarks"]:
            item["show_distribution"] = False
    with pytest.raises(ValueError):
        calculate_figure(TripleWellQuadrature(points=1001), [0, 1.5], supplied)


def test_chance_landmark_and_no_three_wells():
    supplied = metadata()
    supplied["landmarks"][0]["auc"] = 0.5
    data = calculate_figure(TripleWellQuadrature(points=1001), [0, 1.5], supplied)
    assert data["landmarks"].alpha.iloc[0] == 0
    with pytest.raises(ValueError, match="three wells"):
        calculate_figure(TripleWellQuadrature(points=1001), [0, 3], supplied)
    assert not triple_well(3, points=1001)["three_wells"]


@pytest.mark.parametrize("grid", [[0, 0], [1, 2], [0, -1], [0, np.nan]])
def test_invalid_grid(grid):
    with pytest.raises(ValueError):
        calculate_figure(TripleWellQuadrature(points=1001), grid, metadata())


def test_figure_cli_and_path_restrictions(private_dir, capsys):
    landmarks = private_dir / "landmarks.json"
    landmarks.write_text(json.dumps(metadata()))
    config_file = private_dir / "figure.json"
    config = {
        "output_dir": str(private_dir / "outputs"),
        "landmarks_file": str(landmarks),
        "make_figure": True,
        "figure_prefix": "custom",
        "points": 1001,
        "alpha_values": [0, 0.5, 1, 1.5],
    }
    config_file.write_text(json.dumps(config))
    main(["mechanical", "--config", str(config_file)])
    assert capsys.readouterr().out == "Completed.\n"
    assert {file.name for file in (private_dir / "outputs").iterdir()} == {
        "custom.png",
        "custom.pdf",
        "custom_curves.csv",
        "custom_landmarks.csv",
        "custom_densities.csv",
        "custom_validation.json",
    }
    table = pd.read_csv(private_dir / "outputs" / "custom_landmarks.csv")
    np.testing.assert_allclose(table.auc, table.target_auc, atol=1e-12)
    for field, value in [
        ("output_dir", str(CODE_ROOT / "figures")),
        ("landmarks_file", str(CODE_ROOT / "configs" / "mechanical.json")),
        ("landmarks_file", "<LANDMARKS_FILE>"),
    ]:
        config_file.write_text(json.dumps({**config, field: value}))
        with pytest.raises(ValueError):
            load_config(config_file)
    config_file.write_text(json.dumps({**config, "figure_prefix": "../escape"}))
    with pytest.raises(ValueError):
        run_mechanical(load_config(config_file))
    link = private_dir / "linked"
    link.symlink_to(CODE_ROOT, target_is_directory=True)
    config_file.write_text(json.dumps({**config, "output_dir": str(link / "output")}))
    with pytest.raises(ValueError):
        load_config(config_file)


def test_numerical_only_remains_default(private_dir):
    run_mechanical(
        {"output_dir": str(private_dir), "points": 1001, "alpha_values": [0, 1]}
    )
    assert {file.name for file in private_dir.iterdir()} == {"mechanical.csv"}
