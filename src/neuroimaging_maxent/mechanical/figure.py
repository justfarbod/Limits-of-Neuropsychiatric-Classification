import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import is_color_like
from matplotlib.markers import MarkerStyle
from scipy.optimize import brentq

from ..reporting.style import (
    BACKGROUND,
    BOUND_COLOR,
    HIGH_TILT_COLOR,
    INTERMEDIATE_COLOR,
    LOW_TILT_COLOR,
    MODEL_COLOR,
    MUTED_INK,
    PAPER_STYLE,
    REFERENCE_COLOR,
    SECONDARY_COLOR,
    panel_heading,
)
from ..utils.io import read_json, write_csv, write_json
from ..utils.paths import external_path

GRAY = REFERENCE_COLOR
ROSE = LOW_TILT_COLOR
BLUE = HIGH_TILT_COLOR
PURPLE = INTERMEDIATE_COLOR
PALETTE = {"grey": GRAY, "rose": ROSE, "blue": BLUE, "purple": PURPLE}


def _color(value):
    if not isinstance(value, str):
        raise ValueError("Invalid figure color.")
    color = PALETTE.get(value, value)
    if not is_color_like(color):
        raise ValueError("Invalid figure color.")
    return color


def _number(value):
    try:
        if isinstance(value, bool):
            raise ValueError("Expected a numeric value.")
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("Expected a numeric value.") from error


def validate_metadata(metadata):
    if not isinstance(metadata, dict) or not isinstance(
        metadata.get("landmarks"), list
    ):
        raise ValueError("Landmark metadata must contain a landmarks list.")
    if not isinstance(metadata.get("bands", []), list):
        raise ValueError("Outcome ranges must be supplied as a list.")
    landmarks = []
    for item in metadata["landmarks"]:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("label"), str)
            or not item["label"].strip()
        ):
            raise ValueError("Each landmark needs a nonempty label.")
        auc = _number(item.get("auc"))
        show = item.get("show_distribution", False)
        if not np.isfinite(auc) or not 0.5 <= auc <= 1 or not isinstance(show, bool):
            raise ValueError("Invalid landmark AUC or distribution flag.")
        marker = item.get("marker", "o")
        if not isinstance(marker, str):
            raise ValueError("Invalid landmark marker.")
        MarkerStyle(marker)
        landmarks.append(
            {
                "label": item["label"],
                "target_auc": auc,
                "color": _color(item.get("color", PURPLE)),
                "show_distribution": show,
                "marker": marker,
            }
        )
    if (
        not landmarks
        or not 1 <= sum(item["show_distribution"] for item in landmarks) <= 2
    ):
        raise ValueError(
            "Select one or two landmark distributions for the upper panels."
        )
    if len({item["label"] for item in landmarks}) != len(landmarks):
        raise ValueError("Landmark labels must be unique.")
    bands = []
    for item in metadata.get("bands", []):
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("label"), str)
            or not item["label"].strip()
        ):
            raise ValueError("Each outcome range needs a label.")
        lower, upper = _number(item.get("lower")), _number(item.get("upper"))
        if not np.isfinite([lower, upper]).all() or not 0.5 <= lower < upper <= 1:
            raise ValueError("Invalid outcome range.")
        bands.append(
            {
                "label": item["label"],
                "lower": lower,
                "upper": upper,
                "color": _color(item.get("color", GRAY)),
            }
        )
    return landmarks, bands


def calculate_figure(quadrature, alpha_values, metadata):
    alphas = np.asarray(alpha_values, dtype=float)
    if (
        alphas.ndim != 1
        or len(alphas) < 2
        or not np.isfinite(alphas).all()
        or alphas[0] != 0
        or np.any(np.diff(alphas) <= 0)
    ):
        raise ValueError("The figure grid must increase strictly from zero.")
    landmarks, bands = validate_metadata(metadata)
    curves = pd.DataFrame([quadrature.evaluate(alpha) for alpha in alphas])
    if not curves.three_wells.all():
        raise ValueError("The figure deformation range must retain three wells.")
    solved = []
    for item in landmarks:
        target = item["target_auc"]
        if target > curves.auc.iloc[-1]:
            raise ValueError("A landmark AUC is unreachable in the deformation range.")
        alpha = (
            0.0
            if target == 0.5
            else brentq(
                lambda value, target=target: quadrature.evaluate(value)["auc"] - target,
                0.0,
                float(alphas[-1]),
                xtol=1e-12,
            )
        )
        solved.append({**item, **quadrature.evaluate(alpha)})
    points = pd.DataFrame(solved)
    all_rows = pd.concat([curves, points], ignore_index=True)
    occupancy_sum = all_rows[
        ["left_occupancy", "middle_occupancy", "right_occupancy"]
    ].sum(axis=1)
    report = {
        "density_normalization_max_error": float(
            np.abs(all_rows.density_mass - 1).max()
        ),
        "occupancy_sum_max_error": float(np.abs(occupancy_sum - 1).max()),
        "three_wells_retained": bool(all_rows.three_wells.all()),
        "chance_auc": float(curves.auc.iloc[0]),
        "work_identity_max_error": float(np.abs(all_rows.work_identity_error).max()),
        "jeffreys_work_max_error": float(np.abs(all_rows.jeffreys_work_error).max()),
        "landmark_auc_max_error": float(np.abs(points.auc - points.target_auc).max()),
        "auc_bound_max_excess": float(
            np.maximum(all_rows.auc - all_rows.auc_bound, 0).max()
        ),
        "band_scope": "outcome ranges, not confidence intervals",
        "parameters": {
            "a": quadrature.a,
            "h": quadrature.h,
            "beta": quadrature.beta,
            "bound": float(quadrature.x[-1]),
            "points": len(quadrature.x),
        },
    }
    errors = [
        value for key, value in report.items() if key.endswith(("error", "excess"))
    ]
    if max(errors) > 1e-10:
        raise RuntimeError("Mechanical figure failed numerical validation.")
    distributions = [
        {"label": "reference", "alpha": 0.0, "color": GRAY, **quadrature.reference}
    ]
    distributions.extend(
        {**item, **quadrature.equilibrium(item["alpha"])}
        for item in solved
        if item["show_distribution"]
    )
    return {
        "x": quadrature.x,
        "curves": curves,
        "landmarks": points,
        "bands": bands,
        "distributions": distributions,
        "report": report,
    }


def render_figure(data):
    with matplotlib.rc_context(PAPER_STYLE):
        figure_height = 5.85
        panel_height = 5.55 * (0.93 - 0.11) / (2 + 0.42)
        fig, axes = plt.subplots(2, 2, figsize=(9.55, figure_height))
        potential, density, occupancy, auc = axes.flat
        for item in data["distributions"]:
            label = item["label"] + rf" ($\alpha={item['alpha']:.2f}$)"
            if item["alpha"] == 0:
                label = r"reference ($\alpha=0$)"
            potential.plot(
                data["x"], item["potential"], color=item["color"], lw=1.7, label=label
            )
            density.plot(data["x"], item["density"], color=item["color"], lw=1.7)
        potential.set(
            xlim=(-1.5, 1.5),
            ylim=(-1.75, 4.55),
            xlabel=r"Mechanical coordinate, $x$",
            ylabel=r"Potential, $U_\alpha(x)$",
        )
        panel_heading(potential, "a", "Shared repertoire,\ndifferent tilt")
        potential.legend(frameon=False, loc="lower left", handlelength=1.6)
        potential.text(
            0.98,
            0.97,
            r"$U_\alpha(x)=U_{\mathrm{str}}(x)-\alpha hx$",
            transform=potential.transAxes,
            ha="right",
            va="top",
            fontsize=7.4,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9},
        )
        density.set(
            xlim=(-1.5, 1.5),
            ylim=(0, 2.82),
            xlabel=r"Mechanical coordinate, $x$",
            ylabel=r"Density, $p_\alpha(x)$",
        )
        panel_heading(density, "b", "Snapshot overlap,\nnot new wells")
        selected = data["landmarks"][data["landmarks"].show_distribution].sort_values(
            "alpha"
        )
        text = "exact snapshot AUC\n" + "\n".join(
            f"{row.label}  {row.auc:.3f}" for row in selected.itertuples()
        )
        density.text(
            0.02,
            0.97,
            text,
            transform=density.transAxes,
            ha="left",
            va="top",
            fontsize=7.3,
            color=MUTED_INK,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.94, "pad": 1},
        )
        curves = data["curves"]
        for column, color, label in [
            ("left_occupancy", GRAY, "left well"),
            ("middle_occupancy", SECONDARY_COLOR, "middle well"),
            ("right_occupancy", MODEL_COLOR, "right well"),
        ]:
            occupancy.plot(
                curves.alpha, curves[column], color=color, lw=1.8, label=label
            )
        max_alpha = float(curves.alpha.iloc[-1])
        for index, row in enumerate(selected.itertuples()):
            occupancy.axvline(row.alpha, color=row.color, ls="--", lw=1.0, alpha=0.9)
            if index == 0:
                occupancy.text(
                    row.alpha + 0.05 * max_alpha,
                    0.12,
                    f"{row.label}\n$P_{{\\mathrm{{right}}}}={row.right_occupancy:.2f}$\nodds $R/L={row.right_left_odds:.1f}$",
                    color=row.color,
                    fontsize=6.4,
                    ha="left",
                    va="bottom",
                )
            else:
                occupancy.text(
                    row.alpha - 0.04 * max_alpha,
                    0.93,
                    f"{row.label}\n$P_{{\\mathrm{{right}}}}={row.right_occupancy:.2f}$",
                    color=row.color,
                    fontsize=6.4,
                    ha="right",
                    va="top",
                )
        occupancy.set(
            xlim=(0, max_alpha),
            ylim=(0, 1.03),
            xlabel=r"Deformation strength, $\alpha$",
            ylabel="Equilibrium well occupancy",
        )
        panel_heading(occupancy, "c", "Occupancy shifts\ninside shared wells")
        occupancy.legend(
            frameon=True,
            facecolor=BACKGROUND,
            edgecolor="none",
            framealpha=1,
            loc="upper left",
            bbox_to_anchor=(0, 0.86),
            handlelength=1.6,
        )
        for band in data["bands"]:
            auc.axhspan(
                band["lower"], band["upper"], color=band["color"], alpha=0.13, zorder=0
            )
            # Label the range explicitly so its width cannot be read as uncertainty.
            above = band["lower"] >= 0.7
            y = band["upper"] + 0.007 if above else (band["lower"] + band["upper"]) / 2
            auc.text(
                0.04 * max_alpha if above else 0.97 * max_alpha,
                y,
                band["label"] + (" (outcome range)" if above else "\n(outcome range)"),
                color=band["color"],
                fontsize=6.3,
                va="bottom" if above else "center",
                ha="left" if above else "right",
            )
        auc.plot(
            curves.alpha,
            curves.auc,
            color=MODEL_COLOR,
            lw=1.8,
            label="Exact snapshot AUC",
        )
        auc.plot(
            curves.alpha,
            curves.auc_bound,
            color=BOUND_COLOR,
            lw=1.8,
            ls="--",
            label="Jeffreys bound",
        )
        auc.axhline(0.5, color=GRAY, ls="--", lw=1.0)
        for row in data["landmarks"].itertuples():
            auc.scatter(
                [row.alpha],
                [row.auc],
                s=38,
                marker=row.marker,
                color=row.color,
                edgecolor="white",
                linewidth=0.6,
                zorder=5,
            )
            if not row.show_distribution:
                auc.annotate(
                    row.label,
                    (row.alpha, row.auc),
                    xytext=(7, -8),
                    textcoords="offset points",
                    color=row.color,
                    fontsize=6.3,
                    ha="left",
                    va="top",
                )
        auc.set(
            xlim=(0, max_alpha),
            ylim=(0.48, 1.02),
            xlabel=r"Deformation strength, $\alpha$",
            ylabel="Optimal snapshot ROC-AUC",
        )
        panel_heading(auc, "d", "Same representation,\ndifferent separable AUC")
        auc.legend(frameon=False, loc="lower right", handlelength=1.8)
        for axis in axes.flat:
            axis.grid(False)
            axis.tick_params(length=3, width=0.9)
        potential.set_yticks([-1, 0, 1, 2, 3, 4])
        for axis in [potential, density]:
            axis.set_xticks([-1, -0.5, 0, 0.5, 1])
        fig.subplots_adjust(
            left=0.08,
            right=0.985,
            top=(0.93 * 5.55 + 0.30) / figure_height,
            bottom=0.11 * 5.55 / figure_height,
            wspace=0.32,
            hspace=0.42 + 0.30 / panel_height,
        )
    return fig


def make_mechanical_figure(config, quadrature):
    output = external_path(config["output_dir"])
    prefix = config.get("figure_prefix", "mechanical_figure")
    if not isinstance(prefix, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", prefix):
        raise ValueError("The figure prefix must be a simple filename stem.")
    data = calculate_figure(
        quadrature,
        config.get("alpha_values", np.linspace(0, 1.5, 151)),
        read_json(config["landmarks_file"]),
    )
    figure = render_figure(data)
    try:
        output.mkdir(parents=True, exist_ok=True)
        with matplotlib.rc_context({"pdf.fonttype": 42}):
            figure.savefig(
                external_path(output / f"{prefix}.png"), dpi=450, bbox_inches="tight"
            )
            figure.savefig(external_path(output / f"{prefix}.pdf"), bbox_inches="tight")
    finally:
        plt.close(figure)
    write_csv(output / f"{prefix}_curves.csv", data["curves"])
    write_csv(output / f"{prefix}_landmarks.csv", data["landmarks"])
    density_table = pd.concat(
        [
            pd.DataFrame(
                {
                    "label": item["label"],
                    "alpha": item["alpha"],
                    "x": data["x"],
                    "potential": item["potential"],
                    "density": item["density"],
                }
            )
            for item in data["distributions"]
        ],
        ignore_index=True,
    )
    write_csv(output / f"{prefix}_densities.csv", density_table)
    write_json(output / f"{prefix}_validation.json", data["report"])
    return data["curves"]
