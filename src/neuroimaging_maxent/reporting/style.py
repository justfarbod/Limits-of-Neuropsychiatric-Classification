"""Typography and palette shared by the paper's analysis figures."""

REFERENCE_COLOR = "#666666"
LOW_TILT_COLOR = "#CC6677"
HIGH_TILT_COLOR = "#2171B5"
INTERMEDIATE_COLOR = "#332288"
SECONDARY_COLOR = "#6BAED6"
MODEL_COLOR = "#087E8B"
BOUND_COLOR = "#E4572E"
BACKGROUND = "#FCFDFE"
EDGE = "#8C949D"
MUTED_INK = "#626A73"

PAPER_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Arial",
        "Helvetica",
        "Liberation Sans",
        "Nimbus Sans",
        "DejaVu Sans",
    ],
    "font.size": 8.5,
    "axes.labelsize": 8.6,
    "axes.titlesize": 9.5,
    "axes.titleweight": "bold",
    "axes.linewidth": 0.9,
    "axes.edgecolor": EDGE,
    "axes.facecolor": BACKGROUND,
    "xtick.labelsize": 7.3,
    "ytick.labelsize": 7.3,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "legend.frameon": False,
    "legend.fontsize": 7.0,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "mathtext.fontset": "dejavusans",
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
}


def panel_heading(axis, letter, title):
    axis.text(
        -0.16,
        1.075,
        letter,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=12,
        weight="bold",
    )
    axis.text(
        0.5,
        1.075,
        title,
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=9.5,
        weight="bold",
    )
