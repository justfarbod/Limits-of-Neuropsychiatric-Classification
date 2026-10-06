import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid, trapezoid

from ..utils.io import write_csv
from ..utils.paths import external_path


def triple_well(alpha, *, a=6.0, h=1.0, beta=1.5, bound=2.25, points=120001):
    if (
        min(a, h, beta, bound) <= 0
        or points < 101
        or not np.isfinite([alpha, a, h, beta, bound]).all()
    ):
        raise ValueError("Invalid quadrature parameters.")
    x = np.linspace(-bound, bound, int(points))
    potential = a * x * x * (x * x - h * h) ** 2
    reference_log = -beta * potential
    target_log = -beta * (potential - alpha * x)
    shift0, shift1 = reference_log.max(), target_log.max()
    weight0, weight1 = np.exp(reference_log - shift0), np.exp(target_log - shift1)
    z0, z1 = trapezoid(weight0, x), trapezoid(weight1, x)
    p0, p1 = weight0 / z0, weight1 / z1
    log_ratio = np.log(z1 / z0) + shift1 - shift0
    cdf0 = cumulative_trapezoid(p0, x, initial=0)
    auc = float(trapezoid(p1 * cdf0, x))
    if alpha < 0:
        auc = 1 - auc
    elif alpha == 0:
        auc = 0.5
    mean0, mean1 = trapezoid(p0 * x, x), trapezoid(p1 * x, x)
    forward = float(-beta * alpha * mean0 + log_ratio)
    reverse = float(beta * alpha * mean1 - log_ratio)
    reciprocal = float(beta * alpha * (mean1 - mean0))
    roots = np.roots([6 * a, 0, -8 * a * h * h, 0, 2 * a * h**4, -alpha])
    stationary = np.sort(roots.real[np.abs(roots.imag) < 1e-10])
    left = right = odds = np.nan
    if len(stationary) == 5 and stationary[0] > -bound and stationary[-1] < bound:
        cdf1 = cumulative_trapezoid(p1, x, initial=0)
        left = float(np.interp(stationary[1], x, cdf1))
        right = 1 - float(np.interp(stationary[3], x, cdf1))
        odds = right / left
    return {
        "alpha": float(alpha),
        "auc": auc,
        "mean_reference": float(mean0),
        "mean_target": float(mean1),
        "left_occupancy": left,
        "right_occupancy": right,
        "right_left_odds": odds,
        "forward_work": forward,
        "reverse_work": reverse,
        "reciprocal_work": reciprocal,
        "work_identity_error": reciprocal - forward - reverse,
    }


def run_mechanical(config):
    parameters = {
        key: config[key]
        for key in ["a", "h", "beta", "bound", "points"]
        if key in config
    }
    table = pd.DataFrame(
        [
            triple_well(alpha, **parameters)
            for alpha in config.get("alpha_values", np.linspace(0, 1.5, 31))
        ]
    )
    write_csv(external_path(config["output_dir"]) / "mechanical.csv", table)
    return table
