import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid, trapezoid

from ..utils.io import write_csv
from ..utils.paths import external_path


class TripleWellQuadrature:
    def __init__(self, *, a=6.0, h=1.0, beta=1.5, bound=2.25, points=120001):
        if (
            not np.isfinite([a, h, beta, bound, points]).all()
            or min(a, h, beta, bound) <= 0
            or points < 101
            or int(points) != points
        ):
            raise ValueError("Invalid quadrature parameters.")
        self.a, self.h, self.beta = float(a), float(h), float(beta)
        self.x = np.linspace(-bound, bound, int(points))
        self.base_potential = a * self.x**2 * (self.x**2 - 1) ** 2
        self.reference = self.equilibrium(0.0)
        self.reference_cdf = cumulative_trapezoid(
            self.reference["density"], self.x, initial=0
        )

    def equilibrium(self, alpha):
        if not np.isfinite(alpha):
            raise ValueError("Deformation must be finite.")
        potential = self.base_potential - alpha * self.h * self.x
        log_weight = -self.beta * potential
        shift = log_weight.max()
        weight = np.exp(log_weight - shift)
        z = trapezoid(weight, self.x)
        log_z = float(np.log(z) + shift)
        return {
            "potential": potential,
            "density": weight / z,
            "log_density": log_weight - log_z,
            "log_z": log_z,
        }

    def evaluate(self, alpha):
        target = self.equilibrium(alpha)
        p0, p1 = self.reference["density"], target["density"]
        auc = float(trapezoid(p1 * self.reference_cdf, self.x))
        if alpha < 0:
            auc = 1 - auc
        elif alpha == 0:
            auc = 0.5
        mean0, mean1 = trapezoid(p0 * self.x, self.x), trapezoid(p1 * self.x, self.x)
        log_z_ratio = target["log_z"] - self.reference["log_z"]
        forward = float(-self.beta * alpha * self.h * mean0 + log_z_ratio)
        reverse = float(self.beta * alpha * self.h * mean1 - log_z_ratio)
        reciprocal = float(self.beta * alpha * self.h * (mean1 - mean0))
        # Integrate both KL terms independently of the mean-based work identity.
        log_ratio = target["log_density"] - self.reference["log_density"]
        jeffreys = float(
            trapezoid(p1 * log_ratio, self.x) - trapezoid(p0 * log_ratio, self.x)
        )
        roots = np.roots([6 * self.a, 0, -8 * self.a, 0, 2 * self.a, -alpha * self.h])
        stationary = np.sort(roots.real[np.abs(roots.imag) < 1e-10])
        left = middle = right = odds = np.nan
        three_wells = bool(
            len(stationary) == 5
            and stationary[0] > self.x[0]
            and stationary[-1] < self.x[-1]
        )
        if three_wells:
            cdf = cumulative_trapezoid(p1, self.x, initial=0)
            # Interpolating at moving saddles avoids dropping boundary grid cells.
            left = float(np.interp(stationary[1], self.x, cdf))
            upper = float(np.interp(stationary[3], self.x, cdf))
            middle, right = upper - left, 1 - upper
            odds = right / left
        return {
            "alpha": float(alpha),
            "auc": auc,
            "auc_bound": min(1.0, 0.5 + np.sqrt(max(jeffreys, 0.0) / 8)),
            "jeffreys": jeffreys,
            "mean_reference": float(mean0),
            "mean_target": float(mean1),
            "left_occupancy": left,
            "middle_occupancy": middle,
            "right_occupancy": right,
            "right_left_odds": odds,
            "three_wells": three_wells,
            "density_mass": float(trapezoid(p1, self.x)),
            "forward_work": forward,
            "reverse_work": reverse,
            "reciprocal_work": reciprocal,
            "work_identity_error": reciprocal - forward - reverse,
            "jeffreys_work_error": jeffreys - (forward + reverse),
        }


def triple_well(alpha, **parameters):
    return TripleWellQuadrature(**parameters).evaluate(alpha)


def run_mechanical(config):
    parameters = {
        key: config[key]
        for key in ["a", "h", "beta", "bound", "points"]
        if key in config
    }
    quadrature = TripleWellQuadrature(**parameters)
    if config.get("make_figure", False):
        from .figure import make_mechanical_figure

        return make_mechanical_figure(config, quadrature)
    table = pd.DataFrame(
        [
            quadrature.evaluate(alpha)
            for alpha in config.get("alpha_values", np.linspace(0, 1.5, 31))
        ]
    )
    write_csv(external_path(config["output_dir"]) / "mechanical.csv", table)
    return table
