import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ..utils.paths import external_path


def plot_results(config):
    table = pd.read_csv(external_path(config["results_file"], must_exist=True))
    output = external_path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    kind = config.get("plot_kind", "auc")
    fig, ax = plt.subplots(figsize=(6, 4))
    if kind == "marginals":
        if not {"observed", "modeled"}.issubset(table):
            raise ValueError("Marginal plots require observed and modeled columns.")
        ax.scatter(table.observed, table.modeled, s=8, alpha=0.35)
        ax.plot([0, 1], [0, 1], color="black", linewidth=0.8)
        ax.set(
            xlabel="Observed probability",
            ylabel="Model probability",
            xlim=(0, 1),
            ylim=(0, 1),
        )
    elif kind == "profile":
        wide = table.pivot(index="outcome", columns="method", values="auc")
        for method in config.get("methods", [c for c in wide if c != "maxent"]):
            ax.scatter(wide.maxent, wide[method], label=method)
        ax.plot([0.5, 1], [0.5, 1], color="black", linewidth=0.8)
        ax.set(xlabel="Maximum-entropy AUC", ylabel="Classifier AUC")
        ax.legend()
    elif "alpha" in table:
        for method, frame in (
            table.groupby("method") if "method" in table else [("quadrature", table)]
        ):
            frame = frame.sort_values("alpha")
            ax.plot(frame.alpha, frame.auc, label=method)
            if {"ci_lower", "ci_upper"}.issubset(frame):
                ax.fill_between(frame.alpha, frame.ci_lower, frame.ci_upper, alpha=0.15)
        ax.set(xlabel="Interpolation parameter", ylabel="AUC")
        ax.legend()
    else:
        if not {"outcome", "method", "auc"}.issubset(table):
            raise ValueError("AUC plots require outcome, method, and auc columns.")
        table.pivot(index="outcome", columns="method", values="auc").plot.bar(ax=ax)
        ax.set(xlabel="Outcome", ylabel="AUC")
    fig.tight_layout()
    if kind not in {"auc", "profile", "marginals"}:
        plt.close(fig)
        raise ValueError("Unknown plot kind.")
    path = external_path(output / f"{kind}.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
