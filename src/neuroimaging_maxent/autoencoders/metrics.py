import numpy as np


def row_correlations(original, reconstructed):
    x = original - original.mean(axis=1, keepdims=True)
    y = reconstructed - reconstructed.mean(axis=1, keepdims=True)
    denominator = np.sqrt(np.sum(x * x, axis=1) * np.sum(y * y, axis=1))
    return np.divide(
        np.sum(x * y, axis=1),
        denominator,
        out=np.full(len(x), np.nan),
        where=denominator > 0,
    )


def reconstruction_metrics(original, reconstructed):
    squared = np.sum((original - reconstructed) ** 2, axis=1)
    total = np.sum((original - original.mean(axis=1, keepdims=True)) ** 2, axis=1)
    r2 = 1 - np.divide(squared, total, out=np.full(len(total), np.nan), where=total > 0)
    return {
        "mean_correlation": float(
            np.nanmean(row_correlations(original, reconstructed))
        ),
        "mean_rmse": float(np.mean(np.sqrt(squared / original.shape[1]))),
        "mean_r2": float(np.nanmean(r2)),
    }
