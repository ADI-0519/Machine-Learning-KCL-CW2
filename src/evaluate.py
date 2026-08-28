import numpy as np


def _class_counts(labels: np.ndarray, num_classes: int) -> np.ndarray:
    values = np.asarray(labels)
    if values.ndim != 1 or not len(values) or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("labels must be a non-empty one-dimensional integer array")
    if not isinstance(num_classes, int) or isinstance(num_classes, bool) or num_classes <= 0:
        raise ValueError("num_classes must be a positive integer")
    if np.any(values < 0) or np.any(values >= num_classes):
        raise ValueError("labels contain a class outside [0, num_classes)")
    return np.bincount(values, minlength=num_classes)


def compute_class_distribution(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """Compute per-class proportions for a non-empty label array."""
    counts = _class_counts(labels, num_classes)
    return counts / counts.sum()


def summarise_labels(
    labels: np.ndarray, num_classes: int = 10
) -> dict[str, list[float] | list[int]]:
    """Return JSON-compatible class counts and proportions."""
    counts = _class_counts(labels, num_classes)
    proportions = counts / counts.sum()
    return {"counts": counts.tolist(), "proportions": proportions.tolist()}
