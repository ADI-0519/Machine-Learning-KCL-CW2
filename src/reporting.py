"""Reporting-only analyses that may explicitly consume ground-truth labels."""

from __future__ import annotations

from numbers import Integral
from typing import Any

import numpy as np


def posthoc_class_balance(
    labels: np.ndarray,
    selected_indices: np.ndarray,
    *,
    num_classes: int,
) -> dict[str, Any]:
    """Summarize selected-label balance without participating in selection."""
    if not isinstance(num_classes, Integral) or isinstance(num_classes, (bool, np.bool_)):
        raise ValueError("num_classes must be a positive integer")
    num_classes = int(num_classes)
    if num_classes <= 0:
        raise ValueError("num_classes must be a positive integer")

    label_values = np.asarray(labels)
    if label_values.ndim != 1 or not np.issubdtype(label_values.dtype, np.integer):
        raise ValueError("labels must be a one-dimensional integer vector")
    label_values = label_values.astype(int, copy=False)
    if np.any(label_values < 0) or np.any(label_values >= num_classes):
        raise ValueError("labels contain a class outside [0, num_classes)")

    selected = np.asarray(selected_indices)
    if selected.ndim != 1 or not len(selected):
        raise ValueError("selected_indices cannot be empty and must be one-dimensional")
    if not np.issubdtype(selected.dtype, np.integer):
        raise ValueError("selected_indices must contain integers")
    selected = selected.astype(int, copy=False)
    if len(np.unique(selected)) != len(selected):
        raise ValueError("selected_indices must be unique")
    if np.any(selected < 0) or np.any(selected >= len(label_values)):
        raise ValueError("selected_indices contain an out-of-range index")

    counts = np.bincount(label_values[selected], minlength=num_classes)
    proportions = counts.astype(np.float64) / len(selected)
    nonzero = proportions > 0.0
    entropy = float(-(proportions[nonzero] * np.log(proportions[nonzero])).sum())
    normalized_entropy = 1.0 if num_classes == 1 else entropy / float(np.log(num_classes))
    normalized_entropy = float(np.clip(normalized_entropy, 0.0, 1.0))
    observed_classes = int(np.count_nonzero(counts))
    return {
        "class_counts": counts.astype(int).tolist(),
        "class_proportions": proportions.tolist(),
        "observed_classes": observed_classes,
        "class_coverage_fraction": observed_classes / num_classes,
        "normalized_entropy": normalized_entropy,
    }
