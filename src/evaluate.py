import numpy as np

def compute_class_distribution(labels,num_classes=10):
    """Compute per class counts and proportions for a label array."""
    counts = np.bincount(labels,minlength=num_classes)
    return counts / counts.sum()

def summarise_labels(labels,num_classes=10):
    """Return a compact class distribution summary for selected labels"""
    counts = np.bincount(labels,minlength=num_classes)
    proportions = counts / counts.sum()
    return {"counts":counts.tolist(), "proportions":proportions.tolist()} 