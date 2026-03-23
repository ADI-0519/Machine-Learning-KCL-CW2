import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .clustering import cluster_embeddings
from .config import load_configurations
from .data import (
    get_cifar10_test,
    get_cifar10_train,
    get_classifier_train_transform,
    get_eval_transform,
    make_subset_loader,
)
from .embeddings import grab_embeddings
from .evaluate import summarise_labels
from .models import SimCLRModel
from .seed import set_seed
from .selectors import (
    random_selector,
    tpcrand_selector,
    tpcrp_modified_selector,
    tpcrp_selector,
)
from .train_classifier import train_classifier


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_simclr_encoder(
    checkpoint_path: str | Path,
    projection_dim: int,
    device: torch.device,
):
    """
    Load a trained SimCLR checkpoint and return the encoder only.
    """
    model = SimCLRModel(proj_dim=projection_dim).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    return model.encoder


def build_embedding_loader(
    data_root: str,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    """
    Build deterministic loader for embedding extraction.
    """
    dataset = get_cifar10_train(root=data_root, transform=get_eval_transform())
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def load_or_compute_embeddings(
    embedding_path: str | Path,
    simclr_checkpoint_path: str | Path,
    projection_dim: int,
    data_root: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> np.ndarray:
    """
    Load cached embeddings if available, otherwise compute and save them.
    """
    embedding_path = Path(embedding_path)
    embedding_path.parent.mkdir(parents=True, exist_ok=True)

    if embedding_path.exists():
        print(f"Loading cached embeddings from {embedding_path}")
        return np.load(embedding_path)

    print("Cached embeddings not found. Computing embeddings from SimCLR encoder...")
    encoder = load_simclr_encoder(
        checkpoint_path=simclr_checkpoint_path,
        projection_dim=projection_dim,
        device=device,
    )

    loader = build_embedding_loader(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    embeddings = grab_embeddings(encoder=encoder, loader=loader, device=device)
    np.save(embedding_path, embeddings)
    print(f"Saved embeddings to {embedding_path}")

    return embeddings


def ensure_budget_size(
    selected_indices: np.ndarray,
    num_samples: int,
    budget: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Guard against rare cases where KMeans yields an empty cluster and the selector
    returns fewer than `budget` samples.

    Pads the selection with random, previously unselected indices.
    """
    selected_indices = np.unique(selected_indices).astype(int)

    if len(selected_indices) == budget:
        return np.sort(selected_indices)

    if len(selected_indices) > budget:
        return np.sort(selected_indices[:budget])

    missing = budget - len(selected_indices)
    all_indices = np.arange(num_samples)
    remaining = np.setdiff1d(all_indices, selected_indices, assume_unique=False)

    filler = rng.choice(remaining, size=missing, replace=False)
    completed = np.concatenate([selected_indices, filler])

    return np.sort(completed.astype(int))


def select_indices(
    method: str,
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    centroids: np.ndarray,
    budget: int,
    knn_k: int,
    modified_alpha: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Run one of the supported selection methods.
    """
    if method == "random":
        selected = random_selector(
            num_samples=len(embeddings),
            budget=budget,
            rng=rng,
        )
    elif method == "tpcrand":
        selected = tpcrand_selector(
            cluster_labels=cluster_labels,
            budget=budget,
            rng=rng,
        )
    elif method == "tpcrp":
        selected = tpcrp_selector(
            embeddings=embeddings,
            cluster_labels=cluster_labels,
            budget=budget,
            knn_k=knn_k,
        )
    elif method == "tpcrp_modified":
        selected = tpcrp_modified_selector(
            embeddings=embeddings,
            cluster_labels=cluster_labels,
            centroids=centroids,
            budget=budget,
            knn_k=knn_k,
            alpha=modified_alpha,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    return ensure_budget_size(
        selected_indices=selected,
        num_samples=len(embeddings),
        budget=budget,
        rng=rng,
    )


def save_selected_indices(
    selected_indices: np.ndarray,
    train_dataset,
    output_path: str | Path,
) -> None:
    """
    Save selected indices and their labels for later analysis.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = np.array(train_dataset.targets)[selected_indices]

    payload = {
        "selected_indices": selected_indices.tolist(),
        "selected_labels": labels.tolist(),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def append_metrics_row(metrics_path: str | Path, row: dict[str, Any]) -> None:
    """
    Append one experiment result row to a CSV file.
    """
    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not metrics_path.exists()

    with open(metrics_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_single_experiment(
    config_path: str | Path,
    method: str,
    budget: int,
    seed: int,
) -> dict[str, Any]:
    """
    Run one full experiment:
    embeddings -> clustering -> selection -> supervised training -> evaluation
    """
    cfg = load_configurations(config_path)
    set_seed(seed)

    rng = np.random.default_rng(seed)
    device = get_device()

    data_root = cfg["data"]["root"]
    num_workers = cfg["data"]["num_workers"]

    simclr_cfg = cfg["simclr"]
    selection_cfg = cfg["selection"]
    classifier_cfg = cfg["classifier"]

    embedding_dir = ensure_dir("./results/embeddings")
    selection_dir = ensure_dir("./results/selections")
    metrics_dir = ensure_dir("./results/metrics")
    checkpoint_dir = ensure_dir("./results/checkpoints")

    simclr_checkpoint_path = simclr_cfg["save_path"]
    checkpoint_stem = Path(simclr_checkpoint_path).stem
    embedding_path = embedding_dir / f"{checkpoint_stem}_train_embeddings.npy"

    # 1. Load or compute embeddings
    embeddings = load_or_compute_embeddings(
        embedding_path=embedding_path,
        simclr_checkpoint_path=simclr_checkpoint_path,
        projection_dim=simclr_cfg["projection_dim"],
        data_root=data_root,
        batch_size=simclr_cfg["batch_size"],
        num_workers=num_workers,
        device=device,
    )

    # 2. Cluster embeddings
    cluster_labels, centroids = cluster_embeddings(
        embeddings=embeddings,
        n_clusters=budget,
        random_state=seed,
    )

    # 3. Select indices
    selected_indices = select_indices(
        method=method,
        embeddings=embeddings,
        cluster_labels=cluster_labels,
        centroids=centroids,
        budget=budget,
        knn_k=selection_cfg["knn_k"],
        modified_alpha=selection_cfg["modified_alpha"],
        rng=rng,
    )
    assert len(selected_indices) == budget, f"Expected {budget} selected samples, got {len(selected_indices)}"
    assert len(np.unique(selected_indices)) == len(selected_indices), "Duplicate selected indices found"

    # 4. Build classifier datasets/loaders
    train_dataset = get_cifar10_train(
        root=data_root,
        transform=get_classifier_train_transform(),
    )
    test_dataset = get_cifar10_test(
        root=data_root,
        transform=get_eval_transform(),
    )

    train_loader = make_subset_loader(
        dataset=train_dataset,
        indices=selected_indices,
        batch_size=classifier_cfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=classifier_cfg["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # 5. Train downstream classifier
    classifier_ckpt_path = checkpoint_dir / f"{method}_budget{budget}_seed{seed}.pt"
    train_result = train_classifier(
        train_loader=train_loader,
        test_loader=test_loader,
        num_classes=10,
        epochs=classifier_cfg["epochs"],
        lr=classifier_cfg["lr"],
        momentum=classifier_cfg["momentum"],
        weight_decay=classifier_cfg["weight_decay"],
        device=device,
        checkpoint_path=classifier_ckpt_path,
        verbose=True,
    )

    # 6. Save selected indices and class distribution
    selection_output_path = selection_dir / f"{method}_budget{budget}_seed{seed}.json"
    save_selected_indices(
        selected_indices=selected_indices,
        train_dataset=train_dataset,
        output_path=selection_output_path,
    )

    selected_labels = np.array(train_dataset.targets)[selected_indices]
    selection_summary = summarise_labels(selected_labels, num_classes=10)

    # 7. Save metrics
    metrics_row = {
        "method": method,
        "budget": budget,
        "seed": seed,
        "best_epoch": train_result["best_epoch"],
        "best_test_accuracy": train_result["best_test_accuracy"],
        "final_test_accuracy": train_result["final_test_accuracy"],
        "final_test_loss": train_result["final_test_loss"],
        "num_selected": len(selected_indices),
    }
    append_metrics_row(metrics_dir / "metrics.csv", metrics_row)

    summary_output = {
        "metrics": metrics_row,
        "selection_summary": selection_summary,
        "selection_file": str(selection_output_path),
        "embedding_file": str(embedding_path),
        "classifier_checkpoint": str(classifier_ckpt_path),
    }

    print("\nExperiment complete:")
    print(json.dumps(summary_output, indent=2))

    return summary_output