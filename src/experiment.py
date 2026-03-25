import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, log_loss
from sklearn.semi_supervised import LabelSpreading
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
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
    kcenter_selector,
    random_selector,
    tpcinv_selector,
    tpcnoclust_selector,
    tpcrp_ccfl_selector,
    tpcrand_selector,
    tpcrp_modified_selector,
    tpcrp_selector,
)
from .typicality import compute_cluster_aware_scores, compute_typicality_scores
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
    model = SimCLRModel(proj_dim=projection_dim).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    try:
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    except RuntimeError as exc:
        raise RuntimeError(
            "Failed to load SimCLR checkpoint due to model architecture mismatch. "
            "If you recently changed the projector (e.g., added BatchNorm), retrain "
            "SimCLR and regenerate cached embeddings."
        ) from exc

    return model.encoder


def build_embedding_loader(data_root: str, split: str, batch_size: int, num_workers: int) -> DataLoader:
    if split == "train":
        dataset = get_cifar10_train(root=data_root, transform=get_eval_transform())
    else:
        dataset = get_cifar10_test(root=data_root, transform=get_eval_transform())

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
    split: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> np.ndarray:
    embedding_path = Path(embedding_path)
    embedding_path.parent.mkdir(parents=True, exist_ok=True)

    if embedding_path.exists():
        print(f"Loading cached {split} embeddings from {embedding_path}")
        return np.load(embedding_path)

    print(f"Cached {split} embeddings not found. Computing from SimCLR encoder...")
    encoder = load_simclr_encoder(
        checkpoint_path=simclr_checkpoint_path,
        projection_dim=projection_dim,
        device=device,
    )
    loader = build_embedding_loader(
        data_root=data_root,
        split=split,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    embeddings = grab_embeddings(encoder=encoder, loader=loader, device=device)
    np.save(embedding_path, embeddings)
    print(f"Saved {split} embeddings to {embedding_path}")
    return embeddings


def ensure_budget_size(
    selected_indices: np.ndarray,
    pool_size: int,
    budget: int,
    rng: np.random.Generator,
) -> np.ndarray:
    selected_indices = np.unique(selected_indices).astype(int)

    if len(selected_indices) == budget:
        return np.sort(selected_indices)
    if len(selected_indices) > budget:
        return np.sort(selected_indices[:budget])

    missing = budget - len(selected_indices)
    all_indices = np.arange(pool_size)
    remaining = np.setdiff1d(all_indices, selected_indices, assume_unique=False)
    filler = rng.choice(remaining, size=missing, replace=False)
    return np.sort(np.concatenate([selected_indices, filler]).astype(int))


def _select_from_embeddings(
    method: str,
    pool_embeddings: np.ndarray,
    query_size: int,
    knn_k: int,
    modified_alpha: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if method == "random":
        return random_selector(num_samples=len(pool_embeddings), budget=query_size, rng=rng)
    if method == "tpcnoclust":
        return tpcnoclust_selector(embeddings=pool_embeddings, budget=query_size, knn_k=knn_k)
    if method == "kcenter":
        return kcenter_selector(embeddings=pool_embeddings, budget=query_size)

    cluster_labels, centroids = cluster_embeddings(
        embeddings=pool_embeddings,
        n_clusters=query_size,
        random_state=int(rng.integers(0, 1_000_000_000)),
    )

    if method == "tpcrand":
        return tpcrand_selector(cluster_labels=cluster_labels, budget=query_size, rng=rng)
    if method == "tpcrp":
        return tpcrp_selector(
            embeddings=pool_embeddings,
            cluster_labels=cluster_labels,
            budget=query_size,
            knn_k=knn_k,
        )
    if method == "tpcrp_modified":
        return tpcrp_modified_selector(
            embeddings=pool_embeddings,
            cluster_labels=cluster_labels,
            centroids=centroids,
            budget=query_size,
            knn_k=knn_k,
            alpha=modified_alpha,
        )
    if method == "tpcinv":
        return tpcinv_selector(
            embeddings=pool_embeddings,
            cluster_labels=cluster_labels,
            budget=query_size,
            knn_k=knn_k,
        )

    raise ValueError(f"Unknown embedding-based selection method: {method}")


def _sort_clusters(uncovered: np.ndarray, sizes: np.ndarray, rng: np.random.Generator) -> list[int]:
    shuffled = uncovered.copy()
    rng.shuffle(shuffled)
    return sorted(shuffled.tolist(), key=lambda c: sizes[c], reverse=True)


def _select_cluster_based_round(
    method: str,
    full_embeddings: np.ndarray,
    pool_indices: np.ndarray,
    labeled_indices: np.ndarray,
    query_size: int,
    knn_k: int,
    modified_alpha: float,
    rng: np.random.Generator,
    max_clusters: int | None,
    min_cluster_size: int,
    ccfl_candidates_per_cluster: int,
    ccfl_refine_steps: int,
) -> np.ndarray:
    if query_size <= 0:
        return np.array([], dtype=int)

    n = len(full_embeddings)
    target_k = len(labeled_indices) + query_size
    if max_clusters is not None and max_clusters > 0:
        target_k = min(target_k, max_clusters)
    target_k = max(1, min(target_k, n))

    cluster_labels, centroids = cluster_embeddings(
        embeddings=full_embeddings,
        n_clusters=target_k,
        random_state=int(rng.integers(0, 1_000_000_000)),
    )

    cluster_sizes = np.bincount(cluster_labels, minlength=target_k)
    labeled_counts = np.zeros(target_k, dtype=int)
    if len(labeled_indices) > 0:
        labeled_counts = np.bincount(cluster_labels[labeled_indices], minlength=target_k)

    uncovered = np.where((labeled_counts == 0) & (cluster_sizes > 0))[0]
    ordered_clusters: list[int] = _sort_clusters(uncovered, cluster_sizes, rng)

    if len(ordered_clusters) < query_size:
        covered = np.setdiff1d(np.arange(target_k), uncovered, assume_unique=False)
        covered_list = covered.tolist()
        rng.shuffle(covered_list)
        covered_list.sort(key=lambda c: (labeled_counts[c], -cluster_sizes[c]))
        ordered_clusters.extend(covered_list)

    pool_list = pool_indices.tolist()
    pool_set = set(pool_list)
    pool_pos = {idx: pos for pos, idx in enumerate(pool_list)}
    selected: list[int] = []
    selected_set: set[int] = set()

    if method == "tpcrp_ccfl":
        selected_cluster_ids: list[int] = []
        for cluster_id in ordered_clusters:
            if len(selected_cluster_ids) >= query_size:
                break
            members = np.where(cluster_labels == cluster_id)[0]
            if len(members) == 0:
                continue
            has_pool_member = any(idx in pool_set for idx in members.tolist())
            if has_pool_member:
                selected_cluster_ids.append(int(cluster_id))

        ccfl_selected = tpcrp_ccfl_selector(
            embeddings=full_embeddings,
            cluster_labels=cluster_labels,
            centroids=centroids,
            selected_cluster_ids=selected_cluster_ids,
            pool_indices=pool_indices,
            knn_k=knn_k,
            candidates_per_cluster=ccfl_candidates_per_cluster,
            refine_steps=ccfl_refine_steps,
            cluster_sizes=cluster_sizes,
            min_cluster_size=min_cluster_size,
            rng=rng,
        )
        selected = [int(idx) for idx in ccfl_selected.tolist()]
        selected_set = set(selected)

        if len(selected) < query_size:
            remaining = [idx for idx in pool_indices.tolist() if idx not in selected_set]
            if remaining:
                filler = rng.choice(
                    np.array(remaining, dtype=int),
                    size=min(query_size - len(selected), len(remaining)),
                    replace=False,
                )
                selected.extend([int(x) for x in np.atleast_1d(filler)])

        selected = selected[:query_size]
        selected_local = [pool_pos[idx] for idx in selected if idx in pool_pos]
        return np.array(selected_local, dtype=int)

    for cluster_id in ordered_clusters:
        if len(selected) >= query_size:
            break

        members = np.where(cluster_labels == cluster_id)[0]
        if len(members) == 0:
            continue

        candidate_members = np.array([idx for idx in members if idx in pool_set], dtype=int)
        if len(candidate_members) == 0:
            continue

        if len(members) < min_cluster_size or method == "tpcrand":
            pick = int(rng.choice(candidate_members))
        else:
            cluster_emb = full_embeddings[members]
            if method == "tpcrp":
                scores = compute_typicality_scores(cluster_emb, knn_k)
            elif method == "tpcinv":
                scores = -compute_typicality_scores(cluster_emb, knn_k)
            elif method == "tpcrp_modified":
                scores = compute_cluster_aware_scores(
                    cluster_embeddings=cluster_emb,
                    centroid=centroids[cluster_id],
                    k=knn_k,
                    alpha=modified_alpha,
                )
            else:
                raise ValueError(f"Unknown cluster-based method: {method}")

            member_to_local = {m: i for i, m in enumerate(members.tolist())}
            candidate_locals = np.array([member_to_local[m] for m in candidate_members.tolist()], dtype=int)
            best_local = int(candidate_locals[np.argmax(scores[candidate_locals])])
            pick = int(members[best_local])

        if pick not in selected_set:
            selected.append(pick)
            selected_set.add(pick)

    if len(selected) < query_size:
        remaining = [idx for idx in pool_indices.tolist() if idx not in selected_set]
        if remaining:
            filler = rng.choice(np.array(remaining, dtype=int), size=min(query_size - len(selected), len(remaining)), replace=False)
            selected.extend([int(x) for x in np.atleast_1d(filler)])

    selected = selected[:query_size]
    selected_local = [pool_pos[idx] for idx in selected if idx in pool_pos]
    return np.array(selected_local, dtype=int)


def _select_from_probabilities(method: str, probs: np.ndarray, query_size: int) -> np.ndarray:
    if method == "uncertainty":
        scores = probs.max(axis=1)
        return np.argsort(scores)[:query_size]
    if method == "margin":
        top2 = np.sort(np.partition(probs, -2, axis=1)[:, -2:], axis=1)
        margins = top2[:, 1] - top2[:, 0]
        return np.argsort(margins)[:query_size]
    if method == "entropy":
        ent = -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1)
        return np.argsort(ent)[-query_size:]
    raise ValueError(f"Unknown uncertainty method: {method}")


@torch.no_grad()
def _predict_probs_linear_head(
    model: nn.Module,
    embeddings: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    x = torch.from_numpy(embeddings.astype(np.float32)).to(device, non_blocking=True)
    logits = model(x)
    return torch.softmax(logits, dim=1).cpu().numpy()


@torch.no_grad()
def _predict_mc_probs_linear_head(
    model: nn.Module,
    embeddings: np.ndarray,
    device: torch.device,
    mc_passes: int = 10,
) -> np.ndarray:
    """
    MC-dropout predictions for ssl_embedding.
    Returns shape (T, N, C).
    """
    prev_mode = model.training
    model.train()  # keep dropout active at inference for MC sampling
    x = torch.from_numpy(embeddings.astype(np.float32)).to(device, non_blocking=True)
    all_mc: list[np.ndarray] = []
    for _ in range(max(1, mc_passes)):
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_mc.append(probs)
    model.train(prev_mode)
    return np.stack(all_mc, axis=0)


@torch.no_grad()
def _predict_probs_torch_model(
    model: torch.nn.Module,
    dataset,
    indices: np.ndarray,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> np.ndarray:
    subset_loader = make_subset_loader(
        dataset=dataset,
        indices=indices.tolist(),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    model.eval()
    chunks: list[np.ndarray] = []
    for images, _ in subset_loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        chunks.append(probs)
    return np.concatenate(chunks, axis=0)


def _forward_logits_and_features_cifar(model: torch.nn.Module, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    features = model.forward_features(images)
    logits = model.forward_logits_from_features(features)
    return logits, features


@torch.no_grad()
def _predict_probs_and_features_torch_model(
    model: torch.nn.Module,
    dataset,
    indices: np.ndarray,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    subset_loader = make_subset_loader(
        dataset=dataset,
        indices=indices.tolist(),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    model.eval()
    probs_chunks: list[np.ndarray] = []
    feat_chunks: list[np.ndarray] = []
    for images, _ in subset_loader:
        images = images.to(device, non_blocking=True)
        logits, features = _forward_logits_and_features_cifar(model, images)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        probs_chunks.append(probs)
        feat_chunks.append(features.cpu().numpy())
    return np.concatenate(probs_chunks, axis=0), np.concatenate(feat_chunks, axis=0)


@torch.no_grad()
def _predict_mc_probs_torch_model(
    model: torch.nn.Module,
    dataset,
    indices: np.ndarray,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    mc_passes: int = 10,
    dropout_p: float = 0.2,
) -> np.ndarray:
    subset_loader = make_subset_loader(
        dataset=dataset,
        indices=indices.tolist(),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    model.eval()
    all_mc: list[np.ndarray] = []
    for _ in range(mc_passes):
        chunks: list[np.ndarray] = []
        for images, _ in subset_loader:
            images = images.to(device, non_blocking=True)
            _, features = _forward_logits_and_features_cifar(model, images)
            dropped = F.dropout(features, p=dropout_p, training=True)
            logits = model.forward_logits_from_features(dropped)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            chunks.append(probs)
        all_mc.append(np.concatenate(chunks, axis=0))
    return np.stack(all_mc, axis=0)  # (T, N, C)


def _badge_gradient_embeddings(probs: np.ndarray, features: np.ndarray) -> np.ndarray:
    num_classes = probs.shape[1]
    y_hat = np.argmax(probs, axis=1)
    one_hot = np.eye(num_classes)[y_hat]
    coeff = probs - one_hot
    grad = coeff[:, :, None] * features[:, None, :]
    return grad.reshape(len(features), -1)


def _kmeanspp_indices(embeddings: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = len(embeddings)
    if k >= n:
        return np.arange(n, dtype=int)

    norms = np.sum(embeddings * embeddings, axis=1)
    first = int(np.argmax(norms))
    selected = [first]

    d2 = np.sum((embeddings - embeddings[first]) ** 2, axis=1)
    for _ in range(1, k):
        total = float(d2.sum())
        if total <= 1e-12:
            remaining = np.setdiff1d(np.arange(n), np.array(selected, dtype=int), assume_unique=False)
            next_idx = int(rng.choice(remaining))
        else:
            probs = d2 / total
            next_idx = int(rng.choice(np.arange(n), p=probs))
        selected.append(next_idx)
        new_d2 = np.sum((embeddings - embeddings[next_idx]) ** 2, axis=1)
        d2 = np.minimum(d2, new_d2)
    return np.array(selected, dtype=int)


def _select_bald_from_mc(mc_probs: np.ndarray, query_size: int) -> np.ndarray:
    # BALD score = H[E[p(y|x,w)]] - E[H[p(y|x,w)]]
    mean_probs = mc_probs.mean(axis=0)
    entropy_mean = -(mean_probs * np.log(np.clip(mean_probs, 1e-12, 1.0))).sum(axis=1)
    entropy_each = -(mc_probs * np.log(np.clip(mc_probs, 1e-12, 1.0))).sum(axis=2)
    expected_entropy = entropy_each.mean(axis=0)
    mi = entropy_mean - expected_entropy
    return np.argsort(mi)[-query_size:]


def _predict_mc_probs_label_spreading(
    model: LabelSpreading,
    embeddings: np.ndarray,
    mc_passes: int = 10,
    dropout_p: float = 0.2,
) -> np.ndarray:
    """
    Lightweight stochastic predictions for semi_supervised selection.
    We apply feature dropout to embeddings at inference and query predict_proba.
    Returns shape (T, N, C).
    """
    p = float(np.clip(dropout_p, 0.0, 0.95))
    all_mc: list[np.ndarray] = []
    for _ in range(max(1, int(mc_passes))):
        if p > 0.0:
            mask = (np.random.rand(*embeddings.shape) >= p).astype(np.float32)
            noisy = embeddings.astype(np.float32) * mask / (1.0 - p)
        else:
            noisy = embeddings.astype(np.float32)
        probs = model.predict_proba(noisy)
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        row_sums = probs.sum(axis=1, keepdims=True)
        zero_rows = (row_sums <= 1e-12).reshape(-1)
        if np.any(zero_rows):
            probs[zero_rows] = 1.0 / probs.shape[1]
            row_sums = probs.sum(axis=1, keepdims=True)
        probs = probs / np.clip(row_sums, 1e-12, None)
        all_mc.append(probs)
    return np.stack(all_mc, axis=0)


def _round_query_sizes(total_budget: int, rounds: int) -> list[int]:
    base = total_budget // rounds
    rem = total_budget % rounds
    sizes = [base] * rounds
    for i in range(rem):
        sizes[i] += 1
    return sizes


def append_metrics_row(metrics_path: str | Path, row: dict[str, Any]) -> None:
    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = (not metrics_path.exists()) or metrics_path.stat().st_size == 0
    with open(metrics_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _train_eval_fully_supervised(
    selected_indices: np.ndarray,
    data_root: str,
    num_workers: int,
    classifier_cfg: dict[str, Any],
    device: torch.device,
    checkpoint_path: Path | None,
) -> dict[str, Any]:
    train_dataset = get_cifar10_train(root=data_root, transform=get_classifier_train_transform())
    test_dataset = get_cifar10_test(root=data_root, transform=get_eval_transform())
    train_loader = make_subset_loader(
        dataset=train_dataset,
        indices=selected_indices.tolist(),
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
    return train_classifier(
        train_loader=train_loader,
        test_loader=test_loader,
        num_classes=10,
        epochs=classifier_cfg["epochs"],
        lr=classifier_cfg["lr"],
        momentum=classifier_cfg["momentum"],
        weight_decay=classifier_cfg["weight_decay"],
        device=device,
        checkpoint_path=checkpoint_path,
        verbose=True,
    )


def _train_eval_ssl_embedding(
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    selected_indices: np.ndarray,
    epochs: int,
    classifier_cfg: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    x_train = torch.from_numpy(train_embeddings[selected_indices].astype(np.float32))
    y_train = torch.from_numpy(train_labels[selected_indices].astype(np.int64))
    x_test = torch.from_numpy(test_embeddings.astype(np.float32))
    y_test = torch.from_numpy(test_labels.astype(np.int64))

    linear_dropout_p = float(classifier_cfg.get("ssl_embedding_dropout_p", 0.2))
    linear = nn.Sequential(
        nn.Dropout(p=linear_dropout_p),
        nn.Linear(train_embeddings.shape[1], 10),
    ).to(device)
    criterion = nn.CrossEntropyLoss()

    # Paper-style linear eval uses a much higher LR than end-to-end supervised training.
    ssl_lr = float(classifier_cfg.get("ssl_embedding_lr", classifier_cfg["lr"] * 100.0))
    optimizer = SGD(
        linear.parameters(),
        lr=ssl_lr,
        momentum=classifier_cfg.get("momentum", 0.9),
        weight_decay=classifier_cfg.get("weight_decay", 0.0),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    best_acc = -1.0
    best_epoch = 1
    best_state = None
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        linear.train()
        optimizer.zero_grad(set_to_none=True)
        logits = linear(x_train.to(device, non_blocking=True))
        loss = criterion(logits, y_train.to(device, non_blocking=True))
        loss.backward()
        optimizer.step()
        scheduler.step()

        linear.eval()
        with torch.no_grad():
            test_logits = linear(x_test.to(device, non_blocking=True))
            probs_test = torch.softmax(test_logits, dim=1).cpu().numpy()
            preds_test = probs_test.argmax(axis=1)
            acc = accuracy_score(test_labels, preds_test)
            test_loss = log_loss(test_labels, probs_test, labels=np.arange(10))

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(loss.item()),
                "test_loss": float(test_loss),
                "test_accuracy": float(acc),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )

        if acc > best_acc:
            best_acc = float(acc)
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in linear.state_dict().items()}

    if best_state is None:
        raise RuntimeError("No best linear-head state was recorded for ssl_embedding.")

    linear.load_state_dict(best_state)
    linear.eval()
    with torch.no_grad():
        final_logits = linear(x_test.to(device, non_blocking=True))
        final_probs = torch.softmax(final_logits, dim=1).cpu().numpy()
        final_preds = final_probs.argmax(axis=1)
        final_acc = accuracy_score(test_labels, final_preds)
        final_loss = log_loss(test_labels, final_probs, labels=np.arange(10))

    return {
        "model": linear,
        "best_test_accuracy": float(best_acc),
        "best_epoch": int(best_epoch),
        "final_test_loss": float(final_loss),
        "final_test_accuracy": float(final_acc),
        "history": history,
    }


def _train_eval_semi_supervised(
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    selected_indices: np.ndarray,
) -> dict[str, Any]:
    y_semi = np.full(len(train_labels), -1, dtype=int)
    y_semi[selected_indices] = train_labels[selected_indices]

    model = LabelSpreading(kernel="knn", n_neighbors=7, alpha=0.2, max_iter=30)
    model.fit(train_embeddings, y_semi)

    probs_test_partial = model.predict_proba(test_embeddings)
    probs_test = np.zeros((len(test_embeddings), 10), dtype=np.float64)
    probs_test[:, model.classes_.astype(int)] = probs_test_partial

    # LabelSpreading can output NaNs in extreme low-label settings.
    probs_test = np.nan_to_num(probs_test, nan=0.0, posinf=0.0, neginf=0.0)
    row_sums = probs_test.sum(axis=1, keepdims=True)
    zero_rows = (row_sums <= 1e-12).reshape(-1)
    if np.any(zero_rows):
        probs_test[zero_rows] = 1.0 / probs_test.shape[1]
        row_sums = probs_test.sum(axis=1, keepdims=True)
    probs_test = probs_test / np.clip(row_sums, 1e-12, None)

    preds_test = probs_test.argmax(axis=1)
    acc = accuracy_score(test_labels, preds_test)
    loss = log_loss(test_labels, probs_test, labels=np.arange(10))
    return {
        "model": model,
        "best_test_accuracy": float(acc),
        "best_epoch": 1,
        "final_test_loss": float(loss),
        "final_test_accuracy": float(acc),
        "history": [],
    }


def run_single_experiment(
    config_path: str | Path,
    method: str,
    budget: int,
    seed: int,
    framework: str = "fully_supervised",
) -> dict[str, Any]:
    cfg = load_configurations(config_path)
    set_seed(seed)
    rng = np.random.default_rng(seed)
    device = get_device()

    data_root = cfg["data"]["root"]
    num_workers = cfg["data"]["num_workers"]
    simclr_cfg = cfg["simclr"]
    selection_cfg = cfg["selection"]
    classifier_cfg = cfg["classifier"]
    experiment_cfg = cfg.get("experiment", {})

    rounds = int(experiment_cfg.get("iterative_rounds", 1))
    initial_labeled = int(experiment_cfg.get("initial_labeled", 0))
    max_clusters = experiment_cfg.get("max_clusters")
    min_cluster_size = int(experiment_cfg.get("min_cluster_size", 5))
    ccfl_candidates_per_cluster = int(experiment_cfg.get("ccfl_candidates_per_cluster", 5))
    ccfl_refine_steps = int(experiment_cfg.get("ccfl_refine_steps", 1))

    embedding_dir = ensure_dir("./results/embeddings")
    selection_dir = ensure_dir("./results/selections")
    metrics_dir = ensure_dir("./results/metrics")
    checkpoint_dir = ensure_dir("./results/checkpoints")

    train_dataset_eval = get_cifar10_train(root=data_root, transform=get_eval_transform())
    train_labels = np.array(train_dataset_eval.targets)
    test_dataset_eval = get_cifar10_test(root=data_root, transform=get_eval_transform())
    test_labels = np.array(test_dataset_eval.targets)

    simclr_checkpoint_path = simclr_cfg["save_path"]
    checkpoint_stem = Path(simclr_checkpoint_path).stem

    train_emb_path = embedding_dir / f"{checkpoint_stem}_train_embeddings.npy"
    test_emb_path = embedding_dir / f"{checkpoint_stem}_test_embeddings.npy"

    need_embeddings = framework in {"ssl_embedding", "semi_supervised"} or method in {
        "tpcrand",
        "tpcrp",
        "tpcrp_ccfl",
        "tpcrp_modified",
        "tpcinv",
        "tpcnoclust",
        "kcenter",
    }

    train_embeddings = None
    test_embeddings = None
    if need_embeddings:
        train_embeddings = load_or_compute_embeddings(
            embedding_path=train_emb_path,
            simclr_checkpoint_path=simclr_checkpoint_path,
            projection_dim=simclr_cfg["projection_dim"],
            data_root=data_root,
            split="train",
            batch_size=simclr_cfg["batch_size"],
            num_workers=num_workers,
            device=device,
        )
        test_embeddings = load_or_compute_embeddings(
            embedding_path=test_emb_path,
            simclr_checkpoint_path=simclr_checkpoint_path,
            projection_dim=simclr_cfg["projection_dim"],
            data_root=data_root,
            split="test",
            batch_size=simclr_cfg["batch_size"],
            num_workers=num_workers,
            device=device,
        )

    all_indices = np.arange(len(train_labels), dtype=int)
    if initial_labeled > 0:
        initial_labeled = min(initial_labeled, budget)
        labeled_indices = rng.choice(all_indices, size=initial_labeled, replace=False).astype(int)
    else:
        labeled_indices = np.array([], dtype=int)

    model_state: Any = None
    round_sizes = _round_query_sizes(max(0, budget - len(labeled_indices)), rounds)
    round_metrics: list[dict[str, Any]] = []

    for round_id, query_size in enumerate(round_sizes, start=1):
        if query_size == 0:
            continue

        pool_indices = np.setdiff1d(all_indices, labeled_indices, assume_unique=False)
        if len(pool_indices) < query_size:
            query_size = len(pool_indices)

        if method == "random":
            local_selected = random_selector(num_samples=len(pool_indices), budget=query_size, rng=rng)
        elif method in {"uncertainty", "margin", "entropy", "dbal", "bald", "badge"} and model_state is not None:
            if framework == "fully_supervised":
                pool_dataset = get_cifar10_train(root=data_root, transform=get_eval_transform())
                if method == "badge":
                    probs_pool, feats_pool = _predict_probs_and_features_torch_model(
                        model=model_state,
                        dataset=pool_dataset,
                        indices=pool_indices,
                        batch_size=classifier_cfg["batch_size"],
                        num_workers=num_workers,
                        device=device,
                    )
                    grad_emb = _badge_gradient_embeddings(probs_pool, feats_pool)
                    local_selected = _kmeanspp_indices(grad_emb, query_size, rng)
                elif method in {"dbal", "bald"}:
                    mc_probs = _predict_mc_probs_torch_model(
                        model=model_state,
                        dataset=pool_dataset,
                        indices=pool_indices,
                        batch_size=classifier_cfg["batch_size"],
                        num_workers=num_workers,
                        device=device,
                        mc_passes=int(experiment_cfg.get("mc_passes", 10)),
                        dropout_p=float(experiment_cfg.get("mc_dropout_p", 0.2)),
                    )
                    if method == "dbal":
                        probs_pool = mc_probs.mean(axis=0)
                        local_selected = _select_from_probabilities("entropy", probs_pool, query_size)
                    else:
                        local_selected = _select_bald_from_mc(mc_probs, query_size)
                else:
                    probs_pool = _predict_probs_torch_model(
                        model=model_state,
                        dataset=pool_dataset,
                        indices=pool_indices,
                        batch_size=classifier_cfg["batch_size"],
                        num_workers=num_workers,
                        device=device,
                    )
                    local_selected = _select_from_probabilities(method=method, probs=probs_pool, query_size=query_size)
            elif framework == "ssl_embedding":
                if method == "badge":
                    local_selected = kcenter_selector(train_embeddings[pool_indices], query_size)
                elif method in {"dbal", "bald"}:
                    mc_probs = _predict_mc_probs_linear_head(
                        model=model_state,
                        embeddings=train_embeddings[pool_indices],
                        device=device,
                        mc_passes=int(experiment_cfg.get("mc_passes", 10)),
                    )
                    if method == "dbal":
                        probs_pool = mc_probs.mean(axis=0)
                        local_selected = _select_from_probabilities("entropy", probs_pool, query_size)
                    else:
                        local_selected = _select_bald_from_mc(mc_probs, query_size)
                else:
                    probs_pool = _predict_probs_linear_head(
                        model=model_state,
                        embeddings=train_embeddings[pool_indices],
                        device=device,
                    )
                    local_selected = _select_from_probabilities(method=method, probs=probs_pool, query_size=query_size)
            elif framework == "semi_supervised":
                if method == "badge":
                    local_selected = kcenter_selector(train_embeddings[pool_indices], query_size)
                elif method in {"dbal", "bald"}:
                    mc_probs = _predict_mc_probs_label_spreading(
                        model=model_state,
                        embeddings=train_embeddings[pool_indices],
                        mc_passes=int(experiment_cfg.get("mc_passes", 10)),
                        dropout_p=float(experiment_cfg.get("semi_mc_dropout_p", experiment_cfg.get("mc_dropout_p", 0.2))),
                    )
                    if method == "dbal":
                        probs_pool = mc_probs.mean(axis=0)
                        local_selected = _select_from_probabilities("entropy", probs_pool, query_size)
                    else:
                        local_selected = _select_bald_from_mc(mc_probs, query_size)
                else:
                    probs_pool = model_state.label_distributions_[pool_indices]
                    local_selected = _select_from_probabilities(method=method, probs=probs_pool, query_size=query_size)
            else:
                raise ValueError(f"Unknown framework: {framework}")
        elif method in {"uncertainty", "margin", "entropy", "dbal", "bald", "badge"}:
            # Cold-start fallback for uncertainty-based methods.
            local_selected = random_selector(num_samples=len(pool_indices), budget=query_size, rng=rng)
        else:
            cluster_based_methods = {"tpcrand", "tpcrp", "tpcrp_ccfl", "tpcrp_modified", "tpcinv"}
            if method in cluster_based_methods:
                local_selected = _select_cluster_based_round(
                    method=method,
                    full_embeddings=train_embeddings,
                    pool_indices=pool_indices,
                    labeled_indices=labeled_indices,
                    query_size=query_size,
                    knn_k=selection_cfg["knn_k"],
                    modified_alpha=selection_cfg["modified_alpha"],
                    rng=rng,
                    max_clusters=max_clusters,
                    min_cluster_size=min_cluster_size,
                    ccfl_candidates_per_cluster=ccfl_candidates_per_cluster,
                    ccfl_refine_steps=ccfl_refine_steps,
                )
            else:
                local_selected = _select_from_embeddings(
                    method=method,
                    pool_embeddings=train_embeddings[pool_indices],
                    query_size=query_size,
                    knn_k=selection_cfg["knn_k"],
                    modified_alpha=selection_cfg["modified_alpha"],
                    rng=rng,
                )

        local_selected = ensure_budget_size(
            selected_indices=local_selected,
            pool_size=len(pool_indices),
            budget=query_size,
            rng=rng,
        )
        newly_selected = pool_indices[local_selected]
        labeled_indices = np.unique(np.concatenate([labeled_indices, newly_selected])).astype(int)

        if framework == "fully_supervised":
            ckpt = checkpoint_dir / f"{framework}_{method}_budget{budget}_seed{seed}_r{round_id}.pt"
            train_result = _train_eval_fully_supervised(
                selected_indices=labeled_indices,
                data_root=data_root,
                num_workers=num_workers,
                classifier_cfg=classifier_cfg,
                device=device,
                checkpoint_path=ckpt,
            )
            model_state = train_result["model"]
        elif framework == "ssl_embedding":
            train_result = _train_eval_ssl_embedding(
                train_embeddings=train_embeddings,
                test_embeddings=test_embeddings,
                train_labels=train_labels,
                test_labels=test_labels,
                selected_indices=labeled_indices,
                epochs=classifier_cfg["epochs"],
                classifier_cfg=classifier_cfg,
                device=device,
            )
            model_state = train_result["model"]
        elif framework == "semi_supervised":
            train_result = _train_eval_semi_supervised(
                train_embeddings=train_embeddings,
                test_embeddings=test_embeddings,
                train_labels=train_labels,
                test_labels=test_labels,
                selected_indices=labeled_indices,
            )
            model_state = train_result["model"]
        else:
            raise ValueError(f"Unknown framework: {framework}")

        round_metrics.append(
            {
                "round": round_id,
                "num_selected": int(len(labeled_indices)),
                "test_accuracy": float(train_result["final_test_accuracy"]),
                "test_loss": float(train_result["final_test_loss"]),
            }
        )
        print(
            f"[{framework}][{method}] round {round_id}/{rounds} "
            f"selected={len(labeled_indices)} test_acc={train_result['final_test_accuracy']:.4f}"
        )

    if not round_metrics:
        raise RuntimeError("No rounds were executed. Check budget/round settings.")

    final_result = train_result
    selection_output_path = selection_dir / f"{framework}_{method}_budget{budget}_seed{seed}.json"
    payload = {
        "selected_indices": labeled_indices.tolist(),
        "selected_labels": train_labels[labeled_indices].tolist(),
        "round_metrics": round_metrics,
    }
    with open(selection_output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    selection_summary = summarise_labels(train_labels[labeled_indices], num_classes=10)

    metrics_row = {
        "framework": framework,
        "method": method,
        "budget": budget,
        "seed": seed,
        "rounds": rounds,
        "best_epoch": final_result["best_epoch"],
        "best_test_accuracy": final_result["best_test_accuracy"],
        "final_test_accuracy": final_result["final_test_accuracy"],
        "final_test_loss": final_result["final_test_loss"],
        "num_selected": len(labeled_indices),
    }
    append_metrics_row(metrics_dir / "metrics.csv", metrics_row)

    summary_output = {
        "metrics": metrics_row,
        "selection_summary": selection_summary,
        "selection_file": str(selection_output_path),
        "embedding_file_train": str(train_emb_path) if train_embeddings is not None else None,
        "embedding_file_test": str(test_emb_path) if test_embeddings is not None else None,
    }
    print("\nExperiment complete:")
    print(json.dumps(summary_output, indent=2))
    return summary_output
