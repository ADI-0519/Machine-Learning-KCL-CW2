import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, log_loss
from sklearn.semi_supervised import LabelSpreading
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .artifacts import (
    artifact_path_for,
    atomic_write_json,
    build_effective_config,
    build_run_artifact,
    build_run_id,
    canonical_json,
    collect_environment,
    config_digest,
)
from .clustering import cluster_embeddings
from .config import CCFL_METHODS, PROTOCOL_METHODS, load_configurations, validate_protocol_config
from .data import (
    get_cifar10_test,
    get_cifar10_train,
    get_classifier_train_transform,
    get_eval_transform,
    make_subset_loader,
)
from .diagnostics import selection_diagnostics, timed_selection
from .embeddings import grab_embeddings
from .protocol import (
    EvaluationMetrics,
    SeedBundle,
    TrainingOutcome,
    derive_seed,
    validate_query,
    validate_round_query_sizes,
)
from .representations import (
    build_embedding_transform,
    load_representation_encoder,
    representation_cache_path,
)
from .seed import make_generator, seed_worker, set_seed
from .selectors import (
    decimal_grid,
    estimate_probcover_delta,
    kcenter_selector,
    probcover_selector,
    random_selector,
    tpcrp_ccfl_selector,
)
from .train_classifier import train_classifier
from .typicality import compute_selected_typicality_scores, compute_typicality_scores


def get_device() -> torch.device:
    """Select CUDA if available, otherwise fall back to CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_embedding_loader(
    data_root: str,
    split: str,
    batch_size: int,
    num_workers: int,
    *,
    seed: int,
    transform: Any | None = None,
) -> DataLoader:
    """Build a non-shuffled loader for train/test embedding extraction"""
    if transform is None:
        transform = get_eval_transform()
    if split == "train":
        dataset = get_cifar10_train(root=data_root, transform=transform)
    elif split == "test":
        dataset = get_cifar10_test(root=data_root, transform=transform)
    else:
        raise ValueError(f"unsupported embedding split: {split}")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=make_generator(seed),
    )


def load_or_compute_embeddings(
    embedding_path: str | Path,
    representation: dict[str, Any],
    data_root: str,
    split: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    *,
    dataloader_seed: int,
) -> np.ndarray:
    """Load cached embeddings or compute them from the frozen representation."""
    embedding_path = Path(embedding_path)
    embedding_path.parent.mkdir(parents=True, exist_ok=True)

    if embedding_path.exists():
        print(f"Loading cached {split} embeddings from {embedding_path}")
        return np.load(embedding_path)

    backend = representation["backend"]
    print(f"Cached {split} embeddings not found. Computing with {backend}...")
    encoder = load_representation_encoder(representation, device)
    loader = build_embedding_loader(
        data_root=data_root,
        split=split,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=dataloader_seed,
        transform=build_embedding_transform(representation),
    )
    embeddings = grab_embeddings(encoder=encoder, loader=loader, device=device)
    np.save(embedding_path, embeddings)
    print(f"Saved {split} embeddings to {embedding_path}")
    return embeddings


def _embedding_digest(embeddings: np.ndarray) -> str:
    """Hash an embedding array including dtype, shape, and byte content."""
    contiguous = np.ascontiguousarray(embeddings)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(canonical_json(list(contiguous.shape)).encode("ascii"))
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _load_or_estimate_probcover_delta(
    *,
    cache_root: str | Path,
    embeddings: np.ndarray,
    num_classes: int,
    candidates: np.ndarray,
    alpha: float,
    clustering_seed: int,
) -> tuple[float, str]:
    """Return a ProbCover radius cached under a digest of every estimator input."""
    cache_inputs = {
        "algorithm": "probcover_pseudo_label_purity_v1",
        "alpha": float(alpha),
        "candidates": np.asarray(candidates, dtype=np.float64).tolist(),
        "clustering_seed": int(clustering_seed),
        "embedding_digest": _embedding_digest(embeddings),
        "num_classes": int(num_classes),
    }
    digest = config_digest(cache_inputs)
    cache_path = Path(cache_root) / f"{digest}.json"
    if cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid ProbCover radius cache at {cache_path}: {exc}") from exc
        if (
            not isinstance(payload, dict)
            or set(payload) != {"cache_digest", "delta", "inputs"}
            or payload["cache_digest"] != digest
            or payload["inputs"] != cache_inputs
            or not isinstance(payload["delta"], (int, float))
            or float(payload["delta"]) not in cache_inputs["candidates"]
        ):
            raise ValueError(f"invalid ProbCover radius cache at {cache_path}: content mismatch")
        return float(payload["delta"]), digest

    delta = estimate_probcover_delta(
        embeddings,
        num_classes=num_classes,
        candidates=np.asarray(candidates, dtype=np.float64),
        alpha=alpha,
        clustering_seed=clustering_seed,
    )
    atomic_write_json(
        cache_path,
        {"cache_digest": digest, "delta": delta, "inputs": cache_inputs},
    )
    return delta, digest


def _pool_local_to_global_query(
    local_query: np.ndarray,
    *,
    pool_indices: np.ndarray,
    query_size: int,
) -> np.ndarray:
    """Convert validated pool-local indices into global training indices."""
    validate_query(
        local_query,
        pool_indices=np.arange(len(pool_indices), dtype=int),
        query_size=query_size,
    )
    global_query = np.asarray(pool_indices, dtype=int)[np.asarray(local_query, dtype=int)]
    validate_query(
        global_query,
        pool_indices=pool_indices,
        query_size=query_size,
    )
    return global_query


def _sort_clusters(uncovered: np.ndarray, sizes: np.ndarray, rng: np.random.Generator) -> list[int]:
    """Random tie break then sort cluster IDs by descending cluster size"""
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
    rng: np.random.Generator,
    max_clusters: int,
    min_cluster_size: int,
    ccfl_variant: dict[str, int | bool] | None,
    *,
    clustering_seed: int,
) -> np.ndarray:
    """Perform one cluster query round and return global training indices."""
    if query_size <= 0:
        return np.array([], dtype=int)

    n = len(full_embeddings)
    target_k = len(labeled_indices) + query_size
    target_k = min(target_k, max_clusters)
    target_k = max(1, min(target_k, n))

    cluster_labels, centroids = cluster_embeddings(
        embeddings=full_embeddings,
        n_clusters=target_k,
        random_state=clustering_seed,
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
    selected: list[int] = []
    selected_set: set[int] = set()

    if method in CCFL_METHODS:
        if ccfl_variant is None:
            raise ValueError(f"method={method} requires explicit CCFL variant parameters")
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
            candidates_per_cluster=int(ccfl_variant["candidates_per_cluster"]),
            refine_steps=int(ccfl_variant["refine_steps"]),
            use_cluster_weights=bool(ccfl_variant["use_cluster_weights"]),
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
        return np.array(selected, dtype=int)

    for cluster_id in ordered_clusters:
        if len(selected) >= query_size:
            break

        members = np.where(cluster_labels == cluster_id)[0]
        if len(members) == 0:
            continue

        candidate_members = np.array([idx for idx in members if idx in pool_set], dtype=int)
        if len(candidate_members) == 0:
            continue

        if len(members) < min_cluster_size:
            pick = int(rng.choice(candidate_members))
        else:
            cluster_emb = full_embeddings[members]
            if method == "tpcrp":
                scores = compute_typicality_scores(cluster_emb, knn_k)
            else:
                raise ValueError(f"method is not enabled by protocol v2: {method}")

            member_to_local = {m: i for i, m in enumerate(members.tolist())}
            candidate_locals = np.array(
                [member_to_local[m] for m in candidate_members.tolist()], dtype=int
            )
            best_local = int(candidate_locals[np.argmax(scores[candidate_locals])])
            pick = int(members[best_local])

        if pick not in selected_set:
            selected.append(pick)
            selected_set.add(pick)

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
    return np.array(selected, dtype=int)


def _select_protocol_round(
    *,
    method: str,
    full_embeddings: np.ndarray,
    pool_indices: np.ndarray,
    labeled_indices: np.ndarray,
    query_size: int,
    knn_k: int,
    rng: np.random.Generator,
    max_clusters: int,
    min_cluster_size: int,
    ccfl_variant: dict[str, int | bool] | None,
    probcover_delta: float | None,
    clustering_seed: int,
) -> np.ndarray:
    """Select one protocol-v2 query and return pool-only global indices."""
    if method not in PROTOCOL_METHODS:
        raise ValueError(f"method is not enabled by protocol v2: {method}")
    if method == "random":
        local_query = random_selector(
            num_samples=len(pool_indices),
            budget=query_size,
            rng=rng,
        )
        return _pool_local_to_global_query(
            local_query,
            pool_indices=pool_indices,
            query_size=query_size,
        )
    if method == "kcenter":
        return kcenter_selector(
            embeddings=full_embeddings,
            pool_indices=pool_indices,
            labeled_indices=labeled_indices,
            query_size=query_size,
        )
    if method == "probcover":
        if probcover_delta is None:
            raise ValueError("method=probcover requires an estimated radius")
        return probcover_selector(
            embeddings=full_embeddings,
            pool_indices=pool_indices,
            labeled_indices=labeled_indices,
            query_size=query_size,
            delta=probcover_delta,
        )
    return _select_cluster_based_round(
        method=method,
        full_embeddings=full_embeddings,
        pool_indices=pool_indices,
        labeled_indices=labeled_indices,
        query_size=query_size,
        knn_k=knn_k,
        rng=rng,
        max_clusters=max_clusters,
        min_cluster_size=min_cluster_size,
        ccfl_variant=ccfl_variant,
        clustering_seed=clustering_seed,
    )


def _train_eval_fully_supervised(
    selected_indices: np.ndarray,
    data_root: str,
    num_workers: int,
    classifier_cfg: dict[str, Any],
    device: torch.device,
    checkpoint_path: Path | None,
    *,
    dataloader_seed: int,
    num_classes: int,
) -> TrainingOutcome:
    """Train and evaluate the supervised CNN on current labeled indices"""
    train_dataset = get_cifar10_train(root=data_root, transform=get_classifier_train_transform())
    test_dataset = get_cifar10_test(root=data_root, transform=get_eval_transform())
    train_loader = make_subset_loader(
        dataset=train_dataset,
        indices=selected_indices.tolist(),
        batch_size=classifier_cfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        seed=dataloader_seed,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=classifier_cfg["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=make_generator(dataloader_seed),
    )
    return train_classifier(
        train_loader=train_loader,
        test_loader=test_loader,
        num_classes=num_classes,
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
    *,
    num_classes: int,
) -> TrainingOutcome:
    """Train a fixed-epoch linear head, then evaluate test embeddings once."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")

    x_train = torch.from_numpy(train_embeddings[selected_indices].astype(np.float32))
    y_train = torch.from_numpy(train_labels[selected_indices].astype(np.int64))
    x_test = torch.from_numpy(test_embeddings.astype(np.float32))

    linear_dropout_p = float(classifier_cfg["ssl_embedding_dropout_p"])
    linear = nn.Sequential(
        nn.Dropout(p=linear_dropout_p),
        nn.Linear(train_embeddings.shape[1], num_classes),
    ).to(device)
    criterion = nn.CrossEntropyLoss()

    # Paper-style linear eval uses a much higher LR than end-to-end supervised training.
    ssl_lr = float(classifier_cfg["ssl_embedding_lr"])
    optimizer = SGD(
        linear.parameters(),
        lr=ssl_lr,
        momentum=classifier_cfg["momentum"],
        weight_decay=classifier_cfg["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    history: list[dict[str, float | int]] = []
    x_train_device = x_train.to(device, non_blocking=True)
    y_train_device = y_train.to(device, non_blocking=True)

    for epoch in range(1, epochs + 1):
        linear.train()
        optimizer.zero_grad(set_to_none=True)
        logits = linear(x_train_device)
        loss = criterion(logits, y_train_device)
        loss.backward()
        optimizer.step()
        scheduler.step()

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(loss.item()),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )

    linear.eval()
    with torch.no_grad():
        test_logits = linear(x_test.to(device, non_blocking=True))
        test_probs = torch.softmax(test_logits, dim=1).cpu().numpy()

    metrics = EvaluationMetrics(
        trained_epochs=epochs,
        test_loss=float(log_loss(test_labels, test_probs, labels=np.arange(num_classes))),
        test_accuracy=float(accuracy_score(test_labels, test_probs.argmax(axis=1))),
    )
    return TrainingOutcome(model=linear, metrics=metrics, history=history)


def _train_eval_label_spreading_proxy(
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    selected_indices: np.ndarray,
    *,
    num_classes: int,
) -> TrainingOutcome:
    """Fit the LabelSpreading proxy and evaluate test embeddings exactly once."""
    y_semi = np.full(len(train_labels), -1, dtype=int)
    y_semi[selected_indices] = train_labels[selected_indices]

    model = LabelSpreading(kernel="knn", n_neighbors=7, alpha=0.2, max_iter=30)
    model.fit(train_embeddings, y_semi)

    probs_test_partial = model.predict_proba(test_embeddings)
    probs_test = np.zeros((len(test_embeddings), num_classes), dtype=np.float64)
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
    loss = log_loss(test_labels, probs_test, labels=np.arange(num_classes))
    metrics = EvaluationMetrics(
        trained_epochs=1,
        test_loss=float(loss),
        test_accuracy=float(acc),
    )
    return TrainingOutcome(model=model, metrics=metrics, history=[])


def run_single_experiment(
    config_path: str | Path,
    method: str,
    seed: int,
    framework: str = "fully_supervised",
) -> dict[str, Any]:
    """Execute and atomically persist one protocol-v2 acquisition trajectory."""
    cfg = load_configurations(config_path)
    validate_protocol_config(cfg)
    if method not in cfg["experiment"]["methods"]:
        raise ValueError(f"method is not configured: {method}")
    if seed not in cfg["experiment"]["replicate_seeds"]:
        raise ValueError(f"replicate seed is not configured: {seed}")
    if framework != cfg["evaluation"]["framework"]:
        raise ValueError(
            f"framework mismatch: requested={framework} configured={cfg['evaluation']['framework']}"
        )

    effective_config = build_effective_config(
        cfg,
        method=method,
        replicate_seed=seed,
        framework=framework,
    )
    run_id = build_run_id(effective_config)
    output_root = Path(cfg["output"]["root"])
    run_artifact_path = artifact_path_for(output_root, run_id)
    if run_artifact_path.exists():
        raise FileExistsError(f"immutable run artifact already exists: {run_artifact_path}")

    run_started = time.perf_counter()
    set_seed(seed)
    device = get_device()

    data_root = cfg["data"]["root"]
    num_workers = cfg["data"]["num_workers"]
    num_classes = int(cfg["data"]["num_classes"])
    representation_cfg = cfg["representation"]
    selection_cfg = cfg["selection"]
    evaluation_cfg = cfg["evaluation"]
    ccfl_variant = cfg["ccfl_variants"][method] if method in CCFL_METHODS else None
    classifier_cfg = {
        "batch_size": evaluation_cfg["batch_size"],
        "epochs": evaluation_cfg["epochs"],
        "lr": evaluation_cfg["lr"],
        "momentum": evaluation_cfg["momentum"],
        "weight_decay": evaluation_cfg["weight_decay"],
        "ssl_embedding_lr": evaluation_cfg["lr"],
        "ssl_embedding_dropout_p": evaluation_cfg["dropout_p"],
    }

    round_query_sizes = validate_round_query_sizes(selection_cfg["round_query_sizes"])
    rounds = len(round_query_sizes)
    max_clusters = selection_cfg["max_clusters"]
    min_cluster_size = int(selection_cfg["min_cluster_size"])

    embedding_dir = ensure_dir(output_root / "cache" / "embeddings")
    checkpoint_dir = ensure_dir(output_root / "checkpoints")

    representation_checkpoint_path = representation_cfg["checkpoint_path"]
    environment = collect_environment(
        device=device,
        checkpoint_path=representation_checkpoint_path,
    )
    if environment["git_dirty"]:
        raise RuntimeError(
            "protocol-v2 runs require a clean Git worktree so every result maps to "
            "reviewed, immutable code"
        )
    checkpoint_digest = environment["checkpoint_sha256"]
    if not isinstance(checkpoint_digest, str):
        raise RuntimeError("representation checkpoint provenance is incomplete")
    checkpoint_cache_key = checkpoint_digest[:12]
    train_emb_path = representation_cache_path(
        embedding_dir,
        dataset=cfg["data"]["name"],
        split="train",
        representation=representation_cfg,
        checkpoint_sha256=checkpoint_digest,
    )
    test_emb_path = representation_cache_path(
        embedding_dir,
        dataset=cfg["data"]["name"],
        split="test",
        representation=representation_cfg,
        checkpoint_sha256=checkpoint_digest,
    )

    train_dataset_eval = get_cifar10_train(root=data_root, transform=get_eval_transform())
    train_labels = np.array(train_dataset_eval.targets)
    test_dataset_eval = get_cifar10_test(root=data_root, transform=get_eval_transform())
    test_labels = np.array(test_dataset_eval.targets)
    for split_name, labels in (("train", train_labels), ("test", test_labels)):
        if (
            labels.ndim != 1
            or not len(labels)
            or not np.issubdtype(labels.dtype, np.integer)
            or np.any(labels < 0)
            or np.any(labels >= num_classes)
        ):
            raise ValueError(
                f"{split_name} labels must be non-empty integers in [0, {num_classes})"
            )

    train_embeddings = load_or_compute_embeddings(
        embedding_path=train_emb_path,
        representation=representation_cfg,
        data_root=data_root,
        split="train",
        batch_size=representation_cfg["batch_size"],
        num_workers=num_workers,
        device=device,
        dataloader_seed=int(representation_cfg["embedding_seed"]),
    )
    test_embeddings = load_or_compute_embeddings(
        embedding_path=test_emb_path,
        representation=representation_cfg,
        data_root=data_root,
        split="test",
        batch_size=representation_cfg["batch_size"],
        num_workers=num_workers,
        device=device,
        dataloader_seed=int(representation_cfg["embedding_seed"]),
    )
    if len(train_embeddings) != len(train_labels) or len(test_embeddings) != len(test_labels):
        raise ValueError("embedding cache lengths do not match their dataset splits")
    if (
        train_embeddings.ndim != 2
        or test_embeddings.ndim != 2
        or train_embeddings.shape[1] == 0
        or train_embeddings.shape[1] != test_embeddings.shape[1]
        or not np.issubdtype(train_embeddings.dtype, np.number)
        or not np.issubdtype(test_embeddings.dtype, np.number)
        or not np.isfinite(train_embeddings).all()
        or not np.isfinite(test_embeddings).all()
    ):
        raise ValueError("embedding caches must be finite numeric matrices with matching widths")
    if (
        representation_cfg["backend"] == "dinov2"
        and train_embeddings.shape[1] != int(representation_cfg["feature_dim"])
    ):
        raise ValueError("DINOv2 embedding width does not match representation.feature_dim")

    probcover_delta: float | None = None
    probcover_radius_seed: int | None = None
    probcover_cache_digest: str | None = None
    if method == "probcover":
        delta_search = cfg["probcover"]["delta_search"]
        candidates = decimal_grid(
            delta_search["minimum"],
            delta_search["maximum"],
            delta_search["step"],
        )
        probcover_radius_seed = derive_seed(
            int(representation_cfg["embedding_seed"]),
            cfg["data"]["name"],
            checkpoint_cache_key,
            "probcover_radius",
        )
        probcover_delta, probcover_cache_digest = _load_or_estimate_probcover_delta(
            cache_root=output_root / "cache" / "probcover_delta",
            embeddings=train_embeddings,
            num_classes=num_classes,
            candidates=candidates,
            alpha=float(cfg["probcover"]["alpha"]),
            clustering_seed=probcover_radius_seed,
        )

    all_indices = np.arange(len(train_labels), dtype=int)
    selected_indices_ordered: list[int] = []
    selected_membership: set[int] = set()
    round_metrics: list[dict[str, Any]] = []
    round_seconds: list[float] = []
    selector_seconds_by_round: list[float] = []

    for round_id, query_size in enumerate(round_query_sizes, start=1):
        round_started = time.perf_counter()
        pool_indices = np.array(
            [index for index in all_indices if int(index) not in selected_membership],
            dtype=int,
        )
        labeled_indices = np.array(selected_indices_ordered, dtype=int)
        if query_size > len(pool_indices):
            raise ValueError(
                f"method={method} round={round_id} query_size={query_size} "
                f"exceeds pool={len(pool_indices)}"
            )

        round_seeds = SeedBundle.for_round(
            replicate=seed,
            dataset=str(cfg["data"]["name"]),
            framework=framework,
            method=method,
            round_id=round_id,
        )
        selector_rng = np.random.default_rng(round_seeds.selector)
        set_seed(round_seeds.selector)
        newly_selected, selector_seconds = timed_selection(
            _select_protocol_round,
            method=method,
            full_embeddings=train_embeddings,
            pool_indices=pool_indices,
            labeled_indices=labeled_indices,
            query_size=query_size,
            knn_k=int(selection_cfg["knn_k"]),
            rng=selector_rng,
            max_clusters=int(max_clusters),
            min_cluster_size=min_cluster_size,
            ccfl_variant=ccfl_variant,
            probcover_delta=probcover_delta,
            clustering_seed=round_seeds.clustering,
        )

        validate_query(
            newly_selected,
            pool_indices=pool_indices,
            query_size=query_size,
        )
        previous_order = selected_indices_ordered.copy()
        selected_indices_ordered.extend(int(index) for index in newly_selected)
        selected_membership.update(int(index) for index in newly_selected)
        if selected_indices_ordered[: len(previous_order)] != previous_order:
            raise RuntimeError(f"method={method} round={round_id} violated nestedness")
        if len(selected_membership) != len(selected_indices_ordered):
            raise RuntimeError(f"method={method} round={round_id} selected duplicates")
        labeled_indices = np.array(selected_indices_ordered, dtype=int)
        remaining_indices = np.array(
            [index for index in all_indices if int(index) not in selected_membership],
            dtype=int,
        )
        if not len(remaining_indices):
            raise ValueError(
                f"method={method} round={round_id} leaves no unlabeled points for diagnostics"
            )
        selected_typicality_scores = compute_selected_typicality_scores(
            train_embeddings,
            labeled_indices,
            int(selection_cfg["knn_k"]),
        )
        round_diagnostics = selection_diagnostics(
            train_embeddings,
            labeled_indices,
            evaluation_indices=remaining_indices,
            selected_typicality_scores=selected_typicality_scores,
        )
        if method in CCFL_METHODS:
            if ccfl_variant is None:
                raise RuntimeError(f"method={method} has no resolved CCFL metadata")
            method_metadata: dict[str, Any] = {
                "ccfl_variant": method,
                "candidates_per_cluster": int(ccfl_variant["candidates_per_cluster"]),
                "refine_steps": int(ccfl_variant["refine_steps"]),
                "use_cluster_weights": bool(ccfl_variant["use_cluster_weights"]),
            }
        elif method == "probcover":
            method_metadata = {
                "probcover_delta": probcover_delta,
                "probcover_radius_seed": probcover_radius_seed,
                "probcover_cache_digest": probcover_cache_digest,
            }
        else:
            method_metadata = {}

        set_seed(round_seeds.training)

        if framework == "fully_supervised":
            ckpt = checkpoint_dir / (
                f"{framework}_{method}_budget{len(labeled_indices)}_seed{seed}_r{round_id}.pt"
            )
            train_result = _train_eval_fully_supervised(
                selected_indices=labeled_indices,
                data_root=data_root,
                num_workers=num_workers,
                classifier_cfg=classifier_cfg,
                device=device,
                checkpoint_path=ckpt,
                dataloader_seed=round_seeds.dataloader,
                num_classes=num_classes,
            )
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
                num_classes=num_classes,
            )
        elif framework == "label_spreading_proxy":
            train_result = _train_eval_label_spreading_proxy(
                train_embeddings=train_embeddings,
                test_embeddings=test_embeddings,
                train_labels=train_labels,
                test_labels=test_labels,
                selected_indices=labeled_indices,
                num_classes=num_classes,
            )
        else:
            raise ValueError(f"Unknown framework: {framework}")

        round_metrics.append(
            {
                "round": round_id,
                "query_size": query_size,
                "cumulative_budget": len(labeled_indices),
                "new_indices": newly_selected.tolist(),
                "selected_indices": selected_indices_ordered.copy(),
                "seeds": asdict(round_seeds),
                "trained_epochs": train_result.metrics.trained_epochs,
                "test_loss": train_result.metrics.test_loss,
                "test_accuracy": train_result.metrics.test_accuracy,
                "selection_diagnostics": round_diagnostics,
                "method_metadata": method_metadata,
            }
        )
        selector_seconds_by_round.append(selector_seconds)
        round_seconds.append(time.perf_counter() - round_started)
        print(
            f"[{framework}][{method}] round {round_id}/{rounds} "
            f"selected={len(labeled_indices)} test_acc={train_result.metrics.test_accuracy:.4f}"
        )

    if not round_metrics:
        raise RuntimeError("No rounds were executed. Check selection.round_query_sizes.")

    artifact = build_run_artifact(
        effective_config=effective_config,
        environment=environment,
        rounds=round_metrics,
        timings={
            "total_seconds": time.perf_counter() - run_started,
            "round_seconds": round_seconds,
            "selector_seconds": selector_seconds_by_round,
        },
    )
    atomic_write_json(run_artifact_path, artifact)

    print("\nExperiment complete:")
    print(
        json.dumps(
            {
                "run_id": artifact["run_id"],
                "artifact_path": str(run_artifact_path),
                "rounds": len(round_metrics),
            },
            indent=2,
        )
    )
    return artifact
