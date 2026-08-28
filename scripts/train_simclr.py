"""Train and freeze the SimCLR representation used by protocol-v2 experiments."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.artifacts import (
    atomic_write_json,
    collect_git_state,
    config_digest,
    file_sha256,
)
from src.config import load_configurations
from src.data import SimCLRTransform, get_cifar10_train
from src.models import SimCLRModel
from src.seed import make_generator, seed_worker, set_seed
from src.simclr import NTXentLoss, train_simclr_epoch


def _manifest_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_suffix(f"{checkpoint_path.suffix}.manifest.json")


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    """Write a PyTorch checkpoint with atomic replacement and temporary cleanup."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def train_simclr_checkpoint(
    config_path: str | Path = "configs/default.yaml",
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Train for the fixed schedule and atomically save the final-epoch checkpoint."""
    cfg = load_configurations(config_path)
    simclr_cfg = cfg["simclr"]
    data_cfg = cfg["data"]
    save_path = Path(simclr_cfg["save_path"])
    manifest_path = _manifest_path(save_path)
    existing = [path for path in (save_path, manifest_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite frozen SimCLR output(s): "
            + ", ".join(str(path) for path in existing)
        )

    git_state = collect_git_state()
    if git_state["git_dirty"]:
        raise RuntimeError("SimCLR training requires a clean Git worktree")

    set_seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = get_cifar10_train(
        root=data_cfg["root"],
        transform=SimCLRTransform(),
    )
    loader = DataLoader(
        dataset,
        batch_size=simclr_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=make_generator(int(cfg["seed"])),
    )

    model = SimCLRModel(proj_dim=simclr_cfg["projection_dim"]).to(device)
    criterion = NTXentLoss(temperature=simclr_cfg["temperature"])
    optimizer = SGD(
        model.parameters(),
        lr=simclr_cfg["lr"],
        momentum=simclr_cfg["momentum"],
        nesterov=simclr_cfg["nesterov"],
        weight_decay=simclr_cfg["weight_decay"],
    )
    scheduler = CosineAnnealingLR(
        optimizer=optimizer,
        T_max=simclr_cfg["epochs"],
        eta_min=simclr_cfg["min_lr"],
    )

    final_loss: float | None = None
    for epoch in range(1, simclr_cfg["epochs"] + 1):
        final_loss = train_simclr_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[SimCLR Epoch {epoch:03d}/{simclr_cfg['epochs']:03d}] "
            f"loss={final_loss:.4f} lr={current_lr:.6f}"
        )

    if final_loss is None:  # pragma: no cover - positive epochs are required by configuration
        raise RuntimeError("SimCLR training completed without an epoch")
    training_config = {
        "seed": int(cfg["seed"]),
        "data": data_cfg,
        "simclr": simclr_cfg,
    }
    checkpoint = {
        "checkpoint_format_version": 2,
        "epoch": int(simclr_cfg["epochs"]),
        "model_state_dict": model.state_dict(),
        "final_training_loss": final_loss,
        "training_config": training_config,
        "training_config_sha256": config_digest(training_config),
        "git_commit": git_state["git_commit"],
    }
    _atomic_torch_save(checkpoint, save_path)

    manifest: dict[str, Any] = {
        "manifest_version": 1,
        "checkpoint_path": str(save_path.resolve()),
        "checkpoint_sha256": file_sha256(save_path),
        "checkpoint_format_version": checkpoint["checkpoint_format_version"],
        "git_commit": git_state["git_commit"],
        "training_config_sha256": checkpoint["training_config_sha256"],
        "training_seed": int(cfg["seed"]),
        "trained_epochs": int(simclr_cfg["epochs"]),
        "final_training_loss": final_loss,
    }
    atomic_write_json(manifest_path, manifest)
    print(f"Saved final SimCLR checkpoint to {save_path}")
    print(f"Saved provenance manifest to {manifest_path}")
    print(f"SimCLR checkpoint SHA-256: {manifest['checkpoint_sha256']}")
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace an existing checkpoint and manifest.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the SimCLR checkpoint trainer from command-line arguments."""
    args = _parse_args()
    train_simclr_checkpoint(args.config, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
