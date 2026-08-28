from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import yaml
from torch.utils.data import Dataset

import scripts.train_simclr as trainer
from src.artifacts import file_sha256


class TinyPairDataset(Dataset):
    def __init__(self) -> None:
        self.inputs = torch.arange(32, dtype=torch.float32).reshape(8, 4) / 31.0

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int):
        value = self.inputs[index]
        return (value, value + 0.01), 0


class TinySimCLR(nn.Module):
    def __init__(self, proj_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(4, 4)
        self.projector = nn.Linear(4, proj_dim)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(inputs)
        return features, self.projector(features)


def _write_config(path: Path, checkpoint_path: Path) -> None:
    config = {
        "seed": 7,
        "data": {"root": "unused", "num_workers": 0},
        "simclr": {
            "batch_size": 4,
            "epochs": 1,
            "lr": 0.05,
            "momentum": 0.0,
            "nesterov": False,
            "min_lr": 0.0,
            "weight_decay": 0.0,
            "temperature": 0.5,
            "projection_dim": 3,
            "save_path": str(checkpoint_path),
        },
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _patch_training_dependencies(monkeypatch) -> None:
    monkeypatch.setattr(
        trainer,
        "collect_git_state",
        lambda: {"git_commit": "a" * 40, "git_dirty": False},
    )
    monkeypatch.setattr(trainer, "get_cifar10_train", lambda **_kwargs: TinyPairDataset())
    monkeypatch.setattr(trainer, "SimCLRModel", TinySimCLR)
    monkeypatch.setattr(trainer.torch.cuda, "is_available", lambda: False)


def test_train_simclr_writes_final_checkpoint_and_matching_manifest(monkeypatch, tmp_path) -> None:
    _patch_training_dependencies(monkeypatch)
    checkpoint_path = tmp_path / "simclr.pt"
    config_path = tmp_path / "train.yaml"
    _write_config(config_path, checkpoint_path)

    manifest = trainer.train_simclr_checkpoint(config_path)

    persisted = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    manifest_path = checkpoint_path.with_suffix(".pt.manifest.json")
    assert persisted["checkpoint_format_version"] == 2
    assert persisted["epoch"] == 1
    assert persisted["git_commit"] == "a" * 40
    assert persisted["final_training_loss"] == pytest.approx(manifest["final_training_loss"])
    assert manifest["checkpoint_sha256"] == file_sha256(checkpoint_path)
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest
    assert list(tmp_path.glob(".*.tmp")) == []

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        trainer.train_simclr_checkpoint(config_path)


def test_train_simclr_rejects_dirty_worktree_before_loading_data(monkeypatch, tmp_path) -> None:
    checkpoint_path = tmp_path / "simclr.pt"
    config_path = tmp_path / "train.yaml"
    _write_config(config_path, checkpoint_path)
    monkeypatch.setattr(
        trainer,
        "collect_git_state",
        lambda: {"git_commit": "a" * 40, "git_dirty": True},
    )
    data_loaded = False

    def fail_if_loaded(**_kwargs):
        nonlocal data_loaded
        data_loaded = True
        raise AssertionError("dataset should not be loaded")

    monkeypatch.setattr(trainer, "get_cifar10_train", fail_if_loaded)

    with pytest.raises(RuntimeError, match="clean Git worktree"):
        trainer.train_simclr_checkpoint(config_path)
    assert data_loaded is False
    assert not checkpoint_path.exists()


def test_atomic_torch_save_preserves_target_when_replace_fails(monkeypatch, tmp_path) -> None:
    path = tmp_path / "checkpoint.pt"
    torch.save({"previous": True}, path)

    def fail_replace(_source: Path, _target: Path) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(trainer.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replace failure"):
        trainer._atomic_torch_save({"replacement": True}, path)

    assert torch.load(path, map_location="cpu", weights_only=True) == {"previous": True}
    assert list(tmp_path.glob(".*.tmp")) == []
