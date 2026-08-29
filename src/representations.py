"""Frozen representation backends and provenance-keyed embedding caches."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision import transforms

from .artifacts import config_digest
from .models import SimCLRModel
from .protocol import PROTOCOL_VERSION

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_component(value: str, context: str) -> str:
    if not isinstance(value, str) or not value or not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError(f"{context} must contain only filename-safe characters")
    return value


def build_embedding_transform(representation: dict[str, Any]):
    """Return deterministic image preprocessing for the configured frozen backend."""
    backend = representation.get("backend")
    if backend == "simclr":
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2023, 0.1994, 0.2010),
                ),
            ]
        )
    if backend == "dinov2":
        return transforms.Compose(
            [
                transforms.Resize(
                    int(representation["resize_size"]),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.CenterCrop(int(representation["crop_size"])),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    raise ValueError(f"unsupported representation backend: {backend!r}")


def representation_cache_path(
    cache_dir: str | Path,
    *,
    dataset: str,
    split: str,
    representation: dict[str, Any],
    checkpoint_sha256: str,
) -> Path:
    """Return a cache path keyed by every representation-defining input."""
    dataset_name = _safe_component(dataset, "dataset")
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    backend = _safe_component(str(representation.get("backend", "")), "backend")
    if not re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha256):
        raise ValueError("checkpoint_sha256 must be a lowercase SHA-256")
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": dataset_name,
        "split": split,
        "representation": representation,
        "checkpoint_sha256": checkpoint_sha256,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    cache_digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]
    return Path(cache_dir) / f"{dataset_name}_{backend}_{split}_{cache_digest}.npy"


class Dinov2Encoder(nn.Module):
    """Adapt a Hugging Face DINOv2 backbone to the tensor-only encoder contract."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.backbone(pixel_values=inputs)
        pooled = output.pooler_output
        if not isinstance(pooled, torch.Tensor) or pooled.ndim != 2:
            raise RuntimeError("DINOv2 backbone returned an invalid pooler_output")
        return pooled


def _load_simclr_encoder(representation: dict[str, Any], device: torch.device) -> nn.Module:
    checkpoint_path = Path(representation["checkpoint_path"])
    manifest_path = checkpoint_path.with_suffix(f"{checkpoint_path.suffix}.manifest.json")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"required SimCLR checkpoint is missing: {checkpoint_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"required SimCLR manifest is missing: {manifest_path}")
    actual_digest = _file_sha256(checkpoint_path)
    expected_digest = str(representation["weights_sha256"])
    if actual_digest != expected_digest:
        raise ValueError(
            f"SimCLR checkpoint SHA-256 mismatch: expected {expected_digest}, found {actual_digest}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid SimCLR manifest at {manifest_path}: {exc}") from exc
    expected_manifest_fields = {
        "manifest_version",
        "checkpoint_path",
        "checkpoint_sha256",
        "checkpoint_format_version",
        "git_commit",
        "training_config_sha256",
        "training_seed",
        "trained_epochs",
        "final_training_loss",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_manifest_fields:
        raise ValueError("SimCLR manifest has an invalid schema")
    if (
        manifest["manifest_version"] != 1
        or manifest["checkpoint_format_version"] != 2
        or manifest["checkpoint_sha256"] != expected_digest
        or Path(str(manifest["checkpoint_path"])).resolve() != checkpoint_path.resolve()
        or not isinstance(manifest["git_commit"], str)
        or not re.fullmatch(r"[0-9a-f]{40}", manifest["git_commit"])
        or not isinstance(manifest["training_config_sha256"], str)
        or not re.fullmatch(r"[0-9a-f]{64}", manifest["training_config_sha256"])
        or not isinstance(manifest["training_seed"], int)
        or isinstance(manifest["training_seed"], bool)
        or not isinstance(manifest["trained_epochs"], int)
        or isinstance(manifest["trained_epochs"], bool)
        or manifest["trained_epochs"] <= 0
        or not isinstance(manifest["final_training_loss"], (int, float))
        or isinstance(manifest["final_training_loss"], bool)
        or not math.isfinite(float(manifest["final_training_loss"]))
    ):
        raise ValueError("SimCLR manifest provenance is inconsistent")

    model = SimCLRModel(proj_dim=int(representation["projection_dim"])).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    expected_checkpoint_fields = {
        "checkpoint_format_version",
        "epoch",
        "model_state_dict",
        "final_training_loss",
        "training_config",
        "training_config_sha256",
        "git_commit",
    }
    if not isinstance(checkpoint, dict) or set(checkpoint) != expected_checkpoint_fields:
        raise ValueError("SimCLR checkpoint has an invalid schema")
    if (
        checkpoint["checkpoint_format_version"] != manifest["checkpoint_format_version"]
        or checkpoint["epoch"] != manifest["trained_epochs"]
        or checkpoint["git_commit"] != manifest["git_commit"]
        or checkpoint["final_training_loss"] != manifest["final_training_loss"]
        or not isinstance(checkpoint["training_config"], dict)
        or config_digest(checkpoint["training_config"]) != manifest["training_config_sha256"]
        or checkpoint["training_config_sha256"] != manifest["training_config_sha256"]
        or checkpoint["training_config"].get("seed") != manifest["training_seed"]
    ):
        raise ValueError("SimCLR checkpoint disagrees with its provenance manifest")
    state_dict = checkpoint["model_state_dict"]
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            "Failed to load SimCLR checkpoint due to model architecture mismatch. "
            "Retrain SimCLR and regenerate cached embeddings."
        ) from exc
    model.encoder.eval()
    return model.encoder


def _load_dinov2_encoder(representation: dict[str, Any], device: torch.device) -> nn.Module:
    checkpoint_path = Path(representation["checkpoint_path"])
    config_path = checkpoint_path.parent / "config.json"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"required DINOv2 weights are missing: {checkpoint_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"required DINOv2 config.json is missing: {config_path}")
    actual_digest = _file_sha256(checkpoint_path)
    expected_digest = str(representation["weights_sha256"])
    if actual_digest != expected_digest:
        raise ValueError(
            f"DINOv2 checkpoint SHA-256 mismatch: expected {expected_digest}, found {actual_digest}"
        )
    try:
        from transformers import Dinov2Model
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "DINOv2 requires the project dependency 'transformers'; reinstall the project"
        ) from exc
    backbone = Dinov2Model.from_pretrained(
        checkpoint_path.parent,
        local_files_only=True,
        use_safetensors=True,
    )
    backbone.to(device)
    backbone.eval()
    return Dinov2Encoder(backbone)


def load_representation_encoder(
    representation: dict[str, Any],
    device: torch.device,
) -> nn.Module:
    """Load a frozen local representation encoder with strict provenance checks."""
    backend = representation.get("backend")
    if backend == "simclr":
        return _load_simclr_encoder(representation, device)
    if backend == "dinov2":
        return _load_dinov2_encoder(representation, device)
    raise ValueError(f"unsupported representation backend: {backend!r}")
