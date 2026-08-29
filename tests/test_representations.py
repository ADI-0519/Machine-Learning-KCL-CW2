from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

import src.experiment as experiment
from src.artifacts import config_digest, file_sha256
from src.config import load_configurations, validate_protocol_config
from src.models import SimCLRModel
from src.representations import (
    Dinov2Encoder,
    build_embedding_transform,
    load_representation_encoder,
    representation_cache_path,
)

DINOV2_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"


def test_embedding_cache_is_written_atomically(monkeypatch, tmp_path) -> None:
    expected = np.arange(12, dtype=np.float32).reshape(3, 4)
    path = tmp_path / "embeddings.npy"
    monkeypatch.setattr(experiment, "load_representation_encoder", lambda *_args: object())
    monkeypatch.setattr(experiment, "build_embedding_loader", lambda **_kwargs: object())
    monkeypatch.setattr(experiment, "build_embedding_transform", lambda _config: object())
    monkeypatch.setattr(experiment, "grab_embeddings", lambda **_kwargs: expected)

    actual = experiment.load_or_compute_embeddings(
        embedding_path=path,
        representation={"backend": "simclr"},
        data_root="unused",
        split="train",
        batch_size=2,
        num_workers=0,
        device=torch.device("cpu"),
        dataloader_seed=7,
    )

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(np.load(path), expected)
    assert list(tmp_path.glob(".*.tmp")) == []


def _dinov2_config(checkpoint_path: Path, checkpoint_sha256: str) -> dict:
    config = load_configurations("configs/protocol_v2_pilot.yaml")
    config["representation"] = {
        "backend": "dinov2",
        "checkpoint_path": str(checkpoint_path),
        "model_id": "facebook/dinov2-small",
        "model_revision": DINOV2_REVISION,
        "weights_sha256": checkpoint_sha256,
        "feature_dim": 384,
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 32,
        "embedding_seed": 21,
    }
    return config


def test_protocol_config_accepts_exact_simclr_and_dinov2_schemas(tmp_path) -> None:
    simclr = load_configurations("configs/protocol_v2_pilot.yaml")
    validate_protocol_config(simclr)

    dinov2 = _dinov2_config(tmp_path / "model.safetensors", "a" * 64)
    validate_protocol_config(dinov2)


def test_protocol_config_requires_pinned_simclr_weights() -> None:
    config = load_configurations("configs/protocol_v2_pilot.yaml")
    config["representation"].pop("weights_sha256")

    with pytest.raises(ValueError, match="weights_sha256"):
        validate_protocol_config(config)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda cfg: cfg["representation"].update(backend="unknown"), "backend"),
        (
            lambda cfg: cfg["representation"].update(model_revision="main"),
            "model_revision",
        ),
        (
            lambda cfg: cfg["representation"].update(weights_sha256="not-a-digest"),
            "weights_sha256",
        ),
        (
            lambda cfg: cfg["representation"].update(crop_size=225),
            "multiple of 14",
        ),
    ],
)
def test_protocol_config_rejects_unpinned_or_invalid_dinov2(mutation, message, tmp_path) -> None:
    config = _dinov2_config(tmp_path / "model.safetensors", "a" * 64)
    mutation(config)
    with pytest.raises(ValueError, match=message):
        validate_protocol_config(config)


def test_embedding_transforms_have_backend_specific_shapes() -> None:
    image = Image.new("RGB", (32, 32), color=(20, 40, 60))
    simclr_config = load_configurations("configs/protocol_v2_pilot.yaml")["representation"]
    dinov2_config = _dinov2_config(Path("unused"), "a" * 64)["representation"]

    simclr_tensor = build_embedding_transform(simclr_config)(image)
    dinov2_tensor = build_embedding_transform(dinov2_config)(image)

    assert simclr_tensor.shape == (3, 32, 32)
    assert dinov2_tensor.shape == (3, 224, 224)
    assert torch.isfinite(simclr_tensor).all()
    assert torch.isfinite(dinov2_tensor).all()


def test_cache_path_keys_dataset_split_backend_config_and_checkpoint(tmp_path) -> None:
    simclr = load_configurations("configs/protocol_v2_pilot.yaml")["representation"]
    baseline = representation_cache_path(
        tmp_path,
        dataset="cifar10",
        split="train",
        representation=simclr,
        checkpoint_sha256="a" * 64,
    )

    changed_values = []
    for dataset, split, backend_config, digest in (
        ("cifar100", "train", simclr, "a" * 64),
        ("cifar10", "test", simclr, "a" * 64),
        ("cifar10", "train", simclr, "b" * 64),
        (
            "cifar10",
            "train",
            _dinov2_config(Path("unused"), "c" * 64)["representation"],
            "c" * 64,
        ),
    ):
        changed_values.append(
            representation_cache_path(
                tmp_path,
                dataset=dataset,
                split=split,
                representation=backend_config,
                checkpoint_sha256=digest,
            )
        )

    assert baseline.parent == tmp_path
    assert baseline.suffix == ".npy"
    assert len({baseline, *changed_values}) == 5
    assert baseline == representation_cache_path(
        tmp_path,
        dataset="cifar10",
        split="train",
        representation=deepcopy(simclr),
        checkpoint_sha256="a" * 64,
    )


class _FakeDinov2Backbone(nn.Module):
    def forward(self, *, pixel_values: torch.Tensor):
        return SimpleNamespace(pooler_output=pixel_values.mean(dim=(2, 3)))


def test_dinov2_wrapper_returns_pooler_output() -> None:
    wrapper = Dinov2Encoder(_FakeDinov2Backbone())
    inputs = torch.arange(24, dtype=torch.float32).reshape(2, 3, 2, 2)
    assert torch.equal(wrapper(inputs), inputs.mean(dim=(2, 3)))


def test_load_simclr_checkpoint_round_trip(tmp_path) -> None:
    checkpoint_path = tmp_path / "simclr.pt"
    source = SimCLRModel(proj_dim=128)
    training_config = {"seed": 21, "data": {}, "simclr": {}}
    training_digest = config_digest(training_config)
    payload = {
        "checkpoint_format_version": 2,
        "epoch": 1,
        "model_state_dict": source.state_dict(),
        "final_training_loss": 1.25,
        "training_config": training_config,
        "training_config_sha256": training_digest,
        "git_commit": "a" * 40,
    }
    torch.save(payload, checkpoint_path)
    checkpoint_digest = file_sha256(checkpoint_path)
    manifest = {
        "manifest_version": 1,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_digest,
        "checkpoint_format_version": 2,
        "git_commit": "a" * 40,
        "training_config_sha256": training_digest,
        "training_seed": 21,
        "trained_epochs": 1,
        "final_training_loss": 1.25,
    }
    checkpoint_path.with_suffix(".pt.manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    representation = {
        "backend": "simclr",
        "checkpoint_path": str(checkpoint_path),
        "weights_sha256": checkpoint_digest,
        "projection_dim": 128,
        "batch_size": 8,
        "embedding_seed": 21,
    }

    loaded = load_representation_encoder(representation, torch.device("cpu"))

    source_state = source.encoder.state_dict()
    loaded_state = loaded.state_dict()
    assert source_state.keys() == loaded_state.keys()
    assert all(torch.equal(source_state[key], loaded_state[key]) for key in source_state)

    representation["weights_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SimCLR checkpoint SHA-256 mismatch"):
        load_representation_encoder(representation, torch.device("cpu"))

    representation["weights_sha256"] = checkpoint_digest
    checkpoint_path.with_suffix(".pt.manifest.json").unlink()
    with pytest.raises(FileNotFoundError, match="SimCLR manifest"):
        load_representation_encoder(representation, torch.device("cpu"))


def test_load_dinov2_uses_local_pinned_weights_and_checks_digest(monkeypatch, tmp_path) -> None:
    checkpoint_path = tmp_path / "model.safetensors"
    checkpoint_path.write_bytes(b"fixed-safe-weights")
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    representation = _dinov2_config(checkpoint_path, digest)["representation"]
    calls = []

    class FakeDinov2Model:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append((Path(path), kwargs))
            return _FakeDinov2Backbone()

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(Dinov2Model=FakeDinov2Model))
    loaded = load_representation_encoder(representation, torch.device("cpu"))

    assert isinstance(loaded, Dinov2Encoder)
    assert calls == [
        (
            tmp_path,
            {
                "local_files_only": True,
                "use_safetensors": True,
            },
        )
    ]

    representation["weights_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_representation_encoder(representation, torch.device("cpu"))


def test_load_dinov2_requires_local_config_and_transformers(tmp_path, monkeypatch) -> None:
    checkpoint_path = tmp_path / "model.safetensors"
    checkpoint_path.write_bytes(b"weights")
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    representation = _dinov2_config(checkpoint_path, digest)["representation"]

    with pytest.raises(FileNotFoundError, match=r"config\.json"):
        load_representation_encoder(representation, torch.device("cpu"))

    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(RuntimeError, match="transformers"):
        load_representation_encoder(representation, torch.device("cpu"))


@pytest.mark.parametrize("split", ["validation", "", "TRAIN"])
def test_cache_path_rejects_invalid_splits(tmp_path, split) -> None:
    representation = load_configurations("configs/protocol_v2_pilot.yaml")["representation"]
    with pytest.raises(ValueError, match="split"):
        representation_cache_path(
            tmp_path,
            dataset="cifar10",
            split=split,
            representation=representation,
            checkpoint_sha256="a" * 64,
        )


def test_cache_path_rejects_invalid_checkpoint_digest(tmp_path) -> None:
    representation = load_configurations("configs/protocol_v2_pilot.yaml")["representation"]
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        representation_cache_path(
            tmp_path,
            dataset="cifar10",
            split="train",
            representation=representation,
            checkpoint_sha256="invalid",
        )


def test_dinov2_transform_matches_locked_numeric_regression() -> None:
    pixels = np.zeros((32, 32, 3), dtype=np.uint8)
    pixels[:, :, 0] = 255
    image = Image.fromarray(pixels, mode="RGB")
    representation = _dinov2_config(Path("unused"), "a" * 64)["representation"]

    transformed = build_embedding_transform(representation)(image)

    expected_red = (1.0 - 0.485) / 0.229
    expected_green = (0.0 - 0.456) / 0.224
    expected_blue = (0.0 - 0.406) / 0.225
    torch.testing.assert_close(
        transformed[:, 0, 0],
        torch.tensor([expected_red, expected_green, expected_blue]),
    )
