from __future__ import annotations

import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

import scripts.prepare_dinov2 as prepare


def test_prepare_dinov2_pins_revision_verifies_files_and_writes_manifest(
    monkeypatch, tmp_path
) -> None:
    weights = b"deterministic-test-weights"
    digest = hashlib.sha256(weights).hexdigest()
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        destination = kwargs["local_dir"]
        (destination / "model.safetensors").write_bytes(weights)
        (destination / "config.json").write_text(
            json.dumps({"model_type": "dinov2", "hidden_size": 384, "patch_size": 14}),
            encoding="utf-8",
        )
        (destination / "preprocessor_config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(prepare, "WEIGHTS_SHA256", digest)
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fake_snapshot_download),
    )

    manifest_path = prepare.prepare_dinov2(tmp_path / "dinov2")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert calls == [
        {
            "repo_id": prepare.MODEL_ID,
            "revision": prepare.MODEL_REVISION,
            "local_dir": tmp_path / "dinov2",
            "allow_patterns": list(prepare.REQUIRED_FILES),
        }
    ]
    assert manifest["model_revision"] == prepare.MODEL_REVISION
    assert manifest["files"]["model.safetensors"] == digest


def test_prepare_dinov2_rejects_wrong_weights(monkeypatch, tmp_path) -> None:
    def fake_snapshot_download(**kwargs):
        destination = kwargs["local_dir"]
        (destination / "model.safetensors").write_bytes(b"wrong")
        (destination / "config.json").write_text(
            json.dumps({"model_type": "dinov2", "hidden_size": 384, "patch_size": 14}),
            encoding="utf-8",
        )
        (destination / "preprocessor_config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fake_snapshot_download),
    )
    with pytest.raises(ValueError, match="SHA-256"):
        prepare.prepare_dinov2(tmp_path / "dinov2")
