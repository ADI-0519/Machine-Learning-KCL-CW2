"""Download and verify the exact local DINOv2-small representation artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from src.artifacts import atomic_write_json

MODEL_ID = "facebook/dinov2-small"
MODEL_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
WEIGHTS_SHA256 = "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1"
REQUIRED_FILES = ("config.json", "model.safetensors", "preprocessor_config.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _validate_model_config(config: dict[str, Any]) -> None:
    expected = {"model_type": "dinov2", "hidden_size": 384, "patch_size": 14}
    for field, value in expected.items():
        if config.get(field) != value:
            raise ValueError(f"DINOv2 config {field!r} must equal {value!r}")


def prepare_dinov2(output_dir: str | Path) -> Path:
    """Materialize and verify the pinned model; return its manifest path."""
    try:
        import truststore
        from huggingface_hub import snapshot_download
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError("install project dependencies before preparing DINOv2") from exc

    # Respect the host's trusted certificate authorities on Windows and managed
    # networks. This retains certificate verification; it does not bypass TLS.
    truststore.inject_into_ssl()

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=destination,
        allow_patterns=list(REQUIRED_FILES),
    )

    paths = {name: destination / name for name in REQUIRED_FILES}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"DINOv2 snapshot is missing required files: {missing}")

    actual_weights_digest = _sha256(paths["model.safetensors"])
    if actual_weights_digest != WEIGHTS_SHA256:
        raise ValueError(
            "downloaded DINOv2 weights failed SHA-256 verification: "
            f"expected {WEIGHTS_SHA256}, found {actual_weights_digest}"
        )
    _validate_model_config(_load_json(paths["config.json"]))

    manifest = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "source": f"https://huggingface.co/{MODEL_ID}/tree/{MODEL_REVISION}",
        "files": {name: _sha256(path) for name, path in sorted(paths.items())},
    }
    manifest_path = destination / "manifest.json"
    atomic_write_json(manifest_path, manifest)
    return manifest_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="results/checkpoints/dinov2-small",
        help="local directory for the immutable DINOv2 artifact",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest_path = prepare_dinov2(args.output_dir)
    print(f"Verified pinned DINOv2 checkpoint: {manifest_path}")


if __name__ == "__main__":
    main()
