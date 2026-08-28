"""Immutable protocol-v2 run artifacts and provenance helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
from copy import deepcopy
from importlib import metadata
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import torch

from .config import CCFL_METHODS
from .protocol import PROTOCOL_VERSION

ROOT_FIELDS = {
    "protocol_version",
    "run_id",
    "config_digest",
    "effective_config",
    "environment",
    "rounds",
    "timings",
}
ENVIRONMENT_FIELDS = {
    "git_commit",
    "git_dirty",
    "python_version",
    "platform",
    "packages",
    "device",
    "cuda_runtime",
    "deterministic_algorithms",
    "cublas_workspace_config",
    "checkpoint_path",
    "checkpoint_sha256",
}
ROUND_FIELDS = {
    "round",
    "query_size",
    "cumulative_budget",
    "new_indices",
    "selected_indices",
    "seeds",
    "trained_epochs",
    "test_loss",
    "test_accuracy",
    "selection_diagnostics",
    "method_metadata",
}
SEED_FIELDS = {"replicate", "clustering", "selector", "training", "dataloader"}
TIMING_FIELDS = {"total_seconds", "round_seconds", "selector_seconds"}
DIAGNOSTIC_FIELDS = {
    "nearest_distance_mean",
    "nearest_distance_p95",
    "nearest_distance_max",
    "selected_pairwise_cosine_mean",
    "selected_typicality_mean",
}


def canonical_json(value: Any) -> str:
    """Serialize a JSON-compatible value with stable ordering and no NaN values."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def config_digest(effective_config: dict[str, Any]) -> str:
    """Return the SHA-256 digest of a canonical effective configuration."""
    encoded = canonical_json(effective_config).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_run_id(effective_config: dict[str, Any]) -> str:
    """Build a deterministic protocol-scoped run identifier."""
    return f"v{PROTOCOL_VERSION}-{config_digest(effective_config)[:16]}"


def build_effective_config(
    config: dict[str, Any],
    *,
    method: str,
    replicate_seed: int,
    framework: str,
) -> dict[str, Any]:
    """Freeze source configuration plus the dimensions identifying one run."""
    if "run" in config:
        raise ValueError("source configuration must not contain a reserved run section")
    effective = deepcopy(config)
    effective["run"] = {
        "framework": str(framework),
        "method": str(method),
        "replicate_seed": int(replicate_seed),
    }
    canonical_json(effective)
    return effective


def artifact_path_for(output_root: str | Path, run_id: str) -> Path:
    """Return the immutable JSON path for a protocol-v2 run ID."""
    return Path(output_root) / "runs" / f"{run_id}.json"


def build_run_artifact(
    *,
    effective_config: dict[str, Any],
    environment: dict[str, Any],
    rounds: list[dict[str, Any]],
    timings: dict[str, Any],
) -> dict[str, Any]:
    """Construct and validate one complete in-memory run artifact."""
    digest = config_digest(effective_config)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": build_run_id(effective_config),
        "config_digest": digest,
        "effective_config": deepcopy(effective_config),
        "environment": deepcopy(environment),
        "rounds": deepcopy(rounds),
        "timings": deepcopy(timings),
    }
    validate_artifact(payload)
    return payload


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace a JSON target and remove temporary files on failure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_artifact(path: str | Path) -> dict[str, Any]:
    """Read and validate one artifact, rejecting malformed JSON or schemas."""
    artifact_path = Path(path)
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid artifact at {artifact_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"invalid artifact at {artifact_path}: root must be an object")
    validate_artifact(payload)
    return payload


def _require_exact_fields(value: dict[str, Any], expected: set[str], context: str) -> None:
    missing = expected - value.keys()
    if missing:
        raise ValueError(f"artifact missing fields in {context}: {sorted(missing)}")
    unexpected = value.keys() - expected
    if unexpected:
        raise ValueError(f"artifact has unexpected fields in {context}: {sorted(unexpected)}")


def _contains_legacy_key(value: Any) -> bool:
    legacy_key = "best_" + "test_accuracy"
    if isinstance(value, dict):
        return legacy_key in value or any(_contains_legacy_key(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_legacy_key(item) for item in value)
    return False


def _finite_number(value: Any, context: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f"invalid numeric value for {context}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite {context}")
    return number


def _finite_nonnegative(value: Any, context: str) -> float:
    number = _finite_number(value, context)
    if number < 0.0:
        raise ValueError(f"negative {context}")
    return number


def _integer(value: Any, context: str, *, minimum: int = 0) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool) or int(value) < minimum:
        raise ValueError(f"artifact {context} must be an integer >= {minimum}")
    return int(value)


def validate_artifact(payload: dict[str, Any]) -> None:
    """Validate the complete protocol-v2 artifact schema and nested trajectory."""
    if _contains_legacy_key(payload):
        raise ValueError("legacy best-test metric is forbidden")
    _require_exact_fields(payload, ROOT_FIELDS, "root")
    if payload["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError("artifact protocol version mismatch")
    if not isinstance(payload["effective_config"], dict):
        raise ValueError("artifact effective_config must be an object")

    expected_digest = config_digest(payload["effective_config"])
    if payload["config_digest"] != expected_digest:
        raise ValueError("artifact config digest mismatch")
    expected_run_id = f"v{PROTOCOL_VERSION}-{expected_digest[:16]}"
    if payload["run_id"] != expected_run_id:
        raise ValueError("artifact run ID mismatch")

    run_config = payload["effective_config"].get("run")
    if not isinstance(run_config, dict):
        raise ValueError("artifact effective_config is missing run dimensions")
    _require_exact_fields(
        run_config,
        {"framework", "method", "replicate_seed"},
        "effective_config.run",
    )
    if not isinstance(run_config["framework"], str) or not run_config["framework"]:
        raise ValueError("artifact run framework must be a non-empty string")
    if not isinstance(run_config["method"], str) or not run_config["method"]:
        raise ValueError("artifact run method must be a non-empty string")
    replicate_seed = _integer(run_config["replicate_seed"], "run replicate_seed")

    environment = payload["environment"]
    if not isinstance(environment, dict):
        raise ValueError("artifact environment must be an object")
    _require_exact_fields(environment, ENVIRONMENT_FIELDS, "environment")
    if not isinstance(environment["git_commit"], str) or not environment["git_commit"]:
        raise ValueError("artifact git_commit must be a non-empty string")
    if not isinstance(environment["git_dirty"], bool):
        raise ValueError("artifact git_dirty must be boolean")
    if not isinstance(environment["packages"], dict) or not environment["packages"]:
        raise ValueError("artifact packages must be a non-empty object")
    if any(
        not isinstance(name, str) or not name or not isinstance(version, str) or not version
        for name, version in environment["packages"].items()
    ):
        raise ValueError("artifact packages must map non-empty names to non-empty versions")
    for name in ("python_version", "platform", "device"):
        if not isinstance(environment[name], str) or not environment[name]:
            raise ValueError(f"artifact {name} must be a non-empty string")
    if not isinstance(environment["deterministic_algorithms"], bool):
        raise ValueError("artifact deterministic_algorithms must be boolean")
    cublas_config = environment["cublas_workspace_config"]
    if cublas_config is not None and not isinstance(cublas_config, str):
        raise ValueError("artifact cublas_workspace_config must be null or a string")
    checkpoint_digest = environment["checkpoint_sha256"]
    if checkpoint_digest is not None and (
        not isinstance(checkpoint_digest, str)
        or len(checkpoint_digest) != 64
        or any(character not in "0123456789abcdef" for character in checkpoint_digest)
    ):
        raise ValueError("artifact checkpoint_sha256 must be null or lowercase SHA-256")
    checkpoint_path = environment["checkpoint_path"]
    if (checkpoint_path is None) != (checkpoint_digest is None):
        raise ValueError("artifact checkpoint path and digest must both be null or populated")
    if checkpoint_path is not None and (
        not isinstance(checkpoint_path, str) or not checkpoint_path
    ):
        raise ValueError("artifact checkpoint_path must be null or a non-empty string")
    if run_config["framework"] in {"ssl_embedding", "label_spreading_proxy"} and (
        checkpoint_path is None or checkpoint_digest is None
    ):
        raise ValueError("artifact embedding framework requires checkpoint provenance")

    rounds = payload["rounds"]
    if not isinstance(rounds, list) or not rounds:
        raise ValueError("artifact has no rounds")
    previous_selected: list[int] = []
    cumulative_budget = 0
    for expected_round, record in enumerate(rounds, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"artifact round {expected_round} must be an object")
        _require_exact_fields(record, ROUND_FIELDS, f"round {expected_round}")
        if _integer(record["round"], f"round {expected_round} number", minimum=1) != expected_round:
            raise ValueError(f"artifact round sequence mismatch at round {expected_round}")
        if not isinstance(record["new_indices"], list) or not isinstance(
            record["selected_indices"], list
        ):
            raise ValueError(f"artifact round {expected_round} indices must be lists")
        query_size = _integer(
            record["query_size"],
            f"round {expected_round} query_size",
            minimum=1,
        )
        new_indices = [
            _integer(index, f"round {expected_round} new index") for index in record["new_indices"]
        ]
        selected_indices = [
            _integer(index, f"round {expected_round} selected index")
            for index in record["selected_indices"]
        ]
        if len(new_indices) != query_size or len(set(new_indices)) != query_size:
            raise ValueError(f"artifact round {expected_round} has invalid new_indices")
        if set(previous_selected).intersection(new_indices):
            raise ValueError(f"artifact round {expected_round} reselects an index")
        expected_selected = previous_selected + new_indices
        if selected_indices != expected_selected:
            raise ValueError(f"artifact round {expected_round} violates ordered nestedness")
        cumulative_budget += query_size
        if (
            _integer(
                record["cumulative_budget"],
                f"round {expected_round} cumulative_budget",
                minimum=1,
            )
            != cumulative_budget
        ):
            raise ValueError(f"artifact round {expected_round} has wrong cumulative_budget")

        seeds = record["seeds"]
        if not isinstance(seeds, dict):
            raise ValueError(f"artifact round {expected_round} seeds must be an object")
        _require_exact_fields(seeds, SEED_FIELDS, f"round {expected_round}.seeds")
        if _integer(seeds["replicate"], f"round {expected_round} replicate seed") != replicate_seed:
            raise ValueError(f"artifact round {expected_round} replicate seed mismatch")
        for seed_name, seed_value in seeds.items():
            _integer(seed_value, f"round {expected_round} {seed_name} seed")

        _integer(record["trained_epochs"], f"round {expected_round} trained_epochs", minimum=1)
        _finite_nonnegative(record["test_loss"], "test_loss")
        accuracy = _finite_nonnegative(record["test_accuracy"], "test_accuracy")
        if accuracy > 1.0:
            raise ValueError(f"artifact round {expected_round} test_accuracy exceeds one")
        diagnostics = record["selection_diagnostics"]
        if not isinstance(diagnostics, dict):
            raise ValueError(f"artifact round {expected_round} diagnostics must be an object")
        _require_exact_fields(
            diagnostics,
            DIAGNOSTIC_FIELDS,
            f"round {expected_round}.selection_diagnostics",
        )
        distance_values = {}
        for name in (
            "nearest_distance_mean",
            "nearest_distance_p95",
            "nearest_distance_max",
        ):
            distance_values[name] = _finite_nonnegative(
                diagnostics[name], f"round {expected_round}.{name}"
            )
        if (
            distance_values["nearest_distance_mean"] > distance_values["nearest_distance_max"]
            or distance_values["nearest_distance_p95"] > distance_values["nearest_distance_max"]
        ):
            raise ValueError(f"artifact round {expected_round} distance diagnostics are unordered")
        cosine = _finite_number(
            diagnostics["selected_pairwise_cosine_mean"],
            f"round {expected_round} pairwise cosine",
        )
        if not -1.0 <= cosine <= 1.0:
            raise ValueError(f"artifact round {expected_round} has invalid pairwise cosine")
        _finite_nonnegative(
            diagnostics["selected_typicality_mean"],
            f"round {expected_round} typicality",
        )

        method_metadata = record["method_metadata"]
        if not isinstance(method_metadata, dict):
            raise ValueError(f"artifact round {expected_round} method_metadata must be an object")
        method = run_config["method"]
        if method in CCFL_METHODS:
            _require_exact_fields(
                method_metadata,
                {
                    "ccfl_variant",
                    "candidates_per_cluster",
                    "refine_steps",
                    "use_cluster_weights",
                },
                f"round {expected_round}.method_metadata",
            )
            configured_variants = payload["effective_config"].get("ccfl_variants")
            if not isinstance(configured_variants, dict) or method not in configured_variants:
                raise ValueError("artifact effective_config is missing CCFL variant metadata")
            configured = configured_variants[method]
            if not isinstance(configured, dict):
                raise ValueError("artifact effective_config has invalid CCFL variant metadata")
            expected_metadata = {"ccfl_variant": method, **configured}
            if method_metadata != expected_metadata:
                raise ValueError(f"artifact round {expected_round} CCFL metadata mismatch")
        elif method == "probcover":
            _require_exact_fields(
                method_metadata,
                {
                    "probcover_delta",
                    "probcover_radius_seed",
                    "probcover_cache_digest",
                },
                f"round {expected_round}.method_metadata",
            )
            if (
                _finite_nonnegative(
                    method_metadata["probcover_delta"],
                    f"round {expected_round} ProbCover delta",
                )
                <= 0.0
            ):
                raise ValueError(f"artifact round {expected_round} has invalid ProbCover delta")
            _integer(
                method_metadata["probcover_radius_seed"],
                f"round {expected_round} ProbCover radius seed",
            )
            cache_digest = method_metadata["probcover_cache_digest"]
            if (
                not isinstance(cache_digest, str)
                or len(cache_digest) != 64
                or any(character not in "0123456789abcdef" for character in cache_digest)
            ):
                raise ValueError(
                    f"artifact round {expected_round} has invalid ProbCover cache digest"
                )
        elif method_metadata:
            raise ValueError(
                f"artifact round {expected_round} has unexpected method_metadata for {method}"
            )
        previous_selected = selected_indices

    timings = payload["timings"]
    if not isinstance(timings, dict):
        raise ValueError("artifact timings must be an object")
    _require_exact_fields(timings, TIMING_FIELDS, "timings")
    _finite_nonnegative(timings["total_seconds"], "total_seconds")
    round_seconds = timings["round_seconds"]
    if not isinstance(round_seconds, list) or len(round_seconds) != len(rounds):
        raise ValueError("artifact round_seconds must align with rounds")
    for duration in round_seconds:
        _finite_nonnegative(duration, "round_seconds")
    selector_seconds = timings["selector_seconds"]
    if not isinstance(selector_seconds, list) or len(selector_seconds) != len(rounds):
        raise ValueError("artifact selector_seconds must align with rounds")
    for duration in selector_seconds:
        _finite_nonnegative(duration, "selector_seconds")


def file_sha256(path: str | Path) -> str:
    """Hash a file incrementally with SHA-256."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _required_package_version(package_name: str) -> str:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"cannot record dependency version: {package_name}") from exc


def collect_environment(
    *,
    device: torch.device,
    checkpoint_path: str | Path | None,
) -> dict[str, Any]:
    """Collect code, dependency, device, and checkpoint provenance for a run."""
    try:
        git_commit = _git_output("rev-parse", "HEAD")
        git_dirty = bool(_git_output("status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("cannot record Git provenance for protocol-v2 run") from exc

    package_names = (
        "huggingface-hub",
        "numpy",
        "pandas",
        "safetensors",
        "scikit-learn",
        "scipy",
        "torch",
        "torchvision",
        "transformers",
        "truststore",
    )
    packages = {
        package_name: _required_package_version(package_name) for package_name in package_names
    }
    packages["torch"] = torch.__version__

    resolved_checkpoint: str | None = None
    checkpoint_digest: str | None = None
    if checkpoint_path is not None:
        checkpoint = Path(checkpoint_path).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"required representation checkpoint is missing: {checkpoint}")
        resolved_checkpoint = str(checkpoint)
        checkpoint_digest = file_sha256(checkpoint)

    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
    else:
        device_name = platform.processor() or "cpu"

    return {
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "device": f"{device.type}:{device_name}",
        "cuda_runtime": torch.version.cuda,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "checkpoint_path": resolved_checkpoint,
        "checkpoint_sha256": checkpoint_digest,
    }
