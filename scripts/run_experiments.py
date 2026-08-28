"""CLI for deterministic, resumable protocol-v2 experiment grids."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.artifacts import (
    artifact_path_for,
    build_effective_config,
    build_run_id,
    config_digest,
    read_artifact,
    validate_artifact,
)
from src.config import load_configurations, validate_protocol_config
from src.experiment import run_single_experiment


def run_experiment_grid(
    *,
    config_path: str | Path,
    method: str | None = None,
    seed: int | None = None,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Execute or plan a filtered protocol-v2 grid with validated resumption."""
    config_path = Path(config_path)
    config = load_configurations(config_path)
    validate_protocol_config(config)

    configured_methods = config["experiment"]["methods"]
    configured_seeds = config["experiment"]["replicate_seeds"]
    framework = config["evaluation"]["framework"]
    methods = configured_methods if method is None else [method]
    seeds = configured_seeds if seed is None else [seed]
    if any(candidate not in configured_methods for candidate in methods):
        raise ValueError(f"requested method is not configured: {method}")
    if any(candidate not in configured_seeds for candidate in seeds):
        raise ValueError(f"requested seed is not configured: {seed}")

    resume = config["output"]["resume"]
    records: list[dict[str, Any]] = []
    for replicate_seed in seeds:
        for configured_method in methods:
            effective_config = build_effective_config(
                config,
                method=configured_method,
                replicate_seed=replicate_seed,
                framework=framework,
            )
            run_id = build_run_id(effective_config)
            expected_digest = config_digest(effective_config)
            artifact_path = artifact_path_for(config["output"]["root"], run_id)
            record = {
                "run_id": run_id,
                "method": configured_method,
                "seed": replicate_seed,
                "framework": framework,
                "artifact_path": str(artifact_path),
            }

            if artifact_path.exists():
                artifact = read_artifact(artifact_path)
                if artifact["config_digest"] != expected_digest:
                    raise ValueError(f"existing artifact digest mismatch: {artifact_path}")
                if not resume:
                    raise FileExistsError(
                        f"immutable artifact already exists and resume is disabled: {artifact_path}"
                    )
                records.append({**record, "status": "skipped"})
                continue

            if dry_run:
                records.append({**record, "status": "planned"})
                continue

            artifact = run_single_experiment(
                config_path=config_path,
                method=configured_method,
                seed=replicate_seed,
                framework=framework,
            )
            validate_artifact(artifact)
            if artifact["run_id"] != run_id or artifact["config_digest"] != expected_digest:
                raise ValueError(f"runner returned the wrong artifact for run {run_id}")
            persisted = read_artifact(artifact_path)
            if persisted["config_digest"] != expected_digest:
                raise ValueError(f"persisted artifact digest mismatch: {artifact_path}")
            records.append({**record, "status": "completed"})
    return records


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/protocol_v2_pilot.yaml"),
        help="Protocol-v2 YAML configuration path.",
    )
    parser.add_argument("--method", help="Run one configured method only.")
    parser.add_argument("--seed", type=int, help="Run one configured replicate seed only.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and list jobs without loading data or executing models.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Parse CLI arguments and print one concise status line per run."""
    args = _build_parser().parse_args(argv)
    records = run_experiment_grid(
        config_path=args.config,
        method=args.method,
        seed=args.seed,
        dry_run=args.dry_run,
    )
    for record in records:
        print(
            f"[{record['status']}] {record['run_id']} "
            f"method={record['method']} seed={record['seed']} "
            f"artifact={record['artifact_path']}"
        )


if __name__ == "__main__":
    main()
