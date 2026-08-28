from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml

from .protocol import PROTOCOL_VERSION, TEST_POLICY, validate_round_query_sizes

TOP_LEVEL_FIELDS = {
    "protocol",
    "output",
    "data",
    "representation",
    "selection",
    "evaluation",
    "experiment",
    "ccfl_variants",
    "probcover",
}
CCFL_METHODS = frozenset(
    {
        "tpcrp_ccfl",
        "ccfl_candidate_only",
        "ccfl_unweighted",
        "ccfl_weighted",
    }
)
PROTOCOL_METHODS = frozenset({"random", "tpcrp", "kcenter", "probcover", *CCFL_METHODS})


def _require_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a mapping")
    return value


def _require_exact_fields(
    value: dict[str, Any],
    expected: set[str],
    context: str,
) -> None:
    missing = expected - value.keys()
    if missing:
        raise ValueError(f"{context} is missing keys: {sorted(missing)}")
    unknown = value.keys() - expected
    if unknown:
        raise ValueError(f"{context} has unknown keys: {sorted(unknown)}")


def _positive_int(value: Any, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{context} must be a positive integer")
    return value


def _finite_number(value: Any, context: str, *, minimum: float | None = None) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{context} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{context} must be finite")
    if minimum is not None and number < minimum:
        raise ValueError(f"{context} must be at least {minimum}")
    return number


def validate_protocol_config(config: dict[str, Any]) -> None:
    """Validate the complete protocol-v2 configuration without hidden defaults."""
    _require_exact_fields(config, TOP_LEVEL_FIELDS, "config")

    protocol = _require_mapping(config["protocol"], "protocol")
    _require_exact_fields(
        protocol,
        {"version", "deterministic", "test_evaluations_per_round"},
        "protocol",
    )
    if protocol["version"] != PROTOCOL_VERSION:
        raise ValueError("protocol.version mismatch")
    if protocol["deterministic"] is not True:
        raise ValueError("protocol.deterministic must be true")
    if protocol["test_evaluations_per_round"] != 1:
        raise ValueError("protocol.test_evaluations_per_round must equal 1")

    output = _require_mapping(config["output"], "output")
    _require_exact_fields(output, {"root", "resume"}, "output")
    if not isinstance(output["root"], str) or not output["root"]:
        raise ValueError("output.root must be a non-empty string")
    if not isinstance(output["resume"], bool):
        raise ValueError("output.resume must be boolean")

    data = _require_mapping(config["data"], "data")
    _require_exact_fields(data, {"name", "root", "num_workers", "num_classes"}, "data")
    if data["name"] != "cifar10":
        raise ValueError("protocol v2 supports data.name=cifar10 only")
    if not isinstance(data["root"], str) or not data["root"]:
        raise ValueError("data.root must be a non-empty string")
    if (
        not isinstance(data["num_workers"], int)
        or isinstance(data["num_workers"], bool)
        or data["num_workers"] < 0
    ):
        raise ValueError("data.num_workers must be a non-negative integer")
    _positive_int(data["num_classes"], "data.num_classes")
    if data["num_classes"] != 10:
        raise ValueError("data.num_classes must equal 10 for cifar10")

    representation = _require_mapping(config["representation"], "representation")
    backend = representation.get("backend")
    if backend == "simclr":
        expected_representation_fields = {
            "backend",
            "checkpoint_path",
            "projection_dim",
            "batch_size",
            "embedding_seed",
        }
    elif backend == "dinov2":
        expected_representation_fields = {
            "backend",
            "checkpoint_path",
            "model_id",
            "model_revision",
            "weights_sha256",
            "feature_dim",
            "resize_size",
            "crop_size",
            "batch_size",
            "embedding_seed",
        }
    else:
        raise ValueError("representation.backend must be 'simclr' or 'dinov2'")
    _require_exact_fields(representation, expected_representation_fields, "representation")
    if (
        not isinstance(representation["checkpoint_path"], str)
        or not representation["checkpoint_path"]
    ):
        raise ValueError("representation.checkpoint_path must be a non-empty string")
    if backend == "simclr":
        _positive_int(representation["projection_dim"], "representation.projection_dim")
    else:
        if representation["model_id"] != "facebook/dinov2-small":
            raise ValueError("representation.model_id must equal facebook/dinov2-small")
        revision = representation["model_revision"]
        if (
            not isinstance(revision, str)
            or len(revision) != 40
            or any(character not in "0123456789abcdef" for character in revision)
        ):
            raise ValueError("representation.model_revision must be a lowercase Git commit SHA")
        weights_digest = representation["weights_sha256"]
        if (
            not isinstance(weights_digest, str)
            or len(weights_digest) != 64
            or any(character not in "0123456789abcdef" for character in weights_digest)
        ):
            raise ValueError("representation.weights_sha256 must be a lowercase SHA-256")
        if _positive_int(representation["feature_dim"], "representation.feature_dim") != 384:
            raise ValueError("representation.feature_dim must equal 384 for dinov2-small")
        resize_size = _positive_int(
            representation["resize_size"], "representation.resize_size"
        )
        crop_size = _positive_int(representation["crop_size"], "representation.crop_size")
        if crop_size > resize_size:
            raise ValueError("representation.crop_size must not exceed resize_size")
        if crop_size % 14:
            raise ValueError("representation.crop_size must be a multiple of 14")
    _positive_int(representation["batch_size"], "representation.batch_size")
    if not isinstance(representation["embedding_seed"], int) or isinstance(
        representation["embedding_seed"], bool
    ):
        raise ValueError("representation.embedding_seed must be an integer")

    selection = _require_mapping(config["selection"], "selection")
    _require_exact_fields(
        selection,
        {"round_query_sizes", "knn_k", "max_clusters", "min_cluster_size"},
        "selection",
    )
    round_query_sizes = validate_round_query_sizes(selection["round_query_sizes"])
    _positive_int(selection["knn_k"], "selection.knn_k")
    max_clusters = _positive_int(selection["max_clusters"], "selection.max_clusters")
    _positive_int(selection["min_cluster_size"], "selection.min_cluster_size")
    if max_clusters < sum(round_query_sizes):
        raise ValueError("selection.max_clusters must cover the final cumulative budget")
    if sum(round_query_sizes) >= 50_000:
        raise ValueError("selection schedule must leave an unlabeled CIFAR-10 diagnostic pool")

    evaluation = _require_mapping(config["evaluation"], "evaluation")
    _require_exact_fields(
        evaluation,
        {
            "framework",
            "epochs",
            "batch_size",
            "lr",
            "momentum",
            "weight_decay",
            "dropout_p",
            "test_policy",
        },
        "evaluation",
    )
    if evaluation["framework"] not in {
        "fully_supervised",
        "ssl_embedding",
        "label_spreading_proxy",
    }:
        raise ValueError("evaluation.framework is unsupported")
    _positive_int(evaluation["epochs"], "evaluation.epochs")
    _positive_int(evaluation["batch_size"], "evaluation.batch_size")
    learning_rate = _finite_number(evaluation["lr"], "evaluation.lr", minimum=0.0)
    if learning_rate <= 0.0:
        raise ValueError("evaluation.lr must be positive")
    _finite_number(evaluation["momentum"], "evaluation.momentum", minimum=0.0)
    _finite_number(evaluation["weight_decay"], "evaluation.weight_decay", minimum=0.0)
    dropout = _finite_number(evaluation["dropout_p"], "evaluation.dropout_p", minimum=0.0)
    if dropout >= 1.0:
        raise ValueError("evaluation.dropout_p must be less than 1")
    if evaluation["test_policy"] != TEST_POLICY:
        raise ValueError("evaluation.test_policy mismatch")

    experiment = _require_mapping(config["experiment"], "experiment")
    _require_exact_fields(
        experiment,
        {"methods", "replicate_seeds", "primary_comparison"},
        "experiment",
    )
    methods = experiment["methods"]
    if (
        not isinstance(methods, list)
        or not methods
        or any(not isinstance(method, str) or not method for method in methods)
        or len(methods) != len(set(methods))
    ):
        raise ValueError("experiment.methods must be unique non-empty strings")
    unsupported_methods = set(methods) - PROTOCOL_METHODS
    if unsupported_methods:
        raise ValueError(
            f"methods are not enabled by the baseline-fidelity protocol: "
            f"{sorted(unsupported_methods)}"
        )
    replicate_seeds = experiment["replicate_seeds"]
    if (
        not isinstance(replicate_seeds, list)
        or not replicate_seeds
        or any(
            not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
            for seed in replicate_seeds
        )
        or len(replicate_seeds) != len(set(replicate_seeds))
    ):
        raise ValueError("experiment.replicate_seeds must be unique non-negative integers")
    comparison = _require_mapping(experiment["primary_comparison"], "primary_comparison")
    _require_exact_fields(
        comparison,
        {"method_a", "method_b", "cumulative_budget", "metric"},
        "experiment.primary_comparison",
    )
    if comparison["method_a"] not in methods or comparison["method_b"] not in methods:
        raise ValueError("primary comparison methods must be configured")
    cumulative_budgets: list[int] = []
    running_budget = 0
    for query_size in round_query_sizes:
        running_budget += query_size
        cumulative_budgets.append(running_budget)
    if comparison["cumulative_budget"] not in cumulative_budgets:
        raise ValueError("primary comparison cumulative budget is not in the schedule")
    if comparison["metric"] != "test_accuracy":
        raise ValueError("primary comparison metric must be test_accuracy")

    ccfl_variants = _require_mapping(config["ccfl_variants"], "ccfl_variants")
    _require_exact_fields(ccfl_variants, set(CCFL_METHODS), "ccfl_variants")
    for variant_name in sorted(CCFL_METHODS):
        variant = _require_mapping(
            ccfl_variants[variant_name],
            f"ccfl_variants.{variant_name}",
        )
        _require_exact_fields(
            variant,
            {"candidates_per_cluster", "refine_steps", "use_cluster_weights"},
            f"ccfl_variants.{variant_name}",
        )
        _positive_int(
            variant["candidates_per_cluster"],
            f"ccfl_variants.{variant_name}.candidates_per_cluster",
        )
        if (
            not isinstance(variant["refine_steps"], int)
            or isinstance(variant["refine_steps"], bool)
            or variant["refine_steps"] < 0
        ):
            raise ValueError(
                f"ccfl_variants.{variant_name}.refine_steps must be a non-negative integer"
            )
        if not isinstance(variant["use_cluster_weights"], bool):
            raise ValueError(f"ccfl_variants.{variant_name}.use_cluster_weights must be boolean")

    if ccfl_variants["ccfl_candidate_only"]["refine_steps"] != 0:
        raise ValueError("ccfl_candidate_only must set refine_steps=0")
    for variant_name in ("tpcrp_ccfl", "ccfl_unweighted", "ccfl_weighted"):
        if ccfl_variants[variant_name]["refine_steps"] <= 0:
            raise ValueError(f"{variant_name} must enable facility-location refinement")
    if ccfl_variants["ccfl_unweighted"]["use_cluster_weights"] is not False:
        raise ValueError("ccfl_unweighted must disable cluster weights")
    for variant_name in ("tpcrp_ccfl", "ccfl_weighted"):
        if ccfl_variants[variant_name]["use_cluster_weights"] is not True:
            raise ValueError(f"{variant_name} must enable cluster weights")

    probcover = _require_mapping(config["probcover"], "probcover")
    _require_exact_fields(probcover, {"alpha", "delta_search"}, "probcover")
    alpha = _finite_number(probcover["alpha"], "probcover.alpha", minimum=0.0)
    if alpha <= 0.0 or alpha > 1.0:
        raise ValueError("probcover.alpha must be in (0, 1]")
    delta = _require_mapping(probcover["delta_search"], "probcover.delta_search")
    _require_exact_fields(delta, {"minimum", "maximum", "step"}, "probcover.delta_search")
    minimum = _finite_number(delta["minimum"], "probcover.delta_search.minimum", minimum=0.0)
    maximum = _finite_number(delta["maximum"], "probcover.delta_search.maximum", minimum=0.0)
    step = _finite_number(delta["step"], "probcover.delta_search.step", minimum=0.0)
    if minimum <= 0.0 or maximum < minimum or step <= 0.0:
        raise ValueError("probcover.delta_search has an invalid range")


def load_configurations(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration from disk into a mapping."""
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    return config
