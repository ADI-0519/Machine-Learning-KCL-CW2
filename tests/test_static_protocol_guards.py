from __future__ import annotations

import ast
import subprocess
from pathlib import Path


def test_legacy_test_selected_metric_names_are_absent() -> None:
    forbidden = ("best_test_accuracy", "final_test_accuracy")
    roots = (Path("src"), Path("scripts"), Path("configs"))
    offenders: list[str] = []

    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in {".py", ".yaml", ".yml"}:
                continue
            text = path.read_text(encoding="utf-8")
            offenders.extend(f"{path}: {term}" for term in forbidden if term in text)

    assert offenders == []


def test_public_framework_name_is_explicitly_a_proxy() -> None:
    config = Path("configs/default.yaml").read_text(encoding="utf-8")
    assert "label_spreading_proxy" in config
    assert '"semi_' + 'supervised"' not in config


def test_noncanonical_uncertainty_method_ids_are_explicitly_proxies() -> None:
    config = Path("configs/default.yaml").read_text(encoding="utf-8")
    for method in ("dbal", "bald", "badge"):
        assert f'"{method}_proxy"' in config
        assert f'"{method}"' not in config


def test_selectors_cannot_import_targets_or_posthoc_label_analysis() -> None:
    path = Path("src/selectors.py")
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    forbidden_modules = {"data", "evaluate", "reporting"}
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name.split(".")[-1] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module.split(".")[-1])

    assert imported_modules.isdisjoint(forbidden_modules)
    assert "posthoc_class_balance" not in source
    assert ".targets" not in source


def test_posthoc_class_balance_stays_outside_selection_and_training_modules() -> None:
    offenders = []
    for path in Path("src").glob("*.py"):
        if path.name == "reporting.py":
            continue
        if "posthoc_class_balance" in path.read_text(encoding="utf-8"):
            offenders.append(str(path))
    assert offenders == []


def test_generated_protocol_roots_are_git_ignored() -> None:
    generated_paths = [
        "results/protocol_v2/runs/example.json",
        "results/protocol_v2_confirmation_simclr/runs/example.json",
        "results/protocol_v2_confirmation_dinov2/runs/example.json",
    ]
    for generated_path in generated_paths:
        completed = subprocess.run(
            ["git", "check-ignore", "--quiet", generated_path],
            check=False,
        )
        assert completed.returncode == 0, f"generated output is not ignored: {generated_path}"
