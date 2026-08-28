from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.statistics import holm_adjust, pair_method_rows, paired_bootstrap_ci


def _pairing_fixture() -> pd.DataFrame:
    rows = []
    for seed, accuracy_a, accuracy_b in ((1, 0.60, 0.55), (2, 0.70, 0.65)):
        common = {
            "dataset": "cifar10",
            "framework": "ssl_embedding",
            "cumulative_budget": 10,
            "replicate_seed": seed,
        }
        rows.append(
            {
                **common,
                "method": "tpcrp_ccfl",
                "run_id": f"a-{seed}",
                "test_accuracy": accuracy_a,
            }
        )
        rows.append(
            {
                **common,
                "method": "tpcrp",
                "run_id": f"b-{seed}",
                "test_accuracy": accuracy_b,
            }
        )
    return pd.DataFrame(rows)


def test_pair_method_rows_returns_sorted_one_to_one_pairs() -> None:
    paired = pair_method_rows(
        _pairing_fixture().sample(frac=1.0, random_state=7),
        method_a="tpcrp_ccfl",
        method_b="tpcrp",
    )

    assert paired["replicate_seed"].tolist() == [1, 2]
    assert paired["run_id_a"].tolist() == ["a-1", "a-2"]
    assert paired["run_id_b"].tolist() == ["b-1", "b-2"]
    np.testing.assert_allclose(
        paired["test_accuracy_a"] - paired["test_accuracy_b"],
        np.array([0.05, 0.05]),
    )


def test_pair_method_rows_rejects_duplicate_pairing_key() -> None:
    rows = _pairing_fixture()
    rows = pd.concat([rows, rows.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate pairing key"):
        pair_method_rows(rows, method_a="tpcrp_ccfl", method_b="tpcrp")


def test_pair_method_rows_rejects_missing_method_partner() -> None:
    rows = _pairing_fixture().drop(index=3)
    with pytest.raises(ValueError, match="missing method partner"):
        pair_method_rows(rows, method_a="tpcrp_ccfl", method_b="tpcrp")


def test_pair_method_rows_rejects_same_method_and_missing_columns() -> None:
    with pytest.raises(ValueError, match="must differ"):
        pair_method_rows(_pairing_fixture(), method_a="tpcrp", method_b="tpcrp")
    with pytest.raises(ValueError, match="missing columns"):
        pair_method_rows(
            _pairing_fixture().drop(columns="run_id"),
            method_a="tpcrp_ccfl",
            method_b="tpcrp",
        )


def test_paired_bootstrap_ci_has_fixed_expected_interval() -> None:
    differences = np.array([0.01, 0.02, 0.04, 0.08], dtype=np.float64)
    assert paired_bootstrap_ci(differences) == pytest.approx((0.015, 0.065))
    assert paired_bootstrap_ci(differences) == paired_bootstrap_ci(differences)


@pytest.mark.parametrize(
    ("differences", "kwargs", "message"),
    [
        ([0.1], {}, "at least two"),
        ([0.1, np.nan], {}, "finite vector"),
        ([[0.1, 0.2]], {}, "finite vector"),
        ([0.1, 0.2], {"confidence": 1.0}, "confidence"),
        ([0.1, 0.2], {"resamples": 0}, "resamples"),
        ([0.1, 0.2], {"seed": True}, "seed"),
    ],
)
def test_paired_bootstrap_ci_rejects_invalid_inputs(differences, kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        paired_bootstrap_ci(np.asarray(differences), **kwargs)


def test_holm_adjust_has_fixed_expected_values_and_preserves_input_order() -> None:
    adjusted = holm_adjust(np.array([0.01, 0.04, 0.03, 0.20]))
    np.testing.assert_allclose(adjusted, np.array([0.04, 0.09, 0.09, 0.20]))


@pytest.mark.parametrize("values", [[], [[0.1]], [-0.1], [1.1], [np.nan], [True]])
def test_holm_adjust_rejects_invalid_values(values) -> None:
    with pytest.raises(ValueError, match="p-values"):
        holm_adjust(np.asarray(values))
