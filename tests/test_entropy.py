# Unit tests for compute_entropy() and aggregate_from_granular() in
# 04_metrics/entropy.py: normalized Shannon entropy (H / ln k) of cluster
# occupancy, and averaging it across runs then instances.
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "04_metrics"))

import pandas as pd
import pytest
from entropy import compute_entropy, aggregate_from_granular


def _write_cluster_distribution(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def test_perfectly_balanced_occupancy_has_entropy_one(tmp_path):
    # 2 clusters, counts [5, 5] -> p=[0.5, 0.5] -> maximum entropy for k=2,
    # normalized by ln(2) -> exactly 1.0.
    csv_path = tmp_path / "F1_I1.csv"
    _write_cluster_distribution(csv_path, [
        {"algorithm": "A", "run": 1, "iteration": 1, "c0": 5, "c1": 5},
    ])
    result = compute_entropy(str(csv_path))
    assert len(result) == 1
    assert result.iloc[0]["entropy"] == pytest.approx(1.0, abs=1e-9)


def test_fully_concentrated_occupancy_has_entropy_zero(tmp_path):
    # all agents in one cluster -> p=[1.0, 0.0] -> entropy 0.
    csv_path = tmp_path / "F1_I1.csv"
    _write_cluster_distribution(csv_path, [
        {"algorithm": "A", "run": 1, "iteration": 1, "c0": 10, "c1": 0},
    ])
    result = compute_entropy(str(csv_path))
    assert len(result) == 1
    assert result.iloc[0]["entropy"] == pytest.approx(0.0, abs=1e-9)


def test_algorithms_of_interest_filters_rows(tmp_path):
    csv_path = tmp_path / "F1_I1.csv"
    _write_cluster_distribution(csv_path, [
        {"algorithm": "A", "run": 1, "iteration": 1, "c0": 5, "c1": 5},
        {"algorithm": "B", "run": 1, "iteration": 1, "c0": 5, "c1": 5},
    ])
    result = compute_entropy(str(csv_path), algorithms_of_interest=["A"])
    assert result["algorithm"].unique().tolist() == ["A"]


def test_zero_total_row_is_skipped(tmp_path):
    # an iteration where no agent occupies any cluster (total=0) must not
    # produce a row (would be a 0/0 division otherwise).
    csv_path = tmp_path / "F1_I1.csv"
    _write_cluster_distribution(csv_path, [
        {"algorithm": "A", "run": 1, "iteration": 1, "c0": 0, "c1": 0},
        {"algorithm": "A", "run": 1, "iteration": 2, "c0": 5, "c1": 5},
    ])
    result = compute_entropy(str(csv_path))
    assert result["iteration"].tolist() == [2]


def test_aggregate_from_granular_averages_runs_then_instances():
    # (A, problem 1, instance 1): runs 0.2 and 0.4 -> mean 0.3
    # (A, problem 1, instance 2): run 0.9 -> mean 0.9
    # mean across instances (0.3, 0.9) -> 0.6
    granular = pd.DataFrame([
        {"algorithm": "A", "problem_id": 1, "instance_id": 1, "run": 1, "iteration": 1, "entropy": 0.2},
        {"algorithm": "A", "problem_id": 1, "instance_id": 1, "run": 2, "iteration": 1, "entropy": 0.4},
        {"algorithm": "A", "problem_id": 1, "instance_id": 2, "run": 1, "iteration": 1, "entropy": 0.9},
    ])
    agg = aggregate_from_granular(granular)
    assert len(agg) == 1
    row = agg.iloc[0]
    assert row["algorithm"] == "A"
    assert row["function_class"] == 1
    assert row["iteration"] == 1
    assert row["mean_entropy"] == pytest.approx(0.6, abs=1e-9)


def test_aggregate_from_granular_empty_input_returns_empty_with_right_columns():
    agg = aggregate_from_granular(pd.DataFrame())
    assert len(agg) == 0
    assert list(agg.columns) == ["algorithm", "iteration", "function_class", "mean_entropy"]
