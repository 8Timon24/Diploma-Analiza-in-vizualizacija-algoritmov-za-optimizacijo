# Unit tests for gui.core.coverage: deciding whether a visualization can be
# drawn from the data on disk, and what it would take to fill the gap.
#
# The GUI deliberately lets the user select algorithms and dimensions that
# were never benchmarked - hiding them would hide most of what mealpy offers.
# That makes "missing" a normal, explained state, so the detection has to be
# exact: claiming data exists when it does not produces a confusing crash
# deep inside a plotting function instead of a useful prompt.
#
# Synthetic fixtures only; no Qt, matplotlib, mealpy or real data.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest

from gui.core import coverage as cov


@pytest.fixture
def entropy_dir(tmp_path, monkeypatch):
    """An entropy table with 2 algorithms on functions 1 and 2, dim 2 only."""
    monkeypatch.setattr(cov.config, "ENTROPY_DATA_DIR", str(tmp_path))
    rows = [
        {"algorithm": algorithm, "iteration": 1.0, "function_class": function,
         "mean_entropy": 0.5}
        for algorithm in ("OriginalDE", "OriginalGWO")
        for function in (1, 2)
    ]
    pd.DataFrame(rows).to_csv(tmp_path / "entropy_dim_2.csv", index=False)
    cov.invalidate()
    yield tmp_path
    cov.invalidate()


# -- range formatting ---------------------------------------------------

def test_compact_collapses_consecutive_runs():
    assert cov._compact([1, 2, 3, 7, 8, 12]) == "1-3, 7-8, 12"
    assert cov._compact([5]) == "5"
    assert cov._compact([]) == ""


def test_compact_sorts_and_deduplicates():
    assert cov._compact([3, 1, 2, 2]) == "1-3"


# -- entropy coverage ---------------------------------------------------

def test_fully_present_selection_reports_no_gaps(entropy_dir):
    result = cov.check_entropy(2, ["OriginalDE", "OriginalGWO"], [1, 2])
    assert result.ok
    assert result.summary() == ""
    assert result.recipe == []


def test_unbenchmarked_algorithm_is_detected(entropy_dir):
    result = cov.check_entropy(2, ["OriginalDE", "ChaoticGWO"], [1])
    assert not result.ok
    assert "ChaoticGWO" in result.summary()
    # the one that IS present must not be reported as missing
    assert "OriginalDE" not in result.summary()


def test_missing_function_is_detected(entropy_dir):
    result = cov.check_entropy(2, ["OriginalDE"], [1, 24])
    assert not result.ok
    assert "24" in result.summary()


def test_dimension_with_no_table_at_all_is_detected(entropy_dir):
    result = cov.check_entropy(3, ["OriginalDE"], [1])
    assert not result.ok
    assert "dimension 3" in result.summary()


def test_recipe_names_the_missing_algorithms_and_dimension(entropy_dir):
    result = cov.check_entropy(2, ["ChaoticGWO", "FuzzyGWO"], [7])
    recipe = "\n".join(result.recipe)
    assert "ChaoticGWO" in recipe and "FuzzyGWO" in recipe
    assert "-d 2" in recipe
    assert "run_benchmarks.py" in recipe
    # the whole chain down to the entropy table, not just the benchmark
    assert "preprocess_data.py" in recipe
    assert "cluster_trajectories.py" in recipe
    assert "entropy.py" in recipe


def test_gap_reports_the_clustering_side_effect(entropy_dir):
    # Adding an algorithm re-clusters the problem, which moves every other
    # algorithm's metrics. Silently omitting that would be misleading.
    result = cov.check_entropy(2, ["ChaoticGWO"], [1])
    assert "cluster" in result.caveat.lower()
    assert result.estimate


def test_no_dimension_selected_is_a_gap_not_a_crash(entropy_dir):
    assert not cov.check_entropy(None, ["OriginalDE"], [1]).ok


def test_multi_dimension_check_accumulates_gaps(entropy_dir):
    result = cov.check_entropy_multi([2, 3], ["OriginalDE"], [1])
    assert not result.ok
    assert "dimension 3" in result.summary()


def test_multi_dimension_check_passes_when_all_present(entropy_dir):
    assert cov.check_entropy_multi([2], ["OriginalDE"], [1, 2]).ok


# -- file-backed checks -------------------------------------------------

def test_missing_file_produces_a_gap_with_a_recipe(tmp_path):
    result = cov.check_file(tmp_path / "absent.csv", "nothing here", ["do a thing"])
    assert not result.ok
    assert result.recipe == ["do a thing"]


def test_present_file_produces_no_gap(tmp_path):
    path = tmp_path / "present.csv"
    path.write_text("a,b\n1,2\n")
    assert cov.check_file(path, "nothing here", ["do a thing"]).ok


def test_scalars_check_spots_a_dimension_absent_from_the_table(tmp_path, monkeypatch):
    path = tmp_path / "scalars.csv"
    pd.DataFrame({"dim": [2, 5], "func": [1, 1], "algo": ["A", "A"],
                  "entropy": [0.1, 0.2]}).to_csv(path, index=False)
    monkeypatch.setattr(cov.config, "SCALARS_CSV", str(path))
    assert cov.check_scalars([2, 5]).ok
    assert not cov.check_scalars([2, 20]).ok


def test_invalidate_lets_new_data_be_seen(entropy_dir):
    assert not cov.check_entropy(3, ["OriginalDE"], [1]).ok
    pd.DataFrame([{"algorithm": "OriginalDE", "iteration": 1.0,
                   "function_class": 1, "mean_entropy": 0.5}]).to_csv(
        entropy_dir / "entropy_dim_3.csv", index=False)
    cov.invalidate()   # what the GUI calls after a run finishes
    assert cov.check_entropy(3, ["OriginalDE"], [1]).ok
