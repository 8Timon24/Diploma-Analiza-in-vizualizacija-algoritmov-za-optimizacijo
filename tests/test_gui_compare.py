# Unit tests for gui.viz.compare: the head-to-head view.
#
# The ordering trap is the one to pin. merged_dim_{d}.csv always stores a
# pair with Algorithm1 alphabetically first, so looking up ("WhaleFOA",
# "AugmentedAEO") in the order the user picked them finds nothing at all -
# silently, as an empty frame rather than an error.
#
# Synthetic fixtures only; no Qt, no real data.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest

# gui.viz.compare imports seaborn, matplotlib and scipy at module level, and
# CI installs only pandas/numpy/pytest. Skipping the whole module there is
# correct - these assertions are about plotting-layer logic.
pytest.importorskip("seaborn")
pytest.importorskip("scipy")

from gui.viz import data as viz_data  # noqa: E402

METRICS = ["entropy", "cosine", "cosine_columns", "exploration", "location", "fitness"]


@pytest.fixture
def merged(tmp_path, monkeypatch):
    """Four algorithms -> six pairs, with A/B deliberately the most similar."""
    values = {
        ("A", "B"): 0.1,
        ("A", "C"): 0.5,
        ("A", "D"): 0.6,
        ("B", "C"): 0.7,
        ("B", "D"): 0.8,
        ("C", "D"): 0.9,
    }
    rows = []
    for (first, second), value in values.items():
        for function in (1, 2, 3):
            rows.append({
                "Algorithm1": first, "Algorithm2": second,
                "Function_id": function, "Instance_id": 1, "Run_id": 1,
                **{m: value for m in METRICS},
            })
    pd.DataFrame(rows).to_csv(tmp_path / "merged_dim_2.csv", index=False)
    monkeypatch.setattr(viz_data.config, "MERGED_DIR", str(tmp_path))
    viz_data.load_merged.cache_clear()
    yield tmp_path
    viz_data.load_merged.cache_clear()


def test_pair_order_is_normalised_alphabetically():
    from gui.viz.compare import ordered_pair

    assert ordered_pair("WhaleFOA", "AugmentedAEO") == ("AugmentedAEO", "WhaleFOA")
    assert ordered_pair("AugmentedAEO", "WhaleFOA") == ("AugmentedAEO", "WhaleFOA")


def test_lookup_works_whichever_order_the_user_picked(merged):
    from gui.viz.compare import summary_table

    forward, _ = summary_table(2, "A", "B")
    backward, _ = summary_table(2, "B", "A")
    pd.testing.assert_frame_equal(forward, backward)


def test_percentile_places_the_pair_in_the_distribution(merged):
    from gui.viz.compare import summary_table

    table, total = summary_table(2, "A", "B")
    assert total == 6                      # six pairs from four algorithms
    # A/B is the smallest of the six values, so it is at the bottom.
    assert table["percentile"].max() == pytest.approx(100 / 6)

    table, _ = summary_table(2, "C", "D")  # the largest
    assert table["percentile"].min() == pytest.approx(100.0)


def test_every_metric_is_reported(merged):
    from gui.viz.compare import summary_table

    table, _ = summary_table(2, "A", "C")
    assert len(table) == len(METRICS)
    assert set(table.columns) == {"metric", "value", "percentile"}


def test_comparing_an_algorithm_with_itself_is_refused(merged):
    from gui.viz.compare import summary_table

    with pytest.raises(ValueError, match="two different algorithms"):
        summary_table(2, "A", "A")


def test_a_pair_absent_from_the_table_is_an_explicit_error(merged):
    from gui.viz.compare import summary_table

    with pytest.raises(ValueError, match="not compared"):
        summary_table(2, "A", "NotPresent")


def test_metric_profile_draws_one_row_per_metric(merged):
    pytest.importorskip("seaborn")
    from gui.viz.compare import metric_profile

    figure = metric_profile({"dimension": 2, "algorithm_a": "A", "algorithm_b": "B"})
    assert len(figure.get_axes()) == len(METRICS)


def test_per_function_profile_has_a_column_per_function(merged):
    pytest.importorskip("seaborn")
    from gui.viz.compare import per_function_profile

    figure = per_function_profile(
        {"dimension": 2, "algorithm_a": "A", "algorithm_b": "B"}
    )
    # one heatmap axes plus its colour bar
    assert len(figure.get_axes()) == 2
