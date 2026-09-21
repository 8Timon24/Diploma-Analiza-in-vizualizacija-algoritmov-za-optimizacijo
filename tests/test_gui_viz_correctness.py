# The visualization bugs that render perfectly and show the wrong number.
#
# tests/test_gui_viz.py checks that every catalog entry can be CALLED - it
# catches KeyError and then swallows everything else. That cannot see a figure
# whose values are wrong, which is how all five of the bugs below survived. So
# these pin the values, not the call.
#
# Synthetic fixtures only: no real data, no cocoex, no mealpy. matplotlib,
# pandas, numpy, seaborn and scipy are needed, so the whole module skips in CI
# (which installs only pandas/numpy/pytest).
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("seaborn")

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
from matplotlib.figure import Figure


# -- 1. the running minimum must be per cluster -------------------------
#
# clustering.cluster_occupancy drew "Best fitness found per cluster" from
#   groupby(["iteration", "cluster"]).min().cummin()
# .cummin() runs over the FLATTENED (iteration, cluster) index, so one
# cluster's minimum carried into the next and every line traced the same
# global running minimum. On real dim-2 data all 21 clusters reported one
# identical value at iteration 1.

def _running_minimum(frame):
    # The real function, not a copy of the expression: a reimplementation
    # here would keep passing while the panel stayed broken.
    from gui.viz.clustering import best_fitness_per_cluster

    return best_fitness_per_cluster(frame)


def test_running_minimum_is_taken_within_each_cluster():
    # Cluster 0 is good and flat; cluster 1 is bad and improves. Neither may
    # ever take the other's value.
    frame = pd.DataFrame({
        "iteration": [1, 1, 2, 2, 3, 3],
        "cluster":   [0, 1, 0, 1, 0, 1],
        "scaled_raw_y": [0.1, 0.9, 0.2, 0.8, 0.3, 0.7],
    })
    best = _running_minimum(frame)

    cluster_1 = best[best["cluster"] == 1].sort_values("iteration")
    assert list(cluster_1["scaled_raw_y"]) == [0.9, 0.8, 0.7], (
        "cluster 1 picked up cluster 0's minimum - .cummin() was not grouped"
    )
    cluster_0 = best[best["cluster"] == 0].sort_values("iteration")
    assert list(cluster_0["scaled_raw_y"]) == [0.1, 0.1, 0.1]


def test_clusters_do_not_collapse_to_one_value_at_the_first_iteration():
    """The shape the bug took on real data: every cluster identical."""
    frame = pd.DataFrame({
        "iteration": [1] * 4,
        "cluster": [0, 1, 2, 3],
        "scaled_raw_y": [0.4, 0.1, 0.9, 0.6],
    })
    best = _running_minimum(frame)
    assert best["scaled_raw_y"].nunique() == 4


# -- 2. exploration must be rescaled exactly once -----------------------
#
# 04_metrics/exploration_pairwise.py already divides the percentage-point
# difference by 100, so merged_dim_{d}.csv carries a 0-1 fraction. metrics.py
# divided again, putting the whole column at ~0.003: every cell of the
# Exploration column was annotated "0.00" on the default selection.

def _merged_frame():
    pairs = [("A", "B"), ("A", "C"), ("B", "C")]
    rows = []
    for index, (first, second) in enumerate(pairs):
        rows.append({
            "Algorithm1": first, "Algorithm2": second,
            "Function_id": 1, "Instance_id": 1, "Run_id": 1,
            "entropy": 0.1 * (index + 1),
            "cosine": 0.2 * (index + 1),
            "cosine_columns": 0.3,
            "exploration": 0.25 * (index + 1),   # already a 0-1 fraction
            "location": 1.0 * (index + 1),
            "fitness": 10.0 * (index + 1),
        })
    return pd.DataFrame(rows)


@pytest.fixture
def merged(monkeypatch, tmp_path):
    """Point load_merged at a synthetic merged table."""
    from gui.viz import data as viz_data

    path = tmp_path / "merged_dim_2.csv"
    _merged_frame().to_csv(path, index=False)
    monkeypatch.setattr(viz_data.config, "MERGED_DIR", str(tmp_path))
    viz_data.load_merged.cache_clear()
    yield
    viz_data.load_merged.cache_clear()


def test_exploration_is_not_divided_by_a_hundred_twice(merged):
    from gui.viz import metrics

    figure = metrics.metric_table({"dimension": 2, "algorithms": ["A", "B", "C"]})
    annotations = {t.get_text() for t in figure.get_axes()[0].texts}

    # 0.25 / 0.50 / 0.75 must survive to the plot. Dividing again would make
    # every one of them render as "0.00".
    assert {"0.25", "0.50", "0.75"} <= annotations, (
        f"exploration was rescaled twice; annotations were {sorted(annotations)}"
    )
    assert sum(a == "0.00" for a in annotations) < 3


# -- 3. a similarity matrix's diagonal is self-SIMILAR ------------------
#
# cluster_similarity.py skips self-pairs, so the pivot arrives with an
# all-NaN diagonal. clustering.algorithm_similarity filled it with 0.0 -
# "maximally dissimilar to itself" - which both mis-coloured the diagonal and
# fed a wrong row vector to the linkage the dendrogram is built from.

def test_self_similarity_is_filled_as_similar_not_distant():
    from gui.viz import clustering

    matrix = pd.DataFrame(
        [[np.nan, 0.4, 0.2],
         [0.4, np.nan, 0.6],
         [0.2, 0.6, np.nan]],
        index=["A", "B", "C"], columns=["A", "B", "C"],
    )
    filled = clustering._fill_self_similarity(matrix)

    assert list(np.diag(filled.to_numpy())) == [1.0, 1.0, 1.0]
    # the off-diagonal must be untouched
    assert filled.loc["A", "B"] == 0.4
    assert filled.loc["B", "C"] == 0.6


def test_a_genuinely_missing_pair_is_reported_not_invented():
    from gui.viz import clustering

    matrix = pd.DataFrame(
        [[np.nan, np.nan, 0.2],
         [np.nan, np.nan, 0.6],
         [0.2, 0.6, np.nan]],
        index=["A", "B", "C"], columns=["A", "B", "C"],
    )
    with pytest.raises(ValueError, match="no aggregate similarity"):
        clustering._fill_self_similarity(matrix)


# -- 4. a pair with no data is not "identical" --------------------------
#
# metrics._distance_matrix started from np.zeros, so a pair absent from the
# merged table kept distance 0 and linkage merged those algorithms at height
# 0 - an algorithm with no data looked like a perfect twin of its neighbour.

def test_absent_pair_is_nan_rather_than_zero_distance():
    from gui.viz import metrics

    means = pd.DataFrame({
        "Algorithm1": ["A", "B"],
        "Algorithm2": ["B", "C"],
        "cosine": [0.9, 0.8],
    })
    matrix = metrics._distance_matrix(means, "cosine", ["A", "B", "C"])

    assert np.isnan(matrix.loc["A", "C"]), (
        "an uncomputed pair reads as distance 0, i.e. identical algorithms"
    )
    assert matrix.loc["A", "A"] == 0.0        # the diagonal is still self-distance
    assert matrix.loc["A", "B"] == 0.9


def test_dendrograms_say_so_when_a_pair_had_to_be_substituted(merged):
    from gui.viz import metrics

    # D is selected but appears in no pair, so every D cell is NaN.
    figure = metrics.metric_dendrograms(
        {"dimension": 2, "algorithms": ["A", "B", "C", "D"]}
    )
    assert "maximally distant" in figure._suptitle.get_text(), (
        "a substituted distance must be visible on the figure, not silent"
    )


# -- 5. the theme pass must not erase deliberate colours ----------------
#
# figure_theme.apply_to recoloured every tick label unconditionally, while
# titles and figure texts were guarded by _is_default_colour.
# metrics.metric_dendrograms tints each dendrogram leaf by mealpy family and
# ships a legend explaining it, so the legend ended up describing colours no
# longer on the figure.

def test_apply_to_keeps_a_deliberately_coloured_tick_label():
    from gui.viz import figure_theme

    figure = Figure()
    axes = figure.add_subplot(1, 1, 1)
    axes.set_xticks([0, 1])
    axes.set_xticklabels(["kept", "themed"])
    deliberate, default = axes.get_xticklabels()
    deliberate.set_color("#ff7f0e")          # what family colouring does

    figure_theme.apply_to(figure)

    assert mcolors.to_rgba(deliberate.get_color()) == mcolors.to_rgba("#ff7f0e")
    # and the untouched one is still themed, or dark mode would break
    assert mcolors.to_rgba(default.get_color()) != mcolors.to_rgba("black")


# -- 6. an incomparable pair is explained, not handed to scipy ----------
#
# exploration.difference() returns NaN when two algorithms share no run of
# equal length - the diversity files in a tree do not all cover the same
# number of iterations. That NaN reached sns.clustermap, which answered
# "The condensed distance matrix must contain only finite values", shown
# verbatim in the error label.

def test_incomparable_algorithms_are_named_instead_of_reaching_scipy():
    from gui.viz import exploration

    matrix = pd.DataFrame(
        [[0.0, 1.0, np.nan],
         [1.0, 0.0, np.nan],
         [np.nan, np.nan, 0.0]],
        index=["A", "B", "Odd"], columns=["A", "B", "Odd"],
    )
    with pytest.raises(ValueError) as caught:
        exploration._require_comparable(matrix)

    message = str(caught.value)
    assert "Odd" in message, "the message must name the algorithm at fault"
    assert "condensed distance matrix" not in message
