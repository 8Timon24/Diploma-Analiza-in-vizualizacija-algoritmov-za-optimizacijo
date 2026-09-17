# Unit tests for gui.viz.data: the loaders behind the ported notebook figures.
#
# The naming split is the trap worth pinning: the raw outputs/ tree names run
# directories "{f}_{i}" while everything from data/processed onward uses
# "F{f}_I{i}". Getting that backwards produces a "file not found" for data
# that is sitting right there.
#
# Synthetic fixtures only; no Qt, no cocoex, no real data.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest

from gui.viz import data as viz_data

METRICS = ["entropy", "cosine", "cosine_columns", "exploration", "location", "fitness"]


# -- naming -------------------------------------------------------------

def test_processed_stage_uses_the_F_and_I_prefix():
    assert viz_data.problem_key(10, 3) == "F10_I3"


def test_raw_run_directory_uses_bare_integers(tmp_path):
    # outputs/dim_2/OriginalDE/10_1 - no F/I prefix here.
    path = viz_data.run_dir(2, "OriginalDE", 10, 1, outputs_dir=tmp_path)
    assert path.name == "10_1"
    assert path.parent.name == "OriginalDE"
    assert path.parent.parent.name == "dim_2"


# -- diversity ----------------------------------------------------------

def _write_diversity(root, dimension, algorithm, function, instance, seed, rows=3):
    directory = viz_data.run_dir(dimension, algorithm, function, instance, root)
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "iteration": range(1, rows + 1),
        "diversity": [1.0] * rows,
        "exploration": [100.0, 50.0, 10.0][:rows],
        "exploitation": [0.0, 50.0, 90.0][:rows],
    }).to_csv(directory / f"diversity_{seed}.csv", index=False)


def test_diversity_curves_are_tagged_with_algorithm_and_run(tmp_path):
    _write_diversity(tmp_path, 2, "A", 1, 1, 1)
    _write_diversity(tmp_path, 2, "B", 1, 1, 1)
    frame = viz_data.load_diversity_curves(2, ["A", "B"], 1, 1, [1], outputs_dir=tmp_path)
    assert set(frame["algorithm"]) == {"A", "B"}
    assert set(frame["run"]) == {1}


def test_missing_diversity_files_are_skipped_not_fatal(tmp_path):
    # Diversity is only written when the benchmark ran with -e, so a partial
    # set is a normal state rather than an error.
    _write_diversity(tmp_path, 2, "A", 1, 1, 1)
    frame = viz_data.load_diversity_curves(
        2, ["A", "Missing"], 1, 1, [1, 2], outputs_dir=tmp_path
    )
    assert set(frame["algorithm"]) == {"A"}


def test_no_diversity_at_all_gives_an_empty_frame_with_columns(tmp_path):
    frame = viz_data.load_diversity_curves(2, ["A"], 1, 1, [1], outputs_dir=tmp_path)
    assert frame.empty
    assert "exploration" in frame.columns


# -- merged metrics -----------------------------------------------------

@pytest.fixture
def merged(tmp_path, monkeypatch):
    rows = []
    for first, second in (("A", "B"), ("A", "C"), ("B", "C")):
        for function in (1, 23):
            rows.append({
                "Algorithm1": first, "Algorithm2": second,
                "Function_id": function, "Instance_id": 1, "Run_id": 1,
                **{m: 0.5 for m in METRICS},
            })
    pd.DataFrame(rows).to_csv(tmp_path / "merged_dim_2.csv", index=False)
    monkeypatch.setattr(viz_data.config, "MERGED_DIR", str(tmp_path))
    viz_data.load_merged.cache_clear()
    yield tmp_path
    viz_data.load_merged.cache_clear()


def test_pair_means_averages_over_problems(merged):
    means = viz_data.pair_means(2, METRICS)
    assert len(means) == 3          # three pairs, not six rows
    assert set(means["Algorithm1"]) == {"A", "B"}


def test_pair_means_keeps_only_pairs_where_both_are_selected(merged):
    means = viz_data.pair_means(2, METRICS, ["A", "B"])
    assert len(means) == 1
    assert means.iloc[0]["Algorithm1"] == "A"
    assert means.iloc[0]["Algorithm2"] == "B"


def test_pair_means_reports_which_metrics_exist_when_none_match(merged):
    with pytest.raises(ValueError, match="It has:"):
        viz_data.pair_means(2, ["not_a_metric"])


def test_selecting_an_absent_algorithm_is_an_explicit_error(merged):
    with pytest.raises(ValueError, match="No pairs"):
        viz_data.pair_means(2, METRICS, ["NotPresent"])


def test_missing_merged_file_names_the_script_that_builds_it(tmp_path, monkeypatch):
    monkeypatch.setattr(viz_data.config, "MERGED_DIR", str(tmp_path / "nothing"))
    viz_data.load_merged.cache_clear()
    with pytest.raises(ValueError, match="merge_metrics.py"):
        viz_data.load_merged(2)
    viz_data.load_merged.cache_clear()


# -- distance matrices --------------------------------------------------

def test_distance_matrix_is_symmetric_with_a_zero_diagonal():
    pytest.importorskip("seaborn")
    pytest.importorskip("scipy")
    from gui.viz.metrics import _distance_matrix

    means = pd.DataFrame([
        {"Algorithm1": "A", "Algorithm2": "B", "entropy": 0.4},
        {"Algorithm1": "A", "Algorithm2": "C", "entropy": 0.7},
        {"Algorithm1": "B", "Algorithm2": "C", "entropy": 0.1},
    ])
    matrix = _distance_matrix(means, "entropy", ["A", "B", "C"])
    assert matrix.loc["A", "B"] == matrix.loc["B", "A"] == 0.4
    assert matrix.loc["A", "A"] == 0.0


# -- animated trajectory player ----------------------------------------
#
# The player is a widget rather than a catalog entry, but its data layer is
# ordinary and worth pinning: it must prefer clustered trajectories (so
# points can be coloured by the cluster every downstream metric is built on)
# and fall back to the processed form when clustering has not been run.

def _write_trajectory_csv(root, dimension, function, instance, algorithms, runs, iterations):
    directory = root / f"dim_{dimension}"
    directory.mkdir(parents=True, exist_ok=True)
    rows = [
        {"x0": 0.1 * i, "x1": -0.1 * i, "raw_y": float(i),
         "iteration": float(i), "algorithm": a, "run": r}
        for a in algorithms for r in runs for i in range(1, iterations + 1)
    ]
    pd.DataFrame(rows).to_csv(
        directory / f"F{function}_I{instance}.csv", compression="zip"
    )


def test_player_falls_back_to_processed_when_clustering_is_absent(tmp_path, monkeypatch):
    from gui.viz import trajectory

    monkeypatch.setattr(trajectory.config, "PROCESSED_DIR", str(tmp_path))
    monkeypatch.setattr(
        trajectory.config, "CLUSTERING_LATEST_DIR", str(tmp_path / "no_clustering")
    )
    _write_trajectory_csv(tmp_path, 2, 1, 1, ["A"], [1], 3)

    frame, has_clusters = trajectory.load_positions(2, 1, 1)
    assert not has_clusters          # so the UI disables cluster colouring
    assert len(frame) == 3


def test_player_reports_where_to_get_data_when_there_is_none(tmp_path, monkeypatch):
    from gui.viz import trajectory

    monkeypatch.setattr(trajectory.config, "PROCESSED_DIR", str(tmp_path / "nope"))
    monkeypatch.setattr(
        trajectory.config, "CLUSTERING_LATEST_DIR", str(tmp_path / "nope")
    )
    with pytest.raises(ValueError, match="preprocess_data.py"):
        trajectory.load_positions(2, 1, 1)


def test_selecting_a_run_that_was_never_computed_is_an_explicit_error(tmp_path, monkeypatch):
    from gui.viz import trajectory

    monkeypatch.setattr(trajectory.config, "PROCESSED_DIR", str(tmp_path))
    monkeypatch.setattr(
        trajectory.config, "CLUSTERING_LATEST_DIR", str(tmp_path / "no_clustering")
    )
    _write_trajectory_csv(tmp_path, 2, 1, 1, ["A"], [1], 3)
    frame, _ = trajectory.load_positions(2, 1, 1)
    with pytest.raises(ValueError, match="No trajectory rows"):
        trajectory.select(frame, ["A"], run=99)


def test_iteration_range_drives_the_slider_bounds(tmp_path, monkeypatch):
    from gui.viz import trajectory

    monkeypatch.setattr(trajectory.config, "PROCESSED_DIR", str(tmp_path))
    monkeypatch.setattr(
        trajectory.config, "CLUSTERING_LATEST_DIR", str(tmp_path / "no_clustering")
    )
    _write_trajectory_csv(tmp_path, 2, 1, 1, ["A", "B"], [1], 7)
    frame, _ = trajectory.load_positions(2, 1, 1)
    selection = trajectory.select(frame, ["A", "B"], 1)
    assert trajectory.iteration_range(selection) == (1, 7)
    assert len(selection) == 14


def test_a_non_2d_trajectory_is_refused_rather_than_projected(tmp_path, monkeypatch):
    # Dropping x2..xd would imply a closeness in the plane that is not real.
    from gui.viz import trajectory

    frame = pd.DataFrame({
        "x0": [0.0], "iteration": [1.0], "algorithm": ["A"], "run": [1],
    })
    with pytest.raises(ValueError, match="2-D only"):
        trajectory.select(frame, ["A"], 1)
