# Unit tests for gui.core.results_root: pointing the app at a different tree.
#
# This exists because config.py derives every data directory from REPO_ROOT at
# import time. That is right for the pipeline, which always runs inside the
# checkout, and wrong for a packaged app that has no checkout - where, before
# this, every view was permanently empty with no way to fix it.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import config
from gui.core import results_root


@pytest.fixture(autouse=True)
def restore_default():
    yield
    results_root.reset()


def test_subpaths_match_the_layout_config_itself_builds():
    """If config.py's layout ever changes, this override must change with it,
    or the app would quietly look in the wrong subdirectories."""
    default = Path(results_root.DEFAULT_ROOT)
    results_root.reset()
    for name, subpath in results_root.SUBPATHS.items():
        assert Path(getattr(config, name)) == default / subpath, name


def test_setting_a_root_repoints_every_directory(tmp_path):
    results_root.set_root(tmp_path)
    assert Path(config.OUTPUTS_DIR) == tmp_path / "outputs"
    assert Path(config.PROCESSED_DIR) == tmp_path / "data/processed"
    assert Path(config.MERGED_DIR) == tmp_path / "metrics_data/merged"
    assert Path(config.SCALARS_CSV) == tmp_path / "metrics_data/scalars.csv"
    assert Path(config.REPO_ROOT) == tmp_path


def test_reset_restores_the_original_tree():
    original = config.OUTPUTS_DIR
    results_root.set_root(Path.home())
    assert config.OUTPUTS_DIR != original
    results_root.reset()
    assert config.OUTPUTS_DIR == original
    assert results_root.is_default()


def test_inspect_reports_what_a_folder_actually_has(tmp_path):
    (tmp_path / "outputs").mkdir()
    (tmp_path / "data" / "entropy").mkdir(parents=True)

    summary = results_root.inspect(tmp_path)
    assert summary.is_usable
    assert "Raw trajectories" in summary.present
    assert "Entropy" in summary.present
    assert "Merged metrics" in summary.missing
    assert "Has:" in summary.describe() and "missing:" in summary.describe()


def test_an_empty_folder_is_reported_as_unusable(tmp_path):
    summary = results_root.inspect(tmp_path)
    assert not summary.is_usable
    assert summary.describe() == "No pipeline output found here."


def test_a_gui_run_directory_is_a_valid_results_root(tmp_path):
    # Each run dir contains outputs/, which is what makes the raw-trajectory
    # views work against a run the app just produced.
    (tmp_path / "outputs" / "dim_2" / "OriginalDE" / "1_1").mkdir(parents=True)
    summary = results_root.set_root(tmp_path)
    assert summary.is_usable
    assert summary.present == ("Raw trajectories",)


def test_moving_the_root_drops_cached_tables(tmp_path):
    """A cache surviving the move would show the previous tree's numbers
    under the new tree's name - worse than an error."""
    pytest.importorskip("seaborn")
    import pandas as pd

    from gui.viz import data as viz_data

    merged = tmp_path / "metrics_data" / "merged"
    merged.mkdir(parents=True)
    pd.DataFrame({
        "Algorithm1": ["A"], "Algorithm2": ["B"], "Function_id": [1],
        "Instance_id": [1], "Run_id": [1], "cosine": [0.5],
    }).to_csv(merged / "merged_dim_2.csv", index=False)

    results_root.set_root(tmp_path)
    assert len(viz_data.load_merged(2)) == 1

    other = tmp_path / "other"
    (other / "metrics_data" / "merged").mkdir(parents=True)
    pd.DataFrame({
        "Algorithm1": ["A", "A"], "Algorithm2": ["B", "C"],
        "Function_id": [1, 1], "Instance_id": [1, 1], "Run_id": [1, 1],
        "cosine": [0.5, 0.7],
    }).to_csv(other / "metrics_data" / "merged" / "merged_dim_2.csv", index=False)

    results_root.set_root(other)
    assert len(viz_data.load_merged(2)) == 2, "stale cache from the previous root"


def test_data_browser_follows_the_root(tmp_path, monkeypatch):
    from gui.core import datastore, workspace

    monkeypatch.setenv(workspace.WORKSPACE_ENV_VAR, str(tmp_path / "no_runs"))
    results_root.set_root(tmp_path)
    assert datastore.build_tree() == []

    (tmp_path / "outputs" / "dim_2").mkdir(parents=True)
    results_root.set_root(tmp_path)
    assert [node.label for node in datastore.build_tree()] == ["Benchmark runs"]
