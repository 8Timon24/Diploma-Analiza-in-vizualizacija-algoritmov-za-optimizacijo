# Regressions reported from running the packaged app.
#
# 1. "no data gets saved even if i run the benchmarks"
#    The data WAS saved - 169 files - but to the workspace, which nothing in
#    the data browser listed. In a packaged app the repo tree does not exist
#    at all, so the browser was simply empty and the run looked lost.
#
# 2. "if you generate multiple times pics get overlayed and everything is
#    mixed up"
#    The compare panel reused one canvas and assigned canvas.figure. That
#    leaves BOTH figures pointing at the same canvas, so a draw_idle() queued
#    by the previous render repaints the old figure over the new one.
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest

from gui.core import datastore, workspace
from gui.core.workspace import RunSpec, create_run_dir


def _make_run(root, algorithms=("A",), status="completed"):
    spec = RunSpec(algorithms=list(algorithms), functions=[1], instances=[1],
                   dimensions=[2], seeds=[1])
    run_dir = create_run_dir(spec, root=root)
    workspace.write_manifest(run_dir, spec, status=status, completed_runs=1,
                             failed_runs=0)
    traj = run_dir / "outputs" / "dim_2" / algorithms[0] / "1_1"
    traj.mkdir(parents=True)
    pd.DataFrame({"x1": [0.1], "x2": [0.2], "fitness": [3.0],
                  "iteration": [1.0]}).to_csv(
        traj / "gbest_trajectory_1.csv", index=False)
    return run_dir


# -- 1. runs must be visible in the browser -----------------------------

def test_workspace_runs_appear_in_the_data_browser(tmp_path, monkeypatch):
    monkeypatch.setenv(workspace.WORKSPACE_ENV_VAR, str(tmp_path))
    # Nothing from the repo tree exists - exactly the packaged-app situation.
    for attribute in ("OUTPUTS_DIR", "PROCESSED_DIR", "CLUSTERING_LATEST_DIR",
                      "ENTROPY_DATA_DIR", "METRICS_DIR", "MERGED_DIR",
                      "CLUSTERING_DBSCAN_DIR"):
        monkeypatch.setattr(datastore.config, attribute, str(tmp_path / "absent"))
    monkeypatch.setattr(datastore.config, "SCALARS_CSV", str(tmp_path / "absent.csv"))

    _make_run(tmp_path)
    roots = [node.label for node in datastore.build_tree()]
    assert roots == ["GUI runs"], f"a packaged app would show nothing: {roots}"


def test_gui_runs_root_is_listed_first(tmp_path, monkeypatch):
    # It is where the user's newly created data is; it should not be buried.
    monkeypatch.setenv(workspace.WORKSPACE_ENV_VAR, str(tmp_path))
    _make_run(tmp_path)
    assert datastore.build_tree()[0].label == "GUI runs"


def test_a_run_expands_to_its_manifest_and_trajectories(tmp_path, monkeypatch):
    monkeypatch.setenv(workspace.WORKSPACE_ENV_VAR, str(tmp_path))
    _make_run(tmp_path, algorithms=("OriginalDE",))

    gui_runs = datastore.build_tree()[0]
    runs = gui_runs.children()
    assert len(runs) == 1
    contents = {node.label for node in runs[0].children()}
    assert "manifest.json" in contents
    assert "dim_2" in contents


def test_run_label_summarises_the_outcome(tmp_path, monkeypatch):
    monkeypatch.setenv(workspace.WORKSPACE_ENV_VAR, str(tmp_path))
    _make_run(tmp_path, algorithms=("A", "B"), status="cancelled")
    label = datastore.build_tree()[0].children()[0].label
    assert "cancelled" in label
    assert "2 algorithms" in label


def test_no_gui_runs_root_when_the_workspace_is_empty(tmp_path, monkeypatch):
    monkeypatch.setenv(workspace.WORKSPACE_ENV_VAR, str(tmp_path / "empty"))
    assert all(node.label != "GUI runs" for node in datastore.build_tree())


def test_manifest_json_is_readable_as_a_table(tmp_path, monkeypatch):
    monkeypatch.setenv(workspace.WORKSPACE_ENV_VAR, str(tmp_path))
    run_dir = _make_run(tmp_path, algorithms=("OriginalDE", "SADE"))

    table = datastore.load_table(run_dir / "manifest.json")
    assert table.fmt == "json"
    fields = dict(zip(table.frame["field"], table.frame["value"]))
    assert fields["status"] == "completed"
    # list-valued fields must be flattened, not dropped
    assert "OriginalDE" in str(fields["spec.algorithms"])


def test_frozen_apps_write_outside_the_bundle(monkeypatch):
    # The bundle directory is not a sensible - or necessarily writable -
    # place for results.
    monkeypatch.delenv(workspace.WORKSPACE_ENV_VAR, raising=False)
    monkeypatch.setattr(workspace, "is_frozen", lambda: True)
    root = workspace.workspace_root()
    assert Path.home() in root.parents or root.parent == Path.home()


# -- 2. repeated renders must not share a canvas ------------------------

def test_compare_panel_builds_a_new_canvas_per_render():
    """Pinned as source: assigning canvas.figure is what caused the overlay,
    and it is an easy habit to reintroduce."""
    source = Path(__file__).resolve().parents[1] / "gui" / "panels" / "compare_panel.py"
    text = source.read_text()
    assert "canvas.figure = figure" not in text, (
        "swapping a figure into an existing canvas leaves both figures bound "
        "to it; build a fresh FigureCanvasQTAgg instead"
    )
    assert "FigureCanvasQTAgg(figure)" in text
