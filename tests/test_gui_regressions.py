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


# -- 3. the empty state must not crash the table ------------------------
#
# "the Data tab dies when a file will not open"
#   DataFrameModel.__init__ guarded frame=None but set_frame() did not, and
#   data_panel passes None on exactly two paths: nothing found under the
#   results root, and a file that could not be read. rowCount() then did
#   len(None) inside endResetModel().

def test_set_frame_accepts_none():
    pytest.importorskip("PySide6")
    from gui.models.dataframe_model import DataFrameModel

    model = DataFrameModel(pd.DataFrame({"a": [1, 2]}))
    model.set_frame(None)
    assert model.rowCount() == 0
    assert model.columnCount() == 0
    assert model.frame.empty


def test_sort_ignores_a_column_index_from_a_previous_file():
    """The view keeps its sort indicator across files, so a narrower frame
    can be asked to sort by a column that no longer exists."""
    pytest.importorskip("PySide6")
    from gui.qt import Qt
    from gui.models.dataframe_model import DataFrameModel

    model = DataFrameModel(pd.DataFrame({"a": [3, 1], "b": [1, 2], "c": [0, 0]}))
    model.set_frame(pd.DataFrame({"a": [3, 1]}))
    model.sort(2, Qt.SortOrder.AscendingOrder)  # column 2 is gone
    assert list(model.frame["a"]) == [3, 1]


def test_sorting_is_not_cumulative_and_keeps_file_order_recoverable():
    pytest.importorskip("PySide6")
    from gui.qt import Qt
    from gui.models.dataframe_model import DataFrameModel

    model = DataFrameModel(pd.DataFrame({"a": [2, 3, 1]}))
    model.sort(0, Qt.SortOrder.AscendingOrder)
    assert list(model.frame["a"]) == [1, 2, 3]
    model.sort(0, Qt.SortOrder.DescendingOrder)
    # Sorted from the frame as loaded, not from the already-sorted copy.
    assert list(model.frame["a"]) == [3, 2, 1]


def test_missing_values_of_every_dtype_render_the_same():
    pytest.importorskip("PySide6")
    import numpy as np
    from gui.qt import Qt
    from gui.models.dataframe_model import DataFrameModel

    frame = pd.DataFrame({
        "f64": pd.Series([np.nan], dtype="float64"),
        "f32": pd.Series([np.nan], dtype="float32"),
        "time": pd.Series([pd.NaT], dtype="datetime64[ns]"),
    })
    model = DataFrameModel(frame)
    shown = [
        model.data(model.index(0, column), Qt.ItemDataRole.DisplayRole)
        for column in range(3)
    ]
    assert shown == ["", "", ""], shown


# -- 4. config.FUNCTIONS is overridable, so nothing may index it blindly --

def test_trajectory_panel_builds_with_a_trimmed_function_list(monkeypatch):
    """PIPELINE_TEST_FUNCTIONS made config.FUNCTIONS = [1, 2]; a bare
    .index(16) then raised ValueError inside MainWindow.__init__ and the whole
    window failed to construct."""
    pytest.importorskip("PySide6")
    from gui.qt import QtWidgets
    import config as config_module
    from gui.panels import trajectory_panel

    monkeypatch.setattr(config_module, "FUNCTIONS", [1, 2])
    monkeypatch.setattr(config_module, "INSTANCES", [1])
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    panel = trajectory_panel.TrajectoryPanel()
    assert panel.function.currentData() == 1


# -- 5. a failed run must release the panel the same way a finished one does

def test_run_panel_reports_a_failed_start():
    """_on_failed used to skip runFinished and cache invalidation, so the
    Data tab never refreshed after a run that died on startup.

    invalidate_caches() rather than coverage.invalidate(): the latter clears
    one lru_cache of four, leaving the figure views rendering pre-run numbers.
    """
    source = Path(__file__).resolve().parents[1] / "gui" / "panels" / "run_panel.py"
    body = source.read_text().split("def _on_failed")[1].split("\n    def ")[0]
    assert "runFinished.emit" in body
    assert "results_root.invalidate_caches()" in body


# -- 6. every trajectory redraw must be restyled, not just the first one ---
#
# "when you generate a trajectory image the text at the top turns black"
#   figure_theme.apply_to() was only ever called once, at TrajectoryPanel
#   construction, before any problem was loaded. Loading one calls
#   _draw_background(), which does axes.clear() - that recreates the title
#   (and the legend, spines, facecolor...) as fresh matplotlib-default
#   artists, black text included, and nothing restyled them again afterward.

def test_trajectory_title_is_restyled_on_every_frame(monkeypatch):
    pytest.importorskip("PySide6")
    from gui.qt import QtWidgets
    from gui import theme
    from gui.panels import trajectory_panel

    monkeypatch.setattr(theme, "_current", theme._DARK)
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    panel = trajectory_panel.TrajectoryPanel()

    # axes.clear() is what _draw_background() does on every load - it is
    # the actual point where matplotlib resets the title to its own
    # black-by-default styling, not set_title() itself.
    panel.axes.clear()
    panel._selection = pd.DataFrame({
        "iteration": [1], "x0": [0.0], "x1": [0.0], "algorithm": ["A"],
    })
    panel._iterations = (1, 1)
    panel._function, panel._instance, panel._run = 1, 1, 1
    panel._trail = panel.axes.scatter([], [])
    panel._cluster_scatter = None
    panel._scatters = {}

    panel._update_frame(1)

    import matplotlib.colors as mcolors

    assert panel.axes.title.get_color() != "black"
    assert mcolors.to_rgba(panel.axes.title.get_color()) == mcolors.to_rgba(
        theme.tokens()["text"]
    )


# -- 7. every panel the main window can report as busy needs a job_noun ----
#
# closeEvent() builds "A {panel.job_noun} is still in progress." for whichever
# panel _busy_panel() returns. PipelinePanel extends JobPanel, which defines
# job_noun; RunPanel was a bare QWidget that re-implemented the whole JobPanel
# lifecycle by hand, and that copy had drifted. Closing the window during a
# benchmark therefore raised AttributeError INSIDE closeEvent: the user never
# saw the "quit anyway?" prompt, the event was neither accepted nor ignored,
# and Qt destroyed the panel with a live BenchmarkRunner child - the exact
# "QThread: Destroyed while thread is still running" abort JobPanel exists to
# prevent.

def test_every_busy_panel_can_be_named_in_the_quit_prompt():
    pytest.importorskip("PySide6")
    from gui.qt import QtWidgets
    from gui.panels.job_panel import JobPanel
    from gui.panels.pipeline_panel import PipelinePanel
    from gui.panels.run_panel import RunPanel

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    # Mirrors MainWindow._busy_panel, so the two cannot drift apart again.
    for panel_class in (RunPanel, PipelinePanel):
        assert issubclass(panel_class, JobPanel), (
            f"{panel_class.__name__} must inherit JobPanel rather than copy it"
        )
        panel = panel_class()
        assert isinstance(panel.job_noun, str) and panel.job_noun
        # what closeEvent actually does with it
        assert f"A {panel.job_noun} is still in progress."
        for name in ("is_running", "request_cancel", "wait_for_exit"):
            assert hasattr(panel, name)


# -- 8. a failed Load must not leave artists the next rebuild will remove --
#
# _fail() called axes.clear() but kept _selection/_scatters/_trail. clear()
# leaves each artist's _remove_method None, so the next _rebuild_artists()
# raised NotImplementedError("cannot remove artist") straight out of the
# colour-mode slot. _selection surviving also meant _update_frame kept
# repainting - and retitling - the problem that had just failed to load.

def test_failed_load_then_colour_mode_change_does_not_raise(monkeypatch):
    pytest.importorskip("PySide6")
    from gui.qt import QtWidgets
    from gui.panels import trajectory_panel

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    panel = trajectory_panel.TrajectoryPanel()

    # Stand in for a successful load: artists on the axes, a selection set.
    panel._selection = pd.DataFrame({
        "iteration": [1], "x0": [0.0], "x1": [0.0], "algorithm": ["A"],
    })
    panel._iterations = (1, 1)
    panel._function, panel._instance, panel._run = 1, 1, 1
    panel._scatters = {"A": panel.axes.scatter([], [])}
    panel._trail = panel.axes.scatter([], [])

    panel._fail("no data for this problem")

    assert panel._selection is None, "a failed load must forget the old problem"
    assert panel._scatters == {} and panel._trail is None

    panel._rebuild_artists()          # what the colour-mode slot calls
    assert "Could not load" in panel.axes.get_title()


# -- 9. a render that raises must not leak a figure into pyplot -----------
#
# offscreen_figures only detached the figures that reached show/savefig/close.
# A plot function that created a figure and THEN raised never entered
# `captured`, so its figure stayed in pyplot's Gcf forever - five failed
# renders leaked five figures, and every error path feeds this.

def test_render_with_leaks_nothing_when_the_plot_raises():
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from gui.viz.capture import render_with

    before = set(plt.get_fignums())

    def explodes():
        plt.figure()
        raise ValueError("boom")

    for _ in range(5):
        with pytest.raises(ValueError):
            render_with(explodes)

    assert set(plt.get_fignums()) == before


def test_capture_only_intercepts_its_own_thread(tmp_path):
    """The Process tab runs stages on a worker thread, and spearman.py writes
    a real PDF through Figure.savefig. A patch that suppressed unconditionally
    turned that write into a silent no-op whenever the user rendered a figure
    at the same time - and the stage still reported success."""
    pytest.importorskip("matplotlib")
    import threading
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from gui.viz.capture import offscreen_figures

    inside, done, result = threading.Event(), threading.Event(), {}

    def worker():
        inside.wait(10)                      # save while the patch is held
        figure = Figure()
        figure.add_subplot(1, 1, 1).plot([0, 1], [1, 0])
        target = tmp_path / "stage.pdf"
        figure.savefig(target)
        result["worker_wrote"] = target.exists() and target.stat().st_size > 0
        done.set()

    thread = threading.Thread(target=worker)
    thread.start()
    with offscreen_figures():
        inside.set()
        done.wait(10)
        own = tmp_path / "ui.png"
        figure = Figure()
        figure.add_subplot(1, 1, 1).plot([0, 1], [0, 1])
        figure.savefig(own)
        result["own_suppressed"] = not own.exists()
    thread.join(10)

    assert result.get("worker_wrote") is True, (
        "another thread's savefig was silently swallowed"
    )
    assert result.get("own_suppressed") is True, (
        "the rendering thread's savefig must still be suppressed"
    )


def test_nested_capture_does_not_hand_savefig_back_early(tmp_path):
    """render_with nests offscreen_figures; an inner context exiting must not
    let the outer one start writing to disk."""
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from gui.viz.capture import offscreen_figures

    target = tmp_path / "must_not_appear.png"
    with offscreen_figures():
        with offscreen_figures():
            pass
        figure = Figure()
        figure.add_subplot(1, 1, 1).plot([0, 1], [0, 1])
        figure.savefig(target)

    assert not target.exists()


# -- 10. editing the form must mark the rendered figure stale -------------
#
# _build_form connected nothing to _mark_stale, so only changing the CATALOG
# selection marked the figure stale. Changing a parameter after a render left
# Export enabled on a figure built from different parameters - exactly what
# _mark_stale exists to prevent. And _clear_figure_area unparents self.status
# without _show_figure putting it back, so the message it set could never be
# seen.

def test_changing_a_parameter_disables_export_and_says_why():
    pytest.importorskip("PySide6")
    pytest.importorskip("seaborn")
    from gui.qt import QtWidgets
    from gui.panels.viz_panel import VizPanel

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    panel = VizPanel()
    panel.activate()

    for row in range(panel.list.count()):
        if "Entropy over" in panel.list.item(row).text():
            panel.list.setCurrentRow(row)
            break
    panel._render()
    if panel._figure is None:
        pytest.skip("no entropy data in this tree to render")
    assert panel.export_button.isEnabled()

    combo = next(w for _p, w in panel._inputs.values()
                 if isinstance(w, QtWidgets.QComboBox))
    combo.setCurrentIndex((combo.currentIndex() + 1) % combo.count())

    assert not panel.export_button.isEnabled(), (
        "Export stayed enabled on a figure that no longer matches the form"
    )
    assert "press Render" in panel.status.text()
    assert panel.status.parent() is not None, (
        "the stale message was written to an unparented, invisible widget"
    )
