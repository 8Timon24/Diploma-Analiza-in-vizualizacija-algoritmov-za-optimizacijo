# The Process tab. The tests that matter here are the ones about not
# destroying data by accident: the confirmation must be unskippable, and the
# caches the run invalidates must actually be dropped.
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

pytest.importorskip("PySide6")

from gui.qt import QtWidgets, Qt
from gui.core import pipeline as pipeline_core
from gui.panels.pipeline_panel import PipelinePanel, _bytes, _describe_size


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def panel(app):
    widget = PipelinePanel()
    yield widget
    widget.deleteLater()


# -- selection ------------------------------------------------------------

def test_the_benchmark_is_not_ticked_by_default(panel):
    """Opening the tab must not arm a multi-hour job."""
    assert "benchmark" not in panel.selected_stages()


def test_start_from_ticks_that_stage_and_everything_after(panel):
    panel._select_from("merge")
    expected = [s.name for s in pipeline_core.stages_from("merge")]
    assert panel.selected_stages() == expected


def test_start_from_keeps_the_combo_in_step(panel):
    """The combo said one thing while the ticks said another."""
    panel._select_from("clustering")
    assert panel.start_from.currentData() == "clustering"


def test_selected_stages_follow_the_tick_boxes(panel):
    panel._select_from("spearman")
    item = panel.stage_list.item(pipeline_core.STAGE_NAMES.index("merge"))
    item.setCheckState(Qt.CheckState.Checked)
    assert set(panel.selected_stages()) == {"merge", "spearman"}


def test_run_is_disabled_with_nothing_ticked(panel):
    for row in range(panel.stage_list.count()):
        panel.stage_list.item(row).setCheckState(Qt.CheckState.Unchecked)
    assert panel.run_button.isEnabled() is False


# -- the confirmation is the only thing between a click and 3.5 GB --------

def test_no_run_starts_without_confirmation(panel, monkeypatch):
    started = []
    monkeypatch.setattr(panel, "_confirm", lambda spec: False)
    monkeypatch.setattr(panel, "start", lambda spec: started.append(spec))

    panel._select_from("merge")
    panel._request_run()

    assert started == [], "a run started despite the confirmation being declined"


def test_confirming_starts_the_run(panel, monkeypatch):
    started = []
    monkeypatch.setattr(panel, "_confirm", lambda spec: True)
    monkeypatch.setattr(panel, "start", lambda spec: started.append(spec))

    panel._select_from("merge")
    panel._request_run()

    assert len(started) == 1
    assert started[0].stages == panel.selected_stages()


def test_the_confirmation_names_every_directory_at_risk(panel, monkeypatch):
    shown = {}

    def fake_question(_parent, _title, text, *args, **kwargs):
        shown["text"] = text
        return QtWidgets.QMessageBox.StandardButton.Cancel

    monkeypatch.setattr(QtWidgets.QMessageBox, "question", fake_question)
    panel._select_from("clustering")
    spec = panel.current_spec()
    panel._confirm(spec)

    for target in spec.targets():
        assert target in shown["text"], f"{target} was not named in the dialog"
    # and the two things that are easy to find out too late
    assert "cancelling" in shown["text"].lower()
    assert "stay open" in shown["text"].lower()


# -- finishing ------------------------------------------------------------

def test_finishing_drops_every_path_keyed_cache(panel, monkeypatch):
    """coverage.invalidate() alone clears one of four caches; the figures
    would otherwise render pre-run numbers."""
    called = []
    from gui.core import results_root

    monkeypatch.setattr(results_root, "invalidate_caches",
                        lambda: called.append("invalidated"))
    panel._finish()
    assert called == ["invalidated"]


def test_a_failed_run_still_releases_the_panel(panel, monkeypatch):
    from gui.core import results_root

    monkeypatch.setattr(results_root, "invalidate_caches", lambda: None)
    panel._on_failed("clustering exploded")

    assert panel.is_running is False
    assert panel.run_button.isEnabled() is True
    assert "exploded" in panel.log.toPlainText()


def test_a_cancelled_run_says_the_tree_is_now_mixed(panel, monkeypatch):
    from gui.core import results_root

    monkeypatch.setattr(results_root, "invalidate_caches", lambda: None)
    monkeypatch.setattr(pipeline_core, "write_manifest", lambda *a, **k: None)
    panel._spec = panel.current_spec()
    panel._on_completed({"stages": ["merge"], "written": 3,
                         "cancelled": True, "seconds": 2.0})

    log = panel.log.toPlainText().lower()
    assert "cancelled" in log
    assert "previous run" in log, "the mixed-state warning is missing"


# -- size reporting -------------------------------------------------------

def test_missing_targets_are_reported_as_such(tmp_path):
    assert "does not exist" in _describe_size(tmp_path / "nope")


def test_size_of_a_directory_counts_its_files(tmp_path):
    (tmp_path / "a.csv").write_text("x" * 2048)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.csv").write_text("y" * 1024)
    described = _describe_size(tmp_path)
    assert "2 files" in described


def test_bytes_are_human_readable():
    assert _bytes(512) == "512 B"
    assert _bytes(2048).endswith("KB")
    assert _bytes(5 * 1024 ** 3).endswith("GB")


# -- the "run these steps now" route from the Visualize tab ---------------

def test_select_stages_ticks_exactly_those(panel):
    panel.select_stages(["merge", "spearman"])
    assert panel.selected_stages() == ["merge", "spearman"]


def test_select_stages_points_the_combo_at_the_earliest(panel):
    panel.select_stages(["build_scalars", "clustering"])
    assert panel.start_from.currentData() == "clustering"


def test_select_stages_with_nothing_disables_run(panel):
    panel.select_stages([])
    assert panel.selected_stages() == []
    assert panel.run_button.isEnabled() is False
