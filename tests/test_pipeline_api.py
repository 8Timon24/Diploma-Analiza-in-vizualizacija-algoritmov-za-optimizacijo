# Unit tests for pipeline_api: the progress/cancel contract every stage
# script implements so the GUI can drive it in-process.
import sys
import threading
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline_api import Progress, StageResult


def test_progress_without_a_callback_is_a_no_op():
    """Running a stage from the command line passes neither argument."""
    progress = Progress()
    progress.begin(3, "x")
    assert progress.item("a") is True
    assert progress.done == 1


def test_item_reports_done_total_and_label():
    seen = []
    progress = Progress(lambda done, total, label: seen.append((done, total, label)))
    progress.begin(2, "clustering")
    progress.item("F1_I1")
    progress.item("F1_I2")
    # begin() emits a 0-of-N tick so a bar can size itself before the first item
    assert seen[0] == (0, 2, "clustering")
    assert seen[1] == (1, 2, "clustering  F1_I1")
    assert seen[2] == (2, 2, "clustering  F1_I2")


def test_item_returns_false_once_cancelled():
    event = threading.Event()
    progress = Progress(cancel_event=event)
    progress.begin(10)
    assert progress.item("a") is True
    event.set()
    assert progress.item("b") is False
    # the cancelled item is not counted as done
    assert progress.done == 1


def test_cancelled_reflects_the_event():
    event = threading.Event()
    progress = Progress(cancel_event=event)
    assert progress.cancelled is False
    event.set()
    assert progress.cancelled is True


def test_note_does_not_advance_the_counter():
    seen = []
    progress = Progress(lambda done, total, label: seen.append((done, label)))
    progress.begin(5)
    progress.item("a")
    progress.note("reading results.csv")
    assert progress.done == 1
    assert seen[-1] == (1, "reading results.csv")


def test_stage_result_describes_itself():
    assert "wrote 3 file(s)" in StageResult("entropy", written=3).describe()
    assert "cancelled after" in StageResult("entropy", 3, cancelled=True).describe()
    assert "note" in StageResult("x", notes=["note"]).describe()
