"""Runs a benchmark sweep on a worker thread, reporting progress as it goes.

helper_functions.run_benchmarks is called in-process rather than as a
subprocess: that is what makes the progress_cb and cancel_event hooks usable.
The CLI's parallel paths are deliberately not used here - besides being
unobservable (their futures are never collected, so exceptions vanish), the
-p s branch hands every worker the full seed list and does N times the work.

Qt rule this file obeys: the worker thread only emits signals. Every widget
touch happens on the UI thread, in the slots connected to them.
"""
import sys
import threading
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gui.qt import QtCore, Signal
from gui.core import optimizers as optimizers_core
from gui.core import workspace as workspace_core


class BenchmarkRunner(QtCore.QThread):
    """Executes a RunSpec, writing into `out_dir`."""

    progressed = Signal(dict)   # one finished problem
    message = Signal(str)       # human-readable status line
    completed = Signal(dict)    # summary when the sweep ends
    failed = Signal(str)        # the sweep could not start or blew up

    def __init__(self, spec, out_dir, parent=None):
        super().__init__(parent)
        self.spec = spec
        self.out_dir = str(out_dir)
        self._cancel = threading.Event()
        self._counts = {"ok": 0, "failed": 0}

    def cancel(self):
        """Ask the sweep to stop at the next problem boundary."""
        self._cancel.set()
        self.message.emit("Cancelling after the current run...")

    @property
    def cancelled(self):
        return self._cancel.is_set()

    def run(self):
        import helper_functions as hf
        from gui.core import problems as problems_core

        started = time.time()
        try:
            registry = optimizers_core.load_registry()
            chosen = registry.resolve(self.spec.algorithms)
        except Exception as exc:
            self.failed.emit(str(exc))
            return

        def on_progress(record):
            self._counts["ok" if record.get("status") == "ok" else "failed"] += 1
            self.progressed.emit(record)

        try:
            source = problems_core.source_from_config(self.spec.source)
            # COCO wants an observer object; the adapted sources ignore it.
            observer = None
            if isinstance(source, problems_core.BBOBSource):
                import cocoex

                observer = cocoex.Observer("no_observer", "")
            suite = source.build(
                self.spec.functions, self.spec.instances, self.spec.dimensions
            )
            # An adapted source silently drops functions that cannot take a
            # requested dimension, so a selection can come back empty.
            if isinstance(suite, problems_core.AdaptedSuite) and len(suite) == 0:
                self.failed.emit(
                    "That selection produced no runnable problems - most "
                    "likely every chosen function is fixed at a different "
                    "dimension than the one selected."
                )
                return
            for seed in self.spec.seeds:
                if self._cancel.is_set():
                    break
                self.message.emit(f"Seed {seed}")
                hf.run_benchmarks(
                    suite,
                    observer,
                    chosen,
                    self.out_dir,
                    seed=seed,
                    epoch_per_dim=self.spec.epoch_per_dim,
                    pop_size=self.spec.pop_size,
                    only_best=self.spec.only_best,
                    save_diversity=self.spec.save_diversity,
                    progress_cb=on_progress,
                    cancel_event=self._cancel,
                )
        except Exception:
            self.failed.emit(traceback.format_exc(limit=3))
            return

        self.completed.emit({
            "ok": self._counts["ok"],
            "failed": self._counts["failed"],
            "expected": self.spec.total_runs,
            "cancelled": self._cancel.is_set(),
            "seconds": time.time() - started,
            "out_dir": self.out_dir,
        })


def finalise_run(run_dir, spec, summary):
    """Record the outcome in the run's manifest."""
    status = "cancelled" if summary.get("cancelled") else (
        "failed" if summary.get("ok", 0) == 0 else "completed"
    )
    workspace_core.write_manifest(
        run_dir,
        spec,
        status=status,
        completed_runs=summary.get("ok", 0),
        failed_runs=summary.get("failed", 0),
        duration_seconds=round(summary.get("seconds", 0.0), 2),
    )
    return status
