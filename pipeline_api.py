"""The calling convention every pipeline stage exposes.

Each stage script in 01_optimize/ .. 05_analysis/ defines:

    def run(progress_cb=None, cancel_event=None, **stage_specific):
        ...
        return StageResult(...)

    if __name__ == "__main__":
        run()

That shape is what lets the desktop app drive the pipeline in-process, the
same way gui/core/runner.py already drives helper_functions.run_benchmarks
through its progress_cb/cancel_event pair. Running a stage as a subprocess
is not an option for the packaged app - the bundle ships no interpreter, so
sys.executable is the GUI itself.

Nothing here imports pandas, numpy or mealpy: this module is imported by
every stage and by the GUI, and must stay cheap.
"""
from dataclasses import dataclass, field


@dataclass
class StageResult:
    """What a stage did. Returned instead of exiting or printing a summary."""

    stage: str
    written: int = 0
    cancelled: bool = False
    notes: list = field(default_factory=list)

    def describe(self):
        state = "cancelled after" if self.cancelled else "wrote"
        parts = [f"{self.stage}: {state} {self.written:,} file(s)"]
        parts.extend(self.notes)
        return "  -  ".join(parts)


class Progress:
    """Per-item progress reporting and cooperative cancellation.

    A stage builds one of these per loop it wants to report, then calls
    item() at the head of the loop:

        progress = Progress(progress_cb, cancel_event)
        progress.begin(len(files), "clustering dim 2")
        for name in files:
            if not progress.item(name):
                break          # cancelled - return a partial StageResult

    item() returns False when the caller has asked to stop, so a stage never
    has to know what a cancel_event is. Both arguments are optional, so
    running a stage from the command line costs nothing.
    """

    def __init__(self, callback=None, cancel_event=None):
        self._callback = callback
        self._cancel_event = cancel_event
        self._done = 0
        self._total = 0
        self._prefix = ""

    @property
    def cancelled(self):
        return self._cancel_event is not None and self._cancel_event.is_set()

    @property
    def done(self):
        return self._done

    def begin(self, total, prefix=""):
        """Start a new counted loop of `total` items."""
        self._total = int(total)
        self._done = 0
        self._prefix = prefix
        self._emit(prefix)

    def item(self, label=""):
        """Announce the item about to be processed. False means stop."""
        if self.cancelled:
            return False
        self._done += 1
        self._emit(f"{self._prefix}  {label}".strip() if self._prefix else label)
        return True

    def note(self, label):
        """Report a status change without advancing the counter."""
        self._emit(label)

    def _emit(self, label):
        if self._callback is None:
            return
        self._callback(self._done, self._total, label)


def as_progress(progress_cb, cancel_event):
    """Convenience for stages that only have one loop worth reporting."""
    return Progress(progress_cb, cancel_event)
