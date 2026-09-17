"""What a panel that owns a background worker needs, regardless of the job.

Two panels run long jobs on a QThread - the benchmark sweep and the analysis
pipeline - and they share the same three obligations, none of which are
obvious enough to be worth getting wrong twice:

  * a finished worker must be deleteLater()'d, not merely dereferenced. The
    thread is parented to the panel, so dropping the reference alone leaves
    every finished QThread attached to it for the life of the app.
  * the main window must be able to ask a panel to stop and then wait for it,
    without reaching into private attributes.
  * closing the window mid-job must not destroy a running QThread, which Qt
    turns into "QThread: Destroyed while thread is still running" and an abort.

The panels keep their own layouts and their own signal wiring; only this
lifecycle is shared.
"""
from gui.qt import QtWidgets


class JobPanel(QtWidgets.QWidget):
    """Base for a panel that runs one worker thread at a time."""

    #: what the main window calls this job when it has to talk about it
    job_noun = "job"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._runner = None

    # -- lifecycle -------------------------------------------------------

    @property
    def is_running(self):
        return self._runner is not None and self._runner.isRunning()

    def _adopt_runner(self, runner):
        self._runner = runner
        return runner

    def _release_runner(self):
        """Drop the finished worker and let Qt destroy its C++ side."""
        runner, self._runner = self._runner, None
        if runner is not None:
            runner.deleteLater()

    # -- public API for the main window ----------------------------------

    def request_cancel(self):
        """Ask the running job to stop at its next safe boundary."""
        if self._runner is not None:
            self._runner.cancel()

    def wait_for_exit(self, milliseconds):
        """Block until the worker has actually finished. False on timeout."""
        if self._runner is None:
            return True
        return self._runner.wait(milliseconds)

    # -- logging ---------------------------------------------------------

    def _append(self, text):
        self.log.appendPlainText(text)
