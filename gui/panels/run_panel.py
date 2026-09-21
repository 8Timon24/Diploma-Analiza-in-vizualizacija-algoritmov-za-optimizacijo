"""Watch a benchmark sweep: progress, live convergence, log, cancel.

The convergence plot deliberately shows one problem at a time. Overlaying
curves from different BBOB functions would be meaningless - their fitness
scales differ by orders of magnitude - so the view follows whichever
(function, instance, dimension) is being worked on.

Curves are kept per problem rather than cleared when the problem changes.
run_benchmarks loops algorithms on the OUTSIDE and problems on the inside, so
a given problem is revisited once per algorithm, far apart in time; clearing
on change would mean never showing more than one algorithm at a time, which
defeats the point of the plot.

Figures here are built directly from matplotlib's object API, never pyplot:
pyplot keeps global state that is not thread-safe and would fight the
rendering done in the visualization panel.
"""
import time
from pathlib import Path

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from gui import theme
from gui.panels import layout as panel_layout
from gui.panels.job_panel import JobPanel
from gui.qt import QtWidgets, Qt, Signal
from gui.core import results_root
from gui.core.runner import BenchmarkRunner, finalise_run
from gui.core.workspace import format_duration, write_manifest
from gui.viz import figure_theme, style

# Redrawing on every finished run is wasteful once runs are fast; the curve
# is still correct, just refreshed at most this often.
REDRAW_INTERVAL_SECONDS = 0.25

# Curves are retained per problem so algorithms can be compared even though
# the sweep revisits each problem once per algorithm. Older problems are
# dropped to keep a long sweep bounded.
MAX_TRACKED_PROBLEMS = 40


class RunPanel(JobPanel):
    runFinished = Signal(dict)

    #: what the main window calls this job when it has to talk about it
    job_noun = "benchmark run"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._spec = None
        self._run_dir = None
        self._started = None
        self._done = 0
        self._current_problem = None
        self._curves_by_problem = {}
        self._last_redraw = 0.0

        # -- header
        self.status = QtWidgets.QLabel("No run in progress.")
        self.status.setProperty("class", "metric")

        self.progress = QtWidgets.QProgressBar()
        self.progress.setTextVisible(True)
        self.timing = QtWidgets.QLabel()
        self.timing.setProperty("class", "hint")

        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel)

        header = QtWidgets.QGridLayout()
        header.addWidget(self.status, 0, 0)
        header.addWidget(self.cancel_button, 0, 1)
        header.addWidget(self.progress, 1, 0)
        header.addWidget(self.timing, 2, 0, 1, 2)

        # -- live convergence
        self.figure = Figure(figsize=(6, 4), layout="constrained")
        self.axes = self.figure.add_subplot(1, 1, 1)
        self._reset_axes()
        figure_theme.apply_to(self.figure)
        self.canvas = FigureCanvasQTAgg(self.figure)

        # -- log
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        self.log.setFont(theme.monospace_font(self.log))

        splitter = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
        left = panel_layout.group_box("Live convergence")
        QtWidgets.QVBoxLayout(left).addWidget(self.canvas)
        right = panel_layout.group_box("Log")
        QtWidgets.QVBoxLayout(right).addWidget(self.log)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setChildrenCollapsible(False)

        layout = panel_layout.column(self)
        layout.addLayout(header)
        layout.addWidget(splitter, 1)

    # -- lifecycle -------------------------------------------------------
    # is_running / _release_runner / wait_for_exit / _append come from
    # JobPanel. They used to be copied into this file verbatim, and that
    # copy is what let job_noun go missing: the main window asks every busy
    # panel for it while building the "quit anyway?" prompt.

    def start(self, spec, run_dir, out_dir):
        if self.is_running:
            return False

        self._spec = spec
        self._run_dir = Path(run_dir) if run_dir else None
        self._started = time.time()
        self._done = 0
        self._current_problem = None
        self._curves_by_problem = {}
        self._reset_axes()
        self.canvas.draw_idle()

        self.log.clear()
        self._append(f"Output: {out_dir}")
        self._append(f"{spec.describe()}  ->  {spec.total_runs:,} runs")
        self.progress.setRange(0, spec.total_runs)
        self.progress.setValue(0)
        self.status.setText("Starting...")
        self.cancel_button.setEnabled(True)

        runner = self._adopt_runner(BenchmarkRunner(spec, out_dir, parent=self))
        runner.progressed.connect(self._on_progress)
        runner.message.connect(self._append)
        runner.completed.connect(self._on_completed)
        runner.failed.connect(self._on_failed)
        runner.start()
        return True

    def _cancel(self):
        if self._runner is not None:
            self.cancel_button.setEnabled(False)
            # Cancellation only takes effect at the next run boundary, which
            # can be minutes away. Saying so in the header matters: the status
            # line otherwise keeps counting up as if nothing was asked.
            self.status.setText("Cancelling - stopping after the current run...")
            self._runner.cancel()

    # -- progress --------------------------------------------------------

    def _on_progress(self, record):
        self._done += 1
        self.progress.setValue(self._done)

        if record.get("status") == "ok":
            self._append(
                f"  [{record['algorithm']}] {record['problem']} "
                f"| f*={record['fitness']:.4e} | evals={record['evaluations']}"
            )
            self._track_curve(record)
        else:
            self._append(
                f"  [{record['algorithm']}] {record.get('problem', '?')} "
                f"| FAILED: {record.get('error', '')}"
            )

        elapsed = time.time() - self._started
        total = self.progress.maximum()
        self.status.setText(f"Running  -  {self._done:,} of {total:,}")
        if self._done:
            remaining = (elapsed / self._done) * max(total - self._done, 0)
            left = "almost done" if remaining < 1 else f"about {format_duration(remaining)} left"
            self.timing.setText(
                f"elapsed {format_duration(elapsed)}  -  {left}  -  "
                f"{elapsed / self._done:.2f}s per run"
            )

    def _track_curve(self, record):
        curve = record.get("curve") or []
        if not curve:
            return

        key = (record["function"], record["instance"], record["dimension"])
        self._current_problem = key
        problem_curves = self._curves_by_problem.pop(key, {})
        problem_curves.setdefault(record["algorithm"], []).append(curve)
        # Re-insert so this problem is newest: plain setdefault left it at its
        # original position, so the problem being plotted right now could be
        # the one evicted below, blanking the live chart mid-sweep.
        self._curves_by_problem[key] = problem_curves

        # Bound the memory a long sweep can accumulate: a full sweep visits
        # 360 problems, and only the recent ones are ever displayed.
        while len(self._curves_by_problem) > MAX_TRACKED_PROBLEMS:
            self._curves_by_problem.pop(next(iter(self._curves_by_problem)))

        self._redraw()

    def _reset_axes(self, key=None):
        self.axes.clear()
        self.axes.set_xlabel("Iteration")
        # BBOB fitness spans 1e-12 to 1e8; on a linear axis almost every
        # curve flattens onto zero and the plot says nothing.
        self.axes.set_yscale("symlog", linthresh=1e-8)
        self.axes.set_ylabel("Best fitness so far (symlog)")
        if key is None:
            self.axes.set_title("Waiting for the first run")
        else:
            function, instance, dimension = key
            self.axes.set_title(f"F{function}  instance {instance}  dim {dimension}")

    def _redraw(self, force=False):
        now = time.time()
        if not force and now - self._last_redraw < REDRAW_INTERVAL_SECONDS:
            return
        self._last_redraw = now

        self._reset_axes(self._current_problem)
        curves = self._curves_by_problem.get(self._current_problem, {})
        # Distinct rather than thesis-matching: a live plot with three teal
        # lines tells the viewer nothing.
        palette = style.distinct_palette(list(curves))
        fallback = style.distinct_palette(sorted(curves))
        for algorithm, runs in sorted(curves.items()):
            # A missing key gave color=None, which matplotlib silently
            # replaces from the default cycle - so the same algorithm could
            # change colour between redraws.
            colour = palette.get(algorithm) or fallback.get(algorithm)
            for index, curve in enumerate(runs):
                self.axes.plot(
                    range(1, len(curve) + 1), curve,
                    color=colour, linewidth=1.6, alpha=0.9,
                    label=algorithm if index == 0 else None,
                )
        if curves:
            self.axes.legend(loc="upper right")
        figure_theme.apply_to(self.figure)
        self.canvas.draw_idle()

    # -- completion ------------------------------------------------------

    def _on_completed(self, summary):
        self._redraw(force=True)
        self.cancel_button.setEnabled(False)
        # A cancelled or partly failed run left the bar frozen at, say, 43%
        # under the words "Run cancelled" - the bar contradicting the text.
        self.progress.setValue(self.progress.maximum())

        status = "cancelled" if summary["cancelled"] else "finished"
        parts = [
            f"Run {status}: {summary['ok']:,} succeeded",
            f"{summary['failed']:,} failed",
            f"in {format_duration(summary['seconds'])}",
        ]
        self.status.setText("  -  ".join(parts))
        self._append("")
        self._append("  -  ".join(parts))

        if summary["ok"] == 0 and summary["failed"]:
            # Worth stating plainly: the CLI prints per-problem failures and
            # then exits successfully, so "it ran" can mean "nothing worked".
            self._append("Every run failed - no trajectories were written.")

        if self._run_dir is not None:
            try:
                finalise_run(self._run_dir, self._spec, summary)
                self._append(f"Manifest updated: {self._run_dir / 'manifest.json'}")
            except OSError as exc:
                # A manifest that cannot be written must not wedge is_running:
                # everything below is what releases the panel for the next run.
                self._append(f"Could not write the manifest: {exc}")

        # New data may now exist, so every path-keyed cache is stale - not
        # just coverage's. A benchmark writes raw trajectories, which the
        # trajectory and raw-data views read through their own caches.
        results_root.invalidate_caches()
        self._release_runner()
        self.runFinished.emit(summary)

    def _on_failed(self, message):
        self.cancel_button.setEnabled(False)
        self.progress.setValue(self.progress.maximum())
        self.status.setText("Run failed to start")
        self._append(message)
        if self._run_dir is not None and self._spec is not None:
            try:
                write_manifest(self._run_dir, self._spec, status="error",
                               error=message)
            except OSError as exc:
                self._append(f"Could not write the manifest: {exc}")
        # The runner can fail after writing some trajectories, so the caches
        # are stale either way and the Data tab still needs refreshing -
        # _on_completed does both and this used to do neither.
        results_root.invalidate_caches()
        self._release_runner()
        self.runFinished.emit(
            {"ok": self._done, "failed": 0, "cancelled": False,
             "seconds": time.time() - self._started, "error": message}
        )

    # -- public API for the main window ----------------------------------

    def request_cancel(self):
        """Ask the running sweep to stop at the next run boundary.

        Overrides JobPanel's plain cancel() so the header explains that the
        sweep stops at the next run boundary, not immediately.
        """
        self._cancel()

    def _append(self, text):
        self.log.appendPlainText(text)
