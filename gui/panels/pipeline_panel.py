"""Run the analysis pipeline - clustering and every metric - without a terminal.

The Run tab watches a benchmark sweep: one job, one progress bar, a live
convergence plot keyed on the benchmark's own record schema. This is a
different shape of job - fourteen ordered stages, each with its own item
count - so it is its own panel rather than a mode of that one.

What this panel is careful about, in order of how much damage each would do:

  * It overwrites the real data/ and metrics_data/ in place. Nothing in the
    pipeline is transactional and no stage deletes, so the confirmation names
    every directory and its current size, and cancelling is described
    honestly rather than implied to be safe.
  * It holds the app for as long as the run takes, which for a full rebuild
    is hours. The dialog says so.
  * When it finishes it has rewritten the tables behind four memoised caches,
    so it invalidates all of them - coverage.invalidate() alone is not enough.
"""
import time

from gui import icons, theme
from gui.core import pipeline as pipeline_core
from gui.core import results_root
from gui.panels import layout as panel_layout
from gui.panels.job_panel import JobPanel
from gui.qt import QtWidgets, Qt, Signal

import config

#: Stages ticked when the panel first opens. The benchmark is left out: it
#: needs the Setup tab's form, and ticking it by default would invite a
#: multi-hour run from a panel the user just opened.
DEFAULT_START = "preprocess"


class PipelinePanel(JobPanel):
    job_noun = "pipeline run"

    pipelineFinished = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._started = None
        self._stage_total = 0
        self._stage_index = 0

        controls = self._build_controls()

        # -- header: two bars, because "which stage" and "how far into it"
        #    are different questions and one bar can only answer one.
        self.status = QtWidgets.QLabel("No pipeline run in progress.")
        self.status.setProperty("class", "metric")
        self.stage_label = QtWidgets.QLabel()
        self.stage_label.setProperty("class", "hint")

        self.overall_progress = QtWidgets.QProgressBar()
        self.overall_progress.setFormat("stage %v of %m")
        self.stage_progress = QtWidgets.QProgressBar()
        self.stage_progress.setFormat("%v of %m")

        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel)

        header = QtWidgets.QGridLayout()
        header.setHorizontalSpacing(theme.SPACE_M)
        header.setVerticalSpacing(theme.SPACE_XS)
        header.addWidget(self.status, 0, 0)
        header.addWidget(self.cancel_button, 0, 1, 3, 1, Qt.AlignmentFlag.AlignTop)
        header.addWidget(self.overall_progress, 1, 0)
        header.addWidget(self.stage_progress, 2, 0)
        header.addWidget(self.stage_label, 3, 0, 1, 2)
        header.setColumnStretch(0, 1)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(20000)
        self.log.setFont(theme.monospace_font(self.log))

        right = panel_layout.column(margins=(panel_layout.S, 0, 0, 0))
        right.addLayout(header)
        right.addWidget(self.log, 1)
        right_host = QtWidgets.QWidget()
        right_host.setLayout(right)

        splitter = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(controls)
        splitter.addWidget(right_host)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([420, 880])
        splitter.setChildrenCollapsible(False)

        layout = panel_layout.column(self)
        layout.addWidget(splitter)

        self._select_from(DEFAULT_START)
        self.refresh()

    # -- construction ----------------------------------------------------

    def refresh_icons(self):
        """See setup_panel.SetupPanel.refresh_icons."""
        self.run_button.setIcon(icons.icon("play", theme.tokens()["accent_text"]))

    def refresh(self):
        """Re-read what this panel would run against.

        Nothing here reacts to results_root.set_root() on its own - the main
        window calls this alongside data_panel.refresh() and viz_panel.refresh()
        whenever the results folder changes (File > Open results folder..., a
        restored setting at startup, or the default reset), so this label
        never shows a stale folder.
        """
        dims = ", ".join(str(d) for d in config.discover_dimensions(config.OUTPUTS_DIR))
        self.context.setText(
            f"Results folder: {results_root.current_root()}\n"
            f"Dimensions: {dims}   -   Clustering method: kmeans\n"
            f"Algorithms are not filtered here - every stage processes "
            f"whatever algorithms already have data in that folder."
        )

    def _build_controls(self):
        box = panel_layout.group_box("Stages")

        # What this panel would actually run against, visible before a single
        # box is ticked. "Results folder: <checkout>" in the log after
        # clicking Run used to be the only place this showed up.
        self.context = QtWidgets.QLabel()
        self.context.setWordWrap(True)
        self.context.setProperty("class", "hint")

        self.start_from = QtWidgets.QComboBox()
        for stage in pipeline_core.STAGES:
            self.start_from.addItem(stage.label, stage.name)
        self.start_from.setToolTip(
            "Tick this stage and everything after it - the same thing "
            "`run_pipeline.py --from <stage>` does."
        )
        self.start_from.activated.connect(
            lambda _index: self._select_from(self.start_from.currentData())
        )

        self.stage_list = QtWidgets.QListWidget()
        self.stage_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        for stage in pipeline_core.STAGES:
            item = QtWidgets.QListWidgetItem(stage.label)
            item.setData(Qt.ItemDataRole.UserRole, stage.name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            tip = stage.description
            if stage.slow:
                tip += "\n\nThis stage is slow."
            item.setToolTip(tip)
            self.stage_list.addItem(item)
        self.stage_list.itemChanged.connect(lambda _item: self._refresh_summary())

        self.summary = QtWidgets.QLabel()
        self.summary.setWordWrap(True)
        self.summary.setProperty("class", "hint")

        self.warning = QtWidgets.QLabel()
        self.warning.setWordWrap(True)
        self.warning.setProperty("class", "warning")

        self.run_button = QtWidgets.QPushButton(
            icons.icon("play", theme.tokens()["accent_text"]), "Run pipeline")
        self.run_button.setProperty("class", "primary")
        self.run_button.setDefault(True)
        self.run_button.setShortcut("Ctrl+Return")
        self.run_button.setToolTip("Run the ticked stages in order (Ctrl+Return)")
        self.run_button.clicked.connect(self._request_run)

        form = panel_layout.column(box)
        form.addWidget(self.context)
        form.addWidget(QtWidgets.QLabel("Start from:"))
        form.addWidget(self.start_from)
        form.addWidget(self.stage_list, 1)
        form.addWidget(self.summary)
        form.addWidget(self.warning)
        form.addWidget(self.run_button)
        return box

    # -- selection -------------------------------------------------------

    def _select_from(self, name):
        """Tick `name` and everything downstream of it."""
        try:
            wanted = {s.name for s in pipeline_core.stages_from(name)}
        except KeyError:
            return
        # Keep the combo honest: it said "Benchmark" while the ticks started
        # at Preprocess, which is the sort of mismatch that gets a multi-hour
        # run started by accident.
        index = self.start_from.findData(name)
        if index >= 0 and index != self.start_from.currentIndex():
            self.start_from.blockSignals(True)
            self.start_from.setCurrentIndex(index)
            self.start_from.blockSignals(False)
        self.stage_list.blockSignals(True)
        try:
            for row in range(self.stage_list.count()):
                item = self.stage_list.item(row)
                state = (Qt.CheckState.Checked
                         if item.data(Qt.ItemDataRole.UserRole) in wanted
                         else Qt.CheckState.Unchecked)
                item.setCheckState(state)
        finally:
            self.stage_list.blockSignals(False)
        self._refresh_summary()

    def select_stages(self, names):
        """Tick exactly these stages.

        Used by the Visualize tab's "Run these steps now", which knows which
        stages would produce the data it is missing - a narrower answer than
        "everything from here on".
        """
        wanted = set(names)
        self.stage_list.blockSignals(True)
        try:
            for row in range(self.stage_list.count()):
                item = self.stage_list.item(row)
                item.setCheckState(
                    Qt.CheckState.Checked
                    if item.data(Qt.ItemDataRole.UserRole) in wanted
                    else Qt.CheckState.Unchecked
                )
        finally:
            self.stage_list.blockSignals(False)
        # Point the combo at the earliest stage now ticked, so it is not
        # describing a selection the user did not make.
        for stage in pipeline_core.STAGES:
            if stage.name in wanted:
                index = self.start_from.findData(stage.name)
                if index >= 0:
                    self.start_from.blockSignals(True)
                    self.start_from.setCurrentIndex(index)
                    self.start_from.blockSignals(False)
                break
        self._refresh_summary()

    def selected_stages(self):
        return [
            self.stage_list.item(row).data(Qt.ItemDataRole.UserRole)
            for row in range(self.stage_list.count())
            if self.stage_list.item(row).checkState() == Qt.CheckState.Checked
        ]

    def current_spec(self):
        return pipeline_core.PipelineSpec(stages=self.selected_stages())

    def _refresh_summary(self):
        spec = self.current_spec()
        runnable = spec.runnable()
        self.run_button.setEnabled(bool(runnable) and not self.is_running)

        if not runnable:
            self.summary.setText("Tick at least one stage.")
            self.warning.hide()
            return

        slow = [s.label for s in runnable if s.slow]
        self.summary.setText(
            f"{len(runnable)} stage(s), dimensions "
            f"{', '.join(str(d) for d in config.discover_dimensions(config.OUTPUTS_DIR))}."
        )

        notes = []
        if spec.skipped():
            notes.append(
                "The benchmark stage is configured on the Setup tab and will "
                "be skipped here."
            )
        if slow:
            notes.append(f"Slow stages selected: {', '.join(slow)}.")
        notes.append("This overwrites existing results in place.")
        self.warning.setText("  ".join(notes))
        self.warning.show()

    # -- starting --------------------------------------------------------

    def _request_run(self):
        if self.is_running:
            QtWidgets.QMessageBox.information(
                self, "Pipeline", "A pipeline run is already in progress.")
            return
        spec = self.current_spec()
        if not spec.runnable():
            return
        if not self._confirm(spec):
            return
        self.start(spec)

    def _confirm(self, spec):
        """Name what will be overwritten, and how big it currently is.

        No stage deletes and none is transactional, so this is the only point
        at which the user finds out what they are about to lose.
        """
        targets = spec.targets()
        lines = []
        for target in targets:
            lines.append(f"    {target}   ({_describe_size(target)})")

        body = [
            spec.describe(),
            "",
            "This overwrites the following, in place:",
            *lines,
            "",
            "Files are replaced one at a time and nothing is deleted first, "
            "so cancelling part-way leaves a mix of new and old files - and "
            "the merge stage reads whatever it finds.",
            "",
            "The run happens inside this window, so it has to stay open until "
            "it finishes. A full rebuild can take hours.",
            "",
            "Start the run?",
        ]
        answer = QtWidgets.QMessageBox.question(
            self, "Run the pipeline?", "\n".join(body),
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Cancel,
        )
        return answer == QtWidgets.QMessageBox.StandardButton.Yes

    def start(self, spec):
        if self.is_running:
            return False

        self._spec = spec
        self._started = time.time()
        stages = spec.runnable()
        self._stage_total = len(stages)
        self._stage_index = 0

        self.log.clear()
        self._append(f"Results folder: {results_root.current_root()}")
        self._append(spec.describe())
        self._append("")

        self.overall_progress.setRange(0, len(stages))
        self.overall_progress.setValue(0)
        self.stage_progress.setRange(0, 1)
        self.stage_progress.setValue(0)
        self.status.setText("Starting...")
        self.cancel_button.setEnabled(True)
        self.run_button.setEnabled(False)

        runner = self._adopt_runner(pipeline_core.PipelineRunner(spec, parent=self))
        runner.stageStarted.connect(self._on_stage_started)
        runner.progressed.connect(self._on_progress)
        runner.message.connect(self._append)
        runner.completed.connect(self._on_completed)
        runner.failed.connect(self._on_failed)
        runner.start()
        return True

    def _cancel(self):
        if self._runner is None:
            return
        self.cancel_button.setEnabled(False)
        self.status.setText("Cancelling - stopping at the next item...")
        self.request_cancel()

    # -- progress --------------------------------------------------------

    def _on_stage_started(self, payload):
        self._stage_index = payload["index"]
        self.overall_progress.setValue(payload["index"])
        self.status.setText(
            f"{payload['label']}  -  stage {payload['index'] + 1} "
            f"of {payload['total']}"
        )
        self.stage_progress.setRange(0, 1)
        self.stage_progress.setValue(0)

    def _on_progress(self, payload):
        total = max(int(payload.get("total") or 0), 1)
        self.stage_progress.setRange(0, total)
        self.stage_progress.setValue(int(payload.get("done") or 0))
        label = payload.get("label") or ""
        if label:
            self.stage_label.setText(label)

    # -- completion ------------------------------------------------------

    def _on_completed(self, summary):
        self.cancel_button.setEnabled(False)
        self.overall_progress.setValue(self.overall_progress.maximum())
        self.stage_progress.setValue(self.stage_progress.maximum())

        state = "cancelled" if summary["cancelled"] else "finished"
        text = (f"Pipeline {state}: {len(summary['stages'])} stage(s), "
                f"{summary['written']:,} file(s) written "
                f"in {_duration(summary['seconds'])}")
        self.status.setText(text)
        self._append("")
        self._append(text)

        if summary["cancelled"]:
            self._append(
                "Cancelled part-way: the stages that did not run still hold "
                "output from the previous run. Re-run from the stage named "
                "above to make the tree consistent again."
            )

        manifest = pipeline_core.write_manifest(self._spec, summary)
        if manifest is not None:
            self._append(f"Recorded in {manifest}")

        self._finish()
        self.pipelineFinished.emit(summary)

    def _on_failed(self, message):
        self.cancel_button.setEnabled(False)
        self.status.setProperty("class", "error")
        theme.restyle(self.status)
        self.status.setText("Pipeline failed")
        self._append("")
        self._append(message)
        self._append(
            "Whatever ran before the failure is on disk; the stages after it "
            "did not run."
        )
        self._finish()
        self.pipelineFinished.emit(
            {"stages": [], "written": 0, "cancelled": False,
             "seconds": time.time() - (self._started or time.time()),
             "error": message}
        )

    def _finish(self):
        """Everything both end paths must do, including the cache drop.

        A pipeline run rewrites entropy_dim_{d}.csv, merged_dim_{d}.csv,
        spearman_dim_{d}.csv and scalars.csv - the tables behind four
        memoised caches. coverage.invalidate() alone clears only one of them,
        which would leave the figures rendering pre-run numbers.
        """
        results_root.invalidate_caches()
        self._release_runner()
        self.run_button.setEnabled(True)
        self._refresh_summary()


def _describe_size(path):
    """Rough current size of a target, for the confirmation dialog."""
    from pathlib import Path

    target = Path(path)
    if not target.exists():
        return "does not exist yet"
    if target.is_file():
        return _bytes(target.stat().st_size)
    total = 0
    count = 0
    for item in target.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                continue
            count += 1
            if count > 20000:      # a rough number is enough to warn with
                return f"over {_bytes(total)}"
    return f"{_bytes(total)}, {count:,} files"


def _bytes(size):
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def _duration(seconds):
    from gui.core.workspace import format_duration

    return format_duration(seconds)
