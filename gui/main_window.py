"""Main application window.

Assembles the panels, owns the startup work that must not block the UI
(discovering mealpy's optimizers walks the whole package tree and takes about
a second), and routes the two cross-panel actions: running a sweep, and
jumping from a missing-data prompt to the run form that would fill it.
"""
from pathlib import Path

from gui.qt import QtCore, QtGui, QtWidgets, Qt, Signal
from gui import theme
from gui import icons
from gui.core import optimizers as optimizers_core
from gui.core import results_root
from gui.core import workspace as workspace_core
from gui.panels.setup_panel import SetupPanel
from gui.panels.run_panel import RunPanel
from gui.panels.pipeline_panel import PipelinePanel
from gui.panels.viz_panel import VizPanel
from gui.panels.trajectory_panel import TrajectoryPanel
from gui.panels.compare_panel import ComparePanel
from gui.panels.data_panel import DataPanel

import config

APP_NAME = "Optimizer Trajectory Explorer"

(SETUP_TAB, RUN_TAB, PROCESS_TAB, VIZ_TAB, TRAJECTORY_TAB,
 COMPARE_TAB, DATA_TAB) = range(7)


class RegistryLoader(QtCore.QThread):
    """Runs the slow mealpy discovery off the UI thread."""

    loaded = Signal(object)
    failed = Signal(str)

    def run(self):
        try:
            self.loaded.emit(optimizers_core.warm_up())
        except Exception as exc:  # surfaced in the status bar, not swallowed
            self.failed.emit(str(exc))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        # Also applied by __main__, but the self-test and any embedding code
        # build the window against their own QApplication.
        theme.apply()
        self._restore_appearance_mode()
        self.setWindowTitle(APP_NAME)
        self.resize(1300, 860)
        # Below this width the Setup tab's form can't lay out label-beside-
        # field (a deliberate choice - see gui/panels/layout.py::form) without
        # a horizontal scrollbar and truncated hint text. Measured directly:
        # the scrollbar first appears between 1060 and 1070px wide.
        self.setMinimumSize(1080, 700)

        self.setup_panel = SetupPanel()
        self.setup_panel.runRequested.connect(self._start_run)
        self.run_panel = RunPanel()
        self.run_panel.runFinished.connect(self._on_run_finished)
        self.pipeline_panel = PipelinePanel()
        self.pipeline_panel.pipelineFinished.connect(self._on_pipeline_finished)
        self.viz_panel = VizPanel()
        self.viz_panel.generateRequested.connect(self._prefill_run)
        self.viz_panel.processRequested.connect(self._prefill_pipeline)
        self.trajectory_panel = TrajectoryPanel()
        self.compare_panel = ComparePanel()
        self.data_panel = DataPanel()

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setIconSize(QtCore.QSize(18, 18))
        icon_color = theme.tokens()["muted"]
        # Kept as an instance attribute (not just consumed here) so
        # _refresh_icons() can redraw each tab's icon after an explicit
        # appearance switch, without re-adding the tabs.
        self._tab_icon_names = []
        for panel, label, tip, icon_name in (
            (self.setup_panel, "Setup", "Choose what to run and see what it costs",
             "sliders-horizontal"),
            (self.run_panel, "Run", "Watch a sweep: progress, live convergence, log",
             "play"),
            (self.pipeline_panel, "Process",
             "Cluster the trajectories and compute every metric", "workflow"),
            (self.viz_panel, "Visualize", "Render any figure from the computed metrics",
             "chart-column"),
            (self.trajectory_panel, "Trajectory", "Animate a 2-D search in the real space",
             "route"),
            (self.compare_panel, "Compare", "Two algorithms, every metric, side by side",
             "git-compare"),
            (self.data_panel, "Data", "Browse and export any file the pipeline wrote",
             "database"),
        ):
            index = self.tabs.addTab(panel, icons.icon(icon_name, icon_color), label)
            self.tabs.setTabToolTip(index, tip)
            self._tab_icon_names.append(icon_name)

        # The tab widget sat flush against the window frame on all four sides.
        host = QtWidgets.QWidget()
        host_layout = QtWidgets.QVBoxLayout(host)
        host_layout.setContentsMargins(theme.SPACE_M, theme.SPACE_S,
                                       theme.SPACE_M, theme.SPACE_S)
        host_layout.addWidget(self.tabs)
        self.setCentralWidget(host)
        self._add_tab_shortcuts()

        self.data_panel.changeRootRequested.connect(self.choose_results_folder)
        self._build_menu()
        self._restore_results_root()

        # theme's on_change list is module-global, so a callback registered
        # here would otherwise outlive this window (real risk in the
        # self-test and in ad-hoc scripts, which build more than one
        # MainWindow in one process) and fire against a deleted C++ object.
        theme.on_change(self._refresh_icons)
        self.destroyed.connect(lambda: theme.off_change(self._refresh_icons))

        self.statusBar().showMessage("Discovering mealpy optimizers...")

        self._loader = RegistryLoader(self)
        self._loader.loaded.connect(self._on_registry_loaded)
        self._loader.failed.connect(self._on_registry_failed)
        self._loader.start()

    # -- results folder --------------------------------------------------

    def _add_tab_shortcuts(self):
        """Ctrl+<n> to reach the nth tab. There were no shortcuts in the app
        at all beyond Ctrl+O and Ctrl+Q."""
        for index in range(self.tabs.count()):
            shortcut = QtGui.QShortcut(
                QtGui.QKeySequence(f"Ctrl+{index + 1}"), self
            )
            shortcut.activated.connect(
                lambda checked=False, i=index: self.tabs.setCurrentIndex(i)
            )

    def _build_menu(self):
        menu = self.menuBar().addMenu("&File")

        open_action = menu.addAction("Open results folder...")
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.choose_results_folder)

        run_action = menu.addAction("Open a run from this app...")
        run_action.triggered.connect(self.choose_run_folder)

        reset_action = menu.addAction("Use the default results folder")
        reset_action.triggered.connect(self.reset_results_folder)

        menu.addSeparator()
        quit_action = menu.addAction("Quit")
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)

        view_menu = self.menuBar().addMenu("&View")
        appearance_menu = view_menu.addMenu("Appearance")
        group = QtGui.QActionGroup(self)
        group.setExclusive(True)
        self._appearance_actions = {}
        for mode, label in (
            ("system", "Match system"),
            ("light", "Light"),
            ("dark", "Dark"),
        ):
            action = appearance_menu.addAction(label)
            action.setCheckable(True)
            action.triggered.connect(
                lambda _checked=False, m=mode: self._set_appearance_mode(m)
            )
            group.addAction(action)
            self._appearance_actions[mode] = action
        # _restore_appearance_mode() already ran (before any panel was
        # built, to avoid a startup flash), so theme.mode() here is the
        # restored value, not always "system".
        self._appearance_actions[theme.mode()].setChecked(True)

    def _settings(self):
        return QtCore.QSettings("OptimizerTrajectoryExplorer", "gui")

    def _restore_appearance_mode(self):
        """Applied before any panel/icon is built, so the app never flashes
        the wrong palette and icons are painted in the right colour the
        first time, not redrawn a moment later."""
        stored = self._settings().value("appearance_mode", "system", str)
        if stored not in ("system", "light", "dark"):
            stored = "system"
        theme.set_mode(stored)

    def _set_appearance_mode(self, mode):
        theme.set_mode(mode)
        self._settings().setValue("appearance_mode", mode)

    def _refresh_icons(self):
        """Redo every icon that was baked into a QIcon/QPixmap at
        construction time, for every widget still holding one after this
        window is fully built. Registered with theme.on_change() (see
        __init__) rather than called ad hoc, so it also covers a future
        appearance switch triggered some other way, not just this menu."""
        muted = theme.tokens()["muted"]
        for index, name in enumerate(self._tab_icon_names):
            self.tabs.setTabIcon(index, icons.icon(name, muted))
        for panel in (self.setup_panel, self.pipeline_panel, self.viz_panel,
                      self.compare_panel, self.trajectory_panel):
            panel.refresh_icons()

    def _restore_results_root(self):
        """Reopen whatever folder was in use last time, if it still exists."""
        stored = self._settings().value("results_root", "", str)
        if stored and Path(stored).is_dir():
            results_root.set_root(stored)
            self.data_panel.refresh()
            self.pipeline_panel.refresh()

    def choose_results_folder(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose a results folder", str(results_root.current_root())
        )
        if directory:
            self._apply_results_root(directory)

    def choose_run_folder(self):
        """Pick one of this app's own runs. Each run directory is itself a
        results tree - it has an outputs/ - so the raw-trajectory views work
        against it directly."""
        runs = workspace_core.list_runs()
        if not runs:
            QtWidgets.QMessageBox.information(
                self, APP_NAME,
                f"No runs yet in {workspace_core.workspace_root()}.\n\n"
                "Start one from the Setup tab.",
            )
            return
        labels = [
            f"{entry['path'].name}  ({(entry['manifest'] or {}).get('status', '?')}, "
            f"{(entry['manifest'] or {}).get('completed_runs', '?')} runs)"
            for entry in runs
        ]
        choice, accepted = QtWidgets.QInputDialog.getItem(
            self, "Open a run", "Run:", labels, 0, False
        )
        if accepted and choice:
            self._apply_results_root(runs[labels.index(choice)]["path"])

    def reset_results_folder(self):
        self._apply_results_root(results_root.DEFAULT_ROOT)

    def _apply_results_root(self, directory):
        summary = results_root.set_root(directory)
        self._settings().setValue("results_root", str(summary.path))

        self.data_panel.refresh()
        self.viz_panel.refresh()
        self.pipeline_panel.refresh()
        self.trajectory_panel.reset_view()

        if summary.is_usable:
            self.statusBar().showMessage(
                f"Results folder: {summary.path}  -  {summary.describe()}", 15000
            )
        else:
            # Say so plainly rather than leaving every tab mysteriously empty.
            QtWidgets.QMessageBox.warning(
                self, APP_NAME,
                f"No pipeline output found under:\n{summary.path}\n\n"
                "A results folder is expected to contain outputs/, data/ "
                "and/or metrics_data/. The app will show nothing until it "
                "points at one that does.",
            )
            self.statusBar().showMessage(
                f"Results folder: {summary.path}  -  nothing found here", 15000
            )

    # -- startup ---------------------------------------------------------

    def _on_registry_loaded(self, registry):
        self.setup_panel.set_registry(registry)
        self.viz_panel.activate()
        self.trajectory_panel.set_registry(registry)
        self.compare_panel.set_registry(registry)
        self.statusBar().showMessage(
            f"{len(registry.names)} optimizers available "
            f"across {len(registry.by_family)} families",
            5000,
        )

    def _on_registry_failed(self, message):
        self.statusBar().showMessage(f"Could not load optimizers: {message}")
        QtWidgets.QMessageBox.critical(
            self, APP_NAME, f"Could not load mealpy optimizers:\n\n{message}"
        )

    # -- cross-panel actions ---------------------------------------------

    def _prefill_run(self, request):
        """From 'this has not been computed yet' to a ready-to-run setup."""
        self.setup_panel.prefill(
            algorithms=request.get("algorithms"),
            dimensions=request.get("dimensions"),
            functions=request.get("functions"),
        )
        self.tabs.setCurrentIndex(SETUP_TAB)
        self.statusBar().showMessage(
            "Setup prefilled with the missing selection. Review the cost, then Run.",
            8000,
        )

    def _prefill_pipeline(self, stage_names):
        """From "this has not been computed yet" to the stages that would."""
        self.pipeline_panel.select_stages(stage_names)
        self.tabs.setCurrentIndex(PROCESS_TAB)
        self.statusBar().showMessage(
            "Process tab set to the stages that would produce the missing "
            "data. Review what it overwrites, then Run.",
            8000,
        )

    def _start_run(self, spec, write_to_real_outputs):
        if self.run_panel.is_running:
            QtWidgets.QMessageBox.information(
                self, APP_NAME, "A run is already in progress."
            )
            self.tabs.setCurrentIndex(RUN_TAB)
            return

        if write_to_real_outputs:
            run_dir, out_dir = None, config.OUTPUTS_DIR
        else:
            try:
                run_dir = workspace_core.create_run_dir(spec)
            except OSError as exc:
                QtWidgets.QMessageBox.critical(
                    self, APP_NAME, f"Could not create the run directory:\n\n{exc}"
                )
                return
            out_dir = run_dir / "outputs"

        self.tabs.setCurrentIndex(RUN_TAB)
        self.run_panel.start(spec, run_dir, out_dir)

    def _on_run_finished(self, summary):
        # Show the user where their data actually went: the run wrote to the
        # workspace, not to the repo tree the other Data roots describe.
        self.data_panel.refresh()
        self.data_panel.reveal_latest_run()
        self.statusBar().showMessage(
            f"Run finished: {summary['ok']:,} succeeded, {summary['failed']:,} failed "
            f"- see the Data tab, under \"GUI runs\"",
            15000,
        )

    def _on_pipeline_finished(self, summary):
        # The pipeline rewrites the tables the visualizations read, so the
        # catalog's availability and the data browser are both stale now.
        self.viz_panel.refresh()
        self.data_panel.refresh()
        if summary.get("error"):
            self.statusBar().showMessage("Pipeline failed - see the Process tab", 15000)
            return
        state = "cancelled" if summary.get("cancelled") else "finished"
        self.statusBar().showMessage(
            f"Pipeline {state}: {len(summary.get('stages', []))} stage(s), "
            f"{summary.get('written', 0):,} file(s) written",
            15000,
        )

    # -- shutdown --------------------------------------------------------

    def _busy_panel(self):
        """The panel running a background job, if any.

        Both of them own a QThread that must not be destroyed underneath it,
        so closing has to account for either being active.
        """
        for panel in (self.run_panel, self.pipeline_panel):
            if panel.is_running:
                return panel
        return None

    def _stop_registry_loader(self):
        """Let the startup mealpy discovery finish before the window dies.

        It is a QThread parented to this window, so closing inside the ~1s it
        takes destroyed it mid-run ("QThread: Destroyed while thread is still
        running"). It only reads from mealpy, so waiting for it is safe and
        short; nothing is left half-written either way.
        """
        loader, self._loader = getattr(self, "_loader", None), None
        if loader is None:
            return
        if loader.isRunning():
            loader.wait(5000)
        loader.deleteLater()

    def closeEvent(self, event):
        """Don't let a job be killed mid-write by closing the window."""
        panel = self._busy_panel()
        if panel is None:
            self._stop_registry_loader()
            event.accept()
            return
        answer = QtWidgets.QMessageBox.question(
            self,
            APP_NAME,
            f"A {panel.job_noun} is still in progress.\n\n"
            "Quitting now stops it at the next boundary and leaves "
            "partial results on disk. Quit anyway?",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Cancel,
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            event.ignore()
            return

        self.statusBar().showMessage("Stopping at the next boundary...")
        panel.request_cancel()
        if panel.wait_for_exit(10000):
            self._stop_registry_loader()
            event.accept()
            return

        # Accepting here would destroy the panel while its QThread child is
        # still running, which Qt turns into "QThread: Destroyed while thread
        # is still running" and an abort. Keep the window open instead and let
        # the user decide again once the current item ends.
        QtWidgets.QMessageBox.information(
            self, APP_NAME,
            f"The {panel.job_noun} has not stopped yet.\n\n"
            "It finishes the item it is working on first. The window will "
            "stay open until then - try closing it again in a moment.",
        )
        event.ignore()
