"""Main application window.

Assembles the panels, owns the startup work that must not block the UI
(discovering mealpy's optimizers walks the whole package tree and takes about
a second), and routes the two cross-panel actions: running a sweep, and
jumping from a missing-data prompt to the run form that would fill it.
"""
from pathlib import Path

from gui.qt import QtCore, QtWidgets, Qt, Signal
from gui.core import optimizers as optimizers_core
from gui.core import results_root
from gui.core import workspace as workspace_core
from gui.panels.setup_panel import SetupPanel
from gui.panels.run_panel import RunPanel
from gui.panels.viz_panel import VizPanel
from gui.panels.trajectory_panel import TrajectoryPanel
from gui.panels.compare_panel import ComparePanel
from gui.panels.data_panel import DataPanel

import config

APP_NAME = "Optimizer Trajectory Explorer"

SETUP_TAB, RUN_TAB, VIZ_TAB, TRAJECTORY_TAB, COMPARE_TAB, DATA_TAB = range(6)


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
        self.setWindowTitle(APP_NAME)
        self.resize(1300, 860)

        self.setup_panel = SetupPanel()
        self.setup_panel.runRequested.connect(self._start_run)
        self.run_panel = RunPanel()
        self.run_panel.runFinished.connect(self._on_run_finished)
        self.viz_panel = VizPanel()
        self.viz_panel.generateRequested.connect(self._prefill_run)
        self.trajectory_panel = TrajectoryPanel()
        self.compare_panel = ComparePanel()
        self.data_panel = DataPanel()

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self.setup_panel, "Setup")
        self.tabs.addTab(self.run_panel, "Run")
        self.tabs.addTab(self.viz_panel, "Visualize")
        self.tabs.addTab(self.trajectory_panel, "Trajectory")
        self.tabs.addTab(self.compare_panel, "Compare")
        self.tabs.addTab(self.data_panel, "Data")
        self.setCentralWidget(self.tabs)

        self.data_panel.changeRootRequested.connect(self.choose_results_folder)
        self._build_menu()
        self._restore_results_root()

        self.statusBar().showMessage("Discovering mealpy optimizers...")

        self._loader = RegistryLoader(self)
        self._loader.loaded.connect(self._on_registry_loaded)
        self._loader.failed.connect(self._on_registry_failed)
        self._loader.start()

    # -- results folder --------------------------------------------------

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

    def _settings(self):
        return QtCore.QSettings("OptimizerTrajectoryExplorer", "gui")

    def _restore_results_root(self):
        """Reopen whatever folder was in use last time, if it still exists."""
        stored = self._settings().value("results_root", "", str)
        if stored and Path(stored).is_dir():
            results_root.set_root(stored)
            self.data_panel.refresh()

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

    # -- shutdown --------------------------------------------------------

    def closeEvent(self, event):
        """Don't let a sweep be killed mid-write by closing the window."""
        if not self.run_panel.is_running:
            event.accept()
            return
        answer = QtWidgets.QMessageBox.question(
            self,
            APP_NAME,
            "A benchmark run is still in progress.\n\n"
            "Quitting now stops it at the next run boundary and leaves "
            "partial results on disk. Quit anyway?",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Cancel,
        )
        if answer == QtWidgets.QMessageBox.StandardButton.Yes:
            self.run_panel._cancel()
            self.run_panel._runner and self.run_panel._runner.wait(10000)
            event.accept()
        else:
            event.ignore()
