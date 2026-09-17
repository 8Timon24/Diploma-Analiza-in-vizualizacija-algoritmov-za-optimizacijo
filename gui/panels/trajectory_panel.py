"""Animated 2-D search trajectory player.

The other twenty views render a figure and stop. This one is a control: the
population is redrawn per iteration over a contour of the true objective, so
you watch the search happen rather than reading a summary of it.

It is deliberately not a catalog entry - the registry's contract is
render(params) -> Figure, and an animation is a widget with its own state.

Drawing is incremental: the contour and the optimum marker are drawn once
when a problem is loaded, and each frame only moves the scatter offsets.
Clearing and redrawing the axes per frame would make the contour dominate
the frame time and the playback stutter.
"""
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from gui.panels import layout as panel_layout
from gui.qt import QtCore, QtWidgets, Qt
from gui.core import optimizers as optimizers_core
from gui.panels.widgets import CheckableList
from gui.viz import figure_theme, style, trajectory

import config

MAX_ALGORITHMS = 4
TRAIL_ALPHA = 0.10
DEFAULT_FPS = 4


def _preferred_index(values, wanted):
    """Index of `wanted` in `values`, or 0 when it is not there.

    Every one of these lists comes from config.py, which honours the
    PIPELINE_TEST_* overrides, so no fixed value is guaranteed to be present.
    """
    try:
        return list(values).index(wanted)
    except ValueError:
        return 0


class TrajectoryPanel(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._frame = None
        self._selection = None
        self._has_clusters = False
        self._scatters = {}
        self._trail = None
        self._cluster_scatter = None
        self._iterations = (1, 1)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._advance)

        controls = self._build_controls()

        self.figure = Figure(figsize=(7, 7), layout="constrained")
        self.axes = self.figure.add_subplot(1, 1, 1)
        self.axes.set_title("Choose a problem and press Load")
        figure_theme.apply_to(self.figure)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        playback = self._build_playback()

        right = panel_layout.column(margins=(panel_layout.S, 0, 0, 0))
        right.addWidget(self.toolbar)
        right.addWidget(self.canvas, 1)
        right.addLayout(playback)
        right_host = QtWidgets.QWidget()
        right_host.setLayout(right)

        # Proportions, not pixel caps: a hard setMaximumWidth made the
        # splitter handle look broken - it dragged and the pane refused
        # to grow.
        splitter = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(controls)
        splitter.addWidget(right_host)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([330, 970])
        splitter.setChildrenCollapsible(False)

        layout = panel_layout.column(self)
        layout.addWidget(splitter)
        self._set_playback_enabled(False)

    # -- construction ----------------------------------------------------

    def _build_controls(self):
        box = QtWidgets.QGroupBox("Problem")
        form = QtWidgets.QFormLayout(box)

        self.function = QtWidgets.QComboBox()
        for f in config.FUNCTIONS:
            self.function.addItem(f"F{f}", f)
        # F16 (Weierstrass) is the most instructive default, but config.FUNCTIONS
        # is overridable via PIPELINE_TEST_FUNCTIONS - a bare .index(16) raised
        # ValueError here and took the whole window down, since this panel is
        # built inside MainWindow.__init__.
        self.function.setCurrentIndex(_preferred_index(config.FUNCTIONS, 16))

        self.instance = QtWidgets.QComboBox()
        for i in config.INSTANCES:
            self.instance.addItem(str(i), i)
        self.instance.setCurrentIndex(min(2, max(len(config.INSTANCES) - 1, 0)))

        self.run = QtWidgets.QComboBox()
        for r in config.SEEDS:
            self.run.addItem(str(r), r)

        # Fall back to whatever the first two algorithms are: the named pair is
        # absent from a trimmed ALGORITHMS_OF_INTEREST, and an empty default
        # makes Load fail on a panel the user has not touched yet.
        preferred = [a for a in ("OriginalDE", "OriginalGWO")
                     if a in config.ALGORITHMS_OF_INTEREST]
        self.algorithms = CheckableList(
            list(config.ALGORITHMS_OF_INTEREST),
            preferred or list(config.ALGORITHMS_OF_INTEREST)[:2],
            rows=8,
        )

        self.colour_mode = QtWidgets.QComboBox()
        self.colour_mode.addItem("Colour by algorithm", "algorithm")
        self.colour_mode.addItem("Colour by cluster", "cluster")
        self.colour_mode.currentIndexChanged.connect(self._rebuild_artists)

        self.show_landscape = QtWidgets.QCheckBox("Show objective landscape")
        self.show_landscape.setChecked(True)
        self.show_trail = QtWidgets.QCheckBox("Show trail of earlier iterations")
        self.show_trail.setChecked(True)
        self.show_trail.toggled.connect(lambda _on: self._draw_current())

        self.load_button = QtWidgets.QPushButton("Load")
        self.load_button.setProperty("class", "primary")
        self.load_button.setDefault(True)
        self.load_button.setShortcut("Ctrl+Return")
        self.load_button.setToolTip("Load this problem's trajectories (Ctrl+Return)")
        self.load_button.clicked.connect(self.load)

        self.status = QtWidgets.QLabel("Dimension 2 only - this plots the real "
                                       "search space.")
        self.status.setWordWrap(True)
        self.status.setProperty("class", "hint")

        form.addRow("Function:", self.function)
        form.addRow("Instance:", self.instance)
        form.addRow("Run:", self.run)
        form.addRow("Algorithms:", self.algorithms)
        form.addRow("", self.colour_mode)
        form.addRow("", self.show_landscape)
        form.addRow("", self.show_trail)
        form.addRow("", self.load_button)
        form.addRow("", self.status)
        return box

    def _build_playback(self):
        self.play_button = QtWidgets.QPushButton("Play")
        self.play_button.setShortcut("Space")
        self.play_button.setToolTip("Play / pause the animation (Space)")
        self.play_button.clicked.connect(self.toggle_play)

        self.slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(1)
        self.slider.valueChanged.connect(lambda _v: self._draw_current())

        self.iteration_label = QtWidgets.QLabel("-")
        self.iteration_label.setMinimumWidth(110)

        self.fps = QtWidgets.QSpinBox()
        self.fps.setToolTip("Playback speed, in frames (iterations) per second")
        self.fps.setRange(1, 30)
        self.fps.setValue(DEFAULT_FPS)
        self.fps.setSuffix(" fps")
        self.fps.valueChanged.connect(self._retime)

        self.export_button = QtWidgets.QPushButton("Export GIF...")
        self.export_button.setToolTip(
            "Render every iteration to an animated GIF. This runs in the "
            "window and can take minutes for a long run."
        )
        self.export_button.clicked.connect(self._export_gif)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.play_button)
        row.addWidget(self.slider, 1)
        row.addWidget(self.iteration_label)
        row.addWidget(self.fps)
        row.addWidget(self.export_button)
        return row

    def _set_playback_enabled(self, enabled):
        for widget in (self.play_button, self.slider, self.fps, self.export_button):
            widget.setEnabled(enabled)

    def set_registry(self, registry):
        """Offer every optimizer, not just the thesis set - a GUI run may
        have produced trajectories for something else."""
        current = self.algorithms.values()
        self.algorithms.replace(registry.names, selected=current)

    # -- loading ---------------------------------------------------------

    def load(self):
        algorithms = self.algorithms.values()
        if not algorithms:
            self._fail("Select at least one algorithm.")
            return
        if len(algorithms) > MAX_ALGORITHMS:
            self._fail(f"At most {MAX_ALGORITHMS} algorithms - more is unreadable.")
            return

        function = self.function.currentData()
        instance = self.instance.currentData()
        run = self.run.currentData()

        QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            frame, has_clusters = trajectory.load_positions(2, function, instance)
            selection = trajectory.select(frame, algorithms, run)
        except Exception as exc:
            QtWidgets.QApplication.restoreOverrideCursor()
            self._fail(str(exc))
            return

        self._frame = frame
        self._selection = selection
        self._has_clusters = has_clusters
        self._iterations = trajectory.iteration_range(selection)
        self._function, self._instance, self._run = function, instance, run

        self.colour_mode.setEnabled(has_clusters)
        if not has_clusters and self.colour_mode.currentData() == "cluster":
            self.colour_mode.setCurrentIndex(0)

        self._draw_background()
        self._rebuild_artists()

        low, high = self._iterations
        self.slider.blockSignals(True)
        self.slider.setMinimum(low)
        self.slider.setMaximum(high)
        self.slider.setValue(low)
        self.slider.blockSignals(False)
        self._set_playback_enabled(True)
        self._draw_current()

        note = f"{len(selection):,} points, iterations {low}-{high}"
        if not has_clusters:
            note += " - no clustering output, so cluster colouring is unavailable"
        self.status.setText(note)
        QtWidgets.QApplication.restoreOverrideCursor()

    def reset_view(self):
        """Drop whatever was loaded: it came from a different results tree."""
        self.stop()
        self._frame = None
        self._selection = None
        self._set_playback_enabled(False)
        self.axes.clear()
        self.axes.set_title("Choose a problem and press Load")
        self.canvas.draw_idle()
        self.status.setText("Results folder changed - press Load.")

    def _fail(self, message):
        self.stop()
        self._set_playback_enabled(False)
        self.status.setText(message)
        self.axes.clear()
        self.axes.set_title("Could not load")
        self.axes.text(0.5, 0.5, message, ha="center", va="center", wrap=True,
                       transform=self.axes.transAxes, fontsize=9)
        self.canvas.draw_idle()

    # -- drawing ---------------------------------------------------------

    def _draw_background(self):
        self.axes.clear()
        self._scatters = {}
        self._trail = None
        self._cluster_scatter = None

        if self.show_landscape.isChecked():
            try:
                mesh_x, mesh_y, grid = trajectory.landscape(self._function, self._instance)
                self.axes.contourf(mesh_x, mesh_y, grid, levels=40, cmap="YlOrRd",
                                   alpha=0.45)
            except Exception:
                pass  # a missing landscape must not stop the animation

        found = trajectory.optimum(self._function, self._instance)
        if found is not None:
            _f_opt, x_opt = found
            if len(x_opt) >= 2:
                self.axes.scatter(x_opt[0], x_opt[1], marker="*", s=420, color="red",
                                  edgecolor="black", linewidth=0.6, zorder=6,
                                  label="true optimum")
        self.axes.set_xlabel("x0")
        self.axes.set_ylabel("x1")

    def _rebuild_artists(self):
        if self._selection is None:
            return
        for artist in list(self._scatters.values()):
            artist.remove()
        self._scatters = {}
        if self._cluster_scatter is not None:
            self._cluster_scatter.remove()
            self._cluster_scatter = None
        if self._trail is not None:
            self._trail.remove()
            self._trail = None

        self._trail = self.axes.scatter([], [], s=9, color="#555555",
                                        alpha=TRAIL_ALPHA, zorder=2,
                                        linewidths=0)

        if self._colour_by_cluster():
            clusters = self._selection["cluster"]
            self._cluster_scatter = self.axes.scatter(
                [], [], s=34, cmap="tab20", c=[], zorder=4,
                vmin=float(clusters.min()), vmax=float(clusters.max()),
                edgecolor="white", linewidth=0.3,
            )
        else:
            algorithms = sorted(self._selection["algorithm"].unique())
            palette = style.distinct_palette(algorithms)
            for algorithm in algorithms:
                self._scatters[algorithm] = self.axes.scatter(
                    [], [], s=34, color=palette[algorithm], zorder=4,
                    edgecolor="white", linewidth=0.3, label=algorithm,
                )
        self.axes.legend(loc="upper right", fontsize=8)
        self._draw_current()

    def _colour_by_cluster(self):
        return (
            self._has_clusters
            and self.colour_mode.currentData() == "cluster"
            and "cluster" in (self._selection.columns if self._selection is not None else [])
        )

    def _draw_current(self):
        if self._selection is None:
            return
        self._update_frame(self.slider.value())
        self.canvas.draw_idle()

    def _update_frame(self, iteration):
        """Move the artists to one iteration. Used for playback and export."""
        selection = self._selection
        here = selection[selection["iteration"] == iteration]

        if self.show_trail.isChecked():
            earlier = selection[selection["iteration"] < iteration]
            self._trail.set_offsets(
                earlier[["x0", "x1"]].to_numpy() if len(earlier) else np.empty((0, 2))
            )
        else:
            self._trail.set_offsets(np.empty((0, 2)))

        if self._cluster_scatter is not None:
            self._cluster_scatter.set_offsets(here[["x0", "x1"]].to_numpy())
            self._cluster_scatter.set_array(here["cluster"].to_numpy(dtype=float))
        else:
            for algorithm, artist in self._scatters.items():
                rows = here[here["algorithm"] == algorithm]
                artist.set_offsets(
                    rows[["x0", "x1"]].to_numpy() if len(rows) else np.empty((0, 2))
                )

        low, high = self._iterations
        self.iteration_label.setText(f"iteration {iteration} / {high}")

        title = (
            f"F{self._function}_I{self._instance}, run {self._run} - "
            f"iteration {iteration} of {high}"
        )
        if self._cluster_scatter is not None:
            # In cluster mode the colours are the only thing on screen that
            # carries meaning, so say how many there are to read.
            occupied = here["cluster"].nunique()
            total = self._selection["cluster"].nunique()
            title += f"\npopulation spread over {occupied} of {total} clusters"
        self.axes.set_title(title)
        return list(self._scatters.values()) + [self._trail]

    # -- playback --------------------------------------------------------

    def toggle_play(self):
        if self.timer.isActive():
            self.stop()
        else:
            if self.slider.value() >= self.slider.maximum():
                self.slider.setValue(self.slider.minimum())
            self.timer.start(int(1000 / self.fps.value()))
            self.play_button.setText("Pause")

    def stop(self):
        self.timer.stop()
        self.play_button.setText("Play")

    def _retime(self):
        if self.timer.isActive():
            self.timer.start(int(1000 / self.fps.value()))

    def _advance(self):
        if self.slider.value() >= self.slider.maximum():
            self.stop()          # stop at the end rather than looping silently
            return
        self.slider.setValue(self.slider.value() + 1)

    # -- export ----------------------------------------------------------

    def _export_gif(self):
        if self._selection is None:
            return
        self.stop()
        suggested = str(
            Path.home() / f"trajectory_F{self._function}_I{self._instance}_run{self._run}.gif"
        )
        target, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export animation", suggested, "GIF (*.gif)"
        )
        if not target:
            return

        from matplotlib.animation import FuncAnimation, PillowWriter

        low, high = self._iterations
        QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            animation = FuncAnimation(
                self.figure, self._update_frame,
                frames=range(low, high + 1), interval=1000 / self.fps.value(),
            )
            animation.save(target, writer=PillowWriter(fps=self.fps.value()))
            self.status.setText(f"Saved {high - low + 1} frames to {target}")
        except Exception as exc:
            self.status.setText(f"Could not export: {exc}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            self._draw_current()
