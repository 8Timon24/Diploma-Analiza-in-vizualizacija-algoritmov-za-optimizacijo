"""Head-to-head: two algorithms, all six metrics, and what they actually did.

Built as its own panel rather than three catalog entries because the whole
point is seeing the three views of one pair together - having to re-pick the
same two algorithms in three separate forms would defeat it.
"""
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT

from gui.qt import QtWidgets, Qt
from gui.panels.widgets import CheckableList  # noqa: F401  (kept for parity)
from gui.viz import compare

import config

# Percentile spread above which the metrics are meaningfully disagreeing
# about the pair. Below it they are telling the same story.
DISAGREEMENT_SPREAD = 40.0
SIMILAR_PERCENTILE = 25.0
DIFFERENT_PERCENTILE = 75.0


class ComparePanel(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._canvases = {}

        controls = self._build_controls()

        self.headline = QtWidgets.QLabel("Pick two algorithms and press Compare.")
        self.headline.setWordWrap(True)
        font = self.headline.font()
        font.setPointSize(font.pointSize() + 1)
        self.headline.setFont(font)

        self.detail = QtWidgets.QLabel()
        self.detail.setWordWrap(True)
        self.detail.setStyleSheet("color: palette(mid);")

        self.views = QtWidgets.QTabWidget()
        self._hosts = {}
        for key, label in (
            ("profile", "Metric profile"),
            ("per_function", "Per function"),
            ("behaviour", "Behaviour"),
        ):
            host = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(host)
            layout.setContentsMargins(0, 0, 0, 0)
            self._hosts[key] = layout
            self.views.addTab(host, label)

        right = QtWidgets.QVBoxLayout()
        right.setContentsMargins(8, 0, 0, 0)
        right.addWidget(self.headline)
        right.addWidget(self.detail)
        right.addWidget(self.views, 1)
        right_host = QtWidgets.QWidget()
        right_host.setLayout(right)

        splitter = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(controls)
        splitter.addWidget(right_host)
        splitter.setStretchFactor(1, 1)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

    def _build_controls(self):
        box = QtWidgets.QGroupBox("Compare")
        form = QtWidgets.QFormLayout(box)
        box.setMaximumWidth(320)

        self.algorithm_a = QtWidgets.QComboBox()
        self.algorithm_b = QtWidgets.QComboBox()
        for combo, default in ((self.algorithm_a, "OriginalDE"),
                               (self.algorithm_b, "WhaleFOA")):
            for name in config.ALGORITHMS_OF_INTEREST:
                combo.addItem(name, name)
            index = combo.findData(default)
            if index >= 0:
                combo.setCurrentIndex(index)

        self.dimension = QtWidgets.QComboBox()
        for d in config.DIMENSIONS:
            self.dimension.addItem(str(d), d)

        self.function = QtWidgets.QComboBox()
        for f in config.FUNCTIONS:
            self.function.addItem(f"F{f}", f)

        self.instance = QtWidgets.QComboBox()
        for i in config.INSTANCES:
            self.instance.addItem(str(i), i)

        self.seeds = QtWidgets.QComboBox()
        self.seeds.addItem("all runs", list(config.SEEDS))
        for s in config.SEEDS:
            self.seeds.addItem(f"run {s}", [s])

        self.compare_button = QtWidgets.QPushButton("Compare")
        self.compare_button.clicked.connect(self.compare)

        note = QtWidgets.QLabel(
            "Function, instance and runs apply to the Behaviour view only; "
            "the other two average over every problem."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: palette(mid); font-size: 11px;")

        form.addRow("Algorithm A:", self.algorithm_a)
        form.addRow("Algorithm B:", self.algorithm_b)
        form.addRow("Dimension:", self.dimension)
        form.addRow("Function:", self.function)
        form.addRow("Instance:", self.instance)
        form.addRow("Runs:", self.seeds)
        form.addRow("", self.compare_button)
        form.addRow("", note)
        return box

    def set_registry(self, registry):
        """Offer every optimizer that has pairwise metrics, plus any the user
        has since benchmarked."""
        for combo in (self.algorithm_a, self.algorithm_b):
            current = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            for name in registry.names:
                combo.addItem(name, name)
            index = combo.findData(current)
            combo.setCurrentIndex(max(index, 0))
            combo.blockSignals(False)

    # -- comparison ------------------------------------------------------

    def _params(self):
        return {
            "dimension": self.dimension.currentData(),
            "algorithm_a": self.algorithm_a.currentData(),
            "algorithm_b": self.algorithm_b.currentData(),
            "function": self.function.currentData(),
            "instance": self.instance.currentData(),
            "seeds": self.seeds.currentData(),
        }

    def compare(self):
        params = self._params()
        if params["algorithm_a"] == params["algorithm_b"]:
            self.headline.setText("Pick two different algorithms.")
            self.detail.clear()
            return

        QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self._summarise(params)
            renderers = {
                "profile": compare.metric_profile,
                "per_function": compare.per_function_profile,
                "behaviour": compare.behaviour,
            }
            for key, render in renderers.items():
                self._show(key, render, params)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def _show(self, key, render, params):
        """Replace the whole canvas rather than swapping the figure into it.

        Assigning canvas.figure leaves BOTH figures pointing at the same
        canvas, so a draw_idle() queued by the previous render repaints the
        old figure over the new one - which shows up as overlaid, mixed-up
        plots after comparing a few times. It also leaves the navigation
        toolbar's view stack pointing at axes that no longer exist. Building
        a fresh canvas each time makes both impossible.
        """
        try:
            figure = render(params)
        except Exception as exc:
            figure = _message_figure(f"{type(exc).__name__}: {exc}")

        layout = self._hosts[key]
        old = self._canvases.pop(key, None)
        if old is not None:
            old.figure.clear()
            old.setParent(None)
            old.deleteLater()
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

        canvas = FigureCanvasQTAgg(figure)
        toolbar = NavigationToolbar2QT(canvas, self)
        # A tall figure (six stacked metric rows) must not be silently cut
        # off at the bottom of the viewport.
        width, height = figure.get_size_inches() * figure.dpi
        canvas.setMinimumSize(int(width * 0.75), int(height * 0.75))
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(canvas)
        scroll.setWidgetResizable(True)

        layout.addWidget(toolbar)
        layout.addWidget(scroll, 1)
        self._canvases[key] = canvas
        canvas.draw_idle()

    def _summarise(self, params):
        """State the finding in a sentence instead of leaving it to be read
        off six axes."""
        try:
            table, total = compare.summary_table(
                params["dimension"], params["algorithm_a"], params["algorithm_b"]
            )
        except Exception as exc:
            self.headline.setText(str(exc))
            self.detail.clear()
            return

        first, second = params["algorithm_a"], params["algorithm_b"]
        percentiles = table["percentile"]
        spread = float(percentiles.max() - percentiles.min())

        if spread > DISAGREEMENT_SPREAD:
            lowest = table.loc[percentiles.idxmin()]
            highest = table.loc[percentiles.idxmax()]
            verdict = (
                f"The metrics disagree about {first} and {second}: "
                f"{highest['metric']} puts them at the {highest['percentile']:.0f}th "
                f"percentile of all {total} pairs, while {lowest['metric']} "
                f"puts them at the {lowest['percentile']:.0f}th."
            )
        elif percentiles.mean() < SIMILAR_PERCENTILE:
            verdict = (
                f"Every metric agrees {first} and {second} are unusually "
                f"similar - all within the {percentiles.max():.0f}th percentile "
                f"of {total} pairs."
            )
        elif percentiles.mean() > DIFFERENT_PERCENTILE:
            verdict = (
                f"Every metric agrees {first} and {second} are unusually "
                f"different - all above the {percentiles.min():.0f}th percentile "
                f"of {total} pairs."
            )
        else:
            verdict = (
                f"{first} and {second} sit mid-distribution on every metric "
                f"(percentiles {percentiles.min():.0f}-{percentiles.max():.0f} "
                f"of {total} pairs)."
            )
        self.headline.setText(verdict)
        self.detail.setText(
            "   ".join(
                f"{row['metric']}: {row['percentile']:.0f}%"
                for _, row in table.iterrows()
            )
        )


def _message_figure(text):
    from matplotlib.figure import Figure

    figure = Figure(figsize=(8, 5), layout="constrained")
    axes = figure.add_subplot(1, 1, 1)
    axes.axis("off")
    axes.text(0.5, 0.5, text, ha="center", va="center", wrap=True, fontsize=10,
              color="#b91c1c")
    return figure
