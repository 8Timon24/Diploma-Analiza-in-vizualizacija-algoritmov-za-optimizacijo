"""Pick a visualization, set its parameters, render it, export it.

The parameter form is generated from each Visualization's declared
parameters, so nothing here knows about entropy or Spearman specifically -
adding a catalog entry is enough to get a working form.
"""
from pathlib import Path

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT

from gui.qt import QtWidgets, Qt, Signal
from gui.panels.widgets import CheckableList, VALUE_ROLE
from gui.viz.registry import catalog

_VIZ_ROLE = Qt.ItemDataRole.UserRole


class VizPanel(QtWidgets.QWidget):
    # asks the window to prefill the run form with what is missing
    generateRequested = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._catalog = catalog()
        self._inputs = {}
        self._figure = None
        self._canvas = None
        self._toolbar = None

        # -- catalog list
        self.list = QtWidgets.QListWidget()
        self.list.setMaximumWidth(280)
        for viz in self._catalog:
            item = QtWidgets.QListWidgetItem(viz.title)
            item.setData(_VIZ_ROLE, viz)
            item.setToolTip(viz.description)
            if not viz.is_available():
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
                item.setText(f"{viz.title}  (no data)")
            self.list.addItem(item)
        self.list.currentItemChanged.connect(self._on_viz_changed)

        # -- description + form
        self.description = QtWidgets.QLabel()
        self.description.setWordWrap(True)
        self.source = QtWidgets.QLabel()
        self.source.setWordWrap(True)
        self.source.setStyleSheet("color: palette(mid); font-size: 11px;")

        self.form_host = QtWidgets.QWidget()
        self.form = QtWidgets.QFormLayout(self.form_host)
        self.form.setContentsMargins(0, 4, 0, 4)

        self.render_button = QtWidgets.QPushButton("Render")
        self.render_button.clicked.connect(self._render)
        self.export_button = QtWidgets.QPushButton("Export figure...")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._export)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self.render_button)
        buttons.addWidget(self.export_button)

        controls = QtWidgets.QWidget()
        controls_layout = QtWidgets.QVBoxLayout(controls)
        controls_layout.setContentsMargins(8, 0, 8, 0)
        controls_layout.addWidget(self.description)
        controls_layout.addWidget(self.form_host)
        controls_layout.addLayout(buttons)
        controls_layout.addWidget(self.source)
        controls_layout.addStretch()
        controls.setMaximumWidth(340)

        # -- figure area
        self.figure_host = QtWidgets.QWidget()
        self.figure_layout = QtWidgets.QVBoxLayout(self.figure_host)
        self.figure_layout.setContentsMargins(0, 0, 0, 0)
        self.status = QtWidgets.QLabel("Select a visualization, then press Render.")
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status.setStyleSheet("color: palette(mid);")
        self.figure_layout.addWidget(self.status)

        splitter = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.list)
        splitter.addWidget(controls)
        splitter.addWidget(self.figure_host)
        splitter.setStretchFactor(2, 1)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        self.list.setEnabled(False)

    def activate(self):
        """Select the first visualization. Called once the optimizer registry
        is warm, so resolving the algorithm choices does not block the UI."""
        self.list.setEnabled(True)
        if self.list.currentRow() < 0 and self.list.count():
            self.list.setCurrentRow(0)

    # -- form ------------------------------------------------------------

    @property
    def current(self):
        item = self.list.currentItem()
        return item.data(_VIZ_ROLE) if item else None

    def refresh(self):
        """Rebuild the current form so its choices reflect a new results
        folder (available dimensions, which entries have data, and so on)."""
        self._build_form(self.current) if self.current else None
        self._show_message("Results folder changed - press Render.")
        self.export_button.setEnabled(False)

    def _on_viz_changed(self, current, _previous):
        viz = current.data(_VIZ_ROLE) if current else None
        if viz is None:
            return
        self.description.setText(viz.description)
        self.source.setText(f"Renders: {viz.source}")
        self._build_form(viz)

    def _build_form(self, viz):
        while self.form.rowCount():
            self.form.removeRow(0)
        self._inputs = {}

        for parameter in viz.parameters:
            choices = parameter.resolve_choices()
            default = parameter.resolve_default()

            if parameter.kind == "bool":
                widget = QtWidgets.QCheckBox()
                widget.setChecked(bool(default))
            elif parameter.kind == "multi":
                widget = CheckableList(choices, default)
            else:
                widget = QtWidgets.QComboBox()
                for choice in choices:
                    widget.addItem(str(choice), choice)
                if default is not None:
                    index = widget.findData(default)
                    if index >= 0:
                        widget.setCurrentIndex(index)

            if parameter.help:
                widget.setToolTip(parameter.help)
            self._inputs[parameter.name] = (parameter, widget)
            self.form.addRow(f"{parameter.label}:", widget)

    def _collect(self):
        values = {}
        for name, (parameter, widget) in self._inputs.items():
            if parameter.kind == "bool":
                values[name] = widget.isChecked()
            elif parameter.kind == "multi":
                values[name] = widget.values()
            else:
                values[name] = widget.currentData()
        return values

    # -- rendering -------------------------------------------------------

    def _render(self):
        viz = self.current
        if viz is None:
            return

        params = self._collect()
        coverage = viz.check_coverage(params)
        if not coverage.ok:
            self._show_missing(coverage)
            self.export_button.setEnabled(False)
            return

        self._show_message(f"Rendering {viz.title}...")
        self.render_button.setEnabled(False)
        QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QtWidgets.QApplication.processEvents()
        try:
            figure = viz.render(params)
        except Exception as exc:
            self._show_message(f"{type(exc).__name__}: {exc}")
            self.export_button.setEnabled(False)
        else:
            self._show_figure(figure)
            self.export_button.setEnabled(True)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            self.render_button.setEnabled(True)

    def _clear_figure_area(self):
        while self.figure_layout.count():
            widget = self.figure_layout.takeAt(0).widget()
            if widget is not None:
                widget.setParent(None)
        self._canvas = None
        self._toolbar = None

    def _show_message(self, text):
        self._clear_figure_area()
        self.status.setText(text)
        self.figure_layout.addWidget(self.status)
        self.status.show()

    def _show_missing(self, coverage):
        """Explain exactly what is absent and how to produce it.

        Selecting un-benchmarked algorithms or dimensions is allowed on
        purpose, so this is a normal state rather than an error.
        """
        self._clear_figure_area()

        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(10)

        heading = QtWidgets.QLabel("This has not been computed yet")
        font = heading.font()
        font.setBold(True)
        font.setPointSize(font.pointSize() + 2)
        heading.setFont(font)
        layout.addWidget(heading)

        missing = QtWidgets.QLabel(coverage.summary())
        missing.setWordWrap(True)
        layout.addWidget(missing)

        if coverage.estimate:
            estimate = QtWidgets.QLabel(f"Estimated cost:  {coverage.estimate}")
            estimate.setWordWrap(True)
            layout.addWidget(estimate)

        if coverage.caveat:
            caveat = QtWidgets.QLabel(coverage.caveat)
            caveat.setWordWrap(True)
            caveat.setStyleSheet(
                "background: palette(alternate-base); padding: 8px;"
                " border-left: 3px solid palette(highlight);"
            )
            layout.addWidget(caveat)

        if coverage.recipe:
            layout.addWidget(QtWidgets.QLabel("Steps that would generate it:"))
            commands = QtWidgets.QPlainTextEdit("\n".join(coverage.recipe))
            commands.setReadOnly(True)
            commands.setMaximumHeight(120)
            font = commands.font()
            font.setFamily("monospace")
            commands.setFont(font)
            layout.addWidget(commands)

            row = QtWidgets.QHBoxLayout()
            if coverage.wanted_algorithms or coverage.wanted_dimensions:
                generate = QtWidgets.QPushButton("Generate missing data...")
                generate.setToolTip(
                    "Prefills the Setup tab with exactly what is missing. "
                    "Only the benchmark step runs in the app; the clustering "
                    "and metric steps are still run from the command line."
                )
                generate.clicked.connect(
                    lambda: self.generateRequested.emit({
                        "algorithms": list(coverage.wanted_algorithms),
                        "dimensions": list(coverage.wanted_dimensions),
                        "functions": list(coverage.wanted_functions),
                    })
                )
                row.addWidget(generate)

            copy_button = QtWidgets.QPushButton("Copy commands")
            copy_button.clicked.connect(
                lambda: QtWidgets.QApplication.clipboard().setText(
                    "\n".join(coverage.recipe)
                )
            )
            row.addWidget(copy_button)
            row.addStretch()
            layout.addLayout(row)

        layout.addStretch()
        self.figure_layout.addWidget(panel)
        self.status.hide()

    def _show_figure(self, figure):
        self._clear_figure_area()
        self._figure = figure
        self._canvas = FigureCanvasQTAgg(figure)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(self._canvas)
        scroll.setWidgetResizable(True)
        self.figure_layout.addWidget(self._toolbar)
        self.figure_layout.addWidget(scroll, 1)
        self._canvas.draw_idle()

    # -- export ----------------------------------------------------------

    def _export(self):
        if self._figure is None:
            return
        viz = self.current
        suggested = str(Path.home() / f"{viz.key}.pdf")
        target, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export figure", suggested,
            "PDF (*.pdf);;PNG (*.png);;SVG (*.svg)",
        )
        if not target:
            return
        QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self._figure.savefig(target, bbox_inches="tight", dpi=200)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
