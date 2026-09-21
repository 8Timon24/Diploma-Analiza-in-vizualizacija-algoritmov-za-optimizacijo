"""Pick a visualization, set its parameters, render it, export it.

The parameter form is generated from each Visualization's declared
parameters, so nothing here knows about entropy or Spearman specifically -
adding a catalog entry is enough to get a working form.
"""
import re
from pathlib import Path

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT

from gui import icons, theme
from gui.panels import layout as panel_layout
from gui.qt import QtWidgets, Qt, Signal
from gui.panels.widgets import CheckableList
from gui.viz.registry import catalog

_VIZ_ROLE = Qt.ItemDataRole.UserRole


class VizPanel(QtWidgets.QWidget):
    # asks the window to prefill the run form with what is missing
    generateRequested = Signal(dict)
    # asks the window to open the Process tab with these stages ticked
    processRequested = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._catalog = catalog()
        self._inputs = {}
        self._figure = None
        self._rendered_key = None
        self._canvas = None
        self._toolbar = None

        # -- catalog list
        self.list = QtWidgets.QListWidget()
        for viz in self._catalog:
            item = QtWidgets.QListWidgetItem(viz.title)
            item.setData(_VIZ_ROLE, viz)
            item.setToolTip(viz.description)
            self.list.addItem(item)
        self._refresh_availability()
        self.list.currentItemChanged.connect(self._on_viz_changed)

        # -- description + form
        self.description = QtWidgets.QLabel()
        self.description.setWordWrap(True)
        self.source = QtWidgets.QLabel()
        self.source.setWordWrap(True)
        self.source.setProperty("class", "hint")

        self.form_host = QtWidgets.QWidget()
        self.form = QtWidgets.QFormLayout(self.form_host)
        self.form.setContentsMargins(0, 4, 0, 4)

        self.render_button = QtWidgets.QPushButton(
            icons.icon("chart-column", theme.tokens()["accent_text"]), "Render")
        self.render_button.setProperty("class", "primary")
        self.render_button.setDefault(True)
        self.render_button.setShortcut("Ctrl+R")
        self.render_button.setToolTip("Draw this visualization (Ctrl+R)")
        self.render_button.clicked.connect(self._render)
        self.export_button = QtWidgets.QPushButton("Export figure...")
        self.export_button.setEnabled(False)
        self.export_button.setShortcut("Ctrl+E")
        self.export_button.setToolTip("Save the figure as PDF, PNG or SVG (Ctrl+E)")
        self.export_button.clicked.connect(self._export)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self.render_button)
        buttons.addWidget(self.export_button)

        controls = QtWidgets.QWidget()
        controls_layout = QtWidgets.QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, panel_layout.S, 0)
        controls_layout.setSpacing(panel_layout.S)
        controls_layout.addWidget(self.description)
        controls_layout.addWidget(self.form_host)
        controls_layout.addLayout(buttons)
        controls_layout.addWidget(self.source)
        controls_layout.addStretch()

        # -- figure area
        self.figure_host = QtWidgets.QWidget()
        self.figure_layout = QtWidgets.QVBoxLayout(self.figure_host)
        self.figure_layout.setContentsMargins(0, 0, 0, 0)
        self.status = QtWidgets.QLabel("Select a visualization, then press Render.")
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status.setProperty("class", "hint")
        self.figure_layout.addWidget(self.status)

        # Proportions, not pixel caps: a hard setMaximumWidth made the
        # splitter handle look broken - it dragged and the pane refused
        # to grow.
        splitter = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.list)
        splitter.addWidget(controls)
        splitter.addWidget(self.figure_host)
        for index, stretch in ((0, 0), (1, 0), (2, 1)):
            splitter.setStretchFactor(index, stretch)
        splitter.setSizes([240, 300, 760])
        splitter.setChildrenCollapsible(False)

        layout = panel_layout.column(self)
        layout.addWidget(splitter)

        self.list.setEnabled(False)

    def refresh_icons(self):
        """See setup_panel.SetupPanel.refresh_icons."""
        self.render_button.setIcon(
            icons.icon("chart-column", theme.tokens()["accent_text"]))
        if self._toolbar is not None:
            icons.restyle_toolbar(self._toolbar)

    def activate(self):
        """Select the first visualization. Called once the optimizer registry
        is warm, so resolving the algorithm choices does not block the UI."""
        self.list.setEnabled(True)
        if self.list.currentRow() < 0 and self.list.count():
            self.list.setCurrentRow(self._first_enabled_row())

    def _first_enabled_row(self):
        """Selecting a disabled row builds a form for a visualization that
        cannot render, so Render fails on a panel the user never touched."""
        for row in range(self.list.count()):
            if self.list.item(row).flags() & Qt.ItemFlag.ItemIsEnabled:
                return row
        return 0

    def _refresh_availability(self):
        """Re-evaluate which entries have data behind them.

        This has to run on every refresh, not once in __init__: a benchmark
        run creates data for entries that were greyed out at startup, and the
        main window builds this panel before it restores the saved results
        root, so the very first answer is against the wrong tree.
        """
        for row in range(self.list.count()):
            item = self.list.item(row)
            viz = item.data(_VIZ_ROLE)
            available = viz.is_available()
            flags = item.flags()
            if available:
                item.setFlags(flags | Qt.ItemFlag.ItemIsEnabled)
                item.setText(viz.title)
            else:
                item.setFlags(flags & ~Qt.ItemFlag.ItemIsEnabled)
                item.setText(f"{viz.title}  (no data)")

    # -- form ------------------------------------------------------------

    @property
    def current(self):
        item = self.list.currentItem()
        return item.data(_VIZ_ROLE) if item else None

    def refresh(self):
        """Rebuild the current form so its choices reflect a new results
        folder (available dimensions, which entries have data, and so on)."""
        self._refresh_availability()
        if self.current is not None:
            self._build_form(self.current)
        self._show_message("Results folder changed - press Render.")
        self.export_button.setEnabled(False)

    def _on_viz_changed(self, current, _previous):
        viz = current.data(_VIZ_ROLE) if current else None
        if viz is None:
            # Clear the form too: keeping it meant _collect() could later read
            # widgets belonging to an entry that is no longer selected.
            self._clear_form()
            self.description.clear()
            self.source.clear()
            return
        self.description.setText(viz.description)
        self.source.setText(f"Renders: {viz.source}")
        self._build_form(viz)
        self._mark_stale()

    def _mark_stale(self):
        """The figure on screen no longer matches the form.

        Export used to stay enabled here, so it would happily save a figure
        that was rendered from different parameters than the ones displayed.
        """
        if self._figure is None:
            return
        self.export_button.setEnabled(False)
        self.status.setProperty("class", "hint")
        theme.restyle(self.status)
        self.status.setText("Parameters changed - press Render to update.")
        # _clear_figure_area unparents self.status to keep it alive across
        # renders, and _show_figure does not put it back - so this message
        # was being written to a hidden, parentless widget and could never
        # be read. Show it above the stale figure instead.
        if self.status.parent() is None:
            self.figure_layout.insertWidget(0, self.status)
        self.status.show()

    def _clear_form(self):
        while self.form.rowCount():
            self.form.removeRow(0)
        self._inputs = {}

    def _build_form(self, viz):
        self._clear_form()

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

            # Without this the figure was only ever marked stale when the
            # CATALOG selection changed, so editing the form after a render
            # left Export enabled on a figure built from other parameters -
            # exactly what _mark_stale exists to prevent.
            # Connected after the default is applied above, so building the
            # form does not itself mark the figure stale. The lambdas drop
            # each signal's argument; _mark_stale takes none.
            if parameter.kind == "bool":
                widget.toggled.connect(lambda *_: self._mark_stale())
            elif parameter.kind == "multi":
                widget.itemChanged.connect(lambda *_: self._mark_stale())
            else:
                widget.currentIndexChanged.connect(lambda *_: self._mark_stale())

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
        # processEvents() below pumps the event loop so the "Rendering..."
        # message actually paints. That also lets clicks through, so the
        # catalog has to be frozen too - otherwise the user can select a
        # different entry mid-render and the finished figure is shown against
        # someone else's description, form and export filename.
        self.list.setEnabled(False)
        QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QtWidgets.QApplication.processEvents()
        try:
            figure = viz.draw(params)
        except Exception as exc:
            self._show_error(f"{type(exc).__name__}: {exc}")
            self._figure = None
            self.export_button.setEnabled(False)
        else:
            self._show_figure(figure)
            self._rendered_key = viz.key
            self.export_button.setEnabled(True)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            self.render_button.setEnabled(True)
            self.list.setEnabled(True)

    def _clear_figure_area(self):
        while self.figure_layout.count():
            widget = self.figure_layout.takeAt(0).widget()
            if widget is None:
                continue
            widget.setParent(None)
            # self.status is reused across renders, so it must survive; the
            # canvases and toolbars are per-figure and were only being
            # unparented, leaving them alive until the next Python GC.
            if widget is not self.status:
                widget.deleteLater()
        if self._canvas is not None:
            self._canvas.figure.clear()
        self._canvas = None
        self._toolbar = None

    def _show_message(self, text):
        self._clear_figure_area()
        self.status.setProperty("class", "hint")
        theme.restyle(self.status)
        self.status.setText(text)
        self.figure_layout.addWidget(self.status)
        self.status.show()

    def _show_error(self, text):
        """Same slot as _show_message, but styled as a failure.

        An exception rendered in the same grey as "Select a visualization"
        is indistinguishable from the ordinary placeholder.
        """
        self._show_message(text)
        self.status.setProperty("class", "error")
        theme.restyle(self.status)

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
        heading.setProperty("class", "heading")
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
            caveat.setProperty("class", "callout")
            layout.addWidget(caveat)

        if coverage.recipe:
            layout.addWidget(QtWidgets.QLabel("Steps that would generate it:"))
            commands = QtWidgets.QPlainTextEdit("\n".join(coverage.recipe))
            commands.setReadOnly(True)
            commands.setMaximumHeight(120)
            commands.setFont(theme.monospace_font(commands))
            layout.addWidget(commands)

            row = QtWidgets.QHBoxLayout()

            # Stages other than the benchmark can be run here directly. The
            # benchmark still needs the Setup form, so a gap that includes it
            # offers both buttons and the user picks the order.
            runnable = [name for name in coverage.stages if name != "benchmark"]
            if runnable:
                process = QtWidgets.QPushButton(
                    icons.icon("workflow", theme.tokens()["accent_text"]),
                    "Run these steps now...")
                process.setProperty("class", "primary")
                process.setToolTip(
                    "Opens the Process tab with exactly these stages ticked. "
                    "They overwrite data/ and metrics_data/ in place, and you "
                    "confirm before anything runs."
                )
                process.clicked.connect(
                    lambda _checked=False, names=tuple(runnable):
                        self.processRequested.emit(list(names))
                )
                row.addWidget(process)

            if coverage.wanted_algorithms or coverage.wanted_dimensions:
                generate = QtWidgets.QPushButton("Benchmark the missing runs...")
                generate.setToolTip(
                    "Prefills the Setup tab with exactly what is missing. "
                    "A benchmark writes to its own workspace, so it never "
                    "overwrites the results already in the repo."
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
        icons.restyle_toolbar(self._toolbar)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(self._canvas)
        scroll.setWidgetResizable(True)
        # Without a minimum, setWidgetResizable squashes a tall figure (the
        # occupancy heatmaps are sized per iteration) into the viewport
        # instead of scrolling it. The compare panel has always done this.
        width, height = figure.get_size_inches() * figure.dpi
        self._canvas.setMinimumSize(int(width * 0.75), int(height * 0.75))
        self.figure_layout.addWidget(self._toolbar)
        self.figure_layout.addWidget(scroll, 1)
        self._canvas.draw_idle()

    # -- export ----------------------------------------------------------

    def _export(self):
        if self._figure is None:
            return
        # Name the file after the visualization that produced the figure on
        # screen, which is not necessarily the one selected now.
        key = self._rendered_key or (self.current.key if self.current else "figure")
        suggested = str(Path.home() / f"{key}.pdf")
        target, selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export figure", suggested,
            "PDF (*.pdf);;PNG (*.png);;SVG (*.svg)",
        )
        if not target:
            return
        target = _with_extension(target, selected_filter)
        QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self._figure.savefig(target, bbox_inches="tight", dpi=200)
        except (OSError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(
                self, "Export failed", f"Could not write {target}:\n\n{exc}"
            )
        else:
            self.source.setText(f"Exported to {target}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()


def _with_extension(target, selected_filter):
    """Honour the format chosen in the dialog when the name has no suffix.

    Without this a filename typed without an extension is written in
    matplotlib's default format regardless of the filter the user picked.
    """
    path = Path(target)
    if path.suffix:
        return target
    match = re.search(r"\*(\.\w+)", selected_filter or "")
    return str(path.with_suffix(match.group(1))) if match else target
