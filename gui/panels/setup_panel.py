"""Experiment setup: what to run, on what, at what cost, and where it lands.

The cost estimate and the output destination are the two things this panel
exists to make impossible to miss. A full sweep is a multi-hour job, and
run_benchmarks writes by algorithm and problem id - so pointing a run at the
repo's real outputs/ can overwrite existing trajectories in place. Sandbox is
the default and switching away from it requires confirming.
"""
from gui.panels import layout as panel_layout
from gui.qt import QtWidgets, Qt, Signal
from gui.core import coverage as coverage_core
from gui.core import problems as problems_core
from gui.core import workspace as workspace_core
from gui.core.workspace import RunSpec, format_duration
from gui.panels.algorithm_picker import AlgorithmPicker
from gui.panels.widgets import CheckableList

import config


# Estimated seconds past which a run is announced up front and confirmed.
LONG_RUN_SECONDS = 600


def _with_note(widget, note):
    """Stack a control and its explanation into one form field."""
    host = QtWidgets.QWidget()
    inner = panel_layout.column(host, margins=0, spacing=2)
    inner.addWidget(widget)
    inner.addWidget(note)
    return host


class SetupPanel(QtWidgets.QWidget):
    selectionChanged = Signal(list)
    runRequested = Signal(object, bool)  # RunSpec, write_to_real_outputs

    def __init__(self, parent=None):
        super().__init__(parent)

        self.picker = AlgorithmPicker()
        self.picker.selectionChanged.connect(self.selectionChanged)
        self.picker.selectionChanged.connect(lambda _names: self._refresh_estimate())

        optimizers_box = QtWidgets.QGroupBox("Optimizers")
        QtWidgets.QVBoxLayout(optimizers_box).addWidget(self.picker)

        right = panel_layout.column(margins=(0, 0, panel_layout.S, 0))
        right.addWidget(self._build_problem_box())
        right.addWidget(self._build_parameter_box())
        right.addWidget(self._build_output_box())
        right.addWidget(self._build_run_box())
        right.addStretch()

        right_host = QtWidgets.QWidget()
        right_host.setLayout(right)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(right_host)
        scroll.setWidgetResizable(True)

        # Proportions, not pixel caps: a hard setMaximumWidth made the
        # splitter handle look broken - it dragged and the pane refused
        # to grow.
        splitter = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(optimizers_box)
        splitter.addWidget(scroll)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 940])
        splitter.setChildrenCollapsible(False)

        layout = panel_layout.column(self)
        layout.addWidget(splitter)

        # Only now that every widget exists: _on_source_changed populates the
        # function list, which costs an estimate that reads the parameter
        # spinboxes built after the problem box.
        self._on_source_changed()

    # -- construction ----------------------------------------------------

    def _build_problem_box(self):
        box = QtWidgets.QGroupBox("Benchmark problems")
        form = panel_layout.form(box)

        self.source_combo = QtWidgets.QComboBox()
        self.source_combo.addItem("BBOB (COCO) - 24 functions, 5 instances",
                                  {"kind": problems_core.BBOB_KEY})
        self.source_combo.addItem("opfunu - classic functions",
                                  {"kind": problems_core.OPFUNU_KEY})
        for year in problems_core.cec_years():
            self.source_combo.addItem(
                f"CEC {year.replace('cec', '')} competition suite",
                {"kind": problems_core.CEC_KEY, "year": year},
            )
        self.source_combo.addItem("User-defined function",
                                  {"kind": problems_core.CUSTOM_KEY})
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)

        self.source_note = QtWidgets.QLabel()
        self.source_note.setWordWrap(True)
        self.source_note.setProperty("class", "hint")

        self.functions = CheckableList(list(config.FUNCTIONS), [1, 2])
        self.function_note = QtWidgets.QLabel()
        self.function_note.setWordWrap(True)
        self.function_note.setProperty("class", "hint")

        self.instances = CheckableList(list(config.INSTANCES), [1], rows=5)
        self.instance_note = QtWidgets.QLabel()
        self.instance_note.setWordWrap(True)
        self.instance_note.setProperty("class", "hint")
        self.dimensions = CheckableList(coverage_core.BENCHMARK_DIMENSIONS, [2], rows=4)
        self.seeds = CheckableList(list(config.SEEDS), [1], rows=4)

        self.dimensions.itemChanged.connect(lambda _item: self._refresh_functions())
        for widget in (self.functions, self.instances, self.seeds):
            widget.itemChanged.connect(lambda _item: self._refresh_estimate())

        # Each note goes under the control it explains, in the same field
        # cell. Adding it as its own row with an empty label left a column of
        # blank label cells and spread the fields a screenful apart.
        form.addRow("Source:", _with_note(self.source_combo, self.source_note))
        form.addRow("", self._build_custom_box())
        form.addRow("Functions:", _with_note(self.functions, self.function_note))
        form.addRow("Instances:", _with_note(self.instances, self.instance_note))
        form.addRow("Dimensions:", self.dimensions)
        form.addRow("Seeds:", self.seeds)
        return box

    def _build_custom_box(self):
        self.custom_box = QtWidgets.QGroupBox("Your function")
        form = panel_layout.form(self.custom_box)

        self.custom_expression = QtWidgets.QLineEdit("sum(x**2)")
        self.custom_expression.setToolTip(
            "A Python expression in x, the numpy array of decision variables. "
            "It must reduce x to a single number: sum(x**2), not x**2.\n"
            "numpy functions are available; builtins and imports are not."
        )
        self.custom_lower = QtWidgets.QDoubleSpinBox()
        self.custom_lower.setRange(-1e9, 1e9)
        self.custom_lower.setValue(-5.12)
        self.custom_upper = QtWidgets.QDoubleSpinBox()
        self.custom_upper.setRange(-1e9, 1e9)
        self.custom_upper.setValue(5.12)

        self.custom_status = QtWidgets.QLabel()
        self.custom_status.setWordWrap(True)
        check = QtWidgets.QPushButton("Check expression")
        check.clicked.connect(self._check_expression)

        bounds = QtWidgets.QHBoxLayout()
        bounds.addWidget(self.custom_lower)
        bounds.addWidget(QtWidgets.QLabel("to"))
        bounds.addWidget(self.custom_upper)

        form.addRow("f(x) =", self.custom_expression)
        form.addRow("Bounds:", bounds)
        form.addRow("", check)
        form.addRow("", self.custom_status)
        self.custom_box.hide()
        return self.custom_box

    # -- problem source --------------------------------------------------

    def source_config(self):
        cfg = dict(self.source_combo.currentData() or {"kind": problems_core.BBOB_KEY})
        if cfg["kind"] == problems_core.CUSTOM_KEY:
            cfg.update({
                "expression": self.custom_expression.text().strip(),
                "lower": self.custom_lower.value(),
                "upper": self.custom_upper.value(),
                "name": "Custom",
            })
        return cfg

    def _current_source(self):
        # source_from_config also raises bare ValueError for an unknown kind
        # and KeyError for a malformed custom config, and this runs inside a
        # currentIndexChanged slot where an escape is an unhandled exception.
        try:
            return problems_core.source_from_config(self.source_config())
        except (problems_core.ExpressionError, ValueError, KeyError):
            return None

    def _on_source_changed(self):
        cfg = self.source_combo.currentData() or {}
        kind = cfg.get("kind")
        is_custom = kind == problems_core.CUSTOM_KEY
        is_bbob = kind == problems_core.BBOB_KEY

        self.custom_box.setVisible(is_custom)
        source = self._current_source()

        # Only BBOB has instances; the others are one problem per function.
        self.instances.setEnabled(is_bbob)
        choices = source.instance_choices() if source else [1]
        keep = [i for i in self.instances.values() if i in choices]
        if not keep:
            # Falling back to a literal [1] left nothing checked whenever 1 was
            # not among the choices - and the list is disabled for non-BBOB
            # sources, so validate() then refused to run something the user had
            # no way to correct.
            keep = choices[:1]
        self.instances.replace(choices, selected=keep)
        if is_bbob:
            self.instance_note.setText(
                f"COCO defines {len(choices)} instances of every BBOB function. "
                f"The results stored in this repo use 1-5."
            )
        else:
            self.instance_note.setText("This source has no instances.")

        self.source_note.setText(source.description if source else "")
        self._refresh_functions()

    def _refresh_functions(self):
        """Repopulate the function list for the current source and dimensions.

        opfunu functions are not all dimension-changeable, so only those
        usable at EVERY selected dimension are offered - otherwise a run
        would quietly skip some of what the user picked.
        """
        source = self._current_source()
        dimensions = self.dimensions.values()
        if source is None or not dimensions:
            self.functions.replace([])
            self.function_note.setText(
                "Select at least one dimension to see the available functions."
                if source is not None else ""
            )
            self._refresh_estimate()
            return

        # Union, not intersection: most opfunu functions are fixed at a single
        # dimension, so intersecting would hide the bulk of the library as
        # soon as a second dimension is ticked. A function is offered if it
        # runs at any selected dimension, and is skipped at the others.
        union = source.catalog_union(dimensions)
        items = []
        for spec, runnable in union:
            label = spec.label
            if len(runnable) < len(dimensions):
                shown = ", ".join(str(d) for d in runnable)
                label = f"{label}   [dim {shown} only]"
            items.append((spec.id, label))
        previous = set(self.functions.values())
        keep = [i for i, _label in items if i in previous] or [i for i, _ in items[:2]]
        self.functions.replace(items, selected=keep)

        chosen = ", ".join(str(d) for d in dimensions)
        partial = sum(1 for _spec, runnable in union if len(runnable) < len(dimensions))
        note = f"{len(items)} functions available"
        if partial and len(dimensions) > 1:
            note += f"  -  {partial} run at only some of dimensions {chosen}"
        if isinstance(source, problems_core.CECSource):
            supported = source.supported_dimensions()
            if not items:
                note = (
                    f"No function in this suite runs at dimension {chosen}. "
                    f"It is defined for {', '.join(str(d) for d in supported)}."
                )
            else:
                note += f"  -  suite is defined for dimensions {', '.join(str(d) for d in supported)}"
        elif isinstance(source, problems_core.OpfunuSourceBase):
            # Count against the same union the items came from. Comparing a
            # single dimension's catalog with a multi-dimension union made
            # `hidden` meaningless, and negative often enough.
            total = len({
                spec.id
                for dimension in dimensions
                for spec in source.full_catalog(dimension)
            })
            hidden = total - len(items)
            if hidden > 0:
                note += (
                    f"  -  {hidden} hidden: fixed at a dimension outside "
                    f"{chosen}"
                )
        self.function_note.setText(note)
        self._refresh_estimate()

    def _check_expression(self):
        dimensions = self.dimensions.values() or [2]
        try:
            source = problems_core.source_from_config(self.source_config())
            value = source.validate(dimensions[0])
        except problems_core.ExpressionError as exc:
            self.custom_status.setProperty("class", "error")
            self.custom_status.setText(str(exc))
            return False
        self.custom_status.setProperty("class", "success")
        self.custom_status.setText(
            f"OK - f(centre) = {value:.6g} at dimension {dimensions[0]}"
        )
        return True

    def _build_parameter_box(self):
        box = QtWidgets.QGroupBox("Run parameters")
        form = QtWidgets.QFormLayout(box)

        self.epoch_per_dim = QtWidgets.QSpinBox()
        self.epoch_per_dim.setRange(1, 1000)
        self.epoch_per_dim.setValue(10)
        self.epoch_per_dim.setToolTip(
            "epoch = this x dimension (the ClustOpt convention the thesis uses)."
        )
        self.pop_size = QtWidgets.QSpinBox()
        self.pop_size.setRange(2, 1000)
        self.pop_size.setValue(50)
        for widget in (self.epoch_per_dim, self.pop_size):
            widget.valueChanged.connect(lambda _value: self._refresh_estimate())

        self.save_diversity = QtWidgets.QCheckBox(
            "Save diversity / exploration / exploitation"
        )
        self.save_diversity.setChecked(True)
        self.only_best = QtWidgets.QCheckBox(
            "Best-so-far trajectory only (skip the much larger population trajectory)"
        )
        self.only_best.toggled.connect(lambda _on: self._refresh_estimate())

        form.addRow("Epochs per dimension:", self.epoch_per_dim)
        form.addRow("Population size:", self.pop_size)
        form.addRow("", self.save_diversity)
        form.addRow("", self.only_best)
        return box

    def _build_output_box(self):
        box = QtWidgets.QGroupBox("Output")
        # The one group whose wrong setting destroys data: run_benchmarks
        # writes by algorithm and problem id, so pointing a run at the real
        # outputs/ overwrites existing trajectories in place. It gets a
        # visual weight the other groups do not have.
        box.setProperty("class", "destructive")
        layout = panel_layout.column(box)

        self.sandbox_choice = QtWidgets.QRadioButton(
            "Sandbox workspace (recommended)"
        )
        self.sandbox_choice.setChecked(True)
        self.real_choice = QtWidgets.QRadioButton(
            "The repository's real outputs/ directory"
        )
        self.real_choice.toggled.connect(self._on_real_output_toggled)

        self.output_hint = QtWidgets.QLabel()
        self.output_hint.setWordWrap(True)
        self.output_hint.setProperty("class", "hint")

        layout.addWidget(self.sandbox_choice)
        layout.addWidget(self.real_choice)
        layout.addWidget(self.output_hint)
        self._update_output_hint()
        return box

    def _build_run_box(self):
        box = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)

        self.estimate = QtWidgets.QLabel()
        self.estimate.setWordWrap(True)
        self.estimate.setProperty("class", "metric")

        self.warning = QtWidgets.QLabel()
        self.warning.setWordWrap(True)
        self.warning.setProperty("class", "warning")
        self.warning.hide()

        self.run_button = QtWidgets.QPushButton("Run benchmark")
        self.run_button.setProperty("class", "primary")
        self.run_button.setDefault(True)
        self.run_button.setShortcut("Ctrl+Return")
        self.run_button.setToolTip("Start the sweep described above (Ctrl+Return)")
        self.run_button.clicked.connect(self._request_run)

        layout.addWidget(self.estimate)
        layout.addWidget(self.warning)
        layout.addWidget(self.run_button)
        return box

    # -- state -----------------------------------------------------------

    def set_registry(self, registry):
        self.picker.set_registry(registry)

    def selected_optimizers(self):
        return self.picker.selected_names()

    def current_spec(self):
        return RunSpec(
            algorithms=self.picker.selected_names(),
            functions=self.functions.values(),
            instances=self.instances.values(),
            dimensions=self.dimensions.values(),
            seeds=self.seeds.values(),
            epoch_per_dim=self.epoch_per_dim.value(),
            pop_size=self.pop_size.value(),
            save_diversity=self.save_diversity.isChecked(),
            only_best=self.only_best.isChecked(),
            source=self.source_config(),
            function_names=self._function_names(),
            problem_count=self._problem_count(),
        )

    def _problem_count(self):
        """Real (function, instance, dimension) count for the current
        selection, so the estimate does not overstate a mixed-dimension run."""
        source = self._current_source()
        functions = self.functions.values()
        dimensions = self.dimensions.values()
        if source is None or not functions or not dimensions:
            return None
        try:
            return source.count_problems(functions, self.instances.values(), dimensions)
        except Exception:
            return None

    def _function_names(self):
        """id -> function name. Run directories are named by integer id, so
        without this the manifest could not say what F7 actually was."""
        source = self._current_source()
        selected = self.functions.values()
        if source is None or not selected:
            return {}
        if hasattr(source, "function_names"):
            return {str(k): v for k, v in source.function_names(selected).items()}
        return {str(f): f"F{f}" for f in selected}

    def prefill(self, algorithms=None, dimensions=None, functions=None):
        """Point the form at a specific gap - used by the 'generate missing
        data' button in the visualization panel.

        Forces the source back to BBOB: the gap was found in data derived from
        the BBOB suite, and a function id means something different under
        every other source.
        """
        self.source_combo.setCurrentIndex(0)
        if algorithms:
            self.picker.set_selected(algorithms)
        if dimensions:
            self.dimensions.set_values(dimensions)
        if functions:
            self.functions.set_values(functions)
        self._refresh_estimate()

    # -- feedback --------------------------------------------------------

    def _refresh_estimate(self):
        spec = self.current_spec()
        problems = spec.validate()
        self.run_button.setEnabled(not problems)

        if problems:
            self.estimate.setText("Cannot run yet: " + ", ".join(problems))
            self.warning.hide()
            return

        self.estimate.setText(
            f"{spec.describe()}\n"
            f"{spec.total_runs:,} runs, estimated "
            f"{format_duration(spec.estimate_seconds())}"
        )

        # Anything over a few minutes deserves saying out loud; the full sweep
        # is hours and is the repo's single most expensive operation.
        if spec.estimate_seconds() > LONG_RUN_SECONDS:
            self.warning.setText(
                "This is a long job. It runs in-process and can be cancelled "
                "at a run boundary, but partial results stay on disk."
            )
            self.warning.show()
        else:
            self.warning.hide()

    def _update_output_hint(self):
        if self.real_choice.isChecked():
            self.output_hint.setText(
                f"Writes into {config.OUTPUTS_DIR}. Existing trajectories for the "
                "same algorithm and problem will be overwritten in place."
            )
        else:
            self.output_hint.setText(
                f"Each run gets its own directory under {workspace_core.workspace_root()}, "
                "with a manifest.json recording exactly what produced it. "
                "The repository's outputs/ is not touched."
            )

    def _on_real_output_toggled(self, checked):
        if checked:
            answer = QtWidgets.QMessageBox.warning(
                self,
                "Write to the real outputs directory?",
                "outputs/ holds the benchmark results this project is built on. "
                "They are not in version control.\n\n"
                "A run writes by algorithm and problem id, so any run of an "
                "algorithm already present will overwrite its trajectories in "
                "place, and every metric derived from them will no longer match.\n\n"
                "Write there anyway?",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.Cancel,
                QtWidgets.QMessageBox.StandardButton.Cancel,
            )
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                self.sandbox_choice.setChecked(True)
                return
        self._update_output_hint()

    def _request_run(self):
        if (self.source_combo.currentData() or {}).get("kind") == problems_core.CUSTOM_KEY:
            if not self._check_expression():
                QtWidgets.QMessageBox.warning(
                    self, "Expression problem",
                    "The function could not be evaluated:\n\n"
                    + self.custom_status.text(),
                )
                return
        spec = self.current_spec()
        problems = spec.validate()
        if problems:
            QtWidgets.QMessageBox.warning(
                self, "Incomplete setup", "Cannot run:\n\n- " + "\n- ".join(problems)
            )
            return

        # Anything past a few minutes is worth confirming. Picking the output
        # folder already raises a modal; actually spending the hours did not.
        seconds = spec.estimate_seconds()
        if seconds > LONG_RUN_SECONDS:
            answer = QtWidgets.QMessageBox.question(
                self, "Start this run?",
                f"{spec.describe()}\n\n"
                f"{spec.total_runs:,} runs, estimated "
                f"{format_duration(seconds)}.\n\n"
                "It runs in this window and can be cancelled at a run "
                "boundary, but whatever has finished stays on disk. Start?",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.Cancel,
                QtWidgets.QMessageBox.StandardButton.Cancel,
            )
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                return

        self.runRequested.emit(spec, self.real_choice.isChecked())
