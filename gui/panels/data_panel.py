"""Browse every file the pipeline has written, in a readable table.

The tree populates lazily (outputs/ alone is ~151k files), and the table is
virtualized, so opening a 336k-row file is as cheap as opening a 20-row one.
"""
from pathlib import Path

from gui import theme
from gui.panels import layout as panel_layout
from gui.qt import QtCore, QtWidgets, Qt, Signal
from gui.core import datastore
from gui.core import results_root
from gui.models.dataframe_model import DataFrameModel

_NODE_ROLE = Qt.ItemDataRole.UserRole
_PLACEHOLDER = "__lazy__"


class DataPanel(QtWidgets.QWidget):
    changeRootRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current = None
        self._heatmap = False

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setUniformRowHeights(True)
        self.tree.itemExpanded.connect(self._populate_children)
        self.tree.itemSelectionChanged.connect(self._on_selected)

        self.model = DataFrameModel()
        self.table = QtWidgets.QTableView()
        self.table.setModel(self.model)
        self.table.setSortingEnabled(True)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectItems
        )
        self.table.horizontalHeader().setStretchLastSection(False)
        # Column auto-sizing asks the model for every row unless this is
        # bounded, which on entropy_granular (336k rows) means 336k pandas
        # lookups per column. The first screenful is enough to size by.
        self.table.horizontalHeader().setResizeContentsPrecision(64)
        self.table.setTextElideMode(Qt.TextElideMode.ElideRight)

        self.title = QtWidgets.QLabel("Select a file")
        self.title.setProperty("class", "heading")
        # These are absolute paths; without wrapping they force the whole
        # header pane wide or get elided with no way to read them.
        self.title.setWordWrap(True)
        self.details = QtWidgets.QLabel()
        self.details.setProperty("class", "hint")
        self.details.setWordWrap(True)

        self.export_button = QtWidgets.QPushButton("Export as CSV...")
        self.export_button.setEnabled(False)
        self.export_button.setShortcut("Ctrl+E")
        self.export_button.setToolTip("Save the table as it is shown (Ctrl+E)")
        self.export_button.clicked.connect(self._export)

        header = QtWidgets.QHBoxLayout()
        text = QtWidgets.QVBoxLayout()
        text.setSpacing(2)
        text.addWidget(self.title)
        text.addWidget(self.details)
        header.addLayout(text, 1)
        header.addWidget(self.export_button, 0, Qt.AlignmentFlag.AlignTop)

        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(panel_layout.S, 0, 0, 0)
        right_layout.setSpacing(panel_layout.S)
        right_layout.addLayout(header)
        right_layout.addWidget(self.table, 1)

        # Proportions, not pixel caps: a hard setMaximumWidth made the
        # splitter handle look broken - it dragged and the pane refused
        # to grow.
        splitter = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.tree)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([320, 980])
        splitter.setChildrenCollapsible(False)

        # Which results tree this is showing. Without it, an empty browser is
        # indistinguishable from "looking in the wrong place".
        self.root_label = QtWidgets.QLabel()
        self.root_label.setWordWrap(True)
        self.root_label.setProperty("class", "hint")
        change = QtWidgets.QPushButton("Change...")
        change.setToolTip("Point the app at a different results folder")
        change.clicked.connect(self.changeRootRequested)
        root_row = QtWidgets.QHBoxLayout()
        root_row.addWidget(QtWidgets.QLabel("Results folder:"))
        root_row.addWidget(self.root_label, 1)
        root_row.addWidget(change)

        layout = panel_layout.column(self)
        layout.addLayout(root_row)
        layout.addWidget(splitter, 1)

        self.refresh()

    # -- tree ------------------------------------------------------------

    def refresh(self):
        self.tree.clear()
        summary = results_root.inspect(results_root.current_root())
        self.root_label.setText(f"{summary.path}  -  {summary.describe()}")

        # The table belongs to the tree that was just discarded. Keeping it on
        # screen meant the header named a file from the previous results root
        # and Export wrote that file's data under the new root's name.
        self._current = None
        self._heatmap = False
        self.model.set_frame(None)
        self.export_button.setEnabled(False)

        roots = datastore.build_tree()
        if not roots:
            self.title.setText("No results found")
            self.details.setText(
                f"Nothing under {summary.path}.\n\n"
                "Run a benchmark from the Setup tab, or use Change... to point "
                "the app at a folder that already has results."
            )
            return
        self.title.setText("Select a file")
        self.details.setText("")
        for node in roots:
            self._add_item(self.tree, node)

    def reveal_latest_run(self):
        """Expand and select the newest GUI run.

        Without this a finished run leaves no visible trace anywhere in the
        app - the output goes to the workspace, not to the repo tree the
        other roots describe, so it looks like nothing was saved.
        """
        for index in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(index)
            node = item.data(0, _NODE_ROLE)
            if node is None or node.label != "GUI runs":
                continue
            self.tree.expandItem(item)
            if item.childCount():
                newest = item.child(0)
                self.tree.expandItem(newest)
                self.tree.setCurrentItem(newest)
                self.tree.scrollToItem(newest)
            return True
        return False

    def _add_item(self, parent, node):
        item = QtWidgets.QTreeWidgetItem(parent)
        item.setText(0, node.label)
        item.setData(0, _NODE_ROLE, node)
        if node.hint:
            item.setToolTip(0, node.hint)
        if not node.is_leaf:
            # A dummy child gives the item an expander without listing the
            # real children until the user asks for them.
            QtWidgets.QTreeWidgetItem(item).setText(0, _PLACEHOLDER)
        return item

    def _populate_children(self, item):
        if item.childCount() != 1 or item.child(0).text(0) != _PLACEHOLDER:
            return
        item.removeChild(item.child(0))
        node = item.data(0, _NODE_ROLE)
        QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            for child in node.children():
                self._add_item(item, child)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def _on_selected(self):
        items = self.tree.selectedItems()
        if not items:
            return
        node = items[0].data(0, _NODE_ROLE)
        if node is not None and node.is_leaf:
            self._load(node)

    # -- table -----------------------------------------------------------

    def _load(self, node):
        # One finally for the whole body: an exception anywhere past the read
        # (set_frame, resizeColumnsToContents) used to leave the application
        # with a busy cursor that nothing ever restored.
        QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            try:
                table = datastore.load_table(node.path)
            except Exception as exc:
                self.title.setText(node.label)
                self.details.setProperty("class", "error")
                self.details.setText(f"Could not read this file: {exc}")
                theme.restyle(self.details)
                self.model.set_frame(None)
                self._current = None
                self.export_button.setEnabled(False)
                return

            heatmap = node.kind == "matrix"
            # Drop any sort indicator left over from the previous file before
            # sorting is re-enabled, or the view immediately re-sorts this
            # frame by a column index chosen for a different one.
            self.table.setSortingEnabled(False)
            self.table.horizontalHeader().setSortIndicator(
                -1, Qt.SortOrder.AscendingOrder
            )
            self.model.set_frame(table.frame, heatmap=heatmap)
            self.table.setSortingEnabled(not heatmap)
            self.table.verticalHeader().setVisible(True)
            # Sampling a bounded number of rows keeps this O(columns) instead
            # of walking all 336k rows of entropy_granular through the model.
            self.table.resizeColumnsToContents()

            self._current = table
            self._heatmap = heatmap
            self.export_button.setEnabled(True)
            rows, columns = table.shape
            self.title.setText(str(_display_path(table.path)))
            parts = [f"{rows:,} rows x {columns} columns", f"{table.fmt}"]
            if heatmap:
                parts.append("correlation matrix - shaded, not sortable")
            parts.extend(table.notes)
            self.details.setProperty("class", "hint")
            self.details.setText("  -  ".join(parts))
            theme.restyle(self.details)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def _export(self):
        if self._current is None:
            return
        suggested = str(Path.home() / f"{self._current.path.stem}.csv")
        target, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export table as CSV", suggested, "CSV files (*.csv)"
        )
        if not target:
            return
        QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            # A matrix carries its labels in the index, so dropping it would
            # silently export half the data. This used to be decided by
            # string-matching the filename, which only covered spearman_*.
            self.model.frame.to_csv(target, index=self._heatmap)
        except OSError as exc:
            QtWidgets.QMessageBox.warning(
                self, "Export failed", f"Could not write {target}:\n\n{exc}"
            )
        else:
            self.details.setText(f"Exported to {target}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()


def _display_path(path):
    """Show the path relative to the repo root when it is inside it."""
    path = Path(path)
    try:
        return path.relative_to(Path(datastore.config.REPO_ROOT))
    except ValueError:
        return path
