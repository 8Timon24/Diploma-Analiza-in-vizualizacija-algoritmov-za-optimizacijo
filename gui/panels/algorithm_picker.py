"""Checkable, searchable tree of every mealpy optimizer, grouped by family.

Families come from mealpy itself (see gui.core.optimizers), so this stays
correct when mealpy adds optimizers - there is no hardcoded list anywhere in
the widget.
"""
from gui.qt import QtWidgets, Qt, Signal
from gui.core import optimizers as optimizers_core

_NAME_ROLE = Qt.ItemDataRole.UserRole


class AlgorithmPicker(QtWidgets.QWidget):
    """Multi-select optimizer picker. Emits the selected names on any change."""

    selectionChanged = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._registry = None
        self._family_items = {}
        self._updating = False

        self.search = QtWidgets.QLineEdit(placeholderText="Filter optimizers...")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._apply_filter)

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setUniformRowHeights(True)
        self.tree.itemChanged.connect(self._on_item_changed)

        self.count_label = QtWidgets.QLabel("Loading optimizers...")
        self.count_label.setStyleSheet("color: palette(mid);")

        buttons = QtWidgets.QHBoxLayout()
        for text, tip, slot in (
            ("Thesis set", "The 28 optimizers pinned in config.py", self.select_thesis_set),
            ("All", "Select every discovered optimizer", self.select_all),
            ("None", "Clear the selection", self.select_none),
        ):
            b = QtWidgets.QPushButton(text, toolTip=tip)
            b.clicked.connect(slot)
            buttons.addWidget(b)
        buttons.addStretch()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.search)
        layout.addWidget(self.tree, 1)
        layout.addLayout(buttons)
        layout.addWidget(self.count_label)

        self.setEnabled(False)

    # -- population ------------------------------------------------------

    def set_registry(self, registry):
        """Populate from a loaded registry (see optimizers_core.load_registry)."""
        self._registry = registry
        self._updating = True
        self.tree.clear()
        self._family_items.clear()

        for family, names in registry.by_family.items():
            parent = QtWidgets.QTreeWidgetItem(self.tree)
            parent.setText(0, f"{optimizers_core.family_label(family)}  ({len(names)})")
            parent.setFlags(parent.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            parent.setCheckState(0, Qt.CheckState.Unchecked)
            self._family_items[family] = parent
            for name in names:
                child = QtWidgets.QTreeWidgetItem(parent)
                child.setText(0, name)
                child.setData(0, _NAME_ROLE, name)
                child.setFlags(child.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                child.setCheckState(0, Qt.CheckState.Unchecked)

        self._updating = False
        self.setEnabled(True)
        self.select_thesis_set()

    # -- selection -------------------------------------------------------

    def selected_names(self):
        return [
            child.data(0, _NAME_ROLE)
            for parent in self._family_items.values()
            for child in (parent.child(i) for i in range(parent.childCount()))
            if child.checkState(0) == Qt.CheckState.Checked
        ]

    def set_selected(self, names):
        wanted = set(names)
        self._updating = True
        for parent in self._family_items.values():
            for i in range(parent.childCount()):
                child = parent.child(i)
                state = (
                    Qt.CheckState.Checked
                    if child.data(0, _NAME_ROLE) in wanted
                    else Qt.CheckState.Unchecked
                )
                child.setCheckState(0, state)
            self._sync_parent(parent)
        self._updating = False
        self._emit()

    def select_thesis_set(self):
        self.set_selected(optimizers_core.thesis_selection())

    def select_all(self):
        self.set_selected(self._registry.names if self._registry else [])

    def select_none(self):
        self.set_selected([])

    # -- internals -------------------------------------------------------

    def _on_item_changed(self, item, _column):
        if self._updating:
            return
        self._updating = True
        if item.childCount():  # a family header: push its state down
            for i in range(item.childCount()):
                item.child(i).setCheckState(0, item.checkState(0))
        elif item.parent():  # a leaf: pull the header into line
            self._sync_parent(item.parent())
        self._updating = False
        self._emit()

    def _sync_parent(self, parent):
        """Set a family header to checked / unchecked / partial from its children."""
        checked = sum(
            parent.child(i).checkState(0) == Qt.CheckState.Checked
            for i in range(parent.childCount())
        )
        if checked == 0:
            state = Qt.CheckState.Unchecked
        elif checked == parent.childCount():
            state = Qt.CheckState.Checked
        else:
            state = Qt.CheckState.PartiallyChecked
        parent.setCheckState(0, state)

    def _apply_filter(self, text):
        needle = text.strip().lower()
        for parent in self._family_items.values():
            visible_children = 0
            for i in range(parent.childCount()):
                child = parent.child(i)
                match = needle in child.data(0, _NAME_ROLE).lower()
                child.setHidden(not match)
                visible_children += match
            parent.setHidden(visible_children == 0)
            if needle:
                parent.setExpanded(True)

    def _emit(self):
        names = self.selected_names()
        total = len(self._registry.names) if self._registry else 0
        self.count_label.setText(f"{len(names)} of {total} optimizers selected")
        self.selectionChanged.emit(names)
