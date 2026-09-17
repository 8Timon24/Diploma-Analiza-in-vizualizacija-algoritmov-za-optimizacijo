"""Checkable, searchable tree of every mealpy optimizer, grouped by family.

Families come from mealpy itself (see gui.core.optimizers), so this stays
correct when mealpy adds optimizers - there is no hardcoded list anywhere in
the widget.
"""
from gui.panels import layout as panel_layout
from gui.qt import QtGui, QtWidgets, Qt, Signal
from gui.core import optimizers as optimizers_core

_NAME_ROLE = Qt.ItemDataRole.UserRole
_FAMILY_ROLE = Qt.ItemDataRole.UserRole + 1


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
        self.search.setToolTip("Filter by name (Ctrl+F)")
        find = QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Find, self)
        find.activated.connect(self.search.setFocus)
        self.search.textChanged.connect(self._apply_filter)

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setUniformRowHeights(True)
        self.tree.itemChanged.connect(self._on_item_changed)

        self.count_label = QtWidgets.QLabel("Loading optimizers...")
        self.count_label.setProperty("class", "hint")

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

        layout = panel_layout.column(self, margins=0)
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
        try:
            # Drop our references BEFORE clear() deletes the C++ items, or
            # anything that walks _family_items in between (selected_names,
            # _sync_parent) hits already-deleted objects.
            self._family_items.clear()
            self.tree.clear()

            for family, names in registry.by_family.items():
                parent = QtWidgets.QTreeWidgetItem(self.tree)
                parent.setData(0, _FAMILY_ROLE, family)
                parent.setFlags(parent.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                parent.setCheckState(0, Qt.CheckState.Unchecked)
                self._family_items[family] = parent
                for name in names:
                    child = QtWidgets.QTreeWidgetItem(parent)
                    child.setText(0, name)
                    child.setData(0, _NAME_ROLE, name)
                    child.setFlags(child.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    child.setCheckState(0, Qt.CheckState.Unchecked)
                self._relabel_family(parent)
        finally:
            # Without this, any exception above leaves the flag set and every
            # later click is silently ignored with nothing shown to the user.
            self._updating = False

        self.setEnabled(True)
        self.select_thesis_set()

    def _relabel_family(self, parent):
        """Header text with the count that is actually visible under it."""
        family = parent.data(0, _FAMILY_ROLE)
        total = parent.childCount()
        shown = sum(not parent.child(i).isHidden() for i in range(total))
        label = optimizers_core.family_label(family)
        if shown == total:
            parent.setText(0, f"{label}  ({total})")
        else:
            parent.setText(0, f"{label}  ({shown} of {total})")

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
        try:
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
        finally:
            self._updating = False
        self._emit()

    def select_thesis_set(self):
        self.set_selected(optimizers_core.thesis_selection())

    def select_all(self):
        if self._registry is None:
            return  # before the registry lands this would just clear the tree
        self.set_selected(self._registry.names)

    def select_none(self):
        self.set_selected([])

    # -- internals -------------------------------------------------------

    def _on_item_changed(self, item, _column):
        if self._updating:
            return
        self._updating = True
        try:
            if item.parent() is None:  # a family header: push its state down
                state = item.checkState(0)
                for i in range(item.childCount()):
                    child = item.child(i)
                    # Only what the filter is showing. Pushing the header's
                    # state onto hidden children meant ticking "Swarm" while
                    # filtering for "DE" silently selected 50 optimizers the
                    # user could not see.
                    if not child.isHidden():
                        child.setCheckState(0, state)
                self._sync_parent(item)
            else:  # a leaf: pull the header into line
                self._sync_parent(item.parent())
        finally:
            self._updating = False
        self._emit()

    def _sync_parent(self, parent):
        """Set a family header to checked / unchecked / partial from its children."""
        visible = [parent.child(i) for i in range(parent.childCount())
                   if not parent.child(i).isHidden()]
        children = visible or [parent.child(i) for i in range(parent.childCount())]
        if not children:
            return
        checked = sum(c.checkState(0) == Qt.CheckState.Checked for c in children)
        if checked == 0:
            state = Qt.CheckState.Unchecked
        elif checked == len(children):
            state = Qt.CheckState.Checked
        else:
            state = Qt.CheckState.PartiallyChecked
        parent.setCheckState(0, state)

    def _apply_filter(self, text):
        needle = text.strip().lower()
        self._updating = True
        try:
            for parent in self._family_items.values():
                visible_children = 0
                for i in range(parent.childCount()):
                    child = parent.child(i)
                    match = needle in child.data(0, _NAME_ROLE).lower()
                    child.setHidden(not match)
                    visible_children += match
                parent.setHidden(visible_children == 0)
                # Expanding on a search is helpful; staying expanded after it
                # is cleared is not.
                parent.setExpanded(bool(needle))
                self._relabel_family(parent)
                self._sync_parent(parent)
        finally:
            self._updating = False
        self._emit()

    def _emit(self):
        names = self.selected_names()
        total = len(self._registry.names) if self._registry else 0
        text = f"{len(names)} of {total} optimizers selected"
        hidden = sum(
            1
            for parent in self._family_items.values()
            for i in range(parent.childCount())
            if parent.child(i).isHidden()
            and parent.child(i).checkState(0) == Qt.CheckState.Checked
        )
        if hidden:
            text += f"  ({hidden} hidden by the filter)"
        self.count_label.setText(text)
        self.selectionChanged.emit(names)
