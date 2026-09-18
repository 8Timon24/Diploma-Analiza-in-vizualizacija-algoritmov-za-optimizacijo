"""Small widgets shared between panels."""
from gui.qt import QtWidgets, Qt

VALUE_ROLE = Qt.ItemDataRole.UserRole

# Rows shown before the list starts scrolling. Sized in rows rather than
# pixels so the list still fits its content when the system font is larger.
DEFAULT_VISIBLE_ROWS = 7


class CheckableList(QtWidgets.QListWidget):
    """Compact multi-select list of values."""

    def __init__(self, choices, selected=None, parent=None, rows=DEFAULT_VISIBLE_ROWS):
        super().__init__(parent)
        # Selection stays on so the list can be reached from the keyboard:
        # with NoSelection there is no current item, so arrow keys and Space
        # do nothing and every one of these lists is mouse-only.
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.setUniformItemSizes(True)
        self._visible_rows = rows
        self._fill(choices, set(selected or []))

    # -- sizing ----------------------------------------------------------

    def sizeHint(self):
        """Ask for `rows` rows of the current font, not a fixed pixel box."""
        hint = super().sizeHint()
        row = self.sizeHintForRow(0) if self.count() else self.fontMetrics().height() + 6
        frame = 2 * self.frameWidth()
        hint.setHeight(row * self._visible_rows + frame)
        return hint

    def minimumSizeHint(self):
        hint = super().minimumSizeHint()
        row = self.sizeHintForRow(0) if self.count() else self.fontMetrics().height() + 6
        hint.setHeight(row * 3 + 2 * self.frameWidth())
        return hint

    # -- contents --------------------------------------------------------

    def _make_item(self, choice):
        # A choice is either a bare value, or a (value, label) pair when the
        # thing shown differs from the thing stored - opfunu functions are
        # picked by name but recorded by integer id.
        if isinstance(choice, tuple):
            choice, label = choice
        else:
            label = str(choice)
        item = QtWidgets.QListWidgetItem(label)
        item.setData(VALUE_ROLE, choice)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        # The labels carry qualifiers the panes are too narrow to show, e.g.
        # "Ackley   [dim 2, 5 only]".
        item.setToolTip(label)
        return item, choice

    def _fill(self, choices, chosen):
        for choice in choices:
            item, value = self._make_item(choice)
            item.setCheckState(
                Qt.CheckState.Checked if value in chosen else Qt.CheckState.Unchecked
            )
            self.addItem(item)

    def add_choice(self, value, checked=True):
        """Append one choice outside the fixed set the list was built with -
        e.g. a seed value the user typed in, rather than one of config.SEEDS.

        Re-checks the existing item instead of duplicating if `value` is
        already present.
        """
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for i in range(self.count()):
            existing = self.item(i)
            if existing.data(VALUE_ROLE) == value:
                existing.setCheckState(state)
                self.itemChanged.emit(existing)
                return
        item, _ = self._make_item(value)
        item.setCheckState(state)
        self.addItem(item)
        self.itemChanged.emit(item)

    def values(self):
        return [
            self.item(i).data(VALUE_ROLE)
            for i in range(self.count())
            if self.item(i).checkState() == Qt.CheckState.Checked
        ]

    def replace(self, choices, selected=None):
        """Repopulate in place.

        `selected=None` keeps whatever is still valid of the current
        selection - the opposite of set_values(), which treats None as an
        empty selection. Both are called from the panels, so the difference
        is deliberate.
        """
        keep = set(selected if selected is not None else self.values())
        blocked = self.signalsBlocked()
        self.blockSignals(True)
        try:
            self.clear()
            self._fill(choices, keep)
        finally:
            self.blockSignals(blocked)

    def set_values(self, values):
        """Check exactly `values`, emitting itemChanged at most once.

        Every checkbox used to emit separately, and the setup panel rebuilds
        its whole function catalog on that signal - so prefilling three
        dimensions ran the rebuild three times.
        """
        wanted = set(values)
        changed = False
        blocked = self.signalsBlocked()
        self.blockSignals(True)
        try:
            for i in range(self.count()):
                item = self.item(i)
                state = (
                    Qt.CheckState.Checked
                    if item.data(VALUE_ROLE) in wanted
                    else Qt.CheckState.Unchecked
                )
                if item.checkState() != state:
                    item.setCheckState(state)
                    changed = True
        finally:
            self.blockSignals(blocked)
        if changed and self.count():
            self.itemChanged.emit(self.item(0))
