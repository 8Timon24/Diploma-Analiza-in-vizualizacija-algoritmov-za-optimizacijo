"""Small widgets shared between panels."""
from gui.qt import QtWidgets, Qt

VALUE_ROLE = Qt.ItemDataRole.UserRole


class CheckableList(QtWidgets.QListWidget):
    """Compact multi-select list of values."""

    def __init__(self, choices, selected=None, parent=None, height=140):
        super().__init__(parent)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.setMaximumHeight(height)
        chosen = set(selected or [])
        for choice in choices:
            # A choice is either a bare value, or a (value, label) pair when
            # the thing shown differs from the thing stored - opfunu functions
            # are picked by name but recorded by integer id.
            if isinstance(choice, tuple):
                choice, label = choice
            else:
                label = str(choice)
            item = QtWidgets.QListWidgetItem(label)
            item.setData(VALUE_ROLE, choice)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked if choice in chosen else Qt.CheckState.Unchecked
            )
            self.addItem(item)

    def values(self):
        return [
            self.item(i).data(VALUE_ROLE)
            for i in range(self.count())
            if self.item(i).checkState() == Qt.CheckState.Checked
        ]

    def replace(self, choices, selected=None):
        """Repopulate in place, keeping whatever selection is still valid."""
        keep = set(selected if selected is not None else self.values())
        self.clear()
        for choice in choices:
            if isinstance(choice, tuple):
                choice, label = choice
            else:
                label = str(choice)
            item = QtWidgets.QListWidgetItem(label)
            item.setData(VALUE_ROLE, choice)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked if choice in keep else Qt.CheckState.Unchecked
            )
            self.addItem(item)

    def set_values(self, values):
        wanted = set(values)
        for i in range(self.count()):
            item = self.item(i)
            item.setCheckState(
                Qt.CheckState.Checked
                if item.data(VALUE_ROLE) in wanted
                else Qt.CheckState.Unchecked
            )
