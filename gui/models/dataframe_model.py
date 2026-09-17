"""Qt table model over a pandas DataFrame.

QTableView only asks for the cells it is about to paint, so this stays
responsive on the largest tables the pipeline produces (336k rows in
entropy_granular_dim_{d}.csv, 227k x 11 in merged_dim_{d}.csv) without paging
or chunking.
"""
import numpy as np
import pandas as pd

from gui.qt import QtCore, QtGui, Qt

# Six significant figures reads well for both fitness values (1e-12 .. 1e8)
# and the 0..1 metric columns, without the noise of full float repr.
FLOAT_FORMAT = "{:.6g}"

_POSITIVE = QtGui.QColor(215, 96, 72)
_NEGATIVE = QtGui.QColor(80, 125, 195)


class DataFrameModel(QtCore.QAbstractTableModel):
    def __init__(self, frame=None, heatmap=False, parent=None):
        super().__init__(parent)
        self._frame = frame if frame is not None else pd.DataFrame()
        self._heatmap = heatmap
        self._sort_column = None
        self._sort_ascending = True

    # -- data ------------------------------------------------------------

    @property
    def frame(self):
        return self._frame

    def set_frame(self, frame, heatmap=False):
        self.beginResetModel()
        self._frame = frame
        self._heatmap = heatmap
        self._sort_column = None
        self.endResetModel()

    def rowCount(self, parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() else len(self._frame)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() else len(self._frame.columns)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        value = self._frame.iat[index.row(), index.column()]

        if role == Qt.ItemDataRole.DisplayRole:
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return ""
            if isinstance(value, (float, np.floating)):
                return FLOAT_FORMAT.format(value)
            return str(value)

        if role == Qt.ItemDataRole.TextAlignmentRole:
            if isinstance(value, (int, float, np.number)):
                return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            return int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        if role == Qt.ItemDataRole.BackgroundRole and self._heatmap:
            return self._shade(value)

        return None

    def _shade(self, value):
        """Correlation-matrix shading: blue for negative, red for positive."""
        if not isinstance(value, (float, int, np.number)) or np.isnan(value):
            return None
        magnitude = min(abs(float(value)), 1.0)
        base = _POSITIVE if value >= 0 else _NEGATIVE
        colour = QtGui.QColor(base)
        colour.setAlphaF(0.12 + 0.68 * magnitude)
        return QtGui.QBrush(colour)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return str(self._frame.columns[section])
        # For a matrix the index carries the labels; otherwise show row numbers.
        if self._heatmap:
            return str(self._frame.index[section])
        return str(section + 1)

    # -- sorting ---------------------------------------------------------

    def sort(self, column, order=Qt.SortOrder.AscendingOrder):
        if self._frame.empty or self._heatmap:
            return
        ascending = order == Qt.SortOrder.AscendingOrder
        name = self._frame.columns[column]
        self.beginResetModel()
        self._frame = self._frame.sort_values(
            by=name, ascending=ascending, kind="stable"
        )
        self._sort_column, self._sort_ascending = column, ascending
        self.endResetModel()
