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

# The only roles data() answers. Anything else short-circuits before the
# per-cell pandas lookup.
_HANDLED_ROLES = frozenset({
    Qt.ItemDataRole.DisplayRole,
    Qt.ItemDataRole.ToolTipRole,
    Qt.ItemDataRole.TextAlignmentRole,
    Qt.ItemDataRole.BackgroundRole,
})


def _is_number(value):
    """True for real numbers only - bool is an int subclass and must not
    count, or a boolean column gets right-aligned and heat-shaded."""
    return isinstance(value, (int, float, np.number)) and not isinstance(value, bool)


def _is_scalar_na(value):
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False  # arrays and exotic objects are not missing values


class DataFrameModel(QtCore.QAbstractTableModel):
    def __init__(self, frame=None, heatmap=False, parent=None):
        super().__init__(parent)
        self._frame = frame if frame is not None else pd.DataFrame()
        # sort() works from the frame as loaded, so repeated header clicks
        # never sort an already-sorted copy and the file's own row order
        # stays recoverable.
        self._source_frame = self._frame
        self._heatmap = heatmap
        self._sort_column = None
        self._sort_ascending = True

    # -- data ------------------------------------------------------------

    @property
    def frame(self):
        return self._frame

    def set_frame(self, frame, heatmap=False):
        # None is a routine argument here, not a mistake: the data browser
        # passes it for "nothing selected" and for "this file could not be
        # read". Without this guard rowCount() raised TypeError on len(None)
        # during endResetModel(), i.e. exactly on the two paths that are
        # supposed to degrade gracefully.
        self.beginResetModel()
        self._frame = frame if frame is not None else pd.DataFrame()
        self._source_frame = self._frame
        self._heatmap = heatmap
        self._sort_column = None
        self._sort_ascending = True
        self.endResetModel()

    def rowCount(self, parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() else len(self._frame)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() else len(self._frame.columns)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        # Qt asks for a dozen roles per cell per repaint and only four are
        # answered here. Reading the cell before the role check meant a
        # pandas scalar lookup for every one of them, which is what made
        # resizeColumnsToContents() on a 336k-row file so expensive.
        if role not in _HANDLED_ROLES:
            return None
        row, column = index.row(), index.column()
        if row >= len(self._frame) or column >= len(self._frame.columns):
            return None  # a stale index from before the last set_frame()
        value = self._frame.iat[row, column]

        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole):
            return self._display(value)

        if role == Qt.ItemDataRole.TextAlignmentRole:
            if _is_number(value):
                return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            return int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        if role == Qt.ItemDataRole.BackgroundRole and self._heatmap:
            return self._shade(value)

        return None

    @staticmethod
    def _display(value):
        # pd.isna covers np.float32 NaN, pd.NaT and pd.NA, none of which are
        # instances of the builtin float - they used to render as "nan" while
        # a float64 NaN in the next column rendered as "".
        if value is None or (_is_scalar_na(value)):
            return ""
        if isinstance(value, (float, np.floating)):
            return FLOAT_FORMAT.format(value)
        return str(value)

    def _shade(self, value):
        """Correlation-matrix shading: blue for negative, red for positive."""
        if not _is_number(value) or _is_scalar_na(value):
            return None
        magnitude = min(abs(float(value)), 1.0)
        base = _POSITIVE if value >= 0 else _NEGATIVE
        colour = QtGui.QColor(base)
        # Capped well below opaque: the text on top is drawn in the palette's
        # foreground colour, and at 0.8 alpha the |r|~1 cells - the ones worth
        # reading - were the hardest to read.
        colour.setAlphaF(0.10 + 0.45 * magnitude)
        return QtGui.QBrush(colour)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role not in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole):
            return None
        if orientation == Qt.Orientation.Horizontal:
            if section >= len(self._frame.columns):
                return None
            return str(self._frame.columns[section])
        if section >= len(self._frame.index):
            return None
        # For a matrix the index carries the labels; otherwise show the row's
        # number in the file. Taking it from the index rather than from
        # `section` keeps it pointing at the same record after a sort.
        label = self._frame.index[section]
        if self._heatmap:
            return str(label)
        if isinstance(label, (int, np.integer)):
            return str(int(label) + 1)
        return str(label)

    # -- sorting ---------------------------------------------------------

    def sort(self, column, order=Qt.SortOrder.AscendingOrder):
        if self._heatmap or self._source_frame.empty:
            return
        if not 0 <= column < len(self._source_frame.columns):
            # Reachable: the view keeps its sort indicator across files, so a
            # narrower file can be asked to sort by a column index that only
            # existed in the previous one.
            return
        ascending = order == Qt.SortOrder.AscendingOrder
        name = self._source_frame.columns[column]
        self.beginResetModel()
        try:
            self._frame = self._source_frame.sort_values(
                by=name, ascending=ascending, kind="stable"
            )
            self._sort_column, self._sort_ascending = column, ascending
        except (TypeError, ValueError):
            # Un-orderable or duplicated column: leave the data as loaded
            # rather than half-sorting it.
            self._frame = self._source_frame
            self._sort_column, self._sort_ascending = None, True
        finally:
            self.endResetModel()
