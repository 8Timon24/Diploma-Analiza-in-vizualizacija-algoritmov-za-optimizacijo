"""Layout constructors with the app's spacing already applied.

Every panel used to build its own QVBoxLayout and either set
setContentsMargins(0, 0, 0, 0) or leave the defaults, with no setSpacing
anywhere - which is why controls sat flush against the tab edge and against
each other, and why no two panels had the same rhythm. Panels call these
instead of the Qt constructors.
"""
from gui import theme
from gui.qt import QtWidgets

M = theme.SPACE_M
S = theme.SPACE_S


def column(parent=None, margins=M, spacing=S):
    layout = QtWidgets.QVBoxLayout(parent) if parent else QtWidgets.QVBoxLayout()
    _apply(layout, margins, spacing)
    return layout


def row(parent=None, margins=0, spacing=S):
    layout = QtWidgets.QHBoxLayout(parent) if parent else QtWidgets.QHBoxLayout()
    _apply(layout, margins, spacing)
    return layout


def form(parent=None, margins=M, spacing=S):
    """A QFormLayout that wraps instead of eliding.

    The panes are narrow (the optimizer picker and the figure take most of
    the width) and several labels are long, so the fields have to be allowed
    to grow and the rows to wrap rather than truncate.
    """
    layout = QtWidgets.QFormLayout(parent) if parent else QtWidgets.QFormLayout()
    _apply(layout, margins, spacing)
    layout.setLabelAlignment(_alignment())
    layout.setFieldGrowthPolicy(
        QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
    )
    layout.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.DontWrapRows)
    return layout


def _alignment():
    from gui.qt import Qt

    return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter


def _apply(layout, margins, spacing):
    if isinstance(margins, int):
        margins = (margins,) * 4
    layout.setContentsMargins(*margins)
    layout.setSpacing(spacing)


def note(text, parent=None):
    """Secondary explanatory text, styled once instead of per panel."""
    label = QtWidgets.QLabel(text, parent)
    label.setWordWrap(True)
    label.setProperty("class", "hint")
    return label


def sub_row(form_layout, widget, text):
    """Attach an explanation directly under the control it describes.

    Adding the note as its own form row with an empty label left a ragged
    column of blank label cells and pushed the fields far apart.
    """
    host = QtWidgets.QWidget()
    inner = column(host, margins=0, spacing=2)
    inner.addWidget(widget)
    inner.addWidget(note(text))
    return host
