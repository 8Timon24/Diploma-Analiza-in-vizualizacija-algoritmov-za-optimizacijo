"""The application's one source of colour, spacing and type.

Before this existed the app ran on the bare platform style and every panel
styled its own labels, several with hardcoded hex colours that inverted badly
on a dark desktop. Everything visual now comes from here:

  * a light and a dark palette, chosen from the OS preference by default and
    re-applied live when it changes (Qt 6.5+ exposes styleHints().colorScheme();
    the colorSchemeChanged signal arrived in 6.8, and this app ships 6.11) -
    or pinned to one or the other via set_mode(), overriding the OS
  * semantic tokens - DANGER, SUCCESS, WARNING, ACCENT, MUTED - so a panel
    never names a colour directly
  * one stylesheet with the classes the panels use: hint, error, success,
    warning, heading, metric, mono

Panels ask for a look with setProperty("class", "hint") and call
restyle(widget) if they change it after the widget is shown. Qt only
re-evaluates a stylesheet on a property change when told to.
"""
from gui.qt import QtCore, QtGui, QtWidgets, Qt

# Spacing scale, in px at 100% scaling. Panels use these instead of inventing
# a margin each - that is what made the original layouts look arbitrary.
SPACE_XS, SPACE_S, SPACE_M, SPACE_L = 4, 8, 12, 16

_LIGHT = {
    "window": "#f4f6f8",
    "base": "#ffffff",
    "alt_base": "#eef1f5",
    # Pure white rather than a near-window off-white: at #fafcfc a card was
    # only ~1% brighter than the #f4f6f8 window behind it and read as an
    # outlined box, not an elevated surface - the 1px border was doing all
    # the work. Matches dark mode's own ordering, where card is the
    # lightest of window/base/card, not squeezed between them.
    "card": "#ffffff",
    "text": "#191c21",
    "muted": "#5c6470",
    "border": "#d5dae1",
    # Deep teal rather than a generic SaaS blue - it reads as instrument /
    # data-tool rather than dashboard-product, and sits close to the
    # viridis/magma families figure_theme.py already draws charts with.
    # Darker than the dark-mode accent on purpose: white text/icons on this
    # need to clear WCAG AA's 4.5:1 for normal text (this is QPalette's
    # Highlight colour, so it also sits under every selected list/table row,
    # not just buttons). #0f9488 only measured 3.74:1 - AA-failing - and was
    # darkened until it cleared 5:1.
    "accent": "#0c7d72",
    "accent_text": "#ffffff",
    # accent darkened ~12%: QPushButton:hover on a primary button only
    # changes border-color, which is already accent at rest, so hovering
    # Run/Compare/Render/Load was invisible without a background to move to.
    "accent_hover": "#0a6e64",
    "danger": "#b3261e",
    "success": "#1a7f4b",
    "warning": "#9a5b06",
    "disabled": "#9aa2ad",
}

_DARK = {
    "window": "#1b1e23",
    "base": "#22262d",
    "alt_base": "#272c34",
    "card": "#262b32",
    "text": "#e6e9ed",
    "muted": "#9aa3b0",
    "border": "#343a44",
    "accent": "#2dd4bf",
    "accent_text": "#0d1117",
    "accent_hover": "#27baa8",
    "danger": "#ef6f63",
    "success": "#43c07d",
    "warning": "#dca23c",
    "disabled": "#6b737f",
}

_current = dict(_LIGHT)
_applied_to = None

# "system" follows the OS live (the colorSchemeChanged connection below);
# "light"/"dark" pin the palette regardless of what the OS reports. Persisted
# and restored by gui/main_window.py's View > Appearance menu - theme.py
# itself has no QSettings dependency, so it stays usable from the self-test
# and from scripts that never build a MainWindow.
_mode = "system"

# Widget state that isn't covered by the app-wide QPalette/stylesheet - most
# notably pre-rendered QIcons, which paint a colour baked in at creation time
# and don't re-read tokens() on their own - registers here to be redone after
# every refresh(), whether it was triggered by the OS or by set_mode().
_on_change = []


def tokens():
    """The active palette, for code that needs a colour outside a stylesheet
    (the matplotlib theme, mostly)."""
    return dict(_current)


def is_dark():
    return _current is not None and _current.get("window") == _DARK["window"]


def mode():
    """The current appearance mode: "system", "light", or "dark"."""
    return _mode


def set_mode(new_mode, app=None):
    """Pin the palette to "light"/"dark", or "system" to follow the OS again."""
    global _mode
    if new_mode not in ("system", "light", "dark"):
        raise ValueError(f"unknown appearance mode: {new_mode!r}")
    _mode = new_mode
    refresh(app)


def on_change(callback):
    """Run `callback` after every refresh() from now on (an OS scheme change
    while mode is "system", or an explicit set_mode()). Does not call it
    immediately - callers that need the current state should read tokens()
    themselves first."""
    _on_change.append(callback)


def off_change(callback):
    """Undo on_change(callback). This list is module-global, so anything
    long-lived enough to register (a main window, not a short-lived dialog)
    must also unregister when it's destroyed, or it accumulates one dead
    callback per instance in any process that builds more than one (the
    self-test and ad-hoc scripts both do)."""
    if callback in _on_change:
        _on_change.remove(callback)


# -- applying ------------------------------------------------------------

def apply(app=None):
    """Style the application. Safe to call more than once."""
    global _applied_to
    app = app or QtWidgets.QApplication.instance()
    if app is None:
        return None

    # Fusion rather than the platform style: it honours a custom QPalette on
    # every platform, which the native Windows and macOS styles largely do
    # not, so the app would be themed on Linux only.
    app.setStyle("Fusion")

    refresh(app)

    app.setWindowIcon(app_icon())

    if _applied_to is not app:
        hints = QtGui.QGuiApplication.styleHints()
        signal = getattr(hints, "colorSchemeChanged", None)
        if signal is not None:
            signal.connect(lambda _scheme: refresh(app))
        _applied_to = app
    return app


def refresh(app=None):
    """Re-apply the palette for the current mode() and restyle everything."""
    global _current
    app = app or QtWidgets.QApplication.instance()
    if app is None:
        return
    dark = _prefers_dark() if _mode == "system" else _mode == "dark"
    _current = dict(_DARK if dark else _LIGHT)
    app.setPalette(_build_palette(_current))
    app.setStyleSheet(stylesheet(_current))
    for callback in list(_on_change):
        try:
            callback()
        except RuntimeError:
            # The Qt object behind a bound-method callback was deleted
            # without going through off_change() first (deleteLater()'s
            # actual C++ destruction is asynchronous, so this can outrace a
            # destroyed-signal-based unregister by a frame or two). Drop it
            # rather than let one dead callback block every other one.
            _on_change.remove(callback)


def _prefers_dark():
    hints = QtGui.QGuiApplication.styleHints()
    scheme = getattr(hints, "colorScheme", None)
    if scheme is None:
        return False
    try:
        return scheme() == Qt.ColorScheme.Dark
    except AttributeError:
        return False


def _build_palette(t):
    c = QtGui.QColor
    palette = QtGui.QPalette()
    role = QtGui.QPalette.ColorRole
    group = QtGui.QPalette.ColorGroup

    palette.setColor(role.Window, c(t["window"]))
    palette.setColor(role.WindowText, c(t["text"]))
    palette.setColor(role.Base, c(t["base"]))
    palette.setColor(role.AlternateBase, c(t["alt_base"]))
    palette.setColor(role.Text, c(t["text"]))
    palette.setColor(role.Button, c(t["window"]))
    palette.setColor(role.ButtonText, c(t["text"]))
    palette.setColor(role.Highlight, c(t["accent"]))
    palette.setColor(role.HighlightedText, c(t["accent_text"]))
    palette.setColor(role.ToolTipBase, c(t["base"]))
    palette.setColor(role.ToolTipText, c(t["text"]))
    palette.setColor(role.Link, c(t["accent"]))
    palette.setColor(role.PlaceholderText, c(t["muted"]))
    # `mid` is what every existing "color: palette(mid)" label resolves to,
    # so it has to be the muted token rather than Fusion's own grey.
    palette.setColor(role.Mid, c(t["muted"]))
    palette.setColor(role.Dark, c(t["border"]))

    for state in (group.Disabled,):
        palette.setColor(state, role.WindowText, c(t["disabled"]))
        palette.setColor(state, role.Text, c(t["disabled"]))
        palette.setColor(state, role.ButtonText, c(t["disabled"]))
    return palette


def stylesheet(t=None):
    t = t or _current
    return f"""
    QWidget {{ color: {t['text']}; }}
    QMainWindow, QDialog {{ background: {t['window']}; }}

    /* -- tabs: a real selected state instead of a hairline ------------- */
    QTabWidget::pane {{
        border: 1px solid {t['border']};
        border-radius: 6px;
        background: {t['window']};
        top: -1px;
    }}
    QTabBar::tab {{
        padding: {SPACE_S}px {SPACE_L}px;
        margin-right: 2px;
        border: 1px solid transparent;
        border-top-left-radius: 6px;
        border-top-right-radius: 6px;
        color: {t['muted']};
    }}
    QTabBar::tab:hover {{ color: {t['text']}; }}
    QTabBar::tab:selected {{
        background: {t['window']};
        border-color: {t['border']};
        border-bottom-color: {t['window']};
        /* A real accent-coloured indicator on the active tab, rather than
           font-weight alone carrying "this is the selected one". */
        border-top: 2px solid {t['accent']};
        padding-top: {SPACE_S - 1}px;
        color: {t['text']};
        font-weight: 600;
    }}

    /* -- grouping: a flat "card" rather than a bare outline ------------- */
    QGroupBox {{
        border: 1px solid {t['border']};
        border-radius: 8px;
        margin-top: {SPACE_M + 2}px;
        padding: {SPACE_M + 4}px {SPACE_M}px {SPACE_M}px {SPACE_M}px;
        background: {t['card']};
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: {SPACE_M}px;
        /* The title is drawn over the frame's top edge, so it needs an
           opaque background to punch a gap in the border rather than sit
           on the line. */
        background: {t['window']};
        padding: 0 {SPACE_S}px;
        color: {t['muted']};
        font-weight: 600;
        font-size: 10.5px;
    }}
    /* Fusion's QGroupBox::title ignores CSS text-transform/letter-spacing
       (verified by rendering one in isolation - it painted mixed-case with
       no changes), so the small-caps section-label look is delivered by
       gui/panels/layout.py's group_box() upper-casing the title string
       instead. Panels must go through that helper, not QGroupBox directly,
       or their title silently reverts to plain mixed case. */
    /* The output group decides whether a run overwrites real trajectories,
       so it is the one group that announces itself. */
    QGroupBox[class="destructive"] {{ border-color: {t['warning']}; }}
    QGroupBox[class="destructive"]::title {{ color: {t['warning']}; }}

    /* -- controls ------------------------------------------------------ */
    QPushButton {{
        padding: {SPACE_S}px {SPACE_M}px;
        border: 1px solid {t['border']};
        border-radius: 5px;
        background: {t['base']};
    }}
    QPushButton:hover {{ border-color: {t['accent']}; }}
    QPushButton:disabled {{ color: {t['disabled']}; border-color: {t['border']}; }}
    /* Fusion's native dotted focus rect doesn't show up against a coloured
       border, so a keyboard-focused button was visually identical to an
       idle one. Drawn outside the border box (outline-offset) so it doesn't
       fight with :hover/:default's own border-color changes. */
    QPushButton:focus {{
        outline: 2px solid {t['accent']};
        outline-offset: 2px;
    }}
    QPushButton:default, QPushButton[class="primary"] {{
        background: {t['accent']};
        color: {t['accent_text']};
        border-color: {t['accent']};
        font-weight: 600;
    }}
    /* The general QPushButton:hover rule only moves border-color, which is
       already accent here at rest - primary buttons need their own hover
       state or the app's highest-intent actions (Run/Compare/Render/Load)
       are the only controls that give no hover feedback at all. */
    QPushButton:default:hover, QPushButton[class="primary"]:hover {{
        background: {t['accent_hover']};
        border-color: {t['accent_hover']};
    }}
    QPushButton:default:disabled, QPushButton[class="primary"]:disabled {{
        background: {t['border']};
        color: {t['disabled']};
        border-color: {t['border']};
    }}

    /* Deliberately NOT QComboBox / QSpinBox / QDoubleSpinBox. Giving those a
       border here makes Qt stop drawing their sub-controls natively, and a
       stylesheet cannot draw a replacement arrow without shipping an image:
       overriding ::down-arrow renders a stray dash, and omitting it renders
       nothing at all. Fusion draws them correctly from the palette, which is
       themed, so they are left alone. */
    QLineEdit, QPlainTextEdit, QTextEdit {{
        padding: {SPACE_XS}px {SPACE_S}px;
        border: 1px solid {t['border']};
        border-radius: 5px;
        background: {t['base']};
        selection-background-color: {t['accent']};
        selection-color: {t['accent_text']};
    }}
    QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus {{
        border-color: {t['accent']};
    }}
    QComboBox, QSpinBox, QDoubleSpinBox {{ min-height: 22px; }}
    QComboBox QAbstractItemView {{
        border: 1px solid {t['border']};
        background: {t['base']};
        selection-background-color: {t['accent']};
        selection-color: {t['accent_text']};
        outline: none;
    }}

    QListWidget, QTreeWidget, QTableView {{
        border: 1px solid {t['border']};
        border-radius: 5px;
        background: {t['base']};
        alternate-background-color: {t['alt_base']};
    }}
    QListWidget::item, QTreeWidget::item {{ padding: 3px 2px; }}
    QHeaderView::section {{
        background: {t['alt_base']};
        color: {t['muted']};
        padding: {SPACE_XS}px {SPACE_S}px;
        border: none;
        border-right: 1px solid {t['border']};
        border-bottom: 1px solid {t['border']};
        font-weight: 600;
    }}

    QProgressBar {{
        border: 1px solid {t['border']};
        border-radius: 5px;
        background: {t['alt_base']};
        text-align: center;
        min-height: 18px;
    }}
    QProgressBar::chunk {{ background: {t['accent']}; border-radius: 4px; }}

    QSplitter::handle {{ background: transparent; }}
    QSplitter::handle:horizontal {{ width: {SPACE_S}px; }}
    QSplitter::handle:vertical {{ height: {SPACE_S}px; }}
    QStatusBar {{ color: {t['muted']}; border-top: 1px solid {t['border']}; }}
    QToolTip {{
        background: {t['base']};
        color: {t['text']};
        border: 1px solid {t['border']};
        padding: {SPACE_XS}px;
    }}

    /* -- semantic label classes --------------------------------------- */
    QLabel[class="heading"] {{ font-size: 15px; font-weight: 600; }}
    QLabel[class="metric"] {{ font-size: 14px; font-weight: 600; }}
    QLabel[class="hint"] {{ color: {t['muted']}; }}
    QLabel[class="error"] {{ color: {t['danger']}; font-weight: 600; }}
    QLabel[class="success"] {{ color: {t['success']}; }}
    QLabel[class="warning"] {{ color: {t['warning']}; }}
    QLabel[class="callout"] {{
        background: {t['alt_base']};
        border-left: 3px solid {t['accent']};
        padding: {SPACE_S}px;
    }}
    """


def app_icon(size=256):
    """The window/taskbar icon: a search trajectory converging on an optimum.

    Painted rather than loaded from a file so there is no asset to add to
    packaging/gui.spec and nothing to go missing from a built bundle.
    """
    t = tokens()
    pixmap = QtGui.QPixmap(size, size)
    pixmap.fill(QtGui.QColor(0, 0, 0, 0))

    painter = QtGui.QPainter(pixmap)
    try:
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        painter.setBrush(QtGui.QBrush(QtGui.QColor(t["accent"])))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(0, 0, size, size, size * 0.22, size * 0.22)

        # A damped spiral into the centre: the search trajectory this whole
        # project is about.
        import math

        path = QtGui.QPainterPath()
        centre = size / 2
        points = []
        for step in range(140):
            angle = step * 0.34
            radius = (size * 0.36) * math.exp(-step / 45)
            points.append((centre + radius * math.cos(angle),
                           centre + radius * math.sin(angle)))
        path.moveTo(*points[0])
        for point in points[1:]:
            path.lineTo(*point)

        pen = QtGui.QPen(QtGui.QColor(t["accent_text"]))
        pen.setWidthF(size * 0.035)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QBrush(QtGui.QColor(t["accent_text"])))
        dot = size * 0.055
        painter.drawEllipse(QtCore.QPointF(centre, centre), dot, dot)
    finally:
        painter.end()
    return QtGui.QIcon(pixmap)


def restyle(widget):
    """Re-evaluate `widget`'s style after its `class` property changed.

    Qt caches the resolved stylesheet per widget, so setProperty alone has no
    visible effect once the widget is on screen.
    """
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()


def set_class(widget, name):
    widget.setProperty("class", name)
    restyle(widget)


def monospace_font(widget, delta=-1):
    """A real fixed-width font. Asking for the family "monospace" only works
    where fontconfig aliases it, which is not Windows or macOS."""
    font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)
    font.setPointSize(max(widget.font().pointSize() + delta, 8))
    return font


def heading_font(widget, delta=1):
    font = QtGui.QFont(widget.font())
    font.setBold(True)
    font.setPointSize(max(widget.font().pointSize() + delta, 8))
    return font


__all__ = [
    "SPACE_XS", "SPACE_S", "SPACE_M", "SPACE_L",
    "apply", "refresh", "tokens", "is_dark", "mode", "set_mode",
    "on_change", "off_change",
    "stylesheet", "app_icon", "restyle", "set_class",
    "monospace_font", "heading_font",
]
