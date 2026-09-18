"""Theme-aware SVG icons, from the vendored Lucide set in gui/assets/icons/.

Every icon ships with stroke="currentColor", so recolouring is a text
substitution before rendering rather than a pixel operation - the same
"read the active palette, redraw" approach theme.py already uses everywhere
else. Nothing here is cached across a theme change: callers ask again after
theme.refresh() (main_window reloads tab/button icons the same way it
already reacts to colorSchemeChanged).
"""
from functools import lru_cache
from pathlib import Path

from gui.qt import QtCore, QtGui, QtSvg
from gui import theme

ICONS_DIR = Path(__file__).resolve().parent / "assets" / "icons"


@lru_cache(maxsize=None)
def _raw_svg(name):
    path = ICONS_DIR / f"{name}.svg"
    return path.read_text(encoding="utf-8")


def icon(name, color=None, size=18):
    """A QIcon for `name` (an svg file under assets/icons/, no extension),
    recoloured to `color` (a theme token colour, default the muted text
    token so it matches unselected tab labels) and rendered at `size` px.
    """
    color = color or theme.tokens()["muted"]
    svg = _raw_svg(name).replace("currentColor", color)

    renderer = QtSvg.QSvgRenderer(QtCore.QByteArray(svg.encode("utf-8")))
    pixmap = QtGui.QPixmap(size, size)
    pixmap.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(pixmap)
    try:
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        renderer.render(painter)
    finally:
        painter.end()
    return QtGui.QIcon(pixmap)


# matplotlib's NavigationToolbar2QT ships its own icon set (a different
# visual language than the rest of the app), keyed by these method names in
# its private-but-stable `_actions` dict. Mapped to the closest Lucide icon
# so the Trajectory/Compare/Visualize toolbars read as part of the same app
# as the tab bar rather than a library default dropped in unstyled.
_TOOLBAR_ICONS = {
    "home": "house",
    "back": "arrow-left",
    "forward": "arrow-right",
    "pan": "move",
    "zoom": "zoom-in",
    "configure_subplots": "layout-grid",
    "edit_parameters": "sliders-vertical",
    "save_figure": "save",
}


def restyle_toolbar(toolbar, color=None, size=18):
    """Swap a NavigationToolbar2QT's stock icons for the app's own set.

    Safe to call once, right after construction. Silently skips any action
    name matplotlib doesn't have in this version rather than raising, since
    the toolbar's exact action set has changed across matplotlib releases.
    """
    for name, icon_name in _TOOLBAR_ICONS.items():
        action = toolbar._actions.get(name)
        if action is not None:
            action.setIcon(icon(icon_name, color, size))


__all__ = ["icon", "restyle_toolbar", "ICONS_DIR"]
