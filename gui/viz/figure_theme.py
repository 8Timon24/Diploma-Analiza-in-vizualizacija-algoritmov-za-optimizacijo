"""Matplotlib styling for figures shown inside the app.

Kept separate from gui/viz/style.py, which despite its name is a palette and
label *lookup* (which colour is OriginalGWO, what is this metric called) and
sets no rcParams at all. This module is the actual styling.

Two problems it solves:

  * Every figure rendered with matplotlib's stock defaults - white face, no
    grid, a full four-spine black box, 10pt type - so on a dark desktop each
    canvas was a bright white slab in a dark window.
  * Because seaborn >= 0.11 no longer mutates rcParams on import, the
    sns.heatmap / sns.lineplot calls throughout gui/viz/ were producing
    matplotlib-default chrome rather than the seaborn look the code reads as
    if it expects.

This is deliberately app-only. The pipeline scripts that write figures_*/
keep their own appearance so the figures already in the thesis stay
consistent with each other; the same plot function called from the GUI is
restyled here, at render time, without touching those scripts.
"""
from contextlib import contextmanager

import matplotlib as mpl

from gui import theme

# One sequential map for magnitude and one diverging map for signed values,
# replacing the six that were in use (YlGnBu, YlOrRd, Blues, RdYlBu_r,
# coolwarm, and tab20 misused as a continuous map). Both are perceptually
# uniform and stay readable in greyscale.
SEQUENTIAL = "viridis"
SEQUENTIAL_ALT = "magma"
DIVERGING = "RdBu_r"

# Legends were set as low as 7pt, and one annotation at 5.5pt, on a 100-DPI
# canvas. Nothing in a figure goes below this.
MIN_FONT_SIZE = 9


def rc_params(dark=None):
    """The rcParams for the active theme."""
    dark = theme.is_dark() if dark is None else dark
    t = theme.tokens()
    face = t["base"]
    text = t["text"]
    muted = t["muted"]
    grid = "#3a414c" if dark else "#dfe4ea"

    return {
        "figure.facecolor": face,
        "figure.edgecolor": face,
        "savefig.facecolor": face,
        "savefig.edgecolor": face,
        "axes.facecolor": face,
        "axes.edgecolor": muted,
        "axes.labelcolor": text,
        "axes.titlecolor": text,
        "text.color": text,
        "xtick.color": muted,
        "ytick.color": muted,
        "xtick.labelcolor": text,
        "ytick.labelcolor": text,

        # A grid under the data instead of a box around it.
        "axes.grid": True,
        "axes.grid.axis": "both",
        "grid.color": grid,
        "grid.linewidth": 0.8,
        "grid.alpha": 0.9,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,

        "font.size": 10.5,
        "axes.titlesize": 12.5,
        "axes.titleweight": "600",
        "axes.labelsize": 10.5,
        "xtick.labelsize": MIN_FONT_SIZE,
        "ytick.labelsize": MIN_FONT_SIZE,
        "legend.fontsize": MIN_FONT_SIZE,
        "legend.title_fontsize": MIN_FONT_SIZE + 0.5,
        "figure.titlesize": 13.5,
        "figure.titleweight": "600",

        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.facecolor": face,
        "legend.edgecolor": grid,
        "legend.borderpad": 0.6,

        "image.cmap": SEQUENTIAL,
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
        "patch.edgecolor": face,
    }


@contextmanager
def styled(dark=None):
    """Draw everything inside this block with the app's figure style."""
    with mpl.rc_context(rc_params(dark)):
        yield


def apply_to(figure, dark=None):
    """Restyle a figure that was already drawn.

    rc_context only affects artists created inside it, and the pipeline's
    plot functions build seaborn objects (clustermaps, FacetGrids) whose
    colours are chosen at construction. Anything that came out with a stock
    white face is corrected here.
    """
    if figure is None:
        return figure
    dark = theme.is_dark() if dark is None else dark
    t = theme.tokens()
    face, text, muted = t["base"], t["text"], t["muted"]
    grid = "#3a414c" if dark else "#dfe4ea"

    figure.patch.set_facecolor(face)
    for axes in figure.get_axes():
        axes.set_facecolor(face)
        for spine in axes.spines.values():
            spine.set_edgecolor(muted)
        axes.tick_params(colors=muted, labelcolor=text)
        axes.xaxis.label.set_color(text)
        axes.yaxis.label.set_color(text)
        # Only recolour a title that is still the default. Several figures
        # colour theirs deliberately - metrics.metric_added_value marks the
        # most/least similar pair - and that meaning must survive.
        if axes.get_title() and _is_default_colour(axes.title.get_color()):
            axes.title.set_color(text)
        for label in list(axes.get_xticklabels()) + list(axes.get_yticklabels()):
            label.set_color(text)
            if label.get_fontsize() < MIN_FONT_SIZE:
                label.set_fontsize(MIN_FONT_SIZE)
        for spine_axis in (axes.xaxis, axes.yaxis):
            spine_axis.grid(True, color=grid, linewidth=0.8, alpha=0.9)
        axes.set_axisbelow(True)

        legend = axes.get_legend()
        if legend is not None:
            _restyle_legend(legend, face, text, grid)

    for legend in getattr(figure, "legends", []):
        _restyle_legend(legend, face, text, grid)

    for text_artist in figure.texts:
        if _is_default_colour(text_artist.get_color()):
            text_artist.set_color(text)
    return figure


def _is_default_colour(colour):
    """True for matplotlib's default text colour, so a deliberate one is left
    alone. rcParams already carry the theme, hence both black and the active
    text colour count as 'not chosen by the plot'."""
    import matplotlib.colors as mcolors

    try:
        rgba = mcolors.to_rgba(colour)
    except ValueError:
        return True
    defaults = {mcolors.to_rgba("black"), mcolors.to_rgba("k")}
    for candidate in (theme.tokens()["text"], mpl.rcParams.get("text.color")):
        try:
            defaults.add(mcolors.to_rgba(candidate))
        except (ValueError, TypeError):
            pass
    return rgba in defaults


def _restyle_legend(legend, face, text, grid):
    frame = legend.get_frame()
    frame.set_facecolor(face)
    frame.set_edgecolor(grid)
    frame.set_alpha(0.9)
    for entry in legend.get_texts():
        entry.set_color(text)
        if entry.get_fontsize() < MIN_FONT_SIZE:
            entry.set_fontsize(MIN_FONT_SIZE)
    title = legend.get_title()
    if title is not None:
        title.set_color(text)
        if title.get_fontsize() < MIN_FONT_SIZE:
            title.set_fontsize(MIN_FONT_SIZE)


def cap_figure_size(figure, max_inches=44):
    """Keep a data-driven figure size within something a canvas can show.

    clustering.cluster_occupancy sizes by iteration count (0.35in each) and
    metrics.metric_table by algorithm pair (0.42in each), so a full run asks
    for figures tens of feet tall. Beyond a point the extra height buys no
    legibility and only makes the scroll area unusable.
    """
    width, height = figure.get_size_inches()
    if height > max_inches or width > max_inches:
        scale = max_inches / max(width, height)
        figure.set_size_inches(width * scale, height * scale)
    return figure
