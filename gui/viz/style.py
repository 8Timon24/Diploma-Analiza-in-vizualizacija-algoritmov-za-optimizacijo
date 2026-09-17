"""One palette and one family map for every GUI figure.

The repo currently defines algorithm-family colours three separate ways
(scalar_regression.FAMILY, the family_map in pari_algoritmov_metrike_heatmap,
and utils.get_algorithm_groups) and duplicates METRIC_LABELS in two more
places. Rather than adding a fourth copy, this module builds on the only
non-hardcoded source - utils.get_algorithm_groups(), which asks mealpy - and
re-exports config.METRIC_LABELS.
"""
import sys
from functools import lru_cache
from pathlib import Path

import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import ALGORITHMS_OF_INTEREST, METRIC_LABELS

# The eight optimizers entropy_plotting.py's __main__ plots by default: two
# DE variants, three swarm families, and two deliberate outliers. A sensible
# starting subset, since 28 lines on one axis is unreadable.
DEFAULT_PLOT_ALGORITHMS = [
    "OriginalDE",
    "L_SHADE",
    "OriginalGWO",
    "OriginalWOA",
    "AugmentedAEO",
    "OriginalHC",
    "OriginalMFO",
    "WhaleFOA",
]


@lru_cache(maxsize=1)
def _thesis_palette():
    """The palette entropy_plotting.py uses, rebuilt identically so GUI
    figures match the ones already in figures_entropy/."""
    return dict(
        zip(
            sorted(ALGORITHMS_OF_INTEREST),
            sns.color_palette("husl", n_colors=len(ALGORITHMS_OF_INTEREST)),
        )
    )


def algo_palette(algorithms):
    """Colour per algorithm, stable for the thesis set and extended for any
    optimizer outside it (the GUI can run all 234)."""
    base = _thesis_palette()
    extra = [a for a in sorted(algorithms) if a not in base]
    extra_colours = sns.color_palette("husl", n_colors=max(len(extra), 1))
    palette = {a: base[a] for a in algorithms if a in base}
    palette.update({a: extra_colours[i] for i, a in enumerate(extra)})
    return palette


@lru_cache(maxsize=1)
def family_map():
    """Optimizer name -> mealpy family. Asks mealpy, so it stays current."""
    import utils

    return utils.get_algorithm_groups()


def distinct_palette(algorithms):
    """Maximally distinguishable colours for a handful of algorithms.

    algo_palette() reproduces the thesis palette so GUI figures match the
    ones already in figures_entropy/, but that palette is husl over all 28
    algorithms, which puts neighbours at nearly the same hue - OriginalDE,
    OriginalGWO and OriginalHC all come out teal. That is fine for a
    28-line reference figure nobody reads line-by-line, and useless for a
    live plot comparing three algorithms. Small sets get categorical
    colours instead; large ones fall back to the thesis palette.
    """
    algorithms = list(algorithms)
    if len(algorithms) <= 10:
        colours = sns.color_palette("tab10", n_colors=10)
    elif len(algorithms) <= 20:
        colours = sns.color_palette("tab20", n_colors=20)
    else:
        return algo_palette(algorithms)
    return {name: colours[i] for i, name in enumerate(sorted(algorithms))}


def family_of(algorithm):
    return family_map().get(algorithm, "other")


@lru_cache(maxsize=1)
def family_colours():
    families = sorted(set(family_map().values()))
    colours = sns.color_palette("tab10", n_colors=max(len(families), 1))
    return dict(zip(families, colours))


def family_label(family):
    """Display name for a mealpy family. Re-exported from gui.core.optimizers
    so figure code has a single styling import."""
    from gui.core.optimizers import family_label as _family_label

    return _family_label(family)


def metric_label(metric):
    return METRIC_LABELS.get(metric, metric.replace("_", " ").capitalize())
