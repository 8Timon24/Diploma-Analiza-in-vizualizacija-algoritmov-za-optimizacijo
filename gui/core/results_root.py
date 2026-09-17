"""Point the app at a results tree other than the one it was started from.

config.py derives every data directory from REPO_ROOT at import time, which
is right for the pipeline - it always runs inside the checkout - and wrong
for a GUI, which may be a packaged app with no checkout at all, or may want
to look at a run it just produced in its own workspace.

Every GUI module reads these paths through `config.<NAME>` at call time
rather than aliasing them at import, so rebinding the attributes here really
does repoint the app. The pipeline's own plotting functions are unaffected:
the GUI loads their input itself and hands them a DataFrame.

Caches keyed on those paths must be dropped when the root moves, or the app
keeps showing the previous tree's data.
"""
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config

# The directory layout every results tree has, mirroring config.py.
SUBPATHS = {
    "OUTPUTS_DIR": "outputs",
    "DATA_DIR": "data",
    "PROCESSED_DIR": "data/processed",
    "CLUSTERING_LATEST_DIR": "data/clustering_latest",
    "CLUSTER_DISTRIBUTIONS_LATEST": "data/clustering_latest/cluster_distributions",
    "CLUSTERING_DBSCAN_DIR": "data/clustering_features_10_algorithms_dbscan",
    "ENTROPY_DATA_DIR": "data/entropy",
    "METRICS_DIR": "metrics_data",
    "MERGED_DIR": "metrics_data/merged",
    "SCALARS_CSV": "metrics_data/scalars.csv",
    "FIGURES_ENTROPY_DIR": "figures_entropy",
    "FIGURES_SPEARMAN_DIR": "figures_spearman",
    "FIGURES_RESULTS_DIR": "figures_results",
}

# What has to be present for each part of the app to have anything to show.
CAPABILITIES = {
    "Raw trajectories": "outputs",
    "Processed trajectories": "data/processed",
    "Clustering": "data/clustering_latest/cluster_distributions",
    "Entropy": "data/entropy",
    "Pairwise metrics": "metrics_data",
    "Merged metrics": "metrics_data/merged",
}

DEFAULT_ROOT = Path(config.REPO_ROOT)
_current = DEFAULT_ROOT


@dataclass(frozen=True)
class RootSummary:
    """What a candidate results folder actually contains."""

    path: Path
    present: tuple
    missing: tuple

    @property
    def is_usable(self):
        return bool(self.present)

    def describe(self):
        if not self.present:
            return "No pipeline output found here."
        text = "Has: " + ", ".join(self.present)
        if self.missing:
            text += "   -   missing: " + ", ".join(self.missing)
        return text


def inspect(path):
    """Report which stages of output exist under a candidate root."""
    path = Path(path)
    present, missing = [], []
    for label, subpath in CAPABILITIES.items():
        (present if (path / subpath).exists() else missing).append(label)
    return RootSummary(path=path, present=tuple(present), missing=tuple(missing))


def current_root():
    return _current


def is_default():
    return _current == DEFAULT_ROOT


def set_root(path):
    """Repoint every data directory, and drop anything cached from the old one."""
    global _current
    path = Path(path).resolve()
    for name, subpath in SUBPATHS.items():
        setattr(config, name, str(path / subpath))
    config.REPO_ROOT = path
    _current = path
    invalidate_caches()
    return inspect(path)


def reset():
    return set_root(DEFAULT_ROOT)


def invalidate_caches():
    """Forget every path-keyed cache.

    Each of these memoises a table read from the results tree; leaving any of
    them warm after a move shows the previous tree's numbers under the new
    tree's name, which is worse than an error.
    """
    from gui.core import coverage

    coverage.invalidate()

    # Imported lazily: these pull in seaborn/matplotlib, which a headless
    # caller (and the test suite) should not be forced to load.
    try:
        from gui.viz import data as viz_data

        viz_data.load_merged.cache_clear()
    except Exception:
        pass
    try:
        from gui.viz import registry

        registry._entropy_table.cache_clear()
        registry._spearman_matrix.cache_clear()
        registry._scalars.cache_clear()
    except Exception:
        pass
    try:
        from gui.core import problems

        problems._OpfunuSourceBase._catalog_cache.clear()
    except Exception:
        pass
