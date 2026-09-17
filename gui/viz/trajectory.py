"""Data behind the animated 2-D trajectory player.

The thesis is about search trajectories, but the repo only ever draws them as
static 3-D scatters. 4_example_visualize_trajectories_for_clustering.ipynb
cell 9 builds a 20-frame loop for one hardcoded problem and shows it inline
without saving anything - this is that idea made into a real control.

Two sources of positions, in order of preference:
  clustering_results/*.parquet  - has a `cluster` column, so points can be
                                  coloured by the cluster assignment every
                                  downstream metric is computed from
  data/processed/*.csv          - the same trajectories before clustering
"""
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config
from gui.viz.data import (
    bbob_optimum, clustering_paths, load_clustered_trajectories, problem_key,
)

# 220x220 is ~48k objective evaluations, which cocoex does in about 0.1s -
# cheap enough to recompute whenever the problem changes.
LANDSCAPE_RESOLUTION = 220

POSITION_COLUMNS = ("x0", "x1")


@lru_cache(maxsize=8)
def landscape(function, instance, dimension=2, resolution=LANDSCAPE_RESOLUTION):
    """(X, Y, Z) of the true objective over the search box.

    Evaluated on a fresh cocoex problem, so it never touches the evaluation
    budget of anything else.
    """
    import cocoex

    suite = cocoex.Suite(
        "bbob", "instances:1-110",
        f"function_indices:{function} instance_indices:{instance} "
        f"dimensions:{dimension}",
    )
    problem = suite.next_problem()
    xs = np.linspace(problem.lower_bounds[0], problem.upper_bounds[0], resolution)
    ys = np.linspace(problem.lower_bounds[1], problem.upper_bounds[1], resolution)
    grid = np.array([[problem([x, y]) for x in xs] for y in ys])
    mesh_x, mesh_y = np.meshgrid(xs, ys)
    return mesh_x, mesh_y, grid


def _load_processed(dimension, function, instance):
    path = Path(config.PROCESSED_DIR) / f"dim_{dimension}" / f"{problem_key(function, instance)}.csv"
    if not path.is_file():
        return None
    return pd.read_csv(path, compression="zip", index_col=0)


def load_positions(dimension, function, instance):
    """Population positions per iteration, with cluster labels when available.

    Returns (frame, has_clusters). Raises with an actionable message when
    neither the clustered nor the processed form exists.
    """
    paths = clustering_paths(dimension, function, instance)
    if paths["results"].is_file():
        return load_clustered_trajectories(dimension, function, instance), True

    processed = _load_processed(dimension, function, instance)
    if processed is not None:
        return processed, False

    raise ValueError(
        f"No trajectories for {problem_key(function, instance)} at dimension "
        f"{dimension}. Run 02_preprocess/preprocess_data.py (and optionally "
        f"03_cluster/cluster_trajectories.py for cluster colouring)."
    )


def select(frame, algorithms, run):
    """Rows for the chosen algorithms and run, sorted by iteration."""
    subset = frame[frame["algorithm"].isin(algorithms) & (frame["run"] == run)]
    if subset.empty:
        raise ValueError(
            f"No trajectory rows for run {run} and those algorithms."
        )
    missing = [c for c in POSITION_COLUMNS if c not in subset.columns]
    if missing:
        raise ValueError(
            f"Trajectory data has no {missing} columns - the player is 2-D only."
        )
    return subset.sort_values("iteration")


def iteration_range(frame):
    return int(frame["iteration"].min()), int(frame["iteration"].max())


def optimum(function, instance, dimension=2):
    """(f_opt, x_opt), or None when it cannot be recovered."""
    try:
        return bbob_optimum(function, instance, dimension)
    except Exception:
        return None
