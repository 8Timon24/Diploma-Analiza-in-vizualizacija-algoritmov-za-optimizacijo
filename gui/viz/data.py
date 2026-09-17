"""Loaders for the visualizations that read raw run output.

These read outputs/ directly - the per-run diversity CSVs and results.csv -
rather than any of the aggregated metric tables, which is why the plots built
on them were previously only reachable by running a notebook.
"""
import os
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config

DIVERSITY_COLUMNS = ["iteration", "diversity", "exploration", "exploitation"]


def run_dir(dimension, algorithm, function, instance, outputs_dir=None):
    """outputs/dim_{d}/{algorithm}/{f}_{i} - note the bare-int folder name."""
    base = outputs_dir or config.OUTPUTS_DIR
    return Path(base) / f"dim_{dimension}" / algorithm / f"{function}_{instance}"


def load_diversity_curves(dimension, algorithms, function, instance, seeds,
                          outputs_dir=None):
    """Long frame: algorithm, run, iteration, diversity, exploration, exploitation.

    Missing files are skipped rather than raising: diversity is only written
    when the benchmark ran with -e, so a partial set is a normal state.
    """
    frames = []
    for algorithm in algorithms:
        for seed in seeds:
            path = run_dir(dimension, algorithm, function, instance, outputs_dir) / f"diversity_{seed}.csv"
            if not path.is_file():
                continue
            frame = pd.read_csv(path).sort_values("iteration")
            frame["algorithm"] = algorithm
            frame["run"] = seed
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=DIVERSITY_COLUMNS + ["algorithm", "run"])
    return pd.concat(frames, ignore_index=True)


@lru_cache(maxsize=512)
def bbob_optimum(function, instance, dimension):
    """(f_opt, x_opt) for a BBOB problem.

    cocoex does not expose the optimum publicly on this build; its internal
    _best_parameter writes x_opt to a file in the CURRENT directory. The
    call is therefore made from a temporary directory so nothing is ever
    written into the repo, and f_opt is evaluated on a fresh problem so it
    does not consume a real run's evaluation budget.
    """
    import cocoex

    suite = cocoex.Suite(
        "bbob", "instances:1-110",
        f"function_indices:{function} instance_indices:{instance} dimensions:{dimension}",
    )
    problem = suite.next_problem()

    previous = os.getcwd()
    scratch = tempfile.mkdtemp(prefix="bbob_optimum_")
    try:
        os.chdir(scratch)
        problem._best_parameter(what="print")
        x_opt = np.loadtxt("._bbob_problem_best_parameter.txt")
    finally:
        os.chdir(previous)
        for leftover in Path(scratch).iterdir():
            leftover.unlink()
        Path(scratch).rmdir()

    return float(problem(x_opt)), np.atleast_1d(x_opt)


def load_results(dimension, outputs_dir=None):
    base = outputs_dir or config.OUTPUTS_DIR
    path = Path(base) / f"dim_{dimension}" / "results.csv"
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def results_with_precision(dimension, function, instance, algorithms=None,
                           outputs_dir=None):
    """results.csv for one problem, with precision = fitness - f_opt.

    Clipped at 1e-12 so the log scale stays usable: a run that lands exactly
    on the optimum would otherwise be log(0).
    """
    results = load_results(dimension, outputs_dir)
    if results.empty:
        return results, None, None

    subset = results[
        (results["problem_id"] == function) & (results["instance_id"] == instance)
    ].copy()
    if algorithms:
        subset = subset[subset["algorithm"].isin(algorithms)]
    if subset.empty:
        return subset, None, None

    f_opt, x_opt = bbob_optimum(function, instance, dimension)
    subset["f_opt"] = f_opt
    subset["precision"] = (subset["fitness"] - f_opt).clip(lower=1e-12)
    return subset, f_opt, x_opt


def coordinate_columns(frame):
    """x1..xd in results.csv (1-indexed, unlike the processed data's x0..)."""
    return [c for c in frame.columns if c.startswith("x") and c[1:].isdigit()]


def pairwise_matrix(algorithms, value_for_pair):
    """Symmetric DataFrame from a function over unordered algorithm pairs."""
    from itertools import combinations

    matrix = pd.DataFrame(
        np.full((len(algorithms), len(algorithms)), np.nan),
        index=algorithms, columns=algorithms,
    )
    for first, second in combinations(algorithms, 2):
        value = value_for_pair(first, second)
        matrix.loc[first, second] = value
        matrix.loc[second, first] = value
    for algorithm in algorithms:
        matrix.loc[algorithm, algorithm] = 0.0
    return matrix


# --------------------------------------------------------------------------
# clustering stage outputs
# --------------------------------------------------------------------------

def problem_key(function, instance):
    """F{f}_I{i} - the naming used from data/processed onward. Note that the
    raw outputs/ tree uses bare {f}_{i} instead."""
    return f"F{function}_I{instance}"


def clustering_paths(dimension, function, instance, method="kmeans"):
    """The three files the clustering stage writes per problem."""
    base = Path(config.clustering_dir(method))
    key = problem_key(function, instance)
    return {
        "distributions": base / "cluster_distributions" / f"dim_{dimension}" / f"{key}.csv",
        "centers": base / "cluster_centers" / f"dim_{dimension}" / f"{key}.csv",
        "results": base / "clustering_results" / f"dim_{dimension}" / f"{key}.parquet",
    }


def load_cluster_occupancy(dimension, function, instance, method="kmeans"):
    """(occupancy, centre_labels).

    occupancy is indexed by (algorithm, run, iteration) with one column per
    cluster; centre_labels maps a cluster id to its rounded coordinates so
    columns can say where a cluster actually is.
    """
    paths = clustering_paths(dimension, function, instance, method)
    if not paths["distributions"].is_file():
        raise ValueError(f"No cluster distributions for {problem_key(function, instance)} "
                         f"at dimension {dimension}.")

    occupancy = pd.read_csv(paths["distributions"], index_col=[0, 1, 2])

    labels = {}
    if paths["centers"].is_file():
        centers = pd.read_csv(paths["centers"], index_col=0)
        labels = {
            int(index): ", ".join(f"{row[c]:.2f}" for c in centers.columns)
            for index, row in centers.iterrows()
        }
    return occupancy, labels


def load_clustered_trajectories(dimension, function, instance, method="kmeans"):
    paths = clustering_paths(dimension, function, instance, method)
    if not paths["results"].is_file():
        raise ValueError(f"No clustering results parquet for "
                         f"{problem_key(function, instance)} at dimension {dimension}.")
    return pd.read_parquet(paths["results"])


def load_algorithm_similarity(dimension, statistic="mean", method="kmeans"):
    """The aggregate cosine similarity matrix cluster_similarity.py writes."""
    base = Path(config.clustering_dir(method)) / config.SIMILARITY_OUTPUT_SUBDIR
    path = base / f"algorithm_{statistic}_similarity_{dimension}D.csv"
    if not path.is_file():
        raise ValueError(
            f"No aggregate similarity file for dimension {dimension} "
            f"(expected {path.name}). Run 03_cluster/cluster_similarity.py."
        )
    frame = pd.read_csv(path, index_col=0)
    return frame.pivot(index="algorithm", columns="algorithm2", values="value")


# --------------------------------------------------------------------------
# merged pairwise metrics
# --------------------------------------------------------------------------

@lru_cache(maxsize=4)
def load_merged(dimension):
    path = Path(config.MERGED_DIR) / f"merged_dim_{dimension}.csv"
    if not path.is_file():
        raise ValueError(
            f"metrics_data/merged/merged_dim_{dimension}.csv does not exist. "
            f"Run 05_analysis/merge_metrics.py."
        )
    return pd.read_csv(path)


def pair_means(dimension, metrics, algorithms=None):
    """Mean of each metric per algorithm pair, averaged over every problem."""
    merged = load_merged(dimension)
    available = [m for m in metrics if m in merged.columns]
    if not available:
        raise ValueError(
            f"merged_dim_{dimension}.csv has none of {metrics}. "
            f"It has: {sorted(set(merged.columns) - set(config.METRIC_KEYS))}"
        )
    if algorithms:
        chosen = set(algorithms)
        merged = merged[
            merged["Algorithm1"].isin(chosen) & merged["Algorithm2"].isin(chosen)
        ]
    if merged.empty:
        raise ValueError("No pairs in the merged table for those algorithms.")
    return merged.groupby(["Algorithm1", "Algorithm2"])[available].mean().reset_index()
