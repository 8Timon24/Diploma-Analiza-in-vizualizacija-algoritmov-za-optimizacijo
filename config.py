"""Shared constants, paths, and helpers for the trajectory-analysis pipeline.

Single source of truth for the algorithm list, benchmark sweep parameters
(dimensions/functions/instances/seeds), the directory layout, and the
pairwise-metric CSV schema. Import from here instead of re-declaring these
in individual pipeline scripts.

Every directory constant below is an ABSOLUTE path anchored to this file's own
location (the repo root), not a bare relative string. Pipeline scripts now
live in stage subfolders (01_optimize/, 04_metrics/, ...) and are still run
directly by path - a relative string like "data/entropy" would silently
resolve against whatever directory the caller's shell happens to be in, and
create a wrong, nested copy if someone `cd`s into a subfolder first. Anchoring
to REPO_ROOT makes every script write to the real top-level folders no matter
where it's invoked from.

Test isolation: REPO_ROOT and the sweep parameters below can each be
overridden by an environment variable (PIPELINE_TEST_ROOT, and a
PIPELINE_TEST_* variable per sweep list, JSON-encoded). tests/test_pipeline_smoke.py
uses this to point an entire real pipeline run at a throwaway temp directory
with a tiny synthetic sweep, without ever touching the real data/outputs/
metrics_data or running the real 28-algorithm/24-function universe. These
variables are unset in normal use, so this never changes real behavior.
"""

import json
import os
import re
from pathlib import Path

_root_override = os.environ.get("PIPELINE_TEST_ROOT")
REPO_ROOT = Path(_root_override).resolve() if _root_override else Path(__file__).resolve().parent


def _list_override(env_var, default):
    raw = os.environ.get(env_var)
    return json.loads(raw) if raw is not None else default


# --- Algorithms under study (mealpy optimizer class names) ---
ALGORITHMS_OF_INTEREST = _list_override("PIPELINE_TEST_ALGORITHMS", [
    "AugmentedAEO", "GWO_WOA",
    "HI_WOA", "IGWO",
    "ImprovedBSO", "JADE",
    "L_SHADE", "LevyTWO",
    "ModifiedAEO", "OriginalAEO",
    "OriginalALO", "OriginalCSA",
    "OriginalDE", "OriginalFPA",
    "OriginalGWO", "OriginalHC",
    "OriginalHHO", "OriginalMFO",
    "OriginalMPA", "OriginalMRFO",
    "OriginalNMRA", "OriginalSHADE",
    "OriginalSSA", "OriginalSSpiderA",
    "OriginalWOA", "RW_GWO",
    "SADE", "WhaleFOA",
])

# --- BBOB benchmark sweep parameters ---
DIMENSIONS = _list_override("PIPELINE_TEST_DIMENSIONS", [2, 5, 10])
FUNCTIONS = _list_override("PIPELINE_TEST_FUNCTIONS", list(range(1, 25)))
INSTANCES = _list_override("PIPELINE_TEST_INSTANCES", list(range(1, 6)))
SEEDS = _list_override("PIPELINE_TEST_SEEDS", [1, 2, 3, 4, 5])

# Seed for the clustering step. sklearn's KMeans picks its initial centroids
# randomly, so without this the pipeline is not reproducible: re-running it can
# choose a different k AND assign different cluster labels, which shifts every
# metric computed downstream from cluster occupancy (entropy, cosine,
# cosine_columns). Changing this value changes clustering results.
CLUSTERING_SEED = 42

# --- Directory layout (all ABSOLUTE, anchored to REPO_ROOT - see module docstring) ---
OUTPUTS_DIR = str(REPO_ROOT / "outputs")                    # raw per-run benchmark trajectories
DATA_DIR = str(REPO_ROOT / "data")
PROCESSED_DIR = f"{DATA_DIR}/processed"                     # reshaped trajectories, input to clustering
CLUSTERING_LATEST_DIR = f"{DATA_DIR}/clustering_latest"
CLUSTER_DISTRIBUTIONS_LATEST = f"{CLUSTERING_LATEST_DIR}/cluster_distributions"
CLUSTERING_DBSCAN_DIR = f"{DATA_DIR}/clustering_features_10_algorithms_dbscan"
ENTROPY_DATA_DIR = f"{DATA_DIR}/entropy"

METRICS_DIR = str(REPO_ROOT / "metrics_data")               # per-metric pairwise CSVs
MERGED_DIR = f"{METRICS_DIR}/merged"
# Per-ALGORITHM scalars (one row per dim/func/algo), as opposed to the pairwise
# tables above. Written by 05_analysis/build_scalars.py, read by scalar_regression.py.
SCALARS_CSV = f"{METRICS_DIR}/scalars.csv"
CLUSTER_DISTRIBUTIONS_SUBDIR = "cluster_distributions"
SIMILARITY_OUTPUT_SUBDIR = "algorithm_pairwise_similarity"

# Figure output directories (also absolute, for the same reason as above).
FIGURES_ENTROPY_DIR = str(REPO_ROOT / "figures_entropy")
FIGURES_SPEARMAN_DIR = str(REPO_ROOT / "figures_spearman")
FIGURES_RESULTS_DIR = str(REPO_ROOT / "figures_results")

# --- Pairwise-metric CSV schema: metrics_data/<metric>/dim_{d}/F{f}_I{i}.csv ---
METRIC_KEYS = ["Algorithm1", "Algorithm2", "Function_id", "Instance_id", "Run_id"]

# English display labels for figures. The underlying column/metric names above
# stay fixed so merged_dim_*.csv / spearman_dim_*.csv remain stable.
METRIC_LABELS = {
    "entropy": "Entropy",
    "cosine": "Cosine distance",
    "cosine_columns": "Cosine distance (columns)",
    "exploration": "Exploration",
    "location": "Location",
    "fitness": "Quality",
}


CLUSTERING_METHODS = ["kmeans", "dbscan", "dbscan_adaptive"]


def clustering_dir(method):
    """Map a clustering method (the -c flag) to its data directory.

    Both 03_cluster/cluster_trajectories.py (which writes) and
    cluster_similarity.py (which reads) go through this, so the two can't
    disagree about where a given method's data lives. They used to: the
    writer sent 'dbscan_adaptive' to <dbscan>/adaptive/ while the reader's
    else-branch sent it to <dbscan>/, so that mode silently analysed the
    wrong directory.
    """
    if method == "kmeans":
        return f"{CLUSTERING_LATEST_DIR}/"
    if method == "dbscan":
        return f"{CLUSTERING_DBSCAN_DIR}/"
    if method == "dbscan_adaptive":
        return f"{CLUSTERING_DBSCAN_DIR}/adaptive/"
    raise ValueError(f"unknown clustering method {method!r}; expected one of {CLUSTERING_METHODS}")


def discover_seeds(directory, prefix):
    """Seed values actually present as `{prefix}_{seed}.csv` files in `directory`.

    SEEDS is what a fresh benchmark sweep defaults to, not a guarantee about
    what's on disk: a GUI run can use any seed via the Setup tab's add-a-seed
    control (gui/panels/setup_panel.py), and re-running the CLI's own -s flag
    with a narrower list produces the same thing. A stage that assumes SEEDS
    while reading per-seed files by name either crashes on one that was never
    generated or silently ignores one outside that list that was - this is
    the fix for both, used in place of `for seed in SEEDS` wherever a stage
    reads outputs/ files named this way (harvest_results.py, preprocess_data.py,
    build_scalars.py's diversity_scalars).
    """
    if not os.path.isdir(directory):
        return []
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)\.csv$")
    seeds = set()
    for name in os.listdir(directory):
        match = pattern.match(name)
        if match:
            seeds.add(int(match.group(1)))
    return sorted(seeds)


def discover_dimensions(directory, fallback=None):
    """Dimensions actually present as dim_{d} subdirectories of `directory`.

    Same reasoning as discover_seeds: DIMENSIONS is what a fresh benchmark
    sweep defaults to, not a guarantee about what's on disk - a GUI run's
    Setup tab lets any subset of dimensions be benchmarked, and the Process
    tab used to always process exactly DIMENSIONS regardless, crashing on
    os.listdir() the moment one of them (e.g. 5 or 10 out of the default
    [2, 5, 10]) was never generated.

    Falls back to `fallback` (typically DIMENSIONS) if `directory` doesn't
    exist yet or has no dim_* subdirectories - e.g. before any benchmark has
    run - so an empty tree doesn't silently resolve to "process nothing."
    """
    pattern = re.compile(r"^dim_(\d+)$")
    dims = set()
    if os.path.isdir(directory):
        for name in os.listdir(directory):
            match = pattern.match(name)
            if match and os.path.isdir(f"{directory}/{name}"):
                dims.add(int(match.group(1)))
    return sorted(dims) if dims else list(fallback if fallback is not None else DIMENSIONS)


def normalize_keys(df):
    """Coerce Function_id/Instance_id/Run_id to plain ints.

    Different pairwise scripts write Function_id/Instance_id as either an int
    (1) or a string ("F1"/"I1") - this makes them comparable so metric CSVs
    can be merged/joined on METRIC_KEYS regardless of which script wrote them.
    """
    for col, prefix in [("Function_id", "F"), ("Instance_id", "I")]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(prefix, "", regex=False).astype(int)
    if "Run_id" in df.columns:
        df["Run_id"] = df["Run_id"].astype(int)
    return df
