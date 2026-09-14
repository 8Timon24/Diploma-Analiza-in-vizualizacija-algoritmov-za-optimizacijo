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
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# --- Algorithms under study (mealpy optimizer class names) ---
ALGORITHMS_OF_INTEREST = [
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
]

# --- BBOB benchmark sweep parameters ---
DIMENSIONS = [2, 5, 10]
FUNCTIONS = list(range(1, 25))
INSTANCES = list(range(1, 6))
SEEDS = [1, 2, 3, 4, 5]

# --- Directory layout (all ABSOLUTE, anchored to REPO_ROOT - see module docstring) ---
OUTPUTS_DIR = str(REPO_ROOT / "outputs")                    # raw per-run benchmark trajectories
DATA_DIR = str(REPO_ROOT / "data")
PROCESSED_DIR = f"{DATA_DIR}/processed"                     # reshaped trajectories, input to clustering
CLUSTERING_LATEST_DIR = f"{DATA_DIR}/clustering_latest"
CLUSTER_DISTRIBUTIONS_LATEST = f"{CLUSTERING_LATEST_DIR}/cluster_distributions"
CLUSTERING_DBSCAN_DIR = f"{DATA_DIR}/clustering_features_10_algorithms_dbscan"
ENTROPY_DATA_DIR = f"{DATA_DIR}/entropy"
RETURN_RATE_DATA_DIR = f"{DATA_DIR}/return_rate"

METRICS_DIR = str(REPO_ROOT / "metrics_data")               # per-metric pairwise CSVs
MERGED_DIR = f"{METRICS_DIR}/merged"
CLUSTER_DISTRIBUTIONS_SUBDIR = "cluster_distributions"
SIMILARITY_OUTPUT_SUBDIR = "algorithm_pairwise_similarity"

# Figure output directories (also absolute, for the same reason as above).
FIGURES_ENTROPY_DIR = str(REPO_ROOT / "figures_entropy")
FIGURES_SPEARMAN_DIR = str(REPO_ROOT / "figures_spearman")
FIGURES_REVISITING_DIR = str(REPO_ROOT / "figures_revisiting")
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
