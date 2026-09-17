# Pipeline step "build_scalars": builds the per-algorithm scalar table that
# scalar_regression.py consumes - one row per (dim, func, algo) with one column
# per scalar. The rest of the pipeline only produces PAIRWISE data (one row per
# algorithm *pair*), which is a different shape, so scalar_regression.py had no
# input it could actually read until this step existed.
#
# Writes metrics_data/scalars.csv (all dimensions in one file, since
# scalar_regression.py facets by the `dim` column).
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
import pandas as pd
from tqdm import tqdm
import config
from config import (
    ALGORITHMS_OF_INTEREST, DIMENSIONS, FUNCTIONS, INSTANCES, SEEDS,
    ENTROPY_DATA_DIR, OUTPUTS_DIR, METRICS_DIR, SCALARS_CSV,
)
from pipeline_api import Progress, StageResult

STAGE = "build_scalars"


def entropy_scalars(dimension):
    """Mean normalized entropy per (func, algo), averaged over runs/instances/iterations."""
    path = f"{ENTROPY_DATA_DIR}/entropy_granular_dim_{dimension}.csv"
    if not os.path.isfile(path):
        print(f"  [warn] missing {path} - run 04_metrics/entropy.py first")
        return None
    df = pd.read_csv(path)
    df = df[df["algorithm"].isin(ALGORITHMS_OF_INTEREST)]
    return (df.groupby(["problem_id", "algorithm"])["entropy"]
              .mean()
              .reset_index()
              .rename(columns={"problem_id": "func", "algorithm": "algo"}))


def fitness_scalars(dimension):
    """Mean final objective value per (func, algo), averaged over instances/seeds."""
    path = f"{OUTPUTS_DIR}/dim_{dimension}/results.csv"
    if not os.path.isfile(path):
        print(f"  [warn] missing {path} - run 01_optimize/harvest_results.py first")
        return None
    df = pd.read_csv(path)
    df = df[df["algorithm"].isin(ALGORITHMS_OF_INTEREST)]
    return (df.groupby(["problem_id", "algorithm"])["fitness"]
              .mean()
              .reset_index()
              .rename(columns={"problem_id": "func", "algorithm": "algo"}))


def diversity_scalars(dimension):
    """Mean exploration ratio and raw diversity per (func, algo).

    Reads the per-run diversity_{seed}.csv files mealpy's run wrote. The
    exploration column is a percentage (0-100); it is divided by 100 here so it
    matches the 0-1 convention 04_metrics/exploration_pairwise.py uses.
    """
    rows = []
    combos = [(alg, f, i, s)
              for alg in ALGORITHMS_OF_INTEREST
              for f in FUNCTIONS
              for i in INSTANCES
              for s in SEEDS]
    for alg, f, i, s in tqdm(combos, desc=f"diversity/dim_{dimension}", leave=False):
        path = f"{config.OUTPUTS_DIR}/dim_{dimension}/{alg}/{f}_{i}/diversity_{s}.csv"
        if not os.path.isfile(path):
            continue
        d = pd.read_csv(path)
        if d.empty:
            continue
        rows.append({
            "func": f,
            "algo": alg,
            "exploration": d["exploration"].mean() / 100.0,
            "diversity": d["diversity"].mean(),
        })
    if not rows:
        print(f"  [warn] no diversity files under {OUTPUTS_DIR}/dim_{dimension} - "
              f"run the benchmark step with -e")
        return None
    return pd.DataFrame(rows).groupby(["func", "algo"], as_index=False).mean()


def build_dimension(dimension):
    """Join every available scalar for one dimension into a (func, algo) table."""
    parts = [p for p in (entropy_scalars(dimension),
                         fitness_scalars(dimension),
                         diversity_scalars(dimension)) if p is not None]
    if not parts:
        return None

    table = parts[0]
    for other in parts[1:]:
        table = table.merge(other, on=["func", "algo"], how="outer")

    table.insert(0, "dim", dimension)
    return table.sort_values(["dim", "func", "algo"]).reset_index(drop=True)


def run(progress_cb=None, cancel_event=None, dimensions=None):
    """Build the per-algorithm scalar table scalar_regression.py consumes.

    A different shape from everything else in metrics_data/, which is pairwise:
    one row per (dim, func, algo).
    """
    dims = list(dimensions if dimensions is not None else DIMENSIONS)
    progress = Progress(progress_cb, cancel_event)
    result = StageResult(STAGE)

    tables = []
    progress.begin(len(dims), "building scalars")
    for d in dims:
        if not progress.item(f"dim {d}"):
            result.cancelled = True
            return result
        print(f"Building scalars for dim={d}...")
        t = build_dimension(d)
        if t is not None:
            tables.append(t)

    if not tables:
        # Returned, not raised: a SystemExit here would tear down the caller,
        # which in the desktop app is a worker thread.
        result.notes.append("no scalars could be built - is the pipeline data present?")
        return result

    scalars = pd.concat(tables, ignore_index=True)
    os.makedirs(config.METRICS_DIR, exist_ok=True)
    scalars.to_csv(config.SCALARS_CSV, index=False)
    result.written += 1
    result.notes.append(f"{len(scalars):,} rows")
    print(f"\nsaved -> {config.SCALARS_CSV}  ({len(scalars)} rows, "
          f"columns: {list(scalars.columns)})")
    return result


def main():
    result = run()
    if not result.written:
        raise SystemExit("  ".join(result.notes) or "nothing was written")


if __name__ == "__main__":
    main()
