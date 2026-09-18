# Pipeline step "cosine_columns_pairwise": per-cluster-column cosine distance
# between every pair of algorithms (averaged across clusters), per run.
# Writes metrics_data/cosine_columns/dim_{d}/F{f}_I{i}.csv.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import os
from itertools import combinations
import config
from config import (
    ALGORITHMS_OF_INTEREST, DIMENSIONS,
    CLUSTER_DISTRIBUTIONS_LATEST as INPUT_DIR, METRICS_DIR as OUTPUT_DIR,
)
from pipeline_api import Progress, StageResult

STAGE = "cosine_columns_pairwise"


def column_cosine_distance(tableA, tableB):
    n_clusters = tableA.shape[1]
    sims = []
    for c in range(n_clusters):
        a = tableA[:, c]
        b = tableB[:, c]
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)

        if na == 0 and nb == 0:
            continue  # neither visited this cluster -> irrelevant, skip
        if na == 0 or nb == 0:
            sims.append(0.0)  # exactly one visited -> maximally dissimilar here
            continue

        sims.append(float(np.dot(a, b) / (na * nb)))

    if not sims:
        return None  # no cluster was visited by either (shouldn't happen normally)

    mean_sim = float(np.mean(sims))
    return 1.0 - mean_sim


def run(progress_cb=None, cancel_event=None, dimensions=None):
    """Per-cluster-column cosine distance between every algorithm pair."""
    dims = list(dimensions if dimensions is not None else DIMENSIONS)
    base_input = config.CLUSTER_DISTRIBUTIONS_LATEST
    output_dir = config.METRICS_DIR
    progress = Progress(progress_cb, cancel_event)
    result = StageResult(STAGE)
    os.makedirs(output_dir, exist_ok=True)
    algorithm_pairs = list(combinations(sorted(ALGORITHMS_OF_INTEREST), 2))

    for d in dims:
        os.makedirs(f'{output_dir}/cosine_columns/dim_{d}', exist_ok=True)
        input_dir = f'{base_input}/dim_{d}'

        files = [f for f in sorted(os.listdir(input_dir)) if f.endswith('.csv')]
        progress.begin(len(files), f"cosine columns dim {d}")
        for file in files:
            if not progress.item(file):
                result.cancelled = True
                return result
            problem_name = file.replace('.csv', '')   # "F1_I1"
            prob = problem_name.split('_')[0]          # "F1"
            inst = problem_name.split('_')[1]          # "I1"

            # index levels: algorithm, run, iteration; columns: clusters
            df = pd.read_csv(f'{input_dir}/{file}', index_col=[0, 1, 2])

            rows = []
            # Runs actually present in this file's "run" index level, not
            # config.SEEDS: a GUI run (or a narrower -s) can use any seed.
            for r in sorted(df.index.get_level_values(1).unique()):
                # extract each algorithm's table ONCE for this run, instead of
                # re-doing the (slow) MultiIndex .loc lookup for every pair it
                # appears in (~27x redundant otherwise).
                tables = {}
                for alg in ALGORITHMS_OF_INTEREST:
                    try:
                        tables[alg] = df.loc[(alg, r)].sort_index().to_numpy()
                    except KeyError:
                        tables[alg] = None

                for pair in algorithm_pairs:
                    alg1, alg2 = pair
                    tableA = tables[alg1]
                    tableB = tables[alg2]
                    if tableA is None or tableB is None:
                        continue

                    if tableA.shape != tableB.shape or tableA.size == 0:
                        continue

                    distance = column_cosine_distance(tableA, tableB)
                    if distance is None:
                        continue

                    row = {"Algorithm1": alg1, "Algorithm2": alg2,
                           "Function_id": prob, "Instance_id": inst, "Run_id": r,
                           "Cosine_column_distance": distance}
                    rows.append(row)

            result_frame = pd.DataFrame(rows)
            result_frame.to_csv(f'{output_dir}/cosine_columns/dim_{d}/{problem_name}.csv', index=False)
            result.written += 1

    return result


def main():
    run()


if __name__ == "__main__":
    main()
