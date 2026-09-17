# Pipeline step "cosine_pairwise": global cosine distance between every pair
# of algorithms' flattened cluster-occupancy vectors, per run.
# Writes metrics_data/cosine/dim_{d}/F{f}_I{i}.csv.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import os
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
import config
from config import (
    ALGORITHMS_OF_INTEREST, DIMENSIONS, SEEDS as RUNS,
    CLUSTER_DISTRIBUTIONS_LATEST as INPUT_DIR, METRICS_DIR as OUTPUT_DIR,
)
from pipeline_api import Progress, StageResult

STAGE = "cosine_pairwise"


def flatten_algorithm_run(sub):
    """
    Flatten one (algorithm, run)'s cluster-occupancy table (rows=iterations,
    cols=clusters) into a single 1-D feature vector by concatenating the
    per-iteration cluster counts in iteration order. This is the same feature
    representation the cosine-similarity script uses (unstack over iteration).
    """
    return sub.sort_index().to_numpy().flatten()


def run(progress_cb=None, cancel_event=None, dimensions=None):
    """Global cosine distance between every pair of algorithms, per run."""
    dims = list(dimensions if dimensions is not None else DIMENSIONS)
    base_input = config.CLUSTER_DISTRIBUTIONS_LATEST
    output_dir = config.METRICS_DIR
    progress = Progress(progress_cb, cancel_event)
    result = StageResult(STAGE)

    os.makedirs(output_dir, exist_ok=True)
    algorithm_pairs = list(combinations(sorted(ALGORITHMS_OF_INTEREST), 2))

    for d in dims:
        os.makedirs(f'{output_dir}/cosine/dim_{d}', exist_ok=True)
        input_dir = f'{base_input}/dim_{d}'

        files = [f for f in sorted(os.listdir(input_dir)) if f.endswith('.csv')]
        progress.begin(len(files), f"cosine dim {d}")
        for file in files:
            if not progress.item(file):
                result.cancelled = True
                return result
            problem_name = file.replace('.csv', '')   # e.g. "F1_I1"
            prob = problem_name.split('_')[0]          # "F1"
            inst = problem_name.split('_')[1]          # "I1"

            # index levels: algorithm, run, iteration; columns: clusters
            df = pd.read_csv(f'{input_dir}/{file}', index_col=[0, 1, 2])

            rows = []
            for r in RUNS:
                for pair in algorithm_pairs:
                    alg1, alg2 = pair
                    try:
                        v1 = flatten_algorithm_run(df.loc[(alg1, r)])
                        v2 = flatten_algorithm_run(df.loc[(alg2, r)])
                    except KeyError:
                        continue  # one of the algorithms/runs missing for this problem

                    if v1.shape != v2.shape or v1.size == 0:
                        continue

                    sim = cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0, 0]
                    distance = 1.0 - sim  # 0 = identical distribution vectors, higher = more different

                    row = {"Algorithm1": alg1, "Algorithm2": alg2,
                           "Function_id": prob, "Instance_id": inst, "Run_id": r,
                           "Cosine_distance": distance}
                    rows.append(row)

            result_frame = pd.DataFrame(rows)
            result_frame.to_csv(f'{output_dir}/cosine/dim_{d}/{problem_name}.csv', index=False)
            result.written += 1

    return result


def main():
    run()


if __name__ == "__main__":
    main()
