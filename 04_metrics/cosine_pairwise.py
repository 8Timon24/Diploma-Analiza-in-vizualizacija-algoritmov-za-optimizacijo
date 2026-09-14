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
from config import (
    ALGORITHMS_OF_INTEREST, DIMENSIONS, SEEDS as RUNS,
    CLUSTER_DISTRIBUTIONS_LATEST as INPUT_DIR, METRICS_DIR as OUTPUT_DIR,
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

algorithm_pairs = list(combinations(sorted(ALGORITHMS_OF_INTEREST), 2))


def flatten_algorithm_run(sub):
    """
    Flatten one (algorithm, run)'s cluster-occupancy table (rows=iterations,
    cols=clusters) into a single 1-D feature vector by concatenating the
    per-iteration cluster counts in iteration order. This is the same feature
    representation the cosine-similarity script uses (unstack over iteration).
    """
    return sub.sort_index().to_numpy().flatten()


for d in DIMENSIONS:
    os.makedirs(f'{OUTPUT_DIR}/cosine/dim_{d}', exist_ok=True)
    input_dir = f'{INPUT_DIR}/dim_{d}'

    for file in sorted(os.listdir(input_dir)):
        if not file.endswith('.csv'):
            continue
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

        result = pd.DataFrame(rows)
        result.to_csv(f'{OUTPUT_DIR}/cosine/dim_{d}/{problem_name}.csv', index=False)
