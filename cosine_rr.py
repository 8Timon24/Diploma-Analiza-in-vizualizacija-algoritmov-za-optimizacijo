import numpy as np
import pandas as pd
import os
from itertools import combinations

ALGORITHMS_OF_INTEREST = ["AugmentedAEO", "GWO_WOA",
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
    "SADE", "WhaleFOA"]

DIMENSIONS = [2, 5, 10]
RUNS = [i for i in range(1, 6)]
INPUT_DIR = 'data/clustering_latest/cluster_distributions'
OUTPUT_DIR = "metrics_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

algorithm_pairs = list(combinations(sorted(ALGORITHMS_OF_INTEREST), 2))


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


for d in DIMENSIONS:
    os.makedirs(f'{OUTPUT_DIR}/cosine_columns/dim_{d}', exist_ok=True)
    input_dir = f'{INPUT_DIR}/dim_{d}'

    for file in sorted(os.listdir(input_dir)):
        if not file.endswith('.csv'):
            continue
        problem_name = file.replace('.csv', '')   # "F1_I1"
        prob = problem_name.split('_')[0]          # "F1"
        inst = problem_name.split('_')[1]          # "I1"

        # index levels: algorithm, run, iteration; columns: clusters
        df = pd.read_csv(f'{input_dir}/{file}', index_col=[0, 1, 2])

        rows = []
        for r in RUNS:
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

        result = pd.DataFrame(rows)
        result.to_csv(f'{OUTPUT_DIR}/cosine_columns/dim_{d}/{problem_name}.csv', index=False)