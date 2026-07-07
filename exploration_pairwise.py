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
FUNCTIONS = [i for i in range(1, 25)]
INSTANCES = [i for i in range(1, 6)]
RUNS = [i for i in range(1, 6)]
INPUT_DIR = "outputs"
OUTPUT_DIR = "metrics_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

algorithm_pairs = list(combinations(sorted(ALGORITHMS_OF_INTEREST), 2))


def load_exploration(d, alg, f, i, seed):
    """
    Reads one diversity_{seed}.csv and returns the exploration column as a
    numpy array ordered by iteration. Returns None if the file is missing.
    exploration+exploitation sum to 100, so |expl diff| == |exploit diff|;
    one column suffices for both.
    """
    path = f"{INPUT_DIR}/dim_{d}/{alg}/{f}_{i}/diversity_{seed}.csv"
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path).sort_values('iteration')
    return df['exploration'].to_numpy()


for d in DIMENSIONS:
    os.makedirs(f'{OUTPUT_DIR}/exploration/dim_{d}', exist_ok=True)

    for f in FUNCTIONS:
        for i in INSTANCES:
            rows = []  # (Alg1, Alg2, F, I) run_id, mean_exploration_difference

            for r in RUNS:
                # cache each algorithm's exploration curve for this (f, i, r)
                # so we don't re-read the same file for every pair it appears in
                curves = {}
                for alg in ALGORITHMS_OF_INTEREST:
                    curves[alg] = load_exploration(d, alg, f, i, r)

                for pair in algorithm_pairs:
                    alg1, alg2 = pair
                    e1 = curves[alg1]
                    e2 = curves[alg2]
                    if e1 is None or e2 is None:
                        continue  # missing diversity file for one algorithm
                    if len(e1) != len(e2) or len(e1) == 0:
                        continue

                    diff = (np.abs(e1 - e2).mean())/100.0

                    row = {"Algorithm1": alg1, "Algorithm2": alg2,
                           "Function_id": f, "Instance_id": i, "Run_id": r,
                           "Mean_exploration_difference": diff}
                    rows.append(row)

            result = pd.DataFrame(rows)
            result.to_csv(f'{OUTPUT_DIR}/exploration/dim_{d}/F{f}_I{i}.csv', index=False)
