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
INPUT_DIR = "data/return_rate"
OUTPUT_DIR = "metrics_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

algorithm_pairs = list(combinations(sorted(ALGORITHMS_OF_INTEREST), 2))

for d in DIMENSIONS:
    os.makedirs(f'{OUTPUT_DIR}/return_rate/dim_{d}', exist_ok=True)
    df = pd.read_csv(f'{INPUT_DIR}/dim_{d}/revisiting_history.csv')

    # revisiting_history uses string problem_class ("F1") and instance ("I1"),
    # unlike the entropy granular file's integer problem_id/instance_id.
    for (prob, inst), tmp in df.groupby(['problem_class', 'instance']):
        rows = []  # (Alg1, Alg2, F, I) run_id, revisit_distance
        for r in RUNS:
            run_data = tmp[tmp['run'] == r]
            for pair in algorithm_pairs:
                alg1, alg2 = pair
                clusters_alg1 = set(run_data[run_data['algorithm'] == alg1]['cluster'])
                clusters_alg2 = set(run_data[run_data['algorithm'] == alg2]['cluster'])

                if not clusters_alg1 and not clusters_alg2:
                    continue  # neither revisited anything this run

                intersection = len(clusters_alg1 & clusters_alg2)
                union = len(clusters_alg1 | clusters_alg2)
                similarity = intersection / union if union > 0 else 0.0
                distance = 1.0 - similarity  # 0 = identical cluster sets, 1 = disjoint

                row = {"Algorithm1": alg1, "Algorithm2": alg2,
                       "Function_id": prob, "Instance_id": inst, "Run_id": r,
                       "Revisit_distance": distance}
                rows.append(row)
        result = pd.DataFrame(rows)
        result.to_csv(f'{OUTPUT_DIR}/return_rate/dim_{d}/{prob}_{inst}.csv', index=False)
