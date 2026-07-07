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
RUNS = [i for i in range(1,6)]
INPUT_DIR = "data/entropy"
OUTPUT_DIR = "metrics_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)
algorithm_pairs = list(combinations(sorted(ALGORITHMS_OF_INTEREST), 2))


for d in DIMENSIONS:
    os.makedirs(f'{OUTPUT_DIR}/entropy/dim_{d}', exist_ok=True)
    df = pd.read_csv(f'{INPUT_DIR}/entropy_granular_dim_{d}.csv')
    for (f, i), tmp in df.groupby(['problem_id', 'instance_id']):
        rows = []  # (Alg1, Alg2, F, I) run_id, mean_entropy_diff
        for r in RUNS:
            run_data = tmp[tmp['run'] == r]
            for pair in algorithm_pairs:
                alg1, alg2 = pair
                a1 = np.array(run_data[run_data['algorithm']==alg1].sort_values('iteration')["entropy"])
                a2 = np.array(run_data[run_data['algorithm']==alg2].sort_values('iteration')["entropy"])
                assert len(a1) == len(a2) == 20, f"unexpected length: {alg1}={len(a1)}, {alg2}={len(a2)} at F{f}_I{i}_R{r}"
                res = np.abs(a1-a2).mean()
                row = {"Algorithm1": alg1, "Algorithm2":alg2, "Function_id": f, "Instance_id": i, "Run_id": r, "Mean_entropy_difference": res}
                rows.append(row)
        result = pd.DataFrame(rows)
        result.to_csv(f'{OUTPUT_DIR}/entropy/dim_{d}/F{f}_I{i}.csv', index=False)