# Pipeline step "entropy_pairwise": for each (function, instance), computes
# the mean absolute per-iteration entropy difference between every pair of
# algorithms, per run. Writes metrics_data/entropy/dim_{d}/F{f}_I{i}.csv.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import os
from itertools import combinations
from config import (
    ALGORITHMS_OF_INTEREST, DIMENSIONS, FUNCTIONS, INSTANCES, SEEDS as RUNS,
    ENTROPY_DATA_DIR as INPUT_DIR, METRICS_DIR as OUTPUT_DIR,
)
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
                assert len(a1) == len(a2), f"unexpected length: {alg1}={len(a1)}, {alg2}={len(a2)} at F{f}_I{i}_R{r}"
                res = np.abs(a1-a2).mean()
                row = {"Algorithm1": alg1, "Algorithm2":alg2, "Function_id": f, "Instance_id": i, "Run_id": r, "Mean_entropy_difference": res}
                rows.append(row)
        result = pd.DataFrame(rows)
        result.to_csv(f'{OUTPUT_DIR}/entropy/dim_{d}/F{f}_I{i}.csv', index=False)