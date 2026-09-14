# Pipeline step "solutions_pairwise": difference in final solution location
# (Euclidean distance) and fitness between every pair of algorithms, per run.
# Writes metrics_data/{location,fitness}/dim_{d}/F{f}_I{i}.csv.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import os
from itertools import combinations
from config import (
    ALGORITHMS_OF_INTEREST, DIMENSIONS, FUNCTIONS, INSTANCES, SEEDS as RUNS,
    OUTPUTS_DIR as INPUT_DIR, METRICS_DIR as OUTPUT_DIR,
)
os.makedirs(OUTPUT_DIR, exist_ok=True)
algorithm_pairs = list(combinations(sorted(ALGORITHMS_OF_INTEREST), 2))


for d in DIMENSIONS:
    os.makedirs(f'{OUTPUT_DIR}/fitness/dim_{d}', exist_ok=True)
    os.makedirs(f'{OUTPUT_DIR}/location/dim_{d}', exist_ok=True)
    df = pd.read_csv(f'{INPUT_DIR}/dim_{d}/results.csv')
    for (f, i), tmp in df.groupby(['problem_id', 'instance_id']):
        rows_location = []  # (Alg1, Alg2, F, I) run_id, metric
        rows_fitness = []
        for r in RUNS:
            run_data = tmp[tmp['seed'] == r]
            for pair in algorithm_pairs:
                alg1, alg2 = pair
                cols = [f'x{k}' for k in range(1, d+1)]
                
                loc_1 = np.array(run_data[run_data['algorithm']==alg1][cols])
                loc_2 = np.array(run_data[run_data['algorithm']==alg2][cols])
                loc_diff = np.linalg.norm(loc_1 - loc_2)
                fit_1 = run_data[run_data['algorithm']==alg1]['fitness'].item()
                fit_2 = run_data[run_data['algorithm']==alg2]['fitness'].item()
                fit_diff = abs(fit_1-fit_2)
                row_loc = {"Algorithm1": alg1, "Algorithm2":alg2, "Function_id": f, "Instance_id": i, "Run_id": r, "Location_difference": loc_diff}
                row_fit = {"Algorithm1": alg1, "Algorithm2":alg2, "Function_id": f, "Instance_id": i, "Run_id": r, "Fitness_difference": fit_diff}
                rows_fitness.append(row_fit)
                rows_location.append(row_loc)
        
        result_fitness = pd.DataFrame(rows_fitness)
        result_location = pd.DataFrame(rows_location)
        result_fitness.to_csv(f'{OUTPUT_DIR}/fitness/dim_{d}/F{f}_I{i}.csv', index=False)
        result_location.to_csv(f'{OUTPUT_DIR}/location/dim_{d}/F{f}_I{i}.csv', index=False)
        