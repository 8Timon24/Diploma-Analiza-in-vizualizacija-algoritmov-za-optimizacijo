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
import config
from config import (
    ALGORITHMS_OF_INTEREST, DIMENSIONS,
    OUTPUTS_DIR as INPUT_DIR, METRICS_DIR as OUTPUT_DIR,
)
from pipeline_api import Progress, StageResult

STAGE = "solutions_pairwise"

def run(progress_cb=None, cancel_event=None, dimensions=None):
    """Pairwise difference in final solution location and fitness."""
    dims = list(dimensions if dimensions is not None else DIMENSIONS)
    input_dir = config.OUTPUTS_DIR
    output_dir = config.METRICS_DIR
    progress = Progress(progress_cb, cancel_event)
    result = StageResult(STAGE)

    os.makedirs(output_dir, exist_ok=True)
    algorithm_pairs = list(combinations(sorted(ALGORITHMS_OF_INTEREST), 2))

    for d in dims:
        os.makedirs(f'{output_dir}/fitness/dim_{d}', exist_ok=True)
        os.makedirs(f'{output_dir}/location/dim_{d}', exist_ok=True)
        df = pd.read_csv(f'{input_dir}/dim_{d}/results.csv')
        # Hoisted out of the innermost loop: it depends only on the dimension,
        # and was being rebuilt once per pair per run.
        cols = [f'x{k}' for k in range(1, d+1)]
        groups = list(df.groupby(['problem_id', 'instance_id']))
        progress.begin(len(groups), f"solutions dim {d}")
        for (f, i), tmp in groups:
            if not progress.item(f"F{f}_I{i}"):
                result.cancelled = True
                return result
            rows_location = []  # (Alg1, Alg2, F, I) run_id, metric
            rows_fitness = []
            # Seeds actually present for this problem, not config.SEEDS: a
            # GUI run (or a narrower -s) can use any seed.
            for r in sorted(tmp['seed'].unique()):
                run_data = tmp[tmp['seed'] == r]
                for pair in algorithm_pairs:
                    alg1, alg2 = pair

                    rows_1 = run_data[run_data['algorithm']==alg1]
                    rows_2 = run_data[run_data['algorithm']==alg2]
                    if len(rows_1) != 1 or len(rows_2) != 1:
                        continue  # one algorithm missing this run for this problem

                    loc_1 = np.array(rows_1[cols])
                    loc_2 = np.array(rows_2[cols])
                    loc_diff = np.linalg.norm(loc_1 - loc_2)
                    fit_1 = rows_1['fitness'].item()
                    fit_2 = rows_2['fitness'].item()
                    fit_diff = abs(fit_1-fit_2)
                    row_loc = {"Algorithm1": alg1, "Algorithm2":alg2, "Function_id": f, "Instance_id": i, "Run_id": r, "Location_difference": loc_diff}
                    row_fit = {"Algorithm1": alg1, "Algorithm2":alg2, "Function_id": f, "Instance_id": i, "Run_id": r, "Fitness_difference": fit_diff}
                    rows_fitness.append(row_fit)
                    rows_location.append(row_loc)

            result_fitness = pd.DataFrame(rows_fitness)
            result_location = pd.DataFrame(rows_location)
            result_fitness.to_csv(f'{output_dir}/fitness/dim_{d}/F{f}_I{i}.csv', index=False)
            result_location.to_csv(f'{output_dir}/location/dim_{d}/F{f}_I{i}.csv', index=False)
            result.written += 2

    return result


def main():
    run()


if __name__ == "__main__":
    main()
