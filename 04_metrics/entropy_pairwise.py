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
import config
from config import (
    ALGORITHMS_OF_INTEREST, DIMENSIONS,
    ENTROPY_DATA_DIR as INPUT_DIR, METRICS_DIR as OUTPUT_DIR,
)
from pipeline_api import Progress, StageResult

STAGE = "entropy_pairwise"

def run(progress_cb=None, cancel_event=None, dimensions=None):
    """Mean absolute per-iteration entropy difference for every algorithm pair."""
    dims = list(dimensions if dimensions is not None else DIMENSIONS)
    input_dir = config.ENTROPY_DATA_DIR
    output_dir = config.METRICS_DIR
    progress = Progress(progress_cb, cancel_event)
    result = StageResult(STAGE)

    os.makedirs(output_dir, exist_ok=True)
    algorithm_pairs = list(combinations(sorted(ALGORITHMS_OF_INTEREST), 2))

    for d in dims:
        os.makedirs(f'{output_dir}/entropy/dim_{d}', exist_ok=True)
        df = pd.read_csv(f'{input_dir}/entropy_granular_dim_{d}.csv')
        groups = list(df.groupby(['problem_id', 'instance_id']))
        progress.begin(len(groups), f"entropy pairs dim {d}")
        for (f, i), tmp in groups:
            if not progress.item(f"F{f}_I{i}"):
                result.cancelled = True
                return result
            rows = []  # (Alg1, Alg2, F, I) run_id, mean_entropy_diff
            # Runs actually present for this problem, not config.SEEDS: a GUI
            # run (or a narrower -s) can use any seed.
            for r in sorted(tmp['run'].unique()):
                run_data = tmp[tmp['run'] == r]
                for pair in algorithm_pairs:
                    alg1, alg2 = pair
                    a1 = np.array(run_data[run_data['algorithm']==alg1].sort_values('iteration')["entropy"])
                    a2 = np.array(run_data[run_data['algorithm']==alg2].sort_values('iteration')["entropy"])
                    if len(a1) == 0 or len(a2) == 0:
                        continue  # one algorithm missing this run for this problem
                    assert len(a1) == len(a2), f"unexpected length: {alg1}={len(a1)}, {alg2}={len(a2)} at F{f}_I{i}_R{r}"
                    res = np.abs(a1-a2).mean()
                    row = {"Algorithm1": alg1, "Algorithm2":alg2, "Function_id": f, "Instance_id": i, "Run_id": r, "Mean_entropy_difference": res}
                    rows.append(row)
            result_frame = pd.DataFrame(rows)
            result_frame.to_csv(f'{output_dir}/entropy/dim_{d}/F{f}_I{i}.csv', index=False)
            result.written += 1

    return result


def main():
    run()


if __name__ == "__main__":
    main()
