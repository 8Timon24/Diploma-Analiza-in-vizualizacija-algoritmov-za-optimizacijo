# Pipeline step "exploration_pairwise": mean absolute difference in
# exploration percentage between every pair of algorithms, per run.
# Writes metrics_data/exploration/dim_{d}/F{f}_I{i}.csv.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import os
from itertools import combinations
import config
from config import (
    ALGORITHMS_OF_INTEREST, DIMENSIONS, FUNCTIONS, INSTANCES, SEEDS as RUNS,
    OUTPUTS_DIR as INPUT_DIR, METRICS_DIR as OUTPUT_DIR,
)
from pipeline_api import Progress, StageResult

STAGE = "exploration_pairwise"


def load_exploration(d, alg, f, i, seed):
    """
    Reads one diversity_{seed}.csv and returns the exploration column as a
    numpy array ordered by iteration. Returns None if the file is missing.
    exploration+exploitation sum to 100, so |expl diff| == |exploit diff|;
    one column suffices for both.
    """
    path = f"{config.OUTPUTS_DIR}/dim_{d}/{alg}/{f}_{i}/diversity_{seed}.csv"
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path).sort_values('iteration')
    return df['exploration'].to_numpy()


def run(progress_cb=None, cancel_event=None, dimensions=None):
    """Pairwise difference in exploration/exploitation balance.

    Needs the benchmark to have been run with -e (save_diversity); a missing
    diversity file for an algorithm simply drops that pair.
    """
    dims = list(dimensions if dimensions is not None else DIMENSIONS)
    output_dir = config.METRICS_DIR
    progress = Progress(progress_cb, cancel_event)
    result = StageResult(STAGE)

    os.makedirs(output_dir, exist_ok=True)
    algorithm_pairs = list(combinations(sorted(ALGORITHMS_OF_INTEREST), 2))

    for d in dims:
        os.makedirs(f'{output_dir}/exploration/dim_{d}', exist_ok=True)
        progress.begin(len(FUNCTIONS) * len(INSTANCES), f"exploration dim {d}")

        for f in FUNCTIONS:
            if progress.cancelled:
                break
            for i in INSTANCES:
                if not progress.item(f"F{f}_I{i}"):
                    break
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

                result_frame = pd.DataFrame(rows)
                result_frame.to_csv(f'{output_dir}/exploration/dim_{d}/F{f}_I{i}.csv', index=False)
                result.written += 1

        if progress.cancelled:
            result.cancelled = True
            return result

    return result


def main():
    run()


if __name__ == "__main__":
    main()
