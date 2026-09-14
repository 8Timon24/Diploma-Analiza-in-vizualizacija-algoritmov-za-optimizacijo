# Pipeline step "return_rate_pairwise": from the revisiting history, computes
# a per-pair "revisit distance" (1 - Jaccard similarity of visited-cluster
# sets) per run. Writes metrics_data/return_rate/dim_{d}/{F}_{I}.csv.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import os
from itertools import combinations
from config import (
    ALGORITHMS_OF_INTEREST, DIMENSIONS, SEEDS as RUNS,
    RETURN_RATE_DATA_DIR as INPUT_DIR, METRICS_DIR as OUTPUT_DIR,
)

if __name__ == "__main__":
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
