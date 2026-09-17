# Pipeline step "harvest_results": scans outputs/ g_best trajectories and
# builds outputs/dim_{d}/results.csv (one row per algorithm/problem/instance/
# seed best solution).
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import os
import config
from config import OUTPUTS_DIR, DIMENSIONS, SEEDS
from pipeline_api import Progress, StageResult

STAGE = "harvest_results"


def best_row_from_gbest(path):
    df = pd.read_csv(path)
    coord_cols = [c for c in df.columns if c.startswith('x')]
    best = df.loc[df['fitness'].idxmin()]
    return coord_cols, best


def run(progress_cb=None, cancel_event=None, dimensions=None):
    """Build outputs/dim_{d}/results.csv from the g_best trajectories.

    Reads OUTPUTS_DIR off `config` at call time rather than using the name
    imported above, so the desktop app's "open a different results folder"
    (gui/core/results_root.py) reaches this stage too.
    """
    dimensions = list(dimensions if dimensions is not None else DIMENSIONS)
    outputs_dir = config.OUTPUTS_DIR
    progress = Progress(progress_cb, cancel_event)
    result_stage = StageResult(STAGE)

    for d in dimensions:
        base = f"{outputs_dir}/dim_{d}"
        rows = []
        coord_cols_ref = None

        algorithms = [a for a in sorted(os.listdir(base))
                      if os.path.isdir(f"{base}/{a}")]
        progress.begin(len(algorithms), f"harvesting dim {d}")

        for algorithm in algorithms:
            if not progress.item(algorithm):
                result_stage.cancelled = True
                return result_stage
            alg_dir = f"{base}/{algorithm}"
            for problem_folder in sorted(os.listdir(alg_dir)):
                pf = problem_folder
                parts = pf.replace('F', '').replace('I', '').split('_')
                problem_id, instance_id = int(parts[0]), int(parts[1])

                for seed in SEEDS:
                    path = f"{alg_dir}/{pf}/gbest_trajectory_{seed}.csv"
                    coord_cols, best = best_row_from_gbest(path)
                    coord_cols_ref = coord_cols
                    row = {
                        "algorithm": algorithm,
                        "problem_id": problem_id,
                        "instance_id": instance_id,
                        "seed": seed,
                        "fitness": best["fitness"],
                    }
                    for c in coord_cols:
                        row[c] = best[c]
                    rows.append(row)

        result = pd.DataFrame(rows)
        ordered = (["algorithm", "problem_id", "instance_id", "seed"]
                   + coord_cols_ref + ["fitness"])
        result = result[ordered]
        out_path = f"{base}/results.csv"
        result.to_csv(out_path, index=False)
        result_stage.written += 1
        result_stage.notes.append(f"dim {d}: {len(result)} rows")
        print(f"wrote {out_path}  ({len(result)} rows)")

    return result_stage


def main():
    run()


if __name__ == "__main__":
    main()
