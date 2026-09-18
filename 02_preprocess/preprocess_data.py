# Pipeline step "preprocess": reshapes outputs/ population trajectories into
# data/processed/dim_{d}/F{f}_I{i}.csv (the shared input format for clustering).
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import os
import config
from config import DIMENSIONS as dimensions
from pipeline_api import Progress, StageResult

STAGE = "preprocess"


def run(progress_cb=None, cancel_event=None, dimensions_=None):
    """Reshape the raw population trajectories into the clustering input.

    Memory note: one dimension's frames are concatenated in one go, which is
    the heaviest allocation in the pipeline. Kept as-is deliberately - the
    concat is what gives the groupby below a single pass per problem.
    """
    dims = list(dimensions_ if dimensions_ is not None else dimensions)
    base = config.OUTPUTS_DIR
    processed_dir = config.PROCESSED_DIR
    progress = Progress(progress_cb, cancel_event)
    result_stage = StageResult(STAGE)

    for d in dims:

        frames = []
        base_d = f'{base}/dim_{d}'

        algorithms = [a for a in os.listdir(base_d)
                      if os.path.isdir(f'{base_d}/{a}')]
        progress.begin(len(algorithms), f"preprocessing dim {d}")

        for algorithm in algorithms:
            if not progress.item(algorithm):
                result_stage.cancelled = True
                return result_stage
            print(f'Preprocessing algorithm {algorithm} in dimension {d}')
            for problem_folder in os.listdir(f'{base_d}/{algorithm}'):
                problem_id, instance_id = problem_folder.split('_')
                problem_dir = f'{base_d}/{algorithm}/{problem_folder}'
                # Discovered per problem folder rather than assumed from
                # config.SEEDS: a GUI run (or a narrower -s) can produce any
                # seed, and this must neither skip one outside SEEDS nor
                # break on one from SEEDS that was never generated.
                for seed in config.discover_seeds(problem_dir, "trajectory"):
                    path = f'{problem_dir}/trajectory_{seed}.csv'
                    df = pd.read_csv(path)
                    column_map = {f'x{i+1}':f'x{i}' for i in range(d)}
                    column_map['fitness'] = 'raw_y'
                    df = df.rename(columns=column_map)
                    df['algorithm'] = algorithm
                    df['run'] = seed
                    df['problem_id'] = int(problem_id)
                    df['instance_id'] = int(instance_id)
                    frames.append(df)

        if not frames:
            result_stage.notes.append(f"dim {d}: no population trajectories")
            continue

        result = pd.concat(frames, ignore_index=True)
        os.makedirs(f'{processed_dir}/dim_{d}', exist_ok=True)

        for (pid, iid), group in result.groupby(['problem_id', 'instance_id']):  # adjust col names as needed
            group.to_csv(f'{processed_dir}/dim_{d}/F{pid}_I{iid}.csv', compression='zip')
            result_stage.written += 1

    for d in dims:
        sample_path = f'{processed_dir}/dim_{d}/F1_I1.csv'
        if not os.path.isfile(sample_path):
            continue
        sample = pd.read_csv(sample_path, compression='zip', index_col=0)
        print(f'dim {d}:', sample.shape, sample.columns.tolist())

    return result_stage


def main():
    run()


if __name__ == "__main__":
    main()
