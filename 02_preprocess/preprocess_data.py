# Pipeline step "preprocess": reshapes outputs/ population trajectories into
# data/processed/dim_{d}/F{f}_I{i}.csv (the shared input format for clustering).
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import os
from config import DIMENSIONS as dimensions, OUTPUTS_DIR as base, PROCESSED_DIR

if __name__ == "__main__":
    for d in dimensions:

        frames = []
        base_d = f'{base}/dim_{d}'

        for algorithm in os.listdir(base_d):
            if not os.path.isdir(f'{base_d}/{algorithm}'):
                    continue
            print(f'Preprocessing algorithm {algorithm} in dimension {d}')
            for problem_folder in os.listdir(f'{base_d}/{algorithm}'):
                problem_id, instance_id = problem_folder.split('_')
                for seed in [1, 2, 3, 4, 5]:
                    path = f'{base_d}/{algorithm}/{problem_folder}/trajectory_{seed}.csv'
                    if not os.path.isfile(path):
                        continue
                    df = pd.read_csv(path)
                    column_map = {f'x{i+1}':f'x{i}' for i in range(d)}
                    column_map['fitness'] = 'raw_y'
                    df = df.rename(columns=column_map)
                    df['algorithm'] = algorithm
                    df['run'] = seed
                    df['problem_id'] = int(problem_id)
                    df['instance_id'] = int(instance_id)
                    frames.append(df)


        result = pd.concat(frames, ignore_index=True)
        os.makedirs(f'{PROCESSED_DIR}/dim_{d}', exist_ok=True)

        for (pid, iid), group in result.groupby(['problem_id', 'instance_id']):  # adjust col names as needed
            group.to_csv(f'{PROCESSED_DIR}/dim_{d}/F{pid}_I{iid}.csv', compression='zip')


    for d in dimensions:
        sample = pd.read_csv(f'{PROCESSED_DIR}/dim_{d}/F1_I1.csv', compression='zip', index_col=0)
        print(f'dim {d}:', sample.shape, sample.columns.tolist())