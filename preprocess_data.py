import pandas as pd
import os

base = 'outputs'
frames = []

for algorithm in os.listdir(base):
    for problem_folder in os.listdir(f'{base}/{algorithm}'):
        problem_id, instance_id = problem_folder.split('_')
        for seed in [1, 2, 3, 4, 5]:
            path = f'{base}/{algorithm}/{problem_folder}/trajectory_{seed}.csv'
            if not os.path.isfile(path):
                continue
            df = pd.read_csv(path)
            df = df.rename(columns={'x1': 'x0', 'x2': 'x1', 'fitness': 'raw_y'})
            df['algorithm'] = algorithm
            df['run'] = seed
            df['problem_id'] = int(problem_id)      # explicitly set here
            df['instance_id'] = int(instance_id)    # explicitly set here
            #df['iteration'] = range(1, len(df) + 1)  # 1-indexed, so iteration 0 is excluded
            #df['evaluations'] = 1000       # 1 eval per row for dim 2
            #print(df.head)
            frames.append(df)

result = pd.concat(frames, ignore_index=True)
os.makedirs('data/processed/dim_2', exist_ok=True)
# Script 1 iterates files in the folder, one file per problem
for (pid, iid), group in result.groupby(['problem_id', 'instance_id']):  # adjust col names as needed
    group.to_csv(f'data/processed/dim_2/F{pid}_I{iid}.csv', compression='zip')


sample = pd.read_csv('data/processed/dim_2/F1_I1.csv', compression='zip', index_col=0)
print(sample.head(10))
print(sample.columns.tolist())
print(sample.shape)