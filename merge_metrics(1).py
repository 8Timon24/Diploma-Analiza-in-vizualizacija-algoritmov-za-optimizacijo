import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from functools import reduce

metrics_dir = 'metrics_data'
KEYS = ['Algorithm1', 'Algorithm2', 'Function_id', 'Instance_id', 'Run_id']

# The metric-value column is the one column in each file that ISN'T a key.
# We rename it to the metric name so columns don't collide after merging.
def normalize_keys(df):
    # Function_id / Instance_id may be int (1) or str ("F1"/"I1") depending on
    # which pairwise script wrote the file - coerce both to plain ints so the
    # merge keys line up across all metrics.
    for col, prefix in [('Function_id', 'F'), ('Instance_id', 'I')]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(prefix, '', regex=False).astype(int)
    if 'Run_id' in df.columns:
        df['Run_id'] = df['Run_id'].astype(int)
    return df
# ---- 1. collect: {dim: {metric: concatenated_dataframe}} ----
per_dim_metric = {}   # per_dim_metric[dim][metric] = df with keys + one value col

for metric in os.listdir(metrics_dir):
    metric_path = f'{metrics_dir}/{metric}'
    if not os.path.isdir(metric_path):
        continue
    for dim_folder in os.listdir(metric_path):
        dim_path = f'{metric_path}/{dim_folder}'
        if not os.path.isdir(dim_path):
            continue
        dim = dim_folder.replace('dim_', '')

        frames = []
        for file in tqdm(sorted(os.listdir(dim_path)), desc=f'{metric}/{dim_folder}'):
            if not file.endswith('.csv'):
                continue
            df = pd.read_csv(f'{dim_path}/{file}')
            if df.empty:
                continue
            df = normalize_keys(df)
            frames.append(df)

        if not frames:
            continue
        metric_df = pd.concat(frames, ignore_index=True)

        # find the value column (the non-key column) and rename it to the metric
        value_cols = [c for c in metric_df.columns if c not in KEYS]
        if len(value_cols) != 1:
            print(f"  [warn] {metric}/{dim_folder}: expected 1 value col, got {value_cols}")
        val = value_cols[0]
        metric_df = metric_df[KEYS + [val]].rename(columns={val: metric})

        per_dim_metric.setdefault(dim, {})[metric] = metric_df

# ---- 2. merge all metrics within each dimension on the shared keys ----
merged_by_dim = {}
for dim, metric_dict in per_dim_metric.items():
    dfs = list(metric_dict.values())
    # outer merge so a pair present in one metric but missing in another isn't dropped;
    # switch to 'inner' if you only want rows where ALL metrics exist.
    merged = reduce(lambda l, r: pd.merge(l, r, on=KEYS, how='outer'), dfs)
    merged_by_dim[dim] = merged
    print(f"\ndim {dim}: merged shape {merged.shape}, metrics: {list(metric_dict.keys())}")
    # save
    os.makedirs('metrics_data/merged', exist_ok=True)
    merged.to_csv(f'metrics_data/merged/merged_dim_{dim}.csv', index=False)
    print(f"  saved -> metrics_data/merged/merged_dim_{dim}.csv")