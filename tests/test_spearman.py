# Ad hoc check: independently recomputes the Spearman correlation between
# two metrics (cosine vs cosine_columns, dim 10) as a sanity cross-check of
# spearman.py's merge logic.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from config import METRIC_KEYS as KEYS, normalize_keys, METRICS_DIR as metrics_dir

DIM = 10  # test on one dimension
METRICS_TO_TEST = ['cosine', 'cosine_columns']


def load_metric(metric, dim):
    dim_path = f'{metrics_dir}/{metric}/dim_{dim}'
    frames = []
    for file in tqdm(sorted(os.listdir(dim_path)), desc=f'{metric}/dim_{dim}'):
        if not file.endswith('.csv'):
            continue
        df = pd.read_csv(f'{dim_path}/{file}')
        if df.empty:
            continue
        df = normalize_keys(df)
        frames.append(df)
    metric_df = pd.concat(frames, ignore_index=True)
    value_col = [c for c in metric_df.columns if c not in KEYS][0]
    return metric_df[KEYS + [value_col]].rename(columns={value_col: metric})


# load the two metrics and merge on shared keys
dfs = [load_metric(m, DIM) for m in METRICS_TO_TEST]
merged = dfs[0].merge(dfs[1], on=KEYS, how='inner')

print(f"\nmerged shape: {merged.shape}")
print(f"NaNs per metric:\n{merged[METRICS_TO_TEST].isna().sum()}")

# spearman between the two metrics
rho = merged['cosine'].corr(merged['cosine_columns'], method='spearman')
print(f"\nSpearman correlation (cosine vs cosine_columns), dim {DIM}: {rho:.4f}")

# also show the full (2x2) matrix for sanity
print("\nfull matrix:")
print(merged[METRICS_TO_TEST].corr(method='spearman').round(4))