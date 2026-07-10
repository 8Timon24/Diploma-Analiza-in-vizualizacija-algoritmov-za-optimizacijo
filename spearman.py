import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from functools import reduce
import matplotlib.pyplot as plt
import seaborn as sns

metrics_dir = 'metrics_data'
KEYS = ['Algorithm1', 'Algorithm2', 'Function_id', 'Instance_id', 'Run_id']
DIMENSIONS = [2, 5, 10]
OUTPUT_DIR = 'figures_spearman'
MERGED_DIR = 'metrics_data/merged'

# metrics to include (folder names under metrics_data/). Excludes 'merged'.
# Edit this list to add/drop metrics.
METRICS = ['entropy', 'return_rate', 'cosine', 'cosine_columns',
           'exploration', 'location', 'fitness']


def normalize_keys(df):
    # Function_id / Instance_id may be int (1) or str ("F1"/"I1") depending on
    # which pairwise script wrote the file - coerce both to plain ints.
    for col, prefix in [('Function_id', 'F'), ('Instance_id', 'I')]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(prefix, '', regex=False).astype(int)
    if 'Run_id' in df.columns:
        df['Run_id'] = df['Run_id'].astype(int)
    return df


def load_metric(metric, dim):
    """Concatenate all F*_I* files for one metric+dim, rename value col to metric name."""
    dim_path = f'{metrics_dir}/{metric}/dim_{dim}'
    if not os.path.isdir(dim_path):
        return None
    frames = []
    for file in tqdm(sorted(os.listdir(dim_path)), desc=f'{metric}/dim_{dim}', leave=False):
        if not file.endswith('.csv'):
            continue
        df = pd.read_csv(f'{dim_path}/{file}')
        if df.empty:
            continue
        frames.append(normalize_keys(df))
    if not frames:
        return None
    metric_df = pd.concat(frames, ignore_index=True)
    value_col = [c for c in metric_df.columns if c not in KEYS][0]
    return metric_df[KEYS + [value_col]].rename(columns={value_col: metric})


def build_merged(dim):
    dfs = []
    present = []
    for m in METRICS:
        md = load_metric(m, dim)
        if md is not None:
            dfs.append(md)
            present.append(m)
        else:
            print(f"  [warn] {m} missing for dim {dim}, skipping")
    # inner merge: keep only comparisons present in ALL metrics (clean for corr)
    merged = reduce(lambda l, r: pd.merge(l, r, on=KEYS, how='inner'), dfs)
    return merged, present


def plot_spearman(corr, dim, present, output_dir):
    plt.figure(figsize=(10, 8))
    # diverging colormap centered at 0 since correlations run -1..1
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                vmin=-1, vmax=1, square=True, linewidths=0.5,
                cbar_kws={'label': "Spearman's rho"})
    plt.title(f'Spearman correlation between metrics (dim {dim})')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/spearman_dim_{dim}.pdf', bbox_inches='tight')
    plt.close()


def interpret(corr, present):
    """Print redundant / complementary / independent structure from the matrix."""
    pairs = []
    for i in range(len(present)):
        for j in range(i + 1, len(present)):
            pairs.append((present[i], present[j], corr.iloc[i, j]))
    pairs.sort(key=lambda x: -abs(x[2]))

    def band(r):
        a = abs(r)
        if a >= 0.8: return "very strong (largely redundant)"
        if a >= 0.6: return "strong"
        if a >= 0.4: return "moderate (complementary)"
        if a >= 0.2: return "weak"
        return "negligible (independent)"

    print("  metric pair relationships (by |rho|):")
    for a, b, r in pairs:
        print(f"    {a:16s} <-> {b:16s}  rho={r:+.3f}  {band(r)}")


if __name__ == '__main__':
    os.makedirs(MERGED_DIR, exist_ok=True)
    for dim in DIMENSIONS:
        print(f"\n{'='*60}\nDIM {dim}\n{'='*60}")
        merged, present = build_merged(dim)
        print(f"merged shape: {merged.shape}")
        print(f"NaNs:\n{merged[present].isna().sum().to_dict()}")

        merged.to_csv(f'{MERGED_DIR}/merged_dim_{dim}.csv', index=False)

        corr = merged[present].corr(method='spearman')
        corr.to_csv(f'{MERGED_DIR}/spearman_dim_{dim}.csv')
        print("\nSpearman matrix:")
        print(corr.round(3).to_string())

        interpret(corr, present)
        plot_spearman(corr, dim, present, OUTPUT_DIR)
        print(f"\n  saved -> {OUTPUT_DIR}/spearman_dim_{dim}.pdf")
        print(f"  saved -> {MERGED_DIR}/merged_dim_{dim}.csv, spearman_dim_{dim}.csv")

    print("\nDONE")