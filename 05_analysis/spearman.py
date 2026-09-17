# Pipeline step "spearman": inner-joins the metrics in METRICS below on the
# shared keys, computes the Spearman correlation matrix between them per
# dimension, and saves the matrix + a heatmap to figures_spearman/.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from functools import reduce
from matplotlib.figure import Figure
import seaborn as sns
import config
from config import (
    METRIC_KEYS as KEYS, DIMENSIONS, MERGED_DIR, METRIC_LABELS, normalize_keys,
    METRICS_DIR as metrics_dir, FIGURES_SPEARMAN_DIR as OUTPUT_DIR,
)
from pipeline_api import Progress, StageResult

STAGE = "spearman"

# metrics to include (folder names under metrics_data/). Excludes 'merged'.
# Edit this list to add/drop metrics.
METRICS = ['entropy', 'cosine', 'cosine_columns',
           'exploration', 'location', 'fitness']


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
    """Draw the correlation heatmap, save it, and return the Figure.

    Built through the Figure API rather than pyplot: pyplot keeps global
    state and creates a canvas on whatever backend is active, which in the
    desktop app is QtAgg - and building a Qt canvas off the main thread
    crashes. A bare Figure has neither problem, so the same function is safe
    from the pipeline's worker thread and from the GUI's renderer.

    It still calls savefig, which is what writes figures_spearman/; when the
    GUI renders this for display, gui/viz/capture.py intercepts that call and
    takes the returned Figure instead, so nothing is written.
    """
    # rename a copy for display only - `corr` itself is written to CSV unchanged
    corr_disp = corr.rename(index=METRIC_LABELS, columns=METRIC_LABELS)

    figure = Figure(figsize=(10, 8), layout="constrained")
    axes = figure.add_subplot(1, 1, 1)
    # diverging colormap centered at 0 since correlations run -1..1
    sns.heatmap(corr_disp, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                vmin=-1, vmax=1, square=True, linewidths=0.5,
                cbar_kws={'label': 'Spearman \u03c1'}, ax=axes)
    axes.set_title(f'Spearman correlation between metrics (dimension {dim})')
    axes.tick_params(axis='x', rotation=45)
    for label in axes.get_xticklabels():
        label.set_horizontalalignment('right')
    axes.tick_params(axis='y', rotation=0)
    os.makedirs(output_dir, exist_ok=True)
    figure.savefig(f'{output_dir}/spearman_dim_{dim}.pdf', bbox_inches='tight')
    return figure


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


def run(progress_cb=None, cancel_event=None, dimensions=None):
    """Correlate the metrics against each other, per dimension."""
    dims = list(dimensions if dimensions is not None else DIMENSIONS)
    merged_dir = config.MERGED_DIR
    figures_dir = config.FIGURES_SPEARMAN_DIR
    progress = Progress(progress_cb, cancel_event)
    result = StageResult(STAGE)

    os.makedirs(merged_dir, exist_ok=True)
    progress.begin(len(dims), "spearman")
    for dim in dims:
        if not progress.item(f"dim {dim}"):
            result.cancelled = True
            return result
        print(f"\n{'='*60}\nDIM {dim}\n{'='*60}")
        merged, present = build_merged(dim)
        print(f"merged shape: {merged.shape}")
        print(f"NaNs:\n{merged[present].isna().sum().to_dict()}")

        # Written under its own name, NOT merged_dim_{dim}.csv - that file is
        # merge_metrics.py's outer-joined, all-metrics output (metrics.py's
        # "merge" step runs right before this one); this is only the
        # narrower inner join spearman.py itself correlates over, restricted
        # to METRICS above and to rows where every one of them is present.
        merged.to_csv(f'{merged_dir}/spearman_input_dim_{dim}.csv', index=False)

        corr = merged[present].corr(method='spearman')
        corr.to_csv(f'{merged_dir}/spearman_dim_{dim}.csv')
        print("\nSpearman matrix:")
        print(corr.round(3).to_string())

        interpret(corr, present)
        plot_spearman(corr, dim, present, figures_dir)
        result.written += 3
        print(f"\n  saved -> {figures_dir}/spearman_dim_{dim}.pdf")
        print(f"  saved -> {merged_dir}/spearman_input_dim_{dim}.csv, spearman_dim_{dim}.csv")

    print("\nDONE")
    return result


def main():
    run()


if __name__ == '__main__':
    main()
