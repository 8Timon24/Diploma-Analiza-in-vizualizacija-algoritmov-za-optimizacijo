"""
Figure: added value of the proposed metrics over the inherited ClustOpt cosine.

Four algorithm pairs (2 with high, 2 with low ClustOpt similarity); for each
pair, a heatmap of all six metrics on problems F1 and F23, plus the average
across all problems.

Point: for pairs with LOW ClustOpt similarity (bottom row), entropy and
exploration show that the two algorithms are nonetheless similar - information
the cosine distance alone doesn't capture.

Normalization: min-max WITHIN each problem, across ALL pairs on that problem
(not just the four shown) - a cell shows where the pair sits relative to the
range of all pairs. If fitness/location end up compressed due to outliers,
switch NORMALIZATION = 'rank'.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from config import MERGED_DIR, FIGURES_RESULTS_DIR

# ---- settings ----
DIM = 2
MERGED_PATH = f'{MERGED_DIR}/merged_dim_{DIM}.csv'
NORMALIZATION = 'rank'    # 'minmax' or 'rank'
PROBLEMS = [(1, 1), (23, 1)]    # (Function_id, Instance_id) columns

# pairs: (algorithm1, algorithm2) - alphabetically ordered, as in the data
PAIRS_HIGH = [
    ('JADE', 'OriginalSHADE'),
    ('OriginalSHADE', 'SADE'),
]
PAIRS_LOW = [
    ('AugmentedAEO', 'OriginalALO'),
    ('OriginalDE', 'OriginalMRFO'),
]

REFERENCE = 'cosine'
PROPOSED = ['entropy', 'cosine_columns', 'exploration', 'location', 'fitness']
ALL_METRICS = [REFERENCE] + PROPOSED

METRIC_NAMES = {
    'cosine':          'ClustOpt cosine\ndistance',
    'entropy':         'Entropy\ndifference',
    'cosine_columns':  'Cosine distance\nper cluster',
    'exploration':     'Exploration\ndifference',
    'location':        'Final location\ndifference',
    'fitness':         'Final quality\ndifference',
}
COLUMN_NAMES = ['Prob. 1', 'Prob. 23', 'Mean']

KEYS = ['Algorithm1', 'Algorithm2']

# ---- 1. load ----
if not os.path.isfile(MERGED_PATH):
    raise FileNotFoundError(MERGED_PATH)
df = pd.read_csv(MERGED_PATH)

# average across runs -> one row per (pair, function, instance)
pp = (df.groupby(KEYS + ['Function_id', 'Instance_id'])[ALL_METRICS]
        .mean().reset_index())

# ---- 2. normalization WITHIN a problem, across ALL pairs ----
for m in ALL_METRICS:
    grp = pp.groupby(['Function_id', 'Instance_id'])[m]
    if NORMALIZATION == 'rank':
        # percentile rank: robust to outliers
        pp[f'{m}_n'] = grp.rank(pct=True)
    else:
        lo, hi = grp.transform('min'), grp.transform('max')
        span = hi - lo
        norm = (pp[m] - lo) / span.where(span > 0)
        pp[f'{m}_n'] = norm.mask(span.eq(0) & pp[m].notna(), 0.0)

norm_cols = [f'{m}_n' for m in ALL_METRICS]

# average of normalized values across all problems
pair_avg = pp.groupby(KEYS)[norm_cols].mean()
# raw ClustOpt distance (for the similarity badge)
pair_avg_raw = pp.groupby(KEYS)[REFERENCE].mean()


def values_for_pair(a1, a2):
    """Returns a (6 x 3) matrix: rows = metrics, columns = F1, F23, mean."""
    columns = []
    for f, i in PROBLEMS:
        sel = pp[(pp['Algorithm1'] == a1) & (pp['Algorithm2'] == a2) &
                 (pp['Function_id'] == f) & (pp['Instance_id'] == i)]
        if sel.empty:
            print(f'  [warning] no data for {a1}/{a2} on F{f}_I{i}')
            columns.append(np.full(len(ALL_METRICS), np.nan))
        else:
            columns.append(sel[norm_cols].iloc[0].to_numpy())
    if (a1, a2) not in pair_avg.index:
        raise KeyError(f'Pair {a1}/{a2} not in the data - check the name order.')
    columns.append(pair_avg.loc[(a1, a2), norm_cols].to_numpy())
    return np.column_stack(columns)


# ---- 3. figure: 2x2 grid of cards ----
all_pairs = [(p, 'high') for p in PAIRS_HIGH] + [(p, 'low') for p in PAIRS_LOW]

fig, axes = plt.subplots(2, 2, figsize=(13, 11))
axes = axes.flatten()

print('=' * 78)
print(f'VALUES IN THE FIGURE (normalization: {NORMALIZATION})')
print('=' * 78)

for ax, ((a1, a2), kind) in zip(axes, all_pairs):
    M = values_for_pair(a1, a2)
    similarity = 1.0 - pair_avg_raw.loc[(a1, a2)]

    table = pd.DataFrame(M, index=[METRIC_NAMES[m] for m in ALL_METRICS],
                          columns=COLUMN_NAMES)

    print(f'\n{a1} / {a2}   (ClustOpt similarity = {similarity:.2f}, {kind})')
    print(table.to_string(float_format='%.3f'))

    sns.heatmap(table, ax=ax, cmap='Blues', vmin=0, vmax=1, annot=True,
                fmt='.2f', linewidths=1.5, linecolor='white', cbar=False,
                annot_kws={'size': 10})

    color = '#2a9d8f' if kind == 'high' else '#e63946'
    ax.set_title(f'{a1}  vs  {a2}\n{kind} ClustOpt similarity ({similarity:.2f})',
                 fontsize=11, fontweight='bold', color=color, pad=12)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(axis='y', rotation=0, labelsize=9)
    ax.tick_params(axis='x', labelsize=10)

# shared color scale at the bottom
fig.subplots_adjust(bottom=0.10, hspace=0.45, wspace=0.55)
cax = fig.add_axes([0.25, 0.04, 0.5, 0.016])
sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=0, vmax=1))
cb = fig.colorbar(sm, cax=cax, orientation='horizontal')
cb.set_label('normalized difference between the two algorithms in the pair '
             '(0 = most similar, 1 = most different)', fontsize=9)

os.makedirs(FIGURES_RESULTS_DIR, exist_ok=True)
out = f'{FIGURES_RESULTS_DIR}/added_value_of_metrics_dim{DIM}.pdf'
plt.savefig(out, bbox_inches='tight')
print(f'\nsaved -> {out}')
plt.show()
