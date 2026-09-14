"""
STEP 1: finds candidates for the "added value of the proposed metrics" figure.

Idea: find pairs of algorithms that the ClustOpt cosine distance rates as very
SIMILAR (low distance), but that still DIFFER under the remaining five
metrics. Such pairs are evidence that the proposed metrics carry information
the ClustOpt cosine distance alone doesn't capture.

For contrast, it also finds pairs with LOW ClustOpt similarity (high distance).

Normalization: min-max WITHIN each problem, across all pairs on that problem.
A cell thus shows where that pair sits relative to the range of all pairs on
the same problem - this makes location and fitness (unbounded, across
functions of very different magnitudes) comparable with the other metrics.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
import os
from config import MERGED_DIR

# ---- settings ----
DIM = 2
MERGED_PATH = f'{MERGED_DIR}/merged_dim_{DIM}.csv'

ALGOS = ["OriginalDE", "SADE", "OriginalSHADE", "JADE", "AugmentedAEO",
         "GWO_WOA", "OriginalGWO", "OriginalMRFO", "OriginalALO", "OriginalAEO"]

REFERENCE = 'cosine'                       # inherited ClustOpt metric
PROPOSED = ['entropy', 'cosine_columns', 'exploration', 'location', 'fitness']
ALL_METRICS = [REFERENCE] + PROPOSED
KEYS = ['Algorithm1', 'Algorithm2']

N_SHOW = 10   # how many candidates to print per list

# ---- 1. load and filter ----
if not os.path.isfile(MERGED_PATH):
    raise FileNotFoundError(f'File not found: {MERGED_PATH}')

df = pd.read_csv(MERGED_PATH)
df = df[df['Algorithm1'].isin(ALGOS) & df['Algorithm2'].isin(ALGOS)].copy()
if df.empty:
    raise ValueError('Nothing left after the ALGOS filter - check the algorithm names.')

missing = set(ALGOS) - (set(df['Algorithm1']) | set(df['Algorithm2']))
if missing:
    print(f'  [warning] these algorithms were not found in the data: {missing}')

# ---- 2. average across runs -> one row per (pair, function, instance) ----
pp = (df.groupby(KEYS + ['Function_id', 'Instance_id'])[ALL_METRICS]
        .mean().reset_index())

# ---- 3. min-max normalization WITHIN each problem ----
# transform instead of groupby.apply: in newer pandas, apply on grouped
# columns doesn't return Function_id/Instance_id back in the result
for m in ALL_METRICS:
    grp = pp.groupby(['Function_id', 'Instance_id'])[m]
    lo = grp.transform('min')
    hi = grp.transform('max')
    span = hi - lo
    normalized = (pp[m] - lo) / span.where(span > 0)
    # where all pairs on a problem are equal (span == 0), set 0, but ONLY
    # where the raw value isn't NaN - NaN must stay NaN, not become a false zero
    normalized = normalized.mask(span.eq(0) & pp[m].notna(), 0.0)
    pp[f'{m}_n'] = normalized

    if pp[m].isna().all():
        print(f'  [warning] metric "{m}" is entirely missing (NaN)')

norm_cols = [f'{m}_n' for m in ALL_METRICS]

# ---- 4. average normalized values across ALL problems ----
pair_avg = pp.groupby(KEYS)[norm_cols].mean().reset_index()
pair_avg['pair'] = pair_avg['Algorithm1'] + ' / ' + pair_avg['Algorithm2']

# mean of the five proposed metrics (normalized)
pair_avg['proposed_mean'] = pair_avg[[f'{m}_n' for m in PROPOSED]].mean(axis=1)

# ---- 5a. CANDIDATES: low ClustOpt distance (= high similarity),
#          but high difference under the proposed metrics ----
pair_avg['gap'] = pair_avg['proposed_mean'] - pair_avg[f'{REFERENCE}_n']

print('=' * 78)
print('A) HIGH ClustOpt SIMILARITY, but LARGE difference under the proposed metrics')
print('   (these are the pairs that support the added-value argument)')
print('=' * 78)
top_a = pair_avg.sort_values('gap', ascending=False).head(N_SHOW)
print(top_a[['pair', f'{REFERENCE}_n', 'proposed_mean', 'gap']
            + [f'{m}_n' for m in PROPOSED]].to_string(index=False, float_format='%.3f'))

# ---- 5b. CONTRAST: low ClustOpt similarity (high distance) ----
print()
print('=' * 78)
print('B) LOW ClustOpt SIMILARITY (high cosine distance) - for contrast')
print('=' * 78)
top_b = pair_avg.sort_values(f'{REFERENCE}_n', ascending=False).head(N_SHOW)
print(top_b[['pair', f'{REFERENCE}_n', 'proposed_mean']
            + [f'{m}_n' for m in PROPOSED]].to_string(index=False, float_format='%.3f'))

# ---- 5c. reference: most similar under ClustOpt (regardless of the gap) ----
print()
print('=' * 78)
print('C) HIGHEST ClustOpt SIMILARITY (lowest cosine distance)')
print('   - compare with A: if these pairs are also similar under the')
print('     proposed metrics, the added-value argument is weaker')
print('=' * 78)
top_c = pair_avg.sort_values(f'{REFERENCE}_n', ascending=True).head(N_SHOW)
print(top_c[['pair', f'{REFERENCE}_n', 'proposed_mean']
            + [f'{m}_n' for m in PROPOSED]].to_string(index=False, float_format='%.3f'))

# ---- 6. check availability of F1 and F23 (columns in the final figure) ----
print()
print('=' * 78)
print('D) CHECK: are F1_I1 and F23_I1 available?')
print('=' * 78)
for f in (1, 23):
    sub = pp[(pp['Function_id'] == f) & (pp['Instance_id'] == 1)]
    print(f'  F{f}_I1: {len(sub)} pairs'
          + ('' if len(sub) else '   <-- MISSING!'))
print(f'  total problems (function x instance) in the data: '
      f'{pp.groupby(["Function_id", "Instance_id"]).ngroups}')
