"""
Diagnostics for the merged file merged_dim_{d}.csv.

Checks three risks from merge_metrics.py:
  1. PAIR ORDER MISMATCH - if one pairwise script writes (A,B) and another
     writes (B,A), the rows don't merge under how='outer' -> duplicated rows
     with NaN. Silent, no warning.
  2. NaN from the outer merge - which metrics are missing, and how many.
  3. Duplicate keys - the same (pair, problem, run) combination more than once.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
import os
from itertools import combinations
from config import METRIC_KEYS as KEYS, MERGED_DIR

DIM = 5
MERGED_PATH = f'{MERGED_DIR}/merged_dim_{DIM}.csv'

N_ALGOS = 28
N_RUNS = 5
N_FUNCTIONS = 24
N_INSTANCES = 5

if not os.path.isfile(MERGED_PATH):
    raise FileNotFoundError(MERGED_PATH)

df = pd.read_csv(MERGED_PATH)
metric_cols = [c for c in df.columns if c not in KEYS]

print('=' * 78)
print(f'DIAGNOSTICS {MERGED_PATH}')
print('=' * 78)
print(f'rows: {len(df):,}')
expected = (N_ALGOS * (N_ALGOS - 1) // 2) * N_RUNS * N_FUNCTIONS * N_INSTANCES
print(f'expected: {expected:,}  '
      f'(C({N_ALGOS},2)={N_ALGOS*(N_ALGOS-1)//2} pairs x {N_RUNS} runs '
      f'x {N_FUNCTIONS} functions x {N_INSTANCES} instances)')
if len(df) > expected * 1.05:
    print('  >>> MORE than expected - possible sign of a pair-order mismatch!')
elif len(df) < expected * 0.95:
    print('  >>> FEWER than expected - data missing somewhere')
else:
    print('  OK')
print(f'metric columns: {metric_cols}')

# ---- 1. pair order ----
print('\n' + '=' * 78)
print('1) PAIR ORDER')
print('=' * 78)
unsorted = df[df['Algorithm1'] > df['Algorithm2']]
print(f'rows where Algorithm1 > Algorithm2 (unsorted): {len(unsorted):,}')
if len(unsorted):
    print('  >>> WARNING: some rows are not in alphabetical order.')
    print('      Examples:')
    print(unsorted[KEYS].head(5).to_string(index=False))

# does the same pair appear in BOTH orders?
pairs_fwd = set(map(tuple, df[['Algorithm1', 'Algorithm2']].drop_duplicates().values))
both_ways = {(a, b) for (a, b) in pairs_fwd if (b, a) in pairs_fwd}
print(f'pairs present in BOTH orders: {len(both_ways)}')
if both_ways:
    print('  >>> SERIOUS ERROR: the same pair written in both directions, rows did not merge!')
    for p in list(both_ways)[:5]:
        print(f'      {p}')
else:
    print('  OK - each pair appears in only one order')

# ---- 2. NaN per metric ----
print('\n' + '=' * 78)
print('2) MISSING VALUES (NaN) PER METRIC')
print('=' * 78)
for m in metric_cols:
    n_nan = df[m].isna().sum()
    pct = 100 * n_nan / len(df)
    flag = ''
    if pct > 50:
        flag = '   <<< MORE THAN HALF - likely a key mismatch!'
    elif pct > 0:
        flag = '   <-- some missing'
    print(f'  {m:20s} {n_nan:>8,} ({pct:5.1f} %){flag}')

n_complete = df[metric_cols].notna().all(axis=1).sum()
print(f'\n  rows with ALL metrics: {n_complete:,} ({100*n_complete/len(df):.1f} %)')

# ---- 3. duplicate keys ----
print('\n' + '=' * 78)
print('3) DUPLICATE KEYS')
print('=' * 78)
dup = df.duplicated(subset=KEYS, keep=False)
print(f'duplicated rows (same key combination): {dup.sum():,}')
if dup.sum():
    print('  >>> WARNING: the same (pair, problem, run) combination appears more than once')
    print(df[dup].sort_values(KEYS).head(6).to_string(index=False))
else:
    print('  OK')

# ---- 4. metric ranges (sanity) ----
print('\n' + '=' * 78)
print('4) METRIC RANGES (sanity check of polarity and scale)')
print('=' * 78)
print(f'{"metric":20s} {"min":>10s} {"max":>10s} {"mean":>10s}')
print('-' * 78)
for m in metric_cols:
    print(f'  {m:18s} {df[m].min():>10.4f} {df[m].max():>10.4f} {df[m].mean():>10.4f}')
print('\n  expected: entropy/cosine/cosine_columns/exploration in [0,1];')
print('            location/fitness unbounded, >= 0')
