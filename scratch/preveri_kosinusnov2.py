"""
INDEPENDENT VERIFICATION of the global cosine-distance calculation.

Recomputes d_cos from the raw cluster_distributions CSVs using the formula
from the thesis, and compares it against the values in merged_dim_{d}.csv
(column 'cosine', filled in by cosine_pairwise.py).

Formula (thesis, cosine-distance section):
    v = (n_{1,1},...,n_{1,k}, n_{2,1},...,n_{T,k})  flattened occupancy table
    d_cos(A,B) = 1 - (v_A . v_B) / (||v_A|| ||v_B||)

Also checks:
  - polarity (distance vs. similarity)
  - seed pairing (same run for both algorithms)
  - whether any scaling was applied before the calculation
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
import os
from itertools import combinations
from config import CLUSTER_DISTRIBUTIONS_LATEST, MERGED_DIR

# ---- settings ----
DIM = 5
F, I = 1, 1
CLUSTER_DIR = f'{CLUSTER_DISTRIBUTIONS_LATEST}/dim_{DIM}'
MERGED_PATH = f'{MERGED_DIR}/merged_dim_{DIM}.csv'

N_CHECK = 15          # how many (pair, run) combinations to check
TOL = 1e-6             # allowed tolerance


def cos_distance(vA, vB):
    """d_cos = 1 - cos(vA, vB); no scaling, exactly per the formula."""
    na, nb = np.linalg.norm(vA), np.linalg.norm(vB)
    if na == 0 or nb == 0:
        return np.nan
    return 1.0 - float(np.dot(vA, vB) / (na * nb))


# ---- 1. load raw cluster occupancies ----
path = f'{CLUSTER_DIR}/F{F}_I{I}.csv'
if not os.path.isfile(path):
    raise FileNotFoundError(f'File not found: {path}')

df = pd.read_csv(path, index_col=[0, 1, 2])   # (algorithm, run, iteration) x clusters
print(f'Source: {path}')
print(f'  shape: {df.shape}, index: {df.index.names}, columns (clusters): {df.shape[1]}')

algos = sorted(df.index.get_level_values('algorithm').unique())
runs = sorted(df.index.get_level_values('run').unique())
print(f'  algorithms: {len(algos)}, runs: {runs}')

# ---- 2. load merged ----
if not os.path.isfile(MERGED_PATH):
    raise FileNotFoundError(f'File not found: {MERGED_PATH}')
merged = pd.read_csv(MERGED_PATH)
sub = merged[(merged['Function_id'] == F) & (merged['Instance_id'] == I)]
print(f'\nmerged: {len(sub)} rows for F{F}_I{I}')
print(f'  cosine column range: [{sub["cosine"].min():.4f}, {sub["cosine"].max():.4f}]')

# ---- 3. compare ----
print('\n' + '=' * 90)
print('COMPARISON: independent calculation vs. merged')
print('=' * 90)
print(f'{"pair":42s} {"run":>4s} {"my d_cos":>11s} {"merged":>10s} {"diff":>10s}  match')
print('-' * 90)

checked = 0
mismatches = []
for a1, a2 in combinations(algos, 2):
    if checked >= N_CHECK:
        break
    for r in runs:
        if checked >= N_CHECK:
            break
        try:
            occA = df.loc[(a1, r)]
            occB = df.loc[(a2, r)]
        except KeyError:
            continue
        if occA.shape != occB.shape:
            print(f'  [skipped] {a1}/{a2} run {r}: different shapes {occA.shape} vs {occB.shape}')
            continue

        vA = occA.values.flatten()   # rows = iterations -> iteration-major
        vB = occB.values.flatten()
        mine = cos_distance(vA, vB)

        row = sub[(((sub['Algorithm1'] == a1) & (sub['Algorithm2'] == a2)) |
                   ((sub['Algorithm1'] == a2) & (sub['Algorithm2'] == a1))) &
                  (sub['Run_id'] == r)]
        if row.empty:
            continue
        theirs = float(row['cosine'].iloc[0])

        diff = abs(mine - theirs)
        ok = diff < TOL
        if not ok:
            mismatches.append((a1, a2, r, mine, theirs, diff))
        print(f'{a1 + " / " + a2:42s} {r:>4} {mine:>11.6f} {theirs:>10.6f} {diff:>10.2e}  '
              + ('YES' if ok else 'NO'))
        checked += 1

# ---- 4. diagnostics on mismatch ----
print('\n' + '=' * 90)
if not mismatches:
    print(f'ALL MATCH ({checked} combinations checked, tolerance {TOL})')
else:
    print(f'MISMATCHES: {len(mismatches)} of {checked}')
    print('\nDiagnostics - checking alternative explanations on the first mismatch:')
    a1, a2, r, mine, theirs, diff = mismatches[0]
    occA = df.loc[(a1, r)]
    occB = df.loc[(a2, r)]
    vA, vB = occA.values.flatten(), occB.values.flatten()

    # (a) is merged perhaps SIMILARITY instead of distance?
    print(f'  (a) 1 - merged = {1 - theirs:.6f}  (my d_cos = {mine:.6f})'
          + ('   <-- POLARITY FLIPPED!' if abs((1 - theirs) - mine) < TOL else ''))

    # (b) was min-max scaling applied before the calculation?
    from sklearn.preprocessing import MinMaxScaler
    both = pd.concat([occA, occB])
    scaled = MinMaxScaler().fit_transform(both)
    sA = scaled[:len(occA)].flatten()
    sB = scaled[len(occA):].flatten()
    d_scaled = cos_distance(sA, sB)
    print(f'  (b) with MinMaxScaler: {d_scaled:.6f}'
          + ('   <-- MATCHES, so it is scaled!' if abs(d_scaled - theirs) < TOL else ''))

    # (c) is the flattening column-major instead of row-major?
    d_T = cos_distance(occA.values.T.flatten(), occB.values.T.flatten())
    print(f'  (c) column-major flattening: {d_T:.6f}'
          + ('   <-- MATCHES, different order!' if abs(d_T - theirs) < TOL else ''))

    # (d) is it perhaps averaged across runs instead of per-run?
    rows_all = sub[((sub['Algorithm1'] == a1) & (sub['Algorithm2'] == a2)) |
                   ((sub['Algorithm1'] == a2) & (sub['Algorithm2'] == a1))]
    print(f'  (d) merged has {len(rows_all)} rows for this pair '
          f'(expected {len(runs)} if per-run)')

# ---- 5. extra check: range and polarity ----
print('\n' + '=' * 90)
print('POLARITY CHECK (d_cos must be a DISTANCE: 0 = identical behavior)')
print('=' * 90)
self_pairs = []
for a in algos[:5]:
    for r1, r2 in combinations(runs, 2):
        try:
            v1 = df.loc[(a, r1)].values.flatten()
            v2 = df.loc[(a, r2)].values.flatten()
        except KeyError:
            continue
        self_pairs.append(cos_distance(v1, v2))
if self_pairs:
    print(f'  same algorithm, different runs: mean d_cos = {np.mean(self_pairs):.4f}')
    print(f'  different algorithms (from merged):  mean d_cos = {sub["cosine"].mean():.4f}')
    print('  -> the first must be LOWER than the second, otherwise polarity is flipped')
