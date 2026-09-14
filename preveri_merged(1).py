"""
Diagnostika zdruzene datoteke merged_dim_{d}.csv.

Preverja tri tveganja iz merge_metrics.py:
  1. NEUJEMANJE VRSTNEGA REDA V PARU - ce ena pairwise skripta zapise (A,B),
     druga pa (B,A), se vrstici pri how='outer' NE zdruzita -> podvojene
     vrstice z NaN. Tiho, brez opozorila.
  2. NaN zaradi outer merge - katere mere manjkajo in koliko.
  3. Podvojeni kljuci - ista kombinacija (par, problem, zagon) veckrat.
"""
import pandas as pd
import numpy as np
import os
from itertools import combinations

DIM = 5
MERGED_PATH = f'metrics_data/merged/merged_dim_{DIM}.csv'
KEYS = ['Algorithm1', 'Algorithm2', 'Function_id', 'Instance_id', 'Run_id']

N_ALGOS = 28
N_RUNS = 5
N_FUNCTIONS = 24
N_INSTANCES = 5

if not os.path.isfile(MERGED_PATH):
    raise FileNotFoundError(MERGED_PATH)

df = pd.read_csv(MERGED_PATH)
metric_cols = [c for c in df.columns if c not in KEYS]

print('=' * 78)
print(f'DIAGNOSTIKA {MERGED_PATH}')
print('=' * 78)
print(f'vrstic: {len(df):,}')
expected = (N_ALGOS * (N_ALGOS - 1) // 2) * N_RUNS * N_FUNCTIONS * N_INSTANCES
print(f'pricakovano: {expected:,}  '
      f'(C({N_ALGOS},2)={N_ALGOS*(N_ALGOS-1)//2} parov x {N_RUNS} zagonov '
      f'x {N_FUNCTIONS} funkcij x {N_INSTANCES} instanc)')
if len(df) > expected * 1.05:
    print('  >>> VEC kot pricakovano - mozen znak neujemanja vrstnega reda v paru!')
elif len(df) < expected * 0.95:
    print('  >>> MANJ kot pricakovano - nekje manjkajo podatki')
else:
    print('  OK')
print(f'stolpci mer: {metric_cols}')

# ---- 1. vrstni red v paru ----
print('\n' + '=' * 78)
print('1) VRSTNI RED V PARU')
print('=' * 78)
unsorted = df[df['Algorithm1'] > df['Algorithm2']]
print(f'vrstic, kjer Algorithm1 > Algorithm2 (nesortirano): {len(unsorted):,}')
if len(unsorted):
    print('  >>> OPOZORILO: nekatere vrstice niso v abecednem vrstnem redu.')
    print('      Primeri:')
    print(unsorted[KEYS].head(5).to_string(index=False))

# ali se isti par pojavi v OBEH vrstnih redih?
pairs_fwd = set(map(tuple, df[['Algorithm1', 'Algorithm2']].drop_duplicates().values))
both_ways = {(a, b) for (a, b) in pairs_fwd if (b, a) in pairs_fwd}
print(f'parov, prisotnih v OBEH vrstnih redih: {len(both_ways)}')
if both_ways:
    print('  >>> RESNA NAPAKA: isti par zapisan v obeh smereh, vrstici se nista zdruzili!')
    for p in list(both_ways)[:5]:
        print(f'      {p}')
else:
    print('  OK - vsak par le v enem vrstnem redu')

# ---- 2. NaN po merah ----
print('\n' + '=' * 78)
print('2) MANJKAJOCE VREDNOSTI (NaN) po merah')
print('=' * 78)
for m in metric_cols:
    n_nan = df[m].isna().sum()
    pct = 100 * n_nan / len(df)
    flag = ''
    if pct > 50:
        flag = '   <<< VEC KOT POLOVICA - verjetno neujemanje kljucev!'
    elif pct > 0:
        flag = '   <-- nekaj manjka'
    print(f'  {m:20s} {n_nan:>8,} ({pct:5.1f} %){flag}')

n_complete = df[metric_cols].notna().all(axis=1).sum()
print(f'\n  vrstic z VSEMI merami: {n_complete:,} ({100*n_complete/len(df):.1f} %)')

# ---- 3. podvojeni kljuci ----
print('\n' + '=' * 78)
print('3) PODVOJENI KLJUCI')
print('=' * 78)
dup = df.duplicated(subset=KEYS, keep=False)
print(f'podvojenih vrstic (ista kombinacija kljucev): {dup.sum():,}')
if dup.sum():
    print('  >>> OPOZORILO: ista kombinacija (par, problem, zagon) se pojavi veckrat')
    print(df[dup].sort_values(KEYS).head(6).to_string(index=False))
else:
    print('  OK')

# ---- 4. razponi mer (sanity) ----
print('\n' + '=' * 78)
print('4) RAZPONI MER (sanity check polaritete in skale)')
print('=' * 78)
print(f'{"mera":20s} {"min":>10s} {"max":>10s} {"povprecje":>10s}')
print('-' * 78)
for m in metric_cols:
    print(f'  {m:18s} {df[m].min():>10.4f} {df[m].max():>10.4f} {df[m].mean():>10.4f}')
print('\n  pricakovano: entropy/cosine/cosine_columns/exploration v [0,1];')
print('               location/fitness neomejena, >= 0')
