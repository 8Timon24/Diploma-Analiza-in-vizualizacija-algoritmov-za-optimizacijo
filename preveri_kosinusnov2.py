"""
NEODVISNA PREVERBA izracuna globalne kosinusne razdalje.

Iz surovih cluster_distributions CSV-jev znova izracuna d_cos po formuli iz
diplome in primerja z vrednostmi v merged_dim_{d}.csv (stolpec 'cosine',
ki ga napolni cosine_pairwise.py).

Formula (diploma, razdelek o kosinusni razdalji):
    v = (n_{1,1},...,n_{1,k}, n_{2,1},...,n_{T,k})  ploscena tabela zasedenosti
    d_cos(A,B) = 1 - (v_A . v_B) / (||v_A|| ||v_B||)

Preverja tudi:
  - polariteto (razdalja vs podobnost)
  - parjenje po semenu (isti run za oba algoritma)
  - ali je bilo pred izracunom uporabljeno kakrsnokoli skaliranje
"""
import pandas as pd
import numpy as np
import os
from itertools import combinations

# ---- nastavitve ----
DIM = 5
F, I = 1, 1
CLUSTER_DIR = f'data/clustering_latest/cluster_distributions/dim_{DIM}'
MERGED_PATH = f'metrics_data/merged/merged_dim_{DIM}.csv'

N_CHECK = 15          # koliko (par, zagon) kombinacij preveriti
TOL = 1e-6            # dovoljeno odstopanje


def cos_distance(vA, vB):
    """d_cos = 1 - cos(vA, vB); brez skaliranja, natanko po formuli."""
    na, nb = np.linalg.norm(vA), np.linalg.norm(vB)
    if na == 0 or nb == 0:
        return np.nan
    return 1.0 - float(np.dot(vA, vB) / (na * nb))


# ---- 1. nalozi surove zasedenosti gruc ----
path = f'{CLUSTER_DIR}/F{F}_I{I}.csv'
if not os.path.isfile(path):
    raise FileNotFoundError(f'Ni datoteke {path}')

df = pd.read_csv(path, index_col=[0, 1, 2])   # (algorithm, run, iteration) x gruce
print(f'Vir: {path}')
print(f'  oblika: {df.shape}, indeks: {df.index.names}, stolpcev (gruc): {df.shape[1]}')

algos = sorted(df.index.get_level_values('algorithm').unique())
runs = sorted(df.index.get_level_values('run').unique())
print(f'  algoritmov: {len(algos)}, zagonov: {runs}')

# ---- 2. nalozi merged ----
if not os.path.isfile(MERGED_PATH):
    raise FileNotFoundError(f'Ni datoteke {MERGED_PATH}')
merged = pd.read_csv(MERGED_PATH)
sub = merged[(merged['Function_id'] == F) & (merged['Instance_id'] == I)]
print(f'\nmerged: {len(sub)} vrstic za F{F}_I{I}')
print(f'  razpon stolpca cosine: [{sub["cosine"].min():.4f}, {sub["cosine"].max():.4f}]')

# ---- 3. primerjaj ----
print('\n' + '=' * 90)
print('PRIMERJAVA: neodvisen izracun proti merged')
print('=' * 90)
print(f'{"par":42s} {"run":>4s} {"moj d_cos":>11s} {"merged":>10s} {"razlika":>10s}  ujemanje')
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
            print(f'  [preskok] {a1}/{a2} run {r}: razlicni obliki {occA.shape} vs {occB.shape}')
            continue

        vA = occA.values.flatten()   # vrstice = iteracije -> iteration-major
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
              + ('DA' if ok else 'NE'))
        checked += 1

# ---- 4. diagnostika ob neujemanju ----
print('\n' + '=' * 90)
if not mismatches:
    print(f'VSE UJEMA ({checked} preverjenih kombinacij, toleranca {TOL})')
else:
    print(f'NEUJEMANJ: {len(mismatches)} od {checked}')
    print('\nDiagnostika - preverjam alternativne razlage na prvem neujemanju:')
    a1, a2, r, mine, theirs, diff = mismatches[0]
    occA = df.loc[(a1, r)]
    occB = df.loc[(a2, r)]
    vA, vB = occA.values.flatten(), occB.values.flatten()

    # (a) je merged morda PODOBNOST namesto razdalje?
    print(f'  (a) 1 - merged = {1 - theirs:.6f}  (moj d_cos = {mine:.6f})'
          + ('   <-- POLARITETA OBRNJENA!' if abs((1 - theirs) - mine) < TOL else ''))

    # (b) je bilo uporabljeno min-max skaliranje pred izracunom?
    from sklearn.preprocessing import MinMaxScaler
    both = pd.concat([occA, occB])
    scaled = MinMaxScaler().fit_transform(both)
    sA = scaled[:len(occA)].flatten()
    sB = scaled[len(occA):].flatten()
    d_scaled = cos_distance(sA, sB)
    print(f'  (b) z MinMaxScaler: {d_scaled:.6f}'
          + ('   <-- UJEMA, torej se skalira!' if abs(d_scaled - theirs) < TOL else ''))

    # (c) je ploscenje po stolpcih namesto po vrsticah?
    d_T = cos_distance(occA.values.T.flatten(), occB.values.T.flatten())
    print(f'  (c) ploscenje po stolpcih: {d_T:.6f}'
          + ('   <-- UJEMA, drugacen vrstni red!' if abs(d_T - theirs) < TOL else ''))

    # (d) je morda povprecje cez zagone namesto per-zagon?
    rows_all = sub[((sub['Algorithm1'] == a1) & (sub['Algorithm2'] == a2)) |
                   ((sub['Algorithm1'] == a2) & (sub['Algorithm2'] == a1))]
    print(f'  (d) merged ima {len(rows_all)} vrstic za ta par '
          f'(pricakovano {len(runs)}, ce je per-zagon)')

# ---- 5. dodatna preverba: razpon in polariteta ----
print('\n' + '=' * 90)
print('PREVERBA POLARITETE (d_cos mora biti RAZDALJA: 0 = enako vedenje)')
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
    print(f'  isti algoritem, razlicna zagona: povprecje d_cos = {np.mean(self_pairs):.4f}')
    print(f'  razlicni algoritmi (iz merged):  povprecje d_cos = {sub["cosine"].mean():.4f}')
    print('  -> prvo mora biti NIZJE od drugega, sicer je polariteta obrnjena')
