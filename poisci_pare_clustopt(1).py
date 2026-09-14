"""
KORAK 1: poisce kandidate za sliko "dodana vrednost predlaganih mer".

Ideja: najti pare algoritmov, ki so si po ClustOpt kosinusni razdalji zelo
PODOBNI (nizka razdalja), a se po preostalih petih merah vseeno RAZLIKUJEJO.
Taki pari so dokaz, da predlagane mere prinesejo informacijo, ki je ClustOpt
kosinusna sama ne zajame.

Za kontrast poisce tudi pare z NIZKO ClustOpt podobnostjo (visoka razdalja).

Normalizacija: min-max ZNOTRAJ vsakega problema, cez vse pare na tem
problemu. Celica tako pove, kje je ta par glede na razpon vseh parov na
istem problemu - s tem sta lokacija in fitness (neomejeni, med funkcijami
razlicnih velikostnih redov) primerljiva z ostalimi merami.
"""
import pandas as pd
import numpy as np
import os

# ---- nastavitve ----
DIM = 2
MERGED_PATH = f'metrics_data/merged/merged_dim_{DIM}.csv'

ALGOS = ["OriginalDE", "SADE", "OriginalSHADE", "JADE", "AugmentedAEO",
         "GWO_WOA", "OriginalGWO", "OriginalMRFO", "OriginalALO", "OriginalAEO"]

REFERENCE = 'cosine'                       # ClustOpt mera (podedovana)
PROPOSED = ['entropy', 'cosine_columns', 'exploration', 'location', 'fitness']
ALL_METRICS = [REFERENCE] + PROPOSED
KEYS = ['Algorithm1', 'Algorithm2']

N_SHOW = 10   # koliko kandidatov izpisati na seznam

# ---- 1. nalozi in filtriraj ----
if not os.path.isfile(MERGED_PATH):
    raise FileNotFoundError(f'Ni datoteke {MERGED_PATH}')

df = pd.read_csv(MERGED_PATH)
df = df[df['Algorithm1'].isin(ALGOS) & df['Algorithm2'].isin(ALGOS)].copy()
if df.empty:
    raise ValueError('Po filtru ALGOS ni ostalo nic - preveri imena algoritmov.')

missing = set(ALGOS) - (set(df['Algorithm1']) | set(df['Algorithm2']))
if missing:
    print(f'  [opozorilo] ti algoritmi niso najdeni v podatkih: {missing}')

# ---- 2. povprecje cez zagone -> ena vrstica na (par, funkcija, instanca) ----
pp = (df.groupby(KEYS + ['Function_id', 'Instance_id'])[ALL_METRICS]
        .mean().reset_index())

# ---- 3. min-max normalizacija ZNOTRAJ vsakega problema ----
# transform namesto groupby.apply: apply v novejsi pandas skupinskih stolpcev
# ne vrne nazaj (Function_id/Instance_id izginili iz rezultata)
for m in ALL_METRICS:
    grp = pp.groupby(['Function_id', 'Instance_id'])[m]
    lo = grp.transform('min')
    hi = grp.transform('max')
    span = hi - lo
    normalized = (pp[m] - lo) / span.where(span > 0)
    # kjer so vsi pari na problemu enaki (span == 0), postavi 0, a LE kjer
    # surova vrednost ni NaN - NaN mora ostati NaN, ne postati lazna nicla
    normalized = normalized.mask(span.eq(0) & pp[m].notna(), 0.0)
    pp[f'{m}_n'] = normalized

    if pp[m].isna().all():
        print(f'  [opozorilo] mera "{m}" je v celoti manjkajoca (NaN)')

norm_cols = [f'{m}_n' for m in ALL_METRICS]

# ---- 4. povprecje normaliziranih vrednosti cez VSE probleme ----
pair_avg = pp.groupby(KEYS)[norm_cols].mean().reset_index()
pair_avg['par'] = pair_avg['Algorithm1'] + ' / ' + pair_avg['Algorithm2']

# povprecje petih predlaganih mer (normaliziranih)
pair_avg['predlagane_povp'] = pair_avg[[f'{m}_n' for m in PROPOSED]].mean(axis=1)

# ---- 5a. KANDIDATI: nizka ClustOpt razdalja (= visoka podobnost),
#          a visoka razlika po predlaganih merah ----
pair_avg['razkorak'] = pair_avg['predlagane_povp'] - pair_avg[f'{REFERENCE}_n']

print('=' * 78)
print('A) VISOKA ClustOpt PODOBNOST, a VELIKA razlika po predlaganih merah')
print('   (to so pari, ki podpirajo argument o dodani vrednosti)')
print('=' * 78)
top_a = pair_avg.sort_values('razkorak', ascending=False).head(N_SHOW)
print(top_a[['par', f'{REFERENCE}_n', 'predlagane_povp', 'razkorak']
            + [f'{m}_n' for m in PROPOSED]].to_string(index=False, float_format='%.3f'))

# ---- 5b. KONTRAST: nizka ClustOpt podobnost (visoka razdalja) ----
print()
print('=' * 78)
print('B) NIZKA ClustOpt PODOBNOST (visoka kosinusna razdalja) - za kontrast')
print('=' * 78)
top_b = pair_avg.sort_values(f'{REFERENCE}_n', ascending=False).head(N_SHOW)
print(top_b[['par', f'{REFERENCE}_n', 'predlagane_povp']
            + [f'{m}_n' for m in PROPOSED]].to_string(index=False, float_format='%.3f'))

# ---- 5c. referenca: najbolj podobni po ClustOpt (ne glede na razkorak) ----
print()
print('=' * 78)
print('C) NAJVISJA ClustOpt PODOBNOST (najnizja kosinusna razdalja)')
print('   - primerjaj z A: ce so ti pari podobni tudi po predlaganih merah,')
print('     je argument o dodani vrednosti sibkejsi')
print('=' * 78)
top_c = pair_avg.sort_values(f'{REFERENCE}_n', ascending=True).head(N_SHOW)
print(top_c[['par', f'{REFERENCE}_n', 'predlagane_povp']
            + [f'{m}_n' for m in PROPOSED]].to_string(index=False, float_format='%.3f'))

# ---- 6. preverba razpolozljivosti F1 in F23 (stolpca v koncni sliki) ----
print()
print('=' * 78)
print('D) PREVERBA: ali sta F1_I1 in F23_I1 na voljo?')
print('=' * 78)
for f in (1, 23):
    sub = pp[(pp['Function_id'] == f) & (pp['Instance_id'] == 1)]
    print(f'  F{f}_I1: {len(sub)} parov'
          + ('' if len(sub) else '   <-- MANJKA!'))
print(f'  skupaj problemov (funkcija x instanca) v podatkih: '
      f'{pp.groupby(["Function_id", "Instance_id"]).ngroups}')
