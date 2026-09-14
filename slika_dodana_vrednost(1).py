"""
Slika: dodana vrednost predlaganih mer nad podedovano ClustOpt kosinusno.

Stiri pari algoritmov (2 z visoko, 2 z nizko ClustOpt podobnostjo), za vsak
par heatmap vseh sestih mer na problemih F1 in F23 ter povprecje cez vse
probleme.

Poanta: pri parih z NIZKO ClustOpt podobnostjo (spodnja vrstica) entropija in
raziskovanje pokazeta, da sta si algoritma vseeno podobna - informacija, ki je
kosinusna razdalja sama ne zajame.

Normalizacija: min-max ZNOTRAJ vsakega problema, cez VSE pare na tem problemu
(ne le cez prikazane stiri) - celica pove, kje je par glede na razpon vseh
parov. Ce se fitness/lokacija zaradi osamelcev izkazeta za stisnjeni, preklopi
NORMALIZACIJA = 'rank'.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ---- nastavitve ----
DIM = 2
MERGED_PATH = f'metrics_data/merged/merged_dim_{DIM}.csv'
NORMALIZACIJA = 'rank'    # 'minmax' ali 'rank'
PROBLEMI = [(1, 1), (23, 1)]    # (Function_id, Instance_id) stolpca

# pari: (algoritem1, algoritem2, oznaka) - abecedno urejeni, kot v podatkih
PARI_VISOKA = [
    ('JADE', 'OriginalSHADE'),
    ('OriginalSHADE', 'SADE'),
]
PARI_NIZKA = [
    ('AugmentedAEO', 'OriginalALO'),
    ('OriginalDE', 'OriginalMRFO'),
]

REFERENCE = 'cosine'
PROPOSED = ['entropy', 'cosine_columns', 'exploration', 'location', 'fitness']
ALL_METRICS = [REFERENCE] + PROPOSED

IMENA_MER = {
    'cosine':          'ClustOpt kosinusna\nrazdalja',
    'entropy':         'Razlika v entropiji',
    'cosine_columns':  'Kosinusna razdalja\npo gručah',
    'exploration':     'Razlika v\nraziskovanju',
    'location':        'Razlika v končni\nlokaciji',
    'fitness':         'Razlika v končni\nkakovosti',
}
IMENA_STOLPCEV = ['Prob. 1', 'Prob. 23', 'Povp.']

KEYS = ['Algorithm1', 'Algorithm2']

# ---- 1. nalozi ----
if not os.path.isfile(MERGED_PATH):
    raise FileNotFoundError(MERGED_PATH)
df = pd.read_csv(MERGED_PATH)

# povprecje cez zagone -> ena vrstica na (par, funkcija, instanca)
pp = (df.groupby(KEYS + ['Function_id', 'Instance_id'])[ALL_METRICS]
        .mean().reset_index())

# ---- 2. normalizacija ZNOTRAJ problema, cez VSE pare ----
for m in ALL_METRICS:
    grp = pp.groupby(['Function_id', 'Instance_id'])[m]
    if NORMALIZACIJA == 'rank':
        # percentilni rang: robusten na osamelce
        pp[f'{m}_n'] = grp.rank(pct=True)
    else:
        lo, hi = grp.transform('min'), grp.transform('max')
        span = hi - lo
        norm = (pp[m] - lo) / span.where(span > 0)
        pp[f'{m}_n'] = norm.mask(span.eq(0) & pp[m].notna(), 0.0)

norm_cols = [f'{m}_n' for m in ALL_METRICS]

# povprecje normaliziranih vrednosti cez vse probleme
pair_avg = pp.groupby(KEYS)[norm_cols].mean()
# surova ClustOpt razdalja (za znacko podobnosti)
pair_avg_raw = pp.groupby(KEYS)[REFERENCE].mean()


def vrednosti_za_par(a1, a2):
    """Vrne (6 x 3) matriko: vrstice = mere, stolpci = F1, F23, povprecje."""
    stolpci = []
    for f, i in PROBLEMI:
        sel = pp[(pp['Algorithm1'] == a1) & (pp['Algorithm2'] == a2) &
                 (pp['Function_id'] == f) & (pp['Instance_id'] == i)]
        if sel.empty:
            print(f'  [opozorilo] ni podatkov za {a1}/{a2} na F{f}_I{i}')
            stolpci.append(np.full(len(ALL_METRICS), np.nan))
        else:
            stolpci.append(sel[norm_cols].iloc[0].to_numpy())
    if (a1, a2) not in pair_avg.index:
        raise KeyError(f'Par {a1}/{a2} ni v podatkih - preveri vrstni red imen.')
    stolpci.append(pair_avg.loc[(a1, a2), norm_cols].to_numpy())
    return np.column_stack(stolpci)


# ---- 3. graf: 2x2 mreza kartic ----
vsi_pari = [(p, 'visoka') for p in PARI_VISOKA] + [(p, 'nizka') for p in PARI_NIZKA]

fig, axes = plt.subplots(2, 2, figsize=(13, 11))
axes = axes.flatten()

print('=' * 78)
print(f'VREDNOSTI NA SLIKI (normalizacija: {NORMALIZACIJA})')
print('=' * 78)

for ax, ((a1, a2), tip) in zip(axes, vsi_pari):
    M = vrednosti_za_par(a1, a2)
    podobnost = 1.0 - pair_avg_raw.loc[(a1, a2)]

    tabela = pd.DataFrame(M, index=[IMENA_MER[m] for m in ALL_METRICS],
                          columns=IMENA_STOLPCEV)

    print(f'\n{a1} / {a2}   (ClustOpt podobnost = {podobnost:.2f}, {tip})')
    print(tabela.to_string(float_format='%.3f'))

    sns.heatmap(tabela, ax=ax, cmap='Blues', vmin=0, vmax=1, annot=True,
                fmt='.2f', linewidths=1.5, linecolor='white', cbar=False,
                annot_kws={'size': 10})

    barva = '#2a9d8f' if tip == 'visoka' else '#e63946'
    ax.set_title(f'{a1}  vs  {a2}\n{tip} ClustOpt podobnost ({podobnost:.2f})',
                 fontsize=11, fontweight='bold', color=barva, pad=12)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(axis='y', rotation=0, labelsize=9)
    ax.tick_params(axis='x', labelsize=10)

# skupna barvna lestvica spodaj
fig.subplots_adjust(bottom=0.10, hspace=0.45, wspace=0.55)
cax = fig.add_axes([0.25, 0.04, 0.5, 0.016])
sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=0, vmax=1))
cb = fig.colorbar(sm, cax=cax, orientation='horizontal')
cb.set_label('normalizirana razlika med algoritmoma v paru '
             '(0 = najbolj podobna, 1 = najbolj različna)', fontsize=9)

os.makedirs('figures_rezultati', exist_ok=True)
out = f'figures_rezultati/dodana_vrednost_mer_dim{DIM}.pdf'
plt.savefig(out, bbox_inches='tight')
print(f'\nshranjeno -> {out}')
plt.show()
