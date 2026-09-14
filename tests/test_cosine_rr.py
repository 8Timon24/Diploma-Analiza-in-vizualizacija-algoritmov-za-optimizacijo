# Ad hoc check: prints the most-similar algorithm pair per function (dim 2)
# under both the column-wise and global cosine-distance metrics.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import os
from config import METRICS_DIR

input_file = f'{METRICS_DIR}/cosine_columns/dim_2/F14_I1.csv'
df = pd.read_csv(input_file)

tmp = df.groupby(['Algorithm1', 'Algorithm2'])['Cosine_column_distance'].mean()
closest_pair = tmp.idxmin()
print(f"Most similar (cosine_columns): {closest_pair[0]} & {closest_pair[1]}  (mean distance {tmp.min():.4f})")

input_filec = f'{METRICS_DIR}/cosine/dim_2/F14_I1.csv'
dfc = pd.read_csv(input_filec)
tmpc = dfc.groupby(['Algorithm1', 'Algorithm2'])['Cosine_distance'].mean()
closest_pair_cosine = tmpc.idxmin()
print(f"Most similar (cosine): {closest_pair_cosine[0]} & {closest_pair_cosine[1]}  (mean distance {tmpc.min():.4f})")

for f in range(1, 25):

    IN_cc = f'{METRICS_DIR}/cosine_columns/dim_2/F{f}_I1.csv'
    df_cc = pd.read_csv(IN_cc)
    tmp_cc = df_cc.groupby(['Algorithm1', 'Algorithm2'])['Cosine_column_distance'].mean()
    closest_pair_cc = tmp_cc.idxmin()

    IN_c = f'{METRICS_DIR}/cosine/dim_2/F{f}_I1.csv'
    df_c = pd.read_csv(IN_c)
    tmp_c = df_c.groupby(['Algorithm1', 'Algorithm2'])['Cosine_distance'].mean()
    closest_pair_c = tmp_c.idxmin()

    print(f'Most similar pairs for cosine column distance: {closest_pair_cc[0]} & {closest_pair_cc[1]} (mean distance {tmp_cc.min():.4f}) \n \
          Most similar pairs for cosine distance: {closest_pair_c[0]} & {closest_pair_c[1]} (mean distance {tmp_c.min():.4f}) ')