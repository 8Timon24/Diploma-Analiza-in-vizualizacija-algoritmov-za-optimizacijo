# Shared helpers for the clustering scripts: algorithm-family lookup,
# per-(algorithm, run) feature aggregation, min-max rescaling, and the
# removed-algorithms filter list.
import pandas as pd
import os
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.patches as mpatches
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
import mealpy
import itertools


def get_algorithm_groups():
    optimizers=mealpy.get_all_optimizers(verbose=False)
    optimizer_group={}
    for k,v in optimizers.items():
        optimizer_group[k.replace('Original','Base')]=str(v).split('.')[1]
        optimizer_group[k]=str(v).split('.')[1]
    optimizer_group['ModifiedBA']='swarm_based'
    optimizer_group['SHADE']='evolutionary_based'
    
    return optimizer_group

def calculate_dynamorep_features(df, x_y_columns, id_columns):
    df=df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    grouped = df[x_y_columns + id_columns].groupby(id_columns)
    features = pd.concat([grouped.mean(), grouped.min(), grouped.max(), grouped.std()], axis=1)
    feature_names = [f'{j}_{i}' for j in
                         ['mean', 'min', 'max', 'std'] for i in x_y_columns]
    
    features.columns = feature_names
    return features

def rescale(d,x_y_columns):
    scaled=list(filter(lambda x: x.startswith('scaled_'), d.columns))
    d=d.drop(columns=scaled)
    d[[f'scaled_{x}' for x in x_y_columns]]=MinMaxScaler().fit_transform(d[x_y_columns])
    return d

def get_removed_algorithms():
    return pd.DataFrame(index=pd.Index([], name='algorithm'))
