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

ALGORITHMS_OF_INTEREST = [
    "ModifiedAEO",
    "OriginalAEO",
    "AugmentedAEO",
    "OriginalSHADE",
    "SADE",
    "JADE",
    "OriginalDE",
    "OriginalWOA",
    "HI_WOA",
]

INPUT_DIR = 'data/clustering_features_x_only_10_algorithms_kmeans_2pow_no_init/cluster_distributions/dim_2'

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

def revisiting_history(d,threshold=3):
    visit_history = {}
    returns = []
    for iteration, row in d.iterrows():
        active_clusters = set(row[row > threshold].index)
        
        for cluster in active_clusters:
            if cluster in visit_history:
                # algorithm has been here before
                last_visit = visit_history[cluster]
                if last_visit < iteration - 1:  
                    returns.append({
                        'iteration': iteration,
                        'cluster': cluster,
                        'last_visited': last_visit,
                        'gap': iteration - last_visit,
                        'agents_count': row[cluster]
                    })
            visit_history[cluster] = iteration

    
    return pd.DataFrame(returns)



def shared_revisit_rate(history, alg1, alg2):
    """
    For each problem, find clusters that both alg1 and alg2 revisited.
    Returns the Jaccard similarity of their revisited cluster sets per problem,
    averaged across all problems.
    """
    results = []
    
    for problem_class in history['problem_class'].unique():
        prob_data = history.query('problem_class == @problem_class')
        
        clusters_alg1 = set(prob_data.query('algorithm == @alg1')['cluster'].unique())
        clusters_alg2 = set(prob_data.query('algorithm == @alg2')['cluster'].unique())
        
        if not clusters_alg1 and not clusters_alg2:
            continue
            
        intersection = len(clusters_alg1 & clusters_alg2)
        union = len(clusters_alg1 | clusters_alg2)
        
        similarity = intersection / union if union > 0 else 0
        results.append({
            'problem_class': problem_class,
            'similarity': similarity,
            'shared_clusters': intersection,
            'alg1_only': len(clusters_alg1 - clusters_alg2),
            'alg2_only': len(clusters_alg2 - clusters_alg1)
        })
    
    return pd.DataFrame(results)

def get_all_problems_revisiting_history(input_dir, algorithms_of_interest):
    all_problems_history = []
    for filename in tqdm(os.listdir(input_dir)):
        
        filepath = f'{input_dir}/{filename}'
        history = []
        problem_name = filename.replace('.csv', '')
        problem_class = problem_name.split('_')[0]
        instance = problem_name.split('_')[1]
        d = pd.read_csv(filepath, index_col=[0, 1, 2])
        
        for alg in algorithms_of_interest:
            df = revisiting_history(d.loc[alg, 1])
            if not df.empty:
                df["algorithm"] = alg
                history.append(df)
        
        if history:
            problem_history = pd.concat(history, ignore_index=True)
            problem_history['problem'] = problem_name
            problem_history['problem_class'] = problem_class
            problem_history['instance'] = instance
            all_problems_history.append(problem_history)

    return pd.concat(all_problems_history, ignore_index=True) if all_problems_history else pd.DataFrame()


def pairwise_revisit_matrix(input_dir, algorithms_of_interest):
    pairs = list(itertools.combinations(algorithms_of_interest, 2))
    pairwise = []
    all_problems_history = get_all_problems_revisiting_history(input_dir, algorithms_of_interest)
    for alg1, alg2 in pairs:
        result = shared_revisit_rate(all_problems_history, alg1, alg2)
        if not result.empty:
            pairwise.append({
                'algorithm': alg1,
                'algorithm2': alg2,
                'mean_similarity': result['similarity'].mean(),
                'mean_shared_clusters': result['shared_clusters'].mean()
            })

    pairwise_df = pd.DataFrame(pairwise)
    algo_order = sorted(ALGORITHMS_OF_INTEREST)
    matrix_full = pd.DataFrame(index=algo_order, columns=algo_order, dtype=float)

    for _, row in pairwise_df.iterrows():
        alg1, alg2 = row['algorithm'], row['algorithm2']
        val = row['mean_similarity']
        matrix_full.loc[alg1, alg2] = val
        matrix_full.loc[alg2, alg1] = val  # mirror

    for alg in algo_order:
        matrix_full.loc[alg, alg] = 1.0

    print(matrix_full)
    return matrix_full