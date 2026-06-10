from utils import get_removed_algorithms
import pandas as pd
import math
import seaborn as sns
import matplotlib.pyplot as plt
import os

ALL_REMOVED_ALGORITHMS = get_removed_algorithms()
# The specific subset of algorithms we want to cluster
ALGORITHMS_OF_INTEREST1 = ["JADE", "OriginalDE",
                        "SADE", "OriginalSHADE",
                        "ModifiedAEO","OriginalAEO", 
                        "AugmentedAEO","HI_WOA", 
                        "OriginalWOA", "OriginalALO", 
                        "OriginalSSA", "OriginalMFO", 
                        "OriginalHHO", "OriginalMPA", 
                        "OriginalMRFO", "WhaleFOA", 
                        "GWO_WOA", "OriginalGWO", 
                        "IGWO", "RW_GWO"]
ALGORITHMS_OF_INTEREST = ["HI_WOA", 
                        "OriginalWOA", "WhaleFOA", 
                        "GWO_WOA", "OriginalGWO", 
                        "IGWO", "RW_GWO"]

FUNCTION_GROUPS = {
    'separable': [1, 2, 3, 4, 5],
    'low_conditioning': [6, 7, 8, 9],
    'high_conditioning': [10, 11, 12, 13, 14],
    'multimodal_adequate': [15, 16, 17, 18, 19],
    'multimodal_weak': [20, 21, 22, 23, 24]
}
DIMENSIONS = [2, 5, 10]
FUNCTIONS = [i for i in range(1, 25)]
INSTANCES = [i for i in range(1, 6)]
INPUT_DIR = 'data/clustering_features_20_algorithms_kmeans/cluster_distributions'
OUTPUT_DIR = 'figures_entropy'

def compute_entropy(filepath, algorithms_of_interest=None):
    df = pd.read_csv(filepath, index_col=[0, 1, 2])
    rows = []
    for (alg, run), group in df.groupby(level=[0, 1]):
        if algorithms_of_interest is not None and alg not in algorithms_of_interest:
            continue
        for iteration, row in group.iterrows():
            counts = row.values
            total = counts.sum()
            if total == 0:
                continue
            p = counts / total
            entropy_i = -sum(p_i * math.log(p_i) for p_i in p if p_i > 0)
            rows.append({
                'algorithm': alg,
                'run': run,
                'iteration': iteration[2],
                'entropy': entropy_i
            })
    return pd.DataFrame(rows)

def plot_entropy(data, function, instance, dimension, output_dir = None, save=False):
    plt.figure(figsize=(10, 5))
    
    sns.lineplot(data=data,x='iteration', y='entropy',hue='algorithm',palette='tab10', errorbar=None)

    plt.title(f'Mean Population Entropy over Iterations in function F{function}_{instance}_D{dimension}')
    plt.xlabel('Iteration')
    plt.ylabel('Entropy')
    plt.tight_layout()
    if save and output_dir is not None:
        plt.savefig(f'{output_dir}/Entropy_F{function}_I{instance}_D{dimension}.pdf')
    else:
        plt.show()
    plt.close()

def plot_entropy_by_function_group(all_entropy, output_dir, function_groups):
    function_to_group = {f: group for group, functions in function_groups.items() for f in functions}

    all_entropy['function_group'] = all_entropy['function_class'].map(function_to_group)

    group_entropy = (all_entropy.groupby(['algorithm', 'iteration', 'function_group'])['mean_entropy'].mean().reset_index())

    g = sns.FacetGrid(group_entropy, col='function_group', col_wrap=3, height=4)
    g.map_dataframe(sns.lineplot, x='iteration', y='mean_entropy', hue='algorithm', palette='tab10')
    g.add_legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/entropy_across_function_groups.pdf')
    plt.close()

def plot_entropy_clustermap(all_entropy, output_dir):
    algo_entropy_matrix = (all_entropy.groupby(['algorithm', 'iteration'])['mean_entropy'].mean().unstack(level='iteration'))
 
    sns.clustermap(algo_entropy_matrix, cmap='YlGnBu', figsize=(14, 6),annot=False, standard_scale=0, col_cluster=False)
    plt.savefig(f'{output_dir}/clustermap.pdf', bbox_inches='tight')
    plt.close()

def plot_entropy_overlay(all_entropy_by_dim, functions_of_interest, dims, output_dir, save = False):
    # all_entropy_by_dim: dict {dim: dataframe}
    rows = []
    for dim, df in all_entropy_by_dim.items():
        sub = df[df['function_class'].isin(functions_of_interest)].copy()
        sub['dim'] = dim
        rows.append(sub)
    combined = pd.concat(rows, ignore_index=True)
    combined['function_class'] = combined['function_class'].astype(str)

    g = sns.FacetGrid(combined, row='dim', col='algorithm', height=3, aspect=1.2)
    g.map_dataframe(sns.lineplot, x='iteration', y='mean_entropy',
                    hue='function_class', palette='tab10')
    g.add_legend()
    
    if save : 
        plt.savefig(f'{output_dir}/entropy_separability_overlay.pdf')

def get_all_entropy(input_dir, functions, instances):
    all_entropy = []
    for F in functions:
        problem_entropy_by_function_class = []
        for I in instances:
            problem_entropy = compute_entropy(f'{input_dir}/F{F}_I{I}.csv', ALGORITHMS_OF_INTEREST)
            mean_problem_entropy = problem_entropy.groupby(['algorithm', 'iteration'])['entropy'].agg(mean_entropy='mean', std_entropy = 'std').reset_index()
            mean_problem_entropy['instance'] = I
            mean_problem_entropy['function_class'] = F
            problem_entropy_by_function_class.append(mean_problem_entropy)
        df = pd.concat(problem_entropy_by_function_class, ignore_index=True)
        df = df.groupby(['algorithm', 'iteration', 'function_class'])['mean_entropy'].mean().reset_index()
        #plot_entropy(df, F)
        all_entropy.append(df)

    all_entropy = pd.concat(all_entropy, ignore_index=True)
    return all_entropy

if __name__ == '__main__':

    for d in DIMENSIONS:
        input_dir = f'{INPUT_DIR}/dim_{d}'
        os.makedirs(input_dir, exist_ok=True)
        output_dir = f'{OUTPUT_DIR}/dim_{d}'
        os.makedirs(output_dir, exist_ok=True)
        all_entropy = get_all_entropy(input_dir, FUNCTIONS, INSTANCES)
        plot_entropy_by_function_group(all_entropy, output_dir, FUNCTION_GROUPS)
        plot_entropy_clustermap(all_entropy, output_dir)    
    

