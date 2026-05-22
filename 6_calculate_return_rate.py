from utils import get_removed_algorithms
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
import os

ALL_REMOVED_ALGORITHMS = get_removed_algorithms()
# The specific subset of algorithms we want to cluster
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



def shared_revisit_rate(history, alg1, alg2, by='problem'):
    """
    For each problem, find clusters that both alg1 and alg2 revisited.
    Returns the Jaccard similarity of their revisited cluster sets per problem,
    averaged across all problems or problem class.
    """
    results = []
    
    for data in history[by].unique():
        if by=='problem_class':
            prob_data = history.query('problem_class == @data')
        else:
            prob_data=history.query('problem == @data')

        clusters_alg1 = set(prob_data.query('algorithm == @alg1')['cluster'].unique())
        clusters_alg2 = set(prob_data.query('algorithm == @alg2')['cluster'].unique())
        
        if not clusters_alg1 and not clusters_alg2:
            continue
            
        intersection = len(clusters_alg1 & clusters_alg2)
        union = len(clusters_alg1 | clusters_alg2)
        
        similarity = intersection / union if union > 0 else 0
        if by=='problem_class':
            prob_data = history.query('problem_class == @data')
            results.append({'problem_class': data, 'similarity': similarity,'shared_clusters': intersection,
                'alg1_only': len(clusters_alg1 - clusters_alg2),'alg2_only': len(clusters_alg2 - clusters_alg1)})
        else:
            prob_data=history.query('problem == @data')
            results.append({'problem': data, 'similarity': similarity,'shared_clusters': intersection,
                'alg1_only': len(clusters_alg1 - clusters_alg2),'alg2_only': len(clusters_alg2 - clusters_alg1)})
    
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


def pairwise_revisit_matrix(input_dir, algorithms_of_interest, by='problem'):
    pairs = list(itertools.combinations(algorithms_of_interest, 2))
    pairwise = []
    all_problems_history = get_all_problems_revisiting_history(input_dir, algorithms_of_interest)
    for alg1, alg2 in pairs:
        result = shared_revisit_rate(all_problems_history, alg1, alg2, by=by)
        if not result.empty:
            pairwise.append({
                'algorithm': alg1,
                'algorithm2': alg2,
                'mean_similarity': result['similarity'].mean(),
                'mean_shared_clusters': result['shared_clusters'].mean()
            })

    pairwise_df = pd.DataFrame(pairwise)
    algo_order = sorted(algorithms_of_interest)
    matrix_full = pd.DataFrame(index=algo_order, columns=algo_order, dtype=float)

    for _, row in pairwise_df.iterrows():
        alg1, alg2 = row['algorithm'], row['algorithm2']
        val = row['mean_similarity']
        matrix_full.loc[alg1, alg2] = val
        matrix_full.loc[alg2, alg1] = val  

    for alg in algo_order:
        matrix_full.loc[alg, alg] = 1.0

    print(matrix_full)
    return matrix_full

def get_per_algorithm_revisits(all_problem_history):
    return all_problem_history.groupby(["algorithm"]).agg(total_revisits=('cluster', 'count')).sort_values(by='total_revisits', ascending=False)

def plot_mean_rr(matrix, type="problem"): 
    sns.set_theme(font_scale=0.8)  # smaller font
    fig, ax = plt.subplots(figsize=(10, 8))  # adjust size as needed
    sns.heatmap(matrix, annot=True, fmt='.2f', cmap='YlGnBu', ax=ax)
    plt.title(f'Shared Revisit Rate between Algorithm Pairs in same {type}')
    plt.xticks(rotation=45, ha='right')  # rotate x labels so they don't overlap
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'figures_revisiting/mean_return_rate_to_same_cluster_in_{type}.pdf', bbox_inches='tight', pad_inches=0.2)
    sns.set_theme(font_scale=1)  # reset font scale
    plt.close()

def plot_clustermap(matrix, type="problem"):
    sns.clustermap(matrix, cmap='YlGnBu', figsize=(14, 6), annot=False, standard_scale=0  )
    plt.savefig(f'figures_revisiting/{type}_revisiting_clustermap.pdf', bbox_inches='tight')
    plt.close()

def plot_revisits(per_alg):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=per_alg.reset_index(), x='algorithm', y='total_revisits', palette='tab10', ax=ax)
    ax.set_title('Total Revisits per Algorithm')
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Total Revisits')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'figures_revisiting/total_revisits.pdf')
    plt.close()

if __name__ == "__main__":
    aph = get_all_problems_revisiting_history(INPUT_DIR, algorithms_of_interest=ALGORITHMS_OF_INTEREST)
    per_alg = get_per_algorithm_revisits(aph)
    matrix_full_pc = pairwise_revisit_matrix(INPUT_DIR, ALGORITHMS_OF_INTEREST, by='problem_class')
    matrix_full_p = pairwise_revisit_matrix(INPUT_DIR, ALGORITHMS_OF_INTEREST)

    plot_mean_rr(matrix=matrix_full_p)
    plot_mean_rr(matrix=matrix_full_pc, type="problem class")
    plot_clustermap(matrix=matrix_full_pc, type="problem class")
    plot_revisits(per_alg)


