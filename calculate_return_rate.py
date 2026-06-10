from utils import get_removed_algorithms
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
import os
import numpy as np
from sklearn.manifold import MDS

ALL_REMOVED_ALGORITHMS = get_removed_algorithms()
ALGORITHMS_OF_INTEREST = ["JADE", "OriginalDE",
                        "SADE", "OriginalSHADE",
                        "ModifiedAEO","OriginalAEO", 
                        "AugmentedAEO","HI_WOA", 
                        "OriginalWOA", "OriginalALO", 
                        "OriginalSSA", "OriginalMFO", 
                        "OriginalHHO", "OriginalMPA", 
                        "OriginalMRFO", "WhaleFOA", 
                        "GWO_WOA", "OriginalGWO", 
                        "IGWO", "RW_GWO"]
DIMENSIONS = [2, 5, 10]
INPUT_DIR = 'data/clustering_features_20_algorithms_kmeans/cluster_distributions'
OUTPUT_DIR = 'figures_revisiting'

def revisiting_history(d,threshold=3):
    visit_history = {}
    returns = []
    for iteration, row in d.iterrows():
        active_clusters = set(row[row > threshold].index)
        
        for cluster in active_clusters:
            if cluster in visit_history:
                # algorithm has been here before
                last_visit = visit_history[cluster]
                if last_visit < iteration:  
                    returns.append({
                        'iteration': iteration,
                        'cluster': cluster,
                        'last_visited': last_visit,
                        'gap': iteration - last_visit,
                        'agents_count': row[cluster]
                    })
            visit_history[cluster] = iteration

    
    return pd.DataFrame(returns)

def shared_rr_for_problem(history, alg1, alg2, problem):
    prob_data = history.query('problem == @problem')
    
    clusters_alg1 = set(prob_data.query('algorithm == @alg1')['cluster'].unique())
    clusters_alg2 = set(prob_data.query('algorithm == @alg2')['cluster'].unique())
    
    if not clusters_alg1 and not clusters_alg2:
        return pd.DataFrame()
    
    intersection = len(clusters_alg1 & clusters_alg2)
    union = len(clusters_alg1 | clusters_alg2)
    similarity = intersection / union if union > 0 else 0
    
    return pd.DataFrame([{
        'problem': problem,
        'similarity': similarity,
        'shared_clusters': intersection,
        'alg1_only': len(clusters_alg1 - clusters_alg2),
        'alg2_only': len(clusters_alg2 - clusters_alg1)
    }])

def shared_revisit_rate(history, alg1, alg2, by='problem', weighted=False):
    results = []
    for data in history[by].unique():
        if by == 'problem_class':
            prob_data = history.query('problem_class == @data')
        else:
            prob_data = history.query('problem == @data')

        run_similarities = []
        for run in prob_data['run'].unique():
            run_data = prob_data[prob_data['run'] == run]
            clusters_alg1 = set(run_data.query('algorithm == @alg1')['cluster'].unique())
            clusters_alg2 = set(run_data.query('algorithm == @alg2')['cluster'].unique())

            if not clusters_alg1 and not clusters_alg2:
                continue

            if weighted:
                w1 = run_data.query('algorithm==@alg1')['cluster'].value_counts()
                w2 = run_data.query('algorithm==@alg2')['cluster'].value_counts()
                all_clusters = w1.index.union(w2.index)
                w1 = w1.reindex(all_clusters, fill_value=0)
                w2 = w2.reindex(all_clusters, fill_value=0)
                intersection = (np.minimum(w1, w2)).sum()
                union = (np.maximum(w1, w2)).sum()
            else:
                intersection = len(clusters_alg1 & clusters_alg2)
                union = len(clusters_alg1 | clusters_alg2)

            similarity = intersection / union if union > 0 else 0
            run_similarities.append(similarity)

        if not run_similarities:
            continue

        similarity = np.mean(run_similarities)
        results.append({
            by: data,
            'similarity': similarity,
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
            for run in range(1, 6):
                try:
                    df = revisiting_history(d.loc[alg, run])
                except KeyError:
                    continue
                if not df.empty:
                    df["algorithm"] = alg
                    df["run"] = run
                    history.append(df)
        if history:
            problem_history = pd.concat(history, ignore_index=True)
            problem_history['problem'] = problem_name
            problem_history['problem_class'] = problem_class
            problem_history['instance'] = instance
            all_problems_history.append(problem_history)

    return pd.concat(all_problems_history, ignore_index=True) if all_problems_history else pd.DataFrame()


def pairwise_revisit_matrix(input_dir, algorithms_of_interest, specified_problem = None, by='problem', weighted = False, history = None):
    pairs = list(itertools.combinations(algorithms_of_interest, 2))
    pairwise = []
    if history is None:
        all_problems_history = get_all_problems_revisiting_history(input_dir, algorithms_of_interest)
    else:
        all_problems_history = history
        
    for alg1, alg2 in pairs:
        if specified_problem is not None:
            result = shared_rr_for_problem(all_problems_history, alg1, alg2, problem=specified_problem)
        else:
            result = shared_revisit_rate(all_problems_history, alg1, alg2, by=by, weighted=weighted)
        
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

def plot_mean_rr(matrix,output_dir, type="problem", w = False): 
    sns.set_theme(font_scale=0.8)  # smaller font
    fig, ax = plt.subplots(figsize=(10, 8))  # adjust size as needed
    sns.heatmap(matrix, annot=True, fmt='.2f', cmap='YlGnBu', ax=ax)
    title = f'Shared Revisit Rate between Algorithm Pairs in same {type}, weighted' if w else f'Shared Revisit Rate between Algorithm Pairs in same {type}'
    plt.title(title)
    plt.xticks(rotation=45, ha='right')  # rotate x labels so they don't overlap
    plt.yticks(rotation=0)
    plt.tight_layout()
    save_path = f'{output_dir}/mean_return_rate_to_same_cluster_in_{type}_weighted.pdf' if w else f'{output_dir}/mean_return_rate_to_same_cluster_in_{type}.pdf'
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.2)
    sns.set_theme(font_scale=1)  # reset font scale
    plt.close()

def plot_clustermap(matrix, output_dir, type="problem"):
    sns.clustermap(matrix, cmap='YlGnBu', figsize=(14, 6), annot=False, standard_scale=0  )
    plt.savefig(f'{output_dir}/{type}_revisiting_clustermap.pdf', bbox_inches='tight')
    plt.close()

def plot_revisits(per_alg, output_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=per_alg.reset_index(), x='algorithm', y='total_revisits', palette='tab10', ax=ax)
    ax.set_title('Total Revisits per Algorithm')
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Total Revisits')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/total_revisits.pdf')
    plt.close()

def plot_mds(matrix, output_dir, type="problem"):
    distance_matrix = 1 - matrix.astype(float)
    distance_matrix = distance_matrix.fillna(1.0)
    np.fill_diagonal(distance_matrix.values, 0.0)  

    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, normalized_stress='auto')
    coords = mds.fit_transform(distance_matrix.values)

    algo_names = matrix.index.tolist()

    # group by family for color coding
    family_map = {
        'DE': ['JADE', 'OriginalDE', 'SADE', 'OriginalSHADE'],
        'AEO': ['ModifiedAEO', 'OriginalAEO', 'AugmentedAEO'],
        'WOA': ['OriginalWOA', 'HI_WOA', 'WhaleFOA', 'GWO_WOA'],
        'GWO': ['OriginalGWO', 'IGWO', 'RW_GWO'],
        'Other': ['OriginalSSA', 'OriginalMFO', 'OriginalHHO', 'OriginalMPA', 'OriginalMRFO']
    }
    palette = {'DE': '#e63946', 'AEO': '#2a9d8f', 'WOA': '#457b9d', 'GWO': '#f4a261', 'Other': '#9b5de5'}

    def get_family(alg):
        for family, members in family_map.items():
            if alg in members:
                return family
        return 'Other'

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('#f8f9fa')

    for i, alg in enumerate(algo_names):
        family = get_family(alg)
        color = palette[family]
        ax.scatter(coords[i, 0], coords[i, 1], color=color, s=120, zorder=3, edgecolors='white', linewidths=1.5)
        ax.annotate(alg, (coords[i, 0], coords[i, 1]),
                    textcoords="offset points", xytext=(8, 4),
                    fontsize=8, color=color, fontweight='bold')

    # legend
    for family, color in palette.items():
        ax.scatter([], [], color=color, label=family, s=80)
    ax.legend(title='Family', framealpha=0.8, fontsize=8)

    ax.set_title(f'MDS — Behavioral Distance between Algorithms ({type})', fontsize=13, pad=15)
    ax.set_xlabel('MDS Dimension 1')
    ax.set_ylabel('MDS Dimension 2')
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/mds_behavioral_distance_{type}.pdf', bbox_inches='tight')
    plt.close()

def plot_mean_rr_problem(matrix, output_dir, problem_name):
    sns.set_theme(font_scale=0.8)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt='.2f', cmap='YlGnBu', ax=ax)
    plt.title(f'Shared Revisit Rate between Algorithm Pairs — {problem_name}')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/mean_return_rate_{problem_name}.pdf', bbox_inches='tight', pad_inches=0.2)
    sns.set_theme(font_scale=1)
    plt.close()


if __name__ == "__main__":
    for d in DIMENSIONS:    
        input_dir = f'{INPUT_DIR}/dim_{d}'
        os.makedirs(input_dir, exist_ok=True)
        output_dir = f'{OUTPUT_DIR}/dim_{d}'
        os.makedirs(output_dir, exist_ok=True)

        aph = get_all_problems_revisiting_history(input_dir, algorithms_of_interest=ALGORITHMS_OF_INTEREST)
        per_alg = get_per_algorithm_revisits(aph)
        problem = "F1_I1"
        matrix = pairwise_revisit_matrix(input_dir, ALGORITHMS_OF_INTEREST,specified_problem=problem)
        plot_mean_rr_problem(matrix, output_dir, problem)
        plot_revisits(per_alg, output_dir)
        
        """
        matrix_full_pc = pairwise_revisit_matrix(input_dir, ALGORITHMS_OF_INTEREST, by='problem_class')
        matrix_full_p = pairwise_revisit_matrix(input_dir, ALGORITHMS_OF_INTEREST)
        matrix_full_p_weighted = pairwise_revisit_matrix(input_dir, ALGORITHMS_OF_INTEREST, weighted=True)
        matrix_full_pc_weighted = pairwise_revisit_matrix(input_dir, ALGORITHMS_OF_INTEREST, by='problem_class', weighted=True)

        plot_mean_rr(matrix=matrix_full_p, output_dir=output_dir)
        plot_mean_rr(matrix=matrix_full_pc, output_dir=output_dir, type="problem class")
        plot_mean_rr(matrix=matrix_full_p_weighted, output_dir=output_dir, w = True)
        plot_mean_rr(matrix=matrix_full_pc_weighted, output_dir = output_dir, type="problem_class", w = True)
        plot_mds(matrix_full_p, output_dir=output_dir)
        plot_mds(matrix_full_pc, output_dir=output_dir, type="problem_class")
        """
    #plot_clustermap(matrix=matrix_full_pc, type="problem class")
    
    #plot_revisits(per_alg)


