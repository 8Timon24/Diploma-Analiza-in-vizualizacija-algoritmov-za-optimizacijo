import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.manifold import MDS
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from return_rate_calculation import DIMENSIONS, RR_DATA_DIR

FIGURES_OUTPUT_DIR = 'figures_revisiting'
SPECIFIED_PROBLEM = "F17_I1"


def plot_mean_rr(matrix, fig_dir, type="problem", w=False):
    sns.set_theme(font_scale=0.8)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt='.2f', cmap='YlGnBu', ax=ax)
    title = (f'Shared Revisit Rate between Algorithm Pairs in same {type}, weighted'
             if w else f'Shared Revisit Rate between Algorithm Pairs in same {type}')
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    save_path = (f'{fig_dir}/mean_return_rate_to_same_cluster_in_{type}_weighted.pdf'
                 if w else f'{fig_dir}/mean_return_rate_to_same_cluster_in_{type}.pdf')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.2)
    sns.set_theme(font_scale=1)
    plt.close()


def plot_clustermap(matrix, fig_dir, type="problem"):
    sns.clustermap(matrix, cmap='YlGnBu', figsize=(14, 6), annot=False, standard_scale=0)
    plt.savefig(f'{fig_dir}/{type}_revisiting_clustermap.pdf', bbox_inches='tight')
    plt.close()

def plot_dendrogram(matrix, fig_dir, type="problem", algorithms=None, method='average'):
    """
    Hierarchical-clustering dendrogram of algorithms based on the pairwise
    DISTANCE matrix (1 - similarity), using scipy linkage. This is a more
    honest view of grouping structure than MDS when MDS stress is high, since
    it doesn't force everything into 2D. Leaf labels are colored by algorithm
    family so you can see whether families cohere in the tree.

    method: linkage method ('average', 'ward', 'complete', ...). 'average'
    works directly on the precomputed distances; 'ward' assumes Euclidean and
    is less principled on a precomputed distance matrix, so 'average' is the
    safer default here.
    """
    if algorithms is not None:
        keep = [a for a in matrix.index if a in set(algorithms)]
        matrix = matrix.loc[keep, keep]

    # distance = 1 - similarity, symmetric, zero diagonal
    dist = 1.0 - matrix.astype(float)
    dist_vals = dist.values.copy()
    np.fill_diagonal(dist_vals, 0.0)
    dist_vals = (dist_vals + dist_vals.T) / 2  # enforce exact symmetry
    dist_vals = np.nan_to_num(dist_vals, nan=1.0)

    condensed = squareform(dist_vals, checks=False)
    Z = linkage(condensed, method=method)

    # family color map for leaf labels
    family_map = {
        'DE': ['JADE', 'OriginalDE', 'SADE', 'OriginalSHADE', 'L_SHADE'],
        'AEO': ['ModifiedAEO', 'OriginalAEO', 'AugmentedAEO'],
        'WOA': ['OriginalWOA', 'HI_WOA', 'WhaleFOA', 'GWO_WOA'],
        'GWO': ['OriginalGWO', 'IGWO', 'RW_GWO'],
    }
    palette = {'DE': '#e63946', 'AEO': '#2a9d8f', 'WOA': '#457b9d',
               'GWO': '#f4a261', 'Other': '#9b5de5'}
    def fam(a):
        for f, members in family_map.items():
            if a in members:
                return f
        return 'Other'

    fig, ax = plt.subplots(figsize=(max(10, len(matrix) * 0.5), 7))
    dn = dendrogram(Z, labels=list(matrix.index), ax=ax, leaf_rotation=90)

    # color each leaf label by its family
    for lbl in ax.get_xticklabels():
        lbl.set_color(palette[fam(lbl.get_text())])
        lbl.set_fontweight('bold')
        lbl.set_fontsize(9)

    # legend
    for f, c in palette.items():
        ax.plot([], [], color=c, label=f, marker='s', linestyle='None')
    ax.legend(title='Family', fontsize=8)

    ax.set_title(f'Revisit-Rate Behavioral Dendrogram ({type})')
    ax.set_ylabel('Distance (1 - shared revisit rate)')
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/dendrogram_{type}.pdf', bbox_inches='tight')
    plt.close()

def plot_revisits(per_alg, fig_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=per_alg.reset_index(), x='algorithm', y='total_revisits',
                hue='algorithm', palette='tab10', legend=False, ax=ax)
    ax.set_title('Total Revisits per Algorithm')
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Total Revisits')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/total_revisits.pdf')
    plt.close()


def plot_mds(matrix, fig_dir, type="problem"):
    distance_matrix = 1 - matrix.astype(float)
    distance_matrix = distance_matrix.fillna(1.0)
    np.fill_diagonal(distance_matrix.values, 0.0)

    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, normalized_stress='auto')
    coords = mds.fit_transform(distance_matrix.values)

    algo_names = matrix.index.tolist()

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

    for family, color in palette.items():
        ax.scatter([], [], color=color, label=family, s=80)
    ax.legend(title='Family', framealpha=0.8, fontsize=8)

    ax.set_title(f'MDS — Behavioral Distance between Algorithms ({type})', fontsize=13, pad=15)
    ax.set_xlabel('MDS Dimension 1')
    ax.set_ylabel('MDS Dimension 2')
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/mds_behavioral_distance_{type}.pdf', bbox_inches='tight')
    plt.close()


def plot_mean_rr_problem(matrix, fig_dir, problem_name):
    sns.set_theme(font_scale=0.8)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt='.2f', cmap='YlGnBu', ax=ax)
    plt.title(f'Shared Revisit Rate between Algorithm Pairs — {problem_name}')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/mean_return_rate_{problem_name}.pdf', bbox_inches='tight', pad_inches=0.2)
    sns.set_theme(font_scale=1)
    plt.close()


def _load_matrix(path):
    # matrices were saved with the algorithm names as the index column
    return pd.read_csv(path, index_col=0)


if __name__ == "__main__":
    for d in DIMENSIONS:
        data_dir = f'{RR_DATA_DIR}/dim_{d}'
        if not os.path.isdir(data_dir):
            print(f"Missing {data_dir} - run return_rate_calculation.py first.")
            continue

        fig_dir = f'{FIGURES_OUTPUT_DIR}/dim_{d}'
        os.makedirs(fig_dir, exist_ok=True)

        per_alg = pd.read_csv(f'{data_dir}/per_algorithm_revisits.csv', index_col=0)

        matrix_problem = _load_matrix(f'{data_dir}/matrix_{SPECIFIED_PROBLEM}.csv')
        matrix_pc = _load_matrix(f'{data_dir}/matrix_problem_class.csv')
        matrix_p = _load_matrix(f'{data_dir}/matrix_problem.csv')
        matrix_p_w = _load_matrix(f'{data_dir}/matrix_problem_weighted.csv')
        matrix_pc_w = _load_matrix(f'{data_dir}/matrix_problem_class_weighted.csv')

        plot_mean_rr_problem(matrix_problem, fig_dir, SPECIFIED_PROBLEM)
        plot_revisits(per_alg, fig_dir)

        plot_mean_rr(matrix_p, fig_dir)
        plot_mean_rr(matrix_pc, fig_dir, type="problem class")
        plot_mean_rr(matrix_p_w, fig_dir, w=True)
        plot_mean_rr(matrix_pc_w, fig_dir, type="problem_class", w=True)

        plot_mds(matrix_p, fig_dir)
        plot_mds(matrix_pc, fig_dir, type="problem_class")

        print(f"saved figures for dim={d} -> {fig_dir}")
