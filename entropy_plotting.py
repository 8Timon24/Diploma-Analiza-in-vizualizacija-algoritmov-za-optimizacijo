import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from entropy_calculation import ALGORITHMS_OF_INTEREST, DIMENSIONS, ENTROPY_DATA_DIR

FUNCTION_GROUPS = {
    'separable': [1, 2, 3, 4, 5],
    'low_conditioning': [6, 7, 8, 9],
    'high_conditioning': [10, 11, 12, 13, 14],
    'multimodal_adequate': [15, 16, 17, 18, 19],
    'multimodal_weak': [20, 21, 22, 23, 24]
}
OUTPUT_DIR = 'figures_entropy'

ALGO_PALETTE = dict(zip(
    sorted(ALGORITHMS_OF_INTEREST),
    sns.color_palette("husl", n_colors=len(ALGORITHMS_OF_INTEREST))
))


def _filter_algorithms(data, algorithms):
    """
    Optionally restrict a dataframe to a subset of algorithms.
    algorithms=None -> no filtering (all algorithms kept).
    Warns (but doesn't crash) if any requested algorithm isn't present in
    the data, so a typo in the subset list is visible rather than silent.
    """
    if algorithms is None:
        return data
    present = set(data['algorithm'].unique())
    missing = [a for a in algorithms if a not in present]
    if missing:
        print(f"  [warning] requested algorithms not found in data: {missing}")
    return data[data['algorithm'].isin(algorithms)].copy()


def plot_entropy(data, function, instance, dimension, output_dir=None, save=False, algorithms=None):
    data = _filter_algorithms(data, algorithms)
    plt.figure(figsize=(10, 5))

    algo_order = sorted(data['algorithm'].unique())
    palette = {a: ALGO_PALETTE[a] for a in algo_order}
    sns.lineplot(data=data, x='iteration', y='entropy', hue='algorithm',
                 hue_order=algo_order, palette=palette, errorbar=None)

    plt.title(f'Mean Normalized Population Entropy over Iterations in function F{function}_{instance}_D{dimension}')
    plt.xlabel('Iteration')
    plt.ylabel('Normalized Entropy')
    plt.ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7)
    plt.tight_layout()
    if save and output_dir is not None:
        plt.savefig(f'{output_dir}/Entropy_F{function}_I{instance}_D{dimension}.pdf', bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_entropy_per_function(all_entropy, output_dir, dimension, algorithms=None):
    """
    Optional: one plot per function_class (24 PDFs), each showing all
    algorithms' normalized entropy over iterations for that function
    (already averaged across instances/runs in all_entropy).
    Off by default - call explicitly if you want these;
    entropy_across_function_groups.pdf already gives a more compact
    grouped summary.
    """
    all_entropy = _filter_algorithms(all_entropy, algorithms)
    for f_class, sub in all_entropy.groupby('function_class'):
        data = sub.rename(columns={'mean_entropy': 'entropy'})
        plot_entropy(data, function=f_class, instance='all', dimension=dimension,
                     output_dir=output_dir, save=True)


def plot_entropy_by_function_group(all_entropy, output_dir, function_groups, algorithms=None):
    all_entropy = _filter_algorithms(all_entropy, algorithms)
    function_to_group = {f: group for group, functions in function_groups.items() for f in functions}

    all_entropy = all_entropy.copy()
    all_entropy['function_group'] = all_entropy['function_class'].map(function_to_group)

    group_entropy = (all_entropy.groupby(['algorithm', 'iteration', 'function_group'])['mean_entropy'].mean().reset_index())
    algo_order = sorted(group_entropy['algorithm'].unique())
    palette = {a: ALGO_PALETTE[a] for a in algo_order}

    g = sns.FacetGrid(group_entropy, col='function_group', col_wrap=3, height=4,
                       hue='algorithm', hue_order=algo_order, palette=palette)
    g.map_dataframe(sns.lineplot, x='iteration', y='mean_entropy')
    g.set(ylim=(0, 1))
    g.set_axis_labels('Iteration', 'Normalized Entropy')
    g.add_legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/entropy_across_function_groups.pdf', bbox_inches='tight')
    plt.close()


def plot_entropy_clustermap(all_entropy, output_dir, algorithms=None):
    all_entropy = _filter_algorithms(all_entropy, algorithms)
    algo_entropy_matrix = (all_entropy.groupby(['algorithm', 'iteration'])['mean_entropy'].mean().unstack(level='iteration'))

    n_algos = algo_entropy_matrix.shape[0]
    if n_algos < 2:
        raise ValueError(f"  [Error] clustermap needs >=2 algorithms, got {n_algos}; ")

    fig_height = max(6, n_algos * 0.35)
    g = sns.clustermap(algo_entropy_matrix, cmap='YlGnBu', figsize=(14, fig_height),
                        annot=False, standard_scale=None, col_cluster=False,
                        vmin=0, vmax=1)
    
    # Labels must be pulled in the DENDROGRAM-REORDERED order (clustermap
    # rearranges rows), otherwise every row gets mislabeled with the wrong
    # algorithm name.
    reordered_idx = g.dendrogram_row.reordered_ind
    g.ax_heatmap.set_yticks([i + 0.5 for i in range(n_algos)])
    g.ax_heatmap.set_yticklabels(
        [algo_entropy_matrix.index[i] for i in reordered_idx],
        fontsize=7, rotation=0
    )

    g.ax_heatmap.set_xlabel('Iteration')
    g.ax_heatmap.set_ylabel('Algorithm')
    g.cax.set_ylabel('Normalized Entropy (H / ln k)')
    plt.savefig(f'{output_dir}/clustermap.pdf', bbox_inches='tight')
    plt.close()


def plot_entropy_overlay(all_entropy_by_dim, functions_of_interest, dims, output_dir, save=False, algorithms=None):
    rows = []
    for dim, df in all_entropy_by_dim.items():
        df = _filter_algorithms(df, algorithms)
        sub = df[df['function_class'].isin(functions_of_interest)].copy()
        sub['dim'] = dim
        rows.append(sub)
    combined = pd.concat(rows, ignore_index=True)
    combined['function_class'] = combined['function_class'].astype(str)

    g = sns.FacetGrid(combined, row='dim', col='algorithm', height=3, aspect=1.2,
                       hue='function_class', palette='tab10')
    g.map_dataframe(sns.lineplot, x='iteration', y='mean_entropy')
    g.set(ylim=(0, 1))
    g.set_axis_labels('Iteration', 'Normalized Entropy')
    g.add_legend()

    if save:
        plt.savefig(f'{output_dir}/entropy_separability_overlay.pdf', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    PLOT_ALGORITHMS = None

    for d in DIMENSIONS:
        entropy_csv = f'{ENTROPY_DATA_DIR}/entropy_dim_{d}.csv'
        if not os.path.isfile(entropy_csv):
            print(f"Missing {entropy_csv} - run entropy_calculation.py first.")
            continue

        output_dir = f'{OUTPUT_DIR}/dim_{d}'
        os.makedirs(output_dir, exist_ok=True)

        all_entropy = pd.read_csv(entropy_csv)
        plot_entropy_by_function_group(all_entropy, output_dir, FUNCTION_GROUPS, algorithms=PLOT_ALGORITHMS)
        plot_entropy_clustermap(all_entropy, output_dir, algorithms=PLOT_ALGORITHMS)
