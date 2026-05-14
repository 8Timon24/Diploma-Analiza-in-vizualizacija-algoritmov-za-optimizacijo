import pandas as pd
import os
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.patches as mpatches
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
from utils import *
import argparse


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DIMENSIONS = [2]
parser = argparse.ArgumentParser(prog='Clustering on meta-heuristic algorithm\'s trajectories', usage='%(prog)s [options]')
parser.add_argument('-c', choices=['kmeans', 'dbscan'], help="Choose the clustering method")
args = parser.parse_args()
if args.c == 'kmeans':
    CLUSTER_FEATURES_DIR = f'data/clustering_features_x_only_10_algorithms_kmeans_2pow_no_init/'
else:
    CLUSTER_FEATURES_DIR = f'data/clustering_features_10_algorithms_dbscan/'

# Sub-directory pattern for cluster distributions per dimension
CLUSTER_DISTRIBUTIONS_SUBDIR = 'cluster_distributions'

# Output sub-directory for pairwise similarity results
SIMILARITY_OUTPUT_SUBDIR = 'algorithm_pairwise_similarity'

# Aggregation methods to compute pairwise similarity over
AGGREGATIONS = ['mean', 'median']


# ---------------------------------------------------------------------------
# Per-file similarity computation
# ---------------------------------------------------------------------------

def compute_trajectory_similarities(directory):
    """
    Load all cluster-distribution CSV files in a directory and compute
    pairwise cosine similarities between every (algorithm, run) pair.

    For each file:
      1. Read the cluster distribution (multi-index: algorithm, run, iteration).
      2. Unstack the iteration level to get one row per (algorithm, run).
      3. Compute the full cosine-similarity matrix between all rows.
      4. Melt the matrix into long format and tag with the problem name.

    Parameters
    ----------
    directory : str
        Path to the folder containing per-problem cluster-distribution CSVs.

    Returns
    -------
    pd.DataFrame
        Long-format dataframe with columns:
        algorithm, run, algorithm2, run2, variable (cluster), value (similarity),
        and problem (filename without .csv extension).
    """
    trajectory_similarities = []

    for file in tqdm(os.listdir(directory)):
        file_loc = f'{directory}/{file}'

        # Skip anything that is not a regular file (e.g. sub-directories)
        if not os.path.isfile(file_loc):
            continue
        
        # Load cluster distribution; index levels: algorithm, run, iteration
        df = pd.read_csv(file_loc, index_col=[0, 1, 2])

        # Unstack iteration → one row per (algorithm, run); drop incomplete rows
        d_algorithm_run = df.unstack(level=-1).dropna()

        # Cosine similarity matrix: rows and columns are (algorithm, run) tuples
        cs = pd.DataFrame(
            cosine_similarity(d_algorithm_run),
            index=d_algorithm_run.index,
            columns=d_algorithm_run.index
        )

        # Reshape from wide matrix to long format
        # Rename the melted index columns so they don't clash with the value columns
        c = (
            cs.reset_index()
            .rename(columns={'algorithm': 'algorithm2', 'run': 'run2'})
            .melt(
                id_vars=[('algorithm2', ''), ('run2', '')],
                value_vars=list(cs.columns)
            )
            .rename(columns={
                ('algorithm2', ''): 'algorithm2',
                ('run2', ''): 'run2'
            })
        )
        c['run'] = c['run'].astype(int)   # fix dtype
        c['run2'] = c['run2'].astype(int) # fix dtype

        # Tag each row with the problem name (strip .csv extension)
        trajectory_similarities.append(c.assign(problem=file.replace('.csv', '')))

    return pd.concat(trajectory_similarities)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def remove_incomplete_algorithms(trajectory_similarities):
    """
    Drop algorithms that are missing from more than a certain 
    percentage of problems, rather than requiring perfect coverage.
    """
    t = trajectory_similarities.groupby('algorithm').count()['run']
    t_max = t.max()
    
    # Allow up to 20% missing coverage instead of requiring perfect coverage
    threshold = t_max * 0.8
    algorithms_to_remove = list(t[t < threshold].index)
    
    print(f"Max count: {t_max}, threshold: {threshold}")
    print(f"Removing algorithms: {algorithms_to_remove}")
    
    if not algorithms_to_remove:
        return trajectory_similarities
    
    return trajectory_similarities.query(
        'algorithm not in @algorithms_to_remove'
        ' and algorithm2 not in @algorithms_to_remove'
    )

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def add_problem_metadata(trajectory_similarities):
    """
    Parse the problem name string into numeric problem_class and instance columns.

    Problem names are assumed to follow the pattern: 'F{class}_I{instance}_...'
    e.g. 'F3_I2_something' → problem_class=3, instance=2.

    Parameters
    ----------
    trajectory_similarities : pd.DataFrame
        Long-format similarity dataframe containing a 'problem' column.

    Returns
    -------
    pd.DataFrame
        Same dataframe with two new integer columns added:
        'problem_class' and 'instance'.
    """
    trajectory_similarities['problem_class'] = [
        int(tt.split('_')[0].replace('F', ''))
        for tt in trajectory_similarities['problem']
    ]
    trajectory_similarities['instance'] = [
        int(tt.split('_')[1].replace('I', ''))
        for tt in trajectory_similarities['problem']
    ]
    return trajectory_similarities


# ---------------------------------------------------------------------------
# Pairwise similarity aggregation and saving
# ---------------------------------------------------------------------------

def compute_and_save_pairwise_similarity(trajectory_similarities, optimizer_group,
                                          aggregation, dimension, output_dir):
    """
    Aggregate pairwise same-run similarities and save to CSV.

    Only rows where run == run2 are used (i.e. comparing the same run of two
    different algorithms, not cross-run comparisons). Self-comparisons
    (algorithm == algorithm2) are excluded from the output.

    Algorithm group labels (from utils.get_algorithm_groups()) are joined in
    so downstream analysis can colour/group by algorithm family.

    Parameters
    ----------
    trajectory_similarities : pd.DataFrame
        Filtered, metadata-enriched long-format similarity dataframe.
    optimizer_group : dict
        Mapping from algorithm name → group label.
    aggregation : str
        Either 'mean' or 'median' — how to aggregate similarity across problems.
    dimension : int
        Problem dimensionality; used only for the output filename.
    output_dir : str
        Directory where the output CSV will be written.
    """
    # Filter to same-run pairs and aggregate across problems
    same_run = trajectory_similarities.query('run == run2')

    if aggregation == 'mean':
        tt = same_run.groupby(['algorithm', 'algorithm2'])['value'].mean().dropna()
    else:
        tt = same_run.groupby(['algorithm', 'algorithm2'])['value'].median().dropna()

    # Reshape to long format, drop self-comparisons, and sort by similarity
    sorted_sim = (
        tt.to_frame()
          .reset_index()
          .query('algorithm != algorithm2')
          .sort_values(by='value', ascending=False)
    )

    # Annotate with algorithm family groups
    sorted_sim['algorithm_group'] = sorted_sim['algorithm'].apply(
        lambda x: optimizer_group[x]
    )
    sorted_sim['algorithm2_group'] = sorted_sim['algorithm2'].apply(
        lambda x: optimizer_group[x]
    )

    output_path = f'{output_dir}/algorithm_{aggregation}_similarity_{dimension}D.csv'
    sorted_sim.to_csv(output_path)


# ---------------------------------------------------------------------------
# Per-dimension processing
# ---------------------------------------------------------------------------

def process_dimension(dimension, cluster_features_dir,
                      cluster_distributions_subdir, similarity_output_subdir):
    """
    Run the full similarity pipeline for a single dimensionality.

    Steps:
      1. Compute per-file pairwise cosine similarities.
      2. Concatenate all files and filter incomplete algorithms.
      3. Parse problem metadata (class and instance) from filenames.
      4. For each aggregation method, compute and save pairwise similarities.

    Parameters
    ----------
    dimension : int
        Problem dimensionality (e.g. 2, 5, 10).
    cluster_features_dir : str
        Root data directory containing both input and output sub-folders.
    cluster_distributions_subdir : str
        Sub-directory name for cluster distribution input files.
    similarity_output_subdir : str
        Sub-directory name for similarity output files.
    """
    print("Dimension ", dimension)

    input_dir = f'{cluster_features_dir}/{cluster_distributions_subdir}/dim_{dimension}'
    output_dir = f'{cluster_features_dir}/{similarity_output_subdir}'
    os.makedirs(output_dir, exist_ok=True)

    trajectory_similarities = compute_trajectory_similarities(input_dir)
    #print("After compute:", trajectory_similarities.shape)
    #print(trajectory_similarities.head())
    #print("run dtype:", trajectory_similarities['run'].dtype)
    #print("run2 dtype:", trajectory_similarities['run2'].dtype)
    #print("run==run2 matches:", trajectory_similarities.query('run==run2').shape[0])

    trajectory_similarities = remove_incomplete_algorithms(trajectory_similarities)
    #print("After remove_incomplete:", trajectory_similarities.shape)

    trajectory_similarities = add_problem_metadata(trajectory_similarities)
    #print("After metadata:", trajectory_similarities.shape)

    # --- Load algorithm family groupings from utils ---
    optimizer_group = get_algorithm_groups()

    # --- Aggregate and save for each aggregation method ---
    for aggregation in AGGREGATIONS:
        compute_and_save_pairwise_similarity(
            trajectory_similarities, optimizer_group,
            aggregation, dimension, output_dir
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """
    Main entry point: iterate over all configured dimensions and process each.
    """
    for dimension in DIMENSIONS:
        process_dimension(
            dimension,
            CLUSTER_FEATURES_DIR,
            CLUSTER_DISTRIBUTIONS_SUBDIR,
            SIMILARITY_OUTPUT_SUBDIR
        )


if __name__ == '__main__':
    main()
