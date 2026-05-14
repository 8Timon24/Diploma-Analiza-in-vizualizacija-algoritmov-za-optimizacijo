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
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.datasets import make_blobs
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
import math
import argparse


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Algorithms that have been flagged for removal (loaded from utils)
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

# Dimensions to process
DIMENSIONS = [2]
parser = argparse.ArgumentParser(prog='Clustering on meta-heuristic algorithm\'s trajectories', usage='%(prog)s [options]')
parser.add_argument('-c', choices=['kmeans', 'dbscan'], help="Choose the clustering method")
args = parser.parse_args()
if args.c == 'kmeans':
    DATA_DIR = f'data/clustering_features_x_only_10_algorithms_kmeans_2pow_no_init/'
else:
    DATA_DIR = f'data/clustering_features_10_algorithms_dbscan/'
# ---------------------------------------------------------------------------
# Cluster count selection
# ---------------------------------------------------------------------------

def determine_number_of_clusters(X):
    """
    Use the elbow method to determine the optimal number of clusters for KMeans.

    Candidate cluster counts are powers of 2 from 2^2=4 up to 2^9=512,
    giving a broad, geometrically-spaced search space.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The feature matrix to cluster.

    Returns
    -------
    int or None
        The elbow value (optimal k) detected by KElbowVisualizer,
        or None if no clear elbow is found.
    """
    model = KMeans()

    # Candidate k values: [4, 8, 16, 32, 64, 128, 256, 512]
    cluster_options = [int(math.pow(2, x)) for x in range(2, 10)]

    visualizer = KElbowVisualizer(model, k=cluster_options)
    visualizer.fit(X)
    visualizer.show()

    return visualizer.elbow_value_


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def fit_kmeans(X):
    """
    Fit a KMeans model using the automatically determined number of clusters.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The feature matrix to cluster.

    Returns
    -------
    clusters : ndarray of shape (n_samples,)
        Integer cluster label for each sample.
    cluster_centers : ndarray of shape (n_clusters, n_features)
        Coordinates of each cluster centroid.
    """
    number_of_clusters = determine_number_of_clusters(X)

    model = KMeans(number_of_clusters)
    model.fit(X)
    clusters = model.predict(X)

    return clusters, model.cluster_centers_


# ---------------------------------------------------------------------------
# Output directory setup
# ---------------------------------------------------------------------------

def create_output_directories(data_dir, dimension):
    """
    Create the required output sub-directories for a given dimension if they
    do not already exist.

    Directories created:
      - <data_dir>/cluster_centers/dim_<dimension>/
      - <data_dir>/cluster_distributions/dim_<dimension>/
      - <data_dir>/clustering_results/dim_<dimension>/

    Parameters
    ----------
    data_dir : str
        Root output directory.
    dimension : int
        Problem dimensionality (e.g. 2, 5, 10).
    """
    os.makedirs(f'{data_dir}/cluster_centers/dim_{dimension}', exist_ok=True)
    os.makedirs(f'{data_dir}/cluster_distributions/dim_{dimension}', exist_ok=True)
    os.makedirs(f'{data_dir}/clustering_results/dim_{dimension}', exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading and filtering
# ---------------------------------------------------------------------------

def load_and_filter_data(filepath, dimension, all_removed_algorithms, algorithms_of_interest):
    """
    Load a compressed CSV for one problem file and apply all row-level filters.

    Filters applied (in order):
      1. Keep only rows where evaluations <= 500 * dimension  (budget cap).
      2. Drop algorithms that are in the global removal list.
      3. Keep only algorithms that are in the interest list.
      4. Drop rows from iteration 0 (initialisation / pre-search rows).

    Parameters
    ----------
    filepath : str
        Full path to the compressed CSV file.
    dimension : int
        Problem dimensionality; used to compute the evaluation budget cap.
    all_removed_algorithms : pd.Index or list
        Algorithms that must be excluded from the analysis.
    algorithms_of_interest : list of str
        Algorithms to retain.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe ready for rescaling and clustering.
    """
    d = pd.read_csv(filepath, compression='zip', index_col=0)

    # Apply evaluation budget cap
    d = d.query('evaluations <= 500 * @dimension')

    # Remove globally excluded algorithms and keep only the target subset
    d = d.query(
        'algorithm not in @all_removed_algorithms.index'
        ' and algorithm in @algorithms_of_interest'
    )

    # Drop initialisation rows (iteration 0)
    d = d.query('iteration > 0')

    return d


# ---------------------------------------------------------------------------
# Feature column helpers
# ---------------------------------------------------------------------------

def get_column_names(dimension):
    """
    Build the standard column-name lists used throughout the pipeline.

    Parameters
    ----------
    dimension : int
        Problem dimensionality.

    Returns
    -------
    x_columns : list of str
        Raw feature column names, e.g. ['x0', 'x1', ...].
    x_y_columns : list of str
        Raw feature columns plus the raw objective column.
    x_columns_scaled : list of str
        Scaled feature column names, e.g. ['scaled_x0', 'scaled_x1', ...].
    x_y_columns_scaled : list of str
        Scaled feature columns plus the scaled objective column.
    """
    x_columns = [f'x{i}' for i in range(0, dimension)]
    x_y_columns = x_columns + ['raw_y']
    x_columns_scaled = [f'scaled_x{i}' for i in range(0, dimension)]
    x_y_columns_scaled = x_columns_scaled + ['scaled_raw_y']

    return x_columns, x_y_columns, x_columns_scaled, x_y_columns_scaled


# ---------------------------------------------------------------------------
# Cluster distribution computation
# ---------------------------------------------------------------------------

def compute_cluster_distribution(d):
    """
    Count how many evaluations each (algorithm, run, iteration) tuple
    contributed to each cluster, producing a wide-format table.

    The result has one row per (algorithm, run, iteration) combination and
    one column per cluster label; missing combinations are filled with 0.

    Parameters
    ----------
    d : pd.DataFrame
        Dataframe that already contains a 'cluster' column.

    Returns
    -------
    pd.DataFrame
        Wide-format cluster distribution table.
    """
    cluster_distribution = (
        d.groupby(['algorithm', 'run', 'iteration', 'cluster'])
         .count()['evaluations']
         .unstack(level=-1)
         .fillna(0)
    )
    return cluster_distribution


# ---------------------------------------------------------------------------
# Result saving
# ---------------------------------------------------------------------------

def save_results(data_dir, dimension, filename, cluster_centers, d, cluster_distribution, x_columns_scaled, kmeans=False):
    """
    Persist all three output artefacts for a single problem file.

    Files written:
      - cluster_centers CSV  : centroid coordinates (scaled column names).
      - clustering_results   : full dataframe with cluster labels (parquet/gzip).
      - cluster_distributions: wide-format evaluation counts per cluster (CSV).

    Parameters
    ----------
    data_dir : str
        Root output directory.
    dimension : int
        Problem dimensionality.
    filename : str
        Original filename (e.g. 'problem_f1.csv'); used to derive output names.
    cluster_centers : ndarray of shape (n_clusters, n_features)
        Cluster centroid coordinates.
    d : pd.DataFrame
        Full dataframe including the 'cluster' column.
    cluster_distribution : pd.DataFrame
        Wide-format cluster distribution table.
    x_columns_scaled : list of str
        Column names to assign to the cluster-centers CSV.
    """
    # Save cluster centroids
    if kmeans:
        pd.DataFrame(cluster_centers, columns=x_columns_scaled).to_csv(
            f'{data_dir}/cluster_centers/dim_{dimension}/{filename}'
        )
    else:
        cluster_centers.to_csv(f'{data_dir}/cluster_centers/dim_{dimension}/{filename}')
    
    # Save full clustering results as compressed parquet
    d.to_parquet(
        f'{data_dir}/clustering_results/dim_{dimension}/{filename.replace(".csv", ".parquet")}',
        compression='gzip'
    )

    # Save per-(algorithm, run, iteration) cluster distribution
    cluster_distribution.to_csv(
        f'{data_dir}/cluster_distributions/dim_{dimension}/{filename}'
    )


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def process_file(filepath, filename, dimension, data_dir,
                 all_removed_algorithms, algorithms_of_interest, kmeans=False):
    """
    Run the full clustering pipeline for a single problem file.

    Steps:
      1. Load and filter the data.
      2. Rescale features and objective.
      3. Fit KMeans and assign cluster labels.
      4. Compute cluster distributions.
      5. Save all output artefacts.

    Parameters
    ----------
    filepath : str
        Full path to the input compressed CSV.
    filename : str
        Bare filename (used for output naming).
    dimension : int
        Problem dimensionality.
    data_dir : str
        Root output directory.
    all_removed_algorithms : pd.Index or list
        Algorithms excluded globally.
    algorithms_of_interest : list of str
        Algorithms to retain.
    """
    print(filename)

    # --- Load & filter ---
    x_columns, x_y_columns, x_columns_scaled, x_y_columns_scaled = get_column_names(dimension)
    d = load_and_filter_data(filepath, dimension, all_removed_algorithms, algorithms_of_interest)
    print(d['algorithm'].unique())

    # --- Rescale features + objective to [0, 1] ---
    d = rescale(d, x_y_columns)

    # --- Cluster on raw (unscaled) x columns ---
    X = d[x_columns]
    print(X.shape)

    if kmeans:
        cluster_labels, cluster_centers = fit_kmeans(X)
        d['cluster'] = cluster_labels
        cluster_distribution = compute_cluster_distribution(d)
    else:
        cluster_labels, noise_label = fit_dbscan(X, eps=0.1, min_samples=100)
        d['cluster'] = cluster_labels  
        d_no_noise = d[d['cluster'] != noise_label] if noise_label is not None else d
        cluster_centers = d_no_noise.groupby('cluster')[x_columns].mean()
        cluster_distribution = compute_cluster_distribution(d_no_noise)

    # --- Persist results ---
    save_results(data_dir, dimension, filename, cluster_centers, d, cluster_distribution, x_columns_scaled)

def find_eps(X, min_samples, show_plot=True):
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors.fit(X)
    distances, _ = neighbors.kneighbors(X)
    
    # Distance to the k-th nearest neighbour, sorted ascending
    distances = np.sort(distances[:, -1])
    
    if show_plot:
        plt.figure(figsize=(8, 4))
        plt.plot(distances)
        plt.xlabel('Points sorted by distance')
        plt.ylabel(f'{min_samples}-NN distance')
        plt.title('K-distance graph — look for the elbow')
        plt.tight_layout()
        plt.savefig("kdistance_plot.png")
        plt.close()
    
    return distances


def fit_dbscan(X, eps=0.05, min_samples=50):
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.labels_
    unique_labels = set(labels)
    num_of_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    noise = np.sum(labels==-1)
    print(f"  DBSCAN found {num_of_clusters} clusters, {noise} noise points ({100*noise/len(labels):.1f}%)")
    noise_label=None
    if -1 in labels:
        noise_label = labels.max() + 1
        labels[labels == -1] = noise_label
    
    return labels, noise_label

# ---------------------------------------------------------------------------
# Per-dimension processing
# ---------------------------------------------------------------------------

def process_dimension(dimension, data_dir, all_removed_algorithms, algorithms_of_interest):
    """
    Process all problem files for a given dimensionality.

    Creates output directories, then iterates over every file in the
    processed data folder for this dimension, calling process_file for each.

    Parameters
    ----------
    dimension : int
        Problem dimensionality (e.g. 2, 5, 10).
    data_dir : str
        Root output directory.
    all_removed_algorithms : pd.Index or list
        Algorithms excluded globally.
    algorithms_of_interest : list of str
        Algorithms to retain.
    """
    create_output_directories(data_dir, dimension)

    input_dir = f'data/processed/dim_{dimension}'
    for filename in tqdm(os.listdir(input_dir)):
        filepath = f'{input_dir}/{filename}'
        process_file(
            filepath, filename, dimension, data_dir,
            all_removed_algorithms, algorithms_of_interest
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """
    Main entry point: iterate over all configured dimensions and process each.
    """
    print(DATA_DIR)
    for dimension in DIMENSIONS:
        process_dimension(
            dimension, DATA_DIR,
            ALL_REMOVED_ALGORITHMS, ALGORITHMS_OF_INTEREST
        )


if __name__ == '__main__':
    main()
