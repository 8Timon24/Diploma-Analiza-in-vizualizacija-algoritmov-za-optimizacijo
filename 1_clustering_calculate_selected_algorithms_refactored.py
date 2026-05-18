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
from kneed import KneeLocator
import itertools

# Algorithms flagged for removal (loaded from utils)
ALL_REMOVED_ALGORITHMS = get_removed_algorithms()
# The specific subset of algorithms we want to cluster
ALGORITHMS_OF_INTEREST = ["ModifiedAEO","OriginalAEO","AugmentedAEO",
                          "OriginalSHADE","SADE","JADE","OriginalDE",
                          "OriginalWOA","HI_WOA",]

# Dimensions to process
DIMENSIONS = [2]
#Arguements to distinguish the type of clustering we want to do
parser = argparse.ArgumentParser(prog='Clustering on meta-heuristic algorithm\'s trajectories', usage='%(prog)s [options]')
parser.add_argument('-c', choices=['kmeans', 'dbscan', 'dbscan_adaptive'], help="Choose the clustering method")
args = parser.parse_args()

if args.c == 'kmeans':
    DATA_DIR = f'data/clustering_features_x_only_10_algorithms_kmeans_2pow_no_init/'
elif args.c == 'dbscan':
    DATA_DIR = f'data/clustering_features_10_algorithms_dbscan/'
elif args.c == 'dbscan_adaptive':
    DATA_DIR = f'data/clustering_features_10_algorithms_dbscan/adaptive/'

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



def fit_kmeans(X):
    """
    Fit a KMeans model using the determined number of clusters.

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

    budget = 500*dimension
    d = d.query('evaluations <= @budget')

    d = d.query(
        'algorithm not in @all_removed_algorithms.index and algorithm in @algorithms_of_interest'
    )

    d = d.query('iteration > 0')

    return d


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
        ['x0', 'x1', ...].
    x_y_columns : list of str
        ['x0', 'x1', ..., 'xd', 'raw_y'] where d=dimension
    x_columns_scaled : list of str
        Scaled feature column names: ['scaled_x0', 'scaled_x1', ...].
    x_y_columns_scaled : list of str
        Scaled feature columns plus the scaled objective column:
        ['scaled_x0', 'scaled_x1', ..., 'scaled_xd', 'scaled_raw_y']
    """
    x_columns = [f'x{i}' for i in range(0, dimension)]
    x_y_columns = x_columns + ['raw_y']
    x_columns_scaled = [f'scaled_x{i}' for i in range(0, dimension)]
    x_y_columns_scaled = x_columns_scaled + ['scaled_raw_y']

    return x_columns, x_y_columns, x_columns_scaled, x_y_columns_scaled


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



def save_results(data_dir, dimension, filename, cluster_centers, d, cluster_distribution, x_columns_scaled, kmeans=False, dbscan_data=None):
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
    eps = dbscan_data['eps']
    ms = dbscan_data['min_samples']
    eps_method = dbscan_data['eps_method']

    if kmeans:
        pd.DataFrame(cluster_centers, columns=x_columns_scaled).to_csv(f'{data_dir}/cluster_centers/dim_{dimension}/{filename}')
        cluster_distribution.to_csv(f'{data_dir}/cluster_distributions/dim_{dimension}/{filename}')
        d.to_parquet(f'{data_dir}/clustering_results/dim_{dimension}/{filename.replace(".csv", ".parquet")}', compression='gzip')
    else:
        print(type(eps), eps)
        print(type(ms), ms)
        os.makedirs(f'{data_dir}/cluster_centers/dim_{dimension}/eps{eps}_ms{ms}/', exist_ok=True)
        cluster_centers.to_csv(f'{data_dir}/cluster_centers/dim_{dimension}/eps{eps}_ms{ms}/{filename}')
        os.makedirs(f'{data_dir}/clustering_results/dim_{dimension}/eps{eps}_ms{ms}', exist_ok=True)
        d.to_parquet(f'{data_dir}/clustering_results/dim_{dimension}/eps{eps}_ms{ms}/{filename.replace(".csv", ".parquet")}',compression='gzip')
        # Save per-(algorithm, run, iteration) cluster distribution
        os.makedirs(f'{data_dir}/cluster_distributions/dim_{dimension}/eps{eps}_ms{ms}', exist_ok=True)
        cluster_distribution.to_csv(f'{data_dir}/cluster_distributions/dim_{dimension}/eps{eps}_ms{ms}/{filename}')
    
    """
    if dbscan_data is not None:
        log_path = f'{data_dir}/dbscan_params/dim_{dimension}_params.csv'
        os.makedirs(f'{data_dir}/dbscan_params', exist_ok=True)
        
        log_row = pd.DataFrame([{'problem': filename.replace('.csv', ''), **dbscan_data}])
        
        # Append to existing log or create new one
        if os.path.isfile(log_path):
            existing = pd.read_csv(log_path)
            pd.concat([existing, log_row], ignore_index=True).to_csv(log_path, index=False)
        else:
            log_row.to_csv(log_path, index=False)
    """


def process_file(filepath, filename, dimension, data_dir,
                 all_removed_algorithms, algorithms_of_interest, kmeans=False, adaptive=False, eps=0.1, ms=100):
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

    x_columns, x_y_columns, x_columns_scaled, x_y_columns_scaled = get_column_names(dimension)
    d = load_and_filter_data(filepath, dimension, all_removed_algorithms, algorithms_of_interest)
    print(d['algorithm'].unique())

    d = rescale(d, x_y_columns)

    X = d[x_columns]
    print(X.shape)

    if kmeans:
        cluster_labels, cluster_centers = fit_kmeans(X)
        d['cluster'] = cluster_labels
        cluster_distribution = compute_cluster_distribution(d)
        save_results(data_dir, dimension, filename, cluster_centers, d, cluster_distribution, x_columns_scaled)
        return
    elif not adaptive: #dbscan
        cluster_labels, noise_label = fit_dbscan(X, eps=eps, min_samples=ms)
        d['cluster'] = cluster_labels  
        d_no_noise = d[d['cluster'] != noise_label] if noise_label is not None else d
        cluster_centers = d_no_noise.groupby('cluster')[x_columns].mean()
        cluster_distribution = compute_cluster_distribution(d_no_noise)
        data = {'eps': eps, 'eps_method': 'manual', 'min_samples': ms}
        save_results(data_dir, dimension, filename, cluster_centers, d, cluster_distribution, x_columns_scaled, dbscan_data=data)
        return
    """
    else:
        cluster_labels, noise_label, data = fit_dbscan_adaptive(X)
        d['cluster'] = cluster_labels  
        d_no_noise = d[d['cluster'] != noise_label] if noise_label is not None else d
        cluster_centers = d_no_noise.groupby('cluster')[x_columns].mean()
        cluster_distribution = compute_cluster_distribution(d_no_noise)
        save_results(data_dir, dimension, filename, cluster_centers, d, cluster_distribution, x_columns_scaled, dbscan_data=data)
    """
    
def k_distance_graph(X, min_samples, show_plot=True):
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors.fit(X)
    distances, _ = neighbors.kneighbors(X)
    
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

def eps_range(X, min_samples):
    """
    Returns an approximate range out of which to pick the eps for dbscan.
    
    Parameters
    ---------
    X -> the clustering data
    min_samples -> parameter for dbscan

    """
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors.fit(X)
    distances, _ = neighbors.kneighbors(X)
    mean_dist = distances[:, -1].mean()
    std_dist = distances[:, -1].std()
    left = mean_dist-std_dist
    right = mean_dist+std_dist
    mid = mean_dist
    return [left, mid, right]

def eps_from_k_distance(X, min_samples):
    """
    Returns an estimated optimal eps by calculating it from the k-distance graph

    Parameters
    ---------
    X -> the clustering data
    min_samples -> parameter for dbscan
    """
    dist = k_distance_graph(X, min_samples, show_plot=False)
    kneedle = KneeLocator(range(len(dist)), dist, S=1.0, curve='convex', direction='increasing')
    eps = dist[kneedle.knee]
    return eps

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

"""
def fit_dbscan_adaptive(X, min_samples=100):

    Doesnt work well, dont use it. 

    X_np = X.values if hasattr(X, 'values') else np.array(X)
    sample_size = min(5000, len(X_np))
    idx = np.random.choice(len(X_np), sample_size, replace=False)
    X_sample = X_np[idx]

    nn = NearestNeighbors(n_neighbors=min_samples, algorithm='ball_tree')
    nn.fit(X_sample)
    distances, _ = nn.kneighbors(X_sample)
    distances_sorted = np.sort(distances[:, -1])

    #Doesnt work as expected, knee is way too high on problems where algorithms converge quickly
    kneedle = KneeLocator(
        range(len(distances_sorted)),
        distances_sorted,
        S=1.0,
        curve='convex',
        direction='increasing'
    )

    if kneedle.knee is None:
        eps = distances_sorted.mean()
        eps_method = 'mean_fallback'
    else:
        eps = distances_sorted[kneedle.knee]
        eps_method = 'knee'

    print(f"  Adaptive eps ({eps_method}): {eps:.4f}")

    # --- Run DBSCAN on full data ---
    model = DBSCAN(eps=eps, min_samples=min_samples, algorithm='ball_tree', n_jobs=-1)
    labels = model.fit_predict(X_np)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    print(f"  Found {n_clusters} clusters, {n_noise} noise points ({100*n_noise/len(X_np):.1f}%)")

    # --- Relabel noise to max_cluster + 1 ---
    noise_label = None
    if -1 in labels:
        noise_label = int(labels.max()) + 1
        labels[labels == -1] = noise_label

    data = {
        'eps': eps,
        'eps_method': eps_method,
        'min_samples': min_samples
    }

    return labels, noise_label, data
"""

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
        if args.c == 'dbscan':
            for epsilon, ms in itertools.product([0.1], [100, 150]):
                process_file(
                    filepath, filename, dimension, data_dir,
                    all_removed_algorithms, algorithms_of_interest, kmeans=False, adaptive =  args.c == 'dbscan_adaptive', eps = epsilon, ms = ms
                )


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
