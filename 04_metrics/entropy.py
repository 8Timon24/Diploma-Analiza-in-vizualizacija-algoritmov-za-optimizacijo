# Pipeline step "entropy_calc": computes normalized Shannon entropy of
# cluster occupancy per algorithm/run/iteration from the cluster-distribution
# CSVs, and writes both a granular and an aggregated table to data/entropy/.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import math
import os
from config import (
    ALGORITHMS_OF_INTEREST, DIMENSIONS, FUNCTIONS, INSTANCES,
    CLUSTER_DISTRIBUTIONS_LATEST as INPUT_DIR, ENTROPY_DATA_DIR,
)

def compute_entropy(filepath, algorithms_of_interest=None):
    """
    Computes normalized Shannon entropy (H / ln k, in [0, 1]) of cluster
    occupancy for every (algorithm, run, iteration) in one
    (function, instance) cluster-distribution file.
    """
    df = pd.read_csv(filepath, index_col=[0, 1, 2])
    k = df.shape[1]
    max_entropy = math.log(k) if k > 1 else 1
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
            entropy_i /= max_entropy
            rows.append({
                'algorithm': alg,
                'run': run,
                'iteration': iteration[2],
                'entropy': entropy_i
            })
    return pd.DataFrame(rows)


def get_granular_entropy(input_dir, functions, instances, dimension, algorithms_of_interest=None):
    """
    Collects FULLY LOSSLESS per-iteration entropy across all
    (function, instance) files for one dimension - nothing averaged away.
    One row per (algorithm, problem_id, instance_id, dimension, run,
    iteration, entropy).
    """
    if algorithms_of_interest is None:
        algorithms_of_interest = ALGORITHMS_OF_INTEREST

    all_rows = []
    for F in functions:
        for I in instances:
            problem_entropy = compute_entropy(f'{input_dir}/F{F}_I{I}.csv', algorithms_of_interest)
            if problem_entropy.empty:
                continue
            problem_entropy['problem_id'] = F
            problem_entropy['instance_id'] = I
            problem_entropy['dimension'] = dimension
            all_rows.append(problem_entropy)

    if not all_rows:
        return pd.DataFrame()

    granular = pd.concat(all_rows, ignore_index=True)
    # reorder columns to the natural identifying-keys-then-value order
    granular = granular[['algorithm', 'problem_id', 'instance_id',
                          'dimension', 'run', 'iteration', 'entropy']]
    return granular


def get_all_entropy(input_dir, functions, instances, algorithms_of_interest=None):
    """
    Aggregates normalized entropy up to (algorithm, iteration,
    function_class) level, averaged first across runs (within
    compute_entropy's caller here) then across instances.
    """
    if algorithms_of_interest is None:
        algorithms_of_interest = ALGORITHMS_OF_INTEREST

    all_entropy = []
    for F in functions:
        problem_entropy_by_function_class = []
        for I in instances:
            problem_entropy = compute_entropy(f'{input_dir}/F{F}_I{I}.csv', algorithms_of_interest)
            mean_problem_entropy = (
                problem_entropy.groupby(['algorithm', 'iteration'])['entropy']
                .agg(mean_entropy='mean', std_entropy='std')
                .reset_index()
            )
            mean_problem_entropy['instance'] = I
            mean_problem_entropy['function_class'] = F
            problem_entropy_by_function_class.append(mean_problem_entropy)
        df = pd.concat(problem_entropy_by_function_class, ignore_index=True)
        df = df.groupby(['algorithm', 'iteration', 'function_class'])['mean_entropy'].mean().reset_index()
        all_entropy.append(df)

    all_entropy = pd.concat(all_entropy, ignore_index=True)
    return all_entropy


def aggregate_from_granular(granular):
    """
    Produces the same aggregated table as get_all_entropy, but from an
    already-computed granular frame (from get_granular_entropy) instead of
    re-scanning the raw files. Averages entropy across runs then across
    instances, down to (algorithm, iteration, function_class) with column
    'mean_entropy'. problem_id is renamed to function_class to match the
    schema the plotting code expects.
    """
    if granular.empty:
        return pd.DataFrame(columns=['algorithm', 'iteration', 'function_class', 'mean_entropy'])

    # mean across runs within each (algorithm, problem, instance, iteration)
    per_instance = (
        granular.groupby(['algorithm', 'problem_id', 'instance_id', 'iteration'])['entropy']
        .mean()
        .reset_index()
    )
    # then mean across instances -> (algorithm, problem, iteration)
    per_problem = (
        per_instance.groupby(['algorithm', 'problem_id', 'iteration'])['entropy']
        .mean()
        .reset_index()
        .rename(columns={'problem_id': 'function_class', 'entropy': 'mean_entropy'})
    )
    return per_problem[['algorithm', 'iteration', 'function_class', 'mean_entropy']]


if __name__ == '__main__':
    os.makedirs(ENTROPY_DATA_DIR, exist_ok=True)

    for d in DIMENSIONS:
        input_dir = f'{INPUT_DIR}/dim_{d}'
        print(f"Computing entropy for dim={d}...")
        # Fully lossless per-iteration table (one row per algorithm/problem/
        # instance/dimension/run/iteration). Base table for any later
        # aggregation or pairwise analysis.
        granular = get_granular_entropy(input_dir, FUNCTIONS, INSTANCES, d, ALGORITHMS_OF_INTEREST)
        granular_path = f'{ENTROPY_DATA_DIR}/entropy_granular_dim_{d}.csv'
        granular.to_csv(granular_path, index=False)
        print(f"  saved granular -> {granular_path}  ({len(granular)} rows)")
        # Aggregated table (averaged across runs then instances) - what the
        # current plots consume. Derived from the granular table in memory
        # rather than re-scanning every raw file a second time.
        all_entropy = aggregate_from_granular(granular)
        out_path = f'{ENTROPY_DATA_DIR}/entropy_dim_{d}.csv'
        all_entropy.to_csv(out_path, index=False)
        print(f"  saved aggregated -> {out_path}  ({len(all_entropy)} rows)")
