# Pipeline step "entropy_calc": computes normalized Shannon entropy of
# cluster occupancy per algorithm/run/iteration from the cluster-distribution
# CSVs, and writes both a granular and an aggregated table to data/entropy/.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import math
import os
import config
from config import (
    ALGORITHMS_OF_INTEREST, DIMENSIONS,
    CLUSTER_DISTRIBUTIONS_LATEST as INPUT_DIR, ENTROPY_DATA_DIR,
)
from pipeline_api import Progress, StageResult

STAGE = "entropy_calc"

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


def get_granular_entropy(input_dir, dimension,
                         algorithms_of_interest=None, progress=None):
    """
    Collects FULLY LOSSLESS per-iteration entropy across every
    (function, instance) file actually present in `input_dir` for one
    dimension - nothing averaged away. One row per (algorithm, problem_id,
    instance_id, dimension, run, iteration, entropy).

    Discovers which F{f}_I{i}.csv files exist rather than assuming
    config.FUNCTIONS x config.INSTANCES (see config.discover_problems): a
    GUI run can cover any subset of BBOB's functions/instances, and
    compute_entropy's pd.read_csv has no existence check, so the full cross
    product crashed on the first (function, instance) pair narrower than
    that.
    """
    if algorithms_of_interest is None:
        algorithms_of_interest = ALGORITHMS_OF_INTEREST

    all_rows = []
    for F, I in config.discover_problems(input_dir):
        if progress is not None and not progress.item(f"F{F}_I{I}"):
            break
        problem_entropy = compute_entropy(f'{input_dir}/F{F}_I{I}.csv', algorithms_of_interest)
        if problem_entropy.empty:
            continue
        problem_entropy['problem_id'] = F
        problem_entropy['instance_id'] = I
        problem_entropy['dimension'] = dimension
        all_rows.append(problem_entropy)

    # Explicit columns even when empty: a totally columnless pd.DataFrame()
    # writes a CSV pandas itself can't read back later (EmptyDataError:
    # "No columns to parse from file") - the same failure mode fixed in the
    # pairwise metric writers, reached here if a dimension's cluster
    # distributions yield zero entropy rows (e.g. nothing discovered, or
    # every count summed to zero).
    granular_columns = ['algorithm', 'problem_id', 'instance_id',
                         'dimension', 'run', 'iteration', 'entropy']
    if not all_rows:
        return pd.DataFrame(columns=granular_columns)

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


def run(progress_cb=None, cancel_event=None, dimensions=None):
    """Compute the granular and aggregated entropy tables per dimension."""
    dims = list(dimensions if dimensions is not None else DIMENSIONS)
    entropy_dir = config.ENTROPY_DATA_DIR
    progress = Progress(progress_cb, cancel_event)
    result = StageResult(STAGE)

    os.makedirs(entropy_dir, exist_ok=True)

    for d in dims:
        input_dir = f'{config.CLUSTER_DISTRIBUTIONS_LATEST}/dim_{d}'
        print(f"Computing entropy for dim={d}...")
        problems = config.discover_problems(input_dir)
        progress.begin(len(problems), f"entropy dim {d}")
        # Fully lossless per-iteration table (one row per algorithm/problem/
        # instance/dimension/run/iteration). Base table for any later
        # aggregation or pairwise analysis.
        granular = get_granular_entropy(input_dir, d,
                                        ALGORITHMS_OF_INTEREST, progress=progress)
        if progress.cancelled:
            result.cancelled = True
            return result
        granular_path = f'{entropy_dir}/entropy_granular_dim_{d}.csv'
        granular.to_csv(granular_path, index=False)
        result.written += 1
        print(f"  saved granular -> {granular_path}  ({len(granular)} rows)")
        # Aggregated table (averaged across runs then instances) - what the
        # current plots consume. Derived from the granular table in memory
        # rather than re-scanning every raw file a second time.
        all_entropy = aggregate_from_granular(granular)
        out_path = f'{entropy_dir}/entropy_dim_{d}.csv'
        all_entropy.to_csv(out_path, index=False)
        result.written += 1
        print(f"  saved aggregated -> {out_path}  ({len(all_entropy)} rows)")

    return result


def main():
    run()


if __name__ == '__main__':
    main()
