# Pipeline step "return_rate_calc": detects cluster "revisit" events per
# algorithm/run (an already-visited cluster becoming active again) and writes
# the revisiting history + per-algorithm revisit counts to data/return_rate/.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from tqdm import tqdm
import itertools
import os
import numpy as np
from config import (
    ALGORITHMS_OF_INTEREST, DIMENSIONS, FUNCTIONS, INSTANCES,
    CLUSTER_DISTRIBUTIONS_LATEST as INPUT_DIR, RETURN_RATE_DATA_DIR as RR_DATA_DIR,
)


def revisiting_history(d, threshold=3):
    """
    For one (algorithm, run)'s cluster-occupancy table (rows = iterations,
    columns = clusters, values = agent counts), logs every time an
    already-visited cluster becomes 'active' (> threshold agents) again in
    a later iteration. Returns one row per revisit event.
    """
    visit_history = {}
    returns = []
    for iteration, row in d.iterrows():
        active_clusters = set(row[row > threshold].index)

        for cluster in active_clusters:
            if cluster in visit_history:
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


def shared_revisit_rate(history, alg1, alg2, by='problem', weighted=False):
    """
    Jaccard similarity of the two algorithms' visited-cluster sets, computed
    PER RUN and then averaged across runs.
    """
    results = []
    for data in history[by].unique():
        if by == 'problem_class':
            prob_data = history.query('problem_class == @data')
        else:
            prob_data = history.query('problem == @data')

        run_similarities = []
        for run in prob_data['run'].unique():
            run_data = prob_data[prob_data['run'] == run]
            clusters_alg1 = set(run_data.query('algorithm == @alg1')['cluster'])
            clusters_alg2 = set(run_data.query('algorithm == @alg2')['cluster'])

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
    """
    Scans every cluster-distribution file in input_dir and builds the full
    long-format revisiting-event table across all problems/instances,
    tagged with algorithm, run, problem, problem_class, instance.
    This is the expensive step (one pass over all raw cluster files).
    """
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


def pairwise_revisit_matrix(algorithms_of_interest, all_problems_history,
                            specified_problem=None, by='problem', weighted=False):
    """
    Builds the symmetric algorithm x algorithm similarity matrix from an
    already-computed revisiting history. 
    """
    pairs = list(itertools.combinations(algorithms_of_interest, 2))
    pairwise = []

    for alg1, alg2 in pairs:
        if specified_problem is not None:
            problem_history = all_problems_history.query('problem == @specified_problem')
            result = shared_revisit_rate(problem_history, alg1, alg2, by='problem', weighted=weighted)
        else:
            result = shared_revisit_rate(all_problems_history, alg1, alg2, by=by, weighted=weighted)

        if not result.empty:
            pairwise.append({
                'algorithm': alg1,
                'algorithm2': alg2,
                'mean_similarity': result['similarity'].mean(),
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

    return matrix_full


def get_per_algorithm_revisits(all_problem_history):
    return (all_problem_history.groupby(["algorithm"])
            .agg(total_revisits=('cluster', 'count'))
            .sort_values(by='total_revisits', ascending=False))


if __name__ == "__main__":
    # one specified single-problem heatmap to also cache, matching the original
    SPECIFIED_PROBLEM = "F17_I1"

    for d in DIMENSIONS:
        input_dir = f'{INPUT_DIR}/dim_{d}'
        out_dir = f'{RR_DATA_DIR}/dim_{d}'
        os.makedirs(out_dir, exist_ok=True)

        print(f"\nComputing revisiting history for dim={d}...")
        aph = get_all_problems_revisiting_history(input_dir, ALGORITHMS_OF_INTEREST)
        if aph.empty:
            print(f"  no data found in {input_dir}, skipping.")
            continue

        # 1. granular revisiting history (base table)
        aph.to_csv(f'{out_dir}/revisiting_history.csv', index=False)
        print(f"  saved revisiting_history.csv ({len(aph)} rows)")
        
        
        # 2. per-algorithm total revisit counts (for the barplot)
        per_alg = get_per_algorithm_revisits(aph)
        per_alg.to_csv(f'{out_dir}/per_algorithm_revisits.csv')
        
        """
        # 3. derived pairwise matrices (what the plots consume)
        matrix_problem = pairwise_revisit_matrix(ALGORITHMS_OF_INTEREST, aph, specified_problem=SPECIFIED_PROBLEM)
        matrix_problem.to_csv(f'{out_dir}/matrix_{SPECIFIED_PROBLEM}.csv')

        matrix_pc = pairwise_revisit_matrix(ALGORITHMS_OF_INTEREST, aph, by='problem_class')
        matrix_pc.to_csv(f'{out_dir}/matrix_problem_class.csv')

        matrix_p = pairwise_revisit_matrix(ALGORITHMS_OF_INTEREST, aph)
        matrix_p.to_csv(f'{out_dir}/matrix_problem.csv')

        matrix_p_w = pairwise_revisit_matrix(ALGORITHMS_OF_INTEREST, aph, weighted=True)
        matrix_p_w.to_csv(f'{out_dir}/matrix_problem_weighted.csv')

        matrix_pc_w = pairwise_revisit_matrix(ALGORITHMS_OF_INTEREST, aph, by='problem_class', weighted=True)
        matrix_pc_w.to_csv(f'{out_dir}/matrix_problem_class_weighted.csv')

        print(f"  saved all matrices for dim={d}")
        """
    print("\nDONE")
