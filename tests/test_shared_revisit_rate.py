# Unit tests (plain assert, run directly) for shared_revisit_rate() in
# 04_metrics/return_rate.py: unweighted/weighted Jaccard, disjoint/empty
# edge cases, and identical algorithms.
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "04_metrics"))

import pandas as pd
import numpy as np
from return_rate import shared_revisit_rate


def make_history(events):
    """
    events: list of dicts with keys algorithm, run, cluster (+ we fill problem).
    One row per revisit EVENT (a cluster can appear multiple times for the same
    algorithm/run - that's what the weighted mode counts).
    """
    rows = []
    for e in events:
        rows.append({
            'algorithm': e['algorithm'],
            'run': e['run'],
            'cluster': e['cluster'],
            'problem': 'F1_I1',
            'problem_class': 'F1',
            'instance': 'I1',
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Scenario: two algorithms A and B, ONE problem (F1_I1), TWO runs.
# ---------------------------------------------------------------------------
# RUN 1 revisit events:
#   A revisits: c0 (x2), c1 (x1), c2 (x1)   -> counts {c0:2, c1:1, c2:1}
#   B revisits: c1 (x3), c2 (x1), c3 (x1)   -> counts {c1:3, c2:1, c3:1}
#
#   UNWEIGHTED (distinct clusters):
#     A set = {c0,c1,c2}, B set = {c1,c2,c3}
#     intersection = {c1,c2} = 2 ; union = {c0,c1,c2,c3} = 4
#     Jaccard_run1 = 2/4 = 0.5
#
#   WEIGHTED (per-cluster counts, over union of clusters c0,c1,c2,c3):
#     A: [c0=2, c1=1, c2=1, c3=0]
#     B: [c0=0, c1=3, c2=1, c3=1]
#     min = [0,1,1,0] -> sum 2 ; max = [2,3,1,1] -> sum 7
#     WeightedJaccard_run1 = 2/7 = 0.285714...
#
# RUN 2 revisit events:
#   A revisits: c0 (x1), c5 (x1)            -> counts {c0:1, c5:1}
#   B revisits: c0 (x1), c5 (x1)            -> counts {c0:1, c5:1}
#     identical sets AND identical counts
#     UNWEIGHTED: {c0,c5} vs {c0,c5} -> 2/2 = 1.0
#     WEIGHTED:   min=[1,1] sum2, max=[1,1] sum2 -> 2/2 = 1.0
#
# AVERAGE across the two runs (what shared_revisit_rate returns):
#   UNWEIGHTED: mean(0.5, 1.0)         = 0.75
#   WEIGHTED:   mean(2/7, 1.0)         = (0.285714 + 1)/2 = 0.642857...

EVENTS = [
    # run 1, algorithm A
    {'algorithm': 'A', 'run': 1, 'cluster': 'c0'},
    {'algorithm': 'A', 'run': 1, 'cluster': 'c0'},
    {'algorithm': 'A', 'run': 1, 'cluster': 'c1'},
    {'algorithm': 'A', 'run': 1, 'cluster': 'c2'},
    # run 1, algorithm B
    {'algorithm': 'B', 'run': 1, 'cluster': 'c1'},
    {'algorithm': 'B', 'run': 1, 'cluster': 'c1'},
    {'algorithm': 'B', 'run': 1, 'cluster': 'c1'},
    {'algorithm': 'B', 'run': 1, 'cluster': 'c2'},
    {'algorithm': 'B', 'run': 1, 'cluster': 'c3'},
    # run 2, algorithm A
    {'algorithm': 'A', 'run': 2, 'cluster': 'c0'},
    {'algorithm': 'A', 'run': 2, 'cluster': 'c5'},
    # run 2, algorithm B
    {'algorithm': 'B', 'run': 2, 'cluster': 'c0'},
    {'algorithm': 'B', 'run': 2, 'cluster': 'c5'},
]


def test_unweighted():
    history = make_history(EVENTS)
    result = shared_revisit_rate(history, 'A', 'B', by='problem', weighted=False)
    val = result['similarity'].iloc[0]
    print("UNWEIGHTED")
    print("  run1 Jaccard = 0.5, run2 Jaccard = 1.0")
    print("  expected average = 0.75")
    print("  got:", round(val, 6))
    assert abs(val - 0.75) < 1e-9, val
    print("  PASS\n")


def test_weighted():
    history = make_history(EVENTS)
    result = shared_revisit_rate(history, 'A', 'B', by='problem', weighted=True)
    val = result['similarity'].iloc[0]
    expected = np.mean([2/7, 1.0])
    print("WEIGHTED")
    print("  run1 weighted Jaccard = 2/7 =", round(2/7, 6), ", run2 = 1.0")
    print("  expected average =", round(expected, 6))
    print("  got:", round(val, 6))
    assert abs(val - expected) < 1e-9, val
    print("  PASS\n")


def test_disjoint_and_empty_edge_cases():
    # One run where the two algorithms revisit completely disjoint clusters,
    # and one run where NEITHER revisits anything (should be skipped entirely,
    # not counted as similarity 0).
    events = [
        # run 1: disjoint
        {'algorithm': 'A', 'run': 1, 'cluster': 'c0'},
        {'algorithm': 'B', 'run': 1, 'cluster': 'c9'},
        # run 2: only A revisits, B revisits nothing -> union>0, intersection 0
        {'algorithm': 'A', 'run': 2, 'cluster': 'c0'},
        # run 3: neither revisits anything -> no rows at all for this run
    ]
    history = make_history(events)
    result = shared_revisit_rate(history, 'A', 'B', by='problem', weighted=False)
    val = result['similarity'].iloc[0]
    # run1: {c0} vs {c9} -> 0/2 = 0.0
    # run2: {c0} vs {}   -> 0/1 = 0.0
    # (run3 doesn't exist in the data -> not averaged in)
    # average = mean(0.0, 0.0) = 0.0
    print("DISJOINT / ONE-EMPTY")
    print("  run1 = 0.0 (disjoint), run2 = 0.0 (B empty), expected avg = 0.0")
    print("  got:", round(val, 6))
    assert abs(val - 0.0) < 1e-9, val
    print("  PASS\n")


def test_identical_algorithms():
    # An algorithm compared to another with identical events -> similarity 1.0
    events = [
        {'algorithm': 'A', 'run': 1, 'cluster': 'c0'},
        {'algorithm': 'A', 'run': 1, 'cluster': 'c1'},
        {'algorithm': 'B', 'run': 1, 'cluster': 'c0'},
        {'algorithm': 'B', 'run': 1, 'cluster': 'c1'},
    ]
    history = make_history(events)
    unw = shared_revisit_rate(history, 'A', 'B', weighted=False)['similarity'].iloc[0]
    wgt = shared_revisit_rate(history, 'A', 'B', weighted=True)['similarity'].iloc[0]
    print("IDENTICAL")
    print("  unweighted:", round(unw, 6), "| weighted:", round(wgt, 6))
    assert abs(unw - 1.0) < 1e-9 and abs(wgt - 1.0) < 1e-9
    print("  PASS\n")


if __name__ == "__main__":
    test_unweighted()
    test_weighted()
    test_disjoint_and_empty_edge_cases()
    test_identical_algorithms()
    print("All shared_revisit_rate tests passed.")
