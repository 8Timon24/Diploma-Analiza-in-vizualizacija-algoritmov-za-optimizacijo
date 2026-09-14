# Unit tests for column_cosine_distance() in 04_metrics/cosine_columns_pairwise.py:
# per-cluster-column cosine distance between two algorithms' occupancy
# tables (rows = iterations, columns = clusters), averaged across clusters.
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "04_metrics"))

import numpy as np
import pytest
from cosine_columns_pairwise import column_cosine_distance


def test_identical_tables_have_zero_distance():
    table = np.array([[1.0, 2.0], [3.0, 4.0]])
    dist = column_cosine_distance(table, table.copy())
    assert dist == pytest.approx(0.0, abs=1e-9)


def test_orthogonal_single_column_has_max_distance():
    # column: A = [1, 0], B = [0, 1] -> cosine similarity 0 -> distance 1
    tableA = np.array([[1.0], [0.0]])
    tableB = np.array([[0.0], [1.0]])
    dist = column_cosine_distance(tableA, tableB)
    assert dist == pytest.approx(1.0, abs=1e-9)


def test_one_side_unvisited_column_counts_as_maximally_dissimilar():
    # A visited this cluster, B never did (all-zero column) -> treated the
    # same as orthogonal: contributes similarity 0, not skipped.
    tableA = np.array([[1.0], [2.0]])
    tableB = np.array([[0.0], [0.0]])
    dist = column_cosine_distance(tableA, tableB)
    assert dist == pytest.approx(1.0, abs=1e-9)


def test_both_sides_unvisited_column_is_skipped_not_penalized():
    # col0: neither algorithm ever visits it -> excluded from the average.
    # col1: identical -> similarity 1. Only col1 should count, so distance
    # is 0, not penalized by the jointly-unvisited col0.
    tableA = np.array([[0.0, 1.0], [0.0, 2.0]])
    tableB = np.array([[0.0, 1.0], [0.0, 2.0]])
    dist = column_cosine_distance(tableA, tableB)
    assert dist == pytest.approx(0.0, abs=1e-9)


def test_all_columns_jointly_unvisited_returns_none():
    tableA = np.zeros((2, 2))
    tableB = np.zeros((2, 2))
    assert column_cosine_distance(tableA, tableB) is None


def test_mean_is_taken_across_columns():
    # col0 identical (sim=1), col1 orthogonal (sim=0) -> mean sim = 0.5,
    # distance = 1 - 0.5 = 0.5.
    tableA = np.array([[1.0, 1.0], [1.0, 0.0]])
    tableB = np.array([[1.0, 0.0], [1.0, 1.0]])
    dist = column_cosine_distance(tableA, tableB)
    assert dist == pytest.approx(0.5, abs=1e-9)
