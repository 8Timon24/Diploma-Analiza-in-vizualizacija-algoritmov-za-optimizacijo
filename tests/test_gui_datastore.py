# Unit tests for gui.core.datastore: the GUI's read path over the pipeline's
# output. Three physical formats hide behind the same .csv extension (plain,
# zip-wrapped in data/processed/, parquet in clustering_results/), and the
# leading unnamed column means something different in each family - both are
# easy to get wrong silently, so they are pinned here.
#
# Synthetic fixtures only: no real data, no Qt, no mealpy/cocoex.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest

from gui.core import datastore


# -- format sniffing ---------------------------------------------------

def test_sniffs_plain_csv(tmp_path):
    path = tmp_path / "trajectory_1.csv"
    pd.DataFrame({"x1": [1.0], "fitness": [2.0]}).to_csv(path, index=False)
    assert datastore.sniff_format(path) == "csv"


def test_sniffs_zip_wrapped_csv_despite_csv_extension(tmp_path):
    # data/processed/*.csv are really zip archives; trusting the extension
    # here is what would break the browser on that whole family.
    path = tmp_path / "F1_I1.csv"
    pd.DataFrame({"x0": [1.0]}).to_csv(path, compression="zip")
    assert datastore.sniff_format(path) == "zip"


def test_sniffs_parquet(tmp_path):
    pytest.importorskip("pyarrow")
    path = tmp_path / "F1_I1.parquet"
    pd.DataFrame({"x0": [1.0]}).to_parquet(path)
    assert datastore.sniff_format(path) == "parquet"


def test_each_format_round_trips_through_load_table(tmp_path):
    pytest.importorskip("pyarrow")
    frame = pd.DataFrame({"x0": [1.0, 2.0], "raw_y": [3.0, 4.0]})

    plain = tmp_path / "plain.csv"
    frame.to_csv(plain, index=False)
    zipped = tmp_path / "zipped.csv"
    frame.to_csv(zipped, compression="zip", index=False)
    parquet = tmp_path / "table.parquet"
    frame.to_parquet(parquet, index=False)

    for path, expected_fmt in ((plain, "csv"), (zipped, "zip"), (parquet, "parquet")):
        table = datastore.load_table(path)
        assert table.fmt == expected_fmt
        assert len(table.frame) == 2
        assert {"x0", "raw_y"} <= set(table.frame.columns)


# -- the leading unnamed column ---------------------------------------

def test_redundant_range_index_is_dropped(tmp_path):
    path = tmp_path / "results.csv"
    pd.DataFrame({"algorithm": ["A", "B"], "fitness": [1.0, 2.0]}).to_csv(path)
    table = datastore.load_table(path)
    assert "Unnamed: 0" not in table.frame.columns
    assert list(table.frame.columns) == ["algorithm", "fitness"]


def test_non_monotonic_index_is_kept_not_dropped(tmp_path):
    # algorithm_pairwise_similarity/ carries a meaningful source index.
    path = tmp_path / "algorithm_mean_similarity_2D.csv"
    frame = pd.DataFrame({"algorithm": ["A", "B"], "value": [0.5, 0.6]}, index=[189, 594])
    frame.to_csv(path)
    table = datastore.load_table(path)
    assert "source_index" in table.frame.columns
    assert table.frame["source_index"].tolist() == [189, 594]


def test_cluster_centers_unnamed_column_is_the_cluster_id(tmp_path):
    centers = tmp_path / "cluster_centers"
    centers.mkdir()
    path = centers / "F1_I1.csv"
    pd.DataFrame({"scaled_x0": [0.1, 0.2], "scaled_x1": [0.3, 0.4]}).to_csv(path)
    table = datastore.load_table(path)
    assert "cluster" in table.frame.columns
    assert table.frame["cluster"].tolist() == [0, 1]


# -- key normalisation --------------------------------------------------

def test_metric_keys_are_normalised_on_read(tmp_path):
    # Pairwise metric CSVs write Function_id as "F10" and Instance_id as "I1";
    # merged_dim_{d}.csv writes them as ints. The browser shows one shape.
    path = tmp_path / "F10_I1.csv"
    pd.DataFrame({
        "Algorithm1": ["A"], "Algorithm2": ["B"],
        "Function_id": ["F10"], "Instance_id": ["I1"], "Run_id": ["3"],
        "Cosine_distance": [0.5],
    }).to_csv(path, index=False)
    table = datastore.load_table(path)
    assert table.frame["Function_id"].tolist() == [10]
    assert table.frame["Instance_id"].tolist() == [1]
    assert table.frame["Run_id"].tolist() == [3]


def test_frames_without_metric_keys_are_left_alone(tmp_path):
    path = tmp_path / "diversity_1.csv"
    original = pd.DataFrame({"iteration": [1.0], "diversity": [1.7], "exploration": [100.0]})
    original.to_csv(path, index=False)
    table = datastore.load_table(path)
    pd.testing.assert_frame_equal(table.frame, original)
    assert table.notes == []


# -- matrix detection ---------------------------------------------------

def test_spearman_matrices_are_recognised_and_keep_their_labels(tmp_path):
    path = tmp_path / "spearman_dim_2.csv"
    frame = pd.DataFrame(
        [[1.0, 0.3], [0.3, 1.0]],
        index=["entropy", "cosine"], columns=["entropy", "cosine"],
    )
    frame.to_csv(path)
    assert datastore.is_matrix(path)
    table = datastore.load_table(path)
    assert list(table.frame.index) == ["entropy", "cosine"]
    assert list(table.frame.columns) == ["entropy", "cosine"]


def test_ordinary_metric_files_are_not_matrices(tmp_path):
    assert not datastore.is_matrix(tmp_path / "merged_dim_2.csv")
    assert not datastore.is_matrix(tmp_path / "F10_I1.csv")


# -- lazy tree ----------------------------------------------------------

def test_natural_sort_orders_f2_before_f10():
    names = ["F10_I1", "F2_I1", "F1_I1"]
    assert sorted(names, key=datastore._natural_key) == ["F1_I1", "F2_I1", "F10_I1"]


def test_nodes_do_not_touch_the_filesystem_until_asked(tmp_path):
    calls = []

    def loader():
        calls.append(1)
        return [datastore.Node(label="child", kind="table", path=tmp_path / "x.csv")]

    node = datastore.Node(label="parent", loader=loader)
    assert calls == []            # constructing a node lists nothing
    assert len(node.children()) == 1
    assert calls == [1]
    node.children()               # cached, not re-listed
    assert calls == [1]


def test_build_tree_omits_roots_that_do_not_exist(tmp_path, monkeypatch):
    # The GUI workspace is a root too now, so it must be pointed somewhere
    # empty as well - otherwise this picks up the developer's real runs.
    from gui.core import workspace

    monkeypatch.setenv(workspace.WORKSPACE_ENV_VAR, str(tmp_path / "no_runs"))
    monkeypatch.setattr(datastore.config, "OUTPUTS_DIR", str(tmp_path / "nope"))
    monkeypatch.setattr(datastore.config, "PROCESSED_DIR", str(tmp_path / "nope"))
    monkeypatch.setattr(datastore.config, "CLUSTERING_LATEST_DIR", str(tmp_path / "nope"))
    monkeypatch.setattr(datastore.config, "ENTROPY_DATA_DIR", str(tmp_path / "nope"))
    monkeypatch.setattr(datastore.config, "METRICS_DIR", str(tmp_path / "nope"))
    monkeypatch.setattr(datastore.config, "MERGED_DIR", str(tmp_path / "nope"))
    monkeypatch.setattr(datastore.config, "SCALARS_CSV", str(tmp_path / "nope.csv"))
    monkeypatch.setattr(datastore.config, "CLUSTERING_DBSCAN_DIR", str(tmp_path / "nope"))
    assert datastore.build_tree() == []
