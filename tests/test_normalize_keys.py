# Unit tests for config.normalize_keys(): coerces Function_id/Instance_id/
# Run_id to plain ints regardless of whether a pairwise script wrote them as
# an int (1) or a string ("F1"/"I1") - this is what lets merge_metrics.py and
# spearman.py join metric CSVs from different scripts on the same keys.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from config import normalize_keys


def test_string_prefixed_ids_become_ints():
    df = pd.DataFrame({
        "Function_id": ["F3", "F17"],
        "Instance_id": ["I1", "I2"],
        "Run_id": ["1", "5"],
    })
    out = normalize_keys(df)
    assert out["Function_id"].tolist() == [3, 17]
    assert out["Instance_id"].tolist() == [1, 2]
    assert out["Run_id"].tolist() == [1, 5]
    assert out["Function_id"].dtype.kind == "i"
    assert out["Instance_id"].dtype.kind == "i"
    assert out["Run_id"].dtype.kind == "i"


def test_already_plain_int_ids_stay_equivalent():
    df = pd.DataFrame({
        "Function_id": [3, 17],
        "Instance_id": [1, 2],
        "Run_id": [1, 5],
    })
    out = normalize_keys(df)
    assert out["Function_id"].tolist() == [3, 17]
    assert out["Instance_id"].tolist() == [1, 2]
    assert out["Run_id"].tolist() == [1, 5]


def test_mixed_prefixed_and_plain_are_comparable_after_normalizing():
    # this is the actual scenario normalize_keys exists for: two metric
    # files, one written with string ids, one with plain ints, must line up
    # on the same merge keys afterward.
    written_as_strings = normalize_keys(pd.DataFrame({
        "Function_id": ["F5"], "Instance_id": ["I2"], "Run_id": ["3"],
    }))
    written_as_ints = normalize_keys(pd.DataFrame({
        "Function_id": [5], "Instance_id": [2], "Run_id": [3],
    }))
    assert written_as_strings.iloc[0].tolist() == written_as_ints.iloc[0].tolist()


def test_missing_columns_are_left_alone():
    # a dataframe that doesn't have one of the key columns shouldn't crash -
    # some callers only have a subset of the keys.
    df = pd.DataFrame({"Function_id": ["F1"], "value": [0.5]})
    out = normalize_keys(df)
    assert out["Function_id"].tolist() == [1]
    assert "Run_id" not in out.columns
    assert out["value"].tolist() == [0.5]


def test_no_key_columns_at_all_is_a_noop():
    df = pd.DataFrame({"value": [1, 2, 3]})
    out = normalize_keys(df)
    assert out["value"].tolist() == [1, 2, 3]
