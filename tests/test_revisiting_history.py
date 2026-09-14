# Unit tests for revisiting_history() in 04_metrics/return_rate.py: logs a
# "revisit" event whenever a cluster that was already active in an earlier
# iteration (agent count > threshold) becomes active again later.
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "04_metrics"))

import pandas as pd
from return_rate import revisiting_history


def test_single_revisit_is_logged_with_correct_gap():
    # iter1: c0 active (5>3) - first visit, no event.
    # iter2: c1 active (5>3) - first visit, no event.
    # iter3: c0 active again (5>3) - REVISIT: last seen at iter1, gap=2.
    # iter4: nothing active.
    d = pd.DataFrame(
        {"c0": [5, 0, 5, 0], "c1": [0, 5, 0, 0]},
        index=[1, 2, 3, 4],
    )
    events = revisiting_history(d, threshold=3)
    assert len(events) == 1
    row = events.iloc[0]
    assert row["cluster"] == "c0"
    assert row["iteration"] == 3
    assert row["last_visited"] == 1
    assert row["gap"] == 2
    assert row["agents_count"] == 5


def test_no_revisits_returns_empty():
    # each cluster only ever becomes active once - nothing to log.
    d = pd.DataFrame(
        {"c0": [5, 0, 0], "c1": [0, 5, 0]},
        index=[1, 2, 3],
    )
    events = revisiting_history(d, threshold=3)
    assert len(events) == 0


def test_threshold_is_strict_not_inclusive():
    # a count exactly AT the threshold does not count as "active".
    d = pd.DataFrame({"c0": [3, 3]}, index=[1, 2])
    events = revisiting_history(d, threshold=3)
    assert len(events) == 0


def test_consecutive_active_iterations_count_as_a_revisit():
    # active in back-to-back iterations still logs an event - "revisit"
    # only means "seen active in an earlier iteration", not "left and came
    # back".
    d = pd.DataFrame({"c0": [5, 5]}, index=[1, 2])
    events = revisiting_history(d, threshold=3)
    assert len(events) == 1
    assert events.iloc[0]["gap"] == 1


def test_custom_threshold_changes_what_counts_as_active():
    d = pd.DataFrame({"c0": [10, 0, 10]}, index=[1, 2, 3])
    # with a high threshold, count=10 no longer counts as active anywhere
    events = revisiting_history(d, threshold=20)
    assert len(events) == 0
