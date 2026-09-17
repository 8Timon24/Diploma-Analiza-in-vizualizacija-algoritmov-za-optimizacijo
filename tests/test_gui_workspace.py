# Unit tests for gui.core.workspace: where GUI runs write and what they record.
#
# The sandbox is the safety property that matters here. outputs/ holds 11 GB
# of results that are not in version control, and run_benchmarks writes by
# algorithm and problem id - so a GUI run of an algorithm already present
# would overwrite real trajectories in place. Runs therefore go to their own
# directory by default, and carry a manifest describing what produced them.
#
# Synthetic only: no Qt, no mealpy, no cocoex, nothing written outside tmp_path.
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from gui.core.workspace import (
    RunSpec, create_run_dir, format_duration, list_runs, read_manifest,
    write_manifest, workspace_root, WORKSPACE_ENV_VAR,
)


def make_spec(**overrides):
    defaults = dict(
        algorithms=["OriginalDE", "OriginalGWO"],
        functions=[1, 2],
        instances=[1],
        dimensions=[2],
        seeds=[1],
    )
    defaults.update(overrides)
    return RunSpec(**defaults)


# -- sizing and cost ----------------------------------------------------

def test_total_runs_is_the_product_of_every_axis():
    spec = make_spec(instances=[1, 2], seeds=[1, 2, 3])
    assert spec.total_runs == 2 * 2 * 2 * 1 * 3


def test_total_runs_is_zero_when_an_axis_is_empty():
    assert make_spec(functions=[]).total_runs == 0


def test_estimate_grows_with_dimension():
    # epoch = epoch_per_dim * dimension, so cost must scale with dimension.
    small = make_spec(dimensions=[2]).estimate_seconds()
    large = make_spec(dimensions=[10]).estimate_seconds()
    assert large > small


def test_estimate_grows_with_population_size():
    assert (
        make_spec(pop_size=100).estimate_seconds()
        > make_spec(pop_size=10).estimate_seconds()
    )


def test_full_thesis_sweep_is_estimated_in_hours():
    # 28 x 24 x 5 x 3 x 5 = 50,400 runs. CLAUDE.md calls it an hours-long job;
    # an estimate that said "minutes" would be actively misleading.
    sweep = RunSpec(
        algorithms=[f"a{i}" for i in range(28)],
        functions=list(range(1, 25)),
        instances=[1, 2, 3, 4, 5],
        dimensions=[2, 5, 10],
        seeds=[1, 2, 3, 4, 5],
    )
    assert sweep.total_runs == 50_400
    assert sweep.estimate_seconds() > 3600


@pytest.mark.parametrize("seconds,expected", [
    (0.4, "under a second"), (45, "45s"), (600, "10 min"), (7200, "2.0 hours"),
])
def test_duration_formatting(seconds, expected):
    assert format_duration(seconds) == expected


# -- validation ---------------------------------------------------------

def test_a_complete_spec_validates():
    assert make_spec().validate() == []


@pytest.mark.parametrize("field", ["algorithms", "functions", "instances", "dimensions", "seeds"])
def test_every_empty_axis_is_reported(field):
    problems = make_spec(**{field: []}).validate()
    assert any(field.rstrip("s") in problem for problem in problems)


def test_degenerate_population_is_rejected():
    assert make_spec(pop_size=1).validate()


# -- run directories ----------------------------------------------------

def test_workspace_root_is_overridable(tmp_path, monkeypatch):
    # The packaged app points this at a user-writable location.
    monkeypatch.setenv(WORKSPACE_ENV_VAR, str(tmp_path))
    assert workspace_root() == tmp_path


def test_create_run_dir_makes_an_outputs_subdir_and_manifest(tmp_path):
    spec = make_spec()
    run_dir = create_run_dir(spec, root=tmp_path)
    assert (run_dir / "outputs").is_dir()
    manifest = read_manifest(run_dir)
    assert manifest["status"] == "started"
    assert manifest["total_runs"] == spec.total_runs
    assert manifest["spec"]["algorithms"] == ["OriginalDE", "OriginalGWO"]


def test_run_dirs_never_collide(tmp_path):
    # Two runs started in the same second must not share a directory.
    first = create_run_dir(make_spec(), root=tmp_path, label="x")
    second = create_run_dir(make_spec(), root=tmp_path, label="x")
    assert first != second
    assert (second / "outputs").is_dir()


def test_manifest_records_what_is_needed_to_reproduce(tmp_path):
    run_dir = create_run_dir(make_spec(), root=tmp_path)
    manifest = read_manifest(run_dir)
    # Library versions, because a mealpy change alters the optimizers, and the
    # clustering seed, because it drives everything computed downstream.
    assert "mealpy" in manifest["versions"]
    assert "python" in manifest["versions"]
    assert "clustering_seed" in manifest


def test_write_manifest_merges_rather_than_replacing(tmp_path):
    spec = make_spec()
    run_dir = create_run_dir(spec, root=tmp_path)
    created = read_manifest(run_dir)["created"]
    write_manifest(run_dir, spec, status="completed", completed_runs=4)
    manifest = read_manifest(run_dir)
    assert manifest["status"] == "completed"
    assert manifest["completed_runs"] == 4
    assert manifest["created"] == created   # original fields survive


def test_read_manifest_tolerates_a_corrupt_file(tmp_path):
    run_dir = tmp_path / "broken"
    (run_dir / "outputs").mkdir(parents=True)
    (run_dir / "manifest.json").write_text("{not json")
    assert read_manifest(run_dir) is None


def test_list_runs_is_newest_first_and_skips_stray_dirs(tmp_path):
    create_run_dir(make_spec(), root=tmp_path, label="aaa")
    create_run_dir(make_spec(), root=tmp_path, label="zzz")
    (tmp_path / "not-a-run").mkdir()
    runs = list_runs(root=tmp_path)
    assert len(runs) == 2
    assert runs[0]["path"].name > runs[1]["path"].name


def test_list_runs_on_a_missing_root_is_empty(tmp_path):
    assert list_runs(root=tmp_path / "nothing") == []


# -- outcome recording --------------------------------------------------

@pytest.mark.parametrize("summary,expected", [
    ({"ok": 4, "failed": 0, "cancelled": False, "seconds": 1.0}, "completed"),
    ({"ok": 2, "failed": 1, "cancelled": True, "seconds": 1.0}, "cancelled"),
    ({"ok": 0, "failed": 4, "cancelled": False, "seconds": 1.0}, "failed"),
])
def test_finalise_run_records_the_right_outcome(tmp_path, summary, expected):
    # "every run failed" must not be recorded as success: the CLI prints
    # per-problem failures and still exits 0, which is exactly the confusion
    # the progress hook exists to remove.
    pytest.importorskip("PySide6")
    from gui.core.runner import finalise_run

    spec = make_spec()
    run_dir = create_run_dir(spec, root=tmp_path)
    assert finalise_run(run_dir, spec, summary) == expected
    assert read_manifest(run_dir)["status"] == expected
