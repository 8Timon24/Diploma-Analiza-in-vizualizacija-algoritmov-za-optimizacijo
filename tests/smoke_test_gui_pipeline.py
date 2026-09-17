"""End-to-end smoke test for the IN-PROCESS pipeline the desktop app drives.

tests/smoke_test_pipeline.py proves the command-line path still works: every
stage invoked as a subprocess by run_pipeline.py. This one proves the other
path - gui/core/pipeline.PipelineRunner calling each stage's run() inside one
process - produces the same files from the same input.

Both paths have to keep working. The CLI is what the thesis was computed with
and what run_pipeline.py still uses; the in-process path is the only one
available to a packaged app, which ships no interpreter to spawn.

Deliberately NOT named test_*.py, for the same reason as its sibling: it runs
real mealpy/cocoex optimization and real KMeans. Run it explicitly:

    venv/bin/python3 -m pytest tests/smoke_test_gui_pipeline.py -v -s

Isolation is identical: PIPELINE_TEST_ROOT and the PIPELINE_TEST_* sweep
overrides point everything at pytest's tmp_path. The stage code reads those
through config.py at import, so the child process below is where they must be
set - not this one, where config is already imported.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import pytest

from smoke_test_pipeline import (
    TEST_ALGORITHMS, TEST_DIMENSIONS, TEST_FUNCTIONS, TEST_INSTANCES,
    TEST_SEEDS, _build_env,
)

pytest.importorskip("PySide6")


# The child process: drives PipelineRunner directly. Written to a file rather
# than passed with -c so a traceback has real line numbers.
DRIVER = '''
import json, sys
sys.path.insert(0, {repo!r})

from gui.core import pipeline

# Every stage except the benchmark, which needs a full RunSpec and is driven
# by gui/core/runner.py - the raw data it would produce is made by the CLI
# step this test runs first.
stages = [name for name in pipeline.STAGE_NAMES if name != "benchmark"]
spec = pipeline.PipelineSpec(stages=stages)

events = {{"stages": [], "progress": 0, "failed": [], "completed": []}}
runner = pipeline.PipelineRunner(spec)
runner.stageStarted.connect(lambda payload: events["stages"].append(payload["name"]))
runner.progressed.connect(lambda _payload: events.__setitem__("progress", events["progress"] + 1))
runner.failed.connect(events["failed"].append)
runner.completed.connect(lambda summary: events["completed"].append({{
    "stages": summary["stages"],
    "written": summary["written"],
    "cancelled": summary["cancelled"],
}}))

runner.run()   # called directly: a QThread needs no event loop to be exercised

print("PIPELINE_EVENTS " + json.dumps(events))
'''


def test_in_process_pipeline_matches_the_cli(tmp_path):
    env = _build_env(tmp_path)
    d = TEST_DIMENSIONS[0]
    i0 = TEST_INSTANCES[0]

    # ---- 1. produce the raw data with the real benchmark CLI ----
    benchmark = [
        sys.executable, str(_REPO_ROOT / "01_optimize" / "run_benchmarks.py"),
        "-a", *TEST_ALGORITHMS,
        "-f", *[str(f) for f in TEST_FUNCTIONS],
        "-i", *[str(i) for i in TEST_INSTANCES],
        "-d", *[str(dim) for dim in TEST_DIMENSIONS],
        "-s", *[str(s) for s in TEST_SEEDS],
        "-e",  # save diversity - exploration_pairwise needs it
    ]
    assert subprocess.run(benchmark, env=env).returncode == 0

    # ---- 2. run every remaining stage IN PROCESS ----
    driver = tmp_path / "_drive_pipeline.py"
    driver.write_text(DRIVER.format(repo=str(_REPO_ROOT)))
    env["QT_QPA_PLATFORM"] = "offscreen"

    finished = subprocess.run(
        [sys.executable, str(driver)], env=env,
        capture_output=True, text=True,
    )
    print(finished.stdout[-4000:])
    print(finished.stderr[-4000:], file=sys.stderr)
    assert finished.returncode == 0, "the in-process driver crashed"

    line = [l for l in finished.stdout.splitlines()
            if l.startswith("PIPELINE_EVENTS ")]
    assert line, "the driver produced no result line"
    events = json.loads(line[-1][len("PIPELINE_EVENTS "):])

    assert events["failed"] == [], f"a stage failed: {events['failed']}"
    assert events["completed"], "the run never reported completion"
    summary = events["completed"][0]
    assert summary["cancelled"] is False
    assert summary["written"] > 0, "the run claims it wrote nothing"

    # every non-benchmark stage announced itself, in pipeline order
    expected = [n for n in _stage_names() if n != "benchmark"]
    assert events["stages"] == expected
    assert events["progress"] > 0, "no per-item progress was reported"

    # ---- 3. the same artefacts the CLI smoke test checks ----
    results_csv = tmp_path / "outputs" / f"dim_{d}" / "results.csv"
    assert results_csv.is_file(), "harvest_results did not write results.csv"
    results = pd.read_csv(results_csv)
    assert set(results["algorithm"].unique()) == set(TEST_ALGORITHMS)

    for f in TEST_FUNCTIONS:
        assert (tmp_path / "data" / "processed" / f"dim_{d}" /
                f"F{f}_I{i0}.csv").is_file(), "preprocess produced nothing"
        assert (tmp_path / "data" / "clustering_latest" / "cluster_distributions"
                / f"dim_{d}" / f"F{f}_I{i0}.csv").is_file(), "clustering produced nothing"

    metric_columns = {
        "entropy": "Mean_entropy_difference",
        "cosine": "Cosine_distance",
        "cosine_columns": "Cosine_column_distance",
        "exploration": "Mean_exploration_difference",
        "location": "Location_difference",
        "fitness": "Fitness_difference",
    }
    for subdir, value_col in metric_columns.items():
        found = list((tmp_path / "metrics_data" / subdir / f"dim_{d}").glob("F*_I*.csv"))
        assert found, f"no {subdir} metric file was written"
        assert value_col in pd.read_csv(found[0]).columns

    merged = pd.read_csv(tmp_path / "metrics_data" / "merged" / f"merged_dim_{d}.csv")
    assert len(merged) > 0
    for column in metric_columns:
        assert column in merged.columns, f"merged table lost '{column}'"

    scalars = pd.read_csv(tmp_path / "metrics_data" / "scalars.csv")
    assert set(scalars["algo"]) == set(TEST_ALGORITHMS)
    assert not scalars[["entropy", "fitness", "exploration", "diversity"]].isna().any().any()

    assert (tmp_path / "metrics_data" / "merged" / f"spearman_dim_{d}.csv").is_file()
    # the Figure-API rewrite of plot_spearman must still write the PDF when
    # the pipeline runs it (only the GUI's renderer suppresses savefig)
    assert (tmp_path / "figures_spearman" / f"spearman_dim_{d}.pdf").is_file()


def _stage_names():
    from gui.core import pipeline

    return pipeline.STAGE_NAMES


# The driver above calls PipelineRunner.run() directly, which never starts a
# real thread. This one drives the Process tab exactly as a user would, so the
# QThread is genuinely started and its signals genuinely cross threads - the
# part no unit test can reach.
PANEL_DRIVER = '''
import json, sys, time
sys.path.insert(0, {repo!r})

from gui.qt import QtWidgets
from gui import theme
from gui.panels.pipeline_panel import PipelinePanel

app = QtWidgets.QApplication([])
theme.apply(app)

panel = PipelinePanel()
panel._select_from("harvest_results")
panel._confirm = lambda spec: True          # the dialog is unit-tested separately

events = {{"finished": [], "stage_labels": []}}
panel.pipelineFinished.connect(events["finished"].append)

panel._request_run()
assert panel.is_running, "the panel did not start a worker thread"

deadline = time.time() + 600
while panel.is_running and time.time() < deadline:
    app.processEvents()
    time.sleep(0.01)
for _ in range(50):
    app.processEvents()
    time.sleep(0.01)

print("PANEL_EVENTS " + json.dumps({{
    "finished": events["finished"],
    "overall_at_max": panel.overall_progress.value() == panel.overall_progress.maximum(),
    "status": panel.status.text(),
    "log_lines": len(panel.log.toPlainText().splitlines()),
}}))
'''


def test_the_process_tab_runs_a_real_worker_thread(tmp_path):
    env = _build_env(tmp_path)

    benchmark = [
        sys.executable, str(_REPO_ROOT / "01_optimize" / "run_benchmarks.py"),
        "-a", *TEST_ALGORITHMS,
        "-f", *[str(f) for f in TEST_FUNCTIONS],
        "-i", *[str(i) for i in TEST_INSTANCES],
        "-d", *[str(dim) for dim in TEST_DIMENSIONS],
        "-s", *[str(s) for s in TEST_SEEDS],
        "-e",
    ]
    assert subprocess.run(benchmark, env=env).returncode == 0

    driver = tmp_path / "_drive_panel.py"
    driver.write_text(PANEL_DRIVER.format(repo=str(_REPO_ROOT)))
    env["QT_QPA_PLATFORM"] = "offscreen"

    finished = subprocess.run([sys.executable, str(driver)], env=env,
                              capture_output=True, text=True)
    print(finished.stdout[-3000:])
    print(finished.stderr[-3000:], file=sys.stderr)
    assert finished.returncode == 0, "the panel driver crashed"

    line = [l for l in finished.stdout.splitlines() if l.startswith("PANEL_EVENTS ")]
    assert line, "the panel never reported a result"
    events = json.loads(line[-1][len("PANEL_EVENTS "):])

    assert events["finished"], "pipelineFinished never fired across the thread"
    summary = events["finished"][0]
    assert "error" not in summary, f"the run failed: {summary.get('error')}"
    assert summary["cancelled"] is False
    assert summary["written"] > 0
    assert events["overall_at_max"], "the overall bar did not reach its maximum"
    assert "finished" in events["status"].lower()
    assert events["log_lines"] > 10, "the log stayed suspiciously empty"

    # the run wrote a manifest recording the clustering seed
    manifest = tmp_path / "pipeline_run.json"
    assert manifest.is_file(), "no pipeline_run.json was written"
    recorded = json.loads(manifest.read_text())
    assert recorded["clustering_seed"] is not None
    assert "spearman" in recorded["stages_completed"]
