# gui/core/pipeline.py: the stage table and the in-process runner that lets
# the app run the analysis pipeline without a terminal.
#
# The runner itself is a QThread, so the tests that touch it drive run()
# directly on the calling thread rather than starting it - a QThread does not
# need to be started to be exercised, and starting one needs an event loop.
import sys
import threading
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

pytest.importorskip("PySide6")

import run_pipeline
from gui.core import pipeline


# -- the stage table must not drift from run_pipeline.py -----------------

def test_stage_names_and_order_match_run_pipeline():
    """"Start from clustering" in the app and `--from clustering` on the
    command line have to mean the same thing."""
    assert pipeline.STAGE_NAMES == run_pipeline.STEP_NAMES


def test_stages_from_mirrors_the_from_flag():
    names = [s.name for s in pipeline.stages_from("clustering")]
    index = run_pipeline.STEP_NAMES.index("clustering")
    assert names == run_pipeline.STEP_NAMES[index:]


def test_stages_from_rejects_an_unknown_name():
    with pytest.raises(KeyError):
        pipeline.stages_from("not_a_stage")


def test_every_stage_declares_what_it_overwrites():
    """The confirmation dialog is built from this; a stage with no declared
    target would silently overwrite data the user was never warned about."""
    for stage in pipeline.STAGES:
        assert stage.writes, f"{stage.name} declares no writes"
        for name in stage.writes:
            assert hasattr(pipeline.config, name), (
                f"{stage.name} names {name}, which is not a config attribute"
            )


def test_targets_follow_a_rebound_results_root(monkeypatch, tmp_path):
    """gui/core/results_root.py rebinds config attributes at runtime; the
    stage table must read them at call time, not capture them at import."""
    stage = pipeline.BY_NAME["merge"]
    monkeypatch.setattr(pipeline.config, "MERGED_DIR", str(tmp_path / "merged"))
    assert stage.targets() == [str(tmp_path / "merged")]


# -- PipelineSpec ---------------------------------------------------------

def test_benchmark_is_skipped_without_a_run_spec():
    spec = pipeline.PipelineSpec(stages=["benchmark", "merge"])
    assert [s.name for s in spec.runnable()] == ["merge"]
    assert [s.name for s in spec.skipped()] == ["benchmark"]


def test_benchmark_is_runnable_once_a_run_spec_is_supplied():
    spec = pipeline.PipelineSpec(stages=["benchmark"], run_spec=object())
    assert [s.name for s in spec.runnable()] == ["benchmark"]


def test_targets_are_deduplicated():
    # four metric stages all write METRICS_DIR
    spec = pipeline.PipelineSpec(stages=[
        "entropy_pairwise", "cosine_pairwise", "solutions_pairwise",
    ])
    assert spec.targets() == [pipeline.config.METRICS_DIR]


# -- the runner -----------------------------------------------------------

class _FakeStageResult:
    def __init__(self, stage, written=1, cancelled=False):
        self.stage = stage
        self.written = written
        self.cancelled = cancelled

    def describe(self):
        return f"{self.stage}: {self.written}"


def _collect(runner):
    """Wire every signal into lists, without starting the thread."""
    seen = {"stages": [], "progress": [], "messages": [],
            "completed": [], "failed": []}
    runner.stageStarted.connect(seen["stages"].append)
    runner.progressed.connect(seen["progress"].append)
    runner.message.connect(seen["messages"].append)
    runner.completed.connect(seen["completed"].append)
    runner.failed.connect(seen["failed"].append)
    return seen


def test_runner_runs_the_selected_stages_in_order(monkeypatch):
    called = []

    def fake_load(stage):
        def run(progress_cb=None, cancel_event=None, **kwargs):
            called.append(stage.name)
            progress_cb(1, 1, "item")
            return _FakeStageResult(stage.name)
        return run

    monkeypatch.setattr(pipeline, "load_stage", fake_load)
    spec = pipeline.PipelineSpec(stages=["merge", "build_scalars", "spearman"])
    runner = pipeline.PipelineRunner(spec)
    seen = _collect(runner)
    runner.run()

    assert called == ["merge", "build_scalars", "spearman"]
    assert [s["label"] for s in seen["stages"]] == [
        pipeline.BY_NAME[n].label for n in called
    ]
    assert seen["completed"][0]["written"] == 3
    assert seen["completed"][0]["cancelled"] is False
    assert seen["failed"] == []


def test_progress_carries_the_stage_it_came_from(monkeypatch):
    def fake_load(stage):
        def run(progress_cb=None, cancel_event=None, **kwargs):
            progress_cb(2, 7, "F1_I1")
            return _FakeStageResult(stage.name)
        return run

    monkeypatch.setattr(pipeline, "load_stage", fake_load)
    runner = pipeline.PipelineRunner(pipeline.PipelineSpec(stages=["merge"]))
    seen = _collect(runner)
    runner.run()

    assert seen["progress"][0] == {
        "done": 2, "total": 7, "label": "F1_I1", "stage": "merge",
    }


def test_a_failing_stage_names_itself_and_stops_the_chain(monkeypatch):
    reached = []

    def fake_load(stage):
        def run(progress_cb=None, cancel_event=None, **kwargs):
            if stage.name == "build_scalars":
                raise ValueError("boom")
            reached.append(stage.name)
            return _FakeStageResult(stage.name)
        return run

    monkeypatch.setattr(pipeline, "load_stage", fake_load)
    spec = pipeline.PipelineSpec(stages=["merge", "build_scalars", "spearman"])
    runner = pipeline.PipelineRunner(spec)
    seen = _collect(runner)
    runner.run()

    assert reached == ["merge"], "the chain continued past a failure"
    assert seen["completed"] == [], "a failed run must not report completion"
    assert "Build scalars failed" in seen["failed"][0]
    assert "boom" in seen["failed"][0]


def test_cancelling_stops_before_the_next_stage(monkeypatch):
    started = []

    def fake_load(stage):
        def run(progress_cb=None, cancel_event=None, **kwargs):
            started.append(stage.name)
            cancel_event.set()          # as if the user hit Cancel mid-stage
            return _FakeStageResult(stage.name, cancelled=True)
        return run

    monkeypatch.setattr(pipeline, "load_stage", fake_load)
    spec = pipeline.PipelineSpec(stages=["merge", "build_scalars"])
    runner = pipeline.PipelineRunner(spec)
    seen = _collect(runner)
    runner.run()

    assert started == ["merge"]
    assert seen["completed"][0]["cancelled"] is True


def test_an_empty_selection_fails_rather_than_reporting_success():
    runner = pipeline.PipelineRunner(pipeline.PipelineSpec(stages=[]))
    seen = _collect(runner)
    runner.run()
    assert seen["completed"] == []
    assert "Nothing to run" in seen["failed"][0]


def test_dimensions_are_only_passed_to_stages_that_accept_them(monkeypatch):
    """merge_metrics discovers dimensions from the filesystem and takes no
    such argument; passing one would be a TypeError."""
    received = {}

    def fake_load(stage):
        def run(progress_cb=None, cancel_event=None):   # no dimensions kwarg
            received[stage.name] = "called"
            return _FakeStageResult(stage.name)
        return run

    monkeypatch.setattr(pipeline, "load_stage", fake_load)
    spec = pipeline.PipelineSpec(stages=["merge"], dimensions=[2])
    runner = pipeline.PipelineRunner(spec)
    runner.run()
    assert received == {"merge": "called"}


# -- the manifest ---------------------------------------------------------

def test_manifest_records_the_clustering_seed(tmp_path):
    spec = pipeline.PipelineSpec(stages=["merge"], dimensions=[2])
    summary = {"stages": ["merge"], "written": 1, "cancelled": False,
               "seconds": 1.23}
    path = pipeline.write_manifest(spec, summary, root=tmp_path)

    import json
    payload = json.loads(Path(path).read_text())
    assert payload["clustering_seed"] == pipeline.config.CLUSTERING_SEED
    assert payload["stages_completed"] == ["merge"]
    assert payload["cancelled"] is False


def test_manifest_failure_never_fails_the_run(tmp_path):
    spec = pipeline.PipelineSpec(stages=["merge"])
    # a directory where the file should go - write_text will raise
    (tmp_path / pipeline.MANIFEST_NAME).mkdir()
    assert pipeline.write_manifest(spec, {}, root=tmp_path) is None


def test_the_completed_payload_is_plain_data(monkeypatch):
    """It crosses into the UI thread and lands in a JSON manifest, so it must
    not carry StageResult instances."""
    import json

    def fake_load(stage):
        def run(progress_cb=None, cancel_event=None, **kwargs):
            return _FakeStageResult(stage.name)
        return run

    monkeypatch.setattr(pipeline, "load_stage", fake_load)
    runner = pipeline.PipelineRunner(pipeline.PipelineSpec(stages=["merge"]))
    seen = _collect(runner)
    runner.run()

    json.dumps(seen["completed"][0])   # raises if anything exotic got in
    assert seen["completed"][0]["per_stage"] == [
        {"stage": "merge", "written": 1, "cancelled": False}
    ]


def test_a_stage_that_cancels_itself_marks_the_run_cancelled(monkeypatch):
    """A stage can stop on its own cancel_event check without the runner's
    own flag being read first; the summary must still say cancelled."""
    def fake_load(stage):
        def run(progress_cb=None, cancel_event=None, **kwargs):
            return _FakeStageResult(stage.name, cancelled=True)
        return run

    monkeypatch.setattr(pipeline, "load_stage", fake_load)
    runner = pipeline.PipelineRunner(pipeline.PipelineSpec(stages=["merge"]))
    seen = _collect(runner)
    runner.run()
    assert seen["completed"][0]["cancelled"] is True


# -- coverage hands the panel real stage names ----------------------------

def test_every_coverage_stage_name_is_a_real_stage():
    """coverage.py mirrors the pipeline order rather than importing it (that
    module must stay Qt-free), so the two can drift. They must not."""
    from gui.core import coverage

    assert list(coverage._PIPELINE_ORDER) == pipeline.STAGE_NAMES
    for name in coverage.ENTROPY_STAGES:
        assert name in pipeline.BY_NAME, f"{name} is not a pipeline stage"


def test_coverage_recipes_carry_runnable_stage_names(tmp_path, monkeypatch):
    """Whatever a gap suggests, the Process tab has to be able to run it."""
    from gui.core import coverage

    monkeypatch.setattr(coverage.config, "MERGED_DIR", str(tmp_path))
    gap = coverage.check_merged(2)
    assert gap.ok is False
    assert gap.stages == ("merge",)
    for name in gap.stages:
        assert name in pipeline.BY_NAME


def test_spearman_gap_asks_for_every_stage_it_depends_on(tmp_path, monkeypatch):
    from gui.core import coverage

    monkeypatch.setattr(coverage.config, "MERGED_DIR", str(tmp_path))
    gap = coverage.check_spearman(2)
    assert gap.stages[0] == "cosine_pairwise"
    assert gap.stages[-1] == "spearman"
    assert "benchmark" not in gap.stages, "a gap should not demand a re-benchmark"
