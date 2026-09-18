# Every pipeline stage must be callable in-process, not just runnable as a
# script. The desktop app drives them directly - a packaged app has no
# interpreter to spawn, so subprocess is not an option there.
#
# These tests derive the stage list from run_pipeline.STEPS rather than
# repeating it, so a step added to the pipeline is automatically covered.
import importlib
import importlib.util
import inspect
import sys
import threading
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import run_pipeline


# The benchmark is the one step the GUI does NOT drive through a stage run():
# helper_functions.run_benchmarks already takes progress_cb/cancel_event, and
# gui/core/runner.py has driven it that way since before this contract existed.
# Its script stays a thin CLI over that function.
DRIVEN_ELSEWHERE = {"benchmark"}


def _stage_scripts():
    """(step_name, script_path) for every step that runs a stage script."""
    found = []
    for name, cmd, _description in run_pipeline.STEPS:
        script = Path(cmd[1])
        if script.suffix == ".py" and name not in DRIVEN_ELSEWHERE:
            found.append((name, script))
    return found


def _load(script):
    """Import a stage module from its path, with its folder on sys.path.

    The numbered stage folders are not importable packages, which is why the
    GUI and these tests load them this way.
    """
    folder = str(script.parent)
    if folder not in sys.path:
        sys.path.insert(0, folder)
    spec = importlib.util.spec_from_file_location(script.stem, script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STAGES = _stage_scripts()


def test_every_step_is_either_a_callable_stage_or_driven_elsewhere():
    """No step may be silently unreachable from the GUI."""
    covered = {name for name, _ in STAGES} | DRIVEN_ELSEWHERE
    assert covered == set(run_pipeline.STEP_NAMES)
    # order matters: the GUI's "start from here" must agree with --from
    assert [name for name, _ in STAGES] == [
        name for name in run_pipeline.STEP_NAMES if name not in DRIVEN_ELSEWHERE
    ]


@pytest.mark.parametrize("name,script", STAGES, ids=[n for n, _ in STAGES])
def test_stage_exposes_a_callable_run(name, script, monkeypatch):
    """The contract in pipeline_api: run(progress_cb=None, cancel_event=None).

    Importing must also be side-effect free. Two of these scripts used to call
    parse_args() at module scope, so importing them from a process with its own
    command line parsed THAT argv and could SystemExit.
    """
    monkeypatch.setattr(sys, "argv", ["something-else", "--not-a-stage-flag"])
    try:
        module = _load(script)
    except ImportError as exc:
        pytest.skip(f"{name}: heavy dependency not installed ({exc})")

    assert hasattr(module, "run"), f"{name} has no run()"
    assert callable(module.run)

    parameters = inspect.signature(module.run).parameters
    for required in ("progress_cb", "cancel_event"):
        assert required in parameters, f"{name}.run() takes no {required}"
        assert parameters[required].default is None, (
            f"{name}.run()'s {required} must default to None so the CLI path "
            f"costs nothing"
        )


@pytest.mark.parametrize("name,script", STAGES, ids=[n for n, _ in STAGES])
def test_stage_still_has_a_main_guard(name, script):
    """Importing a stage must never start doing its work - several used to
    have no guard at all, and importing one for a unit test kicked off its
    whole real computation."""
    text = script.read_text()
    assert "__main__" in text, f"{name} lost its __main__ guard"


def _load_merge_metrics():
    """merge_metrics.py imports tqdm at module scope, which CI's minimal
    dependency set (pandas, numpy, pytest - see .github/workflows/tests.yml)
    does not install. Same skip-on-ImportError pattern as
    test_stage_exposes_a_callable_run, so this stays passing whether or not
    the full requirements.txt is installed."""
    try:
        return _load(REPO_ROOT / "05_analysis" / "merge_metrics.py")
    except ImportError as exc:
        pytest.skip(f"merge_metrics: heavy dependency not installed ({exc})")


def test_a_preset_cancel_event_stops_a_stage_before_it_works(tmp_path, monkeypatch):
    """The cancel contract, exercised on a real stage rather than on a mock."""
    pytest.importorskip("pandas")
    import config

    metrics = tmp_path / "metrics_data"
    (metrics / "entropy" / "dim_2").mkdir(parents=True)
    monkeypatch.setattr(config, "METRICS_DIR", str(metrics))
    monkeypatch.setattr(config, "MERGED_DIR", str(tmp_path / "merged"))

    module = _load_merge_metrics()

    already_cancelled = threading.Event()
    already_cancelled.set()
    result = module.run(cancel_event=already_cancelled)

    assert result.cancelled is True
    assert result.written == 0
    assert not (tmp_path / "merged").exists(), "a cancelled stage wrote output"


def test_progress_is_reported_with_a_total(tmp_path, monkeypatch):
    pytest.importorskip("pandas")
    import config

    metrics = tmp_path / "metrics_data"
    (metrics / "entropy" / "dim_2").mkdir(parents=True)
    (metrics / "cosine" / "dim_2").mkdir(parents=True)
    monkeypatch.setattr(config, "METRICS_DIR", str(metrics))
    monkeypatch.setattr(config, "MERGED_DIR", str(tmp_path / "merged"))

    module = _load_merge_metrics()

    seen = []
    module.run(progress_cb=lambda done, total, label: seen.append((done, total)))

    assert seen, "no progress was reported"
    # a bar needs the total up front, on the very first tick
    assert seen[0][1] == 2
    assert seen[-1][0] == 2
