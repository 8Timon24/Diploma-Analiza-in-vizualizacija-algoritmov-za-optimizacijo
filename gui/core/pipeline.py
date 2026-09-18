"""Driving the analysis pipeline from the app.

run_pipeline.py runs the stages as subprocesses. That is not available to a
packaged app - the bundle ships no interpreter, so sys.executable is the GUI
itself - and it gives no per-item progress and no way to cancel. So the GUI
imports each stage and calls the run() function pipeline_api.py defines.

STAGES below deliberately mirrors run_pipeline.STEPS: same names, same order.
tests/test_pipeline_stages.py pins that, because "start from clustering" in
the app and `--from clustering` on the command line must mean the same thing.

The benchmark is the one stage not driven through a stage run(): the GUI has
driven it through helper_functions.run_benchmarks and gui/core/runner.py since
before this existed, and it needs a full RunSpec rather than a dimension list.
"""
import importlib
import importlib.util
import inspect
import json
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from gui.qt import QtCore, Signal

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config


@dataclass(frozen=True)
class Stage:
    """One pipeline step, and what running it would overwrite."""

    name: str            # must match a run_pipeline.py step name
    label: str           # short label for the UI
    description: str
    folder: str          # the numbered stage folder, e.g. "03_cluster"
    module: str          # module name inside it, e.g. "cluster_trajectories"
    writes: tuple = ()   # config attribute names this stage rewrites
    needs_run_spec: bool = False
    slow: bool = False   # worth calling out in the confirmation dialog

    def targets(self):
        """The directories this stage overwrites, read at call time so a
        rebound results root (gui/core/results_root.py) is reflected."""
        return [getattr(config, name) for name in self.writes
                if getattr(config, name, None)]


STAGES = [
    Stage("benchmark", "Benchmark",
          "Run the optimizers against the benchmark suite. Configured on the "
          "Setup tab; this is the only stage that produces new raw data.",
          "01_optimize", "run_benchmarks",
          writes=("OUTPUTS_DIR",), needs_run_spec=True, slow=True),
    Stage("harvest_results", "Harvest results",
          "Scan the g_best trajectories into outputs/dim_{d}/results.csv.",
          "01_optimize", "harvest_results",
          writes=("OUTPUTS_DIR",)),
    Stage("preprocess", "Preprocess",
          "Reshape the raw population trajectories into the clustering input.",
          "02_preprocess", "preprocess_data",
          writes=("PROCESSED_DIR",), slow=True),
    Stage("clustering", "Clustering",
          "KMeans-cluster each problem's trajectories. Everything downstream "
          "depends on this, and it is the stage the clustering seed affects.",
          "03_cluster", "cluster_trajectories",
          writes=("CLUSTERING_LATEST_DIR",), slow=True),
    Stage("aggregate_cosine", "Aggregate cosine",
          "Aggregate cosine similarity between algorithm trajectory vectors.",
          "03_cluster", "cluster_similarity",
          writes=("CLUSTERING_LATEST_DIR",)),
    Stage("entropy_calc", "Entropy",
          "Shannon entropy of cluster occupancy per algorithm/run/iteration.",
          "04_metrics", "entropy",
          writes=("ENTROPY_DATA_DIR",), slow=True),
    Stage("entropy_pairwise", "Entropy (pairwise)",
          "Mean per-iteration entropy difference between algorithm pairs.",
          "04_metrics", "entropy_pairwise",
          writes=("METRICS_DIR",)),
    Stage("cosine_pairwise", "Cosine (pairwise)",
          "Global cosine distance between algorithm pairs.",
          "04_metrics", "cosine_pairwise",
          writes=("METRICS_DIR",)),
    Stage("cosine_columns_pairwise", "Cosine columns (pairwise)",
          "Per-cluster-column cosine distance. The slowest metric stage.",
          "04_metrics", "cosine_columns_pairwise",
          writes=("METRICS_DIR",), slow=True),
    Stage("exploration_pairwise", "Exploration (pairwise)",
          "Difference in exploration/exploitation balance. Needs the "
          "benchmark to have saved diversity.",
          "04_metrics", "exploration_pairwise",
          writes=("METRICS_DIR",)),
    Stage("solutions_pairwise", "Solutions (pairwise)",
          "Difference in final solution location and fitness.",
          "04_metrics", "solutions_pairwise",
          writes=("METRICS_DIR",)),
    Stage("merge", "Merge metrics",
          "Outer-join every metric into merged_dim_{d}.csv.",
          "05_analysis", "merge_metrics",
          writes=("MERGED_DIR",)),
    Stage("build_scalars", "Build scalars",
          "Per-algorithm scalar table (a different shape from the pairwise "
          "metrics) that the regression view consumes.",
          "05_analysis", "build_scalars",
          writes=("SCALARS_CSV",)),
    Stage("spearman", "Spearman",
          "Correlate the metrics against each other, and draw the heatmap.",
          "05_analysis", "spearman",
          writes=("MERGED_DIR", "FIGURES_SPEARMAN_DIR")),
]

STAGE_NAMES = [stage.name for stage in STAGES]
BY_NAME = {stage.name: stage for stage in STAGES}


def stages_from(name):
    """Every stage from `name` onward - the GUI's version of --from."""
    if name not in BY_NAME:
        raise KeyError(f"unknown stage: {name}")
    return STAGES[STAGE_NAMES.index(name):]


def load_stage(stage):
    """Import a stage module and hand back its run().

    Imported on demand, never at GUI startup: cluster_trajectories alone
    pulls in mealpy, yellowbrick and kneed, which costs seconds. The numbered
    folders are not importable packages, so the folder goes on sys.path first
    - the same trick the notebooks and gui/viz/registry.py use.
    """
    folder = str(REPO_ROOT / stage.folder)
    if folder not in sys.path:
        sys.path.insert(0, folder)
    module = importlib.import_module(stage.module)
    runner = getattr(module, "run", None)
    if not callable(runner):
        raise AttributeError(
            f"{stage.module}.run() is missing - see pipeline_api.py for the "
            f"contract every stage implements"
        )
    return runner


@dataclass
class PipelineSpec:
    """What to run. Deliberately not workspace.RunSpec, which is benchmark-
    shaped (it carries algorithms/seeds and a total_runs derived from them)."""

    stages: list = field(default_factory=lambda: list(STAGE_NAMES))
    dimensions: list = None          # None means config.DIMENSIONS
    method: str = "kmeans"           # clustering method
    run_spec: object = None          # a workspace.RunSpec, only for benchmark

    def selected(self):
        return [BY_NAME[name] for name in self.stages if name in BY_NAME]

    def runnable(self):
        """Stages that can actually run, given what was supplied."""
        return [s for s in self.selected()
                if not (s.needs_run_spec and self.run_spec is None)]

    def skipped(self):
        return [s for s in self.selected() if s not in self.runnable()]

    def targets(self):
        """Every directory this run would overwrite, de-duplicated."""
        seen = []
        for stage in self.runnable():
            for target in stage.targets():
                if target not in seen:
                    seen.append(target)
        return seen

    def describe(self):
        names = [s.label for s in self.runnable()]
        dims = self.dimensions or config.discover_dimensions(config.OUTPUTS_DIR)
        return (f"{len(names)} stage(s): {', '.join(names)}\n"
                f"dimensions {', '.join(str(d) for d in dims)}")


class PipelineRunner(QtCore.QThread):
    """Runs the selected stages in order, off the UI thread.

    Same contract as gui/core/runner.py's BenchmarkRunner: the worker only
    emits signals, every widget touch happens on the UI thread.
    """

    stageStarted = Signal(dict)   # {"name", "label", "index", "total"}
    progressed = Signal(dict)     # {"done", "total", "label", "stage"}
    message = Signal(str)
    completed = Signal(dict)
    failed = Signal(str)

    def __init__(self, spec, parent=None):
        super().__init__(parent)
        self.spec = spec
        self._cancel = threading.Event()
        self._results = []

    def cancel(self):
        """Ask the run to stop at the next item boundary."""
        self._cancel.set()
        self.message.emit("Cancelling - stopping at the next item...")

    @property
    def cancelled(self):
        return self._cancel.is_set()

    def run(self):
        started = time.time()
        stages = self.spec.runnable()
        if not stages:
            self.failed.emit("Nothing to run - no stage was selected.")
            return

        for skipped in self.spec.skipped():
            self.message.emit(
                f"Skipping {skipped.label}: it needs a benchmark configured "
                f"on the Setup tab."
            )

        for index, stage in enumerate(stages):
            if self._cancel.is_set():
                break
            self.stageStarted.emit({
                "name": stage.name, "label": stage.label,
                "index": index, "total": len(stages),
            })
            self.message.emit(f"[{index + 1}/{len(stages)}] {stage.label}")

            try:
                result = self._run_stage(stage)
            except Exception:
                # Name the stage: "it failed" with a bare traceback is not
                # actionable when fourteen things could have produced it.
                self.failed.emit(
                    f"{stage.label} failed:\n\n{traceback.format_exc(limit=4)}"
                )
                return

            self._results.append(result)
            self.message.emit(f"    {result.describe()}")
            if getattr(result, "cancelled", False):
                break

        # Plain data only. A signal payload crosses into the UI thread and
        # ends up in a manifest, so it holds no StageResult instances - just
        # what a reader and a JSON encoder can both use.
        self.completed.emit({
            "stages": [r.stage for r in self._results],
            "written": sum(r.written for r in self._results),
            "cancelled": self._cancel.is_set() or any(
                getattr(r, "cancelled", False) for r in self._results
            ),
            "seconds": time.time() - started,
            "per_stage": [
                {"stage": r.stage, "written": r.written,
                 "cancelled": bool(getattr(r, "cancelled", False))}
                for r in self._results
            ],
        })

    # -- one stage -------------------------------------------------------

    def _run_stage(self, stage):
        def on_progress(done, total, label):
            self.progressed.emit({
                "done": done, "total": total,
                "label": label, "stage": stage.name,
            })

        if stage.needs_run_spec:
            return self._run_benchmark(stage, on_progress)

        runner = load_stage(stage)
        kwargs = {"progress_cb": on_progress, "cancel_event": self._cancel}
        # Only pass what a stage accepts: merge_metrics has no dimensions
        # argument because it discovers them from the filesystem.
        parameters = inspect.signature(runner).parameters
        if "dimensions" in parameters:
            kwargs["dimensions"] = self.spec.dimensions or config.discover_dimensions(
                config.OUTPUTS_DIR)
        if "method" in parameters:
            kwargs["method"] = self.spec.method
        return runner(**kwargs)

    def _run_benchmark(self, stage, on_progress):
        """Delegate to the same function gui/core/runner.py uses."""
        from pipeline_api import StageResult
        import helper_functions as hf
        from gui.core import optimizers as optimizers_core
        from gui.core import problems as problems_core

        spec = self.spec.run_spec
        result = StageResult(stage.name)
        registry = optimizers_core.load_registry()
        chosen = registry.resolve(spec.algorithms)
        source = problems_core.source_from_config(spec.source)

        observer = None
        if isinstance(source, problems_core.BBOBSource):
            import cocoex

            observer = cocoex.Observer("no_observer", "")

        suite = source.build(spec.functions, spec.instances, spec.dimensions)
        done = {"n": 0}

        def record_cb(record):
            done["n"] += 1
            on_progress(done["n"], spec.total_runs,
                        f"{record.get('algorithm', '?')} "
                        f"{record.get('problem', '?')}")

        for seed in spec.seeds:
            if self._cancel.is_set():
                result.cancelled = True
                break
            hf.run_benchmarks(
                suite, observer, chosen, config.OUTPUTS_DIR,
                seed=seed, epoch_per_dim=spec.epoch_per_dim,
                pop_size=spec.pop_size, only_best=spec.only_best,
                save_diversity=spec.save_diversity,
                progress_cb=record_cb, cancel_event=self._cancel,
            )
        result.written = done["n"]
        return result


# -- the record a run leaves behind --------------------------------------

MANIFEST_NAME = "pipeline_run.json"


def write_manifest(spec, summary, root=None):
    """Record what a pipeline run did, next to the data it wrote.

    Separate from workspace.write_manifest, which takes a RunSpec and calls
    asdict() on it. The clustering seed is the field that matters most here:
    results produced before it existed are not reproducible, so a tree with
    no manifest and a tree written by a seeded run are worth telling apart.
    """
    from gui.core import workspace

    root = Path(root or config.REPO_ROOT)
    path = root / MANIFEST_NAME
    payload = {
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "stages_requested": list(spec.stages),
        "stages_completed": list(summary.get("stages", [])),
        "cancelled": bool(summary.get("cancelled")),
        "seconds": round(float(summary.get("seconds", 0.0)), 1),
        "files_written": int(summary.get("written", 0)),
        "dimensions": list(spec.dimensions or config.DIMENSIONS),
        "clustering_method": spec.method,
        "clustering_seed": config.CLUSTERING_SEED,
        "versions": workspace._versions(),
    }
    try:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    except OSError:
        return None   # a manifest is a record, never a reason to fail a run
    return path
