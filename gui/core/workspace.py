"""Where GUI runs write, and what they record about themselves.

Runs go to a sandbox workspace by default, never to the repo's outputs/.
That is not paranoia: outputs/ holds 11 GB of results that are not in git and
took months to compute, and run_benchmarks writes into it by algorithm and
problem id, so a GUI run of an algorithm already present would overwrite real
trajectories in place. Writing to the real tree stays possible, but only as a
deliberate, separately confirmed choice.

Every run directory carries a manifest.json describing exactly what produced
it. That is what makes a GUI run reproducible by someone other than the
person who clicked the button.
"""
import json
import os
import platform
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config

WORKSPACE_ENV_VAR = "GUI_WORKSPACE"
DEFAULT_WORKSPACE = Path(config.REPO_ROOT) / "gui_runs"

# Name used for the workspace when the app is running from a frozen bundle,
# where the "repo root" is inside the (possibly read-only) application dir.
FROZEN_WORKSPACE_NAME = "OptimizerTrajectoryExplorer"


def is_frozen():
    """True when running from a PyInstaller bundle."""
    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


def workspace_root():
    """Where GUI runs are written.

    In a checkout this is gui_runs/ beside the code. In a packaged app the
    bundle directory is not a sensible - or necessarily writable - place to
    put results, so runs go to the user's home instead. GUI_WORKSPACE
    overrides both.
    """
    override = os.environ.get(WORKSPACE_ENV_VAR)
    if override:
        return Path(override)
    if is_frozen():
        return Path.home() / FROZEN_WORKSPACE_NAME / "runs"
    return DEFAULT_WORKSPACE


@dataclass
class RunSpec:
    """A benchmark sweep the user has asked for."""

    algorithms: list = field(default_factory=list)
    functions: list = field(default_factory=list)
    instances: list = field(default_factory=lambda: [1])
    dimensions: list = field(default_factory=lambda: [2])
    seeds: list = field(default_factory=lambda: [1])
    epoch_per_dim: int = 10
    pop_size: int = 50
    save_diversity: bool = True
    only_best: bool = False
    # Which benchmark source the functions come from. Stored in the manifest,
    # since the run directories name functions by integer id only.
    source: dict = field(default_factory=lambda: {"kind": "bbob"})
    function_names: dict = field(default_factory=dict)
    # How many (function, instance, dimension) problems the selection really
    # yields. Not always the product: an opfunu function fixed at 2-D
    # contributes only at dimension 2, so the product would overstate it.
    # None falls back to the product.
    problem_count: int = None

    @property
    def problems(self):
        if self.problem_count is not None:
            return self.problem_count
        return len(self.functions) * len(self.instances) * len(self.dimensions)

    @property
    def total_runs(self):
        return len(self.algorithms) * self.problems * len(self.seeds)

    def estimate_seconds(self):
        """Rough wall-clock estimate.

        Cost per run scales with epoch = epoch_per_dim * dimension and with
        pop_size, since that is how many objective evaluations happen.
        Calibrated against this machine; the run panel replaces it with a
        measured rate once a few runs have finished.
        """
        if not self.dimensions:
            return 0.0
        mean_dimension = sum(self.dimensions) / len(self.dimensions)
        evaluations = self.epoch_per_dim * mean_dimension * self.pop_size
        per_run = evaluations * 1.1e-4
        return self.total_runs * max(per_run, 0.02)

    @property
    def source_kind(self):
        return (self.source or {}).get("kind", "bbob")

    def describe(self):
        naive = len(self.functions) * len(self.instances) * len(self.dimensions)
        text = (
            f"{len(self.algorithms)} algorithms x {len(self.functions)} functions "
            f"x {len(self.instances)} instances x {len(self.dimensions)} dimensions "
            f"x {len(self.seeds)} seeds"
        )
        if self.problem_count is not None and self.problem_count != naive:
            # Say so rather than letting the arithmetic look wrong.
            text += (
                f"\n{self.problems} problems, not {naive}: some functions are "
                f"fixed at one dimension and are skipped at the others"
            )
        return text

    def validate(self):
        """Human-readable reasons this cannot be run, if any."""
        problems = []
        if not self.algorithms:
            problems.append("no algorithms selected")
        if not self.functions:
            problems.append("no functions selected")
        if not self.instances:
            problems.append("no instances selected")
        if not self.dimensions:
            problems.append("no dimensions selected")
        if not self.seeds:
            problems.append("no seeds selected")
        if self.pop_size < 2:
            problems.append("population size must be at least 2")
        if self.epoch_per_dim < 1:
            problems.append("epochs per dimension must be at least 1")
        return problems


def format_duration(seconds):
    if seconds < 1:
        return "under a second"
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} hours"


def create_run_dir(spec, root=None, label=None):
    """A fresh timestamped directory for one run, with its manifest."""
    root = Path(root) if root is not None else workspace_root()
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    name = f"{stamp}_{label}" if label else stamp
    run_dir = root / name
    counter = 2
    while run_dir.exists():
        run_dir = root / f"{name}_{counter}"
        counter += 1
    (run_dir / "outputs").mkdir(parents=True)
    write_manifest(run_dir, spec, status="started")
    return run_dir


def _versions():
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for module in ("mealpy", "cocoex", "numpy", "pandas", "sklearn"):
        try:
            versions[module] = __import__(module).__version__
        except Exception:
            versions[module] = None
    return versions


def write_manifest(run_dir, spec, **extra):
    """Record what produced this run, so it can be reproduced or explained."""
    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "spec": asdict(spec),
        "total_runs": spec.total_runs,
        "versions": _versions(),
        "clustering_seed": config.CLUSTERING_SEED,
    }
    manifest.update(extra)

    path = Path(run_dir) / "manifest.json"
    if path.exists():
        existing = json.loads(path.read_text())
        existing.update(manifest)
        manifest = existing
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return path


def read_manifest(run_dir):
    path = Path(run_dir) / "manifest.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError):
        return None


def list_runs(root=None):
    """Previous GUI runs, newest first."""
    root = Path(root) if root is not None else workspace_root()
    if not root.is_dir():
        return []
    runs = []
    for directory in sorted(root.iterdir(), reverse=True):
        if not directory.is_dir():
            continue
        manifest = read_manifest(directory)
        if manifest is not None:
            runs.append({"path": directory, "manifest": manifest})
    return runs
