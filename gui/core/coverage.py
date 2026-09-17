"""What data exists, what a visualization needs, and how to make up the gap.

The catalog deliberately lets the user ask for dimensions and algorithms that
have never been computed - hiding them would hide most of what the tool can
do. This module answers "can that actually be drawn?" and, when it cannot,
says precisely what is missing and which pipeline steps would produce it.

Everything here is read-only and cheap: it inspects the small summary tables
(data/entropy/entropy_dim_{d}.csv and friends), never the 17 GB of raw runs.
"""
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config

# The dimensions run_benchmarks.py will accept (its -d choices). Wider than
# config.DIMENSIONS, which is only what this thesis actually swept.
BENCHMARK_DIMENSIONS = [2, 3, 5, 10, 20, 40]


@dataclass
class Gap:
    """One missing thing, and what it would take to produce it."""

    what: str
    dimension: int = None
    algorithms: tuple = ()
    functions: tuple = ()

    def describe(self):
        parts = [self.what]
        if self.algorithms:
            shown = ", ".join(sorted(self.algorithms)[:6])
            if len(self.algorithms) > 6:
                shown += f", +{len(self.algorithms) - 6} more"
            parts.append(f"algorithms: {shown}")
        if self.functions:
            parts.append(f"functions: {_compact(self.functions)}")
        return "  -  ".join(parts)


@dataclass
class Coverage:
    """Whether a visualization can be drawn, and what is missing if not."""

    gaps: list = field(default_factory=list)
    recipe: list = field(default_factory=list)
    # The same steps as `recipe`, but as pipeline stage names the Process tab
    # can actually run. `recipe` stays the human-readable command list, for
    # people who would rather run it themselves.
    stages: tuple = ()
    caveat: str = ""
    estimate: str = ""
    # What a run would have to cover to close these gaps. Used to prefill the
    # setup form when the user asks to generate the missing data.
    wanted_algorithms: tuple = ()
    wanted_functions: tuple = ()
    wanted_dimensions: tuple = ()

    @property
    def ok(self):
        return not self.gaps

    def summary(self):
        return "\n".join(gap.describe() for gap in self.gaps)


def _compact(numbers):
    """[1,2,3,7] -> '1-3, 7'"""
    numbers = sorted(set(int(n) for n in numbers))
    if not numbers:
        return ""
    runs, start, previous = [], numbers[0], numbers[0]
    for value in numbers[1:]:
        if value == previous + 1:
            previous = value
            continue
        runs.append((start, previous))
        start = previous = value
    runs.append((start, previous))
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)


# --------------------------------------------------------------------------
# what exists
# --------------------------------------------------------------------------

def _entropy_path(dimension):
    return Path(config.ENTROPY_DATA_DIR) / f"entropy_dim_{dimension}.csv"


@lru_cache(maxsize=16)
def entropy_contents(dimension):
    """(algorithms, functions) present in the entropy table for a dimension."""
    path = _entropy_path(dimension)
    if not path.exists():
        return frozenset(), frozenset()
    frame = pd.read_csv(path, usecols=["algorithm", "function_class"])
    return (
        frozenset(frame["algorithm"].unique()),
        frozenset(int(f) for f in frame["function_class"].unique()),
    )


def invalidate():
    """Forget what was on disk - call after a run produces new data."""
    entropy_contents.cache_clear()


# --------------------------------------------------------------------------
# recipes
# --------------------------------------------------------------------------

def _entropy_recipe(dimension, algorithms, functions):
    """The pipeline steps that turn raw runs into an entropy table."""
    algorithm_flag = " ".join(sorted(algorithms)) if algorithms else "<algorithms>"
    function_flag = " ".join(str(f) for f in sorted(functions)) if functions else "<functions>"
    return [
        f"python 01_optimize/run_benchmarks.py -a {algorithm_flag} "
        f"-d {dimension} -f {function_flag} -e",
        "python 02_preprocess/preprocess_data.py",
        "python 03_cluster/cluster_trajectories.py -c kmeans",
        "python 04_metrics/entropy.py",
    ]


CLUSTERING_CAVEAT = (
    "Adding an algorithm re-runs KMeans on the combined trajectories for that "
    "problem, so cluster assignments - and therefore the entropy and cosine "
    "metrics of every OTHER algorithm on it - will change. A benchmark run "
    "from this app writes to a separate workspace, but running the clustering "
    "and metric stages overwrites data/ and metrics_data/ in place."
)


# Pipeline stage names, kept here rather than imported from gui.core.pipeline:
# that module imports Qt, and this one is used by plain unit tests.
ENTROPY_STAGES = ("benchmark", "preprocess", "clustering", "entropy_calc")

#: The pipeline order, mirrored. gui/core/pipeline.py owns the real table and
#: a test pins the two against run_pipeline.py.
_PIPELINE_ORDER = (
    "benchmark", "harvest_results", "preprocess", "clustering",
    "aggregate_cosine", "entropy_calc", "entropy_pairwise", "cosine_pairwise",
    "cosine_columns_pairwise", "exploration_pairwise", "solutions_pairwise",
    "merge", "build_scalars", "spearman",
)


def _stages_from(name):
    return _PIPELINE_ORDER[_PIPELINE_ORDER.index(name):]


def _merge_stages(left, right):
    """Union of two stage tuples, kept in pipeline order."""
    wanted = set(left) | set(right)
    return tuple(n for n in _PIPELINE_ORDER if n in wanted)


def _estimate(n_algorithms, n_functions, dimension, instances=None, seeds=None):
    instances = instances if instances is not None else len(config.INSTANCES)
    seeds = seeds if seeds is not None else len(config.SEEDS)
    runs = max(n_algorithms, 1) * max(n_functions, 1) * instances * seeds
    # Measured on this machine: a dim-2 run (20 epochs x 50 agents) takes
    # roughly a second, and cost scales with epoch = 10 * dimension.
    seconds = runs * dimension * 0.5
    return f"{runs:,} runs, roughly {_duration(seconds)} plus clustering"


def _duration(seconds):
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} hours"


# --------------------------------------------------------------------------
# per-visualization checks
# --------------------------------------------------------------------------

def check_entropy(dimension, algorithms, functions):
    """Can the entropy plots be drawn for this selection?"""
    coverage = Coverage()
    if dimension is None:
        coverage.gaps.append(Gap(what="No dimension selected"))
        return coverage

    have_algorithms, have_functions = entropy_contents(dimension)
    algorithms = list(algorithms or [])
    functions = [int(f) for f in (functions or [])]

    if not have_algorithms:
        coverage.gaps.append(
            Gap(
                what=f"No entropy data at all for dimension {dimension}",
                dimension=dimension,
                algorithms=tuple(algorithms),
                functions=tuple(functions),
            )
        )
    else:
        missing_algorithms = [a for a in algorithms if a not in have_algorithms]
        missing_functions = [f for f in functions if f not in have_functions]
        if missing_algorithms:
            coverage.gaps.append(
                Gap(
                    what=f"Not benchmarked at dimension {dimension}",
                    dimension=dimension,
                    algorithms=tuple(missing_algorithms),
                )
            )
        if missing_functions:
            coverage.gaps.append(
                Gap(
                    what=f"No data for these functions at dimension {dimension}",
                    dimension=dimension,
                    functions=tuple(missing_functions),
                )
            )

    if coverage.gaps:
        wanted_algorithms = {a for gap in coverage.gaps for a in gap.algorithms} or set(algorithms)
        wanted_functions = {f for gap in coverage.gaps for f in gap.functions} or set(functions)
        coverage.recipe = _entropy_recipe(dimension, wanted_algorithms, wanted_functions)
        coverage.stages = ENTROPY_STAGES
        coverage.caveat = CLUSTERING_CAVEAT
        coverage.estimate = _estimate(
            len(wanted_algorithms), len(wanted_functions), dimension
        )
        coverage.wanted_algorithms = tuple(sorted(wanted_algorithms))
        coverage.wanted_functions = tuple(sorted(wanted_functions))
        coverage.wanted_dimensions = (dimension,)
    return coverage


def check_entropy_multi(dimensions, algorithms, functions):
    """Same, across several dimensions at once (the overlay plot)."""
    combined = Coverage()
    for dimension in dimensions or []:
        single = check_entropy(dimension, algorithms, functions)
        combined.gaps.extend(single.gaps)
        combined.recipe.extend(r for r in single.recipe if r not in combined.recipe)
        combined.stages = _merge_stages(combined.stages, single.stages)
        combined.caveat = combined.caveat or single.caveat
        combined.wanted_algorithms = tuple(
            sorted(set(combined.wanted_algorithms) | set(single.wanted_algorithms))
        )
        combined.wanted_functions = tuple(
            sorted(set(combined.wanted_functions) | set(single.wanted_functions))
        )
        combined.wanted_dimensions = tuple(
            sorted(set(combined.wanted_dimensions) | set(single.wanted_dimensions))
        )
    if combined.gaps and not combined.estimate:
        combined.estimate = "see the steps below"
    return combined


def check_file(path, what, recipe, stages=()):
    """A visualization whose input is a single computed file."""
    coverage = Coverage()
    if not Path(path).exists():
        coverage.gaps.append(Gap(what=what))
        coverage.recipe = recipe
        coverage.stages = tuple(stages)
    return coverage


def _benchmark_recipe(dimension, algorithms, functions, instances, extra_flag=""):
    algorithm_flag = " ".join(sorted(algorithms)) if algorithms else "<algorithms>"
    function_flag = " ".join(str(f) for f in sorted(functions)) if functions else "<functions>"
    instance_flag = " ".join(str(i) for i in sorted(instances)) if instances else "1"
    return [
        f"python 01_optimize/run_benchmarks.py -a {algorithm_flag} "
        f"-d {dimension} -f {function_flag} -i {instance_flag}{extra_flag}"
    ]


def check_diversity(dimension, algorithms, function, instance, seeds):
    """The exploration views read per-run diversity_{seed}.csv files, which
    the benchmark writes only when it is run with -e."""
    coverage = Coverage()
    algorithms = list(algorithms or [])
    seeds = list(seeds or [])
    if not algorithms or not seeds:
        coverage.gaps.append(Gap(what="Select at least one algorithm and one run"))
        return coverage

    missing = []
    for algorithm in algorithms:
        directory = (
            Path(config.OUTPUTS_DIR) / f"dim_{dimension}" / algorithm
            / f"{function}_{instance}"
        )
        if not any((directory / f"diversity_{seed}.csv").is_file() for seed in seeds):
            missing.append(algorithm)

    if missing:
        coverage.gaps.append(
            Gap(
                what=(
                    f"No diversity data for F{function}_I{instance} at "
                    f"dimension {dimension}"
                ),
                dimension=dimension,
                algorithms=tuple(missing),
                functions=(int(function),),
            )
        )
        coverage.recipe = _benchmark_recipe(
            dimension, missing, [function], [instance], extra_flag=" -e"
        )
        coverage.caveat = (
            "Diversity, exploration and exploitation are only recorded when "
            "the benchmark runs with -e. A run without it produces "
            "trajectories but no diversity file."
        )
        coverage.estimate = _estimate(len(missing), 1, dimension, instances=1,
                                      seeds=len(seeds))
        coverage.wanted_algorithms = tuple(sorted(missing))
        coverage.wanted_functions = (int(function),)
        coverage.wanted_dimensions = (dimension,)
    return coverage


def check_results(dimension, function, instance, algorithms):
    """The final-solution views read outputs/dim_{d}/results.csv."""
    coverage = Coverage()
    path = Path(config.OUTPUTS_DIR) / f"dim_{dimension}" / "results.csv"
    if not path.is_file():
        coverage.gaps.append(
            Gap(what=f"outputs/dim_{dimension}/results.csv does not exist")
        )
        coverage.recipe = _benchmark_recipe(
            dimension, algorithms, [function], [instance]
        ) + ["python 01_optimize/harvest_results.py"]
        coverage.stages = ("benchmark", "harvest_results")
        coverage.wanted_algorithms = tuple(sorted(algorithms or ()))
        coverage.wanted_dimensions = (dimension,)
        coverage.wanted_functions = (int(function),)
        return coverage

    results = pd.read_csv(path, usecols=["algorithm", "problem_id", "instance_id"])
    here = results[
        (results["problem_id"] == function) & (results["instance_id"] == instance)
    ]
    present = set(here["algorithm"].unique())
    missing = [a for a in (algorithms or []) if a not in present]
    if not present:
        coverage.gaps.append(
            Gap(what=f"No results for F{function}_I{instance} at dimension {dimension}",
                dimension=dimension, functions=(int(function),))
        )
    elif missing:
        coverage.gaps.append(
            Gap(what=f"Not benchmarked on F{function}_I{instance} at dimension {dimension}",
                dimension=dimension, algorithms=tuple(missing),
                functions=(int(function),))
        )

    if coverage.gaps:
        wanted = missing or list(algorithms or [])
        coverage.recipe = _benchmark_recipe(
            dimension, wanted, [function], [instance]
        ) + ["python 01_optimize/harvest_results.py"]
        coverage.estimate = _estimate(len(wanted), 1, dimension, instances=1)
        coverage.wanted_algorithms = tuple(sorted(wanted))
        coverage.wanted_functions = (int(function),)
        coverage.wanted_dimensions = (dimension,)
    return coverage


def check_clustering(dimension, function, instance, method="kmeans"):
    """The cluster views read the clustering stage's per-problem output."""
    base = Path(config.clustering_dir(method))
    path = base / "cluster_distributions" / f"dim_{dimension}" / f"F{function}_I{instance}.csv"
    return check_file(
        path,
        f"No clustering output for F{function}_I{instance} at dimension {dimension}",
        [
            "python 02_preprocess/preprocess_data.py",
            f"python 03_cluster/cluster_trajectories.py -c {method}",
        ],
        stages=("preprocess", "clustering"),
    )


def check_similarity(dimension, method="kmeans", statistic="mean"):
    base = Path(config.clustering_dir(method)) / config.SIMILARITY_OUTPUT_SUBDIR
    return check_file(
        base / f"algorithm_{statistic}_similarity_{dimension}D.csv",
        f"No aggregate similarity matrix for dimension {dimension}",
        [f"python 03_cluster/cluster_similarity.py -c {method}"],
        stages=("aggregate_cosine",),
    )


def check_merged(dimension):
    return check_file(
        Path(config.MERGED_DIR) / f"merged_dim_{dimension}.csv",
        f"metrics_data/merged/merged_dim_{dimension}.csv does not exist",
        ["python 05_analysis/merge_metrics.py"],
        stages=("merge",),
    )


def check_spearman(dimension):
    if dimension is None:
        return Coverage(gaps=[Gap(what="No dimension selected")])
    return check_file(
        Path(config.MERGED_DIR) / f"spearman_dim_{dimension}.csv",
        f"No Spearman matrix for dimension {dimension}",
        [
            "python run_pipeline.py --from cosine_pairwise",
            "  (needs every pairwise metric computed first)",
        ],
        stages=_stages_from("cosine_pairwise"),
    )


def check_scalars(dimensions):
    coverage = check_file(
        config.SCALARS_CSV,
        "metrics_data/scalars.csv has not been built",
        ["python 05_analysis/build_scalars.py"],
        stages=("build_scalars",),
    )
    if coverage.ok and dimensions:
        frame = pd.read_csv(config.SCALARS_CSV, usecols=["dim"])
        have = set(int(d) for d in frame["dim"].unique())
        missing = [d for d in dimensions if int(d) not in have]
        if missing:
            coverage.gaps.append(
                Gap(what=f"scalars.csv has no rows for dimension(s) {_compact(missing)}")
            )
            coverage.recipe = [
                "python run_pipeline.py --from benchmark   (for the missing dimension)",
                "python 05_analysis/build_scalars.py",
            ]
            coverage.stages = ("build_scalars",)
    return coverage
