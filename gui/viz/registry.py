"""The catalog of visualizations the GUI can render.

Each entry declares what parameters it needs and how to draw itself; the
panel builds its form from that declaration, so adding a visualization is one
render function plus one Visualization() here.

Tier 1 (this module) wraps the plotting functions that already exist in the
pipeline rather than reimplementing them, so a GUI figure is by construction
the same figure the pipeline produces. The scripts live in directories whose
names start with a digit ("04_metrics"), which are not importable as
packages, so they are loaded by putting the directory on sys.path - the same
trick the notebooks use.
"""
import importlib
import sys
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import config
from gui.core import coverage as coverage_core
from gui.viz import clustering, exploration, metrics, solutions, style
from gui.viz import figure_theme
from gui.viz.capture import render_with

SCALAR_METRICS = ["entropy", "fitness", "exploration", "diversity"]

# Handed to the pipeline plotting functions as their output directory.
# Nothing is ever written there - capture.py disables savefig while a
# figure is being rendered - but some of them call os.makedirs() on it
# first, so it has to be a real, writable path.
_UNUSED_OUTPUT_DIR = tempfile.gettempdir()


@lru_cache(maxsize=None)
def _pipeline_module(folder, name):
    """Import a module from a numbered stage folder."""
    folder_path = str(REPO_ROOT / folder)
    if folder_path not in sys.path:
        sys.path.insert(0, folder_path)
    return importlib.import_module(name)


# --------------------------------------------------------------------------
# parameter + visualization specs
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Parameter:
    name: str
    label: str
    kind: str  # choice | multi | bool
    choices: object = ()
    default: object = None
    help: str = ""

    def resolve_choices(self):
        return list(self.choices() if callable(self.choices) else self.choices)

    def resolve_default(self):
        return self.default() if callable(self.default) else self.default


@dataclass(frozen=True)
class Visualization:
    key: str
    title: str
    description: str
    source: str
    render: object = field(repr=False, default=None)
    parameters: tuple = ()
    requires: object = field(repr=False, default=None)
    coverage: object = field(repr=False, default=None)

    def is_available(self):
        return True if self.requires is None else bool(self.requires())

    def draw(self, params):
        """Render this entry with the app's figure style applied.

        The single point every catalog entry passes through. The entries that
        wrap a pipeline function are already styled inside render_with; the
        ones that build their own figure (exploration, clustering, metrics,
        solutions) are styled here, so no viz module has to remember to.
        """
        with figure_theme.styled():
            figure = self.render(params)
        figure_theme.apply_to(figure)
        return figure_theme.cap_figure_size(figure)

    def check_coverage(self, params):
        """What is missing for this parameter set, if anything."""
        if self.coverage is None:
            return coverage_core.Coverage()
        return self.coverage(params)


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------

def _entropy_path(dim):
    return Path(config.ENTROPY_DATA_DIR) / f"entropy_dim_{dim}.csv"


def available_entropy_dims():
    return [d for d in config.DIMENSIONS if _entropy_path(d).exists()]


def available_spearman_dims():
    return [
        d
        for d in config.DIMENSIONS
        if (Path(config.MERGED_DIR) / f"spearman_dim_{d}.csv").exists()
    ]


@lru_cache(maxsize=8)
def _entropy_table(dim):
    return pd.read_csv(_entropy_path(dim))


@lru_cache(maxsize=8)
def _spearman_matrix(dim):
    return pd.read_csv(Path(config.MERGED_DIR) / f"spearman_dim_{dim}.csv", index_col=0)


@lru_cache(maxsize=1)
def _scalars():
    module = _pipeline_module("05_analysis", "scalar_regression")
    return module.load_scalars(config.SCALARS_CSV)


def _default_dim(dims):
    return dims[0] if dims else None


# --------------------------------------------------------------------------
# render functions
# --------------------------------------------------------------------------

def _render_entropy_curves(params):
    plotting = _pipeline_module("04_metrics", "entropy_plotting")
    table = _entropy_table(params["dimension"])
    subset = table[table["function_class"] == params["function"]].rename(
        columns={"mean_entropy": "entropy"}
    )
    if subset.empty:
        raise ValueError(f"No entropy data for F{params['function']}")
    return render_with(
        plotting.plot_entropy,
        subset,
        function=params["function"],
        instance="all",
        dimension=params["dimension"],
        save=False,
        algorithms=params["algorithms"] or None,
    )


def _require_entropy_rows(table, algorithms, dimension):
    """Fail readably when none of the chosen algorithms has entropy data.

    The pipeline plot functions filter by algorithm and then hand the empty
    frame to seaborn, which answers "Number of rows must be a positive
    integer, not 0" - a matplotlib grid-sizing error shown verbatim to
    someone who simply picked an algorithm that was never benchmarked.
    Guarded here rather than in 04_metrics/, so the thesis scripts are
    untouched; this is the same place the clustermap's own guard lives.
    """
    if not algorithms:
        return
    have = set(table["algorithm"].unique())
    missing = [a for a in algorithms if a not in have]
    if len(missing) == len(algorithms):
        raise ValueError(
            f"No entropy data at dimension {dimension} for "
            f"{', '.join(missing[:4])}{'...' if len(missing) > 4 else ''}. "
            f"Benchmark them first, or pick algorithms that have been run."
        )


def _render_entropy_function_groups(params):
    plotting = _pipeline_module("04_metrics", "entropy_plotting")
    table = _entropy_table(params["dimension"])
    _require_entropy_rows(table, params["algorithms"], params["dimension"])
    return render_with(
        plotting.plot_entropy_by_function_group,
        table,
        _UNUSED_OUTPUT_DIR,  # savefig is disabled while rendering
        plotting.FUNCTION_GROUPS,
        algorithms=params["algorithms"] or None,
    )


def _render_entropy_clustermap(params):
    plotting = _pipeline_module("04_metrics", "entropy_plotting")
    algorithms = params["algorithms"] or None
    if algorithms is not None and len(algorithms) < 2:
        raise ValueError("A clustermap needs at least 2 algorithms")
    return render_with(
        plotting.plot_entropy_clustermap,
        _entropy_table(params["dimension"]),
        _UNUSED_OUTPUT_DIR,
        algorithms=algorithms,
    )


def _render_entropy_overlay(params):
    plotting = _pipeline_module("04_metrics", "entropy_plotting")
    dims = params["dimensions"] or available_entropy_dims()
    functions = params["functions"]
    if not functions:
        raise ValueError("Select at least one function")
    by_dim = {d: _entropy_table(d) for d in dims}
    for dimension, table in by_dim.items():
        _require_entropy_rows(table, params["algorithms"], dimension)
    return render_with(
        plotting.plot_entropy_overlay,
        by_dim,
        [int(f) for f in functions],
        list(dims),
        _UNUSED_OUTPUT_DIR,
        save=False,
        algorithms=params["algorithms"] or None,
    )


def _render_spearman(params):
    spearman = _pipeline_module("05_analysis", "spearman")
    dim = params["dimension"]
    corr = _spearman_matrix(dim)
    return render_with(
        spearman.plot_spearman, corr, dim, list(corr.index), _UNUSED_OUTPUT_DIR
    )


def _render_scalar_regression(params):
    module = _pipeline_module("05_analysis", "scalar_regression")
    x_metric, y_metric = params["x_metric"], params["y_metric"]
    if x_metric == y_metric:
        raise ValueError("Pick two different metrics for the axes")
    aggregated = module.aggregate_over_functions(
        _scalars(),
        [x_metric, y_metric],
        normalise_per_function=params["normalise"],
    )
    dims = [int(d) for d in params["dimensions"]] or None
    return render_with(
        module.facet_by_dim, aggregated, x_metric, y_metric, dims=dims
    )


# --------------------------------------------------------------------------
# the catalog
# --------------------------------------------------------------------------

def _ALGORITHM_CHOICES():
    """Every optimizer mealpy ships, not just the 28 already benchmarked.

    Picking an un-benchmarked one is allowed on purpose: the panel then
    reports what is missing and how to generate it. Safe to call on the UI
    thread - the registry is warmed on a worker at startup.
    """
    from gui.core import optimizers

    return optimizers.load_registry().names


def _DEFAULT_ALGORITHMS():
    return list(style.DEFAULT_PLOT_ALGORITHMS)


def _DIMENSION_CHOICES():
    return list(coverage_core.BENCHMARK_DIMENSIONS)


def _RUN_CHOICES():
    return list(config.SEEDS)


def _PAIR_ALGORITHM_DEFAULT():
    """Enough algorithms for pairwise structure to be visible, few enough to
    read: two DE variants, two swarm, two others."""
    return ["OriginalDE", "SADE", "L_SHADE", "OriginalGWO", "OriginalWOA",
            "AugmentedAEO"]


def _SEED_CHOICES():
    return list(config.SEEDS)


def _DEFAULT_SEEDS():
    return list(config.SEEDS)


def _INSTANCE_CHOICES():
    return list(config.INSTANCES)


def _RAW_ALGORITHM_DEFAULT():
    """A readable handful for the per-problem raw-output views."""
    return list(style.DEFAULT_PLOT_ALGORITHMS[:6])


def _problem_parameters(default_dim=None, algorithms="multi", seeds=True):
    """The parameter set shared by views that read raw per-run output.

    Every parameter here is one the render function actually consumes - a
    form control that changes nothing is worse than no control at all.
    `algorithms` is "multi", "single" or None; `seeds` is only offered by the
    views that read per-run files.
    """
    parameters = [
        Parameter("dimension", "Dimension", "choice",
                  choices=_DIMENSION_CHOICES,
                  default=default_dim or _preferred_dim),
        Parameter("function", "Function", "choice",
                  choices=lambda: list(config.FUNCTIONS), default=23,
                  help="F23 is weakly structured multimodal - a revealing case."),
        Parameter("instance", "Instance", "choice",
                  choices=_INSTANCE_CHOICES, default=1),
    ]
    if algorithms == "multi":
        parameters.append(
            Parameter("algorithms", "Algorithms", "multi",
                      choices=_ALGORITHM_CHOICES, default=_RAW_ALGORITHM_DEFAULT)
        )
    elif algorithms == "single":
        parameters.append(
            Parameter("algorithm", "Algorithm", "choice",
                      choices=_ALGORITHM_CHOICES, default="OriginalGWO")
        )
    if seeds:
        parameters.append(
            Parameter("seeds", "Runs", "multi",
                      choices=_SEED_CHOICES, default=_DEFAULT_SEEDS)
        )
    return tuple(parameters)


def _preferred_dim():
    """Default to a dimension that already has data, if any."""
    computed = available_entropy_dims()
    return computed[0] if computed else coverage_core.BENCHMARK_DIMENSIONS[0]


def catalog():
    """Every visualization, in the order the panel should list them."""
    return [
        Visualization(
            key="entropy_curves",
            title="Entropy over iterations",
            description=(
                "Mean normalized population entropy (H / ln k) per iteration, "
                "one line per algorithm, for a single BBOB function."
            ),
            source="04_metrics/entropy_plotting.py :: plot_entropy",
            parameters=(
                Parameter("dimension", "Dimension", "choice",
                          choices=_DIMENSION_CHOICES, default=_preferred_dim,
                          help="Dimensions without data can be selected; the panel will say what is missing."),
                Parameter("function", "Function", "choice",
                          choices=lambda: list(config.FUNCTIONS), default=1),
                Parameter("algorithms", "Algorithms", "multi",
                          choices=_ALGORITHM_CHOICES, default=_DEFAULT_ALGORITHMS,
                          help="28 lines on one axis is unreadable; a subset is the default."),
            ),
            render=_render_entropy_curves,
            coverage=lambda p: coverage_core.check_entropy(p["dimension"], p["algorithms"], [p["function"]]),
        ),
        Visualization(
            key="entropy_function_groups",
            title="Entropy by BBOB function group",
            description=(
                "The same entropy curves faceted across the five BBOB function "
                "groups, from separable through weakly-structured multimodal."
            ),
            source="04_metrics/entropy_plotting.py :: plot_entropy_by_function_group",
            parameters=(
                Parameter("dimension", "Dimension", "choice",
                          choices=_DIMENSION_CHOICES, default=_preferred_dim,
                          help="Dimensions without data can be selected; the panel will say what is missing."),
                Parameter("algorithms", "Algorithms", "multi",
                          choices=_ALGORITHM_CHOICES, default=_DEFAULT_ALGORITHMS),
            ),
            render=_render_entropy_function_groups,
            coverage=lambda p: coverage_core.check_entropy(p["dimension"], p["algorithms"], config.FUNCTIONS),
        ),
        Visualization(
            key="entropy_clustermap",
            title="Entropy clustermap (algorithm x iteration)",
            description=(
                "Algorithms clustered by the shape of their entropy curve. "
                "Rows are reordered by the dendrogram, columns stay in "
                "iteration order."
            ),
            source="04_metrics/entropy_plotting.py :: plot_entropy_clustermap",
            parameters=(
                Parameter("dimension", "Dimension", "choice",
                          choices=_DIMENSION_CHOICES, default=_preferred_dim,
                          help="Dimensions without data can be selected; the panel will say what is missing."),
                Parameter("algorithms", "Algorithms", "multi",
                          choices=_ALGORITHM_CHOICES, default=_DEFAULT_ALGORITHMS,
                          help="At least two are needed to cluster."),
            ),
            render=_render_entropy_clustermap,
            coverage=lambda p: coverage_core.check_entropy(p["dimension"], p["algorithms"], config.FUNCTIONS),
        ),
        Visualization(
            key="entropy_overlay",
            title="Entropy across dimensions",
            description=(
                "Dimension x algorithm grid, one line per function. Shows "
                "whether an algorithm's exploration profile holds up as the "
                "search space grows. Not reachable from the CLI."
            ),
            source="04_metrics/entropy_plotting.py :: plot_entropy_overlay",
            parameters=(
                Parameter("dimensions", "Dimensions", "multi",
                          choices=_DIMENSION_CHOICES,
                          default=lambda: available_entropy_dims() or [2]),
                Parameter("functions", "Functions", "multi",
                          choices=lambda: list(config.FUNCTIONS), default=[1, 23],
                          help="F1 is separable and unimodal; F23 is weakly structured."),
                Parameter("algorithms", "Algorithms", "multi",
                          choices=_ALGORITHM_CHOICES,
                          default=lambda: style.DEFAULT_PLOT_ALGORITHMS[:4],
                          help="One column per algorithm, so keep this small."),
            ),
            render=_render_entropy_overlay,
            coverage=lambda p: coverage_core.check_entropy_multi(p["dimensions"], p["algorithms"], p["functions"]),
        ),
        Visualization(
            key="spearman_heatmap",
            title="Spearman correlation between metrics",
            description=(
                "How much the six pairwise metrics agree. Off-diagonal values "
                "near 1 mean two metrics carry redundant information - the "
                "central question of the thesis."
            ),
            source="05_analysis/spearman.py :: plot_spearman",
            parameters=(
                Parameter("dimension", "Dimension", "choice",
                          choices=_DIMENSION_CHOICES,
                          default=lambda: _default_dim(available_spearman_dims())
                          or _preferred_dim()),
            ),
            render=_render_spearman,
            coverage=lambda p: coverage_core.check_spearman(p["dimension"]),
        ),
        Visualization(
            key="scalar_regression",
            title="Scalar regression across algorithms",
            description=(
                "One point per algorithm, coloured by mealpy family, with an "
                "OLS fit and error bars across BBOB functions. Influential "
                "points (Cook's D > 4/n) are ringed."
            ),
            source="05_analysis/scalar_regression.py :: facet_by_dim",
            parameters=(
                Parameter("x_metric", "X axis", "choice",
                          choices=SCALAR_METRICS, default="entropy"),
                Parameter("y_metric", "Y axis", "choice",
                          choices=SCALAR_METRICS, default="exploration"),
                Parameter("dimensions", "Dimensions", "multi",
                          choices=lambda: list(config.DIMENSIONS),
                          default=lambda: list(config.DIMENSIONS)),
                Parameter("normalise", "Normalise per function", "bool", default=False,
                          help="z-score each scalar within (dim, function) before averaging."),
            ),
            render=_render_scalar_regression,
            coverage=lambda p: coverage_core.check_scalars(p["dimensions"]),
        ),
        Visualization(
            key="mean_exploration",
            title="Exploration over iterations",
            description=(
                "Mean share of the population still exploring, per iteration. "
                "The dashed line at 50% is the explore/exploit boundary."
            ),
            source="gui/viz/exploration.py (was exploration_plots_preview.ipynb cell 5)",
            parameters=_problem_parameters(default_dim=lambda: 5),
            render=exploration.mean_exploration,
            coverage=lambda p: coverage_core.check_diversity(
                p["dimension"], p["algorithms"], p["function"], p["instance"], p["seeds"]),
        ),
        Visualization(
            key="exploration_vs_exploitation",
            title="Exploration vs exploitation (one algorithm)",
            description=(
                "A single algorithm's explore/exploit balance over time, with "
                "the shaded spread across runs."
            ),
            source="gui/viz/exploration.py (was exploration_plots_preview.ipynb cell 7)",
            parameters=_problem_parameters(default_dim=lambda: 5,
                                           algorithms="single"),
            render=exploration.exploration_vs_exploitation,
            coverage=lambda p: coverage_core.check_diversity(
                p["dimension"], [p["algorithm"]], p["function"], p["instance"], p["seeds"]),
        ),
        Visualization(
            key="exploration_difference",
            title="Exploration difference between algorithms",
            description=(
                "Pairwise mean difference in exploration, clustered. Run 1 is "
                "compared against run 1, so run-to-run variance does not "
                "inflate the difference."
            ),
            source="gui/viz/exploration.py (was exploration_plots_preview.ipynb cell 9)",
            parameters=_problem_parameters(default_dim=lambda: 5),
            render=exploration.exploration_difference_clustermap,
            coverage=lambda p: coverage_core.check_diversity(
                p["dimension"], p["algorithms"], p["function"], p["instance"], p["seeds"]),
        ),
        Visualization(
            key="raw_diversity",
            title="Raw population diversity",
            description=(
                "Un-normalised diversity per iteration - the quantity the "
                "exploration percentage is derived from."
            ),
            source="gui/viz/exploration.py (was exploration_plots_preview.ipynb cell 11)",
            parameters=_problem_parameters(default_dim=lambda: 5),
            render=exploration.raw_diversity,
            coverage=lambda p: coverage_core.check_diversity(
                p["dimension"], p["algorithms"], p["function"], p["instance"], p["seeds"]),
        ),
        Visualization(
            key="precision_distribution",
            title="Final precision per algorithm",
            description=(
                "How close each algorithm got, across runs, on a log axis and "
                "ordered best first. Precision is f - f*, using the true "
                "optimum recovered from COCO."
            ),
            source="gui/viz/solutions.py (was location_fitness_preview.ipynb cell 7, inline only)",
            parameters=_problem_parameters(seeds=False),
            render=solutions.precision_distribution,
            coverage=lambda p: coverage_core.check_results(
                p["dimension"], p["function"], p["instance"], p["algorithms"]),
        ),
        Visualization(
            key="precision_difference",
            title="Pairwise difference in final quality",
            description=(
                "Mean absolute difference in log10 precision between every "
                "pair of algorithms, seed-paired."
            ),
            source="gui/viz/solutions.py (was location_fitness_preview.ipynb cell 9, inline only)",
            parameters=_problem_parameters(seeds=False),
            render=solutions.precision_difference,
            coverage=lambda p: coverage_core.check_results(
                p["dimension"], p["function"], p["instance"], p["algorithms"]),
        ),
        Visualization(
            key="location_difference",
            title="Pairwise distance between final solutions",
            description=(
                "How far apart algorithms finish in the search space. Two "
                "algorithms can score alike and still end up nowhere near "
                "each other - which is the point of keeping location and "
                "quality as separate metrics."
            ),
            source="gui/viz/solutions.py (was location_fitness_preview.ipynb cell 11, inline only)",
            parameters=_problem_parameters(seeds=False),
            render=solutions.location_difference,
            coverage=lambda p: coverage_core.check_results(
                p["dimension"], p["function"], p["instance"], p["algorithms"]),
        ),
        Visualization(
            key="final_positions",
            title="Final solutions in the search space (2-D)",
            description=(
                "Every run's final best point plotted in the real search "
                "space, with the true optimum starred. Dimension 2 only."
            ),
            source="gui/viz/solutions.py (was location_fitness_preview.ipynb cell 13, inline only)",
            parameters=_problem_parameters(default_dim=lambda: 2, seeds=False),
            render=solutions.final_positions,
            coverage=lambda p: coverage_core.check_results(
                p["dimension"], p["function"], p["instance"], p["algorithms"]),
        ),
        Visualization(
            key="cluster_occupancy",
            title="Cluster occupancy (one algorithm)",
            description=(
                "How many agents sit in each cluster at each iteration - the "
                "raw quantity every entropy and cosine metric is computed "
                "from. Columns are labelled with the cluster's coordinates."
            ),
            source="gui/viz/clustering.py (was 3_cluster_analysis.ipynb cell 2)",
            parameters=(
                Parameter("dimension", "Dimension", "choice",
                          choices=_DIMENSION_CHOICES, default=lambda: 2),
                Parameter("function", "Function", "choice",
                          choices=lambda: list(config.FUNCTIONS), default=1),
                Parameter("instance", "Instance", "choice",
                          choices=_INSTANCE_CHOICES, default=1),
                Parameter("algorithm", "Algorithm", "choice",
                          choices=_ALGORITHM_CHOICES, default="OriginalDE"),
                Parameter("run", "Run", "choice", choices=_RUN_CHOICES, default=1),
                Parameter("population", "Population size", "choice",
                          choices=[10, 20, 30, 50, 100], default=50,
                          help="Sets the colour scale's upper bound."),
                Parameter("show_fitness", "Show best fitness per cluster", "bool",
                          default=False,
                          help="Adds a second panel; reads the parquet, so slower."),
            ),
            render=clustering.cluster_occupancy,
            coverage=lambda p: coverage_core.check_clustering(
                p["dimension"], p["function"], p["instance"]),
        ),
        Visualization(
            key="cluster_occupancy_compare",
            title="Cluster occupancy (compare algorithms)",
            description=(
                "The same occupancy view stacked for several algorithms on "
                "one problem and run, which is what makes their different "
                "search behaviour visible side by side."
            ),
            source="gui/viz/clustering.py (was 3_cluster_analysis.ipynb cell 3)",
            parameters=(
                Parameter("dimension", "Dimension", "choice",
                          choices=_DIMENSION_CHOICES, default=lambda: 2),
                Parameter("function", "Function", "choice",
                          choices=lambda: list(config.FUNCTIONS), default=1),
                Parameter("instance", "Instance", "choice",
                          choices=_INSTANCE_CHOICES, default=1),
                Parameter("algorithms", "Algorithms", "multi",
                          choices=_ALGORITHM_CHOICES,
                          default=lambda: ["OriginalDE", "OriginalGWO"],
                          help="At most 6 - one tall row each."),
                Parameter("run", "Run", "choice", choices=_RUN_CHOICES, default=1),
                Parameter("population", "Population size", "choice",
                          choices=[10, 20, 30, 50, 100], default=50),
            ),
            render=clustering.cluster_occupancy_compare,
            coverage=lambda p: coverage_core.check_clustering(
                p["dimension"], p["function"], p["instance"]),
        ),
        Visualization(
            key="algorithm_similarity",
            title="Algorithm similarity clustermap",
            description=(
                "Algorithms clustered by aggregate trajectory similarity, "
                "with leaves coloured by mealpy family - so you can see "
                "whether behaviour follows lineage or cuts across it."
            ),
            source="gui/viz/clustering.py (was 3_cluster_analysis.ipynb cells 9/10/14)",
            parameters=(
                Parameter("dimension", "Dimension", "choice",
                          choices=_DIMENSION_CHOICES, default=_preferred_dim),
                Parameter("statistic", "Statistic", "choice",
                          choices=["mean", "median"], default="mean"),
                Parameter("algorithms", "Algorithms", "multi",
                          choices=_ALGORITHM_CHOICES,
                          default=lambda: list(config.ALGORITHMS_OF_INTEREST),
                          help="All 28 is the informative view here."),
            ),
            render=clustering.algorithm_similarity,
            coverage=lambda p: coverage_core.check_similarity(
                p["dimension"], statistic=p["statistic"]),
        ),
        Visualization(
            key="metric_table",
            title="Metric values per algorithm pair",
            description=(
                "Every selected pair against all six metrics on a shared "
                "0-1 scale. Rows that are uniformly dark or light are pairs "
                "the metrics agree on; mixed rows are where they disagree."
            ),
            source="gui/viz/metrics.py (was pari_algoritmov_metrike_heatmap.ipynb cell 7)",
            parameters=(
                Parameter("dimension", "Dimension", "choice",
                          choices=_DIMENSION_CHOICES, default=_preferred_dim),
                Parameter("algorithms", "Algorithms", "multi",
                          choices=_ALGORITHM_CHOICES,
                          default=_PAIR_ALGORITHM_DEFAULT,
                          help="Every pair becomes a row, so this grows fast."),
            ),
            render=metrics.metric_table,
            coverage=lambda p: coverage_core.check_merged(p["dimension"]),
        ),
        Visualization(
            key="metric_dendrograms",
            title="Clustering induced by each metric",
            description=(
                "One dendrogram per metric. If two metrics were redundant "
                "they would produce the same tree - where the trees differ "
                "is where a metric adds information."
            ),
            source="gui/viz/metrics.py (was pari_algoritmov_metrike_heatmap.ipynb cell 9)",
            parameters=(
                Parameter("dimension", "Dimension", "choice",
                          choices=_DIMENSION_CHOICES, default=_preferred_dim),
                Parameter("algorithms", "Algorithms", "multi",
                          choices=_ALGORITHM_CHOICES,
                          default=_PAIR_ALGORITHM_DEFAULT),
            ),
            render=metrics.metric_dendrograms,
            coverage=lambda p: coverage_core.check_merged(p["dimension"]),
        ),
        Visualization(
            key="metric_added_value",
            title="Added value over the reference metric",
            description=(
                "Takes the pairs the ClustOpt cosine distance calls most and "
                "least similar, and shows what the other five metrics say "
                "about them. Pairs are picked from the data, not hardcoded."
            ),
            source="gui/viz/metrics.py (was scratch/slika_dodana_vrednost.py)",
            parameters=(
                Parameter("dimension", "Dimension", "choice",
                          choices=_DIMENSION_CHOICES, default=_preferred_dim),
                Parameter("algorithms", "Algorithms", "multi",
                          choices=_ALGORITHM_CHOICES,
                          default=_PAIR_ALGORITHM_DEFAULT,
                          help="At least 4, so there are pairs to rank."),
                Parameter("problems", "Functions shown", "multi",
                          choices=lambda: list(config.FUNCTIONS), default=[1, 23],
                          help="F1 is separable unimodal, F23 weakly structured."),
            ),
            render=metrics.metric_added_value,
            coverage=lambda p: coverage_core.check_merged(p["dimension"]),
        ),
    ]
