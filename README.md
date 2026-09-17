# Analiza in vizualizacija algoritmov za optimizacijo prek iskalnih trajektorij

**Analysis and visualization of optimization algorithms via their search trajectories** —
a diploma thesis codebase.

Metaheuristic optimizers are usually compared by *how good* their final solution is. This
project compares them by *how they search*: it records the full population trajectory of
28 metaheuristics on the BBOB benchmark suite, discretises each trajectory by clustering
the visited points, and derives several pairwise "behavioural distance" measures between
algorithms.

The research question is which of those measures actually tell you different things.
The final step computes the Spearman correlation between all metrics, separating the ones
that are largely redundant from the ones that carry complementary information.

## What it measures

Each metric produces one number per (algorithm A, algorithm B, function, instance, run):

| Metric | What it captures |
|---|---|
| `entropy` | Difference in normalized Shannon entropy (H / ln k) of cluster occupancy — how spread out the population is |
| `cosine` | Cosine distance between the two flattened cluster-occupancy vectors |
| `cosine_columns` | Cosine distance computed per cluster, then averaged |
| `exploration` | Difference in the exploration / exploitation balance |
| `location` | Euclidean distance between the two final solutions |
| `fitness` | Difference in final objective value |

## Two ways to use it

One codebase with two front-ends over the same code. The desktop app **imports**
the pipeline — `config.py`, `helper_functions.py`, and the plotting functions in
`04_metrics/` and `05_analysis/` — rather than reimplementing any of it, so a
figure drawn in the app is the figure the pipeline produces.

| | What it is | Reach for it when |
|---|---|---|
| **[Command-line pipeline](#the-command-line-pipeline)** | The thesis workflow: 14 batch steps from benchmark to Spearman correlation | Reproducing the thesis results, or running the full sweep |
| **[Desktop app](#the-desktop-app)** | A PySide6 GUI: pick optimizers, run scoped experiments, browse the data, render the figures | Exploring results, trying other optimizers or benchmark suites, showing the work to someone |

The pipeline is the source of truth and runs standalone; the app is optional and
nothing in the pipeline imports it.

## Requirements

Python 3.10, and the packages in `requirements.txt`. The one that may need attention is
`cocoex` (`coco-experiment`), the COCO/BBOB benchmark suite — it is a compiled C
extension.

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

No directories need to be created by hand. `data/`, `outputs/`, `metrics_data/` and the
`figures_*/` folders are generated, gitignored, and absent from a fresh clone; every step
creates what it needs on first write.

## The command-line pipeline

`run_pipeline.py` runs all 14 steps in order, each as a separate subprocess, stopping at
the first failure so nothing runs on broken data:

```bash
python run_pipeline.py                     # everything, start to finish
python run_pipeline.py --dry-run           # print the commands without running them
python run_pipeline.py --from clustering   # resume from a step (inclusive)
python run_pipeline.py --only spearman     # run a single step
```

**The first step is expensive.** It runs 28 optimizers × 24 functions × 5 instances ×
3 dimensions × 5 seeds, with `epoch = 10 × dimension` and a population of 50. Budget
hours, not minutes. Everything after it operates on the cached results, so in practice you
run the benchmark once and re-run later steps with `--from`.

Individual steps can also be run directly; they work from any working directory:

```bash
python 01_optimize/run_benchmarks.py -a OriginalGWO -f 1 -d 2 -i 1 -s 1   # small subset
python 03_cluster/cluster_trajectories.py -c kmeans
```

## Pipeline stages

Scripts live in numbered folders that mirror the execution order:

```
01_optimize/    run the optimizers on BBOB, harvest best-solution results
02_preprocess/  reshape raw trajectories into the clustering input format
03_cluster/     KMeans-cluster the trajectory points; aggregate cosine similarity
04_metrics/     entropy, cosine, cosine-per-column, exploration, location/fitness
05_analysis/    merge all metrics, Spearman correlation, exploratory notebooks
```

Supporting code sits at the repo root: `config.py` (all constants and paths),
`utils.py` and `helper_functions.py` (shared helpers), `run_pipeline.py` (the orchestrator).
`scratch/` holds one-off diagnostic scripts and `tests/` the test suite.

`gui/` is the desktop app and `packaging/` its build spec. Both sit alongside the
pipeline rather than wrapping it: the numbered stages above run exactly as they
always have, with or without the app installed.

`config.py` is the single place to change the algorithm list, the dimensions/functions/
instances/seeds swept, the clustering seed, and every input/output directory.

## Outputs

```
outputs/        raw per-run trajectories from the optimizers
data/           processed trajectories, clustering results, entropy tables
metrics_data/   one CSV per metric per problem, plus merged_dim_{d}.csv
figures_*/      generated plots
```

`metrics_data/merged/merged_dim_{d}.csv` is the main artifact: every metric joined on
(Algorithm1, Algorithm2, Function_id, Instance_id, Run_id).

`metrics_data/scalars.csv` is the companion per-**algorithm** table (one row per dim/func/algo,
with entropy, fitness, exploration and diversity), used for the regression analysis below.

## Regression over per-algorithm scalars

The pairwise metrics above answer "how differently do A and B search?". To ask a question
about the algorithms themselves — *do higher-entropy searchers actually explore more?* — you
need per-algorithm scalars rather than pairwise differences, which is what
`metrics_data/scalars.csv` provides:

```bash
python 05_analysis/build_scalars.py                      # build the scalar table
python 05_analysis/scalar_regression.py                  # entropy vs exploration (default)
python 05_analysis/scalar_regression.py --x entropy --y fitness --dims 2 10
```

It fits one OLS panel per dimension, colours the 28 algorithms by family, draws error bars
for the spread across BBOB functions, and rings high-influence points (Cook's distance) so a
line propped up by one or two outliers is visible rather than hidden. Output goes to
`figures_results/scalar_regression.png`.

Note it deliberately regresses the *scalars*, not averaged pairwise distances: for a
difference-type metric, averaging over partners is V-shaped in the underlying scalar, so it
measures atypicality rather than magnitude (see the module docstring).

## The desktop app

A PySide6 GUI over the same pipeline: pick optimizers and benchmark functions,
run a scoped experiment, browse the raw data, and render the project's figures
without writing any Python.

```bash
python -m gui                 # from the repo root, in the venv
python -m gui --self-test     # headless checks, no window; used by the build CI
```

Six tabs:

| Tab | What it does |
| --- | --- |
| Setup | Pick from all 234 mealpy optimizers and a benchmark source, with a live cost estimate |
| Run | Progress, live convergence plot, streaming log, cancel |
| Visualize | 20 visualizations, 13 of them previously reachable only by running a notebook |
| Trajectory | Animated 2-D search playback over the true landscape, with GIF export |
| Compare | Two algorithms head to head across all six pairwise metrics |
| Data | Every file the pipeline has written, in a sortable table |

Benchmark sources: **BBOB** (24 functions x 110 instances via COCO), **opfunu**
(125 classic functions), the **CEC** suites (cec2005-cec2022), and a
**user-defined** expression.

### It does not touch your results

Runs are written to `gui_runs/<timestamp>/` with a `manifest.json` recording
the exact selection, library versions and clustering seed. Writing into the
repository's real `outputs/` is possible but requires confirming a dialog that
says what will be overwritten. Rendering a figure never writes to any
`figures_*/` directory - the pipeline's plotting functions call `savefig` with
hardcoded paths, so the GUI disables `savefig` while rendering and exporting is
a separate, explicit action.

If a visualization needs data that does not exist yet, the app says exactly
what is missing, estimates what it would cost, and offers to prefill the run
form with that selection.

### Pointing it at your results

**File > Open results folder...** switches which tree the app reads
(`outputs/`, `data/`, `metrics_data/`), and the choice is remembered. This is
what makes the packaged app useful on a machine without the repository, and
the Data tab header always states which folder is active and what it contains.

**File > Open a run from this app...** opens one of your own runs. A run
directory is a results tree in its own right, so the raw-trajectory and
exploration views work on it immediately. The entropy, cosine and Spearman
views need the clustering and metric stages, which the GUI does not run - it
shows you the commands instead.

### Building the Windows executable

PyInstaller cannot cross-compile, so the `.exe` is built on a Windows runner
(`.github/workflows/build-windows.yml`, triggered by a `v*` tag or manually).
To build locally on your own platform:

```bash
pip install -r packaging/requirements-app.txt
pyinstaller packaging/gui.spec --noconfirm
dist/OptimizerTrajectoryExplorer/OptimizerTrajectoryExplorer --self-test
```

The bundle is about 500 MB and the self-test is what proves it is usable: it
checks the things that fail *only* once packaged - mealpy optimizers are found
by `pkgutil.walk_packages`, which PyInstaller cannot trace; opfunu's CEC suites
need ~1190 bundled data files; and cocoex is a compiled C extension.

## Tests

```bash
pytest tests/                          # fast unit tests, ~0.4s, no real data needed
pytest tests/smoke_test_pipeline.py    # end-to-end: the whole pipeline on a tiny sweep
```

The smoke test runs all 14 steps for real (2 algorithms, 2 functions, one dimension) in an
isolated temporary directory, so it never touches your actual results. It is deliberately
named so `pytest tests/` does not pick it up.

## A note on reproducibility

Clustering is seeded (`CLUSTERING_SEED` in `config.py`), so runs are reproducible.
Results produced *before* that seed was introduced are not — sklearn's KMeans initialises
centroids randomly, and on this data two different seeds disagree on roughly half of all
cluster labels, which shifts every metric derived from cluster occupancy. If you need
results that a re-run reproduces exactly, regenerate them from the clustering step onward.
