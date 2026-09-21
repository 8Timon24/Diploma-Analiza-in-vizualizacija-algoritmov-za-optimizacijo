# Analysis and visualization of optimization algorithms via their search trajectories

**Diploma thesis codebase**

*(Ta README je na voljo tudi v slovenščini: [README.sl.md](README.sl.md))*

Metaheuristic optimizers are usually compared by *how good* their final solution is. This
project compares them by *how they search*: it records the full population trajectory of
28 metaheuristics on the BBOB benchmark suite, discretises each trajectory by clustering
the visited points, and derives several pairwise "behavioural distance" measures between
algorithms.

The research question is which of those measures actually tell you different things.
The final step computes the Spearman correlation between all metrics, separating the ones
that are largely redundant from the ones that carry complementary information.

## Contents

- [What it measures](#what-it-measures)
- [Two ways to use it](#two-ways-to-use-it)
- [Requirements](#requirements)
- [The command-line pipeline](#the-command-line-pipeline)
- [Pipeline stages](#pipeline-stages)
- [Outputs](#outputs)
- [Regression over per-algorithm scalars](#regression-over-per-algorithm-scalars)
- [Exploratory notebooks](#exploratory-notebooks)
- [The desktop app](#the-desktop-app)
- [Tests](#tests)
- [A note on reproducibility](#a-note-on-reproducibility)

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

### Every step, run directly

The exact command `run_pipeline.py` runs for each step (from `STEPS` in
`run_pipeline.py`), in order:

| Step | Command | Produces |
|---|---|---|
| `benchmark` | `python 01_optimize/run_benchmarks.py -p a -e` | raw per-run trajectories in `outputs/` |
| `harvest_results` | `python 01_optimize/harvest_results.py` | `outputs/dim_{d}/results.csv` |
| `preprocess` | `python 02_preprocess/preprocess_data.py` | `data/processed/dim_{d}/F{f}_I{i}.csv` |
| `clustering` | `python 03_cluster/cluster_trajectories.py -c kmeans` | `data/clustering_latest/{cluster_centers,cluster_distributions,clustering_results}` |
| `aggregate_cosine` | `python 03_cluster/cluster_similarity.py -c kmeans` | aggregate cosine similarity, for the clustermap figures |
| `entropy_calc` | `python 04_metrics/entropy.py` | cluster-occupancy entropy (granular + aggregated) |
| `entropy_pairwise` | `python 04_metrics/entropy_pairwise.py` | pairwise entropy-difference metric |
| `cosine_pairwise` | `python 04_metrics/cosine_pairwise.py` | pairwise global cosine distance |
| `cosine_columns_pairwise` | `python 04_metrics/cosine_columns_pairwise.py` | pairwise per-cluster-column cosine distance |
| `exploration_pairwise` | `python 04_metrics/exploration_pairwise.py` | pairwise exploration/exploitation difference |
| `solutions_pairwise` | `python 04_metrics/solutions_pairwise.py` | pairwise location/fitness difference |
| `merge` | `python 05_analysis/merge_metrics.py` | `metrics_data/merged/merged_dim_{d}.csv` |
| `build_scalars` | `python 05_analysis/build_scalars.py` | `metrics_data/scalars.csv` |
| `spearman` | `python 05_analysis/spearman.py` | `metrics_data/merged/spearman_dim_{d}.csv`, `figures_spearman/` |

Two more scripts aren't `run_pipeline.py` steps - run them manually after
their dependency:

| Command | Needs | Produces |
|---|---|---|
| `python 04_metrics/entropy_plotting.py` | `entropy_calc` | entropy figures in `figures_entropy/` |
| `python 05_analysis/scalar_regression.py` | `build_scalars` | `figures_results/scalar_regression.png` (see [below](#regression-over-per-algorithm-scalars)) |

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

## Exploratory notebooks

`05_analysis/` also holds the notebooks used to build and sanity-check the
analysis above - `run_pipeline.py` doesn't run them, and nothing downstream
depends on their output. Open them with `jupyter lab 05_analysis/` (or via
an editor's notebook support) from anywhere; each one's **first cell is a
"repo-root bootstrap"** that `os.chdir`s up to the repo root and puts it on
`sys.path`, since Jupyter's own working directory is the notebook's folder
and every path in these notebooks (`data/...`, `metrics_data/...`) assumes
the repo root. Always run that cell first.

| Notebook | What it's for |
|---|---|
| `3_cluster_analysis.ipynb` | Plot the raw population trajectories for a chosen set of dimensions/problems/algorithms/runs, coloured by cluster |
| `4_example_visualize_trajectories_for_clustering.ipynb` | Worked example of visualizing a problem's landscape and trajectories, from setting up the clustering input |
| `entropy_notebook.ipynb` | One example of each entropy plot type (`04_metrics/entropy_plotting.py`), with algorithm filtering |
| `exploration_plots_preview.ipynb` | Exploration/exploitation curves compared across algorithms |
| `location_fitness_preview.ipynb` | Final-solution location and fitness compared against the true BBOB optimum |
| `pari_algoritmov_metrike_heatmap.ipynb` | Pairwise-metric heatmap and table for eyeballing which measures agree, ahead of the automated Spearman step |

They're large (one is several MB with embedded plot output) - if you're
editing one programmatically rather than through Jupyter, load it with
`json.load`/`json.dump(..., indent=1, ensure_ascii=False)` plus a trailing
newline rather than a text editor, to keep the diff to just your change.

## The desktop app

A PySide6 GUI over the same pipeline: pick optimizers and benchmark functions,
run a scoped experiment, browse the raw data, and render the project's figures
without writing any Python.

```bash
python -m gui                 # from the repo root, in the venv
python -m gui --self-test     # headless checks, no window; used by the build CI
```

Seven tabs:

| Tab | What it does |
| --- | --- |
| Setup | Pick from all 234 mealpy optimizers and a benchmark source, with a live cost estimate |
| Run | Progress, live convergence plot, streaming log, cancel |
| Process | Run clustering and every metric stage - the rest of the pipeline after the benchmark - without a terminal |
| Visualize | 20 visualizations, 13 of them previously reachable only by running a notebook |
| Trajectory | Animated 2-D search playback over the true landscape, with GIF export |
| Compare | Two algorithms head to head across all six pairwise metrics |
| Data | Every file the pipeline has written, in a sortable table |

Benchmark sources: **BBOB** (24 functions x 110 instances via COCO), **opfunu**
(125 classic functions), the **CEC** suites (cec2005-cec2022), and a
**user-defined** expression.

### It does not touch your results by accident

Benchmark runs (the Run tab) are written to `gui_runs/<timestamp>/` with a
`manifest.json` recording the exact selection, library versions and
clustering seed - never into the repository's real `outputs/` unless you
explicitly confirm that in the Setup tab. Rendering a figure never writes to
any `figures_*/` directory - the pipeline's plotting functions call `savefig`
with hardcoded paths, so the GUI disables `savefig` while rendering, and
exporting a figure is a separate, explicit action.

The Process tab is the one place that *does* write into the real `data/` and
`metrics_data/` in place - see below - and it never does so without you
confirming exactly what will be overwritten.

If a visualization needs data that does not exist yet, the app says exactly
what is missing, estimates what it would cost, and offers to run those
stages for you (or, for the benchmark, prefills the Run form with that
selection).

### Running the pipeline from the app

The **Process** tab runs every stage after the benchmark - preprocessing,
clustering, all six pairwise metrics, merging, scalars, Spearman - in the
running app itself, no terminal needed. Tick individual stages, or use
"start from" to tick a stage and everything after it (the same thing
`run_pipeline.py --from <step>` does on the command line - the two agree on
step names and order). Two progress bars track overall progress and the
current stage's; Cancel stops at the next item boundary.

It writes into the real `data/` and `metrics_data/` in place, the same as
running the steps from a terminal, so clicking Run asks you to confirm first
- naming every directory it's about to overwrite and its current size. Runs
are not transactional and nothing is deleted first, so cancelling part-way
leaves a mix of old and new files; the confirmation and the log both say so.
A `pipeline_run.json` is written next to the results afterward, recording
which stages ran, how long, and the clustering seed in effect.

This is also what makes a **packaged build** self-contained: it ships no
Python interpreter, so `run_pipeline.py`'s subprocess-per-step approach isn't
available there - the Process tab imports each stage module and calls it
in-process instead, which is why it exists.

### Pointing it at your results

**File > Open results folder...** switches which tree the app reads and
writes (`outputs/`, `data/`, `metrics_data/`), and the choice is remembered.
This is what makes the packaged app useful on a machine without the
repository, and the Data tab header always states which folder is active and
what it contains.

**File > Open a run from this app...** opens one of your own runs. A run
directory is a results tree in its own right, so the raw-trajectory and
exploration views work on it immediately; run the Process tab against it to
get the entropy, cosine and Spearman views too.

### Getting the Windows executable

Push a version tag (`git tag v0.1.0 && git push origin v0.1.0`) and
`.github/workflows/build-windows.yml` builds it and attaches the zipped
bundle to a GitHub Release under that tag - that Release page is what to
hand someone who just wants to run the app, not the repository. Download
the zip, extract it (the `.exe` needs its sibling DLLs and data files next
to it - don't copy it out on its own), and run
`OptimizerTrajectoryExplorer.exe`. It isn't code-signed, so Windows
SmartScreen will flag it on first launch; click **More info > Run anyway**.

Triggering the workflow manually (Actions tab > build-windows-exe > Run
workflow) instead of via a tag builds the same thing but only uploads it as
a 30-day workflow artifact, since there's no tag to name a Release after.

PyInstaller cannot cross-compile, so the `.exe` is only ever built on a
Windows runner - to build locally on your own Windows machine instead:

```bash
pip install -r packaging/requirements-app.txt
pip install --no-deps mealpy==3.0.3   # see the comment in that file for why
pyinstaller packaging/gui.spec --noconfirm
dist/OptimizerTrajectoryExplorer/OptimizerTrajectoryExplorer --self-test
```

The bundle is about 500 MB and the self-test is what proves it is usable: it
checks the things that fail *only* once packaged - mealpy optimizers are found
by `pkgutil.walk_packages`, which PyInstaller cannot trace; opfunu's CEC suites
need ~1190 bundled data files; and cocoex is a compiled C extension.

## Tests

```bash
pytest tests/                             # fast unit + GUI tests, ~10s, no real data needed
pytest tests/smoke_test_pipeline.py -v -s       # end-to-end via subprocess, real pipeline
pytest tests/smoke_test_gui_pipeline.py -v -s   # the same, but in-process through the app
```

Both smoke tests run all 14 steps for real (2 algorithms, 2 functions, one dimension) in an
isolated temporary directory, so neither ever touches your actual results. The first runs
each step the way `run_pipeline.py` does, as a subprocess; the second drives the same sweep
through `gui/core/pipeline.py` - the code the Process tab uses - including one test that
starts a real `QThread` and exercises the Process tab itself. Both are deliberately not
named `test_*.py`, so `pytest tests/` does not pick them up and they must be run explicitly.

## A note on reproducibility

Clustering is seeded (`CLUSTERING_SEED` in `config.py`), so runs are reproducible.
Results produced *before* that seed was introduced are not — sklearn's KMeans initialises
centroids randomly, and on this data two different seeds disagree on roughly half of all
cluster labels, which shifts every metric derived from cluster occupancy. If you need
results that a re-run reproduces exactly, regenerate them from the clustering step onward.
