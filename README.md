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

## Running the pipeline

`run_pipeline.py` runs all 13 steps in order, each as a separate subprocess, stopping at
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

## Tests

```bash
pytest tests/                          # fast unit tests, ~0.4s, no real data needed
pytest tests/smoke_test_pipeline.py    # end-to-end: the whole pipeline on a tiny sweep
```

The smoke test runs all 13 steps for real (2 algorithms, 2 functions, one dimension) in an
isolated temporary directory, so it never touches your actual results. It is deliberately
named so `pytest tests/` does not pick it up.

## A note on reproducibility

Clustering is seeded (`CLUSTERING_SEED` in `config.py`), so runs are reproducible.
Results produced *before* that seed was introduced are not — sklearn's KMeans initialises
centroids randomly, and on this data two different seeds disagree on roughly half of all
cluster labels, which shifts every metric derived from cluster occupancy. If you need
results that a re-run reproduces exactly, regenerate them from the clustering step onward.
