# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

A diploma thesis codebase ("Analiza in vizualizacija algoritmov za optimizacijo prek iskalnih trajektorij" — analysis and visualization of optimization algorithms via search trajectories). It benchmarks 28 metaheuristic optimizers (from `mealpy`) on the BBOB benchmark suite (via `cocoex`/COCO), then clusters their search trajectories and computes several pairwise similarity/difference metrics between algorithms to see which metrics carry redundant vs. complementary information (via Spearman correlation). Code, comments, and output are English throughout; only the thesis-facing prose (docstrings quoting formulas) still references "the thesis."

## Before you run anything

Two things will bite you otherwise:

- **The benchmark step is an hours-long job.** `python run_pipeline.py` with no arguments starts step 1, which runs 28 optimizers × 24 functions × 5 instances × 3 dimensions × 5 seeds. Never kick it off casually or suggest the user do so without saying what it costs. In practice the benchmark is run once and later steps are re-run with `--from <step>`.
- **Pipeline steps overwrite the user's real results.** `data/`, `outputs/` and `metrics_data/` hold months of computed output that is *not* in git (all gitignored). Running a metrics step "just to see if it works" recomputes and rewrites real files. To exercise the pipeline safely, use the smoke test or set the `PIPELINE_TEST_*` env vars to redirect everything into a temp directory (see `config.py`'s docstring and `tests/smoke_test_pipeline.py`). Prefer `--dry-run` when you only need to check wiring.

Every pipeline script is import-safe: they all guard their work behind `if __name__ == "__main__":`, so importing one to test a function does not execute the pipeline. Keep it that way — several used to lack the guard, and importing one for a unit test kicked off its whole real computation.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` is a full `pip freeze` of the venv, committed to the repo. Key dependencies: `mealpy` (optimizer implementations), `cocoex`/`coco-experiment` (BBOB suite — this is a compiled C extension and is the dependency most likely to need special install steps), `scikit-learn`, `pandas`, `pytest`, `yellowbrick`/`kneed` (elbow-method cluster count selection), `dtw-python`, `pymoo`, `opfunu`. Regenerate it with `pip freeze > requirements.txt` after installing something new.

No directories need to be created by hand. `data/`, `outputs/`, `metrics_data/`, and every `figures_*/` folder are gitignored and don't exist on a fresh clone — every pipeline script creates whatever output directories it needs on first write. All of these paths are defined once in `config.py` as **absolute** paths anchored to the repo root (`REPO_ROOT = Path(__file__).resolve().parent`), so a step still writes to the real top-level folders no matter which directory you're in when you run it (e.g. `python 04_metrics/cosine_pairwise.py` from the repo root, or `cd 04_metrics && python cosine_pairwise.py` from inside the folder — both land in the same `metrics_data/`).

## Repo layout

Pipeline scripts live in numbered stage folders that mirror the run order end to end — `ls` at the repo root shows the whole pipeline shape:

```
01_optimize/    run_benchmarks.py, harvest_results.py
02_preprocess/  preprocess_data.py
03_cluster/     cluster_trajectories.py, cluster_similarity.py
04_metrics/     entropy.py, entropy_pairwise.py, entropy_plotting.py,
                cosine_pairwise.py, cosine_columns_pairwise.py,
                exploration_pairwise.py, solutions_pairwise.py
05_analysis/    merge_metrics.py, spearman.py, build_scalars.py,
                scalar_regression.py, and all the exploratory *.ipynb notebooks
scratch/        one-off diagnostic/debug + exploratory check scripts (preveri_*.py,
                poisci_pare_clustopt.py, slika_dodana_vrednost.py, check_*.py,
                testing.py) — not part of the pipeline, no assertions
tests/          pytest unit tests (test_*.py) + smoke_test_pipeline.py (see Testing below)
```

`config.py`, `utils.py`, `helper_functions.py`, and `run_pipeline.py` stay at the repo root since they're shared across every stage. Every script outside the root starts with a small `sys.path.insert(0, ...)` shim pointing at the repo root so `from config import ...` (and similar) resolves regardless of how the script is invoked — this is deliberate, not leftover cruft.

**Notebooks** in `05_analysis/` each open with a "repo-root bootstrap" cell that `os.chdir`s up to the repo root and puts the root + `04_metrics/` on `sys.path`. Jupyter's working directory is the notebook's own folder, so without it every `data/...`-style path inside the notebooks would resolve under `05_analysis/`. Keep that cell first, and run it before the rest of the notebook.

## The pipeline

`run_pipeline.py` orchestrates the full pipeline as a sequence of subprocess steps (each invoked by its full path under the stage folders above), each depending on the previous step's output. Run it whole, or resume/isolate a step:

```bash
python run_pipeline.py                  # run every step in order
python run_pipeline.py --from clustering # resume from a given step (inclusive)
python run_pipeline.py --only benchmark  # run a single step
python run_pipeline.py --dry-run         # print commands without running them
```

Step order and what each stage produces (see `STEPS` in `run_pipeline.py` for the exact commands):

1. **benchmark** (`01_optimize/run_benchmarks.py`) — runs mealpy optimizers against the `cocoex` BBOB suite, `epoch = 10 * dimension` (ClustOpt-paper convention). Writes raw per-run trajectories to `outputs/dim_{d}/{algorithm}/F{f}_I{i}/`. Supports parallelizing over functions/dimensions/seeds/instances/algorithms via `-p {f,d,s,a,i}`; `-a` restricts to specific algorithm names; `-b` saves only the g_best trajectory (skips the much larger full-population trajectory); `-e` also saves diversity/exploration/exploitation.
2. **harvest_results** (`01_optimize/harvest_results.py`) — scans `outputs/` g_best trajectories, builds `outputs/dim_{d}/results.csv` (one row per algorithm/problem/instance/seed best solution).
3. **preprocess** (`02_preprocess/preprocess_data.py`) — reshapes `outputs/` population trajectories into `data/processed/dim_{d}/F{f}_I{i}.csv` (renames `x1..xd`→`x0..x{d-1}`, `fitness`→`raw_y`; this is the shared input format for clustering).
4. **clustering** (`03_cluster/cluster_trajectories.py -c kmeans`) — KMeans-clusters trajectories per problem file (elbow method picks k from powers of 2, 4..512). Writes `cluster_centers/`, `clustering_results/` (parquet), `cluster_distributions/` (per-iteration cluster occupancy counts) under `data/clustering_latest/` (or `data/clustering_features_*_dbscan/` for `-c dbscan`/`dbscan_adaptive`).
5. **aggregate_cosine** (`03_cluster/cluster_similarity.py -c kmeans`) — aggregate cosine similarity between algorithm trajectory vectors, for clustermap figures.
6. **entropy_calc** / **entropy_pairwise** (`04_metrics/entropy.py`, `04_metrics/entropy_pairwise.py`) — Shannon entropy (normalized, H/ln k) of cluster occupancy per algorithm/run/iteration, then a pairwise difference measure between algorithms.
7. **cosine_pairwise** / **cosine_columns_pairwise** (`04_metrics/cosine_pairwise.py`, `04_metrics/cosine_columns_pairwise.py`) — global and per-cluster-column pairwise cosine distance between algorithms.
8. **exploration_pairwise** (`04_metrics/exploration_pairwise.py`) — pairwise difference in exploration/exploitation balance.
9. **solutions_pairwise** (`04_metrics/solutions_pairwise.py`) — pairwise difference in final solution location/fitness.
10. **merge** (`05_analysis/merge_metrics.py`) — outer-joins **every** metric found under `metrics_data/*/dim_{d}/` on the shared keys into `metrics_data/merged/merged_dim_{d}.csv`. This is the canonical "all metrics" table.
11. **build_scalars** (`05_analysis/build_scalars.py`) — builds `metrics_data/scalars.csv`: one row per (dim, func, algo) with per-**algorithm** scalars (entropy, fitness, exploration, diversity), derived from `data/entropy/entropy_granular_dim_{d}.csv`, `outputs/dim_{d}/results.csv` and the per-run `diversity_{seed}.csv` files. This is a *different shape* from everything else in `metrics_data/`, which is pairwise (one row per algorithm **pair**) — and it exists because `scalar_regression.py` needs per-algorithm scalars and previously had no input in the repo that matched.
12. **spearman** (`05_analysis/spearman.py`) — inner-joins only the metrics in its own `METRICS` list, computes the Spearman correlation matrix between them per dimension, and saves the matrix to `metrics_data/merged/spearman_dim_{d}.csv` and a heatmap to `figures_spearman/`. Its own working table is `metrics_data/merged/spearman_input_dim_{d}.csv` — a **different file** from step 10's `merged_dim_{d}.csv`, on purpose: both scripts used to write to the same `merged_dim_{d}.csv`, and since "spearman" always runs right after "merge", it silently clobbered merge_metrics.py's more complete output with its own narrower one on every real run. Don't reintroduce that collision if editing either script's output path.

Two figure-producing scripts aren't `run_pipeline.py` steps — run them manually after the step they depend on: `04_metrics/entropy_plotting.py` (after `entropy_calc`) and `05_analysis/scalar_regression.py` (after `build_scalars`, defaults to reading `metrics_data/scalars.csv` and writing `figures_results/scalar_regression.png`).

Note: a "return rate" / "shared revisit rate" metric (detecting when an algorithm revisits an already-explored cluster) used to be part of this pipeline (`return_rate.py`, `return_rate_pairwise.py`, `return_rate_plotting.py`, a `data/return_rate/` stage, and a `return_rate` column in `merged_dim_{d}.csv`). It was removed as unneeded — if you see stray references to it in old notes/branches, they're stale.

All per-metric pairwise CSVs under `metrics_data/<metric>/dim_{d}/F{f}_I{i}.csv` share the same key columns — `config.METRIC_KEYS` = `['Algorithm1', 'Algorithm2', 'Function_id', 'Instance_id', 'Run_id']` — plus exactly one metric-value column, which is what lets `merge_metrics.py`/`spearman.py` join them generically (they auto-detect the value column as "whatever isn't a key column"). `Function_id`/`Instance_id` may be written as `1` or as `"F1"`/`"I1"` depending on which script wrote the file; always pass values through `config.normalize_keys()` before comparing/joining across metrics.

## config.py

Single source of truth for the algorithm list (`ALGORITHMS_OF_INTEREST`, 28 names), the benchmark sweep (`DIMENSIONS`, `FUNCTIONS`, `INSTANCES`, `SEEDS`), every data/output/figures directory, the pairwise-metric CSV schema (`METRIC_KEYS`, `normalize_keys()`), and English `METRIC_LABELS` for figures. Every pipeline/metrics/analysis script imports the values it needs from here (often aliased on import to match the script's existing local variable name, e.g. `from config import CLUSTER_DISTRIBUTIONS_LATEST as INPUT_DIR`) instead of redeclaring them. When adding or removing an algorithm from the analysis, change it in `config.py` only.

Two things there are easy to miss:
- **`CLUSTERING_SEED`** is passed as `random_state` to both `KMeans(...)` calls in `03_cluster/cluster_trajectories.py`. sklearn's KMeans initializes centroids randomly, so without it a re-run could pick a different k *and* different cluster labels, which moves every metric computed downstream from cluster occupancy. Changing this value changes clustering results — measured on real dim-2 data, two different seeds disagree on ~53% of cluster labels.
  **Caveat:** the results currently stored in `data/clustering_latest/` (and everything derived from them in `metrics_data/` and `figures_*/`) were produced *before* the seed was added, i.e. unseeded. Re-running the pipeline today will therefore **not** reproduce those exact numbers. This was a deliberate choice — the existing results were kept as-is rather than regenerated. Reproducibility applies to runs from this point on.
- **`clustering_dir(method)`** maps the `-c` flag (`kmeans` / `dbscan` / `dbscan_adaptive`) to its data directory, and is used by *both* `cluster_trajectories.py` (writes) and `cluster_similarity.py` (reads). They previously each had their own copy of that mapping and disagreed about `dbscan_adaptive`, so that mode silently read a different directory than the one it wrote. Don't reintroduce a second copy.

## Testing

`pytest` is in `requirements.txt`. Run both before committing anything that touches the pipeline.

```bash
# Fast unit tests for the pure metric functions (normalize_keys,
# column_cosine_distance, compute_entropy/aggregate_from_granular). Synthetic
# fixtures only - no real data, no cocoex/sklearn/mealpy. Sub-second.
python -m pytest tests/

# End-to-end smoke test: runs EVERY run_pipeline.py step for real (real
# mealpy/cocoex optimization + real KMeans) on a tiny 2-algorithm/2-function
# sweep, redirected into a temp dir via config.py's PIPELINE_TEST_* env vars,
# so it never touches the real data/outputs/metrics_data. ~13s. Deliberately
# NOT named test_*.py so the fast suite above skips it - run it explicitly:
python -m pytest tests/smoke_test_pipeline.py -v -s
```

The smoke test is the one that catches wiring bugs (renamed modules, path changes, schema drift between steps) — the unit tests can't see those. Most of the real bugs found in this repo were caught by it, not by inspection.

CI (`.github/workflows/tests.yml`) byte-compiles every pipeline script and runs the fast suite on push and PR. It installs only `pandas numpy pytest` on purpose: the unit tests never reach the heavy scientific stack, and building `cocoex` (a compiled C extension) in CI would be slow and fragile. The smoke test is not run in CI.

### Editing the notebooks

`05_analysis/*.ipynb` are large (one is 7.5 MB with embedded output). Do **not** read them in full — it will flood the context with base64 image data. Grep them, or manipulate the JSON programmatically with `json.load`/`json.dump(..., indent=1, ensure_ascii=False)` plus a trailing newline, which round-trips Jupyter's own formatting and keeps the diff to just your change.

## Common commands

```bash
# run one pipeline step directly, e.g. after only changing filtering logic:
python 03_cluster/cluster_trajectories.py -c kmeans
python 03_cluster/cluster_similarity.py -c kmeans

# run the benchmark for a subset (fast iteration on one algorithm/function):
python 01_optimize/run_benchmarks.py -a OriginalGWO -f 1 -d 2 -i 1 -s 1
```
