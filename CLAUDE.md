# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

A diploma thesis codebase ("Analiza in vizualizacija algoritmov za optimizacijo prek iskalnih trajektorij" — analysis and visualization of optimization algorithms via search trajectories). It benchmarks 28 metaheuristic optimizers (from `mealpy`) on the BBOB benchmark suite (via `cocoex`/COCO), then clusters their search trajectories and computes several pairwise similarity/difference metrics between algorithms to see which metrics carry redundant vs. complementary information (via Spearman correlation). Code, comments, and output are English throughout; only the thesis-facing prose (docstrings quoting formulas) still references "the thesis."

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
                return_rate.py, return_rate_pairwise.py, return_rate_plotting.py,
                cosine_pairwise.py, cosine_columns_pairwise.py,
                exploration_pairwise.py, solutions_pairwise.py
05_analysis/    merge_metrics.py, spearman.py, scalar_regression.py,
                and all the exploratory *.ipynb notebooks
scratch/        one-off diagnostic/debug scripts (preveri_*.py, poisci_pare_clustopt.py,
                slika_dodana_vrednost.py, testing.py) — not part of the pipeline
tests/          pytest unit tests (test_*.py) + smoke_test_pipeline.py (see Testing below)
```

`config.py`, `utils.py`, `helper_functions.py`, and `run_pipeline.py` stay at the repo root since they're shared across every stage. Every moved script starts with a small `sys.path.insert(0, ...)` shim pointing at the repo root so `from config import ...` (and similar) resolves regardless of how the script is invoked — this is deliberate, not leftover cruft.

`Old/` and `trash/` hold pre-refactor/deprecated versions of scripts — treat them as reference only, not live code.

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
7. **return_rate_calc** / **return_rate_pairwise** (`04_metrics/return_rate.py`, `04_metrics/return_rate_pairwise.py`) — detects "revisit" events (an already-visited cluster becoming active again) and a Jaccard-based shared-revisit-rate between algorithm pairs.
8. **cosine_pairwise** / **cosine_columns_pairwise** (`04_metrics/cosine_pairwise.py`, `04_metrics/cosine_columns_pairwise.py`) — global and per-cluster-column pairwise cosine distance between algorithms.
9. **exploration_pairwise** (`04_metrics/exploration_pairwise.py`) — pairwise difference in exploration/exploitation balance.
10. **solutions_pairwise** (`04_metrics/solutions_pairwise.py`) — pairwise difference in final solution location/fitness.
11. **merge** (`05_analysis/merge_metrics.py`) — outer-joins **every** metric found under `metrics_data/*/dim_{d}/` (including `return_rate`) on the shared keys into `metrics_data/merged/merged_dim_{d}.csv`. This is the canonical "all metrics" table.
12. **spearman** (`05_analysis/spearman.py`) — inner-joins only the metrics in its own `METRICS` list (currently excludes `return_rate`), computes the Spearman correlation matrix between them per dimension, and saves the matrix to `metrics_data/merged/spearman_dim_{d}.csv` and a heatmap to `figures_spearman/`. Its own working table is `metrics_data/merged/spearman_input_dim_{d}.csv` — a **different file** from step 11's `merged_dim_{d}.csv`, on purpose: both scripts used to write to the same `merged_dim_{d}.csv`, and since "spearman" always runs right after "merge", it silently clobbered merge_metrics.py's more complete output with its own narrower one on every real run (confirmed in the user's existing data). Don't reintroduce that collision if editing either script's output path.

Two plotting scripts (`04_metrics/entropy_plotting.py`, `04_metrics/return_rate_plotting.py`) aren't `run_pipeline.py` steps — run them manually after their calc-stage counterpart to render figures.

All per-metric pairwise CSVs under `metrics_data/<metric>/dim_{d}/F{f}_I{i}.csv` share the same key columns — `config.METRIC_KEYS` = `['Algorithm1', 'Algorithm2', 'Function_id', 'Instance_id', 'Run_id']` — plus exactly one metric-value column, which is what lets `merge_metrics.py`/`spearman.py` join them generically (they auto-detect the value column as "whatever isn't a key column"). `Function_id`/`Instance_id` may be written as `1` or as `"F1"`/`"I1"` depending on which script wrote the file; always pass values through `config.normalize_keys()` before comparing/joining across metrics.

## config.py

Single source of truth for the algorithm list (`ALGORITHMS_OF_INTEREST`, 28 names), the benchmark sweep (`DIMENSIONS`, `FUNCTIONS`, `INSTANCES`, `SEEDS`), every data/output/figures directory, the pairwise-metric CSV schema (`METRIC_KEYS`, `normalize_keys()`), and English `METRIC_LABELS` for figures. Every pipeline/metrics/analysis script imports the values it needs from here (often aliased on import to match the script's existing local variable name, e.g. `from config import RETURN_RATE_DATA_DIR as RR_DATA_DIR`) instead of redeclaring them. When adding or removing an algorithm from the analysis, change it in `config.py` only.

## Testing

`pytest` is in `requirements.txt`.

```bash
# fast unit tests for the pure metric functions (normalize_keys, column_cosine_distance,
# revisiting_history, compute_entropy/aggregate_from_granular) - synthetic fixtures, no
# real data or cocoex dependency, ~2s total:
python -m pytest tests/

# some pipeline scripts execute their real computation unconditionally at import time
# (no `if __name__ == "__main__":` guard) - never import a name from one of these for a
# test without first checking it has the guard, or the import itself runs the real thing.

# end-to-end smoke test: runs the ENTIRE real pipeline (all 15 run_pipeline.py steps,
# real mealpy/cocoex optimization + real KMeans clustering) against a tiny synthetic
# 2-algorithm/2-function sweep, isolated to a temp dir via config.py's PIPELINE_TEST_*
# env vars (see config.py's docstring) - never touches the real data/outputs/metrics_data.
# Deliberately NOT named test_*.py so `pytest tests/` above doesn't pick it up - run it
# explicitly (takes ~15s, not part of the fast suite):
python -m pytest tests/smoke_test_pipeline.py -v -s
```

## Common commands

```bash
# run one pipeline step directly, e.g. after only changing filtering logic:
python 03_cluster/cluster_trajectories.py -c kmeans
python 03_cluster/cluster_similarity.py -c kmeans

# run the benchmark for a subset (fast iteration on one algorithm/function):
python 01_optimize/run_benchmarks.py -a OriginalGWO -f 1 -d 2 -i 1 -s 1
```
