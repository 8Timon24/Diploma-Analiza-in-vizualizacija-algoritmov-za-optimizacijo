"""
End-to-end smoke test: runs the ENTIRE real pipeline (all run_pipeline.py
steps, unmodified) against a tiny synthetic sweep, writing into a throwaway
temp directory. This is what actually exercises the step-to-step wiring
(column names, merge keys, file formats) that the pure-function unit tests
in this folder can't see - it's what would have caught the import/path bugs
found while restructuring the repo into staged folders.

Deliberately named so pytest's default discovery (test_*.py / *_test.py)
does NOT pick it up - it runs real mealpy/cocoex optimization and real
KMeans clustering, so it takes real time (roughly a minute or more) and
should be run explicitly, not as part of every `pytest tests/`:

    venv/bin/python3 -m pytest tests/smoke_test_pipeline.py -v -s

Isolation: every directory config.py exposes, and every sweep parameter
(ALGORITHMS_OF_INTEREST, DIMENSIONS, FUNCTIONS, INSTANCES, SEEDS), is
overridden via the PIPELINE_TEST_* environment variables config.py reads
(see config.py's module docstring). The real repo's data/, outputs/,
metrics_data/, figures_*/ are never read or written by this test - every
path involved is anchored under pytest's tmp_path.
"""
import sys
import os
import json
import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import pytest

# Tiny synthetic sweep: >=2 algorithms (so algorithm_pairs is non-empty and
# pairwise metrics actually produce rows) and >=2 problems (so the merged
# table has more than one row - a single row makes Spearman correlation
# degenerate, all-NaN, which would hide a real wiring bug behind "it didn't
# crash"). Chosen algorithms are cheap/fast mealpy optimizers.
TEST_ALGORITHMS = ["OriginalDE", "OriginalHC"]
TEST_DIMENSIONS = [2]
TEST_FUNCTIONS = [1, 2]
TEST_INSTANCES = [1]
TEST_SEEDS = [1]


def _build_env(tmp_root):
    env = os.environ.copy()
    env["PIPELINE_TEST_ROOT"] = str(tmp_root)
    env["PIPELINE_TEST_ALGORITHMS"] = json.dumps(TEST_ALGORITHMS)
    env["PIPELINE_TEST_DIMENSIONS"] = json.dumps(TEST_DIMENSIONS)
    env["PIPELINE_TEST_FUNCTIONS"] = json.dumps(TEST_FUNCTIONS)
    env["PIPELINE_TEST_INSTANCES"] = json.dumps(TEST_INSTANCES)
    env["PIPELINE_TEST_SEEDS"] = json.dumps(TEST_SEEDS)
    return env


def test_full_pipeline_end_to_end(tmp_path):
    import run_pipeline  # real repo-root ROOT, unaffected by our env overrides

    steps = list(run_pipeline.STEPS)

    # Scope the benchmark step down via CLI flags too, not just config.py
    # overrides - otherwise mealpy would try the full 24-function/
    # 28-algorithm universe before config.py's overrides even matter.
    benchmark_cmd = [
        sys.executable, str(_REPO_ROOT / "01_optimize" / "run_benchmarks.py"),
        "-a", *TEST_ALGORITHMS,
        "-f", *[str(f) for f in TEST_FUNCTIONS],
        "-i", *[str(i) for i in TEST_INSTANCES],
        "-d", *[str(d) for d in TEST_DIMENSIONS],
        "-s", *[str(s) for s in TEST_SEEDS],
        "-e",  # also save diversity - exploration_pairwise needs it
    ]
    steps[0] = ("benchmark", benchmark_cmd, steps[0][2])

    # Snapshot the mtimes of the real dim_2 directories this test's data
    # would collide with if isolation failed, so we can prove afterward
    # that nothing in the real repo was touched. Not a recursive scan of
    # the (multi-GB) real data/outputs/metrics_data trees - just the
    # specific leaf directories a broken override would write files into.
    d = TEST_DIMENSIONS[0]
    collision_dirs = [
        _REPO_ROOT / "outputs" / f"dim_{d}",
        _REPO_ROOT / "data" / "processed" / f"dim_{d}",
        _REPO_ROOT / "data" / "clustering_latest" / "cluster_distributions" / f"dim_{d}",
        *[_REPO_ROOT / "metrics_data" / m / f"dim_{d}"
          for m in ("entropy", "cosine", "cosine_columns", "exploration", "location", "fitness")],
    ]
    before_snapshot = {
        str(p): sorted((f.name, f.stat().st_mtime) for f in p.iterdir() if f.is_file())
        for p in collision_dirs if p.is_dir()
    }

    env = _build_env(tmp_path)
    for name, cmd, description in steps:
        print(f"\n--- smoke test step: {name} ---")
        result = subprocess.run(cmd, env=env)
        assert result.returncode == 0, f"step '{name}' failed (exit {result.returncode}): {' '.join(cmd)}"

    d = TEST_DIMENSIONS[0]
    i0 = TEST_INSTANCES[0]

    # ---- benchmark + harvest_results ----
    results_csv = tmp_path / "outputs" / f"dim_{d}" / "results.csv"
    assert results_csv.is_file(), "harvest_results.py did not write results.csv"
    results = pd.read_csv(results_csv)
    assert set(results["algorithm"].unique()) == set(TEST_ALGORITHMS)
    assert set(results["problem_id"].unique()) == set(TEST_FUNCTIONS)

    # ---- preprocess ----
    for f in TEST_FUNCTIONS:
        processed = tmp_path / "data" / "processed" / f"dim_{d}" / f"F{f}_I{i0}.csv"
        assert processed.is_file(), f"preprocess_data.py did not write {processed.name}"

    # ---- clustering + aggregate_cosine ----
    for f in TEST_FUNCTIONS:
        cluster_dist = (tmp_path / "data" / "clustering_latest" / "cluster_distributions"
                         / f"dim_{d}" / f"F{f}_I{i0}.csv")
        assert cluster_dist.is_file(), f"clustering step did not write {cluster_dist.name}"

    # ---- every pairwise metric produced at least one real row, with the
    #      expected schema, for both algorithms (not just an empty stub) ----
    metric_files = {
        "entropy": ("Mean_entropy_difference", "entropy"),
        "cosine": ("Cosine_distance", "cosine"),
        "cosine_columns": ("Cosine_column_distance", "cosine_columns"),
        "exploration": ("Mean_exploration_difference", "exploration"),
        "location": ("Location_difference", "location"),
        "fitness": ("Fitness_difference", "fitness"),
    }
    for subdir, (value_col, _) in metric_files.items():
        for f in TEST_FUNCTIONS:
            candidates = list((tmp_path / "metrics_data" / subdir / f"dim_{d}").glob(f"F{f}_I*.csv"))
            assert candidates, f"no {subdir} metric file found for function {f}"
            df = pd.read_csv(candidates[0])
            assert value_col in df.columns, f"{subdir} file missing expected column {value_col}"
            if not df.empty:
                assert set(df["Algorithm1"]) | set(df["Algorithm2"]) <= set(TEST_ALGORITHMS)

    # ---- merge: all metrics joined into one table with real data ----
    merged_csv = tmp_path / "metrics_data" / "merged" / f"merged_dim_{d}.csv"
    assert merged_csv.is_file()
    merged = pd.read_csv(merged_csv)
    assert len(merged) > 0
    for _, (_, merged_col) in metric_files.items():
        # proves normalize_keys() correctly reconciles the different
        # Function_id/Instance_id conventions each pairwise script writes
        # (entropy_pairwise.py: plain ints; the rest: "F1"/"I1" strings) -
        # every metric must actually appear as a column after the merge.
        assert merged_col in merged.columns, f"merged table missing metric column '{merged_col}'"

    # ---- spearman: correlation matrix + figure produced from real data ----
    spearman_csv = tmp_path / "metrics_data" / "merged" / f"spearman_dim_{d}.csv"
    assert spearman_csv.is_file()
    spearman_pdf = tmp_path / "figures_spearman" / f"spearman_dim_{d}.pdf"
    assert spearman_pdf.is_file()

    # ---- isolation: the real repo's own directories were never touched ----
    after_snapshot = {
        str(p): sorted((f.name, f.stat().st_mtime) for f in p.iterdir() if f.is_file())
        for p in collision_dirs if p.is_dir()
    }
    assert after_snapshot == before_snapshot, (
        "the real repo's data/outputs/metrics_data were modified by this test - "
        "PIPELINE_TEST_ROOT isolation is broken"
    )
