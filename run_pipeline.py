"""
Orchestration script: runs the entire data-preparation and analysis pipeline
in order, from raw optimization runs to the final Spearman analysis.

Each step runs as a separate subprocess (the same as running it manually in a
terminal), and the sequence stops immediately if any step returns an error
(non-zero exit code) - so it doesn't continue on broken data.

Usage:
    python run_pipeline.py                    # run every step
    python run_pipeline.py --from clustering   # resume from a given step
    python run_pipeline.py --only benchmark    # run just one step
    python run_pipeline.py --dry-run           # print commands without running them

The pipeline scripts live in numbered stage folders (01_optimize/, 02_preprocess/,
03_cluster/, 04_metrics/, 05_analysis/) that mirror this exact run order - `ls` at
the repo root shows the whole pipeline shape at a glance. No directories need to
be created by hand beforehand: every step creates whatever output folders it
needs (data/, outputs/, metrics_data/, figures_*/) on its own, anchored to the
repo root regardless of the working directory this script is invoked from.
"""
import subprocess
import sys
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Step definitions: (name, command as an argument list, description)
# Order matters - each step depends on the previous step's output. The stage
# folder a step's script lives in (01_optimize, ..., 05_analysis) reflects
# this same order.
# ---------------------------------------------------------------------------
STEPS = [
    ("benchmark",
     [sys.executable, str(ROOT / "01_optimize" / "run_benchmarks.py"), "-p", "a", "-e"],
     "Run all optimization algorithms (mealpy/cocoex) - population "
     "+ g_best trajectory + diversity, epoch=10*dimension."),

    ("harvest_results",
     [sys.executable, str(ROOT / "01_optimize" / "harvest_results.py")],
     "Build results.csv from the g_best trajectories."),

    ("preprocess",
     [sys.executable, str(ROOT / "02_preprocess" / "preprocess_data.py")],
     "Convert outputs/ into data/processed/dim_{d}/F{f}_I{i}.csv (input for clustering)."),

    ("clustering",
     [sys.executable, str(ROOT / "03_cluster" / "cluster_trajectories.py"), "-c", "kmeans"],
     "KMeans-cluster the trajectories -> cluster_centers, cluster_distributions, clustering_results."),

    ("aggregate_cosine",
     [sys.executable, str(ROOT / "03_cluster" / "cluster_similarity.py"), "-c", "kmeans"],
     "Aggregate cosine similarity (for the clustermap figures)."),

    ("entropy_calc",
     [sys.executable, str(ROOT / "04_metrics" / "entropy.py")],
     "Compute cluster-occupancy entropy (granular + aggregated)."),

    ("entropy_pairwise",
     [sys.executable, str(ROOT / "04_metrics" / "entropy_pairwise.py")],
     "Pairwise entropy-difference metric."),

    ("return_rate_calc",
     [sys.executable, str(ROOT / "04_metrics" / "return_rate.py")],
     "Compute the revisiting history + revisit-rate matrices."),

    ("return_rate_pairwise",
     [sys.executable, str(ROOT / "04_metrics" / "return_rate_pairwise.py")],
     "Pairwise revisit-rate metric (granular)."),

    ("cosine_pairwise",
     [sys.executable, str(ROOT / "04_metrics" / "cosine_pairwise.py")],
     "Pairwise global cosine distance."),

    ("cosine_columns_pairwise",
     [sys.executable, str(ROOT / "04_metrics" / "cosine_columns_pairwise.py")],
     "Pairwise per-cluster-column cosine distance."),

    ("exploration_pairwise",
     [sys.executable, str(ROOT / "04_metrics" / "exploration_pairwise.py")],
     "Pairwise exploration/exploitation-difference metric."),

    ("solutions_pairwise",
     [sys.executable, str(ROOT / "04_metrics" / "solutions_pairwise.py")],
     "Pairwise location- and fitness-difference metric."),

    ("merge",
     [sys.executable, str(ROOT / "05_analysis" / "merge_metrics.py")],
     "Merge all metrics into merged_dim_{d}.csv."),

    ("spearman",
     [sys.executable, str(ROOT / "05_analysis" / "spearman.py")],
     "Spearman correlation analysis between metrics."),
]

STEP_NAMES = [name for name, _, _ in STEPS]


def run_step(name, cmd, description, dry_run=False):
    print(f"\n{'='*70}")
    print(f"STEP: {name}")
    print(f"  {description}")
    print(f"  command: {' '.join(cmd)}")
    print(f"{'='*70}")

    if dry_run:
        print("  [dry-run] skipped")
        return True

    start = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n!!! STEP '{name}' FAILED (exit code {result.returncode}), "
              f"after {elapsed:.1f}s. Stopping the pipeline.")
        return False

    print(f"\n  step '{name}' completed successfully in {elapsed:.1f}s")
    return True


def main():
    parser = argparse.ArgumentParser(description="Sequential data-preparation and analysis pipeline")
    parser.add_argument("--from", dest="from_step", choices=STEP_NAMES,
                        help="Resume from this step onward (inclusive)")
    parser.add_argument("--only", dest="only_step", choices=STEP_NAMES,
                        help="Run only this one step")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only print commands, without actually running them")
    args = parser.parse_args()

    if args.only_step:
        steps_to_run = [s for s in STEPS if s[0] == args.only_step]
    elif args.from_step:
        start_idx = STEP_NAMES.index(args.from_step)
        steps_to_run = STEPS[start_idx:]
    else:
        steps_to_run = STEPS

    print(f"Pipeline: {len(steps_to_run)} steps -> {[s[0] for s in steps_to_run]}")

    pipeline_start = time.time()
    for name, cmd, description in steps_to_run:
        ok = run_step(name, cmd, description, dry_run=args.dry_run)
        if not ok:
            sys.exit(1)

    total = time.time() - pipeline_start
    print(f"\n{'='*70}")
    print(f"ENTIRE PIPELINE COMPLETED SUCCESSFULLY in {total/60:.1f} min")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
