# Progress

Working notes for the `desktop-gui` branch. Last updated 2026-09-18.

## Where things stand

Branch `desktop-gui`, pushed to origin. Last commit `7060ace` — "Theme the
desktop GUI and let it run the analysis pipeline" (48 files, +4429/−484).
Phase 5 (packaging) is done on top of that but **not yet committed**.

| | |
|---|---|
| Fast tests | 255 passing (was 176) |
| `tests/smoke_test_pipeline.py` | passes — the CLI/subprocess path |
| `tests/smoke_test_gui_pipeline.py` | passes — the in-process path + the Process tab |
| `python -m gui --self-test` | passes, 7 tabs, 9 checks (was 8) |
| Real `data/` / `metrics_data/` | untouched by any of this work |

## Done

**Visual overhaul.** `gui/theme.py` is now the only place colour, spacing and
type are defined; it follows the OS light/dark preference. `gui/viz/figure_theme.py`
does the same for matplotlib, applied at `Visualization.draw()` and inside
`capture.render_with()`. Deliberately app-only — the scripts that write
`figures_*/` keep their own look so the thesis figures stay consistent.

**The pipeline runs from the GUI.** Every stage script exposes
`run(progress_cb, cancel_event)` per the new `pipeline_api.py`, with `__main__`
reduced to a one-line wrapper. `gui/core/pipeline.py` mirrors
`run_pipeline.STEPS` and drives the stages in process; the Process tab runs
them with per-file progress behind a confirmation naming every directory it
overwrites and its current size. `run_pipeline.py` is untouched and still works.

**~19 bugs fixed**, including two reproduced crashes (`set_frame(None)`;
`config.FUNCTIONS.index(16)` taking out `MainWindow` construction) and the
module-scope `parse_args()` in both `03_cluster/` scripts.

## Done: Phase 5 — packaging

The last item from the plan (`~/.claude/plans/how-do-i-use-spicy-cascade.md`)
is done. In `packaging/gui.spec`:

- `01_optimize`, `02_preprocess`, `03_cluster` added to `pathex` (were
  absent; only `04_metrics`/`05_analysis` were there before)
- every stage module named in `hiddenimports`: `harvest_results`,
  `preprocess_data`, `cluster_trajectories`, `cluster_similarity`,
  `entropy`, `entropy_pairwise`, `cosine_pairwise`,
  `cosine_columns_pairwise`, `exploration_pairwise`, `solutions_pairwise`,
  `merge_metrics`, `build_scalars`, `spearman` (`entropy_plotting` and
  `scalar_regression` were already there for the figure scripts).
  `run_benchmarks` deliberately left out — the GUI drives it through
  `helper_functions.run_benchmarks`, never imports that module.
- `yellowbrick` and `kneed` removed from `excludes`
- `gui/self_test.py` gained a check that imports every non-benchmark stage
  in `pipeline.STAGES` and asserts `run` is callable — same trick as the
  mealpy/opfunu checks: silent in a checkout, loud in a bundle missing a
  module

`CLAUDE.md`'s Packaging section updated to match. Verified: fast suite (255,
unchanged), `python -m gui --self-test` (now 9 checks, all pass, including
the new one — imported all 13 non-benchmark stages successfully), both
files byte-compile.

**Not yet verified: an actual PyInstaller build.** This environment can't
run PyInstaller/produce a Windows exe — that only happens in
`.github/workflows/build-windows.yml`. Push and watch that workflow before
calling packaging fully proven; a module resolving fine via `sys.path`
in a checkout doesn't guarantee PyInstaller's static analysis found it via
`pathex` the same way.

## Then: the clustering re-run

The original goal. `data/clustering_latest/` (3.5 GB) was produced *before*
`CLUSTERING_SEED` existed, i.e. unseeded. Regenerating it and everything
downstream is `--from clustering`, or the Process tab ticked from Clustering.

- **No need to re-run the benchmark.** `helper_functions.run_benchmarks` has
  always taken an explicit seed, so the 11 GB in `outputs/` is unaffected.
- **Disk is fine.** No stage deletes and none writes alongside; files are
  replaced one at a time, so peak transient overhead is one file (<20 MB), not
  3.5 GB. Free space was 7.6 GB on a 90%-full disk.
- **Back up first if it matters.** None of it is in git.
  `data/clustering_latest/cluster_distributions/` (290 MB) plus `metrics_data/`
  (281 MB) is ~570 MB and covers everything downstream without the 3.2 GB of
  raw cluster assignments.
- **Cancelling mid-stage leaves a mixed tree** — nothing is transactional and
  `merge_metrics.py` merges whatever it finds. The Process tab says so.

## Watch out for

- Keep `run_pipeline.py` invoking the scripts as subprocesses.
  `smoke_test_pipeline.py` passing unchanged is what proves the refactor did
  not move the science.
- No `parse_args()` at module scope in a stage script.
- No `pyplot` in a stage — `gui/qt.py` forces `QtAgg` and building a Qt canvas
  off the worker thread crashes.
- After a run call `results_root.invalidate_caches()`, not
  `coverage.invalidate()` — the latter clears one memoised cache of four.
