# Progress

Working notes for the "run the whole pipeline from the GUI" effort.
Last updated 2026-09-18. **Status: done, merged, verified.**

## Where things stand

All four PRs from `desktop-gui` are merged into `main`:

| PR | What | Merge commit |
|---|---|---|
| #1 | Desktop GUI (original) | `0e84bf3` |
| #2 | Theme overhaul + pipeline runs from the GUI (Phases 1-4) | `36ee498` |
| #3 | Phase 5 packaging + two CI-only test fixes | `f7e9c2b` |
| #4 | Process-tab context line + trajectory dark-mode fix | `b725176` |

`main` is at `b725176`. Nothing outstanding from the plan
(`~/.claude/plans/how-do-i-use-spicy-cascade.md`) - all five phases done and
verified.

| Check | Result |
|---|---|
| `pytest tests/` | 259 passed |
| `pytest tests/smoke_test_pipeline.py -v -s` | passes - CLI/subprocess path |
| `pytest tests/smoke_test_gui_pipeline.py -v -s` | passes - in-process path + live Process tab |
| `python -m gui --self-test` | passes, 7 tabs, 9 checks |
| The real clustering re-run | **done** (see below) |

## What shipped

**Visual overhaul.** `gui/theme.py` is the only place colour, spacing and
type are defined; follows the OS light/dark preference.
`gui/viz/figure_theme.py` does the same for matplotlib, applied at
`Visualization.draw()` / `capture.render_with()` - app-only, so `figures_*/`
output keeps its original look.

**The pipeline runs from the GUI, including when packaged.** Every stage
script exposes `run(progress_cb, cancel_event)` per `pipeline_api.py`.
`gui/core/pipeline.py` mirrors `run_pipeline.STEPS` and drives stages
in-process (forced by packaging - a frozen app ships no interpreter to
subprocess out to). The Process tab runs them with per-stage/per-item
progress behind a confirmation naming every directory it will overwrite and
its current size, and now also shows the active results folder, dimensions
and clustering method up front, before you even tick a box.
`packaging/gui.spec` bundles all 13 stage modules; `gui/self_test.py` checks
they actually import in a built bundle.

**~21 bugs found and fixed** along the way, including two reproduced
crashes (`DataFrameModel.set_frame(None)`; `config.FUNCTIONS.index(16)`
taking out `MainWindow` construction), module-scope `parse_args()` in both
`03_cluster/` scripts, a CI failure from two tests missing skip guards
(`tqdm` not installed, `PySide6` not installed - CI intentionally installs
neither), and dark-mode trajectory plots rendering black-on-black because
`figure_theme.apply_to()` was only ever called once, before `axes.clear()`
wiped the styling on every real redraw.

## The clustering re-run

Done, via `python run_pipeline.py --from clustering` in a real terminal,
2026-09-17 17:46-19:03 (confirmed by file mtimes across all three real
dimensions - `data/clustering_latest/cluster_centers/dim_{2,5,10}`,
`metrics_data/merged/spearman_dim_{2,5,10}.csv`, `metrics_data/scalars.csv`
- and by the user's own manual check of `cluster_centers`). `config.CLUSTERING_SEED`
is now in effect for this data, unlike the results it replaced. No
`pipeline_run.json` exists for it, which is expected: that manifest is only
written by the GUI's `PipelineRunner`, never by `run_pipeline.py`.

Before landing on the terminal re-run, the GUI path was also verified
end-to-end against a disposable `PIPELINE_TEST_ROOT` temp tree (tiny
2-algorithm/2-function sweep) through a live Process tab - this is what
surfaced the persisted-`QSettings`-results-root bug (see below) and the
black-text trajectory bug.

## Bugs found via manual testing, beyond the plan's own scope

- **`gui/main_window.py`'s `_restore_results_root()`** silently restores a
  results folder from `QSettings`, persisted from any earlier session,
  overriding `PIPELINE_TEST_ROOT` on every launch. Not fixed (arguably
  correct behaviour for normal use), but worth remembering: a stale
  persisted root can make the safe env-var testing path look broken. Clear
  it with `rm ~/.config/OptimizerTrajectoryExplorer/gui.conf` if it happens
  again.
- **Trajectory panel dark-mode text** - see above, fixed in PR #4.
- **Process tab gave no indication of what it would run against** until the
  confirmation dialog appeared - fixed in PR #4 with a persistent context
  line (results folder / dimensions / method) above the stage list.

## Docs

`CLAUDE.md`, `README.md` both updated to match everything above - the
README's desktop-app section previously still said "the GUI does not run
[clustering/metrics] - it shows you the commands instead", which stopped
being true as of PR #2.

## Watch out for, going forward

- Keep `run_pipeline.py` invoking the scripts as subprocesses -
  `smoke_test_pipeline.py` passing unchanged is what proves any future
  refactor doesn't move the science.
- No `parse_args()` at module scope in a stage script.
- No `pyplot` in a stage - `gui/qt.py` forces `QtAgg`, and building a Qt
  canvas off the worker thread crashes.
- After any run, call `results_root.invalidate_caches()`, not
  `coverage.invalidate()` - the latter clears one memoised cache of four.
- After any figure redraw that touches an axes freshly `.clear()`'d, call
  `figure_theme.apply_to(figure)` again - restyling doesn't survive a clear.
- CI (`tests.yml`) installs only `pandas numpy pytest`. A new test that
  imports a stage module or `gui.qt` needs the matching
  `pytest.importorskip(...)` / skip-on-`ImportError` guard, or it'll pass
  locally and fail in CI.
