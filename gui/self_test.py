"""Checks that only mean something once the app is packaged.

PyInstaller cannot trace two things this app depends on:

  * helper_functions.get_optimizers_safe() finds optimizers with
    pkgutil.walk_packages, which is invisible to static analysis - without
    collect_submodules('mealpy') a built exe starts fine and offers ZERO
    optimizers.
  * opfunu's CEC suites load ~1190 bundled shift/rotation data files. Without
    collect_data_files('opfunu') the catalog looks right and every CEC
    function fails the moment it is evaluated.

Both failures are silent and only appear on the machine the exe ships to, so
they are checked explicitly here and run against the built artifact in CI.
"""
import sys
import traceback


def _check(label, function, expectation=None):
    try:
        value = function()
    except Exception as exc:
        print(f"  FAIL  {label}: {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=2)
        return False
    if expectation is not None and not expectation(value):
        print(f"  FAIL  {label}: unexpected result {value!r}")
        return False
    print(f"  ok    {label}: {value}")
    return True


def run():
    """Return 0 when the bundle is usable, 1 otherwise."""
    from gui.core import workspace

    print(f"frozen: {workspace.is_frozen()}")
    print(f"workspace: {workspace.workspace_root()}")
    print(f"python: {sys.version.split()[0]}")
    print()

    results = []

    def optimizer_count():
        from gui.core.optimizers import load_registry

        return len(load_registry().names)

    # The walk_packages hazard: a bundle missing mealpy submodules reports 0.
    results.append(_check("mealpy optimizers discovered", optimizer_count,
                          lambda n: n > 100))

    def thesis_set():
        from gui.core.optimizers import thesis_selection

        return len(thesis_selection())

    results.append(_check("thesis algorithm set resolves", thesis_set,
                          lambda n: n == 28))

    def bbob_problem():
        import helper_functions as hf

        suite = hf.get_suite([1], [1], [2])
        problem = next(iter(suite))
        return f"{problem.id} f(0,0)={problem([0.0, 0.0]):.4g}"

    # The compiled-extension hazard: cocoex is a C extension with data files.
    results.append(_check("cocoex BBOB suite builds and evaluates", bbob_problem))

    def bbob_instance():
        import helper_functions as hf

        return [p.id_instance for p in hf.get_suite([1], [6, 42], [2])]

    results.append(_check("instance filter is by instance number",
                          bbob_instance, lambda v: v == [6, 42]))

    def opfunu_function():
        from gui.core.problems import OpfunuSource

        source = OpfunuSource()
        catalog = source.catalog(2)
        suite = source.build([catalog[0].id], [1], [2])
        problem = next(iter(suite))
        return f"{len(catalog)} functions, {problem.name} = {problem([0.1, 0.2]):.4g}"

    results.append(_check("opfunu classic functions evaluate", opfunu_function))

    def cec_function():
        from gui.core.problems import CECSource

        source = CECSource("cec2017")
        catalog = source.catalog(10)
        suite = source.build([catalog[0].id], [1], [10])
        problem = next(iter(suite))
        return f"{len(catalog)} functions, {problem.name} = {problem([0.1] * 10):.6g}"

    # The bundled-data hazard: CEC functions need their shift/rotate matrices.
    results.append(_check("CEC suite data files are present", cec_function))

    def short_run():
        import helper_functions as hf
        from gui.core.optimizers import load_registry

        records = []
        suite = hf.get_suite([1], [1], [2])
        optimizers = load_registry().resolve(["OriginalDE"])
        hf.run_benchmarks(suite, None, optimizers, str(workspace.workspace_root() / "_selftest"),
                          seed=1, epoch_per_dim=2, pop_size=6, only_best=True,
                          progress_cb=records.append)
        return f"{len(records)} problem(s), status={records[0]['status'] if records else 'none'}"

    results.append(_check("a real optimization run completes", short_run,
                          lambda v: "status=ok" in v))

    def parquet_roundtrip():
        """The bundle ships fastparquet instead of pyarrow (146 MB smaller).
        pandas picks whichever engine is present, so this proves the
        clustering_results reader still works in the packaged app."""
        import tempfile

        import pandas as pd

        frame = pd.DataFrame({"x0": [1.0, 2.0], "cluster": [0, 1]}, index=[7, 9])
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/probe.parquet"
            frame.to_parquet(path)
            back = pd.read_parquet(path)
        engine = "fastparquet"
        try:
            import pyarrow  # noqa: F401

            engine = "pyarrow"
        except ImportError:
            pass
        return f"{back.shape} via {engine}"

    results.append(_check("parquet reader works", parquet_roundtrip))

    def browser_sees_runs():
        """A packaged app has no repo tree, so the workspace runs are the
        only data there is. If the browser cannot see them, a finished run
        looks like it saved nothing."""
        from gui.core import datastore, workspace

        roots = [node.label for node in datastore.build_tree()]
        runs = workspace.list_runs()
        if runs and "GUI runs" not in roots:
            raise RuntimeError(
                f"{len(runs)} run(s) exist in {workspace.workspace_root()} "
                f"but the data browser does not list them"
            )
        return f"{len(runs)} run(s) in the workspace, roots={roots or ['(none yet)']}"

    results.append(_check("data browser sees workspace runs", browser_sees_runs))

    def qt_window():
        """Build the real window and let its startup work finish.

        Importing Qt proves very little - a bundle can be missing the
        platform plugin and only fail when a window is actually created.
        This also exercises the background optimizer discovery and every
        panel's constructor.
        """
        import time

        from gui.qt import QtWidgets
        from gui.main_window import MainWindow

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        window = MainWindow()
        window.show()

        deadline = time.time() + 60
        while time.time() < deadline and window.setup_panel.picker._registry is None:
            app.processEvents()
            time.sleep(0.02)

        tabs = [window.tabs.tabText(i) for i in range(window.tabs.count())]
        registry = window.setup_panel.picker._registry
        window.close()
        if registry is None:
            raise RuntimeError("optimizer registry never finished loading")
        return f"{len(tabs)} tabs ({', '.join(tabs)})"

    results.append(_check("main window opens", qt_window))

    print()
    if all(results):
        print("SELF-TEST PASSED")
        return 0
    print(f"SELF-TEST FAILED ({results.count(False)} of {len(results)} checks)")
    return 1
