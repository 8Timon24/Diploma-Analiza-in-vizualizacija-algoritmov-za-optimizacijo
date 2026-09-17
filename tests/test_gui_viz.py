# Unit tests for gui.viz.capture: the layer that lets the GUI reuse the
# pipeline's plotting functions.
#
# Those functions were written for a batch pipeline - they end in plt.close()
# and several savefig() unconditionally to a hardcoded path under
# figures_entropy/ or figures_spearman/. The GUI must get the figure back
# WITHOUT writing anything, or every click would overwrite a real thesis
# figure. That guarantee is what these tests pin.
#
# matplotlib is not installed in CI (it only installs pandas/numpy/pytest),
# so these skip there and run locally.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

pytest.importorskip("matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from gui.viz.capture import offscreen_figures, render_with


def test_savefig_is_disabled_inside_the_context(tmp_path):
    target = tmp_path / "must_not_appear.pdf"

    def pipeline_style_plot():
        plt.figure()
        plt.plot([0, 1], [0, 1])
        plt.savefig(target)   # what the real functions do
        plt.close()

    with offscreen_figures():
        pipeline_style_plot()

    assert not target.exists()


def test_savefig_works_again_after_the_context(tmp_path):
    with offscreen_figures():
        plt.figure()

    target = tmp_path / "written.png"
    figure = Figure()
    figure.add_subplot(1, 1, 1).plot([0, 1], [0, 1])
    figure.savefig(target)
    assert target.exists()  # exporting must still work for the user


def test_figure_survives_the_close_the_plot_function_calls():
    def pipeline_style_plot():
        plt.figure()
        plt.plot([0, 1], [1, 0])
        plt.close()

    figure = render_with(pipeline_style_plot)
    assert isinstance(figure, Figure)
    assert len(figure.get_axes()) == 1


def test_figure_is_captured_from_show_too():
    def pipeline_style_plot():
        plt.figure()
        plt.plot([0, 1], [1, 0])
        plt.show()      # plot_entropy(save=False) takes this branch
        plt.close()

    assert isinstance(render_with(pipeline_style_plot), Figure)


def test_explicitly_returned_figure_is_preferred():
    # facet_by_dim returns (fig, stats) instead of closing.
    expected = Figure()

    def returns_a_figure():
        return expected, "some stats"

    assert render_with(returns_a_figure) is expected


def test_no_figures_are_leaked_into_pyplot():
    # Suppressing plt.close() would otherwise pile figures up in pyplot's
    # registry, one per render, until the app runs out of memory.
    before = set(plt.get_fignums())

    def pipeline_style_plot():
        plt.figure()
        plt.plot([0, 1], [0, 1])
        plt.close()

    for _ in range(5):
        render_with(pipeline_style_plot)

    assert set(plt.get_fignums()) == before


def test_a_function_that_draws_nothing_is_an_error():
    with pytest.raises(RuntimeError, match="produced no figure"):
        render_with(lambda: None)


def test_patches_are_restored_even_when_the_plot_raises():
    real_show, real_close, real_savefig = plt.show, plt.close, Figure.savefig

    def explodes():
        plt.figure()
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        render_with(explodes)

    assert plt.show is real_show
    assert plt.close is real_close
    assert Figure.savefig is real_savefig


def test_catalog_entries_are_well_formed():
    pytest.importorskip("seaborn")
    from gui.viz.registry import catalog

    entries = catalog()
    assert entries
    assert len({v.key for v in entries}) == len(entries), "keys must be unique"
    for viz in entries:
        assert viz.title and viz.description and viz.source
        assert callable(viz.render)
        for parameter in viz.parameters:
            assert parameter.kind in {"choice", "multi", "bool"}
            # A form cannot be built from choices that fail to resolve.
            parameter.resolve_choices()
            parameter.resolve_default()


# -- catalog hygiene ----------------------------------------------------

def test_no_entry_offers_both_a_single_and_a_multi_algorithm_control():
    # exploration_vs_exploitation once showed "Algorithms" (unused by its
    # render) next to "Algorithm" (the one it read). Two controls that look
    # like the same thing, one of which does nothing.
    pytest.importorskip("seaborn")
    from gui.viz.registry import catalog

    for viz in catalog():
        names = {p.name for p in viz.parameters}
        assert not {"algorithm", "algorithms"} <= names, (
            f"{viz.key} offers both a single and a multi algorithm control"
        )


def test_parameter_names_are_unique_within_an_entry():
    pytest.importorskip("seaborn")
    from gui.viz.registry import catalog

    for viz in catalog():
        names = [p.name for p in viz.parameters]
        assert len(names) == len(set(names)), f"{viz.key} has duplicate parameters"


def test_every_entry_declares_where_it_came_from():
    # The source line is how a reader finds the code behind a figure - most
    # of these were notebook cells with no file output at all.
    pytest.importorskip("seaborn")
    from gui.viz.registry import catalog

    for viz in catalog():
        assert viz.source and (".py" in viz.source or ".ipynb" in viz.source)


def test_render_never_fails_on_a_missing_parameter_key():
    """Every render must be satisfiable by its own declared defaults.

    A KeyError here means the catalog entry declares one set of parameters
    while the render function reads another - which is how
    exploration_vs_exploitation broke when its "algorithms" control was
    replaced by a single "algorithm". Missing DATA is fine and raises
    ValueError; a missing parameter KEY is a wiring bug.
    """
    pytest.importorskip("seaborn")
    from gui.viz.registry import catalog

    for viz in catalog():
        params = {p.name: p.resolve_default() for p in viz.parameters}
        try:
            viz.render(params)
        except KeyError as exc:
            raise AssertionError(
                f"{viz.key} reads parameter {exc} that it does not declare"
            ) from exc
        except Exception:
            pass  # data availability, not wiring
        finally:
            import matplotlib.pyplot as plt
            plt.close("all")
