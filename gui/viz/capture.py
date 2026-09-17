"""Run the pipeline's plotting functions and keep the figure they draw.

Those functions were written for a batch pipeline, not a GUI: they end in
plt.close(), and several call savefig() unconditionally with a hardcoded path
under figures_entropy/ or figures_spearman/. Calling them as-is from the GUI
would therefore (a) hand back nothing to display and (b) silently overwrite
the user's real thesis figures every time someone clicked a plot.

offscreen_figures() neutralises all three exit paths - show, savefig and
close - and hands back the Figure instead. Rendering in the GUI becomes a
pure read: no file in any figures_* directory is touched. Exporting is a
separate, explicit action the user takes.

Reusing the real functions this way is the point: the GUI shows exactly the
figure the pipeline produces, with no second implementation to drift.
"""
from contextlib import contextmanager

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


@contextmanager
def offscreen_figures():
    """Yield a list that collects every figure the wrapped code tried to
    show, save or close. Nothing is written to disk."""
    captured = []

    def remember(fig):
        if fig is not None and fig not in captured:
            captured.append(fig)

    real_show = plt.show
    real_close = plt.close
    real_savefig = Figure.savefig

    def fake_show(*_args, **_kwargs):
        remember(plt.gcf())

    def fake_close(*args, **_kwargs):
        # Capture before discarding; the figure itself is kept alive so the
        # caller can still display it.
        if args and isinstance(args[0], Figure):
            remember(args[0])
        else:
            remember(plt.gcf())

    def fake_savefig(self, *_args, **_kwargs):
        remember(self)  # deliberately does not write

    plt.show = fake_show
    plt.close = fake_close
    Figure.savefig = fake_savefig
    try:
        yield captured
    finally:
        plt.show = real_show
        plt.close = real_close
        Figure.savefig = real_savefig
        # Hand the figures over to the GUI: drop them from pyplot's registry
        # so they are not leaked there, while keeping the objects alive for a
        # fresh Qt canvas to adopt.
        for fig in captured:
            _detach(fig)


def _detach(fig):
    """Remove a figure from pyplot's registry, keeping the object alive."""
    try:
        plt._pylab_helpers.Gcf.destroy_fig(fig)
    except Exception:
        pass


def render_with(plot_callable, *args, **kwargs):
    """Call a pipeline plotting function and return the figure it drew.

    Returns the last figure captured, which is the one the function was
    building when it tried to save or close.
    """
    with offscreen_figures() as figures:
        result = plot_callable(*args, **kwargs)

    # facet_by_dim and friends already hand a figure back explicitly. Those
    # never hit show/savefig/close, so they are still registered with pyplot
    # and must be detached here too, or every render leaks one.
    returned = None
    if isinstance(result, Figure):
        returned = result
    elif isinstance(result, tuple):
        returned = next((item for item in result if isinstance(item, Figure)), None)

    if returned is not None:
        _detach(returned)
        return returned
    if figures:
        return figures[-1]
    raise RuntimeError(
        f"{getattr(plot_callable, '__name__', plot_callable)} produced no figure"
    )
