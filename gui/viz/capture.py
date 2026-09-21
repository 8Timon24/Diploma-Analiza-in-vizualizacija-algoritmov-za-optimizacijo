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
import threading
from contextlib import contextmanager

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from gui.viz import figure_theme

# plt.show/plt.close/Figure.savefig can only be replaced process-wide, so the
# patch has to be installed globally but BEHAVE per-thread.
#
# The Process tab runs pipeline stages in process on a worker thread, and
# 05_analysis/spearman.py writes a real PDF through Figure.savefig. A patch
# that suppressed unconditionally turned that write into a silent no-op
# whenever the user happened to render a figure at the same time - the stage
# still reported success. So the fakes below apply only to the thread that
# installed them; every other thread gets the real function.
#
# The lock covers install/restore, so two overlapping contexts can never
# restore the FAKES as if they were the originals, which would have killed
# savefig for the rest of the session.
_PATCH_LOCK = threading.RLock()

#: thread whose show/savefig/close calls are currently being intercepted
_owner = None


@contextmanager
def offscreen_figures():
    """Yield a list that collects every figure the wrapped code tried to
    show, save or close. Nothing is written to disk."""
    captured = []

    def remember(fig):
        if fig is not None and fig not in captured:
            captured.append(fig)

    global _owner

    with _PATCH_LOCK:
        # Anything already registered with pyplot belongs to someone else and
        # must be left alone; everything that appears while we hold the patch
        # is ours to clean up.
        pre_existing = set(plt.get_fignums())

        real_show, real_close, real_savefig = plt.show, plt.close, Figure.savefig
        # Saved and restored rather than cleared, so a nested context (the
        # lock is re-entrant) hands ownership back to the outer one instead
        # of leaving it delegating to the real savefig mid-render.
        previous_owner = _owner
        _owner = threading.current_thread()

        def mine():
            return threading.current_thread() is _owner

        def fake_show(*args, **kwargs):
            if not mine():
                return real_show(*args, **kwargs)
            remember(_current_figure())

        def fake_close(*args, **kwargs):
            if not mine():
                return real_close(*args, **kwargs)
            # Capture before discarding; the figure itself is kept alive so
            # the caller can still display it.
            if args and isinstance(args[0], Figure):
                remember(args[0])
            else:
                remember(_current_figure())

        def fake_savefig(self, *args, **kwargs):
            if not mine():
                return real_savefig(self, *args, **kwargs)
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
            _owner = previous_owner
            # Hand the figures over to the GUI: drop them from pyplot's
            # registry so they are not leaked there, while keeping the
            # objects alive for a fresh Qt canvas to adopt. Detaching only
            # the CAPTURED ones leaked a figure on every failed render - a
            # plot function that raises after plt.figure() never reaches
            # show/savefig/close, so its figure stayed in Gcf forever.
            for number in set(plt.get_fignums()) - pre_existing:
                manager = plt._pylab_helpers.Gcf.figs.get(number)
                if manager is not None:
                    remember(manager.canvas.figure)
            for fig in captured:
                _detach(fig)


def _current_figure():
    """The active figure, or None.

    Deliberately not plt.gcf(), which CREATES a figure when there is none:
    a plot function that called show() or close() without drawing anything
    then handed back a blank white canvas instead of raising.
    """
    manager = plt._pylab_helpers.Gcf.get_active()
    return None if manager is None else manager.canvas.figure


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

    The figure is drawn under the app's matplotlib style and then corrected
    in place, because seaborn's clustermaps and FacetGrids fix their colours
    at construction and ignore the surrounding rc_context.
    """
    with figure_theme.styled():
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
        return _finish(returned)
    if figures:
        return _finish(figures[-1])
    raise RuntimeError(
        f"{getattr(plot_callable, '__name__', plot_callable)} produced no figure"
    )


def _finish(figure):
    figure_theme.apply_to(figure)
    figure_theme.cap_figure_size(figure)
    return figure
