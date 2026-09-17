"""Exploration / exploitation views.

These four plots existed only as cells in
05_analysis/exploration_plots_preview.ipynb - three of the four saved a PDF,
but all of them required opening and running the notebook to see anything.
They read the per-run diversity_{seed}.csv files, which the benchmark only
writes when it is run with -e.
"""
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure

from gui.viz import style
from gui.viz.capture import render_with
from gui.viz.data import load_diversity_curves, pairwise_matrix

# The boundary the ClustOpt-style exploration measure is read against: above
# 50% the population is still spreading out, below it is converging.
EXPLORATION_BOUNDARY = 50.0


def _curves(params, algorithms=None):
    """Load the diversity curves this view needs.

    Views that take a single algorithm pass it explicitly rather than
    relying on an "algorithms" key they do not declare.
    """
    return load_diversity_curves(
        params["dimension"], algorithms if algorithms is not None else params["algorithms"],
        params["function"], params["instance"], params["seeds"],
    )


def _problem_title(params):
    return (
        f"F{params['function']}_I{params['instance']}, dim {params['dimension']}"
    )


def _require(data, params):
    if data.empty:
        raise ValueError(
            "No diversity data for this selection. The benchmark writes "
            "diversity_{seed}.csv only when run with -e."
        )


def mean_exploration(params):
    """Mean exploration % per iteration, one line per algorithm."""
    data = _curves(params)
    _require(data, params)

    means = data.groupby(["algorithm", "iteration"])["exploration"].mean().reset_index()
    order = sorted(means["algorithm"].unique())
    palette = style.distinct_palette(order)

    figure = Figure(figsize=(11, 5), layout="constrained")
    axes = figure.add_subplot(1, 1, 1)
    sns.lineplot(data=means, x="iteration", y="exploration", hue="algorithm",
                 hue_order=order, palette=palette, errorbar=None, ax=axes)
    # Labelled, so it reaches the legend: an unexplained dashed line across
    # the middle of the plot told the reader nothing.
    axes.axhline(EXPLORATION_BOUNDARY, color="gray", linestyle="--", alpha=0.6,
                 label=f"{EXPLORATION_BOUNDARY:.0f}% - even split")
    axes.set_ylim(0, 100)
    axes.set_xlabel("Iteration")
    axes.set_ylabel("Exploration (%)")
    axes.set_title(f"Exploration over iterations ({_problem_title(params)})")
    axes.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Algorithm")
    return figure


def exploration_vs_exploitation(params):
    """One algorithm's exploration and exploitation, with the spread across runs."""
    algorithm = params["algorithm"]
    data = _curves(params, algorithms=[algorithm])
    _require(data, params)

    subset = data[data["algorithm"] == algorithm]
    if subset.empty:
        raise ValueError(f"No diversity data for {algorithm} on this problem.")

    aggregated = subset.groupby("iteration").agg(
        exploration=("exploration", "mean"),
        exploration_min=("exploration", "min"),
        exploration_max=("exploration", "max"),
        exploitation=("exploitation", "mean"),
    ).reset_index()

    figure = Figure(figsize=(11, 5), layout="constrained")
    axes = figure.add_subplot(1, 1, 1)
    axes.fill_between(aggregated["iteration"], aggregated["exploration_min"],
                      aggregated["exploration_max"], color="#457b9d", alpha=0.15,
                      label="Exploration spread across runs")
    axes.plot(aggregated["iteration"], aggregated["exploration"], color="#457b9d",
              marker="o", markersize=3, label="Exploration")
    axes.plot(aggregated["iteration"], aggregated["exploitation"], color="#e63946",
              marker="o", markersize=3, label="Exploitation")
    axes.axhline(EXPLORATION_BOUNDARY, color="gray", linestyle="--", alpha=0.5)
    axes.set_ylim(0, 100)
    axes.set_xlabel("Iteration")
    axes.set_ylabel("%")
    axes.set_title(f"{algorithm}: exploration vs exploitation ({_problem_title(params)})")
    axes.legend()
    return figure


def exploration_difference_clustermap(params):
    """Pairwise mean |delta exploration %|, clustered.

    Seed-paired: run 1 of one algorithm is compared against run 1 of the
    other, so the difference is not inflated by run-to-run variance.
    """
    data = _curves(params)
    _require(data, params)

    algorithms = [a for a in params["algorithms"] if a in set(data["algorithm"])]
    if len(algorithms) < 2:
        raise ValueError("A clustermap needs at least 2 algorithms with data.")

    def curve(algorithm, run):
        rows = data[(data["algorithm"] == algorithm) & (data["run"] == run)]
        return rows.sort_values("iteration")["exploration"].to_numpy()

    def difference(first, second):
        per_run = []
        for run in params["seeds"]:
            a, b = curve(first, run), curve(second, run)
            if len(a) == len(b) and len(a):
                per_run.append(np.abs(a - b).mean())
        return float(np.mean(per_run)) if per_run else np.nan

    matrix = pairwise_matrix(algorithms, difference)
    size = max(9, len(algorithms) * 1.1)

    def draw():
        grid = sns.clustermap(
            matrix, cmap="YlOrRd", annot=len(algorithms) <= 12, fmt=".1f",
            figsize=(size, size),
            cbar_kws={"label": "mean |difference| in exploration (%)"},
        )
        # Labels must follow the dendrogram order, or every row is mislabelled.
        rows = grid.dendrogram_row.reordered_ind
        grid.ax_heatmap.set_yticks([i + 0.5 for i in range(len(algorithms))])
        grid.ax_heatmap.set_yticklabels([matrix.index[i] for i in rows],
                                        fontsize=9, rotation=0)
        columns = grid.dendrogram_col.reordered_ind
        grid.ax_heatmap.set_xticks([i + 0.5 for i in range(len(algorithms))])
        grid.ax_heatmap.set_xticklabels([matrix.columns[i] for i in columns],
                                        fontsize=9, rotation=45, ha="right")
        grid.ax_heatmap.set_xlabel("")
        grid.ax_heatmap.set_ylabel("")
        # Inside the layout box: at y=1.02 the title sat outside the figure
        # and was clipped on the live canvas (bbox_inches="tight" only saved
        # it on export).
        grid.figure.suptitle(
            f"Difference in exploration between algorithms ({_problem_title(params)})",
            y=0.995,
        )
        return grid.figure

    return render_with(draw)


def raw_diversity(params):
    """Un-normalised population diversity per iteration."""
    data = _curves(params)
    _require(data, params)

    curves = data.groupby(["algorithm", "iteration"])["diversity"].mean().reset_index()
    order = sorted(curves["algorithm"].unique())

    figure = Figure(figsize=(11, 5), layout="constrained")
    axes = figure.add_subplot(1, 1, 1)
    sns.lineplot(data=curves, x="iteration", y="diversity", hue="algorithm",
                 hue_order=order, palette=style.distinct_palette(order),
                 errorbar=None, ax=axes)
    axes.set_xlabel("Iteration")
    axes.set_ylabel("Diversity (raw)")
    axes.set_title(f"Raw population diversity ({_problem_title(params)})")
    axes.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7,
                title="Algorithm")
    return figure
