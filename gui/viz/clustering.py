"""Cluster-occupancy and algorithm-similarity views.

These read the clustering stage's output: the per-iteration occupancy counts,
the cluster centres that say where each cluster sits, and the aggregate
cosine similarity that cluster_similarity.py writes. All three previously
lived only in 05_analysis/3_cluster_analysis.ipynb.
"""
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from gui.viz import style
from gui.viz.capture import render_with
from gui.viz.data import (
    load_algorithm_similarity, load_cluster_occupancy,
    load_clustered_trajectories, problem_key,
)


def _occupancy_for(occupancy, algorithm, run):
    try:
        return occupancy.loc[(algorithm, run)]
    except KeyError:
        raise ValueError(
            f"No cluster occupancy for {algorithm}, run {run} on this problem."
        ) from None


def _label_columns(frame, labels):
    """Name each cluster column with where that cluster actually is."""
    if not labels:
        return frame
    return frame.rename(
        columns={c: f"{int(c)}\n({labels[int(c)]})" for c in frame.columns
                 if int(c) in labels}
    )


def _draw_occupancy(axes, frame, labels, population, annotate, title):
    frame = _label_columns(frame, labels)
    sns.heatmap(
        frame.replace(0, np.nan), ax=axes, cmap="YlGnBu",
        annot=annotate, fmt=".0f", vmin=0, vmax=population,
        cbar_kws={"label": "agents in cluster"},
    )
    axes.set_title(title, fontsize=11, fontweight="bold")
    axes.set_xlabel("Cluster")
    axes.set_ylabel("Iteration")


def cluster_occupancy(params):
    """Where one algorithm's population sits, cluster by cluster, over time."""
    dimension, function, instance = params["dimension"], params["function"], params["instance"]
    algorithm, run = params["algorithm"], params["run"]

    occupancy, labels = load_cluster_occupancy(dimension, function, instance)
    frame = _occupancy_for(occupancy, algorithm, run)
    annotate = frame.shape[1] <= 32

    height = max(6, frame.shape[0] * 0.35)
    width = max(9, frame.shape[1] * 0.42)
    figure = Figure(figsize=(width, height), layout="constrained")

    if params.get("show_fitness"):
        axes = figure.subplots(1, 2, width_ratios=[3, 2])
        occupancy_axes, fitness_axes = axes[0], axes[1]
    else:
        occupancy_axes, fitness_axes = figure.add_subplot(1, 1, 1), None

    _draw_occupancy(
        occupancy_axes, frame, labels, params["population"], annotate,
        f"{algorithm} - run {run} - {problem_key(function, instance)} (dim {dimension})",
    )

    if fitness_axes is not None:
        trajectories = load_clustered_trajectories(dimension, function, instance)
        subset = trajectories.query("algorithm == @algorithm and run == @run")
        if subset.empty:
            raise ValueError(f"No clustered trajectories for {algorithm}, run {run}.")
        best = (
            subset.groupby(["iteration", "cluster"])["scaled_raw_y"].min()
            .cummin().to_frame().reset_index()
        )
        # legend=False coloured every cluster and then said which was which
        # nowhere at all.
        sns.lineplot(data=best, x="iteration", y="scaled_raw_y", hue="cluster",
                     palette="tab10", ax=fitness_axes)
        fitness_axes.legend(title="Cluster", ncol=2, loc="upper right")
        fitness_axes.set_xlabel("Iteration")
        fitness_axes.set_ylabel("Best scaled fitness (running minimum)")
        fitness_axes.set_title("Best fitness found per cluster", fontsize=11)

    return figure


def cluster_occupancy_compare(params):
    """The same occupancy view for several algorithms, stacked for comparison."""
    dimension, function, instance = params["dimension"], params["function"], params["instance"]
    run = params["run"]
    algorithms = list(params["algorithms"])
    if not algorithms:
        raise ValueError("Select at least one algorithm.")
    if len(algorithms) > 6:
        raise ValueError(
            f"{len(algorithms)} algorithms would make an unreadably tall figure. "
            "Pick at most 6."
        )

    occupancy, labels = load_cluster_occupancy(dimension, function, instance)
    frames = [(a, _occupancy_for(occupancy, a, run)) for a in algorithms]
    columns = frames[0][1].shape[1]
    annotate = columns <= 32

    width = max(9, columns * 0.42)
    height = max(5, frames[0][1].shape[0] * 0.3) * len(frames)
    figure = Figure(figsize=(width, height), layout="constrained")
    axes_list = np.atleast_1d(figure.subplots(len(frames), 1))

    for axes, (algorithm, frame) in zip(axes_list, frames):
        _draw_occupancy(
            axes, frame, labels, params["population"], annotate,
            f"{algorithm} - run {run} - {problem_key(function, instance)} (dim {dimension})",
        )
    return figure


def algorithm_similarity(params):
    """Algorithms clustered by their aggregate trajectory similarity.

    Leaf colours are the mealpy family, so it is immediately visible whether
    the clustering recovers the families or cuts across them - which is the
    interesting question.
    """
    dimension = params["dimension"]
    matrix = load_algorithm_similarity(dimension, params["statistic"])

    algorithms = list(params["algorithms"]) or list(matrix.index)
    present = [a for a in algorithms if a in matrix.index]
    if len(present) < 3:
        raise ValueError("Need at least 3 algorithms present in the similarity matrix.")
    matrix = matrix.loc[present, present].fillna(0.0)

    families = {a: style.family_of(a) for a in present}
    colours = style.family_colours()
    row_colours = [colours.get(families[a], "#999999") for a in present]

    size = max(8, len(present) * 0.45)

    def draw():
        grid = sns.clustermap(
            matrix, row_colors=row_colours, col_colors=row_colours,
            cmap="YlGnBu", figsize=(size, size),
            annot=len(present) <= 12, fmt=".2f",
            # Shrink and raise the colour bar so the family legend below it
            # has room; seaborn's default cbar_pos overlaps that corner.
            cbar_pos=(0.02, 0.84, 0.03, 0.12),
            cbar_kws={"label": "aggregate cosine similarity"},
        )
        # Labels must be pulled in dendrogram order or every row is mislabelled.
        order = grid.dendrogram_row.reordered_ind
        grid.ax_heatmap.set_yticks([i + 0.5 for i in range(len(present))])
        grid.ax_heatmap.set_yticklabels([matrix.index[i] for i in order],
                                        fontsize=8, rotation=0)
        column_order = grid.dendrogram_col.reordered_ind
        grid.ax_heatmap.set_xticks([i + 0.5 for i in range(len(present))])
        grid.ax_heatmap.set_xticklabels([matrix.columns[i] for i in column_order],
                                        fontsize=8, rotation=45, ha="right")
        grid.ax_heatmap.set_xlabel("")
        grid.ax_heatmap.set_ylabel("")

        used = sorted({families[a] for a in present})
        grid.figure.legend(
            handles=[Patch(color=colours.get(f, "#999999"),
                           label=style.family_label(f)) for f in used],
            title="mealpy family", loc="upper left",
            bbox_to_anchor=(0.005, 0.80), fontsize=8, frameon=False,
        )
        grid.figure.suptitle(
            f"Algorithm similarity ({params['statistic']}, dim {dimension})",
            y=0.995,
        )
        return grid.figure

    return render_with(draw)
