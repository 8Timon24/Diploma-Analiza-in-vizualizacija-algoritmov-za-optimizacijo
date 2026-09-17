"""Final-solution quality and location views.

All four existed only as inline cells in
05_analysis/location_fitness_preview.ipynb - none of them wrote a file, so
the only way to see one was to open the notebook and re-run it.

They compare where algorithms actually ended up, against the problem's true
optimum, which cocoex does not expose publicly (see data.bbob_optimum).
"""
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure

from gui.viz import style
from gui.viz.data import (
    coordinate_columns, pairwise_matrix, results_with_precision,
)


def _subset(params):
    subset, f_opt, x_opt = results_with_precision(
        params["dimension"], params["function"], params["instance"],
        params["algorithms"],
    )
    if subset.empty:
        raise ValueError(
            f"No rows in outputs/dim_{params['dimension']}/results.csv for "
            f"F{params['function']}_I{params['instance']} and these algorithms."
        )
    return subset, f_opt, x_opt


def _problem_title(params):
    return f"F{params['function']}_I{params['instance']}, dim {params['dimension']}"


def precision_distribution(params):
    """Final precision per algorithm across seeds, best first, on a log axis."""
    subset, f_opt, _x_opt = _subset(params)
    order = subset.groupby("algorithm")["precision"].median().sort_values().index

    figure = Figure(figsize=(12, 6), layout="constrained")
    axes = figure.add_subplot(1, 1, 1)
    sns.boxplot(data=subset, x="algorithm", y="precision", order=order,
                color="lightsteelblue", showfliers=False, ax=axes)
    sns.stripplot(data=subset, x="algorithm", y="precision", order=order,
                  color="black", size=4, alpha=0.6, ax=axes)
    axes.set_yscale("log")
    axes.set_xlabel("Algorithm")
    axes.set_ylabel("Precision  f - f*  (log scale)")
    axes.set_title(
        f"Final precision per algorithm ({_problem_title(params)}) - lower is better"
        f"\nf* = {f_opt:.6g}"
    )
    axes.tick_params(axis="x", rotation=45)
    for label in axes.get_xticklabels():
        label.set_horizontalalignment("right")
    return figure


def _seed_column(subset):
    return "seed" if "seed" in subset.columns else "run"


def precision_difference(params):
    """Pairwise mean |delta log10 precision|, seed-paired."""
    subset, _f_opt, _x_opt = _subset(params)
    algorithms = [a for a in params["algorithms"] if a in set(subset["algorithm"])]
    if len(algorithms) < 2:
        raise ValueError("Need at least 2 algorithms with results.")

    seed_column = _seed_column(subset)
    seeds = sorted(subset[seed_column].unique())

    def log_precision(algorithm, seed):
        rows = subset[(subset["algorithm"] == algorithm) & (subset[seed_column] == seed)]
        return None if rows.empty else float(np.log10(rows["precision"].iloc[0]))

    def difference(first, second):
        values = []
        for seed in seeds:
            a, b = log_precision(first, seed), log_precision(second, seed)
            if a is not None and b is not None:
                values.append(abs(a - b))
        return float(np.mean(values)) if values else np.nan

    matrix = pairwise_matrix(algorithms, difference)

    figure = Figure(figsize=(11, 9), layout="constrained")
    axes = figure.add_subplot(1, 1, 1)
    sns.heatmap(matrix, annot=len(algorithms) <= 12, fmt=".2f", cmap="YlOrRd",
                square=True, ax=axes,
                cbar_kws={"label": "mean |difference| in log10 precision"})
    axes.set_title(f"Pairwise difference in final quality ({_problem_title(params)})")
    axes.tick_params(axis="x", rotation=45)
    for label in axes.get_xticklabels():
        label.set_horizontalalignment("right")
    return figure


def location_difference(params):
    """Pairwise Euclidean distance between the final best points, seed-paired."""
    subset, _f_opt, _x_opt = _subset(params)
    algorithms = [a for a in params["algorithms"] if a in set(subset["algorithm"])]
    if len(algorithms) < 2:
        raise ValueError("Need at least 2 algorithms with results.")

    coordinates = coordinate_columns(subset)
    seed_column = _seed_column(subset)
    seeds = sorted(subset[seed_column].unique())

    def point(algorithm, seed):
        rows = subset[(subset["algorithm"] == algorithm) & (subset[seed_column] == seed)]
        return None if rows.empty else rows[coordinates].to_numpy()[0]

    def difference(first, second):
        values = []
        for seed in seeds:
            a, b = point(first, seed), point(second, seed)
            if a is not None and b is not None:
                values.append(float(np.linalg.norm(a - b)))
        return float(np.mean(values)) if values else np.nan

    matrix = pairwise_matrix(algorithms, difference)

    figure = Figure(figsize=(11, 9), layout="constrained")
    axes = figure.add_subplot(1, 1, 1)
    sns.heatmap(matrix, annot=len(algorithms) <= 12, fmt=".2f", cmap="YlGnBu",
                square=True, ax=axes,
                cbar_kws={"label": "mean Euclidean distance between final points"})
    axes.set_title(f"Pairwise distance between final solutions ({_problem_title(params)})")
    axes.tick_params(axis="x", rotation=45)
    for label in axes.get_xticklabels():
        label.set_horizontalalignment("right")
    return figure


def final_positions(params):
    """Where every run actually ended up, against the true optimum.

    Two dimensions only - there is no honest way to draw a 10-D endpoint on
    a plane, and projecting would imply a closeness that is not there.
    """
    if params["dimension"] != 2:
        raise ValueError(
            "This view is 2-D only: it plots the final points in the real "
            "search space. Select dimension 2."
        )

    subset, f_opt, x_opt = _subset(params)
    coordinates = coordinate_columns(subset)
    if len(coordinates) < 2:
        raise ValueError("results.csv does not carry two coordinate columns.")

    algorithms = sorted(subset["algorithm"].unique())
    palette = style.distinct_palette(algorithms)

    figure = Figure(figsize=(8.5, 8), layout="constrained")
    axes = figure.add_subplot(1, 1, 1)
    for algorithm in algorithms:
        rows = subset[subset["algorithm"] == algorithm]
        axes.scatter(rows[coordinates[0]], rows[coordinates[1]],
                     color=palette.get(algorithm), s=55, alpha=0.85,
                     edgecolor="white", linewidth=0.6, label=algorithm)
    axes.scatter(x_opt[0], x_opt[1], marker="*", s=420, color="red",
                 edgecolor="black", linewidth=0.6, zorder=5, label="true optimum")
    # The raw dataframe column names are "x1"/"x2"; say what they are.
    axes.set_xlabel(f"{coordinates[0]}  (search space)")
    axes.set_ylabel(f"{coordinates[1]}  (search space)")
    axes.set_title(
        f"Final best solutions in the search space ({_problem_title(params)})"
        f"\nf* = {f_opt:.6g} at ({x_opt[0]:.4g}, {x_opt[1]:.4g})"
    )
    axes.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    return figure
