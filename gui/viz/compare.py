"""Head-to-head comparison of two algorithms.

metrics_data/ exists to answer "how do these two differ, and do the metrics
agree about it" - but answering it previously meant opening a notebook and
filtering a 227k-row table by hand.

Every view here puts the pair in context rather than showing a bare number.
A cosine distance of 0.51 means nothing alone; that it sits at the 8th
percentile of all 378 pairs means these two are unusually alike.
"""
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from gui.viz import style
from gui.viz.data import load_diversity_curves, load_merged
from gui.viz.metrics import METRICS

# merged_dim_{d}.csv always stores a pair with Algorithm1 alphabetically
# first, so a lookup has to normalise the order or it silently finds nothing.
def ordered_pair(first, second):
    return (first, second) if first < second else (second, first)


def _validate(first, second):
    if first == second:
        raise ValueError("Pick two different algorithms.")
    return ordered_pair(first, second)


def _all_pair_means(dimension):
    """Mean of each metric per pair, over every problem. 378 pairs for the
    thesis set - this is the distribution a pair is judged against."""
    merged = load_merged(dimension)
    available = [m for m in METRICS if m in merged.columns]
    if not available:
        raise ValueError(f"merged_dim_{dimension}.csv carries none of {METRICS}.")
    return merged.groupby(["Algorithm1", "Algorithm2"])[available].mean(), available


def _pair_values(means, pair):
    if pair not in means.index:
        raise ValueError(
            f"{pair[0]} and {pair[1]} are not compared in the merged table for "
            f"this dimension."
        )
    return means.loc[pair]


def metric_profile(params):
    """Where this pair sits in the distribution of every pair, per metric.

    Each row is one metric: every pair as a faint dot, this one marked. A
    pair that sits at the same end of every row is one the metrics agree
    about; a pair that jumps around is where they disagree, which is the
    thesis's whole question.
    """
    dimension = params["dimension"]
    pair = _validate(params["algorithm_a"], params["algorithm_b"])
    means, available = _all_pair_means(dimension)
    values = _pair_values(means, pair)

    figure = Figure(figsize=(11, 1.15 * len(available) + 1.8), layout="constrained")
    axes_list = np.atleast_1d(figure.subplots(len(available), 1))

    rng = np.random.default_rng(0)   # jitter must not change between renders
    for axes, metric in zip(axes_list, available):
        column = means[metric].to_numpy()
        axes.scatter(column, rng.normal(0, 0.06, size=len(column)),
                     s=16, color="#b0b0b0", alpha=0.45, linewidths=0)

        value = float(values[metric])
        percentile = float((column <= value).mean() * 100)
        axes.scatter([value], [0], s=190, marker="D", color="#d62728",
                     edgecolor="black", linewidth=0.7, zorder=5)
        axes.axvline(value, color="#d62728", alpha=0.35, linewidth=1)

        axes.set_yticks([])
        axes.set_ylim(-0.3, 0.3)
        axes.set_ylabel(style.metric_label(metric), rotation=0, ha="right",
                        va="center", fontsize=10)
        axes.text(
            0.995, 0.86,
            f"{value:.3f}   -   {percentile:.0f}th percentile of {len(column)} pairs",
            transform=axes.transAxes, ha="right", va="top", fontsize=9,
            color="#d62728",
        )
        for spine in ("top", "right", "left"):
            axes.spines[spine].set_visible(False)

    axes_list[-1].set_xlabel("metric value (each dot is one algorithm pair)")
    figure.suptitle(
        f"{pair[0]}  vs  {pair[1]}   (dim {dimension})\n"
        f"low percentile = unusually similar, high = unusually different",
        fontsize=12,
    )
    return figure


def per_function_profile(params):
    """Which problems the two actually differ on.

    Values are percentile ranks computed within each function separately, so
    a column says "for this function, how does this pair compare with every
    other pair" - which is comparable across functions in a way raw values
    are not.
    """
    dimension = params["dimension"]
    pair = _validate(params["algorithm_a"], params["algorithm_b"])

    merged = load_merged(dimension)
    available = [m for m in METRICS if m in merged.columns]
    per_function = (
        merged.groupby(["Algorithm1", "Algorithm2", "Function_id"])[available]
        .mean().reset_index()
    )
    ranked = per_function.copy()
    for metric in available:
        ranked[metric] = per_function.groupby("Function_id")[metric].rank(pct=True)

    rows = ranked[
        (ranked["Algorithm1"] == pair[0]) & (ranked["Algorithm2"] == pair[1])
    ]
    if rows.empty:
        raise ValueError(f"No per-function rows for {pair[0]} vs {pair[1]}.")

    table = rows.set_index("Function_id")[available].T
    table.index = [style.metric_label(m) for m in table.index]
    table.columns = [f"F{int(f)}" for f in table.columns]

    figure = Figure(figsize=(max(11, 0.5 * table.shape[1] + 4), 5), layout="constrained")
    axes = figure.add_subplot(1, 1, 1)
    sns.heatmap(table, cmap="RdYlBu_r", vmin=0, vmax=1, annot=table.shape[1] <= 24,
                fmt=".2f", linewidths=0.4, ax=axes,
                cbar_kws={"label": "percentile among all pairs on that function"})
    axes.set_xlabel("BBOB function")
    axes.set_ylabel("")
    axes.set_title(
        f"{pair[0]}  vs  {pair[1]}   (dim {dimension})\n"
        f"red = more different than most pairs on that function, blue = more alike"
    )
    return figure


def behaviour(params):
    """What the two actually did, side by side, on one problem.

    The metrics say how far apart they are; this says what the difference
    looks like - entropy collapsing at different rates, or one staying in
    exploration long after the other has committed.
    """
    dimension = params["dimension"]
    first, second = params["algorithm_a"], params["algorithm_b"]
    if first == second:
        raise ValueError("Pick two different algorithms.")
    algorithms = [first, second]
    function, instance = params["function"], params["instance"]
    palette = style.distinct_palette(algorithms)

    figure = Figure(figsize=(13, 5), layout="constrained")
    entropy_axes, exploration_axes = figure.subplots(1, 2)

    # -- entropy, from the aggregated entropy table
    from gui.viz.registry import _entropy_table

    drawn = False
    try:
        table = _entropy_table(dimension)
        subset = table[
            table["algorithm"].isin(algorithms) & (table["function_class"] == function)
        ]
        if not subset.empty:
            sns.lineplot(data=subset, x="iteration", y="mean_entropy",
                         hue="algorithm", hue_order=algorithms, palette=palette,
                         errorbar=None, ax=entropy_axes)
            drawn = True
    except Exception:
        drawn = False
    if not drawn:
        entropy_axes.text(0.5, 0.5, "No entropy data for this selection",
                          ha="center", va="center", transform=entropy_axes.transAxes,
                          color="gray")
    entropy_axes.set_ylim(0, 1)
    entropy_axes.set_xlabel("Iteration")
    entropy_axes.set_ylabel("Normalized entropy (H / ln k)")
    entropy_axes.set_title("Population entropy")

    # -- exploration, from the per-run diversity files
    curves = load_diversity_curves(
        dimension, algorithms, function, instance, params["seeds"]
    )
    if curves.empty:
        exploration_axes.text(
            0.5, 0.5,
            "No diversity data\n(benchmark must be run with -e)",
            ha="center", va="center", transform=exploration_axes.transAxes,
            color="gray",
        )
    else:
        means = (
            curves.groupby(["algorithm", "iteration"])["exploration"]
            .mean().reset_index()
        )
        sns.lineplot(data=means, x="iteration", y="exploration", hue="algorithm",
                     hue_order=[a for a in algorithms if a in set(means["algorithm"])],
                     palette=palette, errorbar=None, ax=exploration_axes)
        exploration_axes.axhline(50, color="gray", linestyle="--", alpha=0.6,
                                 label="50% - even split")
    exploration_axes.set_ylim(0, 100)
    exploration_axes.set_xlabel("Iteration")
    exploration_axes.set_ylabel("Exploration (%)")
    exploration_axes.set_title("Exploration vs exploitation")
    if exploration_axes.get_legend_handles_labels()[0]:
        exploration_axes.legend(loc="upper right")

    figure.suptitle(
        f"{first}  vs  {second}   -   F{function}_I{instance}, dim {dimension}",
        fontsize=12,
    )
    return figure


def summary_table(dimension, algorithm_a, algorithm_b):
    """The same numbers as metric_profile, as text for the panel header."""
    pair = _validate(algorithm_a, algorithm_b)
    means, available = _all_pair_means(dimension)
    values = _pair_values(means, pair)
    rows = []
    for metric in available:
        column = means[metric].to_numpy()
        value = float(values[metric])
        rows.append({
            "metric": style.metric_label(metric),
            "value": value,
            "percentile": float((column <= value).mean() * 100),
        })
    return pd.DataFrame(rows), len(means)
