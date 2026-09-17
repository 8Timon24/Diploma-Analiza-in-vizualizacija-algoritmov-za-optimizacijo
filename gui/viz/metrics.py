"""Views on the relationships between the six pairwise metrics.

This is the thesis's actual question - which metrics say the same thing and
which add something - so these three answer it three ways: the raw values per
pair, the clustering each metric induces, and what the newer metrics reveal
about pairs that the reference metric calls alike or unalike.

Ported from 05_analysis/pari_algoritmov_metrike_heatmap.ipynb (inline plus
two PDFs) and scratch/slika_dodana_vrednost.py.
"""
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

from gui.viz import style
from gui.viz.data import load_merged, pair_means

METRICS = ["entropy", "cosine", "cosine_columns", "exploration", "location", "fitness"]

# Metrics whose raw units are unbounded, so they need rescaling before they
# can share a 0-1 colour scale with the others.
UNBOUNDED = ["location", "fitness"]

# The metric the thesis treats as the reference (the ClustOpt cosine distance)
# and therefore the one the others are judged as adding value over.
REFERENCE_METRIC = "cosine"


def _selected(params):
    algorithms = list(params["algorithms"])
    if len(algorithms) < 2:
        raise ValueError("Select at least 2 algorithms.")
    return algorithms


def _labels():
    return {m: style.metric_label(m) for m in METRICS}


def metric_table(params):
    """Every algorithm pair against every metric, on a shared 0-1 scale.

    exploration is a percentage and location/fitness are unbounded, so they
    are rescaled within the current selection - which means the colours are
    comparable across columns but only within this set of algorithms.
    """
    algorithms = _selected(params)
    means = pair_means(params["dimension"], METRICS, algorithms)

    table = means.copy()
    table["pair"] = table["Algorithm1"] + "  /  " + table["Algorithm2"]
    table = table.set_index("pair")[[m for m in METRICS if m in table.columns]]

    scaled = table.copy()
    if "exploration" in scaled:
        scaled["exploration"] = scaled["exploration"] / 100.0
    for metric in UNBOUNDED:
        if metric not in scaled:
            continue
        low, high = scaled[metric].min(), scaled[metric].max()
        scaled[metric] = (scaled[metric] - low) / (high - low) if high > low else 0.0
    scaled = scaled.rename(columns=_labels())

    figure = Figure(figsize=(10, 0.42 * len(scaled) + 3), layout="constrained")
    axes = figure.add_subplot(1, 1, 1)
    sns.heatmap(scaled, annot=True, fmt=".2f", cmap="YlOrRd", vmin=0, vmax=1,
                linewidths=0.5, ax=axes,
                cbar_kws={"label": "normalised difference [0, 1]"})
    axes.set_title(f"Metric values per algorithm pair (dim {params['dimension']})")
    axes.set_xlabel("Metric")
    axes.set_ylabel("Algorithm pair")
    axes.tick_params(axis="x", rotation=30)
    for label in axes.get_xticklabels():
        label.set_horizontalalignment("right")
    return figure


def _distance_matrix(means, metric, algorithms):
    matrix = pd.DataFrame(
        np.zeros((len(algorithms), len(algorithms))),
        index=algorithms, columns=algorithms,
    )
    for _, row in means.iterrows():
        first, second = row["Algorithm1"], row["Algorithm2"]
        if first in matrix.index and second in matrix.columns:
            matrix.loc[first, second] = row[metric]
            matrix.loc[second, first] = row[metric]
    return matrix


def metric_dendrograms(params):
    """One dendrogram per metric: does each metric group algorithms alike?

    Leaves are coloured by mealpy family, so a metric that recovers the
    families looks different at a glance from one that cuts across them.
    """
    algorithms = _selected(params)
    if len(algorithms) < 3:
        raise ValueError("Hierarchical clustering needs at least 3 algorithms.")
    means = pair_means(params["dimension"], METRICS, algorithms)
    available = [m for m in METRICS if m in means.columns]

    colours = style.family_colours()
    labels = _labels()

    rows = 2 if len(available) > 3 else 1
    columns = int(np.ceil(len(available) / rows))
    figure = Figure(figsize=(6.5 * columns, 6 * rows), layout="constrained")
    axes_list = np.atleast_1d(figure.subplots(rows, columns)).flatten()

    for axes, metric in zip(axes_list, available):
        matrix = _distance_matrix(means, metric, algorithms).to_numpy(dtype=float)
        np.fill_diagonal(matrix, 0.0)
        matrix = (matrix + matrix.T) / 2.0        # enforce exact symmetry
        if np.isnan(matrix).any():
            matrix = np.nan_to_num(matrix, nan=np.nanmax(matrix) if
                                   np.isfinite(matrix).any() else 1.0)
        linkage_matrix = linkage(squareform(matrix, checks=False), method="average")
        dendrogram(linkage_matrix, labels=algorithms, ax=axes, leaf_rotation=90)
        for label in axes.get_xticklabels():
            label.set_color(colours.get(style.family_of(label.get_text()), "#444444"))
            label.set_fontsize(7)
        axes.set_title(labels.get(metric, metric), fontsize=11, fontweight="bold")
        axes.set_ylabel("distance")

    for axes in axes_list[len(available):]:
        axes.set_visible(False)

    # Leaf colours are meaningless without saying what they encode.
    used = sorted({style.family_of(a) for a in algorithms})
    figure.legend(
        handles=[Patch(color=colours.get(f, "#444444"), label=style.family_label(f))
                 for f in used],
        title="mealpy family", loc="lower center", ncol=min(len(used), 5),
        fontsize=9, frameon=False,
    )
    figure.suptitle(
        f"Algorithm clustering under each metric (dim {params['dimension']})",
        fontsize=14,
    )
    return figure


def metric_added_value(params):
    """Do the other metrics agree with the reference on its extreme pairs?

    The pairs are chosen from the data rather than hardcoded: the two most
    and two least similar pairs by the reference metric. If the other metrics
    simply tracked it, every card would be uniform - where they are not is
    exactly the added value the thesis argues for.
    """
    algorithms = _selected(params)
    dimension = params["dimension"]
    means = pair_means(dimension, METRICS, algorithms)
    available = [m for m in METRICS if m in means.columns]
    if REFERENCE_METRIC not in available:
        raise ValueError(f"The merged table has no '{REFERENCE_METRIC}' column.")
    if len(means) < 4:
        raise ValueError(
            f"Only {len(means)} algorithm pairs; select at least 4 algorithms."
        )

    # Rank-normalise each metric across all pairs so the cards share a scale.
    ranked = means.copy()
    for metric in available:
        ranked[metric] = means[metric].rank(pct=True)

    ordered = means.sort_values(REFERENCE_METRIC)
    chosen = (
        [(row, "most similar") for _, row in ordered.head(2).iterrows()]
        + [(row, "least similar") for _, row in ordered.tail(2).iterrows()]
    )

    merged = load_merged(dimension)
    problems = params["problems"]

    figure = Figure(figsize=(13, 11), layout="constrained")
    axes_list = np.atleast_1d(figure.subplots(2, 2)).flatten()
    labels = _labels()

    for axes, (row, kind) in zip(axes_list, chosen):
        first, second = row["Algorithm1"], row["Algorithm2"]
        columns, names = [], []
        for function in problems:
            here = merged[
                (merged["Algorithm1"] == first) & (merged["Algorithm2"] == second)
                & (merged["Function_id"] == function)
            ]
            if here.empty:
                columns.append(np.full(len(available), np.nan))
            else:
                # rank each problem's value against the same pool
                columns.append([
                    float((means[m] <= here[m].mean()).mean()) for m in available
                ])
            names.append(f"F{function}")
        pair_rank = ranked[
            (ranked["Algorithm1"] == first) & (ranked["Algorithm2"] == second)
        ]
        columns.append(pair_rank[available].iloc[0].to_numpy())
        names.append("Mean")

        table = pd.DataFrame(
            np.column_stack(columns),
            index=[labels[m] for m in available], columns=names,
        )
        sns.heatmap(table, ax=axes, cmap="Blues", vmin=0, vmax=1, annot=True,
                    fmt=".2f", linewidths=1.5, linecolor="white", cbar=False)
        colour = "#2a9d8f" if kind == "most similar" else "#e63946"
        axes.set_title(
            f"{first}  vs  {second}\n{kind} by {labels[REFERENCE_METRIC].lower()} "
            f"({row[REFERENCE_METRIC]:.2f})",
            fontsize=10, fontweight="bold", color=colour,
        )
        axes.set_xlabel("")
        axes.set_ylabel("")
        axes.tick_params(axis="y", rotation=0, labelsize=9)

    figure.suptitle(
        f"What the other metrics say about the reference metric's extremes "
        f"(dim {dimension})\n0 = most similar pair, 1 = most different, "
        f"ranked across all selected pairs",
        fontsize=12,
    )
    return figure
