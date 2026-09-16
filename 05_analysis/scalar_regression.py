"""
scalar_regression.py
====================
Regression scatter plots over the *per-algorithm scalars* that underlie the
pairwise metric suite.

Motivation
----------
The pairwise metrics (entropy difference, exploration difference, ...) are
built on top of per-algorithm scalars:

    H_a   Shannon entropy of the discretised search trajectory
    E_a   exploration / exploitation ratio

Regressing H_a against E_a over the 28 algorithms answers the question you
actually want to ask -- *do high-entropy searchers explore more?*

Averaging a pairwise metric over partners, D_a = (1/(n-1)) sum_b d(a, b), does
NOT answer that. For a difference-type metric d(a,b) = |s_a - s_b|, D_a is
V-shaped in s_a: both the lowest- and the highest-entropy algorithm score high.
D_a measures *atypicality*, not magnitude. It also makes the OLS p-value invalid,
since D_a and D_b share the term d(a, b).

This module therefore stays on the scalars. The 28 points are not structurally
coupled, so slope / R^2 / p-value are interpretable at face value.

Expected input
--------------
A tidy CSV with one row per (dim, func, algo) and one column per scalar, which
is exactly what `05_analysis/build_scalars.py` writes to
`metrics_data/scalars.csv` (the default for `--scalars`):

    dim,func,algo,entropy,fitness,exploration,diversity
    2,1,JADE,0.71,0.006,0.63,1.42
    2,1,L_SHADE,0.55,0.011,0.58,1.31
    ...

Note this is a different shape from the rest of the pipeline's output, which is
PAIRWISE (one row per algorithm *pair*: Algorithm1, Algorithm2, ...). Run
`python 05_analysis/build_scalars.py` first; `metrics_data/merged/merged_dim_*.csv`
is not a valid input here.

`load_scalars` also accepts the long form (dim, func, algo, metric, value) and
pivots it.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from config import SCALARS_CSV, FIGURES_RESULTS_DIR

# --------------------------------------------------------------------------
# Algorithm families -- used for point colour and for spotting the case where
# one family collapses to a blob and a handful of outliers fit the line.
# Anything not listed falls through to "other"; fill in the rest of the 28.
# --------------------------------------------------------------------------
FAMILY: dict[str, str] = {
    "JADE": "DE",
    "SADE": "DE",
    "L-SHADE": "DE",
    "OriginalSHADE": "DE",
    "OriginalDE": "DE",
    "OriginalAEO": "AEO",
    "EnhancedAEO": "AEO",
    "ImprovedAEO": "AEO",
    "AugmentedAEO": "AEO",
    "OriginalWOA": "WOA",
    "HI_WOA": "WOA",
    "OriginalGWO": "GWO",
    "RW_GWO": "GWO",
}

FAMILY_COLOURS = {
    "DE": "#1f77b4",
    "AEO": "#2ca02c",
    "WOA": "#d62728",
    "GWO": "#9467bd",
    "other": "#7f7f7f",
}


def family_of(algo: str) -> str:
    return FAMILY.get(algo, "other")


# --------------------------------------------------------------------------
# Loading / aggregation
# --------------------------------------------------------------------------
def load_scalars(path: str | Path) -> pd.DataFrame:
    """Read the per-(dim, func, algo) scalar table, wide or long."""
    df = pd.read_csv(path)
    if {"metric", "value"}.issubset(df.columns):  # long -> wide
        df = df.pivot_table(
            index=["dim", "func", "algo"], columns="metric", values="value"
        ).reset_index()
        df.columns.name = None
    missing = {"dim", "func", "algo"} - set(df.columns)
    if missing:
        raise ValueError(f"scalar table is missing key column(s): {sorted(missing)}")
    return df


def aggregate_over_functions(
    df: pd.DataFrame,
    metrics: list[str],
    normalise_per_function: bool = False,
) -> pd.DataFrame:
    """Collapse the BBOB function axis -> one row per (dim, algo).

    normalise_per_function z-scores each scalar within each (dim, func) group
    before averaging. Use it when a scalar's scale drifts across functions and
    you don't want a couple of high-variance functions dominating the mean;
    leave it off when the raw units are the thing you want on the axis.
    """
    df = df.copy()

    if normalise_per_function:
        # z-score within (dim, func): centres every function on the same scale,
        # so the mean over functions weights each function equally.
        df[metrics] = df.groupby(["dim", "func"])[metrics].transform(
            lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) > 0 else s * 0.0
        )

    agg = df.groupby(["dim", "algo"])[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]  # flatten MultiIndex
    return agg.reset_index()


# --------------------------------------------------------------------------
# Fit + influence diagnostics
# --------------------------------------------------------------------------
def cooks_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Cook's D for simple OLS (p = 2: slope + intercept).

    Flags the points that are single-handedly holding up the regression line --
    exactly the failure mode where the DE family sits in one blob and two swarm
    algorithms out on the edge define the slope.
    """
    n = len(x)
    fit = stats.linregress(x, y)
    resid = y - (fit.intercept + fit.slope * x)
    mse = np.sum(resid**2) / (n - 2)

    sxx = np.sum((x - x.mean()) ** 2)
    leverage = 1.0 / n + (x - x.mean()) ** 2 / sxx  # hat-matrix diagonal

    return (resid**2 / (2 * mse)) * (leverage / (1 - leverage) ** 2)


def fit_stats(x: np.ndarray, y: np.ndarray) -> dict:
    fit = stats.linregress(x, y)
    rho, rho_p = stats.spearmanr(x, y)  # rank-based; comparable to the corr matrix
    return {
        "slope": fit.slope,
        "intercept": fit.intercept,
        "r2": fit.rvalue**2,
        "p": fit.pvalue,
        "rho": rho,
        "rho_p": rho_p,
    }


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
def scatter_regression(
    ax: plt.Axes,
    sub: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    errorbars: bool = True,
    annotate: bool = True,
    influence_threshold: float | None = None,
) -> dict:
    """One panel: 28 algorithms, coloured by family, with an OLS fit.

    `sub` is the aggregated frame filtered to a single dim.
    Returns the fit statistics for that panel.
    """
    x = sub[f"{x_metric}_mean"].to_numpy()
    y = sub[f"{y_metric}_mean"].to_numpy()
    algos = sub["algo"].to_numpy()

    # spread across BBOB functions -- shows whether an algorithm's scalar is a
    # stable property or an artefact of averaging over wildly different funcs
    if errorbars:
        ax.errorbar(
            x, y,
            xerr=sub[f"{x_metric}_std"], yerr=sub[f"{y_metric}_std"],
            fmt="none", ecolor="lightgray", elinewidth=0.8, zorder=1,
        )

    for fam in sorted({family_of(a) for a in algos}):
        m = np.array([family_of(a) == fam for a in algos])
        ax.scatter(
            x[m], y[m], s=46, zorder=3, label=fam,
            color=FAMILY_COLOURS.get(fam, FAMILY_COLOURS["other"]),
            edgecolor="white", linewidth=0.6,
        )

    st = fit_stats(x, y)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, st["intercept"] + st["slope"] * xs, "k--", lw=1.2, zorder=2)

    # ring the high-influence points instead of silently letting them set the slope
    thr = influence_threshold if influence_threshold is not None else 4.0 / len(x)
    infl = cooks_distance(x, y) > thr
    if infl.any():
        ax.scatter(
            x[infl], y[infl], s=170, zorder=2,
            facecolors="none", edgecolors="black", linewidths=1.1,
        )

    if annotate:  # only 28 points, so the labels stay legible
        for xi, yi, a in zip(x, y, algos):
            ax.annotate(a, (xi, yi), fontsize=5.5, xytext=(4, 3),
                        textcoords="offset points", zorder=4)

    ax.set_title(
        f"$R^2$={st['r2']:.2f}   ρ={st['rho']:.2f}   p={st['p']:.1e}",
        fontsize=9,
    )
    return st


def facet_by_dim(
    agg: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    dims: list[int] | None = None,
    outfile: str | Path | None = None,
    **kwargs,
) -> tuple[plt.Figure, pd.DataFrame]:
    """One panel per dimension, shared axes.

    The change in slope across 2 -> 5 -> 10 is itself the result: if the
    entropy/exploration relationship steepens with dimension, that is a claim
    the pooled single-panel version would have hidden.
    """
    dims = dims or sorted(agg["dim"].unique())
    fig, axes = plt.subplots(
        1, len(dims), figsize=(5.2 * len(dims), 4.8), sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes)

    rows = []
    for ax, d in zip(axes, dims):
        st = scatter_regression(
            ax, agg[agg["dim"] == d], x_metric, y_metric, **kwargs
        )
        rows.append({"dim": d, **st})
        ax.set_xlabel(x_metric.replace("_", " "))
        ax.text(0.02, 0.97, f"dim = {d}", transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top")

    axes[0].set_ylabel(y_metric.replace("_", " "))
    axes[-1].legend(fontsize=7, loc="lower right", frameon=True)
    fig.suptitle(f"{y_metric} vs {x_metric} across 28 algorithms", fontsize=12)
    fig.tight_layout()

    if outfile:
        fig.savefig(outfile, dpi=200, bbox_inches="tight")

    return fig, pd.DataFrame(rows)


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scalars", default=SCALARS_CSV,
                    help="per-(dim, func, algo) scalar CSV (default: what build_scalars.py writes)")
    ap.add_argument("--x", default="entropy")
    ap.add_argument("--y", default="exploration")
    ap.add_argument("--dims", type=int, nargs="*", default=None)
    ap.add_argument("--normalise-per-function", action="store_true")
    ap.add_argument("--no-errorbars", action="store_true")
    ap.add_argument("--no-annotate", action="store_true")
    ap.add_argument("--out", default=f"{FIGURES_RESULTS_DIR}/scalar_regression.png")
    args = ap.parse_args()

    if not Path(args.scalars).is_file():
        raise SystemExit(
            f"no scalar table at {args.scalars}\n"
            f"run `python 05_analysis/build_scalars.py` first "
            f"(the pairwise merged_dim_*.csv files are a different shape and won't work here)"
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    df = load_scalars(args.scalars)
    agg = aggregate_over_functions(
        df, [args.x, args.y], normalise_per_function=args.normalise_per_function
    )
    _, summary = facet_by_dim(
        agg, args.x, args.y,
        dims=args.dims,
        outfile=args.out,
        errorbars=not args.no_errorbars,
        annotate=not args.no_annotate,
    )
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
