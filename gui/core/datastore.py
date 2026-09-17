"""Discovery and loading of everything the pipeline has written.

Two jobs, both of which exist because the on-disk layout is less uniform than
it looks:

1. Discovery. outputs/ alone holds ~151k files, so the tree is built lazily -
   a node lists its children only when something asks for them.

2. Loading. Three physical formats hide behind the same .csv extension
   (plain, zip-wrapped in data/processed/, parquet in clustering_results/),
   so format is sniffed from magic bytes rather than trusted from the name.
   Four different key-column conventions are normalised on read so the viewer
   can present one consistent shape.

Every file the pipeline produces fits comfortably in memory (the largest is
merged_dim_{d}.csv at ~31 MB / 227k rows, which pandas reads in 0.3s), so
tables load whole and QTableView virtualizes the display.
"""
import re
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config

# Rendered as a heatmap rather than a table: these are square correlation
# matrices, not tidy rows.
MATRIX_STEMS = ("spearman_dim_",)

_UNNAMED = "Unnamed: 0"

_ZIP_MAGIC = b"PK\x03\x04"
_PARQUET_MAGIC = b"PAR1"

TRAJECTORY_KINDS = {
    "trajectory": "Full population trajectory",
    "gbest_trajectory": "Best-so-far trajectory",
    "diversity": "Diversity / exploration / exploitation",
}


# --------------------------------------------------------------------------
# format sniffing + loading
# --------------------------------------------------------------------------

def sniff_format(path):
    """Return 'parquet', 'zip' or 'csv' from the file's first bytes.

    data/processed/*.csv are zip archives despite the extension, so the
    extension is not usable here.
    """
    with open(path, "rb") as fh:
        head = fh.read(4)
    if head == _PARQUET_MAGIC:
        return "parquet"
    if head == _ZIP_MAGIC:
        return "zip"
    return "csv"


def is_matrix(path):
    return any(Path(path).stem.startswith(stem) for stem in MATRIX_STEMS)


def _resolve_unnamed_index(frame, path):
    """Decide what the leading unnamed column actually means.

    It is not always noise. In cluster_centers/ it is the cluster id, and in
    algorithm_pairwise_similarity/ it is a meaningful non-monotonic source
    index. Only drop it when it is a plain 0..n-1 range duplicating the row
    number.
    """
    if _UNNAMED not in frame.columns:
        return frame, ""

    column = frame[_UNNAMED]
    if "cluster_centers" in Path(path).parts:
        return frame.rename(columns={_UNNAMED: "cluster"}), "Unnamed column is the cluster id"
    if column.equals(pd.Series(range(len(frame)), name=_UNNAMED)):
        return frame.drop(columns=[_UNNAMED]), ""
    return (
        frame.rename(columns={_UNNAMED: "source_index"}),
        "Unnamed column kept as 'source_index'",
    )


@dataclass
class TableData:
    """A loaded table plus what the viewer needs to describe it."""

    frame: pd.DataFrame
    path: Path
    fmt: str
    notes: list = field(default_factory=list)

    @property
    def shape(self):
        return self.frame.shape


def _load_manifest_table(path):
    """A run manifest as a readable two-column table."""
    import json

    manifest = json.loads(Path(path).read_text())
    rows = []

    def walk(prefix, value):
        if isinstance(value, dict):
            for key, item in value.items():
                walk(f"{prefix}.{key}" if prefix else str(key), item)
        elif isinstance(value, list):
            rows.append({"field": prefix, "value": ", ".join(str(v) for v in value)})
        else:
            rows.append({"field": prefix, "value": value})

    walk("", manifest)
    return pd.DataFrame(rows)


def load_table(path, normalise=True):
    """Read any pipeline file into a DataFrame, whatever its physical format."""
    path = Path(path)
    if path.suffix == ".json":
        return TableData(frame=_load_manifest_table(path), path=path, fmt="json",
                         notes=["run manifest"])
    fmt = sniff_format(path)
    notes = []

    if fmt == "parquet":
        frame = pd.read_parquet(path)
        # clustering_results stores its index rather than a column. Surface it
        # as an ordinary column and let the resolver below decide whether it
        # carries meaning or just duplicates the row number.
        if frame.index.name is None:
            frame = frame.reset_index(names=_UNNAMED)
        else:
            frame = frame.reset_index()
    elif fmt == "zip":
        # Read without index_col: data/processed/ writes an index, but a zip
        # written without one would otherwise lose its first real column.
        frame = pd.read_csv(path, compression="zip")
        notes.append("zip-compressed CSV")
    elif is_matrix(path):
        frame = pd.read_csv(path, index_col=0)
    else:
        frame = pd.read_csv(path)

    frame, unnamed_note = _resolve_unnamed_index(frame, path)
    if unnamed_note:
        notes.append(unnamed_note)

    if normalise and set(config.METRIC_KEYS) & set(frame.columns):
        before = frame[[c for c in config.METRIC_KEYS if c in frame.columns]].dtypes
        frame = config.normalize_keys(frame)
        after = frame[[c for c in config.METRIC_KEYS if c in frame.columns]].dtypes
        if not before.equals(after):
            notes.append("Keys normalised via config.normalize_keys()")

    return TableData(frame=frame, path=path, fmt=fmt, notes=notes)


# --------------------------------------------------------------------------
# lazy tree
# --------------------------------------------------------------------------

@dataclass
class Node:
    """One row in the browser tree. Children are produced on demand."""

    label: str
    kind: str = "group"  # group | table | matrix
    path: Path = None
    hint: str = ""
    loader: object = field(default=None, repr=False)
    _children: list = field(default=None, repr=False, compare=False)

    @property
    def is_leaf(self):
        return self.kind != "group"

    def children(self):
        if self._children is None:
            self._children = list(self.loader()) if self.loader else []
        return self._children


def _natural_key(text):
    """Sort F2_I1 before F10_I1, and dim_2 before dim_10."""
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", str(text))]


def _leaf(path, label=None, hint=""):
    path = Path(path)
    return Node(
        label=label or path.name,
        kind="matrix" if is_matrix(path) else "table",
        path=path,
        hint=hint,
    )


def _files_in(directory, pattern="*", hint=""):
    directory = Path(directory)
    if not directory.is_dir():
        return []
    files = sorted(
        (p for p in directory.glob(pattern) if p.is_file()),
        key=lambda p: _natural_key(p.name),
    )
    return [_leaf(p, hint=hint) for p in files]


def _subdirs(directory, make_child):
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return [
        make_child(p)
        for p in sorted(
            (p for p in directory.iterdir() if p.is_dir()), key=lambda p: _natural_key(p.name)
        )
    ]


def _dim_dirs(base, hint="", pattern="*"):
    """base/dim_{d}/<files> - the shape of almost every pipeline output dir."""
    return _subdirs(
        base,
        lambda d: Node(
            label=d.name,
            loader=lambda d=d: _files_in(d, pattern, hint=hint),
        ),
    )


def _run_files(run_dir):
    """The 15 per-run CSVs, grouped by kind so seeds sit together."""
    nodes = []
    for kind, description in TRAJECTORY_KINDS.items():
        seeds = sorted(run_dir.glob(f"{kind}_*.csv"), key=lambda p: _natural_key(p.name))
        nodes.extend(_leaf(p, label=p.name, hint=description) for p in seeds)
    return nodes


def _benchmark_runs():
    """outputs/dim_{d}/{algorithm}/{function}_{instance}/ - lazy at every level."""
    def dim_node(dim_dir):
        def algorithms():
            return _subdirs(
                dim_dir,
                lambda algo: Node(
                    label=algo.name,
                    loader=lambda algo=algo: _subdirs(
                        algo,
                        lambda run: Node(
                            label=f"F{run.name.split('_')[0]}_I{run.name.split('_')[1]}",
                            loader=lambda run=run: _run_files(run),
                        ),
                    ),
                ),
            )

        return Node(label=dim_dir.name, loader=algorithms)

    return _subdirs(Path(config.OUTPUTS_DIR), dim_node)


def _pairwise_metrics():
    base = Path(config.METRICS_DIR)
    if not base.is_dir():
        return []
    metrics = sorted(
        p for p in base.iterdir() if p.is_dir() and p.name != Path(config.MERGED_DIR).name
    )
    return [
        Node(
            label=config.METRIC_LABELS.get(m.name, m.name),
            hint=f"metrics_data/{m.name}",
            loader=lambda m=m: _dim_dirs(m, hint="One row per algorithm pair"),
        )
        for m in metrics
    ]


def _clustering_group(base, label):
    subdirs = [
        ("cluster_distributions", "Per-iteration cluster occupancy"),
        ("clustering_results", "Clustered trajectories (parquet)"),
        ("cluster_centers", "KMeans cluster centres"),
        ("algorithm_pairwise_similarity", "Aggregate cosine similarity"),
    ]
    children = []
    for name, hint in subdirs:
        directory = base / name
        if not directory.is_dir():
            continue
        has_dims = any(p.is_dir() for p in directory.iterdir())
        children.append(
            Node(
                label=name,
                hint=hint,
                loader=(
                    (lambda d=directory, h=hint: _dim_dirs(d, hint=h))
                    if has_dims
                    else (lambda d=directory, h=hint: _files_in(d, hint=h))
                ),
            )
        )
    return Node(label=label, loader=lambda: children)


def _gui_runs():
    """Runs launched from the GUI, newest first.

    These live outside the repo's outputs/ by design (see gui.core.workspace),
    which also means nothing else in this tree would ever show them - and in a
    packaged app they are the ONLY results that exist.
    """
    from gui.core import workspace

    nodes = []
    for entry in workspace.list_runs():
        directory = entry["path"]
        manifest = entry["manifest"] or {}
        spec = manifest.get("spec", {})
        status = manifest.get("status", "?")
        completed = manifest.get("completed_runs")
        summary = f"{status}"
        if completed is not None:
            summary += f", {completed} runs"
        algorithms = spec.get("algorithms") or []
        if algorithms:
            summary += f", {len(algorithms)} algorithms"

        def children(directory=directory):
            items = []
            manifest_path = directory / "manifest.json"
            if manifest_path.is_file():
                items.append(_leaf(manifest_path, label="manifest.json",
                                   hint="What produced this run"))
            outputs = directory / "outputs"
            if outputs.is_dir():
                items.extend(
                    _subdirs(
                        outputs,
                        lambda dim: Node(
                            label=dim.name,
                            loader=lambda dim=dim: _subdirs(
                                dim,
                                lambda algo: Node(
                                    label=algo.name,
                                    loader=lambda algo=algo: _subdirs(
                                        algo,
                                        lambda run: Node(
                                            label=run.name,
                                            loader=lambda run=run: _run_files(run),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    )
                )
            return items

        nodes.append(Node(label=f"{directory.name}  ({summary})", loader=children))
    return nodes


def build_tree():
    """Top-level browser roots, omitting anything not present on disk."""
    from gui.core import workspace

    roots = []

    # Listed first: in a packaged app these are the only results there are,
    # and after a run this is where the user's new data actually is.
    if workspace.list_runs():
        roots.append(
            Node(
                label="GUI runs",
                hint=f"Runs launched from this app ({workspace.workspace_root()})",
                loader=_gui_runs,
            )
        )

    def add(node, exists):
        if exists:
            roots.append(node)

    outputs = Path(config.OUTPUTS_DIR)
    add(
        Node(
            label="Benchmark runs",
            hint="Raw per-run trajectories",
            loader=_benchmark_runs,
        ),
        outputs.is_dir(),
    )
    # Gated on a results.csv actually existing: harvest_results.py writes it,
    # and a fresh benchmark run has trajectories but no summary yet. An empty
    # expandable node would just look broken.
    summaries = sorted(outputs.glob("dim_*/results.csv")) if outputs.is_dir() else []
    add(
        Node(
            label="Run summaries",
            hint="One row per algorithm / problem / instance / seed",
            loader=lambda: [
                _leaf(path, label=f"{path.parent.name}/results.csv")
                for path in sorted(summaries, key=lambda p: _natural_key(p.parent.name))
            ],
        ),
        bool(summaries),
    )
    add(
        Node(
            label="Processed trajectories",
            hint="Clustering input (zip-compressed CSV)",
            loader=lambda: _dim_dirs(
                Path(config.PROCESSED_DIR), hint="zip-compressed CSV"
            ),
        ),
        Path(config.PROCESSED_DIR).is_dir(),
    )

    latest = Path(config.CLUSTERING_LATEST_DIR)
    add(_clustering_group(latest, "Clustering (kmeans)"), latest.is_dir())

    entropy = Path(config.ENTROPY_DATA_DIR)
    add(
        Node(
            label="Entropy",
            hint="Normalised Shannon entropy of cluster occupancy",
            loader=lambda: _files_in(entropy, "*.csv"),
        ),
        entropy.is_dir(),
    )

    metrics = Path(config.METRICS_DIR)
    add(
        Node(label="Pairwise metrics", hint="One row per algorithm pair", loader=_pairwise_metrics),
        metrics.is_dir(),
    )
    merged = Path(config.MERGED_DIR)
    add(
        Node(
            label="Merged & correlation",
            hint="All metrics joined, and the Spearman matrices",
            loader=lambda: _files_in(merged, "*.csv"),
        ),
        merged.is_dir(),
    )
    scalars = Path(config.SCALARS_CSV)
    add(
        Node(
            label="Scalars",
            hint="Per-algorithm scalars (not pairwise)",
            loader=lambda: [_leaf(scalars)],
        ),
        scalars.exists(),
    )

    dbscan = Path(config.CLUSTERING_DBSCAN_DIR)
    add(
        Node(
            label="Experiments (legacy DBSCAN)",
            hint="Partial side experiment, not part of the main grid",
            loader=lambda: [_clustering_group(dbscan, dbscan.name)],
        ),
        dbscan.is_dir(),
    )

    return roots
