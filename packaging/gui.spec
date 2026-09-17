# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Optimizer Trajectory Explorer.

Build:   pyinstaller packaging/gui.spec --noconfirm
Verify:  dist/OptimizerTrajectoryExplorer/OptimizerTrajectoryExplorer --self-test

Four things here are load-bearing; each one produces a bundle that starts
cleanly and then fails at runtime if it is dropped.

1. collect_submodules('mealpy')
   helper_functions.get_optimizers_safe() discovers optimizers with
   pkgutil.walk_packages. Static analysis cannot see that, so without this
   the app launches with an EMPTY optimizer list.

2. collect_data_files('opfunu')
   The CEC suites load ~1190 bundled shift/rotation matrices. Without them
   the catalog still lists every CEC function and each one fails the moment
   it is evaluated.

3. collect_all('cocoex')
   cocoex is a compiled C extension (two .so/.pyd files) plus data.

4. The numbered stage folders on pathex, with their modules named explicitly.
   gui/viz/registry.py imports entropy_plotting, spearman and
   scalar_regression by name at runtime via importlib, because directories
   called "04_metrics" are not importable as packages. Naming them here is
   what puts them in the bundle.

PyQt6 is excluded deliberately: it is installed in the dev venv as a
matplotlib backend, and shipping two Qt bindings in one process is a
C++-level crash waiting to happen.

pyarrow is excluded in favour of fastparquet. Both read the project's
clustering_results parquet files identically (verified on a real 140k x 29
file); pyarrow costs 146 MB of the bundle and fastparquet costs about 9. The
code does not care - pandas picks whichever engine is installed - and the
self-test proves the reader still works in the bundle.
"""
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

REPO_ROOT = Path(SPECPATH).resolve().parent

APP_NAME = "OptimizerTrajectoryExplorer"

# -- 1. mealpy: invisible to static analysis (pkgutil.walk_packages)
hidden = collect_submodules("mealpy")

# -- 4. pipeline modules imported by name at runtime
hidden += [
    "config", "utils", "helper_functions",
    "entropy", "entropy_plotting",          # 04_metrics
    "spearman", "scalar_regression",        # 05_analysis
]

# -- 2. opfunu CEC support data
datas = collect_data_files("opfunu")

# -- 3. cocoex: compiled extension + its data
cocoex_datas, cocoex_binaries, cocoex_hidden = collect_all("cocoex")
datas += cocoex_datas
hidden += cocoex_hidden

analysis = Analysis(
    [str(REPO_ROOT / "gui" / "__main__.py")],
    pathex=[
        str(REPO_ROOT),
        str(REPO_ROOT / "04_metrics"),
        str(REPO_ROOT / "05_analysis"),
    ],
    binaries=cocoex_binaries,
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        "PyQt6", "PyQt5", "PySide2",   # never two Qt bindings in one process
        # Qt modules this app never touches - it is QtWidgets only.
        "PySide6.QtQml", "PySide6.QtQuick", "PySide6.QtQuickWidgets",
        "PySide6.QtPdf", "PySide6.QtPdfWidgets", "PySide6.Qt3DCore",
        "PySide6.QtMultimedia", "PySide6.QtWebEngineCore",
        "PySide6.QtCharts", "PySide6.QtDesigner", "PySide6.QtTest",
        # 146 MB; fastparquet reads the same files (see the note above).
        "pyarrow",
        "tkinter",
        "IPython", "jupyter", "notebook", "nbformat", "nbconvert",
        "pytest", "_pytest",
        "pymoo", "yellowbrick", "kneed", "dtw",   # notebook-only extras
    ],
    noarchive=False,
)

pyz = PYZ(analysis.pure)

exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,          # a GUI app: no console window on Windows
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# One-dir rather than one-file: a one-file build unpacks ~300 MB to a temp
# directory on every launch, which makes startup slow enough to be awkward
# in a live demo.
collect = COLLECT(
    exe,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=APP_NAME,
)
