"""The set of mealpy optimizers the GUI can offer, and how they group.

Nothing here hardcodes optimizer names. Both facts the picker needs already
exist in the repo and are simply cached:

  helper_functions.get_optimizers_safe()  name -> class, found by walking
                                          mealpy's package tree
  utils.get_algorithm_groups()            name -> mealpy family

Both are slow enough to matter (the first walks every mealpy module and takes
a few seconds), so they run once and the result is cached for the process.
Call warm_up() off the UI thread before showing the picker.
"""
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import ALGORITHMS_OF_INTEREST

# mealpy names families after its own subpackages; these are for display only.
FAMILY_LABELS = {
    "bio_based": "Bio-inspired",
    "evolutionary_based": "Evolutionary",
    "human_based": "Human-based",
    "math_based": "Math-based",
    "music_based": "Music-based",
    "physics_based": "Physics-based",
    "sota_based": "State-of-the-art",
    "swarm_based": "Swarm",
    "system_based": "System-based",
    "utils": "Utility",
    "other": "Other",
}

UNKNOWN_FAMILY = "other"


class UnknownOptimizerError(KeyError):
    """Raised for a name mealpy does not provide.

    run_benchmarks.py resolves names with a bare dict lookup, so a typo there
    surfaces as an opaque KeyError. The GUI should never hand mealpy a name it
    has not validated, and this carries enough context to say why.
    """


@dataclass(frozen=True)
class OptimizerRegistry:
    """Every mealpy optimizer available, with its family."""

    classes: dict
    families: dict

    @property
    def names(self):
        return sorted(self.classes)

    @property
    def by_family(self):
        grouped = {}
        for name in self.names:
            grouped.setdefault(self.families.get(name, UNKNOWN_FAMILY), []).append(name)
        return dict(sorted(grouped.items()))

    def family_of(self, name):
        return self.families.get(name, UNKNOWN_FAMILY)

    def resolve(self, names):
        """names -> {name: class}, in the order given.

        This is the guard that run_benchmarks.py:56 lacks.
        """
        missing = [n for n in names if n not in self.classes]
        if missing:
            raise UnknownOptimizerError(
                f"mealpy does not provide: {', '.join(sorted(missing))}"
            )
        return {n: self.classes[n] for n in names}


@lru_cache(maxsize=1)
def load_registry():
    """Build the registry. Seconds on first call, instant afterwards."""
    # Imported lazily: helper_functions pulls in mealpy and cocoex, and utils
    # pulls in seaborn/sklearn/matplotlib. Neither should be paid at import
    # time of this module, which the UI thread touches early.
    from helper_functions import get_optimizers_safe
    import utils

    classes, _names = get_optimizers_safe()
    groups = utils.get_algorithm_groups()
    families = {name: groups.get(name, UNKNOWN_FAMILY) for name in classes}
    return OptimizerRegistry(classes=classes, families=families)


def warm_up():
    """Force the slow discovery now. Safe to call from a worker thread."""
    return load_registry()


def thesis_selection():
    """The 28 optimizers config.py pins as the thesis set, minus any that this
    mealpy version no longer ships (so the preset degrades instead of raising).
    """
    registry = load_registry()
    return [n for n in ALGORITHMS_OF_INTEREST if n in registry.classes]


def family_label(family):
    return FAMILY_LABELS.get(family, family.replace("_", " ").title())
